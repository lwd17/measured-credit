#!/usr/bin/env python
"""ALFWorld training: what is a correct step-level signal worth?

An audit over 192 rollouts / 3695 turns found that a step-level baseline which
estimates credit from the group's observed returns does not track which turns
advanced the task: within a trajectory its rank correlation with planner
progress is -0.072, 95% CI [-0.138, -0.010], against +0.925 for an independent
leave-one-out control. Its pooled correlation is entirely explained by
trajectory-level outcome information, which GRPO already has.

That is a measurement about a signal. This turns it into a measurement about
training. The arms share one combination rule and differ only in what fills the
step-level slot:

    A(i,t) = A_E(i) + omega * A_step(i,t)

    grpo       A_step = 0
    anchor_cf  A_step = the measured counterfactual, z-scored over the
               measured turns of the group and then centred within each
               trajectory

Fairness: one loop, one optimizer, one schedule, one rollout path, one encoder.
The arm selects a vector of per-turn advantages and nothing else.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

STEP_RE = re.compile(r"\(step \d+/\d+\)")


def anchor_key(observation: str) -> str:
    """State key for anchor grouping: same state text, same group.

    The step counter is stripped: leaving it in makes every state unique, every
    anchor group size 1, and A_S identically zero -- a silent reduction of any
    anchor-based arm to plain GRPO that would still log under its own name.
    """
    o = STEP_RE.sub("", observation or "").strip().lower()
    return " ".join(o.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=("grpo", "anchor_cf"))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--prompts_per_step", type=int, default=4)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max_turns", type=int, default=30)
    ap.add_argument("--n_train_tasks", type=int, default=64)
    ap.add_argument("--n_eval_tasks", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--hard_types", action="store_true",
                    help="keep only the four task types whose optimal plans are "
                         "long enough for per-turn credit to have anything to "
                         "assign. pick_and_place_simple and look_at_obj_in_light "
                         "run 14 turns and come back 83%% untrained, against 48%% "
                         "and 21-25 turns for the other four; they are 34%% of the "
                         "pool and can only dilute a per-turn effect.")
    ap.add_argument("--action_hint", default="full",
                    choices=("full", "objects", "none"),
                    help="how much the prompt gives away. Measured base rates "
                         "for an untrained Qwen3-8B: 'full' (the admissible "
                         "command list) 65%%, 'objects' (names only) 0/24, "
                         "'none' 0/24. The list is not a hint but the task's "
                         "grammar -- it says which verbs apply right now -- so "
                         "at this model scale there is no usable middle "
                         "setting and 'full' is the only trainable one.")
    ap.add_argument("--max_tokens", type=int, default=3072,
                    help="generation cap per turn. Distinct from "
                         "--token_budget, which sizes a gradient micro-batch; "
                         "conflating them would silently truncate every reply "
                         "at the accumulation size.")
    ap.add_argument("--token_budget", type=int, default=16384,
                    help="tokens per micro-batch; activation memory tracks "
                         "tokens, so this is what actually bounds the peak")
    ap.add_argument("--max_seq_len", type=int, default=12288)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--collect_workers", type=int, default=32)
    ap.add_argument("--annot_workers", type=int, default=24)
    ap.add_argument("--annot_timeout", type=float, default=300.0,
                    help="deadline for a step's planner annotation. Fast "
                         "Downward's search is unbounded, so a single hard "
                         "instance can hold a step indefinitely; 90s was too "
                         "tight and dropped whole groups.")
    ap.add_argument("--model_path",
                    default=os.environ.get("SSC_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--served_name", default="Qwen3-8B")
    ap.add_argument("--base_url", default="http://localhost:8100/v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dynamic_sampling", action="store_true",
                    help="drop groups whose 8 rollouts all end alike (A_E is "
                         "identically zero) and keep drawing tasks to refill. "
                         "The multiplicative form is identically zero on such "
                         "groups while the additive form still has a step "
                         "term -- that is the main reason the two differ on "
                         "ALFWorld.")
    ap.add_argument("--dyn_max_factor", type=int, default=6,
                    help="with dynamic sampling, draw at most "
                         "prompts_per_step * this value tasks per step.")
    ap.add_argument("--fallback_omega", type=float, default=None,
                    help="scale of the step term on the fallback branch "
                         "(groups with A_E=0); defaults to --omega. Use it to "
                         "hold the magnitude down when three terms add up: on "
                         "QA, omega=1 doubles |adv| and the run collapses.")
    ap.add_argument("--mass_fallback", action="store_true",
                    help="on groups whose outcomes are all equal (A_E=0, so "
                         "there is no mass to conserve) fall back to the "
                         "additive step term instead of letting A = A_E*q be "
                         "identically zero. Such groups are about 74%% of "
                         "ALFWorld.")
    ap.add_argument("--dose", type=float, default=1.0,
                    help="dose for anchor_mass_signed: 0 reduces exactly to "
                         "GRPO; above 1 the least necessary turns get a "
                         "negative weight, so a negative sign can appear even "
                         "inside a successful trajectory")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="dose knob for anchor_mass; at 1.0 every weight is 1 "
                         "and it reduces exactly to GRPO")
    ap.add_argument("--measure_losers", action="store_true",
                    help="measure failed trajectories too: replay each prefix "
                         "followed by the shortest successful plan in the "
                         "group, so delta_t is the change in whether the task "
                         "can still be completed. That measures WHICH STEP "
                         "broke the task instead of using an on-path / "
                         "off-path heuristic")
    ap.add_argument("--nonneg_winners", action="store_true",
                    help="clamp the step term to >=0 on trajectories whose "
                         "advantage is positive: a redundant exploration step "
                         "falls back to A_E instead of being penalised "
                         "(09-13 4B probe: 54%% of 'go to' turns in successful "
                         "trajectories received a negative advantage)")
    ap.add_argument("--zero_uniform_groups", action="store_true",
                    help="zero the step term on groups whose outcomes are all "
                         "equal, reducing exactly to GRPO; this removes an "
                         "outcome-independent, position-shaped signal")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--resume_adapter", default=None,
                    help="path to a saved adapter to continue from. The "
                         "optimizer moments are NOT restored -- only the "
                         "weights -- so a resumed run takes a step or two to "
                         "rebuild Adam's second moment. That is a real "
                         "discontinuity and it is cheaper than the alternative, "
                         "which is discarding the steps already paid for.")
    ap.add_argument("--start_step", type=int, default=0,
                    help="step number to resume at; the schedule and the eval "
                         "cadence both key off this, so a resumed arm still "
                         "evaluates at the same steps as one that ran straight "
                         "through")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ssc.credit.alf_arms import turn_advantages
    from ssc.credit.annotate_pool import annotate_batch
    from ssc.detector.rollout import Rollout
    from ssc.env.alf_pool import collect
    from ssc.env.alfworld_env import (ALFWorldPolicyConfig, HARD_TYPES,
                                      load_alfworld,
                                      system_prompt)
    from ssc.train.batch import collate, encode_turns, token_logprobs
    from ssc.credit.grpo_patch import grpo_loss
    from ssc.train.sync import LoRASync

    out_dir = ROOT / (args.out_dir or f"runs/alf_{args.arm}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"config_from_step{args.start_step}.json" if args.start_step else "config.json"
    (out_dir / tag).write_text(json.dumps(vars(args), indent=2))
    stats_path = out_dir / "steps.jsonl"

    torch.manual_seed(args.seed)

    # Train and eval task pools are disjoint by construction: the eval pool is
    # drawn from valid_unseen, so a gain cannot come from memorising the games
    # the arm trained on.
    # HARD_TYPES is applied to BOTH pools or the arms would train on one
    # difficulty and be scored on another.
    keep = HARD_TYPES if args.hard_types else None
    train_tasks = load_alfworld(args.n_train_tasks, split="train", spread=True,
                                task_types=keep)
    eval_tasks = load_alfworld(args.n_eval_tasks, split="valid_unseen",
                               spread=True, task_types=keep)
    print(f"arm={args.arm} train={len(train_tasks)} eval={len(eval_tasks)}",
          flush=True)

    cfg = ALFWorldPolicyConfig(model=args.served_name, max_turns=args.max_turns,
                               max_tokens_per_turn=args.max_tokens,
                               action_hint=args.action_hint)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.resume_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_adapter,
                                          is_trainable=True)
        print(f"resumed from {args.resume_adapter} at step {args.start_step}",
              flush=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"))
    # Gradient checkpointing is guarded by `if self.gradient_checkpointing and
    # self.training`, and `from_pretrained` returns a model in EVAL mode. The
    # flag reads True while every layer still stores its full activation set --
    # 46.3 GB peak against 6.7 GB once this is called.
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    # One prompt for the whole run: whatever the rollouts were collected under
    # is what the encoder must rebuild, or every turn trains against a context
    # that never existed.
    sys_prompt = system_prompt(args.action_hint)
    sync = LoRASync(args.base_url)
    # Unique per RUN rather than per (arm, seed) -- see the same line in
    # ws_train.py: two runs registering the same adapter name on one sampling
    # service overwrite each other's weights.
    adapter_name = out_dir.name
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id

    def collect_group(tasks, group, seed_base, label=""):
        """Rollouts for one step, keyed by (task, rollout_index)."""
        got = collections.defaultdict(dict)
        want = len(tasks) * group
        done = 0
        t_start = time.time()
        for task, k, rd, err in collect(tasks, group, cfg, args.base_url,
                                        workers=args.collect_workers,
                                        seed_base=seed_base):
            done += 1
            if done % 8 == 0 or done == want:
                print(f"    {label}collect {done}/{want} "
                      f"({done / max(time.time() - t_start, 1e-9) * 60:.1f}/min)",
                      flush=True)
            if err is not None:
                print(f"  ! {task.task_id[:36]} r{k}: {err}", flush=True)
                continue
            got[task.task_id][k] = (task, Rollout.from_dict(rd))
        return got

    def planner_progress(items):
        """a_star per turn, from the PDDL planner.

        Progress alone -- no leave-one-out. The necessity sweep costs
        0.2283 * T^2 and the audit already established that the two agree at
        rho = 0.925 within a trajectory, so paying for both every training step
        would buy nothing.
        """
        anns = annotate_batch(items, gamma=args.gamma, budget=args.max_turns,
                              workers=args.annot_workers, with_necessity=False,
                              timeout=args.annot_timeout)
        return [None if a is None else a["a_star"] for a in anns]

    def evaluate(step):
        got = collect_group(eval_tasks, 1, seed_base=90000 + step, label="eval ")
        rolls = [r for d in got.values() for _, r in d.values()]
        if not rolls:
            return 0.0, 0.0
        succ = sum(r.outcome.success or 0 for r in rolls) / len(rolls)
        turns = sum(r.num_turns for r in rolls) / len(rolls)
        return succ, turns

    # On a resume the new process has never pushed the adapter to the sampler
    # and cfg.model still names the base model, so the very first rollout would
    # be off-policy while the loss assumes ratio = 1 (the same guard exists in
    # ws_train.py). Push the adapter being continued before entering the loop.
    if args.start_step > 0:
        sync.load(adapter_name, str(out_dir / "adapter"))
        sync.verify_active(adapter_name)
        cfg.model = adapter_name
        print(f"pushed resumed adapter to sampler as {adapter_name!r}", flush=True)

    history = []
    for step in range(args.start_step, args.steps + 1):
        t0 = time.time()

        if step % args.eval_every == 0 or step == args.steps:
            succ, turns = evaluate(step)
            print(f"[eval step {step}] success={succ:.1%} turns={turns:.1f}",
                  flush=True)
            history.append({"step": step, "eval_success": succ,
                            "eval_turns": turns})
            with stats_path.open("a") as f:
                f.write(json.dumps(history[-1]) + "\n")
            if step == args.steps:
                break

        # ---- rollouts ----------------------------------------------------
        # Phase timings are logged because an unattended run that stalls looks
        # exactly like one that is merely slow, and the three phases fail for
        # entirely different reasons: sampling on a dead endpoint, annotation on
        # a hung pool, training on an OOM.
        diag = collections.Counter()
        t_phase = time.time()
        # Dynamic sampling (DAPO style). ALFWorld has a binary reward and a
        # group size of 8, so a group is often all-success or all-failure and
        # A_E is then identically 0. The multiplicative form A = A_E * q is
        # identically zero on such a group and learns nothing from it -- the
        # diagnostics measured only 7-9% of turns moved by the signed arm. The
        # additive form A = A_E + w * A_S still has a step term on the same
        # group, and that, rather than dose or the ability to produce a
        # negative sign, is the main reason the two score so differently on
        # ALFWorld. So drop the zero-variance groups and keep drawing tasks
        # until prompts_per_step groups are filled.
        # Applied identically to every arm: GRPO's gradient on such a group is
        # already zero, so dropping it does not change its expected update.
        groups_raw: dict = {}
        n_drawn = n_flat = 0
        cursor = (step * args.prompts_per_step) % max(len(train_tasks), 1)
        max_draws = (args.prompts_per_step * args.dyn_max_factor
                     if args.dynamic_sampling else args.prompts_per_step)
        while len(groups_raw) < args.prompts_per_step and n_drawn < max_draws:
            want = args.prompts_per_step - len(groups_raw)
            batch_tasks = [train_tasks[(cursor + i) % len(train_tasks)]
                           for i in range(want)]
            cursor += want
            n_drawn += want
            got_i = collect_group(batch_tasks, args.group,
                                  seed_base=args.seed * 100000 + step * 1000 + n_drawn,
                                  label=f"step {step} ")
            for task_id, per_k in sorted(got_i.items()):
                ents = [per_k[k] for k in sorted(per_k)]
                rs = [float(r.outcome.reward or 0.0) for _, r in ents]
                if (args.dynamic_sampling and len(ents) >= 2
                        and max(rs) - min(rs) < 1e-9):
                    n_flat += 1
                    continue
                groups_raw[task_id] = per_k
            if not args.dynamic_sampling:
                break
        got = groups_raw if groups_raw else got_i

        t_collect = time.time() - t_phase
        t_phase = time.time()

        # ---- ground truth for the whole step, in one pool ----------------
        # Per-group annotation spawned a fresh 24-worker pool for every task in
        # the batch -- four pool startups a step, each a few seconds, for work
        # that shares one pool perfectly well.
        groups = []
        for task_id, per_k in sorted(got.items()):
            entries = [per_k[k] for k in sorted(per_k)]
            if len(entries) >= 2 and all(r.num_turns for _, r in entries):
                groups.append((task_id, entries))

        a_star_by_group = {}
        if args.arm == "anchor_cf" and groups:
            # Measured in the PARENT process, not the annotation pool. The
            # batched replay needs `asynchronous=True` for correctness -- the
            # synchronous batch shares one game across its slots and reports
            # every ablation as a win -- and pool workers are daemonic, so they
            # cannot spawn the children that path requires.
            from ssc.credit.anchor_cf import anchor_deltas
            n_abl = 0
            for task_id, entries in groups:
                acts = [[str((t.tool_args or {}).get("action", ""))
                         for t in roll.turns] for _, roll in entries]
                obs = [[{"observation": t.observation,
                         "tool_args": t.tool_args} for t in roll.turns]
                       for _, roll in entries]
                try:
                    cred = anchor_deltas(entries[0][0].game_file, acts, obs,
                                         budget=args.max_turns,
                                         measure_losers=args.measure_losers)
                    a_star_by_group[task_id] = cred.deltas
                    n_abl += cred.n_ablations
                except Exception as e:  # noqa: BLE001
                    print(f"  ! anchor_cf failed on {task_id[:32]}: "
                          f"{type(e).__name__}: {e}", flush=True)
            if n_abl:
                print(f"    anchor_cf: {n_abl} replays over {len(groups)} groups",
                      flush=True)

        for gid, (task_id, entries) in enumerate(groups):
            rolls = [r for _, r in entries]
            rewards = [float(r.outcome.reward or 0.0) for r in rolls]
            lengths = [r.num_turns for r in rolls]

            kw = {}
            if args.arm == "anchor_cf":
                a_star = a_star_by_group.get(task_id, [])
                if len(a_star) != len(rolls):
                    print(f"  ! planner returned {len(a_star)} rows for "
                          f"{len(rolls)} rollouts on {task_id[:32]}; group skipped",
                          flush=True)
                    continue
                # A trajectory the planner could not score falls back to a FLAT
                # step signal -- which is exactly GRPO for that trajectory --
                # rather than taking its group out of the batch.
                #
                # Dropping the group was the first version and it was quietly
                # unfair. Fast Downward's slow instances are not spread evenly:
                # every timeout observed landed on `pick_clean_then_place_in_recep`,
                # so an arm that depends on the planner was losing one task
                # family while the others trained on it. The arms would then
                # have differed in WHICH TASKS THEY SAW as well as in their step
                # signal, and no comparison survives that.
                #
                # Replay can also end earlier than the record if an action was
                # rejected, so rows are padded to the recorded length.
                n_missing = sum(1 for a in a_star if a is None)
                if n_missing:
                    print(f"  ~ planner missed {n_missing}/{len(rolls)} rollouts "
                          f"on {task_id[:32]}; those fall back to flat credit",
                          flush=True)
                kw["a_star"] = [([0.0] * L if a is None
                                 else list(a) + [0.0] * (L - len(a)))
                                for a, L in zip(a_star, lengths)]

            per_turn = turn_advantages(args.arm, rewards, lengths,
                                       gamma=args.gamma, omega=args.omega,
                                       alpha=args.alpha, dose=args.dose,
                                       fallback=args.mass_fallback, fallback_omega=args.fallback_omega,
                                       nonneg_winners=args.nonneg_winners,
                                       zero_uniform_groups=args.zero_uniform_groups, **kw)

            # Intermediate-quantity diagnostics. |adv| and |g| alone do not
            # show the mechanism: on ALFWorld the multiplicative form is nearly
            # identical to GRPO (|adv| 0.332 against 0.343) while the additive
            # form is 0.728. What has to be watched is HOW FAR the advantages
            # moved relative to GRPO and WHETHER a negative advantage appears
            # inside a successful trajectory. The latter is the mechanism the
            # additive form wins by, and a sign-preserving multiplicative form
            # cannot produce it by construction.
            try:
                _base = turn_advantages("grpo", rewards, lengths)
                for _r, _b in zip(per_turn, _base):
                    _e = _b[0] if _b else 0.0
                    for _a, _bb in zip(_r, _b):
                        diag["n_turn"] += 1
                        diag["dose"] += abs(_a - _bb)
                        diag["n_moved"] += abs(_a - _bb) > 1e-9
                        if _e > 0:
                            diag["n_pos"] += 1
                            diag["n_neg_in_pos"] += _a < -1e-9
                    if _e > 0 and _r:
                        diag["q_spread"] += (max(_r) - min(_r)) / abs(_e)
                        diag["n_pos_roll"] += 1
                for _row in kw.get("a_star", []):
                    for _v in _row:
                        diag["n_delta"] += 1
                        diag["n_delta_nz"] += abs(_v) > 1e-9
                        diag["delta_abs"] += abs(_v)
            except Exception:  # noqa: BLE001  diagnostics must never break training
                pass

            for (task, roll), adv_row in zip(entries, per_turn):
                exs = encode_turns(roll, tokenizer, sys_prompt, "alfworld",
                                   args.max_seq_len, gid)
                for ex in exs:
                    if ex.turn_index < len(adv_row):
                        examples.append(ex)
                        adv_values.append(adv_row[ex.turn_index])
                n_roll += 1
                n_won += roll.outcome.success or 0

        if not examples:
            print(f"[step {step}] no usable examples; skipping", flush=True)
            continue

        # ---- update ------------------------------------------------------
        # Micro-batches are sized by TOKEN BUDGET, not by example count.
        # ALFWorld turn examples run from ~3k tokens early in an episode to the
        # 12k cap late in one, so a fixed count batches wildly different
        # objects: micro_batch=4 fit for most of a step and then died with
        # `OutOfMemoryError: tried to allocate 5.25 GiB` on the one chunk that
        # happened to draw four long ones. Activation memory tracks tokens, so
        # bounding tokens bounds the peak. Sorting by length also keeps padding
        # waste down, since a chunk is padded to its longest member.
        order = sorted(range(len(examples)),
                       key=lambda i: len(examples[i].input_ids))
        chunks, cur, cur_tok = [], [], 0
        for i in order:
            n_tok = len(examples[i].input_ids)
            if cur and cur_tok + n_tok > args.token_budget:
                chunks.append(cur)
                cur, cur_tok = [], 0
            cur.append(i)
            cur_tok += n_tok
        if cur:
            chunks.append(cur)

        # Each chunk's loss is a token-mean over ITS OWN actor tokens, so the
        # step objective is only the token-mean over all of them if the chunks
        # are weighted by their share of actor tokens. Dividing by the chunk
        # COUNT instead weighted every chunk alike, and the chunks are anything
        # but alike: built to a fixed budget of INPUT tokens, they held 231 to
        # 3497 actor tokens each, so per-token weight varied 15x across a step.
        #
        # The distortion was not random either. Chunks are length-sorted, a
        # turn's context grows with its index while its generation does not, so
        # the over-weighted chunks were the late-turn ones -- correlation +0.49
        # between a chunk's over-weighting and its length rank. That silently
        # applied a turn-position prior to EVERY arm, including the GRPO
        # baseline, and turn position by itself tracks causal progress at
        # rho = +0.464. A baseline quietly receiving the very signal the
        # experiment is trying to isolate compresses the difference it is
        # trying to measure.
        chunk_tokens = [sum(examples[i].n_actor_tokens for i in idx)
                        for idx in chunks]
        total_tokens = max(sum(chunk_tokens), 1)

        optimizer.zero_grad(set_to_none=True)
        n_chunks = max(1, len(chunks))
        total_loss = 0.0
        n_oom = 0

        def run_chunk(indices, weight) -> float:
            b = collate([examples[i] for i in indices], pad).to(model.device)
            adv = torch.tensor([adv_values[i] for i in indices],
                               dtype=torch.float32, device=model.device)
            # One gradient step per batch of rollouts, so the behaviour policy
            # IS the current policy: old_lp equals new_lp at these parameters,
            # the ratio is exactly 1 and the clip is inactive. Detaching one
            # forward pass is therefore exact, and removes a whole extra forward
            # per micro-batch. Restore the separate no_grad pass the moment this
            # loop does more than one inner epoch, or the ratio stops meaning
            # anything.
            new_lp = token_logprobs(model, b.input_ids, b.attention_mask)
            old_lp = new_lp.detach()
            loss = grpo_loss(b.rewards, b.actor_mask, old_lp, new_lp,
                             b.group_ids, args.clip_eps, advantages=adv)
            (loss * weight).backward()
            return float(loss.detach())

        for idx, n_tok in zip(chunks, chunk_tokens):
            # An OOM must not end a multi-hour unattended run. The chunk is
            # retried one example at a time; anything that still fails is
            # dropped and COUNTED, because a step that quietly trained on half
            # its data would otherwise look exactly like a clean one.
            try:
                total_loss += run_chunk(idx, n_tok / total_tokens)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                for j in idx:
                    try:
                        total_loss += run_chunk(
                            [j], examples[j].n_actor_tokens / total_tokens)
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        n_oom += 1

        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        adapter = out_dir / "adapter"
        model.save_pretrained(adapter)
        # Keep a copy at every evaluation point. Overwriting one adapter per
        # step was enough until a run collapsed: grpo peaked at 83.3% on step 10
        # and fell to 20.8% by step 20, and by then the step-10 weights -- the
        # only ones worth evaluating -- had been overwritten nine times. A
        # learning curve that cannot be revisited is a curve that has to be
        # believed on the strength of a 24-task eval, which is exactly what the
        # noise floor says cannot be done.
        if step % args.eval_every == 0:
            ckpt = out_dir / f"adapter_step{step}"
            model.save_pretrained(ckpt)
        sync.load(adapter_name, str(adapter))
        sync.verify_active(adapter_name)
        # Point the SAMPLING policy at the adapter. Without this the rollouts
        # keep naming the base model, vLLM keeps serving the base weights, and
        # every step trains on trajectories from an untrained policy while the
        # adapter loads successfully and the logs read normally.
        cfg.model = adapter_name

        rec = {"step": step, "arm": args.arm, "loss": total_loss / n_chunks,
               "t_collect_s": round(t_collect, 1), "t_annot_s": round(t_annot, 1),
               "t_train_s": round(time.time() - t_phase, 1),
               "grad_norm": float(gnorm), "n_rollouts": n_roll,
               "train_success": n_won / max(n_roll, 1),
               "n_examples": len(examples),
               "mean_abs_adv": sum(abs(a) for a in adv_values) / len(adv_values),
               # Mechanism diagnostics. dose = mean per-turn shift of the
               # advantage relative to GRPO; neg_in_success = fraction of turns
               # in successful trajectories pushed to a negative advantage (the
               # mechanism the additive form wins by); q_spread = range of the
               # weights inside a successful trajectory; delta_* = density and
               # magnitude of the measurement itself.
               "dose": round(diag["dose"] / max(diag["n_turn"], 1), 4),
               "moved_frac": round(diag["n_moved"] / max(diag["n_turn"], 1), 3),
               "neg_in_success": round(diag["n_neg_in_pos"] / max(diag["n_pos"], 1), 3),
               "q_spread": round(diag["q_spread"] / max(diag["n_pos_roll"], 1), 3),
               "delta_nonzero": round(diag["n_delta_nz"] / max(diag["n_delta"], 1), 3),
               "delta_abs_mean": round(diag["delta_abs"] / max(diag["n_delta"], 1), 4),
               "n_groups_drawn": n_drawn, "n_groups_flat": n_flat,
               "n_chunks": n_chunks, "n_oom_dropped": n_oom,
               "wall_s": round(time.time() - t0, 1)}
        history.append(rec)
        with stats_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[step {step}] loss={rec['loss']:+.4f} |g|={rec['grad_norm']:.3f} "
              f"train_succ={rec['train_success']:.0%} "
              f"ex={rec['n_examples']} |adv|={rec['mean_abs_adv']:.3f} "
              f"dose={rec['dose']:.3f} moved={rec['moved_frac']:.0%} "
              f"neg_in_success={rec['neg_in_success']:.0%} q_spread={rec['q_spread']:.2f} "
              f"delta_nz={rec['delta_nonzero']:.0%}/{rec['delta_abs_mean']:.3f} "
              + (f"drawn={rec['n_groups_drawn']} flat={rec['n_groups_flat']} " if args.dynamic_sampling else "")
              + f"({rec['t_collect_s']:.0f}s collect / {rec['t_annot_s']:.0f}s annot "
              + f"/ {rec['t_train_s']:.0f}s train)"
              + (f"  OOM-dropped={n_oom}" if n_oom else ""), flush=True)

    print(f"done -> {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
