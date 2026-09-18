#!/usr/bin/env python
"""GRPO / measured-credit training on multi-hop QA.

Two environments share this trainer:

  --env local   HotpotQA distractor: BM25 over the question's own ten paragraphs
                (the earlier, negative-result line; F1 reward, 5 turns, top-2).
  --env wiki    Search-R1 setting: BM25 over all of Wikipedia through
                the frozen index service, top-3 passages, 4 turns, EM reward,
                NQ + HotpotQA training questions, the paper's seven test sets.

Third environment, so this file is written against the list of things the first
two cost to learn, each of which is a silent failure that logs normally:

  model.train() after get_peft_model   gradient checkpointing is guarded by
                                       `self.training`; peft returns eval mode.
  sync the adapter EVERY step          `run_update` sets old_lp = new_lp because
                                       one update per batch makes the behaviour
                                       policy the current policy.
  replay the shown context verbatim    WebShop rebuilt prompts from stored
                                       pieces and one was captured a step late.
  chunk by PADDED width                a chunk is collated to its longest member.
  keep a checkpoint per eval           a run that peaks and collapses leaves
                                       nothing to evaluate otherwise.

The measured-credit arms and the training hygiene (anneal, dynamic sampling,
lr decay, per-turn clip) are the WebShop `L_mxS` recipe transplanted; see
`ssc/credit/multihop_cf.py` for what is measured here and why.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

# Arms that need a counterfactual measurement (including the grouped form, which
# feeds the measurement into a group-normalised step term).
DELTA_ARMS = ("anchor_cf", "anchor_cvmax")
# Arms credited by "the value of answering now, V*" instead of by ablation.
# Arms credited by "the value of answering now, V" instead of by ablation. The
# combined arms belong here too: in the open-domain setting, dropping one piece
# of evidence and re-answering (multihop_deltas) is a much weaker signal than
# the commit value, so both combined arms must use the same operator as L_mxS --
# only then is the sole difference between them how the measurement and the
# step term are put together.
CV_ARMS = ("anchor_cvmax",)


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=("grpo", "anchor_cf", "anchor_cvmax"))
    ap.add_argument("--env", choices=("local", "wiki"), default="local")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--prompts_per_step", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max_turns", type=int, default=None,
                    help="default local=5, wiki=4 (paper: at most 3 searches + answer)")
    ap.add_argument("--top_k", type=int, default=None, help="default local=2, wiki=3")
    ap.add_argument("--doc_chars", type=int, default=None, help="default local=500, wiki=1200")
    ap.add_argument("--wiki_url", default="http://127.0.0.1:8080")
    ap.add_argument("--reward", choices=("f1", "em"), default=None,
                    help="episode reward; default local=f1, wiki=em (the Search-R1 setting uses EM)")
    ap.add_argument("--fixed_reader", action="store_true",
                    help="run the counterfactual measurement with the **base model** as the "
                         "reader, rather than with the current policy, which drifts during "
                         "training. Rationale: V(E_t) is 'how well one can answer right now "
                         "with this evidence'; measured with the current policy, the more the "
                         "policy drifts the more the ruler drifts with it. Measured: our "
                         "validation-to-test drop was -0.051 against -0.016 for a step-level "
                         "baseline that estimates credit from observed returns, with the "
                         "largest losses out of domain (2Wiki and the like). A fixed reader "
                         "makes V a ruler that does not deform as training proceeds.")
    ap.add_argument("--measure", choices=("f1", "em"), default="f1",
                    help="score used by the counterfactual measurement (V and the regret term). "
                         "F1 is denser; the reward still follows --reward")
    ap.add_argument("--n_train_tasks", type=int, default=2000)
    ap.add_argument("--n_eval_tasks", type=int, default=120)
    ap.add_argument("--eval_split", default=None,
                    help="default local=validation, wiki=val (unseen questions held out of "
                         "the training pool)")
    ap.add_argument("--eval_temperature", type=float, default=0.0)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr_final", type=float, default=None,
                    help="decay the learning rate linearly to this value (default: no decay)")
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--sim_threshold", type=float, default=0.9,
                    help="similarity threshold for grouping anchor states (0.9 in the paper's "
                         "QA setting; <=0 means exact match)")
    ap.add_argument("--no_step_std", action="store_true")
    ap.add_argument("--dose", type=float, default=1.0,
                    help="dose for anchor_mass_signed: 0 degenerates exactly to GRPO, >1 "
                         "produces negative signs")
    ap.add_argument("--fallback_omega", type=float, default=None,
                    help="scale of the step term on the fallback branch (groups with A_E=0); "
                         "defaults to --omega. Use it to hold the magnitude down when three "
                         "terms stack: on QA, omega=1 doubles |adv| and the run collapses.")
    ap.add_argument("--mass_fallback", action="store_true",
                    help="groups whose outcomes are all identical fall back to the additive "
                         "step term (the switch for the unified form)")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="dose for the multiplicative arm: at 1.0 the weight is constantly 1 "
                         "and it degenerates exactly to GRPO")
    ap.add_argument("--w_max", type=float, default=3.0)
    ap.add_argument("--anneal_steps", type=int, default=None,
                    help="anneal the step term: omega decays linearly to 0, alpha rises "
                         "linearly to 1, pure GRPO afterwards")
    ap.add_argument("--regret_lambda", type=float, default=0.0,
                    help="coefficient of the measured relative-value term over sibling "
                         "queries in the same group (0=off). Can be stacked on any arm")
    ap.add_argument("--search_credit", choices=("centred", "regret"), default="centred",
                    help="centred=Q(own)-mean(pool); regret=min(0, Q(own)-max(pool)), where the "
                         "pool is the sibling candidates in the same state")
    ap.add_argument("--no_answer_candidate", action="store_true",
                    help="do not treat 'answer now' as a candidate in the relative-value term")
    ap.add_argument("--unmask_answer", action="store_true",
                    help="for the within-anchor-group form only: the answer turn enters the "
                         "same-state comparison with 0 (answer early vs search again)")
    ap.add_argument("--answer_credit", choices=("mask", "zero", "prior_residual"), default=None,
                    help="measurement on the answer turn: mask=not measurable; zero=enters with "
                         "0; prior_residual=closed-book value + extraction residual (closes the "
                         "telescoping decomposition). Default anchor_cvmax=prior_residual, "
                         "grouped=zero")
    ap.add_argument("--answer_lookahead", action="store_true",
                    help="relative-value term: on the answer turn, add a lookahead candidate "
                         "for 'the policy searches once more' (measured by replay)")
    ap.add_argument("--nonneg_winners", action="store_true",
                    help="clip the step term to >=0 on trajectories with positive advantage: "
                         "surplus turns fall back to A_E instead of being penalised "
                         "(on ALFWorld 4B this lifted ours from 39.5 to 60.9)")
    ap.add_argument("--question_candidate", action="store_true",
                    help="relative-value term: at the initial state, add the fixed candidate "
                         "'use the question text verbatim as the query'")
    ap.add_argument("--repeat_penalty", type=float, default=0.0,
                    help="negative credit of this size for a search that brought back no new "
                         "documents (a repeated query) (0=off)")
    ap.add_argument("--answer_regret", action="store_true",
                    help="on the answer turn, add the extraction regret "
                         "min(0, EM(sampled answer) - EM(greedy short answer on the same evidence))")
    ap.add_argument("--no_dilute", action="store_true",
                    help="keep the appended zero-variance groups from diluting the outcome "
                         "gradient: advantages are scaled by all_tokens/quota_group_tokens, so "
                         "the token-mean gradient of the quota groups matches the baseline")
    ap.add_argument("--no_running_max", action="store_true",
                    help="use V(t+1)-V(t) instead of V*(t+1)-V*(t) (may be negative). This is "
                         "already the default in the wiki setting")
    ap.add_argument("--running_max", action="store_true",
                    help="force the running maximum V* in the wiki setting (off by default: "
                         "evidence accumulates, and the harm done by a distractor passage is real)")
    ap.add_argument("--dynamic_sampling", action="store_true",
                    help="drop groups whose outcomes are all identical and whose measurement "
                         "shows no signal, and keep drawing questions; keep those with a "
                         "counterfactual signal")
    ap.add_argument("--dyn_max_factor", type=int, default=4)
    ap.add_argument("--flat_extra_max", type=int, default=8,
                    help="how many groups with identical outcomes but a measured signal may be "
                         "appended per step (they do not use up the quota)")
    ap.add_argument("--adv_clip", type=float, default=None,
                    help="clip the per-turn advantage to [-c, c]")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--token_budget", type=int, default=16384)
    ap.add_argument("--anneal_regret", action="store_true",
                    help="let the regret term decay linearly to zero along with\n"
                         "anneal_steps. Off by default: in every arm so far the regret\n"
                         "coefficient was a constant applied after annealing, so once\n"
                         "annealing ended the arm was 'GRPO + lambda*regret' rather than\n"
                         "pure GRPO. Turning it on matches the stated contract of\n"
                         "anneal_dose; but ws_L_mxann_s0 (regret off throughout) reached a\n"
                         "gradient norm of 20 at step 90 on WebShop and collapsed, so the\n"
                         "default does not change the behaviour of existing arms.")
    ap.add_argument("--lr_warmup_frac", type=float, default=0.0,
                    help="fraction of the total steps spent on linear learning-rate warmup "
                         "(Search-R1 uses 0.285, the search variant of a step-level baseline "
                         "uses 0.1; default 0 = off, consistent with existing arms)")
    ap.add_argument("--max_seq_len", type=int, default=8192)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_path",
                    default=os.environ.get("SSC_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--served_name", default="Qwen3-8B")
    ap.add_argument("--base_url", default="http://localhost:8100/v1")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume_adapter", default=None,
                    help="continue training from a saved adapter. **The optimizer state is "
                         "not restored**: AdamW's first and second moments start from zero "
                         "again, so the effective step size is too large for the first step "
                         "or two after resuming. That is a real discontinuity, written down "
                         "rather than hidden.")
    ap.add_argument("--start_step", type=int, default=0,
                    help="step number to continue from; the evaluation cadence follows it, so "
                         "step numbers still line up with an arm that was run in one go.")
    args = ap.parse_args()

    wiki = args.env == "wiki"
    if args.max_turns is None:
        args.max_turns = 4 if wiki else 5
    if args.top_k is None:
        args.top_k = 3 if wiki else 2
    if args.doc_chars is None:
        args.doc_chars = 1200 if wiki else 500
    if args.reward is None:
        args.reward = "em" if wiki else "f1"
    if args.eval_split is None:
        args.eval_split = "val" if wiki else "validation"
    if args.answer_credit is None:
        args.answer_credit = ("zero" if (args.unmask_answer or args.arm == "anchor_cvmax_grouped")
                              else "prior_residual")
    running_max = args.running_max or (not wiki and not args.no_running_max)
    args.running_max_effective = running_max

    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ssc.credit.alf_arms import anneal_dose, turn_advantages
    from ssc.credit.grpo_patch import grpo_loss
    from ssc.credit.multihop_cf import (ANSWER_SYSTEM, ANSWER_SYSTEM_OPEN,
                                        multihop_commit_deltas, multihop_deltas,
                                        multihop_regret)
    from ssc.env.multihop_policy import extract_action
    from ssc.train.batch import _multihop_history
    from ssc.env.multihop_env import load_multihop, load_wikiqa
    from ssc.env.multihop_policy import MultiHopPolicy, MultiHopPolicyConfig
    from ssc.train.batch import collate, encode_turns, token_logprobs
    from ssc.train.step import plan_chunks, run_update
    from ssc.train.sync import LoRASync

    prefix = "wq" if wiki else "mh"
    tag = args.tag or args.arm
    out_dir = Path(args.out_dir or ROOT / "runs" / f"{prefix}_{tag}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    stats_path = out_dir / "steps.jsonl"

    torch.manual_seed(args.seed)
    if wiki:
        train_tasks = load_wikiqa(args.n_train_tasks, split="train")
        eval_tasks = load_wikiqa(args.n_eval_tasks, split=args.eval_split)
    else:
        train_tasks = load_multihop(args.n_train_tasks, split="train")
        eval_tasks = load_multihop(args.n_eval_tasks, split=args.eval_split)
    print(f"arm={args.arm} env={args.env} reward={args.reward} train={len(train_tasks)} "
          f"eval={len(eval_tasks)} ({args.eval_split})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.resume_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
        print(f"resumed from {args.resume_adapter} at step {args.start_step}", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"))
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id

    def set_lr(step: int) -> float:
        """Warm up first, then decay linearly to lr_final (both off by default, so
        existing arms are unchanged).

        Warmup is the one thing both reference setups use and we did not have:
        Search-R1's GRPO uses `lr_warmup_steps_ratio=0.285`, and the search
        variant of a step-level baseline uses 0.1. We compute each gradient from
        only 8 questions (they use 512 / 256), so the gradient noise is far
        larger: on 8B the median |g| is 6.75 and 98% of steps are clipped at
        max_grad_norm=1.0. Running at the full learning rate from the very first
        step therefore means several fixed-size jumps in a row along a noise
        direction.
        """
        if args.steps <= 0:
            return args.lr
        warm = int(round(args.lr_warmup_frac * args.steps))
        if warm > 0 and step < warm:
            lr = args.lr * (step + 1) / warm
        elif args.lr_final is None:
            lr = args.lr
        else:
            denom = max(args.steps - warm, 1)
            frac = min(1.0, max(0.0, (step - warm) / denom))
            lr = args.lr + (args.lr_final - args.lr) * frac
        for g in optimizer.param_groups:
            g["lr"] = lr
        return lr

    adapter_name = f"{prefix}_{tag}_s{args.seed}"
    sync = LoRASync(args.base_url)
    common = dict(model=args.served_name, max_turns=args.max_turns,
                  max_tokens_per_turn=args.max_tokens,
                  retriever="wiki" if wiki else "local", wiki_url=args.wiki_url,
                  top_k=args.top_k, doc_chars=args.doc_chars, reward=args.reward)
    cfg = MultiHopPolicyConfig(temperature=1.0, top_p=0.95, **common)
    policy = MultiHopPolicy(cfg, args.base_url)
    eval_cfg = MultiHopPolicyConfig(temperature=args.eval_temperature,
                                    top_p=1.0 if args.eval_temperature == 0 else 0.95,
                                    **common)
    eval_policy = MultiHopPolicy(eval_cfg, args.base_url)
    if wiki:
        eval_policy.wiki = policy.wiki          # share the retrieval cache
    SYSTEM_PROMPT = policy.system_prompt
    answer_system = ANSWER_SYSTEM_OPEN if wiki else ANSWER_SYSTEM
    client = OpenAI(base_url=args.base_url, api_key="x")

    def greedy(messages) -> str:
        # With `--fixed_reader`, read with the base model; otherwise keep using the
        # current policy (cfg.model is synced to the adapter name).
        model = args.served_name if args.fixed_reader else cfg.model
        r = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0, top_p=1.0,
            max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return (r.choices[0].message.content or "").strip().split("\n")[0].strip()

    NUDGE = ("Before answering, do one more search. Reply with exactly one line: "
             "Action: search[your query]")

    def propose_query(roll, state) -> str | None:
        """Ask the current policy (greedily) for one more query, from the context
        it saw just before answering: the lookahead candidate."""
        ans = next((i for i, t in enumerate(roll.turns)
                    if (t.tool_args or {}).get("kind") == "answer"), None)
        if ans is None:
            return None
        msgs = _multihop_history(roll, ans, SYSTEM_PROMPT) + [{"role": "user", "content": NUDGE}]
        r = client.chat.completions.create(
            model=cfg.model, messages=msgs, temperature=0.0, top_p=1.0, max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        act = extract_action((r.choices[0].message.content or "").strip())
        return act[1] if act and act[0] == "search" else None

    def collect(tasks, group, seed_base, pol=None):
        pol = pol or policy
        jobs = [(t, k) for t in tasks for k in range(group)]

        def one(job):
            t, k = job
            try:
                return (t, *pol.rollout(t, rollout_index=k, seed=seed_base + k))
            except Exception as e:  # noqa: BLE001
                print(f"  ! {t.task_id} k={k}: {type(e).__name__}: {e}", flush=True)
                return None

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            return [r for r in ex.map(one, jobs) if r is not None]

    def evaluate(step: int) -> dict:
        got = collect(eval_tasks, 1, seed_base=90000 + step, pol=eval_policy)
        if not got:
            return {"eval_score": 0.0, "eval_success": 0.0, "eval_f1": 0.0}
        by_src = collections.defaultdict(list)
        for t, r, s in got:
            by_src[t.source or "all"].append((float(r.outcome.reward or 0.0), s.em, s.f1))
        rec = {"eval_score": sum(r.outcome.reward or 0.0 for _, r, _ in got) / len(got),
               "eval_success": sum(s.em for _, _, s in got) / len(got),
               "eval_f1": sum(s.f1 for _, _, s in got) / len(got),
               "eval_turns": sum(len(r.turns) for _, r, _ in got) / len(got),
               "eval_no_answer": sum(1 for _, r, _ in got if r.stop_reason) / len(got),
               "eval_n": len(got)}
        if len(by_src) > 1:
            rec["eval_by_source"] = {k: {"em": round(sum(x[1] for x in v) / len(v), 4),
                                         "f1": round(sum(x[2] for x in v) / len(v), 4),
                                         "n": len(v)} for k, v in sorted(by_src.items())}
        return rec

    need_deltas = args.arm in DELTA_ARMS
    need_regret = args.regret_lambda > 0

    def annotate(entries):
        """Counterfactual measurement for one group (several rollouts of the same
        question). Returns (deltas_credit, regret_credit)."""
        states = [s for _, _, s in entries]
        rolls = [r for _, r, _ in entries]
        cache: dict = {}
        cr = reg = None
        if need_deltas:
            if args.arm in CV_ARMS:
                cr = multihop_commit_deltas(greedy, states, rolls, cache=cache,
                                            running_max=running_max,
                                            metric=args.measure,
                                            answer_credit=args.answer_credit,
                                            repeat_penalty=args.repeat_penalty,
                                            answer_system=answer_system)
            else:
                cr = multihop_deltas(greedy, states, rolls, cache=cache, metric=args.measure)
        if need_regret:
            reg = multihop_regret(greedy, states, rolls, cache=cache, mode=args.search_credit,
                                  include_answer=not args.no_answer_candidate,
                                  metric=args.measure, running_max=running_max,
                                  propose_query=propose_query if args.answer_lookahead else None,
                                  question_candidate=args.question_candidate,
                                  answer_system=answer_system,
                                  answer_regret_metric="em" if args.answer_regret else None)
        return cr, reg

    def annotate_groups(pairs):
        if not pairs:
            return []

        def one(item):
            tid, entries = item
            try:
                return tid, *annotate(entries)
            except Exception as e:  # noqa: BLE001
                print(f"  ! deltas failed on {tid}: {type(e).__name__}: {e}", flush=True)
                return tid, None, None
        with ThreadPoolExecutor(max_workers=min(args.workers, len(pairs))) as ex:
            return list(ex.map(one, pairs))

    def has_signal(cr, reg) -> bool:
        ok = False
        if cr is not None:
            ok = any(abs(v) > 1e-9 for row, m in zip(cr.deltas, cr.mask)
                     for v, k in zip(row, m) if k)
        if reg is not None:
            ok = ok or any(abs(v) > 1e-9 for row, m in zip(reg.deltas, reg.mask)
                           for v, k in zip(row, m) if k)
        return ok

    def retrieval_alive() -> bool:
        """Is the retrieval service alive? It died once, on 09-06 at 22:50, from a
        JVM SIGSEGV (the single-process ThreadingHTTPServer starts one thread per
        request and crashed inside Lucene's JNI call on thread 286650). A dead
        retriever does not make training fail: search returns nothing, the policy
        answers anyway, and dozens of steps train on empty evidence. Better to
        stop and wait than to feed garbage into the gradient."""
        if not wiki:
            return True
        try:
            import urllib.request
            with urllib.request.urlopen(f"{args.wiki_url}/manifest", timeout=8) as r:
                return b'"ok"' in r.read(200)
        except Exception:  # noqa: BLE001
            return False

    if args.resume_adapter:
        # On resume the weights must be pushed to the sampler BEFORE entering the
        # loop. Otherwise the first step's rollouts, and that step's evaluation,
        # still sample from the base model while the trainer already holds the
        # resumed weights: the behaviour policy and the current policy disagree,
        # run_update's ratio=1 premise is broken, and we update a policy that has
        # already trained for 60 steps using trajectories from the base model. The
        # observed symptom is that the evaluation at the resumed step falls back
        # to base level (0.285).
        sync.load(adapter_name, args.resume_adapter)
        sync.verify_active(adapter_name)
        cfg.model = adapter_name
        eval_cfg.model = adapter_name
        print(f"pushed the resumed weights to the sampler: {adapter_name}", flush=True)

    rng = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(train_tasks), generator=rng).tolist()
    cursor = args.start_step * args.prompts_per_step
    history = []
    for step in range(args.start_step, args.steps + 1):
        waited = 0
        while not retrieval_alive():
            if waited == 0:
                print(f"[step {step}] retrieval service unreachable, pausing (retry every 60 s)",
                      flush=True)
            time.sleep(60)
            waited += 1
            if waited >= 120:
                print(f"[step {step}] retrieval service down for two hours, exiting", flush=True)
                return 2
        if waited:
            print(f"[step {step}] retrieval service is back after {waited} min, continuing",
                  flush=True)
        if step % args.eval_every == 0:
            if step > 0:
                model.save_pretrained(str(out_dir / f"adapter_step{step}"))
            ev = evaluate(step)
            print(f"[eval step {step}] score={ev['eval_score']:.3f} em={ev['eval_success']:.1%} "
                  f"f1={ev['eval_f1']:.3f} turns={ev.get('eval_turns', 0):.2f}"
                  + (f" by_source={ {k: v['em'] for k, v in ev['eval_by_source'].items()} }"
                     if 'eval_by_source' in ev else ""), flush=True)
            history.append({"step": step, **ev})
            with stats_path.open("a") as f:
                f.write(json.dumps(history[-1]) + "\n")
        if step == args.steps:
            break

        # ---- Rollouts (optional dynamic sampling: a group with no outcome
        # variance and no counterfactual signal is dropped and redrawn) ----
        t0 = time.time()
        groups: dict = collections.OrderedDict()
        cr_by_group: dict = {}
        reg_by_group: dict = {}
        n_drawn = n_flat_dropped = n_flat_kept = 0
        n_calls = 0
        max_draws = (args.prompts_per_step * args.dyn_max_factor
                     if args.dynamic_sampling else args.prompts_per_step)
        last_fresh: dict = {}
        # First fill prompts_per_step groups that carry an outcome signal (A_E
        # nonzero) -- that part is identical to the baseline. Groups whose outcomes
        # are all identical but that do carry a measured signal are APPENDED on top
        # (at most flat_extra_max of them) and do not use up the quota. Under the
        # EM reward, 78% of groups have identical outcomes; if those used up the
        # quota, each step would keep only 1 group with an outcome signal while the
        # baseline has 8 -- that would be a difference in batch size, not a
        # difference between methods.
        extras: dict = collections.OrderedDict()
        while len(groups) < args.prompts_per_step and n_drawn < max_draws:
            want = args.prompts_per_step - len(groups)
            batch = [train_tasks[order[(cursor + i) % len(order)]] for i in range(want)]
            cursor += want
            n_drawn += want
            got = collect(batch, args.group,
                          seed_base=args.seed * 100000 + step * 1000 + n_drawn)
            fresh: dict = collections.OrderedDict()
            for task, roll, state in got:
                fresh.setdefault(task.task_id, []).append((task, roll, state))
            last_fresh = fresh
            flat: dict = {}
            for tid, entries in fresh.items():
                rs = [float(r.outcome.reward or 0.0) for _, r, _ in entries]
                if (args.dynamic_sampling and len(entries) >= 2
                        and max(rs) - min(rs) < 1e-9):
                    flat[tid] = entries
                else:
                    groups[tid] = entries
            room = args.flat_extra_max - len(extras)
            if flat and room > 0 and (need_deltas or need_regret):
                for tid, cr, reg in annotate_groups(list(flat.items())):
                    if cr is not None:
                        cr_by_group[tid] = cr
                        n_calls += cr.n_calls
                    if reg is not None:
                        reg_by_group[tid] = reg
                        n_calls += reg.n_calls
                    if has_signal(cr, reg) and len(extras) < args.flat_extra_max:
                        extras[tid] = flat[tid]
                        n_flat_kept += 1
                    else:
                        n_flat_dropped += 1
            else:
                n_flat_dropped += len(flat)
            if not args.dynamic_sampling:
                break
        if not groups and not extras:
            groups.update(last_fresh)
        n_quota_groups = len(groups)
        groups.update(extras)
        t_collect = time.time() - t0
        if not groups:
            print(f"[step {step}] no rollouts; skipping", flush=True)
            continue

        # ---- Counterfactual measurement (the remaining groups) ----
        t0 = time.time()
        if need_deltas or need_regret:
            todo = [(tid, ent) for tid, ent in groups.items()
                    if tid not in cr_by_group and tid not in reg_by_group]
            for tid, cr, reg in annotate_groups(todo):
                if cr is not None:
                    cr_by_group[tid] = cr
                    n_calls += cr.n_calls
                if reg is not None:
                    reg_by_group[tid] = reg
                    n_calls += reg.n_calls
        t_annot = time.time() - t0
        if n_calls:
            print(f"    {args.arm}: {n_calls} regenerations over {len(groups)} groups",
                  flush=True)

        # ---- Advantages ----
        examples, adv_values = [], []
        n_step_nonzero = n_step_total = 0
        n_tok_quota = n_tok_all = 0
        n_roll, total_score, total_em, total_turns = 0, 0.0, 0.0, 0
        omega_t, alpha_t = anneal_dose(args.omega, args.alpha, step, args.anneal_steps)
        # The regret term has to anneal too. The contract of anneal_dose is "pure
        # GRPO once annealing is over" (omega->0 and alpha->1 both degenerate
        # exactly to the baseline), but regret used to be the constant
        # args.regret_lambda, applied after annealing, so past step 60 our arm was
        # really "GRPO + 2.0*regret" rather than GRPO -- a perturbation that never
        # goes away. On 4B, ours ended at 30.6 against 34.3 for GRPO, and over the
        # last 60 steps this term was the only difference between the two.
        anneal_frac = (min(1.0, max(0.0, step / args.anneal_steps))
                       if args.anneal_steps and args.anneal_steps > 0 else 0.0)
        regret_lambda_t = (args.regret_lambda * (1.0 - anneal_frac)
                           if args.anneal_regret else args.regret_lambda)
        adv_arm = {"anchor_cvmax": "anchor_mass"}.get(args.arm, args.arm)
        for gid, (tid, entries) in enumerate(groups.items()):
            rolls = [r for _, r, _ in entries]
            rewards = [float(r.outcome.reward or 0.0) for r in rolls]
            lengths = [len(r.turns) for r in rolls]
            if min(lengths, default=0) == 0:
                continue
            kw = {}
            if need_deltas:
                cr = cr_by_group.get(tid)
                if cr is None or len(cr.deltas) != len(rolls):
                    continue
                kw["a_star"] = [list(a) + [0.0] * (L - len(a))
                                for a, L in zip(cr.deltas, lengths)]
                kw["a_star_mask"] = [list(m) + [False] * (L - len(m))
                                     for m, L in zip(cr.mask, lengths)]
            per_turn = turn_advantages(adv_arm, rewards, lengths, gamma=args.gamma,
                                       omega=omega_t, step_std=not args.no_step_std,
                                       alpha=alpha_t, w_max=args.w_max,
                                       dose=args.dose, fallback=args.mass_fallback, fallback_omega=args.fallback_omega,
                                       nonneg_winners=args.nonneg_winners, **kw)
            try:   # Is the step term doing anything? (zero dose: alpha=1 mult., omega=0 add.)
                base_turn = turn_advantages(adv_arm, rewards, lengths, gamma=args.gamma,
                                            omega=0.0, step_std=not args.no_step_std,
                                            alpha=1.0, w_max=args.w_max,
                                            dose=0.0, fallback=args.mass_fallback, fallback_omega=args.fallback_omega, **kw)
                for a_row, b_row in zip(per_turn, base_turn):
                    for a, b in zip(a_row, b_row):
                        n_step_total += 1
                        n_step_nonzero += abs(a - b) > 1e-9
            except Exception:  # noqa: BLE001
                pass
            reg = reg_by_group.get(tid)
            if reg is not None and len(reg.deltas) == len(rolls) and regret_lambda_t:
                per_turn = [[a + regret_lambda_t * (r[t] if t < len(r) else 0.0)
                             for t, a in enumerate(row)]
                            for row, r in zip(per_turn, reg.deltas)]
                for row in reg.deltas:
                    for v in row:
                        if abs(v) > 1e-9:
                            n_step_nonzero += 0   # counted in the regret stats below
            if args.adv_clip is not None:
                c = float(args.adv_clip)
                per_turn = [[max(-c, min(c, v)) for v in row] for row in per_turn]
            for (task, roll, state), adv_row in zip(entries, per_turn):
                exs = encode_turns(roll, tokenizer, SYSTEM_PROMPT, "multihop",
                                   args.max_seq_len, gid)
                for ex in exs:
                    if ex.turn_index < len(adv_row):
                        examples.append(ex)
                        adv_values.append(adv_row[ex.turn_index])
                        if gid < n_quota_groups:
                            n_tok_quota += ex.n_actor_tokens
                        n_tok_all += ex.n_actor_tokens
                n_roll += 1
                total_score += float(roll.outcome.reward or 0.0)
                total_em += state.em
                total_turns += len(roll.turns)

        if not examples:
            print(f"[step {step}] no usable examples; skipping", flush=True)
            continue
        dilute = 1.0
        if args.no_dilute and n_tok_quota > 0 and n_tok_all > n_tok_quota:
            # The loss is a token mean: tokens from the appended groups (whose
            # advantages are mostly 0) thin out the quota groups' gradient in
            # proportion; measured |g| 0.6-0.8 against 4-9 for the baseline (which
            # is clipped to 1). Scaling back up makes the effective step size of
            # the quota groups match the baseline.
            dilute = n_tok_all / n_tok_quota
            adv_values = [a * dilute for a in adv_values]

        t0 = time.time()
        chunks = plan_chunks(examples, args.token_budget)
        loss, gnorm, n_oom = run_update(
            model, optimizer, examples, adv_values, chunks,
            collate=collate, pad=pad, token_logprobs=token_logprobs,
            grpo_loss=grpo_loss, clip_eps=args.clip_eps,
            max_grad_norm=args.max_grad_norm)
        t_train = time.time() - t0

        # EVERY step, before the next rollout: the loss assumes ratio 1.
        adapter = out_dir / "adapter"
        model.save_pretrained(str(adapter))
        sync.load(adapter_name, str(adapter))
        sync.verify_active(adapter_name)
        cfg.model = adapter_name
        eval_cfg.model = adapter_name

        lr_now = set_lr(step + 1)
        mean_adv = sum(abs(a) for a in adv_values) / max(len(adv_values), 1)
        n_reg_nz = sum(1 for reg in reg_by_group.values() for row, m in zip(reg.deltas, reg.mask)
                       for v, k in zip(row, m) if k and abs(v) > 1e-9)
        n_reg_meas = sum(1 for reg in reg_by_group.values() for m in reg.mask for k in m if k)
        n_la = sum(getattr(reg, "n_lookahead", 0) for reg in reg_by_group.values())
        n_la_gain = sum(getattr(reg, "n_lookahead_gain", 0) for reg in reg_by_group.values())
        rec = {"step": step, "arm": args.arm, "loss": loss, "grad_norm": gnorm,
               "n_rollouts": n_roll, "train_score": total_score / max(n_roll, 1),
               "train_em": total_em / max(n_roll, 1),
               "train_turns": total_turns / max(n_roll, 1),
               "n_examples": len(examples), "mean_abs_adv": mean_adv,
               "step_nonzero_frac": round(n_step_nonzero / max(n_step_total, 1), 3),
               "regret_measured": n_reg_meas, "regret_nonzero": n_reg_nz,
               "lookahead": n_la, "lookahead_gain": n_la_gain,
               "n_groups_drawn": n_drawn, "n_groups_flat_dropped": n_flat_dropped,
               "n_groups_flat_kept": n_flat_kept, "n_groups_used": len(groups),
               "omega": omega_t, "alpha": alpha_t, "lr": lr_now, "dilute_scale": round(dilute, 3),
               "n_chunks": len(chunks), "n_oom_dropped": n_oom,
               "n_regen": n_calls, "t_collect_s": round(t_collect, 1),
               "t_annot_s": round(t_annot, 1), "t_train_s": round(t_train, 1)}
        history.append(rec)
        with stats_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[step {step}] loss={loss:+.4f} |g|={gnorm:.3f} "
              f"score={rec['train_score']:.3f} turns={rec['train_turns']:.2f} "
              f"ex={len(examples)} |adv|={mean_adv:.3f} step≠0={rec['step_nonzero_frac']:.0%} "
              + (f"reg {n_reg_nz}/{n_reg_meas} " if need_regret else "")
              + (f"lookahead {n_la_gain}/{n_la} " if args.answer_lookahead else "")
              + (f"groups {len(groups)}/{n_drawn} drawn (dropped {n_flat_dropped}, "
                 f"flat-with-signal {n_flat_kept}) "
                 if args.dynamic_sampling else "")
              + f"({t_collect:.0f}s collect / {t_annot:.0f}s annot / {t_train:.0f}s train)"
              + (f"  OOM-dropped={n_oom}" if n_oom else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
