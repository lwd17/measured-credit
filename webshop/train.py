#!/usr/bin/env python
"""Train one arm on WebShop.

The second environment. It shares the update loop (`ssc.train.step`), the
advantage arms (`ssc.credit.alf_arms`) and the encoder with the ALFWorld
trainer; what differs is the environment, and only the environment.

Three things WebShop changes, all checked before this was written:

  the reward is GRADED       attribute match in [0, 1] rather than 0/1, so a
                             measured delta is continuous. Across six groups of
                             the untrained policy, 46% of turns carried a
                             non-zero delta over 13 distinct values in
                             [-0.04, +1.00] -- against 16% and two values on
                             ALFWorld. Negative deltas exist here: some turns
                             actively cost score.
  replay is OUT OF PROCESS   WebShop pins 2022 dependencies and its own Python,
                             so the environment lives behind a worker. There is
                             no batched env to share a reset across, so the 9.5x
                             from batching is gone; pooling still gives 2.7x.
  preconditions are enforced  clicking a product before searching is refused,
                             which is what makes the counterfactual
                             self-checking. Verified before building on it:
                             three sessions replayed three times each gave
                             identical traces, and 64% of ablations moved the
                             outcome.

Rollouts are collected with a thread pool over ONE worker rather than the
process pool ALFWorld needs: WebShop's state lives in the worker process, so
concurrent episodes would interleave on the same store. Generation is the
bottleneck anyway -- 24 rollouts take about a minute against 10-30 for
ALFWorld -- so the serial environment is not what limits the step.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=("grpo", "anchor_cf", "anchor_cvmax"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts_per_step", type=int, default=4)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max_turns", type=int, default=15)
    ap.add_argument("--n_train_tasks", type=int, default=400)
    ap.add_argument("--n_eval_tasks", type=int, default=48)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--omega", type=float, default=1.0)
    # `_fill_offpath` grades a failed rollout's turns by how much of what
    # remains was spent on pages a winner also visited. That is a suffix mean,
    # so it is monotone in turn position whenever on-path turns cluster early --
    # and measured on WebShop it correlates with position at rho = -0.576, which
    # is a position prior wearing a measurement's clothes. It was validated
    # against planner truth on ALFWorld (rho +0.172, position rho -0.020) and
    # never on WebShop, so on WebShop it is a switch, not a default.
    ap.add_argument("--no_offpath", action="store_true")
    # From the published ablation of a step-level baseline that estimates
    # credit from observed returns: the normalisation factor is task-dependent,
    # and fixing it at 1 scores higher on its harder tasks because dividing by
    # a group standard deviation set by a handful of values amplifies them. Our
    # WebShop groups are the extreme case of that.
    ap.add_argument("--no_step_std", action="store_true")
    # Prior step-level work trains WebShop on a rule-based reward -- 10 for
    # success, 0 otherwise -- while we used WebShop's native graded score.
    # Measured on the same rollouts, the choice decides which term of the
    # advantage survives: graded leaves a group variance of 0.037 against
    # binary's 0.133, so A_E is nearly zero and the step term carries almost the
    # whole advantage. Our own ALFWorld sweep found a step share near 50% helps
    # and 90% hurts by 3.3 points, so the graded reward had us running in the
    # regime our own experiment rejected.
    ap.add_argument("--binary_reward", action="store_true")
    # `drop_required` marks a turn whose ablation drives a positive return to
    # exactly zero. Under a GRADED reward that picks out `search` and `buy now`
    # and leaves the partial deltas. Under a BINARY reward the two cases are
    # exhaustive -- pivotal turns score 1 -> 0 and are dropped, redundant ones
    # score 1 -> 1 and are zero -- so the step signal is identically empty and
    # the arm silently runs as plain GRPO. ALFWorld, where the method works,
    # has a binary reward and no such filter at all.
    ap.add_argument("--keep_required", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--token_budget", type=int, default=16384)
    ap.add_argument("--max_seq_len", type=int, default=12288)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--collect_workers", type=int, default=16)
    ap.add_argument("--env_workers", type=int, default=8,
                    help="private WebShop environments; each costs one index build")
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--neutral_unmeasured", action="store_true",
                    help="treat the turns the operator structurally cannot measure\n"
                         "(search / navigation clicks / buy now) as unmeasured: their\n"
                         "multiplicative weight is fixed at 1, and mean(q)=1 is conserved\n"
                         "only over the measurable turns (opening a product, picking an\n"
                         "option). 09-12 probe: without this, credit on search/nav in\n"
                         "successful trajectories is pushed down to 80 percent of GRPO's,\n"
                         "and those are exactly the two kinds of turn a step-level baseline\n"
                         "that estimates credit from observed returns rewards twice.\n"
                         "Only affects ours.")
    ap.add_argument("--graded_measurement", action="store_true",
                    help="under a binary reward, **keep the counterfactual measurement in\n"
                         "the graded currency** (by default a binary reward binarises the\n"
                         "measurement as well). 09-12 probe: with a binary measurement the\n"
                         "running-max commit value is non-zero in a successful trajectory\n"
                         "only on the last turn that makes 'buy now would succeed' true;\n"
                         "the click that actually chose the item scores 0, and the\n"
                         "multiplicative term touches only about 10 percent of the turns.\n"
                         "This switch changes the measurement side only -- the episode\n"
                         "return stays binary. One of the three 09-12 repair arms.")
    ap.add_argument("--binary_measurement", action="store_true",
                    help="binarise the **counterfactual measurement** only; the episode\n"
                         "return stays graded.\n"
                         "Why they have to be separable: WebShop's graded reward is an\n"
                         "attribute-overlap score, so on that reward the per-turn measured\n"
                         "delta is a dense 'push the current basket's overlap higher'\n"
                         "signal, and the optimum of that objective is 'buy whatever is\n"
                         "closest on average' -- hedging. Measured over six arms, the\n"
                         "partial credit (score - success) correlates with the exact-match\n"
                         "rate at rho=-0.902, and the ordering is exactly how dense the\n"
                         "step term is:\n"
                         "no step term 0.191 < binary step term 0.208 < graded step term\n"
                         "0.216 < commit value 0.268 < run-to-optimum 0.305.\n"
                         "--binary_reward binarises the episode return as well, at the cost\n"
                         "of 6 of 8 groups having no within-group variance and therefore a\n"
                         "zero gradient for the whole step. This switch changes the\n"
                         "measurement side only.")
    ap.add_argument("--anneal_steps", type=int, default=None,
                    help="anneal the step term: omega decays linearly to 0 and alpha rises "
                         "linearly to 1 over this many steps, after which the arm is plain "
                         "GRPO. See the notes on ssc.credit.alf_arms.anneal_dose.")
    ap.add_argument("--regret_k", type=int, default=0,
                        help="measured regret against the best available alternative: the "
                             "first K items on a results page, the option completions on a "
                             "product page. Non-positive, not centred within the group, added "
                             "to the advantage with --regret_lambda (0 = off).")
    ap.add_argument("--item_credit", choices=("regret", "centred"), default="regret",
                        help="credit for the turn that opens a product from a results page: "
                             "regret = penalise only falling short of the best item on the page "
                             "(non-positive); centred = Q(a) minus the mean over the first k "
                             "items on the page (a zero-mean virtual anchor group, signed both "
                             "ways).")
    ap.add_argument("--regret_group_center", action="store_true",
                        help="zero-mean the product-opening regret inside the same-state anchor "
                             "group formed by the 8 rollouts of one task (a relative comparison, "
                             "as in a step-level baseline that estimates credit from observed "
                             "returns); the buy-now regret stays absolute. This is what fixes "
                             "'memorise the product id': absolute regret is a cross-task "
                             "quantity.")
    ap.add_argument("--anneal_regret", action="store_true",
                    help="decay the regret term to zero linearly along with anneal_steps.\n"
                         "Off by default: in every existing arm regret was a constant added\n"
                         "after the annealing, so once annealing finished the arm was\n"
                         "'GRPO + lambda*regret' rather than plain GRPO. Turning it on\n"
                         "matches the written contract of anneal_dose; but ws_L_mxann_s0\n"
                         "(regret off throughout) hit a gradient norm of 20 at step 90 on\n"
                         "WebShop and collapsed, so the behaviour of existing arms is not\n"
                         "changed by default.")
    ap.add_argument("--regret_lambda", type=float, default=1.0)
    ap.add_argument("--unmask_commit", action="store_true",
                        help="for the within-anchor-group form only: the buy-now turn is no "
                             "longer masked and joins the anchor group of its product page with "
                             "its measured value (identically 0 under run-to-optimum), so that "
                             "'click one more option' and 'buy now' are actually compared.")
    ap.add_argument("--dynamic_sampling", action="store_true",
                        help="drop groups whose 8 rollouts all end the same way (A_E is "
                             "identically zero, so the whole group carries no gradient) and keep "
                             "drawing from the task pool until prompts_per_step groups are filled "
                             "or the budget runs out.")
    ap.add_argument("--dyn_max_factor", type=int, default=4,
                        help="dynamic sampling draws at most prompts_per_step * this value "
                             "tasks per step.")
    ap.add_argument("--adv_clip", type=float, default=None,
                        help="clip the per-turn advantage to [-adv_clip, adv_clip]. In the "
                             "state-level form the three terms A_E + 0.5*S + 2*G can stack to "
                             "|A| about 3, with a measured median |g| of 5.2 against 1.8 for a "
                             "step-level baseline that estimates credit from observed returns; "
                             "clipping the gradient norm only rescales the whole gradient and "
                             "does not hold down extreme single-turn values.")
    ap.add_argument("--lr_final", type=float, default=None,
                        help="decay the learning rate linearly to this value (no decay by "
                             "default).")
    ap.add_argument("--search_lookahead", type=int, default=0,
                        help="one-step lookahead for V* at a search turn: the best value "
                             "reachable by a single buy among the first K items of the results "
                             "page is recorded as the credit of that search turn (0 = off). It "
                             "covers the blind spot V* has on the anchor group of a search page.")
    ap.add_argument("--dose", type=float, default=1.0,
                    help="dose for anchor_mass_signed: at 0 it degenerates exactly to GRPO; "
                         "above 1 the least necessary turns get a negative weight (a negative "
                         "sign can appear even inside a successful trajectory)")
    ap.add_argument("--fallback_omega", type=float, default=None,
                    help="scale of the step term on the fallback branch (groups with A_E=0); "
                         "defaults to --omega. Use it to hold the magnitude down when the three "
                         "terms stack: on QA, omega=1 doubles |adv| and the run collapses.")
    ap.add_argument("--mass_fallback", action="store_true",
                    help="groups that all end the same way (A_E=0, so there is no mass to "
                         "conserve) fall back to the additive step term. The switch for the "
                         "unified form: on ALFWorld about 74%% of groups are of this kind.")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="dose knob for anchor_mass: at 1.0 every weight is 1 and it "
                         "degenerates exactly to GRPO; at 0 credit is distributed entirely by "
                         "the measured contribution")
    ap.add_argument("--mask_structural", action="store_true",
                    help="mark click[buy now] and the leading search as unmeasurable. "
                         "Deleting either necessarily drives the return to zero, so their "
                         "delta equals the return of the whole trajectory -- a quantity A_E "
                         "already carries. commit_deltas masks the commit turn and "
                         "multihop_deltas masks the answer turn; the deletion operator was "
                         "missing that mask here.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_path",
                    default=os.environ.get("SSC_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--served_name", default="Qwen3-8B")
    ap.add_argument("--base_url", default="http://localhost:8103/v1")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--resume_adapter", default=None,
                    help="continue training from a saved adapter. **The optimizer state is "
                         "NOT restored** -- AdamW's first and second moments start from zero "
                         "again, so the first step or two after a resume take an effectively "
                         "larger step. This is a real discontinuity, written down here rather "
                         "than hidden: a resumed arm and an arm run in one go are not exactly "
                         "the same run. It is still far cheaper than discarding the steps "
                         "already paid for.")
    ap.add_argument("--start_step", type=int, default=0,
                    help="the step number to continue from. The evaluation cadence follows "
                         "it too, so a resumed arm lands on the same step numbers as an arm "
                         "run in one go and the comparison stays aligned.")
    args = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ssc.credit.alf_arms import anneal_dose, turn_advantages
    from ssc.credit.webshop_cf import (webshop_regret, page_key, webshop_commit_deltas, center_item_regret,
                                       webshop_deltas)
    from ssc.env.webshop_env import (SYSTEM_PROMPT, WebShopWorker,
                                     load_webshop)
    from ssc.env.webshop_policy import WebShopPolicy, WebShopPolicyConfig
    from ssc.train.batch import collate, encode_turns, token_logprobs
    from ssc.train.step import plan_chunks, run_update
    from ssc.train.sync import LoRASync
    from ssc.credit.grpo_patch import grpo_loss

    torch.manual_seed(args.seed)
    out_dir = ROOT / (args.out_dir or f"runs/ws_{args.arm}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Dump the launch arguments. `resume_ws_arm.sh` inherits the switches from
    # this file: re-typing them by hand and dropping one `--mask_structural` or
    # `--binary_reward` yields a different arm on resume, with nothing in the
    # logs looking wrong. A resume writes its own file name and does not
    # overwrite the original.
    _cfg = (f"config_from_step{args.start_step}.json" if args.start_step
            else "config.json")
    (out_dir / _cfg).write_text(json.dumps(vars(args), indent=2))
    stats_path = out_dir / "steps.jsonl"

    train_tasks = load_webshop(args.n_train_tasks, split="train")
    eval_tasks = load_webshop(args.n_eval_tasks, split="eval")
    print(f"arm={args.arm} train={len(train_tasks)} eval={len(eval_tasks)}",
          flush=True)

    cfg = WebShopPolicyConfig(model=args.served_name, max_turns=args.max_turns,
                              max_tokens_per_turn=args.max_tokens)
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
    # `gradient_checkpointing_enable()` is guarded by `self.training` inside
    # transformers, and `get_peft_model` hands back a model in EVAL mode, so
    # without this line every one of the 36 layers keeps its full activation
    # set while the flag still reads True. On ALFWorld that was 46.3 GB against
    # 6.7 GB; here it was the whole difference between a 23.5 GB micro-batch in
    # isolation and the 49.5 GB that dropped all 324 examples of every step.
    # lora_dropout is 0, so the two modes are mathematically identical and only
    # the memory differs -- which is exactly why it stays invisible.
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def set_lr(step: int) -> float:
        """Decay linearly to lr_final (no decay by default).

        Late in training only a few groups per step carry a gradient, |g| often
        reaches 8-15, and clipping rescales the whole gradient by 1/|g|, so one
        noisy step can undo dozens of steps of progress.
        """
        if args.lr_final is None or args.steps <= 0:
            return args.lr
        frac = min(1.0, max(0.0, step / args.steps))
        lr = args.lr + (args.lr_final - args.lr) * frac
        for g in optimizer.param_groups:
            g["lr"] = lr
        return lr
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id

    # The LoRA name on the sampler has to be unique per run, not per
    # (arm, seed): in the early hours of 09-12 three 4B ours variants
    # (B4_ours / V4_AR / V4_NR) shared one sampling service under the single
    # name ws_anchor_cvmax_s1, so every step's sync.load overwrote the others,
    # rollouts were drawn from another arm's weights, and five arms were wasted
    # overnight. out_dir is named ws_{arm}_s{seed} by default, so the name of a
    # default run does not change.
    adapter_name = out_dir.name
    sync = LoRASync(args.base_url)
    # The worker holds the store; one process serves every rollout and every
    # ablation, so the 1000-product index is built once for the whole run.
    # ONE WORKER PER THREAD. A worker holds a single live environment, and
    # `_call` writes a request then reads a line with no lock, so two rollouts
    # sharing one interleave their steps and every thread blocks -- which is why
    # collection was a sequential loop. Private workers make it parallel with no
    # change to any rollout: the environment each one drives is its own.
    workers = [WebShopWorker() for _ in range(args.env_workers)]
    worker = workers[0]
    policy = WebShopPolicy(cfg, args.base_url)

    def collect(tasks, group, seed_base):
        """Rollouts for one step, one private environment per thread.

        Generation is what the wall clock is spent on, so overlapping it is the
        whole win: 209 s sequential against about 30 s at eight workers, on a
        step that was 1362 s. Identical rollouts -- the seed and the task decide
        the trajectory, not which worker ran it.
        """
        flat = [(t, k) for t in tasks for k in range(group)]
        jobs = [(n % len(workers), t, k) for n, (t, k) in enumerate(flat)]
        # Each rollout owns one environment exclusively: take it from the
        # queue, hand it back when done. Slots used to be assigned by n % W, but
        # the thread pool picks up the next job as soon as a thread frees up, so
        # job n+W can be handed the same slot while job n is still running --
        # two rollouts then interleave reset/step on one WebShop session and
        # contaminate each other's trajectories (simulated: of the 32 rollouts
        # per step, about a third entered with the environment already taken).
        # See "evaluation protocol differences" in reports/STATUS.md.
        import queue as _queue
        pool: "_queue.Queue" = _queue.Queue()
        for w in workers:
            pool.put(w)

        def one(job):
            _slot, t, k = job
            w = pool.get()
            try:
                return (t, policy.rollout(t, rollout_index=k, seed=seed_base + k,
                                          worker=w))
            except Exception as e:  # noqa: BLE001
                print(f"  ! {t.task_id} k={k}: {type(e).__name__}: {e}", flush=True)
                return None
            finally:
                pool.put(w)

        with ThreadPoolExecutor(max_workers=len(workers)) as ex:
            return [r for r in ex.map(one, jobs) if r is not None]

    def evaluate(step: int) -> tuple[float, float]:
        got = collect(eval_tasks, 1, seed_base=90000 + step)
        if not got:
            return 0.0, 0.0
        return (sum(r.outcome.reward for _, r in got) / len(got),
                sum(1 for _, r in got if r.outcome.success) / len(got))

    # On a resume the adapter has never been pushed to the sampler from this
    # new process and cfg.model is still the base model name -- the very first
    # rollout would be off-policy while the loss assumes a ratio of 1. The guard
    # below was written for exactly this case and correctly stopped four arms on
    # the first resume attempt. This does that one sync before entering the
    # loop: the weights are the ones training continues from, so the adapter
    # directory pushed here is the same one.
    if args.start_step > 0:
        sync.load(adapter_name, str(out_dir / "adapter"))
        sync.verify_active(adapter_name)
        cfg.model = adapter_name
        print(f"pushed resumed adapter to sampler as {adapter_name!r}", flush=True)

    history = []
    for step in range(args.start_step, args.steps + 1):
        # The sampler must already be serving the adapter before any rollout of
        # step > 0 is collected. This reads as a tautology and was false for
        # twenty steps: the logs stayed normal, the adapter loaded fine, and
        # the only symptom was a training score that looked healthy because it
        # was measuring the base model.
        if step > 0 and cfg.model != adapter_name:
            raise RuntimeError(
                f"sampler still on {cfg.model!r} at step {step}: rollouts would "
                f"be off-policy while the loss assumes a ratio of 1")
        if step % args.eval_every == 0:
            if step > 0:
                # The sampler is already current: the per-step sync at the end
                # of the previous iteration did it. This branch only keeps a
                # revisitable copy, because a run that peaks and then collapses
                # leaves nothing to evaluate if one adapter is overwritten in
                # place.
                model.save_pretrained(str(out_dir / f"adapter_step{step}"))
            score, succ = evaluate(step)
            print(f"[eval step {step}] score={score:.3f} success={succ:.1%}",
                  flush=True)
            history.append({"step": step, "eval_score": score,
                            "eval_success": succ})
            with stats_path.open("a") as f:
                f.write(json.dumps(history[-1]) + "\n")
            if step == args.steps:
                break

        t0 = time.time()
        lo = (step * args.prompts_per_step) % max(len(train_tasks), 1)
        # Dynamic sampling (DAPO style): when all 8 rollouts of a group end the
        # same way, A_E is identically zero and the group contributes nothing to
        # the gradient -- late in training such groups are the large majority
        # (measured on the L_mxann@60 policy, 8 of 8 groups ended identically),
        # so only a handful of groups actually push each step and they dominate
        # the direction, which is the mechanism behind the late drift. Here the
        # zero-variance groups are dropped and further tasks are drawn from the
        # pool until prompts_per_step groups are filled or the budget runs out.
        a_star_by_group: dict[str, list[list[float]]] = {}
        mask_by_group: dict[str, list[list[bool]]] = {}
        regret_by_group: dict[str, list[list[float]]] = {}
        regret_kinds_by_group: dict[str, list[list]] = {}
        n_replays = 0
        DELTA_ARMS = ("anchor_cf", "anchor_cvmax")

        # With --search_lookahead on, a search turn IS measurable (the best
        # buyable value among the first k items of the results page), so it must
        # no longer be masked as "structurally silent" -- otherwise the credit
        # the lookahead computes is thrown away by a weight fixed at 1.
        _UNMEASURABLE = (() if args.search_lookahead > 0 else ("search[",)) + (
                         "click[buy now]", "click[back to search]", "click[< prev]",
                         "click[next >]", "click[description]", "click[features]",
                         "click[reviews]", "click[attributes]")
        def _neutral_mask(mask, ents):
            """search / navigation / buy now: the turns the operator is
            structurally silent on, marked unmeasured (weight fixed at 1)."""
            out = []
            for row, (_, roll) in zip(mask, ents):
                acts = [str((x.tool_args or {}).get("action", "")).strip().lower() for x in roll.turns]
                out.append([bool(k) and not any(a.startswith(u) for u in _UNMEASURABLE)
                            for k, a in zip(row, acts)] + list(row[len(acts):]))
            return out
        def annotate_groups(pairs):
            """Counterfactual measurement for several groups (one environment
            per group; with no more groups than workers, none is shared)."""
            def annotate(item):
                slot, tid, entries = item
                sess = [t.session for t, _ in entries]
                acts = [[str((x.tool_args or {}).get("action", ""))
                         for x in r.turns] for _, r in entries]
                obs = [[str((x.tool_args or {}).get("state_before", ""))
                        for x in r.turns] for _, r in entries]
                try:
                    w = workers[slot]
                    reg = (webshop_regret(w, sess, acts, obs, k=args.regret_k,
                                          mode=args.item_credit,
                                          binary=(args.binary_reward and not args.graded_measurement) or args.binary_measurement)
                           if args.regret_k > 0 else None)
                    return tid, reg, (webshop_commit_deltas(
                                     w, sess, acts, obs,
                                     running_max=args.arm in ("anchor_cvmax",),
                                     lookahead_k=args.search_lookahead,
                                     mask_commit=not args.unmask_commit,
                                     binary=(args.binary_reward and not args.graded_measurement) or args.binary_measurement)
                                 if args.arm in ("anchor_cvmax",)
                                 else webshop_deltas(w, sess, acts, obs,
                                                     fill_offpath=not args.no_offpath,
                                                     binary=(args.binary_reward and not args.graded_measurement) or args.binary_measurement,
                                                     drop_required=not args.keep_required,
                                                     mask_structural=args.mask_structural))
                except Exception as e:  # noqa: BLE001
                    print(f"  ! deltas failed on {tid}: {type(e).__name__}: {e}",
                          flush=True)
                    return tid, None, None
            items = [(k % len(workers), tid, ent) for k, (tid, ent) in enumerate(pairs)]
            if not items:
                return []
            with ThreadPoolExecutor(max_workers=min(len(workers), len(items))) as ex:
                return list(ex.map(annotate, items))

        groups = collections.defaultdict(list)
        n_drawn = n_dropped_flat = n_kept_cf = 0
        cursor = lo
        max_draws = (args.prompts_per_step * args.dyn_max_factor
                     if args.dynamic_sampling else args.prompts_per_step)
        while len(groups) < args.prompts_per_step and n_drawn < max_draws:
            want = args.prompts_per_step - len(groups)
            batch = [train_tasks[(cursor + i) % len(train_tasks)] for i in range(want)]
            cursor += want; n_drawn += want
            got = collect(batch, args.group,
                          seed_base=args.seed * 100000 + step * 1000 + n_drawn)
            fresh = collections.defaultdict(list)
            for task, roll in got:
                fresh[task.task_id].append((task, roll))
            flat = {}
            for tid, entries in fresh.items():
                rs = [float(r.outcome.reward or 0.0) for _, r in entries]
                if args.dynamic_sampling and len(entries) >= 2 and max(rs) - min(rs) < 1e-9:
                    flat[tid] = entries
                    continue
                groups[tid] = entries
            # Zero-variance groups must not all be dropped: in a group where
            # all 8 rollouts opened the same wrong product the outcomes are
            # identical (A_E is identically zero), yet the regret term gives
            # every rollout negative credit -- the strongest signal there is for
            # correcting item choice, and it only appears in groups like this.
            # So measure first, keep whatever carries a counterfactual signal,
            # and drop only groups with no variance AND an all-zero measurement.
            if flat and args.arm in DELTA_ARMS:
                for tid, reg, cr in annotate_groups(list(flat.items())):
                    has_signal = False
                    if cr is not None:
                        a_star_by_group[tid] = cr.deltas
                        mask_by_group[tid] = _neutral_mask(cr.mask, flat[tid]) if args.neutral_unmeasured else cr.mask
                        n_replays += cr.n_replays
                        has_signal = any(abs(v) > 1e-9 for row, m in zip(cr.deltas, cr.mask)
                                         for v, k in zip(row, m) if k)
                        if reg is not None:
                            regret_by_group[tid] = reg.deltas
                            regret_kinds_by_group[tid] = reg.kinds
                            n_replays += reg.n_replays
                            has_signal = has_signal or any(v < -1e-9 for row in reg.deltas for v in row)
                    if has_signal:
                        groups[tid] = flat[tid]; n_kept_cf += 1
                    else:
                        n_dropped_flat += 1
            else:
                n_dropped_flat += len(flat)
            if not args.dynamic_sampling:
                break
            if n_drawn >= max_draws and not groups:
                # Budget exhausted and not one group kept: the whole batch
                # really has no variance, and rather than burn an empty step,
                # take the last batch (the gradient is near zero, but it saves
                # another round of sampling and leaves a row in steps.jsonl).
                groups.update(fresh)
        t_collect = time.time() - t0

        t0 = time.time()
        if args.arm in DELTA_ARMS:
            todo = [(tid, ent) for tid, ent in groups.items() if tid not in a_star_by_group]
            for tid, reg, cr in annotate_groups(todo):
                if cr is None:
                    continue
                a_star_by_group[tid] = cr.deltas
                mask_by_group[tid] = _neutral_mask(cr.mask, groups[tid]) if args.neutral_unmeasured else cr.mask
                n_replays += cr.n_replays
                if reg is not None:
                    regret_by_group[tid] = reg.deltas
                    regret_kinds_by_group[tid] = reg.kinds
                    n_replays += reg.n_replays
        if n_replays:
            print(f"    {args.arm}: {n_replays} replays over {len(groups)} groups",
                  flush=True)
        t_annot = time.time() - t0

        examples, adv_values = [], []
        n_step_nonzero, n_step_total = 0, 0  # is the step term doing anything (guards against silently degenerating to GRPO)
        n_roll = 0
        total_score = 0.0
        for gid, (tid, entries) in enumerate(groups.items()):
            rolls = [r for _, r in entries]
            rewards = [(1.0 if (r.outcome.reward or 0.0) >= 1.0 else 0.0)
                       if args.binary_reward else float(r.outcome.reward or 0.0)
                       for r in rolls]
            lengths = [r.num_turns for r in rolls]
            kw = {}
            if args.arm in DELTA_ARMS:
                a_star = a_star_by_group.get(tid, [])
                if len(a_star) != len(rolls):
                    continue
                kw["a_star"] = [list(a) + [0.0] * (L - len(a))
                                for a, L in zip(a_star, lengths)]
                m = mask_by_group.get(tid)
                if m and len(m) == len(rolls):
                    kw["a_star_mask"] = [list(x) + [False] * (L - len(x))
                                         for x, L in zip(m, lengths)]
            # The measurement operator and the combination form are two
            # independent choices; this is where they are decoupled:
            #   anchor_cv     commit value V    + additive (the same combination
            #                                    path as anchor_cf)
            #   anchor_cvmax  run-to-optimum V* + sign-preserving multiplicative
            #                                    (V* is identically zero on
            #                                    navigation, and the additive
            #                                    form would centre those zeros
            #                                    into a penalty)
            #   anchor_cvmax_grouped  run-to-optimum V* + centring within the
            #                 anchor group, as in a step-level baseline that
            #                 estimates credit from observed returns (its eq. 7
            #                 with F_norm=1). From one state the V* increment of
            #                 "click one more option" is positive and that of
            #                 "buy now" is zero, so centring within the group
            #                 gives the former a positive sign and the latter a
            #                 negative one -- exactly the pathology seen in the
            #                 traces (buying with one option left unset). Zeros
            #                 are only compared inside the group and are no
            #                 longer centred into a penalty.
            adv_arm = {"anchor_cvmax": "anchor_mass"}.get(args.arm, args.arm)
            # The regret term has to decay with the annealing as well. The
            # contract of anneal_dose is "after annealing this is plain GRPO"
            # (omega->0 and alpha->1 both degenerate exactly to the baseline),
            # but regret used to be added after the annealing with the constant
            # args.regret_lambda, so beyond anneal_steps the arm was really
            # "GRPO + lambda*regret" and not GRPO -- a perturbation that never
            # goes away. The consequence has already been measured in
            # mh_train.py: on 4B QA, ours ends at 30.6 against GRPO's 34.3, and
            # over the last 60 steps this term is their only difference.
            _af = (min(1.0, max(0.0, step / args.anneal_steps))
                   if args.anneal_steps and args.anneal_steps > 0 else 0.0)
            regret_lambda_t = (args.regret_lambda * (1.0 - _af)
                               if args.anneal_regret else args.regret_lambda)
            omega_t, alpha_t = anneal_dose(args.omega, args.alpha, step,
                                           args.anneal_steps)
            per_turn = turn_advantages(adv_arm, rewards, lengths,
                                       gamma=args.gamma, omega=omega_t,
                                       step_std=not args.no_step_std,
                                       alpha=alpha_t, dose=args.dose,
                                       fallback=args.mass_fallback, fallback_omega=args.fallback_omega,
                                       neutral_unmeasured=args.neutral_unmeasured, **kw)
            # The same batch with the dose zeroed out; the number of turns
            # where the two differ is the number of turns the step term changes.
            # Zero dose means alpha=1 for a multiplicative arm and omega=0 for
            # an additive one; zeroing both is correct for either.
            try:
                base_turn = turn_advantages(adv_arm, rewards, lengths,
                                            gamma=args.gamma, omega=0.0,
                                            step_std=not args.no_step_std,
                                            alpha=1.0, dose=0.0,
                                            fallback=args.mass_fallback, fallback_omega=args.fallback_omega, **kw)
                for a_row, b_row in zip(per_turn, base_turn):
                    for a, b in zip(a_row, b_row):
                        n_step_total += 1
                        n_step_nonzero += abs(a - b) > 1e-9
            except Exception:  # noqa: BLE001  diagnostic only; must never affect training
                pass
            reg_rows = regret_by_group.get(tid)
            if reg_rows and args.regret_group_center and tid in regret_kinds_by_group:
                keys_for_reg = kw.get("anchor_keys") or [
                    [page_key(str((x.tool_args or {}).get("state_before", "")))
                     for x in r.turns] for r in rolls]
                reg_rows = center_item_regret(reg_rows, regret_kinds_by_group[tid], keys_for_reg)
            if reg_rows and len(reg_rows) == len(per_turn) and regret_lambda_t:
                per_turn = [[a + regret_lambda_t * (r[t] if t < len(r) else 0.0)
                             for t, a in enumerate(row)]
                            for row, r in zip(per_turn, reg_rows)]
            if args.adv_clip is not None:
                c = float(args.adv_clip)
                per_turn = [[max(-c, min(c, v)) for v in row] for row in per_turn]
            for (task, roll), adv_row in zip(entries, per_turn):
                exs = encode_turns(roll, tokenizer, SYSTEM_PROMPT, "webshop",
                                   args.max_seq_len, gid)
                for ex in exs:
                    if ex.turn_index < len(adv_row):
                        examples.append(ex)
                        adv_values.append(adv_row[ex.turn_index])
                n_roll += 1
                total_score += roll.outcome.reward or 0.0

        if not examples:
            print(f"[step {step}] no usable examples; skipping", flush=True)
            continue

        t0 = time.time()
        chunks = plan_chunks(examples, args.token_budget)
        loss, gnorm, n_oom = run_update(
            model, optimizer, examples, adv_values, chunks,
            collate=collate, pad=pad, token_logprobs=token_logprobs,
            grpo_loss=grpo_loss, clip_eps=args.clip_eps,
            max_grad_norm=args.max_grad_norm)
        t_train = time.time() - t0

        # Sync EVERY step, not every eval. `run_update` sets `old_lp =
        # new_lp.detach()` because one optimizer step per batch of rollouts
        # makes the behaviour policy the current policy and the ratio exactly
        # 1. Syncing only inside the eval branch broke that premise: for 20
        # steps vLLM kept serving the base weights while the adapter drifted,
        # so every update was off-policy at a ratio the loss assumed was 1.
        # anchor_cf has the largest gradients of the three arms and diverged
        # first -- training score fell from 0.769 to exactly 0.000 the moment
        # step 20 finally pointed the sampler at the adapter, and episodes
        # collapsed to 2 turns. alf_train.py syncs per step and carries a
        # comment saying precisely this; the WebShop trainer was written later
        # and did not carry it over.
        adapter = out_dir / "adapter"
        model.save_pretrained(str(adapter))
        sync.load(adapter_name, str(adapter))
        sync.verify_active(adapter_name)
        cfg.model = adapter_name

        lr_now = set_lr(step)
        mean_adv = sum(abs(a) for a in adv_values) / max(len(adv_values), 1)
        rec = {"step": step, "arm": args.arm, "loss": loss, "grad_norm": gnorm,
               "n_rollouts": n_roll,
               "train_score": total_score / max(n_roll, 1),
               "train_success": total_score / max(n_roll, 1),
               "n_examples": len(examples), "mean_abs_adv": mean_adv,
               "step_nonzero_frac": round(n_step_nonzero / max(n_step_total, 1), 3),
               "n_groups_drawn": n_drawn, "n_groups_flat": n_dropped_flat,
               "n_groups_used": len(groups), "n_groups_flat_kept_cf": n_kept_cf, "lr": lr_now,
               "n_chunks": len(chunks), "n_oom_dropped": n_oom,
               "t_collect_s": round(t_collect, 1),
               "t_annot_s": round(t_annot, 1), "t_train_s": round(t_train, 1)}
        history.append(rec)
        with stats_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[step {step}] loss={loss:+.4f} |g|={gnorm:.3f} "
              f"score={rec['train_score']:.3f} ex={len(examples)} "
              f"|adv|={mean_adv:.3f} step≠0={rec['step_nonzero_frac']:.0%} "
              + (f"groups {len(groups)}/{n_drawn} drawn (dropped {n_dropped_flat}, flat-but-cf {n_kept_cf}) " if args.dynamic_sampling else "")
              + (f"({t_collect:.0f}s collect / {t_annot:.0f}s annot / "
              f"{t_train:.0f}s train)")
              + (f"  OOM-dropped={n_oom}" if n_oom else ""), flush=True)

    for w in workers:
        w.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
