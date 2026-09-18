#!/usr/bin/env python
"""Evaluate saved checkpoints against the untrained policy, paired per task.

The in-training eval cannot answer anything here. Twelve evaluations of ONE
untrained policy on the same 24 tasks with the same seeds spanned 62.5% to
83.3% -- sd 5.2, a 20-point range -- so a 24-task number carries no information
about a difference smaller than that, and every difference in question is
smaller than that.

This runs the real measurement: greedy decoding, the whole 134-game unseen
split, three runs per task scored by their mean, every checkpoint on the same
tasks with the same seeds, compared PER TASK. Resolution is about 4 points.

`base` -- the untrained model -- is evaluated as its own checkpoint and is the
reference every other number is quoted against. Without it a table of scores
cannot separate a gain from a smaller loss: the previous run collapsed, and the
arm that collapsed least would otherwise have read as the winner, with a
"ceiling" computed against it that was really a ranking of damage.

Checkpoints matter because a run can peak and then fall apart. At lr=1e-4 grpo
peaked at step 10 and ended at 20.8%, below the untrained 63.4%, and its step-10
weights had been overwritten nine times by then. Comparing arms at their final
step alone would have compared two wreckages.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mcnemar(a_wins: int, b_wins: int) -> float:
    """Two-sided exact McNemar p-value on discordant pairs."""
    n = a_wins + b_wins
    if n == 0:
        return 1.0
    k = min(a_wins, b_wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_bootstrap(pairs, n_boot=5000, seed=0):
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        s = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        diffs.append(sum(x - y for x, y in s) / len(s))
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="grpo")
    ap.add_argument("--steps", default="20,40,60,80,final",
                    help="checkpoint steps to evaluate; 'final' means the "
                         "last-written adapter")
    ap.add_argument("--n_tasks", type=int, default=134)
    ap.add_argument("--split", default="valid_unseen",
                    choices=("valid_unseen", "valid_seen", "valid_all"),
                    help="'valid_all' pools both held-out splits. Neither was "
                         "trained on here -- training uses the train split -- "
                         "so pooling doubles the sample and halves the interval "
                         "instead of trading one bias for another.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--hard_types", action="store_true",
                    help="score only the four task types the arms trained on. "
                         "Must match start_arm.sh or the arms are graded on a "
                         "mix they never saw.")
    ap.add_argument("--max_turns", type=int, default=30,
                    help="30, not 25: 6 of 61 known wins on the hard types need "
                         "26-30 turns, and 36%% of pick_heat wins land past 25.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--action_hint", default="full",
                    choices=("full", "objects", "none"),
                    help="how much the prompt gives away. 'full' lists the "
                         "admissible commands and leaves an untrained 8B at 65%% "
                         "with no headroom; 'none' gives nothing and leaves it "
                         "at 0/24, which is unlearnable rather than hard; "
                         "'objects' names what is present and makes the policy "
                         "compose the command.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm_seed", type=int, default=0)
    ap.add_argument("--skip_base", action="store_true",
                    help="omit the untrained reference row; quote it from an "
                         "earlier pass instead of paying for it again")
    ap.add_argument("--base_url",
                    default="http://localhost:8100/v1,http://localhost:8101/v1")
    ap.add_argument("--served_name", default="Qwen3-8B")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="reports/alf_checkpoint_eval.json")
    ap.add_argument("--no_cache", action="store_true",
                    help="do not read or write the per-row cache beside --out")
    args = ap.parse_args()

    from ssc.detector.rollout import Rollout
    from ssc.env.alf_pool import collect
    from ssc.env.alfworld_env import ALFWorldPolicyConfig, load_alfworld
    from ssc.train.sync import LoRASync

    from ssc.env.alfworld_env import HARD_TYPES
    keep = HARD_TYPES if args.hard_types else None
    if args.split == "valid_all":
        # Both held-out splits. ALFWorld calls one "seen" because its own
        # demonstrations covered those layouts; neither appears in the split
        # trained on here, so both are unseen for this experiment.
        tasks = (load_alfworld(args.n_tasks, split="valid_unseen", spread=True,
                               task_types=keep)
                 + load_alfworld(args.n_tasks, split="valid_seen", spread=True,
                                 task_types=keep))
    else:
        tasks = load_alfworld(args.n_tasks, split=args.split, spread=True,
                              task_types=keep)
    sync = LoRASync(args.base_url)

    # (label, adapter path or None for the base model)
    # Re-running the base costs a full pass, and it is the one row that does not
    # change between invocations -- so a follow-up that only wants a different
    # checkpoint can skip it and quote the number the first pass produced.
    targets = [] if args.skip_base else [("base", None)]
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        # An arm may be given either as a short name (assembled into
        # alf_<arm>_s<seed>) or as a full directory name, because the 4B family
        # lives in alf4b_<arm>_s1, without the "alf_" prefix.
        run_dir = ROOT / args.runs / f"{arm}_s{args.arm_seed}"
        if not run_dir.is_dir():
            run_dir = ROOT / args.runs / f"alf_{arm}_s{args.arm_seed}"
        for st in [s.strip() for s in args.steps.split(",") if s.strip()]:
            path = run_dir / ("adapter" if st == "final" else f"adapter_step{st}")
            if path.exists():
                targets.append((f"{arm}@{st}", path))
            else:
                print(f"  {arm}@{st}: missing {path}", flush=True)

    print(f"{len(tasks)} unseen tasks x {args.repeats} runs, temperature="
          f"{args.temperature}, {len(targets)} checkpoints", flush=True)

    # A row is ~70 minutes and the table is only written when the LAST row
    # finishes, so anything that kills the process throws away every row
    # already paid for: the 92-task hard-type pass died in its fifth row after
    # 5.8 hours and left four completed rows in the log and nothing on disk.
    # Each row is therefore flushed to a sidecar as it completes, and a rerun
    # with the same measurement settings reuses it. The key is every setting
    # that changes what a row MEANS -- reusing a row measured on other tasks,
    # other seeds or a different turn budget would silently mix two experiments
    # into one column.
    cache_path = ROOT / (str(args.out) + ".partial")
    cache_key = {"split": args.split, "n_tasks": len(tasks),
                 "hard_types": bool(args.hard_types), "repeats": args.repeats,
                 "temperature": args.temperature, "max_turns": args.max_turns,
                 "action_hint": args.action_hint, "seed": args.seed,
                 "arm_seed": args.arm_seed, "runs": args.runs}
    results: dict[str, dict[str, float]] = {}
    if not args.no_cache and cache_path.exists():
        try:
            blob = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            blob = {}
        if blob.get("key") == cache_key:
            results = {k: {t: float(v) for t, v in row.items()}
                       for k, row in blob.get("rows", {}).items()}
            if results:
                print(f"  reusing {len(results)} row(s) from {cache_path.name}: "
                      + ", ".join(sorted(results)), flush=True)
        elif blob:
            print(f"  ignoring {cache_path.name}: measured under different "
                  "settings", flush=True)

    def flush_cache() -> None:
        if args.no_cache:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps({"key": cache_key, "rows": results}))
        tmp.replace(cache_path)  # atomic: a crash mid-write cannot corrupt it

    for label, path in targets:
        if label in results:
            rate = sum(results[label].values()) / max(len(results[label]), 1)
            print(f"  {label}: {rate:.1%} (cached)", flush=True)
            continue
        if path is None:
            name = args.served_name
        else:
            name = f"ck_{label.replace('@', '_')}"
            sync.load(name, str(path))
            sync.verify_active(name)
        cfg = ALFWorldPolicyConfig(model=name, max_turns=args.max_turns,
                                   max_tokens_per_turn=3072,
                                   temperature=args.temperature, top_p=1.0,
                                   action_hint=args.action_hint)
        hits, n_done = {}, 0
        total = len(tasks) * args.repeats
        for task, _k, rd, err in collect(tasks, args.repeats, cfg, args.base_url,
                                         workers=args.workers,
                                         seed_base=args.seed):
            n_done += 1
            won = (0 if err is not None
                   else Rollout.from_dict(rd).outcome.success or 0)
            hits.setdefault(task.task_id, []).append(won)
            # Every 100 was silent for a whole checkpoint when the task set is
            # smaller than that -- a 40-task pass printed nothing for 20 minutes
            # and looked stalled. Scale the interval to the pass instead.
            if n_done % max(min(total // 4, 100), 1) == 0:
                print(f"    {label} {n_done}/{total}", flush=True)
        results[label] = {t: sum(v) / len(v) for t, v in hits.items()}
        flush_cache()
        rate = sum(results[label].values()) / max(len(results[label]), 1)
        print(f"  {label}: {rate:.1%}", flush=True)
        if path is not None:
            sync.unload(name)

    if not results:
        print("no checkpoints evaluated")
        return 1
    common = sorted(set.intersection(*(set(v) for v in results.values())))
    # `--skip_base` leaves no untrained row, and every column below was written
    # assuming one. Fall back to the first checkpoint as the reference and say
    # so, rather than dying after the rollouts have already been paid for --
    # which is what happened the first time the flag was used, losing a
    # 70-minute pass whose per-task results were never written.
    ref_label = "base" if "base" in results else next(iter(results))
    base = results[ref_label]
    if ref_label != "base":
        print(f"\n(no untrained row; deltas are against {ref_label})")
    print(f"\n{len(common)} tasks completed by every checkpoint")
    print(f"\n{'checkpoint':<14}{'success':>10}{'vs base':>10}"
          f"{'95% CI':>22}{'McNemar p':>12}"
          .replace('vs base', f'vs {ref_label}'[:9]))
    out = {}
    for label in results:
        pairs = [(results[label][t], base[t]) for t in common]
        rate = sum(x for x, _ in pairs) / len(pairs)
        delta = sum(x - y for x, y in pairs) / len(pairs)
        lo, hi = paired_bootstrap(pairs, seed=args.seed)
        wins = sum(1 for x, y in pairs if x > y)
        losses = sum(1 for x, y in pairs if x < y)
        p = mcnemar(wins, losses)
        ci = f"[{lo:+.1%}, {hi:+.1%}]"
        print(f"{label:<14}{rate:>10.1%}{delta:>+10.1%}{ci:>22}"
              f"{('' if label == 'base' else f'{p:.3f}'):>12}")
        out[label] = {"success": rate, "delta_vs_base": delta, "ci": [lo, hi],
                      "wins": wins, "losses": losses, "mcnemar_p": p,
                      "per_task": {t: results[label][t] for t in common}}

    p_out = ROOT / args.out
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps({"n_tasks": len(common),
                                 "repeats": args.repeats,
                                 "temperature": args.temperature,
                                 "results": out}, indent=2))
    print(f"\nwritten to {p_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
