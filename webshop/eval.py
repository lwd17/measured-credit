#!/usr/bin/env python
"""Score saved WebShop adapters against each other on one held-out task set.

The trainer's own eval is 48 tasks run once, which is too coarse to separate
arms that finish within a few points of each other: at n = 48 the standard error
on a success rate near 30% is about 6.6 points, so a 4-point difference is
indistinguishable from nothing. This runs every checkpoint over the SAME tasks
with the same seeds and pairs the comparison per task, which removes the
task-difficulty variance that dominates that error.

Both a graded score and an exact-match success rate are reported: WebShop pays
partial credit for attribute overlap, and a method can move one without moving
the other.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="grpo")
    ap.add_argument("--steps", default="20,40,60")
    ap.add_argument("--n_tasks", type=int, default=120)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max_turns", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip_base", action="store_true")
    ap.add_argument("--base_url", default="http://localhost:8102/v1")
    ap.add_argument("--served_name", default="Qwen3-8B")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--seed", type=int, default=0,
                    help="checkpoint directory suffix runs/ws_<arm>_s<seed>; for a "
                         "non-zero seed the label carries _s<seed> so it cannot collide "
                         "with the JSON / summary rows of seed 0")
    ap.add_argument("--out", default="reports/ws_checkpoint_eval.json")
    args = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor

    from ssc.env.webshop_env import WebShopWorker, load_webshop
    from ssc.env.webshop_policy import WebShopPolicy, WebShopPolicyConfig
    from ssc.train.sync import LoRASync

    tasks = load_webshop(args.n_tasks, split="eval")
    sync = LoRASync(args.base_url)

    targets: list[tuple[str, Path | None]] = []
    if not args.skip_base:
        targets.append(("base", None))
    for arm in args.arms.split(","):
        for st in args.steps.split(","):
            path = ROOT / args.runs / f"ws_{arm}_s{args.seed}" / f"adapter_step{st}"
            if path.exists():
                targets.append((f"{arm}{'' if args.seed == 0 else f'_s{args.seed}'}@{st}", path))
            else:
                print(f"  {arm}@{st}: missing {path}", flush=True)

    print(f"{len(tasks)} eval tasks x {args.repeats} runs, "
          f"temperature={args.temperature}, {len(targets)} checkpoints", flush=True)

    # ONE WORKER PER THREAD. A worker holds a single live environment, so two
    # rollouts sharing one interleave their `step` calls on the same session,
    # and `_call` writes a request then reads a line with no lock, so replies
    # come back to the wrong caller and every thread blocks -- observed as a
    # run frozen at 180/240 with the sampler at 0% utilisation. The trainer
    # collects rollouts in a plain sequential loop for exactly this reason.
    # Giving each thread its own worker keeps the generation calls (the wall
    # clock) overlapping while every environment stays private to one rollout.
    workers = [WebShopWorker() for _ in range(args.workers)]
    scores: dict[str, dict[str, float]] = {}
    wins: dict[str, dict[str, float]] = {}
    try:
        for label, path in targets:
            name = args.served_name
            if path is not None:
                name = f"ck_{label.replace('@', '_')}"
                sync.load(name, str(path))
                sync.verify_active(name)
            cfg = WebShopPolicyConfig(model=name, max_turns=args.max_turns,
                                      max_tokens_per_turn=args.max_tokens,
                                      temperature=args.temperature, top_p=1.0)
            policy = WebShopPolicy(cfg, args.base_url)

            got: dict[str, list[tuple[float, float]]] = {}

            # Each rollout owns one environment exclusively: take it from the
            # queue, hand it back when done. Slots used to be assigned by
            # n % workers, but a thread picks up the next job as soon as it
            # frees up, so job n+W can be handed the same slot while job n is
            # still running -- two rollouts then interleave steps on one WebShop
            # session and contaminate each other's trajectories. Exceptions are
            # no longer silently scored 0 either: retry twice, and only score 0
            # (and count it) if it still fails.
            import queue as _queue
            pool: "_queue.Queue" = _queue.Queue()
            for w in workers:
                pool.put(w)
            n_fail = [0]

            def one(job):
                _slot, t, k = job
                for attempt in range(3):
                    w = pool.get()
                    try:
                        r = policy.rollout(t, rollout_index=k, seed=90000 + k,
                                           worker=w)
                        return t.task_id, float(r.outcome.reward or 0.0), \
                            1.0 if r.outcome.success else 0.0
                    except Exception as e:  # noqa: BLE001
                        if attempt == 2:
                            n_fail[0] += 1
                            print(f"    ! {t.task_id} failed three times, scored 0: {type(e).__name__}: {e}",
                                  flush=True)
                            return t.task_id, 0.0, 0.0
                    finally:
                        pool.put(w)

            flat = [(t, k) for t in tasks for k in range(args.repeats)]
            jobs = [(n % args.workers, t, k) for n, (t, k) in enumerate(flat)]
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for n_done, (tid, sc, ok) in enumerate(ex.map(one, jobs), 1):
                    got.setdefault(tid, []).append((sc, ok))
                    if n_done % max(len(jobs) // 4, 1) == 0:
                        print(f"    {label} {n_done}/{len(jobs)}", flush=True)
            scores[label] = {t: sum(a for a, _ in v) / len(v) for t, v in got.items()}
            wins[label] = {t: sum(b for _, b in v) / len(v) for t, v in got.items()}
            print(f"  {label}: score={statistics.mean(scores[label].values()):.3f} "
                  f"success={statistics.mean(wins[label].values()):.1%}"
                  + (f"  rollouts scored 0 after failure: {n_fail[0]}" if n_fail[0] else ""), flush=True)
    finally:
        for w in workers:
            w.close()

    common = set.intersection(*(set(v) for v in scores.values()))
    ref = "base" if "base" in scores else sorted(scores)[0]
    print(f"\n{len(common)} tasks completed by every checkpoint\n")
    print(f"{'checkpoint':<20}{'score':>8}{'success':>10}"
          f"{'vs ' + ref:>12}{'95% CI':>20}")
    out = {}
    for label in scores:
        d = [scores[label][t] - scores[ref][t] for t in common]
        mu = statistics.mean(d)
        se = statistics.pstdev(d) / max(len(d) ** 0.5, 1) if len(d) > 1 else 0.0
        out[label] = {"score": statistics.mean(scores[label][t] for t in common),
                      "success": statistics.mean(wins[label][t] for t in common),
                      "delta_score": mu, "ci": [mu - 1.96 * se, mu + 1.96 * se]}
        print(f"{label:<20}{out[label]['score']:>8.3f}{out[label]['success']:>9.1%}"
              f"{mu:>+12.3f}   [{mu - 1.96 * se:+.3f}, {mu + 1.96 * se:+.3f}]")
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"per_task_score": scores, "per_task_success": wins,
                             "summary": out}, indent=2))
    print(f"\nwritten to {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
