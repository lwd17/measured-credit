#!/usr/bin/env python
"""Strict evaluation for open-domain QA (the Search-R1 setting): the seven test
sets of Table 2 in the paper, greedy decoding, paired per question.

The evaluation inside the training loop only looks at the validation questions
held out of the training pool (`--split val`) and is used to pick checkpoints. The
test sets (`--split test`) are reported only at the end, taking `--n_per_source`
evenly spaced questions per source (all 125 of Bamboogle). The same set of
questions is reused for every checkpoint, so differences between arms can be
paired per question.

    python scripts/wq_eval.py --runs wq_grpo_s0,wq_L_mxS_s0 --steps 40,80,120 \
        --split test --n_per_source 300 --out reports/wq_test.json
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _paired(a: dict, b: dict, common) -> tuple[float, float]:
    d = [a[t] - b[t] for t in common]
    mu = statistics.mean(d) if d else 0.0
    se = statistics.pstdev(d) / max(len(d) ** 0.5, 1) if len(d) > 1 else 0.0
    return mu, 1.96 * se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated directory names under runs/")
    ap.add_argument("--steps", default="120")
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--n_per_source", type=int, default=300)
    ap.add_argument("--n_val", type=int, default=400)
    ap.add_argument("--max_turns", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--doc_chars", type=int, default=1200)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--base", action="store_true", help="also evaluate the base model")
    ap.add_argument("--base_url", default="http://localhost:8100/v1")
    ap.add_argument("--served_name", default="Qwen3-8B")
    ap.add_argument("--wiki_url", default="http://127.0.0.1:8080")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor

    from ssc.env.multihop_env import load_wikiqa
    from ssc.env.multihop_policy import MultiHopPolicy, MultiHopPolicyConfig
    from ssc.train.sync import LoRASync

    if args.split == "test":
        tasks = load_wikiqa(0, split="test", n_per_source=args.n_per_source)
    else:
        tasks = load_wikiqa(args.n_val, split="val")
    print(f"{args.split}: {len(tasks)} questions  sources "
          f"{collections.Counter(t.source for t in tasks)}", flush=True)

    out_path = Path(args.out or ROOT / "reports" / f"wq_{args.split}.json")
    result = {"per_task_em": {}, "per_task_f1": {}, "summary": {}, "n_tasks": len(tasks),
              "config": vars(args)}
    if out_path.exists():
        try:
            old = json.loads(out_path.read_text())
            if old.get("n_tasks") == len(tasks) and old.get("config", {}).get("split") == args.split:
                result["per_task_em"], result["per_task_f1"] = old["per_task_em"], old["per_task_f1"]
                result["summary"] = old.get("summary", {})
        except Exception:  # noqa: BLE001
            pass

    sync = LoRASync(args.base_url)
    cfg = MultiHopPolicyConfig(model=args.served_name, max_turns=args.max_turns,
                               max_tokens_per_turn=args.max_tokens, temperature=0.0, top_p=1.0,
                               retriever="wiki", wiki_url=args.wiki_url, top_k=args.top_k,
                               doc_chars=args.doc_chars, reward="em")
    policy = MultiHopPolicy(cfg, args.base_url)

    def run(label: str, model_name: str):
        if label in result["per_task_em"]:
            print(f"  {label}: already have results, skipping", flush=True)
            return
        cfg.model = model_name
        t0 = time.time()

        def one(t):
            for attempt in range(3):
                try:
                    r, s = policy.rollout(t, rollout_index=0, seed=7)
                    return t, r, s
                except Exception as e:  # noqa: BLE001
                    err = e
            print(f"  ! {t.task_id}: {type(err).__name__}: {err}", flush=True)
            return None

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            got = [x for x in ex.map(one, tasks) if x is not None]
        # The retrieval service crashed once, on 09-06 at 22:50 (JVM SIGSEGV). Only
        # a handful of questions completed in the evaluation that was running, and
        # the result was recorded as if it were complete: `L_gR@80 = 1.000` was
        # computed from 1 question. If coverage is too low, record nothing: a
        # partial result is far more dangerous than no result at all.
        if len(got) < 0.95 * len(tasks):
            print(f"  {label}: only {len(got)}/{len(tasks)} questions completed, coverage too "
                  f"low, not recording (the retrieval service or the sampler may be down)",
                  flush=True)
            return
        em = {t.task_id: s.em for t, _, s in got}
        f1 = {t.task_id: s.f1 for t, _, s in got}
        result["per_task_em"][label] = em
        result["per_task_f1"][label] = f1
        by = collections.defaultdict(list)
        for t, r, s in got:
            by[t.source].append((s.em, s.f1, len(r.turns)))
        summ = {src: {"em": round(statistics.mean(x[0] for x in v), 4),
                      "f1": round(statistics.mean(x[1] for x in v), 4),
                      "turns": round(statistics.mean(x[2] for x in v), 2), "n": len(v)}
                for src, v in sorted(by.items())}
        summ["avg"] = {"em": round(statistics.mean(v["em"] for k, v in summ.items()), 4),
                       "f1": round(statistics.mean(v["f1"] for k, v in summ.items()), 4),
                       "n": len(got)}
        summ["all"] = {"em": round(statistics.mean(em.values()), 4),
                       "f1": round(statistics.mean(f1.values()), 4)}
        result["summary"][label] = summ
        print(f"  {label}: avgEM={summ['avg']['em']:.3f} allEM={summ['all']['em']:.3f} "
              f"{ {k: v['em'] for k, v in summ.items() if k not in ('avg', 'all')} } "
              f"({time.time() - t0:.0f}s)", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))

    if args.base:
        run("base", args.served_name)
    steps = [int(s) for s in args.steps.split(",") if s]
    for run_name in args.runs.split(","):
        run_name = run_name.strip()
        for step in steps:
            adir = ROOT / args.runs_dir / run_name / f"adapter_step{step}"
            if not (adir / "adapter_config.json").exists():
                print(f"  {run_name}@{step}: no checkpoint, skipping", flush=True)
                continue
            label = f"{run_name}@{step}"
            if label in result["per_task_em"]:
                print(f"  {label}: already have results, skipping", flush=True)
                continue
            name = f"ck_{run_name}_{step}"
            sync.load(name, str(adir))
            sync.verify_active(name)
            try:
                run(label, name)
            finally:
                try:
                    sync.unload(name)
                except Exception:  # noqa: BLE001
                    pass

    # Paired differences (against base, or against the first label)
    labels = list(result["per_task_em"])
    ref = "base" if "base" in labels else (labels[0] if labels else None)
    if ref:
        for lab in labels:
            common = set(result["per_task_em"][lab]) & set(result["per_task_em"][ref])
            mu, hw = _paired(result["per_task_em"][lab], result["per_task_em"][ref], common)
            result["summary"].setdefault(lab, {})["delta_em_vs_" + ref] = [round(mu, 4), round(mu - hw, 4), round(mu + hw, 4)]
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print("\nPaired difference (EM, against " + ref + "):")
        for lab in labels:
            d = result["summary"][lab]["delta_em_vs_" + ref]
            print(f"  {lab:28s} {d[0]:+.4f}  [{d[1]:+.4f}, {d[2]:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
