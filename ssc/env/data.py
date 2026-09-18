"""Task loading for the Phase 0 audit (runbook sections 16, 20).

Section 16 names HotpotQA as the primary dataset, with NQ / 2Wiki / MuSiQue as
options. Section 20 needs 50 tasks x 2 rollouts.

Task selection is a deterministic function of the question text, so the audit
set is reproducible and independent of dataset iteration order or of how many
tasks were requested -- asking for 50 and later for 200 must give a superset,
not a different sample.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

SEED_NAMESPACE = "ssc-phase0-20260816"


@dataclass
class Task:
    task_id: str
    question: str
    answer: str
    dataset: str
    n_hops: int | None = None


def stable_key(question: str) -> str:
    q = " ".join(question.strip().split()).lower()
    return hashlib.sha256((SEED_NAMESPACE + "::" + q).encode("utf-8")).hexdigest()


def task_seed(question: str, rollout_index: int = 0) -> int:
    """Per-(task, rollout) sampling seed. Rollouts of one task must differ."""
    h = hashlib.sha256(
        f"{SEED_NAMESPACE}::{stable_key(question)}::{rollout_index}".encode()).hexdigest()
    return int(h[:16], 16) % (2 ** 31)


def load_hotpotqa(n: int, split: str = "validation",
                  level: str | None = "hard") -> list[Task]:
    """HotpotQA distractor split.

    `level="hard"` keeps the multi-hop subset. Single-hop questions are answered
    in one turn, which cannot contain a regime replacement by construction --
    including them would deflate section 22's prevalence for a reason that has
    nothing to do with the phenomenon.
    """
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split=split)
    rows = []
    for r in ds:
        if level and r.get("level") != level:
            continue
        rows.append(Task(task_id="", question=r["question"],
                         answer=r["answer"], dataset="hotpotqa"))
    rows.sort(key=lambda t: stable_key(t.question))
    for t in rows:
        t.task_id = "hp_" + stable_key(t.question)[:12]
    return rows[:n]


def load_musique(n: int, min_hops: int = 4) -> list[Task]:
    """MuSiQue answerable subset, for the harder multi-hop arm (section 16).

    Defaults to 4-hop. Hop count bounds the achievable semantic horizon: a
    correctly executed 2-hop question is 2 searches plus an answer, and a
    section 12 REPLACE needs adoption -> a governed action -> invalidation ->
    switch, so 2-hop tasks can barely host the phenomenon even under ideal
    behaviour. Available: 1252 2-hop, 760 3-hop, 405 4-hop.
    """
    from datasets import load_dataset

    ds = load_dataset("bdsaglam/musique", "answerable", split="validation")
    rows = []
    for r in ds:
        hops = str(r.get("id", "")).count("hop")
        n_hop = int(str(r["id"]).split("hop")[0]) if "hop" in str(r["id"]) else None
        if n_hop is not None and n_hop < min_hops:
            continue
        rows.append(Task(task_id="", question=r["question"],
                         answer=r["answer"], dataset="musique", n_hops=n_hop))
    rows.sort(key=lambda t: stable_key(t.question))
    for t in rows:
        t.task_id = "mq_" + stable_key(t.question)[:12]
    return rows[:n]


LOADERS = {"hotpotqa": load_hotpotqa, "musique": load_musique}
