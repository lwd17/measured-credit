"""A multi-hop QA environment with a DETERMINISTIC retrieval tool.

The counterfactual this project measures needs a replay that returns the same
thing twice, and the two environments already used give that for free: ALFWorld
and WebShop recompute the outcome themselves from a re-executed action script.
Multi-hop QA does not. The final answer is a string the policy produced, so
re-running a shortened script reproduces that string and the reward never moves
-- the measurement would read zero everywhere by construction.

So the ablation here removes a retrieval step's EVIDENCE and regenerates the
answer from what is left. That costs one generation per ablation, against zero
for the other two, and it is what makes the question "was this retrieval
necessary" answerable at all.

Retrieval is token-overlap over the question's own ten paragraphs, ties broken
by paragraph order, so an identical query returns an identical ranking -- the
same property `bm25_searcher` was written to give WebShop.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for",
         "is", "are", "was", "were", "be", "by", "with", "that", "this", "it",
         "as", "from", "which", "who", "what", "when", "where", "did", "does"}


def _toks(s: str) -> list[str]:
    return [w for w in _WORD.findall((s or "").lower()) if w not in _STOP]


@dataclass
class MultiHopTask:
    task_id: str
    question: str
    answer: str
    titles: list[str]
    paragraphs: list[str]
    gold_titles: list[str]
    level: str = ""
    # The Search-R1 test sets (PopQA and others) allow several acceptable answers
    # per question; `answer` keeps the first one and scoring takes the maximum over
    # all of them. The distractor dataset has a single answer, so the list is empty.
    answers: list[str] = field(default_factory=list)
    source: str = ""

    def all_answers(self) -> list[str]:
        return self.answers or [self.answer]

    def gold_index(self) -> set[int]:
        g = {t.lower() for t in self.gold_titles}
        return {i for i, t in enumerate(self.titles) if t.lower() in g}


def load_multihop(n: int, split: str = "validation", seed_offset: int = 0
                  ) -> list[MultiHopTask]:
    """Evenly spaced questions, so a small sample keeps the type mix.

    Taking the first n instead sorted the sample by whatever order the file
    happens to be in; ALFWorld's loader had that bug and it put 126 of one task
    type in train and none in eval.
    """
    import os
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    d = load_dataset("hotpot_qa", "distractor", split=split)
    total = len(d)
    idx = [(i * total) // n for i in range(n)] if n < total else list(range(total))
    out = []
    for j in idx:
        ex = d[(j + seed_offset) % total]
        ctx = ex["context"]
        out.append(MultiHopTask(
            task_id=str(ex["id"]),
            question=ex["question"],
            answer=ex["answer"],
            titles=list(ctx["title"]),
            paragraphs=[" ".join(s) for s in ctx["sentences"]],
            gold_titles=list(ex["supporting_facts"]["title"]),
            level=ex.get("level", "")))
    return out


class Retriever:
    """Deterministic BM25 top-k over ONE question's paragraphs.

    The first version scored Jaccard overlap, `|q & d| / |q | d|`, which divides
    by the DOCUMENT length and so systematically prefers short paragraphs. On
    HotpotQA the supporting paragraphs are the long ones, so that scoring
    handed the agent distractors for structural reasons and any credit measured
    on top of it would have been measuring the retriever's bias. BM25 is the
    standard choice and normalises length instead of penalising it.

    Ties break by paragraph order so an identical query returns an identical
    ranking, which every delta depends on: a delta is a difference of two
    replays, and a retriever that reorders puts that jitter inside the
    measurement.
    """

    K1, B = 1.5, 0.75

    def __init__(self, task: MultiHopTask, k: int = 2):
        import math

        self.task, self.k = task, k
        self._docs = [_toks(t + " " + p)
                      for t, p in zip(task.titles, task.paragraphs)]
        self._tf = [collections.Counter(d) for d in self._docs]
        self._len = [len(d) or 1 for d in self._docs]
        self._avg = sum(self._len) / len(self._len)
        n = len(self._docs)
        df = collections.Counter()
        for d in self._docs:
            df.update(set(d))
        self._idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5))
                     for w, c in df.items()}

    def score(self, query: str, i: int) -> float:
        total = 0.0
        for w in _toks(query):
            f = self._tf[i].get(w, 0)
            if not f:
                continue
            denom = f + self.K1 * (1 - self.B + self.B * self._len[i] / self._avg)
            total += self._idf.get(w, 0.0) * f * (self.K1 + 1) / denom
        return total

    def search(self, query: str) -> list[int]:
        if not _toks(query):
            return []
        scored = sorted(((self.score(query, i), -i, i)
                         for i in range(len(self._docs))), reverse=True)
        return [i for s, _, i in scored[:self.k] if s > 0]

    def doc(self, i) -> tuple[str, str]:
        return self.task.titles[i], self.task.paragraphs[i]


class WikiRetriever:
    """The Search-R1 open-domain setting: BM25 top-k over all of Wikipedia, served
    by the frozen local index service.

    `/search` returns only a 300-character snippet and `/browse` (wiki://doc_id)
    returns the full 100-word passage, so one retrieval is two HTTP calls. The
    service is deterministic (Lucene score, then docid order), so the same query
    returns the same ranking twice -- the property every delta depends on.
    Query results are cached per process: a query repeated across the 8 rollouts of
    a group is looked up once, and the "same state, different query" candidates of
    the counterfactual measurement hit the same cache.
    """

    def __init__(self, url: str = "http://127.0.0.1:8080", k: int = 3,
                 timeout: float = 60.0):
        import threading
        self.url, self.k, self.timeout = url.rstrip("/"), k, timeout
        self.docs: dict[str, tuple[str, str]] = {}
        self._hits: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self.task = None

    def _post(self, path: str, payload: dict, retries: int = 4) -> dict:
        """The retrieval service is a single-process ThreadingHTTPServer, and under
        five training processes with 16 threads each it occasionally resets the
        connection outright (about 36 times per arm per 40 steps). Losing one
        rollout to a reset is tolerable, but a probe or an evaluation must not die
        from it, hence the retry with backoff."""
        import json as _json
        import time as _time
        import urllib.request
        req = urllib.request.Request(
            f"{self.url}{path}", data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        last = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return _json.loads(r.read())
            except Exception as e:  # noqa: BLE001
                last = e
                _time.sleep(0.3 * (attempt + 1))
        raise last

    def search(self, query: str) -> list[str]:
        q = " ".join((query or "").split())
        if not q:
            return []
        with self._lock:
            if q in self._hits:
                return list(self._hits[q])
        out = self._post("/search", {"query": [q]})
        hits = (out.get("results") or {}).get(q) or []
        ids = [h["doc_id"] for h in hits[:self.k]]
        need = [i for i in ids if i not in self.docs]
        for j in range(0, len(need), 3):          # the server browses at most 3 pages at once
            pages = self._post("/browse", {"url": [f"wiki://{i}" for i in need[j:j + 3]]})
            for key, pg in (pages.get("pages") or {}).items():
                did = key.replace("wiki://", "")
                if "text" in pg:
                    self.docs[did] = (pg.get("title") or "", pg["text"])
        for h in hits[:self.k]:                    # fall back to the snippet if browse failed
            self.docs.setdefault(h["doc_id"], (h.get("title") or "", h.get("snippet") or ""))
        with self._lock:
            self._hits[q] = ids
        return list(ids)

    def doc(self, i) -> tuple[str, str]:
        return self.docs.get(i, ("", ""))


SYSTEM_PROMPT = (
    "You answer multi-hop questions by searching a small document set.\n"
    "Each turn, reply with exactly one line, either\n"
    "  Action: search[your query]\n"
    "  Action: answer[your final answer]\n"
    "Search returns the two most relevant paragraphs. Search as many times as\n"
    "you need, then answer. Keep the answer short: a name, a date, or yes/no."
)

# Prompt for the open-domain (Search-R1) setting: retrieval runs over all of
# Wikipedia and returns three passages at a time.
WIKI_SYSTEM_PROMPT = (
    "You answer questions by searching Wikipedia.\n"
    "Each turn, reply with exactly one line, either\n"
    "  Action: search[your query]\n"
    "  Action: answer[your final answer]\n"
    "Search returns the three most relevant passages. You have a limited number\n"
    "of turns, so search only as many times as you need, then answer. Keep the\n"
    "answer short: a name, a date, a number, or yes/no."
)

SEARCHR1_DIR = "data/searchr1"
SEARCHR1_SOURCES = ("nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa",
                    "musique", "bamboogle")


def load_wikiqa(n: int, split: str = "train", sources=None,
                n_per_source: int | None = None, val_frac: float = 0.02,
                data_dir: str | None = None) -> list[MultiHopTask]:
    """The data released with Search-R1 (PeterJinGo/nq_hotpotqa_train).

    train: the NQ + HotpotQA training sets merged (169,615 questions). Per source,
    the first (1-val_frac) forms the training pool and the last val_frac forms the
    validation pool (`split="val"`); the two are disjoint by question. Checkpoint
    selection looks only at the validation pool, and the test sets take no part in
    any selection.
    test: the seven test sets of Table 2 in the paper (51,713 questions), sampled
    evenly with `n_per_source` per source (Bamboogle has only 125 questions, so all
    of them are taken).
    Even spacing is used for the same reason as in load_multihop: taking the first
    n is at the mercy of the order of the file.
    """
    import os
    from pathlib import Path

    import pandas as pd

    root = Path(data_dir or os.environ.get("SEARCHR1_DIR")
                or Path(__file__).resolve().parents[2] / SEARCHR1_DIR)
    file = "test" if split == "test" else "train"
    df = pd.read_parquet(root / f"{file}.parquet",
                         columns=["id", "question", "golden_answers", "data_source"])
    if sources:
        df = df[df.data_source.isin(list(sources))]
    out = []
    for src_name, part in df.groupby("data_source", sort=True):
        part = part.reset_index(drop=True)
        total = len(part)
        if split == "train":
            part = part.iloc[:int(total * (1 - val_frac))]
        elif split == "val":
            part = part.iloc[int(total * (1 - val_frac)):]
        part = part.reset_index(drop=True)
        want = n_per_source if n_per_source is not None else n
        m = min(want, len(part))
        idx = [(i * len(part)) // m for i in range(m)] if m < len(part) else range(len(part))
        for j in idx:
            r = part.iloc[j]
            answers = [str(a) for a in list(r["golden_answers"])]
            out.append(MultiHopTask(
                task_id=f"{src_name}:{r['id']}", question=str(r["question"]).strip(),
                answer=answers[0] if answers else "", titles=[], paragraphs=[],
                gold_titles=[], answers=answers, source=str(src_name)))
    if n_per_source is None and split != "test":
        # For the training pool, take n evenly spaced questions per source, then
        # interleave the sources and truncate to n (keeping the source proportions)
        by = {}
        for t in out:
            by.setdefault(t.source, []).append(t)
        mixed, i = [], 0
        while len(mixed) < n and any(by.values()):
            for k in sorted(by):
                if by[k]:
                    mixed.append(by[k].pop(0))
            i += 1
        out = mixed[:n]
    return out


@dataclass
class MultiHopState:
    task: MultiHopTask
    retriever: Retriever
    retrieved: list[list[int]] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    final: str = ""
    f1: float = 0.0
    em: float = 0.0

    def evidence(self, drop_step: int | None = None) -> str:
        ids = [i for t, step in enumerate(self.retrieved) if t != drop_step for i in step]
        return self.evidence_for(ids)

    def evidence_for(self, ids) -> str:
        """Render a set of documents in retrieval order, deduplicated. The
        counterfactual measurement uses it to render any prefix or candidate."""
        seen, lines = set(), []
        for i in ids:
            if i in seen:
                continue
            seen.add(i)
            title, text = self.retriever.doc(i)
            lines.append(f"[{title}] {text}")
        return "\n\n".join(lines) if lines else "(no documents retrieved)"


def normalise(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold) -> float:
    golds = [gold] if isinstance(gold, str) else list(gold)
    return 1.0 if any(normalise(pred) == normalise(g) for g in golds) else 0.0


def f1(pred: str, gold) -> float:
    if not isinstance(gold, str):
        return max((f1(pred, g) for g in gold), default=0.0)
    p, g = normalise(pred).split(), normalise(gold).split()
    if not p or not g:
        return float(p == g)
    common = 0
    gg = list(g)
    for w in p:
        if w in gg:
            gg.remove(w)
            common += 1
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)
