"""Outcome reward for search QA (runbook sections 10, 16).

Section 10.1's recommended v1 reward is the terminal task reward. For search QA
that is exact match against the reference answer, following Search-R1.

This module is the ONLY place the gold answer is used during training, and it
is deliberately separate from everything the detector touches: section 11
forbids the detector from seeing the gold answer, the success flag or the
reward, and section 55 Test 7 asserts that the detector payload cannot carry
them. Keeping scoring here, rather than on the Rollout builder, is what makes
that separation checkable.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """SQuAD/HotpotQA normalisation: lowercase, strip punctuation/articles/space."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


def f1(prediction: str, gold: str) -> float:
    p = normalize_answer(prediction).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def substring_match(prediction: str, gold: str) -> int:
    """Search-R1 scores a prediction correct if the gold string appears in it.

    Kept available but NOT the default: the policy's final answer here is a
    sentence ("No, Chumbawamba is from England while..."), and a one-token gold
    such as "no" matches almost any sentence containing it. That inflates
    reward in exactly the direction that would make the emergence curve look
    better than it is.
    """
    return int(normalize_answer(gold) in normalize_answer(prediction))


def score_rollout(prediction: str, gold: str, mode: str = "em") -> dict:
    """Terminal task reward. Returns the fields Outcome expects."""
    em = exact_match(prediction, gold)
    f = f1(prediction, gold)
    if mode == "em":
        r = float(em)
    elif mode == "f1":
        r = f
    elif mode == "substring":
        r = float(substring_match(prediction, gold))
    else:
        raise ValueError(f"unknown reward mode {mode!r}")
    return {"success": int(r > 0.5), "reward": r, "em": em, "f1": round(f, 4)}
