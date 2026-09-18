"""Semantic survival coefficients (runbook sections 6, 21, 52).

q_t answers: "what is the probability that the semantic regime governing turn t
was still valid at the end of the rollout?"

Hard version (section 6.1):   q_t = 0 for b <= t < c
Soft version (section 6.2):   q_t = prod over covering events of (1 - p_m)

Only accepted REPLACE events participate (section 6.3). REFINE and COMMIT are
ordinary task progression; gating them would erase exactly the long-range
credit the method is trying to preserve.
"""

from __future__ import annotations

from ssc.detector.schema import SemanticEvent


def build_turn_survival(num_turns: int, events: list[SemanticEvent],
                        *, hard: bool = False) -> list[float]:
    """Section 52. Returns q of length num_turns, one coefficient per action turn.

    `hard=True` forces p=1 for every accepted event, giving section 6.1's
    0/1 coefficient. Otherwise detector confidence is used (section 6.2).

    Overlapping spans multiply, which is the correct reading of section 6.2's
    product: a turn governed by two independently invalidated regimes survives
    less than one governed by either alone.
    """
    if num_turns < 0:
        raise ValueError("num_turns must be non-negative")
    q = [1.0] * num_turns

    for e in events:
        if not e.is_active:            # section 6.3: REPLACE + accepted only
            continue
        p = 1.0 if hard else e.confidence
        for t in e.turns():
            if 0 <= t < num_turns:
                q[t] *= (1.0 - p)
    return q


def semantic_horizon(events: list[SemanticEvent]) -> int:
    """H_sem = 1 + N_replace (section 21).

    A trajectory with no validated replacement has horizon 1: it is a single
    semantic regime, not zero regimes.
    """
    return 1 + sum(1 for e in events if e.is_active)


def expand_to_tokens(turn_q: list[float], turn_of_token: list[int],
                     actor_mask: list[int | bool]) -> list[float]:
    """Broadcast per-turn q to per-token q (section 7).

    `turn_of_token[j]` is the action turn that emitted token j, or -1 for
    tokens that no action turn produced (environment observations, padding).

    Section 9 / Test 5: non-actor tokens carry zero loss weight before and
    after SSC, so they are given 0.0 here and must be excluded from every
    normalisation statistic.
    """
    if len(turn_of_token) != len(actor_mask):
        raise ValueError("turn_of_token and actor_mask must align")
    out = []
    for tok_turn, m in zip(turn_of_token, actor_mask):
        if not m:
            out.append(0.0)
        elif 0 <= tok_turn < len(turn_q):
            out.append(turn_q[tok_turn])
        else:
            # An actor token with no owning turn would silently receive full
            # credit; that is a serialisation bug, not a modelling choice.
            raise ValueError(
                f"actor token maps to turn {tok_turn}, outside 0..{len(turn_q)-1}")
    return out
