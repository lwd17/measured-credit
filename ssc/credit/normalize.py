"""Mass-preserving credit redistribution (runbook sections 7, 53).

Zeroing token weights shrinks the total gradient for exactly those rollouts
that contain the most invalidated computation. That confounds "better credit
assignment" with "smaller effective learning rate on long trajectories" --
which is why section 7 makes the mass-preserving variant the MAIN one and asks
for the raw variant to be reported alongside it.

Normalisation is over ACTOR TOKENS ONLY. Including masked positions would make
the mean depend on padding length, i.e. on batching.
"""

from __future__ import annotations

EPS = 1e-8


def mass_preserve(token_q: list[float], actor_mask: list[int | bool],
                  eps: float = EPS) -> list[float]:
    """Section 53. Rescale so the mean weight over actor tokens is 1.

    The degenerate case -- every actor token invalidated -- falls back to
    uniform weights rather than dividing by ~0. Section 53 calls this
    defensive; it is reachable in practice when a short trajectory is entirely
    one abandoned regime, and blowing up to 1/eps would produce a single
    enormous update.
    """
    if len(token_q) != len(actor_mask):
        raise ValueError("token_q and actor_mask must align")

    active = [q for q, m in zip(token_q, actor_mask) if m]
    if not active:
        return [0.0] * len(token_q)

    mean_q = sum(active) / len(active)
    if mean_q < eps:
        return [1.0 if m else 0.0 for m in actor_mask]

    return [(q / mean_q) if m else 0.0 for q, m in zip(token_q, actor_mask)]


def mean_actor_weight(token_q: list[float], actor_mask: list[int | bool]) -> float:
    """Diagnostic for section 53's assertion."""
    active = [q for q, m in zip(token_q, actor_mask) if m]
    return sum(active) / len(active) if active else 0.0


def assert_mass_preserved(token_q: list[float], actor_mask: list[int | bool],
                          tol: float = 1e-4) -> None:
    """Section 53's diagnostic assertion, callable from the training loop."""
    if not any(actor_mask):
        return
    m = mean_actor_weight(token_q, actor_mask)
    if abs(m - 1.0) > tol:
        raise AssertionError(
            f"mass not preserved: mean actor weight {m:.6f} != 1 (tol {tol})")
