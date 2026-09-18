"""Exact per-turn marginal contribution, by re-executing the environment.

Every criterion tried before this one was a PROXY for how much a turn
contributed, and measured against ground truth all of them were close to
uninformative: dataflow unreachability separated load-bearing turns from
dispensable ones at 1.22x the base rate, the avoidable-waste rule at 1.08x, and
a step-level baseline that estimates credit from observed returns at 1.41x,
against a base rate of 8.8%.

The proxies exist because the quantity they approximate was assumed to be
unobservable. In a deterministic, replayable environment it is not:

    delta_t = score(tau) - score(tau without action t)

is the turn's marginal contribution, and it costs environment steps rather than
model calls. Measured on ScienceWorld the whole leave-one-out sweep for one
trajectory runs in about 0.05 * T^2 seconds -- ~31 s at T = 25 -- so a training
step's worth of rollouts costs seconds of wall clock spread over spare cores,
against a step that already takes ~20 minutes on the GPU.

## What this is not

It is NOT reward-blind. It reads the environment score, so it is privileged
training-time information in the sense the credit-assignment literature uses
that phrase. That constraint was worth holding while the criterion was a
semantic guess -- it stopped the detector from simply keying on success -- but
it has no force here, because nothing is being inferred: the causal effect is
measured directly, and measuring it is what credit assignment means.

## Known bias

Leave-one-out under-counts turns that are REDUNDANT with another turn: if two
actions are alternatives, removing either alone changes nothing, so both look
dispensable while removing both would not be. Every such pair inflates the
dispensable fraction. The direction is known and one-sided, which is why the
weight below keeps a floor of uniform credit rather than zeroing everything the
sweep calls dispensable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ssc.detector.rollout import Rollout


def rollout_actions(rollout: Rollout) -> list[str]:
    return [str((t.tool_args or {}).get("action", "")) for t in rollout.turns]


def variation_of(rollout: Rollout) -> int | None:
    """ScienceWorld task ids carry the variation as a trailing `_<n>`."""
    m = re.search(r"_(\d+)$", rollout.task_id or "")
    return int(m.group(1)) if m else None


def replay_score(env, task_name: str, variation: int, acts: list[str]) -> float:
    """Final score of a scripted action sequence.

    `step()` returns the score DELTA; the cumulative score is `info["score"]`,
    and the rollout recorder keeps its LAST value. Reading the delta instead
    made every replay return the same number and would have scored every turn
    against a baseline that never matched the recorded trajectory.
    """
    env.load(task_name, variation, "")
    env.reset()
    score = 0.0
    for a in acts:
        if not a.strip():
            continue
        _, _, done, info = env.step(a)
        score = float(info.get("score", score))
        if done:
            break
    return score


@dataclass
class TurnContribution:
    base_score: float
    deltas: list[float]          # score(tau) - score(tau \ t), one per turn


def counterfactual_deltas(env, rollout: Rollout, task_name: str,
                          variation: int | None = None) -> TurnContribution:
    var = variation if variation is not None else variation_of(rollout)
    acts = rollout_actions(rollout)
    if var is None or not acts:
        return TurnContribution(0.0, [0.0] * len(acts))
    base = replay_score(env, task_name, var, acts)
    deltas = [base - replay_score(env, task_name, var, acts[:i] + acts[i + 1:])
              for i in range(len(acts))]
    return TurnContribution(base, deltas)


def contribution_weights(deltas: list[float], alpha: float = 0.5,
                         w_max: float = 3.0) -> list[float]:
    """Per-turn allocation weights from marginal contributions.

    ``alpha`` is the floor of uniform credit that survives regardless of
    contribution, and it is the dose knob:

        alpha = 1  ->  s is constant, w == 1 everywhere, EXACTLY GRPO
        alpha = 0  ->  credit is proportional to measured contribution alone

    ``w_max`` caps the amplification that conservation produces. It is not a
    tuning convenience: 91.7% of turns have delta = 0, so conserving mass across
    the survivors alone multiplies them by ~12, and an amplification of that
    size is what drove the earlier collapse -- larger steps degrade the policy,
    a degraded policy produces more discountable turns, and the factor grows
    again. Capping it breaks that loop by construction.

    Capping and conservation fight each other, and resolving that badly would
    have wrecked the dose-response: clamping after normalising drops total
    credit mass, so a smaller alpha would also mean a smaller effective step,
    and the sweep would confound DOSE with LEARNING RATE. So the scale is solved
    for instead --

        find c > 0 such that  mean_t min(c * s_t, w_max) = 1

    -- which is monotone in c and therefore a bisection. Mean weight is exactly
    1 whenever that is attainable, and it is attainable for every alpha > 0,
    since then every s_t > 0 and the attainable mean rises to w_max >= 1. Only
    alpha == 0 exactly can fail, when too few turns carry any contribution at
    all; that case keeps the largest achievable scale and is reported by the
    weights summing to less than n.
    """
    n = len(deltas)
    if n == 0:
        return []
    hi = max(deltas) if deltas else 0.0
    norm = [(d / hi if hi > 0 else 0.0) for d in deltas]
    s = [alpha + (1.0 - alpha) * x for x in norm]
    if max(s) <= 0:
        return [1.0] * n                       # nothing measured: stay uniform

    def mean_at(c: float) -> float:
        return sum(min(c * x, w_max) for x in s) / n

    # The search ceiling is the scale at which the SMALLEST positive weight
    # reaches the cap, not the largest. Bounding it by the largest stops the
    # search while the small entries can still grow -- the big ones merely sit
    # clamped -- and conservation gets declared impossible when it is not: at
    # alpha = 0.1 that reported a mean weight of 0.43 instead of 1.0, which
    # would have made every low-alpha arm train at a quietly smaller effective
    # step and confounded the dose-response with learning rate.
    pos = [x for x in s if x > 0]
    ceiling = w_max / min(pos)
    if mean_at(ceiling) < 1.0:                 # genuinely unattainable
        return [min(ceiling * x, w_max) for x in s]
    lo, high = 0.0, ceiling
    for _ in range(60):
        mid = (lo + high) / 2
        if mean_at(mid) < 1.0:
            lo = mid
        else:
            high = mid
    c = (lo + high) / 2
    return [min(c * x, w_max) for x in s]


EPS = 1e-9


def signed_weights(deltas: list[float], keep: list[bool] | None = None,
                   dose: float = 1.0, w_max: float = 3.0) -> list[float]:
    """Mass-preserving allocation weights that are allowed to go negative.
    Sister form of `contribution_weights`.

        q_t = 1 + dose * (d_t - mean_measured(d)) / max_measured|d - mean|

    Three properties hold by construction, not by tuning:

      dose = 0   q is identically 1, reducing exactly to GRPO
      mean(q)    is exactly 1 over the measured turns (because d - mean has zero
                 mean); unmeasured turns take the neutral value 1, so the total
                 advantage mass of a trajectory equals GRPO's
      dose > 1   the least necessary turns get q < 0, so a negative sign can
                 appear even inside a successful trajectory

    The last one is the only reason this form exists. `contribution_weights`
    forces q >= 0, so the strongest thing it can do to a useless turn is set its
    weight to zero and not update on it. On ALFWorld the additive form's gain
    comes precisely from ACTIVELY PENALISING the replaceable turns of a
    successful trajectory (26-54% of turns measured with a negative final
    advantage, and a placement control with zero effect, which says the negative
    signs land in the right places). Lowering alpha only widens the spread and
    does not change the ability to produce a sign, so it cannot recover this --
    a falsifiable claim, measured on real deltas by
    `scripts/probes/alf_dose.py`.

    `keep` is the measurement mask. The mean is taken over the measured turns
    only: on ALFWorld every turn is measurable, so a measured zero is correctly
    pushed below the mean; WebShop and multi-hop pass a mask, so an unmeasured
    zero neither enters the statistics nor collects credit. That avoids reading
    "never measured" as "measured and zero", which is exactly how the additive
    form failed in those two environments.

    For dose <= 1, q lies in [1-dose, 1+dose], no clamping is needed and
    mean(q) is exactly 1. For dose > 1 the lower bound is negative and the upper
    bound is 1+dose; only values above w_max are clamped, which loses a little
    mass -- visibly in the log rather than silently.
    """
    n = len(deltas)
    if n == 0:
        return []
    keep = [True] * n if keep is None else list(keep)
    measured = [d for d, k in zip(deltas, keep) if k]
    if len(measured) < 2:
        return [1.0] * n
    mu = sum(measured) / len(measured)
    # The scale is the MEAN absolute deviation, not the maximum. ALFWorld's
    # deltas are sparse positives (measured at 45-64% non-zero per trajectory,
    # the rest exactly zero), and with max|d-mu| the denominator is dominated by
    # the positive outlier: for deltas=[1,0,...,0], mu=0.1 and max=0.9, so a
    # zero turn gets weight 1-dose*0.11, which is still 0.78 at dose=2 and never
    # reaches a negative value. That is why the diagnostics reported 0%
    # "negative in successful trajectory" at dose=2 and only 10% at dose=3. With
    # MAD the same data gives MAD=0.18 and dose=2 yields -0.11, so the sign
    # actually appears and the dose knob means something.
    mad = sum(abs(d - mu) for d in measured) / len(measured)
    if mad < EPS:                        # all equal: no difference to allocate
        return [1.0] * n
    q = [1.0 + dose * (d - mu) / mad if k else None
         for d, k in zip(deltas, keep)]
    # Clamping breaks mass conservation: at dose=3 with sparse deltas, 45% of
    # turns are clipped by w_max=3 and the mean falls from 1 to 0.40 -- a 60%
    # shrink of the whole trajectory's advantage, which is a change in learning
    # rate rather than in credit, and confounds DOSE with EFFECTIVE STEP SIZE
    # (the comment on contribution_weights records the same trap on WebShop).
    # So after clamping the mean is pulled back to 1 by an additive shift, then
    # clamped again; a few iterations converge, and a shift preserves the
    # relative ordering of the turns.
    idx = [i for i, x in enumerate(q) if x is not None]
    for _ in range(4):
        for i in idx:
            q[i] = min(w_max, q[i])
        m = sum(q[i] for i in idx) / len(idx)
        if abs(m - 1.0) < 1e-9:
            break
        free = [i for i in idx if q[i] < w_max - 1e-9]
        if not free:
            break
        shift = (1.0 - m) * len(idx) / len(free)
        for i in free:
            q[i] += shift
    return [1.0 if x is None else x for x in q]


# --------------------------------------------------------------------------
# Batch computation for the training loop
# --------------------------------------------------------------------------
#
# The sweep is quadratic in turns and the GPU is busy for ~20 minutes a step, so
# the cost only matters if it blocks. It does not: one worker per process, each
# holding its own ScienceWorld JVM, turns ~500 s of sequential replay for a
# 16-rollout step into ~16 s of wall clock. The pool is created once and the
# environment is built in the initialiser, because building it per task would
# cost more than the work.

_ENV = None
_TASK = None


def _init_worker(task_name: str) -> None:
    global _ENV, _TASK
    from scienceworld import ScienceWorldEnv
    _ENV = ScienceWorldEnv("", envStepLimit=200)
    _TASK = task_name


def _one(payload):
    idx, acts, var = payload
    if var is None or not acts:
        return idx, 0.0, [0.0] * len(acts)
    base = replay_score(_ENV, _TASK, var, acts)
    deltas = [base - replay_score(_ENV, _TASK, var, acts[:i] + acts[i + 1:])
              for i in range(len(acts))]
    return idx, base, deltas


_POOL = None
_POOL_TASK = None


def _get_pool(task_name: str, workers: int):
    """One pool for the whole run, rebuilt only if it has broken.

    Creating it per step meant spawning `workers` JVMs every step -- a fixed
    15-30 s that is invisible against a 20-minute step but dominated the smoke
    configuration, and that is pure waste over 60 steps. Holding it costs one
    startup for the run.

    `maxtasksperchild` still recycles a worker every 32 rollouts. With 16
    rollouts a step and 60 steps each worker would otherwise hold one JVM alive
    across ~60 `env.load()` calls, and an unattended overnight run is the wrong
    place to find out whether that leaks.
    """
    global _POOL, _POOL_TASK
    import multiprocessing as mp
    if _POOL is not None and _POOL_TASK == task_name:
        return _POOL
    _close_pool()
    ctx = mp.get_context("spawn")
    _POOL = ctx.Pool(processes=workers, initializer=_init_worker,
                     initargs=(task_name,), maxtasksperchild=32)
    _POOL_TASK = task_name
    return _POOL


def _close_pool():
    global _POOL, _POOL_TASK
    if _POOL is not None:
        try:
            _POOL.terminate(); _POOL.join()
        except Exception:
            pass
    _POOL, _POOL_TASK = None, None


def counterfactual_deltas_batch(rollouts, task_name: str, workers: int = 16,
                                timeout: float = 1800.0):
    """Per-turn deltas for a batch of rollouts, computed on a persistent pool.

    Returns a list aligned with `rollouts`. A rollout whose replay fails yields
    all-zero deltas, which `contribution_weights` turns into uniform weights --
    the arm degrades to GRPO on that rollout rather than being dropped or, far
    worse, credited arbitrarily. A pool-level failure tears the pool down so the
    next step rebuilds it, because a half-dead pool would return zeros for every
    rollout from then on and the run would look like a method that does nothing.
    """
    payloads = [(i, rollout_actions(r), variation_of(r))
                for i, r in enumerate(rollouts)]
    out = [TurnContribution(0.0, [0.0] * len(p[1])) for p in payloads]
    if not payloads:
        return out
    try:
        pool = _get_pool(task_name, min(workers, max(1, len(payloads))))
        for idx, base, deltas in pool.imap_unordered(_one, payloads):
            out[idx] = TurnContribution(base, deltas)
    except Exception as e:                      # keep training, but rebuild
        print(f"[counterfactual] pool failed ({type(e).__name__}: {e}); "
              f"rebuilding, this step falls back to uniform credit", flush=True)
        _close_pool()
    return out


def shuffled(deltas: list[float], rng) -> list[float]:
    """Same deltas, permuted within the rollout.

    The matched control for the position claim: identical amount of credit is
    moved, only WHERE it goes changes, so a difference between the two arms
    cannot be explained by how much reweighting happened.
    """
    d = list(deltas)
    rng.shuffle(d)
    return d
