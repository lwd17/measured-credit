"""Process-pool workers for ground-truth annotation.

Lives here rather than in the annotation script because the trainer needs the
same workers: a planner-annotated arm labels every step's rollouts inline, and
`scripts/` is not an importable package, so reaching into it would break the
moment the trainer runs from a different working directory.

Processes, not threads. TextWorld keeps global state in the gym registry and in
the PDDL/grammar parser, and building or loading envs concurrently in threads
corrupts both -- eight rollouts across eight threads crashed all eight. Inside
these workers the global state is private by construction.
"""

from __future__ import annotations

import os
import time

_GAMMA = 0.95
_BUDGET = 30
_WITH_NECESSITY = True


def init_worker(gamma: float, budget: int, with_necessity: bool) -> None:
    global _GAMMA, _BUDGET, _WITH_NECESSITY
    _GAMMA, _BUDGET, _WITH_NECESSITY = gamma, budget, with_necessity
    os.environ.setdefault("ALFWORLD_DATA",
                          os.path.expanduser("~/.cache/alfworld"))


def annotate_one(payload):
    """Annotate one trajectory. Returns (index, annotation, error).

    Errors are returned, never raised: one game that fails to replay must not
    take the pool down, and a silently dropped trajectory would make a partial
    annotation look complete.
    """
    index, game_file, actions = payload
    from ssc.credit.counterfactual_alf import leave_one_out
    from ssc.credit.optimal_advantage import trajectory_value
    try:
        tv = trajectory_value(game_file, actions, budget=_BUDGET, gamma=_GAMMA)
        out = {"d": tv.d, "progress": tv.progress, "a_star": tv.a_star,
               "v_star": tv.v_star, "replay_won": tv.won,
               "wasted": tv.wasted, "damaging": tv.damaging,
               "lost_turn": tv.lost_turn}
        if _WITH_NECESSITY:
            nec = leave_one_out(game_file, actions, budget=_BUDGET)
            out["necessity"] = nec.deltas
            out["necessity_base"] = nec.base
        return index, out, None
    except Exception as e:  # noqa: BLE001
        return index, None, f"{type(e).__name__}: {e}"


def annotate_batch(items, gamma: float, budget: int, workers: int,
                   with_necessity: bool = False, timeout: float = 90.0):
    """`a_star` (and optionally necessity) for a list of (game_file, actions).

    Returns a list aligned with `items`; an entry is None where the replay
    failed or ran past `timeout`. The caller decides what to do with a gap --
    the trainer skips that group rather than training it on zeros, which would
    read as a trajectory in which no turn contributed.

    ## Why there is a deadline

    Annotation normally costs ~3 s a trajectory. On a training run it stopped:
    one worker sat at 100% CPU for 15 minutes on a single trajectory while the
    other 23 finished in seconds, and the whole arm blocked behind it. Fast
    Downward's search is not bounded -- proving a goal unreachable means
    exhausting the space -- so a single awkward state can hold a run
    indefinitely.

    The deadline is enforced by killing the POOL, not by an alarm inside the
    worker. The planner runs in a C library through ctypes, and a Python signal
    handler only runs between bytecodes, so `signal.alarm` would not fire until
    the C call returned -- which is the thing that is not returning.
    """
    import multiprocessing as mp

    payload = [(i, gf, acts) for i, (gf, acts) in enumerate(items)]
    out = [None] * len(payload)
    if not payload:
        return out

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=max(1, min(workers, len(payload))),
                    initializer=init_worker,
                    initargs=(gamma, budget, with_necessity),
                    maxtasksperchild=16)
    n_done = 0
    try:
        pending = [(i, pool.apply_async(annotate_one, (item,)))
                   for i, item in enumerate(payload)]
        deadline = time.monotonic() + timeout
        # Poll for whatever is READY rather than blocking on each result in
        # submission order. Blocking in order makes one slow trajectory hold up
        # every result behind it, and when the deadline then fires those are
        # discarded even though they finished long ago: measured on a training
        # step, a single hard planning instance at position 25 cost the six
        # completed results that followed it.
        while pending and time.monotonic() < deadline:
            still = []
            for i, ar in pending:
                if not ar.ready():
                    still.append((i, ar))
                    continue
                try:
                    index, ann, err = ar.get(timeout=0)
                except Exception:  # noqa: BLE001
                    continue
                if err is None:
                    out[index] = ann
                n_done += 1
            pending = still
            if pending:
                time.sleep(0.2)
    finally:
        # terminate(), not close(): a worker stuck inside the planner will not
        # notice a graceful shutdown, and join() would inherit the hang the
        # deadline exists to escape.
        pool.terminate()
        pool.join()

    missing = sum(1 for v in out if v is None)
    if missing:
        print(f"[annotate] {missing}/{len(payload)} trajectories unannotated "
              f"after {timeout:.0f}s; their groups will be skipped", flush=True)
    return out
