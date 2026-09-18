"""Process-pool rollout collection for ALFWorld.

Rollouts must run concurrently or the GPU idles: measured here, one at a time
held vLLM at `Running: 1 reqs` and ~1% KV cache use, roughly 5% of the card,
because a rollout spends most of its wall clock waiting on its own previous turn.

Threads do not work. TextWorld keeps global state in two places -- the gym
registry and the PDDL/grammar parser -- and building or loading envs
concurrently corrupts both. Eight rollouts across eight threads crashed all
eight:

    IndexError: pop from empty list             (gym registry)
    TypeError: 'NoneType' object is not iterable
    FailedToken: (212:5) expecting 'template'   (grammar parser)

Serialising construction alone does not fix it, because the game is parsed on
`reset()`, not at registration; the FailedToken survived a build lock. Rather
than chase every shared structure with locks -- and leave an unattended run
resting on having found them all -- each rollout runs in its own process, where
the global state is private by construction.

The pool is created once and the policy is built in the initialiser: an OpenAI
client per process, not per rollout, so endpoint round-robin still spreads load
across samplers.
"""

from __future__ import annotations

import os
from dataclasses import asdict

_POLICY = None


def _init_worker(cfg_dict: dict, base_url: str) -> None:
    global _POLICY
    from ssc.env.alfworld_env import ALFWorldPolicy, ALFWorldPolicyConfig
    # Each worker takes the endpoint list in a rotated order, so workers do not
    # all send their first request to the same sampler.
    urls = [u.strip() for u in base_url.split(",") if u.strip()]
    shift = os.getpid() % max(len(urls), 1)
    rotated = ",".join(urls[shift:] + urls[:shift])
    _POLICY = ALFWorldPolicy(ALFWorldPolicyConfig(**cfg_dict), rotated)


def _one(payload):
    """Run one rollout. Returns (index, rollout_dict, error).

    Exceptions are returned rather than raised: a single bad game must not take
    down the pool, and an error that vanishes silently would let a short run
    pass as a complete one.
    """
    index, task_dict, rollout_index, seed = payload
    from ssc.env.alfworld_env import ALFTask
    try:
        roll = _POLICY.rollout(ALFTask(**task_dict), rollout_index=rollout_index,
                               seed=seed)
        return index, roll.to_dict(), None
    except Exception as e:  # noqa: BLE001
        return index, None, f"{type(e).__name__}: {e}"


def collect(tasks, group: int, cfg, base_url: str, workers: int,
            seed_base: int = 1000, on_result=None):
    """Yield (task, rollout_index, rollout_dict, error) as rollouts complete.

    `on_result` is called for its side effects (writing, counting) in the
    parent, so the caller never has to hold every rollout in memory at once.
    """
    import multiprocessing as mp

    jobs = [(i, asdict(t), k, seed_base + k)
            for i, (t, k) in enumerate((t, k) for t in tasks
                                       for k in range(group))]
    index_to_task = {i: (tasks[i // group], i % group) for i in range(len(jobs))}

    ctx = mp.get_context("spawn")
    # `maxtasksperchild` recycles workers: TextWorld leaks file handles across
    # many `register_games` calls, and an unattended run is the wrong place to
    # discover the limit.
    pool = ctx.Pool(processes=workers, initializer=_init_worker,
                    initargs=(cfg.to_dict(), base_url), maxtasksperchild=16)
    try:
        for index, roll, err in pool.imap_unordered(_one, jobs):
            task, k = index_to_task[index]
            if on_result is not None:
                on_result(task, k, roll, err)
            yield task, k, roll, err
    finally:
        pool.terminate()
        pool.join()
