"""Leave-one-out necessity on ALFWorld.

`ssc.credit.optimal_advantage` measures PROGRESS -- did this turn move the task
closer to done -- and it is the primary signal because it is graded and exact.
This module measures the complementary question, NECESSITY:

    delta_t = R(tau) - R(tau without action t)

The two come apart in one direction that matters. A turn with progress = 0 is
provably removable, so necessity adds nothing there. A turn with progress = 1
may still be unnecessary: if two routes were equally good, each step of the
taken route advances the plan while the trajectory would have succeeded without
it. Necessity is what separates "advanced the plan" from "was the only way", and
reporting only one of them would let a method look aligned with causality while
crediting substitutable work.

## Why this is a sequential sweep

It was a batched one. Every leave-one-out sequence drawn from a T-action
trajectory has exactly T-1 actions, so they advance in lockstep, and registering
T copies of one game as a single batch env costs ONE reset instead of T -- 0.0129
* T^2 against 0.2283 * T^2, an 18x reduction, since a reset costs ~1.6 s against
~0.08 s for a step.

It was also wrong, in the worst available way. Under `asynchronous=True` the
batch slots are separate processes and the sweep is correct. Under
`asynchronous=False` they are not: registering the SAME game file T times gives
slots that render distinct observations but share the underlying game, so every
slot reported `won=True` the moment any of them finished. The deltas came back
all-zero -- a perfectly plausible "no turn was necessary" that would have been
believed.

Asynchronous batching cannot be used where this is needed. The annotation pool
builds these envs inside multiprocessing workers, which are daemonic and cannot
have children; there the async batch env does not raise, it hangs.

So the sweep resets per ablation and pays 0.2283 * T^2 -- about 3.5 minutes at
T = 30. That is affordable because rollouts are annotated in parallel across the
pool, and correctness here is not negotiable: these deltas are the ground truth
everything else is scored against. `tests/test_alfworld_measurement.py` pins the
result against an independent replay.

## Known bias

Leave-one-out under-counts turns that are REDUNDANT with another turn: if two
actions are alternatives, removing either alone changes nothing, so both look
dispensable while removing both would not be. The direction is known and
one-sided. This is exactly the blind spot that made necessity alone an
inadequate measurement on ScienceWorld -- 91.7% of turns read as dispensable --
and it is why progress, not necessity, is the primary signal here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Necessity:
    base: float                  # outcome of the unablated trajectory
    deltas: list[float]          # base - outcome(tau \ t), one per turn

    @property
    def necessary(self) -> list[int]:
        return [t for t, d in enumerate(self.deltas) if d > 0]

    @property
    def dispensable_fraction(self) -> float:
        if not self.deltas:
            return 0.0
        return sum(1 for d in self.deltas if d <= 0) / len(self.deltas)


def _outcome(info, index: int) -> float:
    return 1.0 if info.get("won", [False] * (index + 1))[index] else 0.0


def leave_one_out(game_file: str, actions: list[str], *, budget: int = 50
                  ) -> Necessity:
    """Exact marginal contribution of every action, by replay.

    The base outcome is replayed here rather than taken from the recorded
    rollout: the recorded reward and a replay can only agree if the environment
    is deterministic and the game is pinned, and taking the recorded value on
    faith would hide a mismatch as a uniform shift in every delta.
    """
    from ssc.env.alfworld_env import pinned_env

    T = len(actions)
    if T == 0:
        return Necessity(0.0, [])

    # One env, reused across every replay. `reset()` on a pinned game replays it
    # rather than advancing to the next, which is the property the whole sweep
    # rests on and is checked in the tests.
    env = pinned_env(game_file, max_steps=budget)

    def replay(seq: list[str]) -> float:
        env.reset()
        outcome = 0.0
        for a in seq:
            _, _, done, info = env.step([a])
            # Read the outcome at EVERY step rather than only when `done` fires:
            # a trajectory that wins on its final action while the step limit
            # ends the episode on the same step is the common case, and keying
            # off `done` alone would make the result depend on which of the two
            # the environment reports first.
            outcome = _outcome(info, 0)
            if done[0]:
                break
        return outcome

    try:
        base = replay(actions)
        deltas = [base - replay(actions[:i] + actions[i + 1:]) for i in range(T)]
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    return Necessity(base, deltas)


def rollout_necessity(rollout, game_file: str, *, budget: int = 50) -> Necessity:
    """`leave_one_out` for a recorded `Rollout`."""
    actions = [str((t.tool_args or {}).get("action", "")) for t in rollout.turns]
    return leave_one_out(game_file, actions, budget=budget)
