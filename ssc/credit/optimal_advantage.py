"""Exact per-turn value, from the domain planner rather than from a proxy.

Leave-one-out ablation (`ssc.credit.counterfactual`) measures NECESSITY: would
the outcome have changed had this action not been taken. Measured on
ScienceWorld it answered "no" for 91.7% of turns, and that number is not a
property of the trajectories -- it is a property of the estimator. Leave-one-out
is the first-order term of a coalition decomposition, so it is blind by
construction to:

    redundancy   two actions that are alternatives: drop either and nothing
                 changes, so both read as dispensable while dropping both loses
                 the task;
    interaction  every higher-order term, which is where a plan's structure
                 lives.

Degenerating to "91.7% zero" is what forced the weighting in
`contribution_weights` to hedge with an alpha floor and a w_max cap: those knobs
compensate for an estimator that could not separate turns, not for anything
about the environment. The fix belongs in the measurement.

ALFWorld is a PDDL domain, and TextWorld exposes Fast Downward's plan from the
CURRENT state at every step. So the optimal distance-to-goal is directly
observable:

    d(s) = length of the optimal plan from s to the goal
         = None when no plan exists (the agent has made the task unsolvable)

and from it three per-turn quantities, which answer different questions and are
kept separate rather than blended:

    progress_t = d(s_t) - d(s_{t+1})     +1 advanced, 0 wasted, negative damaged
    a_star_t   = g^d(s_{t+1}) - g^d(s_t)  graded, the optimal-potential shaping
    v_star_t   = 1[d(s_t) <= B - t]       exact optimal success under the budget

`a_star` is potential-based shaping (Ng et al. 1999) with the OPTIMAL potential,
which is the strongest shaping that leaves the optimal policy unchanged. It is
also exactly what RewardFlow approximates: that method reconstructs g^d by
BFS over a state graph stitched together from sampled trajectories, where the
planner computes d outright.

## What this is NOT

`a_star` is the OPTIMAL advantage A*(s,a) = Q*(s,a) - V*(s). Policy-gradient
methods estimate the CURRENT-POLICY advantage A^pi(s,a) = Q^pi(s,a) - V^pi(s),
and the two differ whenever the policy is not optimal -- which is always. So an
audit that scores a return-based step advantage against `a_star` is not comparing two
estimates of one quantity, and reporting it as though it were would be wrong.

What `a_star` does answer is the causal question the credit-assignment
literature appeals to informally: did this turn move the task toward completion,
waste a step, or damage it. That is the claim these methods make about where
their credit lands, and it is checkable. Where the policy-advantage reading is
the one that matters, A^pi has to be sampled directly -- branch K continuations
from each s_t and average the returns -- which costs model calls and is
therefore run on a subsample, not on every trajectory. `sample_policy_advantage`
is deliberately absent here; it belongs with the rollout machinery.

## Cost

One replay of T actions with replanning at each step: ~0.13 s per step, so ~6.6 s
at T = 50. The leave-one-out sweep over the same trajectory costs 0.0129 * T^2
(~34 s at T = 50) even after batching. The stronger measurement is also the
cheaper one by 5x.
"""

from __future__ import annotations

from dataclasses import dataclass

# Unsolvable states get value 0, not a missing entry: an agent that destroys the
# task has produced the most informative turn in the trajectory, and dropping it
# would bias the measurement toward "nothing the agent did ever mattered".
UNSOLVABLE = None


@dataclass
class TrajectoryValue:
    """Per-state distances and the per-turn quantities derived from them.

    `d` has one more entry than the others: it is the distance at every state
    s_0 .. s_T, while the per-turn values are indexed by the action between two
    states.
    """

    d: list[int | None]              # d(s_0) .. d(s_T); None = unsolvable
    progress: list[int | None]       # d(s_t) - d(s_{t+1})
    a_star: list[float]              # g^d(s_{t+1}) - g^d(s_t)
    v_star: list[float]              # 1[d(s_t) <= budget - t], at s_0 .. s_T
    won: bool
    budget: int

    @property
    def num_turns(self) -> int:
        return len(self.a_star)

    @property
    def wasted(self) -> list[int]:
        """Turns that neither advanced nor damaged the plan."""
        return [t for t, p in enumerate(self.progress) if p == 0]

    @property
    def damaging(self) -> list[int]:
        """Turns after which the goal became strictly further away, including
        the turn that made it unreachable."""
        return [t for t, p in enumerate(self.progress) if p is None or p < 0]

    @property
    def lost_turn(self) -> int | None:
        """The turn at which the task stopped being winnable within budget.

        This is the exact answer to "which action lost the episode" and it is
        defined only for failures; a successful trajectory never crosses.
        """
        for t in range(len(self.v_star) - 1):
            if self.v_star[t] > 0 and self.v_star[t + 1] == 0:
                return t
        return None


def _distance(info, index: int = 0) -> int | None:
    """Optimal remaining steps at the current state.

    A won state has no plan left and must read 0, not `UNSOLVABLE`; conflating
    the two would score every successful ending as a destroyed task.
    """
    if info.get("won", [False] * (index + 1))[index]:
        return 0
    plan = info.get("policy_commands", [[]] * (index + 1))[index]
    return len(plan) if plan else UNSOLVABLE


def _discount(d: int | None, gamma: float) -> float:
    return 0.0 if d is None else gamma ** d


def trajectory_value(game_file: str, actions: list[str], *, budget: int = 50,
                     gamma: float = 0.95, env=None) -> TrajectoryValue:
    """Replay `actions` on `game_file`, recording the planner's d at every state.

    `env` may be supplied to reuse a pinned env across calls; it must have been
    built with `with_plan=True`, and it is reset here rather than by the caller
    so that a stale state cannot leak into the distances.
    """
    from ssc.env.alfworld_env import pinned_env

    own = env is None
    env = env or pinned_env(game_file, max_steps=budget, with_plan=True)
    _, info = env.reset()

    d = [_distance(info)]
    won = False
    for a in actions:
        _, _, done, info = env.step([a])
        d.append(_distance(info))
        won = bool(info.get("won", [False])[0])
        if done[0]:
            break

    progress: list[int | None] = []
    a_star: list[float] = []
    for t in range(len(d) - 1):
        before, after = d[t], d[t + 1]
        progress.append(None if after is None or before is None
                        else before - after)
        a_star.append(_discount(after, gamma) - _discount(before, gamma))

    # V* is budget-aware because the reward is: the environment pays 1 only if
    # the goal is reached within `budget` steps, so a state whose optimal plan
    # no longer fits is already lost even though the planner still returns one.
    v_star = [0.0 if dt is None else float(dt <= budget - t)
              for t, dt in enumerate(d)]

    if own:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
    return TrajectoryValue(d=d, progress=progress, a_star=a_star,
                           v_star=v_star, won=won, budget=budget)


def rollout_value(rollout, game_file: str, *, budget: int = 50,
                  gamma: float = 0.95) -> TrajectoryValue:
    """`trajectory_value` for a recorded `Rollout`."""
    actions = [str((t.tool_args or {}).get("action", "")) for t in rollout.turns]
    return trajectory_value(game_file, actions, budget=budget, gamma=gamma)
