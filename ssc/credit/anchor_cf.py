"""Anchor counterfactual credit: measured contribution at each (state, action).

The step-level slot in this family of updates is usually filled by a proxy --
the group's discounted return z-scored within an anchor state. Measured against
ground truth that proxy carries nothing: within-trajectory rank correlation with
causal progress is -0.072, below the +0.464 that turn index alone reaches. This
module keeps the same structure and replaces the proxy with a measurement.

    for each turn of each WINNING rollout, replay that rollout without that
    action and read the outcome. The difference IS the action's marginal
    contribution -- no model, no planner, no domain knowledge.

Fidelity and cost, both measured on real groups of 8 rollouts:

    signal                              rho vs true progress    cost / group
    leave-one-out on every turn                 +0.925              117 s
    anchor pooling, batched  (until 09-12)      +0.606              3.5 s
    turn position alone                         +0.464                  0
    return-based A_S, same anchors              -0.072                  0

## Why the (state, action) pooling was removed on 2026-09-12

Pooling shared one replay between every turn of the group that took the same
action from the same state text. That assumes an action's marginal contribution
is a property of (state, action). It is not: it depends on the rest of the
trajectory. A rollout that takes the mug three times makes each take
unnecessary (delta 0), and that 0 was handed to the ONE take of a clean 6-turn
rollout in the same group whose own deletion delta is +1 -- the most important
action of the cleanest trajectory came out at A = -2.45 while GRPO gave +1.
Measured on 12 groups x 8 of the hard task types (results/_probes/alf4b_dose_probe):

                                            Qwen3-4B      Qwen3-8B
    winner turns whose pooled delta != own      9.5%          3.2%
    necessary steps handed a pooled 0          17.9%          4.1%
    necessary steps pushed NEGATIVE, pooled    20.3%          6.0%
    necessary steps pushed negative, exact       0%            0%

The weaker policy has more redundant repeats, so more of its groups contain a
"representative" that makes an essential action look optional, and the damage
lands on exactly the clean successes the arm should be reinforcing.

The replacement is exact for every winner (its own leave-one-out sweep) and
shares a replay only between deletions that produce byte-identical action
sequences -- the same counterfactual, hence the same replay. Losers are not
swept: dropping one action from a failed run turns it into a win on ~3% of
turns, so their sweep would buy almost nothing, and they keep the free
off-path fill below. On the probe this costs 919 vs 868 replays on 4B and
653 vs 628 on 8B, against a measurement phase that is 3-5% of a training step.

Batching cuts another 9.5x, and that is where the time actually is -- a reset
costs ~1.6 s against ~0.08 s for a step, so paying one reset for the whole group
rather than one per ablation is most of the win.

Batching MUST use `asynchronous=True`. Under `asynchronous=False` the slots of
one repeated game file share the underlying game: every slot reports `won=True`
as soon as any of them finishes, and the deltas come back all-zero -- which
reads as "no turn was necessary" and is entirely plausible. That silent failure
is why `anchor_deltas` is checked against a sequential reference in the tests,
and why this runs in the parent process: multiprocessing pool workers are
daemonic and cannot spawn the children the async batch needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

STEP_RE = re.compile(r"\(step \d+/\d+\)")

# Filler for a slot that has finished or run out of actions. `look` is inert and
# always admissible; each slot's outcome is latched when it finishes, so the
# filler cannot change what was measured.
NOOP = "look"


def anchor_key(observation: str) -> str:
    """State key for pooling, matching the audit's `default` key.

    The step counter is stripped: leaving it in makes every state unique, which
    would collapse every anchor to a single member and turn the pooling back
    into full leave-one-out at full price.
    """
    o = STEP_RE.sub("", observation or "").strip().lower()
    return " ".join(o.split())


def _outcome(info, index: int) -> float:
    return 1.0 if info.get("won", [False] * (index + 1))[index] else 0.0


@dataclass
class AnchorCredit:
    deltas: list[list[float]]      # per rollout, per turn
    n_ablations: int               # replays actually run
    n_turns: int                   # turns they covered


def state_of(turns: list, t: int) -> str:
    """The observation the turn at index `t` was generated from.

    Turn t's own `observation` is the RESULT of turn t, so the state it acted
    from is turn t-1's observation -- and for the first turn, the opening
    observation the adapter records on it.
    """
    if t > 0:
        return turns[t - 1].get("observation", "")
    return str((turns[0].get("tool_args") or {}).get("opening_obs", ""))


def anchor_deltas(game_file: str, action_seqs: list[list[str]],
                  observations: list[list], budget: int = 30,
                  batched: bool = True, fill_offpath: bool = True,
                  measure_losers: bool = False
                  ) -> AnchorCredit:
    """Measured marginal contribution for every turn of a group.

    Every winning rollout gets an exact leave-one-out sweep on its own action
    sequence: delta[i][t] = base_i - outcome(actions_i without turn t). Two
    deletions share a replay only when they yield the same action sequence,
    which is the same counterfactual. Losing rollouts measure nothing here and
    are handed to `_fill_offpath`. `batched=False` runs the same replays one
    env-reset at a time so the fast path can be checked against it.

    The bases are replayed rather than read off the record: two rollouts of one
    game can end differently, and an ablation scored against the wrong base
    inverts its sign.
    """
    n_roll = len(action_seqs)
    if n_roll == 0:
        return AnchorCredit([], 0, 0)
    replay = _batched if batched else _sequential

    bases = _replay_chunked(replay, game_file, list(action_seqs), budget)
    winners = [i for i, b in enumerate(bases) if b > 0]

    # One slot per DISTINCT post-deletion sequence. Deleting either of two
    # consecutive identical actions, or the same turn of two identical
    # rollouts, produces the same sequence and therefore the same replay.
    slot: dict[tuple[str, ...], int] = {}
    seqs: list[list[str]] = []
    where: dict[tuple[int, int], int] = {}
    for i in winners:
        acts = action_seqs[i]
        for t in range(len(acts)):
            key = tuple(acts[:t] + acts[t + 1:])
            if key not in slot:
                slot[key] = len(seqs)
                seqs.append(list(key))
            where[(i, t)] = slot[key]
    scores = _replay_chunked(replay, game_file, seqs, budget) if seqs else []

    out = [[0.0] * len(acts) for acts in action_seqs]
    for (i, t), j in where.items():
        out[i][t] = bases[i] - scores[j]

    n_loser = 0
    if measure_losers:
        n_loser = _measure_losers(replay, game_file, action_seqs, observations,
                                  bases, out, budget)
    if fill_offpath:
        _fill_offpath(out, action_seqs, observations, bases)
    return AnchorCredit(out, n_roll + len(seqs) + n_loser,
                        sum(len(a) for a in action_seqs))


def _measure_losers(replay, game_file: str, action_seqs: list[list[str]],
                    observations: list[list], bases: list[float],
                    out: list[list[float]], budget: int) -> int:
    """Measured credit for the rollouts that FAILED: which turn broke it.

    Deletion is silent on a failure (0 - 0 on 97% of its turns), and the
    off-path fill below is a heuristic (rho +0.17 against planner progress).
    This asks the environment a different counterfactual question, one that
    is defined for failures:

        r_t = does the group's shortest winning plan still complete the task
              when run after the loser's first t actions?
        delta_t = r_{t+1} - r_t

    delta_t is the effect of taking a_t on whether the task is still
    completable by a plan that demonstrably worked: -1 on the turn that
    made it unrecoverable (put the object in the wrong place, threw it away),
    +1 on a turn that restored it, 0 on a turn that merely wasted time.
    Every prefix is replayed, no bisection, so the non-monotone cases
    (take the wrong object, then put it back) are measured rather than
    assumed away -- that assumption is what sank the bisected version
    (rho +0.514 falling to +0.755 -> +0.514 at 25% more replays).

    A loser whose every prefix stays recoverable measures all zeros, which
    is a real answer ("nothing it did was fatal, it just ran out of time")
    and is then handed to the off-path fill like before. If the winner's
    plan does not replay from scratch (r_0 = 0, the plan depends on
    something the loser's game instance lacks) nothing is measured.
    Prefixes shared between losers, and the empty prefix, replay once.

    Prefixes on which the loser is HOLDING something are not measured: a
    fixed plan cannot know what to do with a full hand, and both ways of
    forcing the question -- failing the plan's `take` (blames every pickup
    of a second instance), or putting the object down where the agent
    stands (flips with whether that spot accepts it) -- produced false
    verdicts on real Qwen3-4B groups. The verdict for a holding stretch
    lands on the turn that empties the hand: put the needed object where
    the plan cannot find it and that put is charged; carry the wrong thing
    around and nothing is, the episode advantage already pays for the time.
    """
    winners = [i for i, b in enumerate(bases) if b > 0]
    losers = [i for i, b in enumerate(bases) if b <= 0 and action_seqs[i]]
    if not winners or not losers:
        return 0
    plan = list(action_seqs[min(winners, key=lambda i: len(action_seqs[i]))])
    slot: dict[tuple[str, ...], int] = {}
    seqs: list[list[str]] = []
    where: dict[tuple[int, int], int] = {}
    helds = {}
    for i in losers:
        acts = action_seqs[i]
        held = _held_after(acts, observations[i])
        helds[i] = held
        for t in range(len(acts) + 1):
            if held[t]:
                continue                      # hand full: not measured here
            key = tuple(acts[:t] + plan)
            if key not in slot:
                slot[key] = len(seqs)
                seqs.append(list(key))
            where[(i, t)] = slot[key]
    # the spliced sequence may be up to `budget` + len(plan) actions long;
    # recoverability, not efficiency, is the question, so give it room.
    scores = _replay_chunked(replay, game_file, seqs, budget + len(plan) + 1)
    for i in losers:
        T = len(action_seqs[i])
        r = [(1.0 if scores[where[(i, t)]] > 0 else 0.0) if (i, t) in where else None
             for t in range(T + 1)]
        if r[0] is None or r[0] <= 0:
            continue
        last = r[0]
        for t in range(T):
            if r[t + 1] is None:
                continue                      # still holding: the verdict waits
            out[i][t] = r[t + 1] - last       # lands on the turn that emptied the hand
            last = r[t + 1]
    return len(seqs)


def _replay_chunked(replay, game_file: str, seqs: list[list[str]], budget: int,
                    chunk: int = 96) -> list[float]:
    """Replay in batches of at most `chunk` slots.

    The async batch env runs one child process per slot; a full leave-one-out
    sweep of eight 20-turn winners is ~160 slots, more than the pooled version
    ever asked for, and one oversized batch is where a machine that is already
    hosting samplers and trainers runs out of file descriptors.
    """
    out: list[float] = []
    for k in range(0, len(seqs), chunk):
        out.extend(replay(game_file, seqs[k:k + chunk], budget))
    return out


def _fill_offpath(out, action_seqs, observations, bases) -> None:
    """Give a rollout that failed and measured nothing SOMETHING, for free.

    Ablation is close to silent on failures: `R(tau) - R(tau minus a_t)` is
    0 - 0 unless dropping the action turns the run into a win, which happens on
    3% of their turns. A row of all zeros survives the group z-score as a single
    constant and is then wiped out entirely by within-trajectory centring, so
    those rollouts contribute no per-turn signal at all -- and they are the
    majority on exactly the hard batches where a step-level term has the most to
    contribute: on those batches it leads GRPO by +18.4 points, against +3.1 on
    the easy ones.

    What is available without touching the environment is the group's own
    successful rollouts. A state one of them passed through is a state from
    which the goal demonstrably WAS reachable, so for a failed rollout

        how much of what remains from turn t was spent on states a winner
        also visited

    grades its turns: high while it is still on a route that worked, falling
    once it has wandered off. Measured against planner progress on the failed
    rollouts of the nine mixed groups:

        turn position (the free prior)          rho -0.020
        on-path / off-path, +-1                 rho +0.154  (+0.127 with the
                                                 position component removed)
        fraction of remaining turns on-path     rho +0.172
        replay-verified recovery point          rho +0.514 overall, DOWN from
                                                 +0.755, at 25% more replays

    The last line is why this is the cheap version rather than the exact one:
    bisecting for the turn where a spliced winner-suffix stops succeeding puts a
    full-magnitude -1 on a single turn, and when the monotonicity it assumes
    does not hold that one wrong value costs more than the row gains.

    Only rows that measured NOTHING are filled. A failure that did produce a
    counterfactual keeps it: that is a real measurement and this is not.
    """
    winners = [i for i, b in enumerate(bases) if b > 0]
    if not winners:
        return
    on_path = set()
    for j in winners:
        for t in range(len(action_seqs[j])):
            on_path.add(anchor_key(state_of(observations[j], t)))
    for i, row in enumerate(out):
        if bases[i] > 0 or any(abs(v) > 1e-9 for v in row):
            continue
        L = len(row)
        on = [1.0 if anchor_key(state_of(observations[i], t)) in on_path
              else 0.0 for t in range(L)]
        for t in range(L):
            row[t] = sum(on[t:]) / max(L - t, 1)



def _replay_one(game_file: str, actions: list[str], budget: int) -> float:
    from ssc.env.alfworld_env import pinned_env
    env = pinned_env(game_file, max_steps=budget)
    try:
        env.reset()
        got = 0.0
        for a in actions:
            _, _, done, info = env.step([a])
            got = _outcome(info, 0)
            if done[0]:
                break
        return got
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


def _sequential(game_file: str, seqs: list[list[str]], budget: int) -> list[float]:
    return [_replay_one(game_file, s, budget) for s in seqs]


def _batched(game_file: str, seqs: list[list[str]], budget: int) -> list[float]:
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import (AlfredDemangler,
                                                           AlfredInfos)

    n = len(seqs)
    infos = textworld.EnvInfos(won=True, admissible_commands=True,
                               extras=["gamefile"])
    env_id = textworld.gym.register_games(
        [game_file] * n, infos, batch_size=n, asynchronous=True,
        max_episode_steps=budget,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
    env = textworld.gym.make(env_id)
    try:
        env.reset()
        finished = [False] * n
        got = [0.0] * n
        for step in range(max(len(s) for s in seqs)):
            cmds = [NOOP if (finished[b] or step >= len(seqs[b]))
                    else seqs[b][step] for b in range(n)]
            _, _, done, info = env.step(cmds)
            for b in range(n):
                if finished[b] or step >= len(seqs[b]):
                    continue
                got[b] = _outcome(info, b)
                if done[b]:
                    finished[b] = True
        return got
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


_OBJ = re.compile(r"^(take|put|move|clean|heat|cool|slice|use|examine)\s+(.+?)\s+(from|in/on|to|with)\s+(.+)$")


def _held_after(actions: list[str], turns: list) -> list:
    """What the agent holds after each prefix length 0..T, read from the
    observations (`You pick up the X from the Y.` / `You move the X to the Y.`).
    A `take` that the game answered with `Nothing happens` leaves the hand as
    it was."""
    held = [None]
    cur = None
    for t, a in enumerate(actions):
        ob = str((turns[t] or {}).get("observation", "") if t < len(turns) else "")
        m = _OBJ.match(a.strip())
        if m and m.group(1) == "take" and ob.startswith("You pick up"):
            cur = m.group(2)
        elif m and m.group(1) in ("put", "move") and (ob.startswith("You move")
                                                   or ob.startswith("You put")):
            cur = None
        held.append(cur)
    return held
