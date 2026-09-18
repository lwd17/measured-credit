"""The ALFWorld adapter and the two credit measurements built on it.

The integration tests here are slow -- a TextWorld reset costs ~1.7 s and the
planner replans at every step -- so they are few and each one covers a claim the
rest of the work rests on:

  * `reset()` REPLAYS a pinned game instead of advancing to the next one. Every
    counterfactual assumes this, and the failure mode is silent: two different
    houses both open with the same TextWorld banner, so an unpinned env
    certifies determinism that was never tested.
  * the batched leave-one-out sweep returns what the sequential sweep returns.
    Batching is an 18x optimisation, and an optimisation that changes the
    numbers is a bug that would be invisible against ground truth we cannot
    otherwise check.
  * a wasted action scores zero progress and a productive one scores positive,
    on a real game rather than a constructed one.

The pure-logic tests are fast and cover the boundaries where a wrong default
would quietly bias every downstream number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssc.credit.anchor_cf import anchor_deltas  # noqa: E402
from ssc.credit.counterfactual_alf import Necessity, leave_one_out  # noqa: E402
from ssc.credit.optimal_advantage import (  # noqa: E402
    TrajectoryValue, _discount, _distance, trajectory_value,
)
from ssc.env.alfworld_env import (  # noqa: E402
    ALFWorldPolicy, _task_type_of, load_alfworld, pinned_env,
    read_task_description,
)


# --------------------------------------------------------------------------
# Pure logic
# --------------------------------------------------------------------------
def test_won_state_reads_distance_zero_not_unsolvable():
    """A won game has no plan left, and so does a destroyed one.

    Both come back as an empty `policy_commands`. Reading the emptiness alone
    would file every success as an unsolvable state -- the exact opposite of its
    meaning -- so `won` has to be checked first.
    """
    assert _distance({"won": [True], "policy_commands": [[]]}) == 0
    assert _distance({"won": [False], "policy_commands": [[]]}) is None
    assert _distance({"won": [False], "policy_commands": [["a", "b"]]}) == 2


def test_distance_reads_the_requested_batch_slot():
    info = {"won": [False, True], "policy_commands": [["a"], []]}
    assert _distance(info, 0) == 1
    assert _distance(info, 1) == 0


def test_unsolvable_state_has_zero_value_not_a_missing_one():
    """gamma^infinity is 0. Dropping the entry instead would bias the measure
    toward 'nothing the agent did ever mattered', which is the bias this whole
    measurement exists to remove."""
    assert _discount(None, 0.95) == 0.0
    assert _discount(0, 0.95) == 1.0
    assert _discount(3, 0.5) == pytest.approx(0.125)


def test_derived_turn_categories():
    tv = TrajectoryValue(d=[3, 2, 2, 1, 0], progress=[1, 0, 1, 1],
                         a_star=[0.1, 0.0, 0.1, 0.1],
                         v_star=[1.0, 1.0, 1.0, 1.0, 1.0], won=True, budget=50)
    assert tv.num_turns == 4
    assert tv.wasted == [1]
    assert tv.damaging == []
    assert tv.lost_turn is None


def test_unsolvable_turn_counts_as_damaging():
    tv = TrajectoryValue(d=[3, 2, None], progress=[1, None], a_star=[0.1, -0.9],
                         v_star=[1.0, 1.0, 0.0], won=False, budget=50)
    assert tv.damaging == [1]
    assert tv.lost_turn == 1


def test_lost_turn_is_the_budget_crossing():
    """V* is budget-aware: a state whose optimal plan no longer fits in the
    remaining steps is already lost even though the planner still returns one."""
    tv = TrajectoryValue(d=[2, 2, 2], progress=[0, 0], a_star=[0.0, 0.0],
                         v_star=[1.0, 1.0, 0.0], won=False, budget=3)
    assert tv.lost_turn == 1


def test_necessity_summary():
    n = Necessity(base=1.0, deltas=[1.0, 0.0, 0.0, 1.0])
    assert n.necessary == [0, 3]
    assert n.dispensable_fraction == 0.5
    assert Necessity(0.0, []).dispensable_fraction == 0.0


def test_extract_action_prefers_the_last_action_line():
    text = "Action: go to desk 1\nOn reflection:\nAction: take mug 1"
    action, ok = ALFWorldPolicy.extract_action(text)
    assert (action, ok) == ("take mug 1", True)


def test_extract_action_reads_past_reasoning():
    text = "<think>Action: go to bed 1 would be wrong</think>\nAction: go to desk 1"
    action, ok = ALFWorldPolicy.extract_action(text)
    assert (action, ok) == ("go to desk 1", True)


def test_missing_action_line_is_a_format_miss_not_a_command():
    """A reply with no Action line must never be handed to the environment as
    prose. Recovering a legal action from the text is allowed, but the miss
    stays visible -- on ScienceWorld the un-flagged version turned 64.5% of
    steps into 'the policy failed' when the adapter had failed to parse."""
    admissible = ["go to desk 1", "take mug 1 from desk 1"]
    text = "I think I should take mug 1 from desk 1 next."
    action, ok = ALFWorldPolicy.extract_action(text, admissible)
    assert action == "take mug 1 from desk 1"
    assert ok is False


def test_format_miss_without_admissible_list_falls_back_to_last_line():
    action, ok = ALFWorldPolicy.extract_action("some prose\nfinal line")
    assert (action, ok) == ("final line", False)


def test_task_type_and_stride_sampling():
    """`[:n]` would draw every task from whichever type sorts first, because the
    game paths begin with the task type."""
    assert _task_type_of("/d/pick_two_obj_and_place-X/t/game.tw-pddl") == \
        "pick_two_obj_and_place"
    assert _task_type_of("/d/nonsense/game.tw-pddl") == "unknown"

    head = load_alfworld(6, stride=1)
    spread = load_alfworld(6, stride=397)
    assert len({t.task_type for t in head}) == 1
    assert len({t.task_type for t in spread}) > 1


def test_read_task_description():
    obs = "-= Welcome =-\nYou are in a room.\nYour task is to: put a mug in desk.\n"
    assert read_task_description(obs) == "put a mug in desk."


# --------------------------------------------------------------------------
# Integration -- these touch the real environment
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def game():
    """One short game, so the slow tests stay under a few seconds each."""
    tasks = load_alfworld(4000, stride=1)
    return next(t for t in tasks if t.task_type == "pick_and_place_simple")


@pytest.fixture(scope="module")
def optimal_plan(game):
    env = pinned_env(game.game_file, max_steps=50, with_plan=True)
    _, info = env.reset()
    plan = list(info["policy_commands"][0])
    env.close()
    return plan


def test_pinned_env_replays_rather_than_advances(game):
    """`reset()` walks to the next game unless the env holds exactly one."""
    env = pinned_env(game.game_file, max_steps=50)
    _, first = env.reset()
    _, second = env.reset()
    env.close()
    assert first["extra.gamefile"][0] == second["extra.gamefile"][0]


def test_optimal_plan_solves_the_game_and_distance_falls_by_one(game, optimal_plan):
    tv = trajectory_value(game.game_file, optimal_plan, budget=50)
    assert tv.won
    assert tv.d[0] == len(optimal_plan)
    assert tv.d[-1] == 0
    assert tv.progress == [1] * len(optimal_plan)


def test_a_wasted_action_scores_zero_progress(game, optimal_plan):
    """The claim the whole measurement rests on, checked on a real game: an
    action that does not advance the plan is scored zero rather than being
    smeared with uniform credit."""
    actions = optimal_plan[:1] + ["look"] + optimal_plan[1:]
    tv = trajectory_value(game.game_file, actions, budget=50)
    assert tv.won
    assert tv.progress[1] == 0
    assert tv.a_star[1] == 0.0
    assert tv.wasted == [1]
    assert all(tv.progress[i] == 1 for i in range(len(actions)) if i != 1)


def test_batched_leave_one_out_equals_the_sequential_sweep(game, optimal_plan):
    """Batching is an 18x optimisation; if it changed the deltas it would be a
    silent bug, since there is no other source of truth to check them against."""
    actions = optimal_plan[:1] + ["look"] + optimal_plan[1:]
    batched = leave_one_out(game.game_file, actions, budget=50)

    def sequential(acts):
        env = pinned_env(game.game_file, max_steps=50)
        env.reset()
        won = 0.0
        for a in acts:
            _, _, done, info = env.step([a])
            won = 1.0 if info["won"][0] else 0.0
            if done[0]:
                break
        env.close()
        return won

    base = sequential(actions)
    expected = [base - sequential(actions[:i] + actions[i + 1:])
                for i in range(len(actions))]
    assert batched.base == base
    assert batched.deltas == expected


def _dummy_turns(actions):
    """`anchor_deltas` reads observations only for the off-path fill of losers;
    with `fill_offpath=False` and all-winning groups they are never consulted."""
    return [{"observation": "", "tool_args": {"opening_obs": ""}} for _ in actions]


def test_anchor_deltas_is_the_exact_sweep_for_every_winner(game, optimal_plan):
    """The group measurement must equal each winner's own leave-one-out sweep.
    Until 2026-09-12 it pooled replays by (state, action) across rollouts, and
    a redundant twin in the group could hand a clean rollout a 0 on its one
    essential action (results/_probes/alf4b_dose_probe: 17.9% of necessary
    steps on Qwen3-4B)."""
    clean = list(optimal_plan)
    wasted = optimal_plan[:1] + ["look"] + optimal_plan[1:]
    cred = anchor_deltas(game.game_file, [clean, wasted],
                         [_dummy_turns(clean), _dummy_turns(wasted)],
                         budget=50, fill_offpath=False)
    for acts, row in zip((clean, wasted), cred.deltas):
        ref = leave_one_out(game.game_file, acts, budget=50)
        assert ref.base == 1.0
        assert row == ref.deltas, (acts, row, ref.deltas)


def test_redundant_twin_does_not_contaminate_the_clean_rollout(game, optimal_plan):
    """The bug the rewrite removes, on a real game: a twin that repeats an
    essential action makes that action optional FOR THE TWIN only."""
    k = next(i for i, a in enumerate(optimal_plan) if a.startswith("take"))
    clean = list(optimal_plan)
    twin = optimal_plan[:k + 1] + [optimal_plan[k]] + optimal_plan[k + 1:]
    cred = anchor_deltas(game.game_file, [twin, clean],
                         [_dummy_turns(twin), _dummy_turns(clean)],
                         budget=50, fill_offpath=False)
    assert cred.deltas[1][k] == 1.0            # the clean rollout's take is necessary
    assert cred.deltas[0][k] == 0.0            # either copy in the twin is optional
    assert cred.deltas[0][k + 1] == 0.0
    # identical counterfactual sequences share one replay: deleting either copy
    # of the repeated take yields the same sequence, so the sweep of the twin
    # costs len(twin) - 1 replays, plus the clean sweep, plus two bases.
    assert cred.n_ablations == 2 + len(clean) + (len(twin) - 1)


def test_necessity_and_progress_agree_on_the_wasted_turn(game, optimal_plan):
    """The two measurements answer different questions, but a turn that makes no
    progress is provably removable, so they must agree in that direction."""
    actions = optimal_plan[:1] + ["look"] + optimal_plan[1:]
    tv = trajectory_value(game.game_file, actions, budget=50)
    nec = leave_one_out(game.game_file, actions, budget=50)
    assert tv.progress[1] == 0
    assert nec.deltas[1] == 0.0
    for t, p in enumerate(tv.progress):
        if p == 0:
            assert nec.deltas[t] == 0.0, f"turn {t} made no progress but reads necessary"


def test_spread_covers_every_task_type_at_any_n():
    """`spread=True` must span the whole filtered pool, for every n.

    The stride form -- `files[::len(files)//n][:n]` -- did not. Over the 92
    unseen games of the four hard types it gave stride 3; `files[::3]` yields 31
    picks and `[:24]` keeps the first 24, reaching only files[69], so the final
    22 -- the entire `pick_two_obj_and_place` block, which sorts last -- were
    unreachable. The eval pool came back 11/7/6/0 while training drew 126 of the
    type that was never scored. Asking for more was worse: at n=48 the derived
    stride is 1, the `stride > 1` branch drops out, and the result is the bare
    prefix, 0 of two types.
    """
    from ssc.env.alfworld_env import HARD_TYPES, load_alfworld

    full = load_alfworld(10_000, split="valid_unseen", spread=True,
                         task_types=HARD_TYPES)
    present = {t.task_type for t in full}
    assert present == set(HARD_TYPES), present

    for n in (12, 24, 32, 48, len(full), len(full) + 40):
        got = load_alfworld(n, split="valid_unseen", spread=True,
                            task_types=HARD_TYPES)
        assert len(got) == min(n, len(full)), (n, len(got))
        files = [t.game_file for t in got]
        assert len(set(files)) == len(files), f"n={n} repeated a game"
        assert not {t.task_type for t in got} - set(HARD_TYPES)
        # every type that fits in a sample of this size must appear
        if n >= 4 * len(HARD_TYPES):
            assert {t.task_type for t in got} == set(HARD_TYPES), (
                n, sorted({t.task_type for t in got}))


def test_train_and_eval_pools_stay_disjoint_under_filtering():
    """A gain must not be memorisation; the filter must not break the split."""
    from ssc.env.alfworld_env import HARD_TYPES, load_alfworld

    tr = load_alfworld(400, split="train", spread=True, task_types=HARD_TYPES)
    ev = load_alfworld(92, split="valid_unseen", spread=True,
                       task_types=HARD_TYPES)
    assert len(tr) == 400 and len(ev) == 92
    assert not {t.game_file for t in tr} & {t.game_file for t in ev}


def test_losers_are_not_swept_and_only_winners_pay_for_replays(game, optimal_plan):
    """A losing rollout gets the free off-path fill, not a sweep: its replays
    would be ~0 - 0 on 97% of turns. Replay count = bases + unique deletions."""
    from ssc.credit.anchor_cf import anchor_deltas
    obs = lambda acts: [{"observation": "", "tool_args": {}} for _ in acts]
    clean = list(optimal_plan)
    loser = ["look"] * 3                      # never reaches the goal
    cred = anchor_deltas(game.game_file, [clean, loser], [obs(clean), obs(loser)],
                         budget=50, fill_offpath=False)
    assert cred.deltas[1] == [0.0, 0.0, 0.0]
    assert cred.n_ablations == 2 + len(clean)


# --------------------------------------------------------------------------
# 09-14 Measuring failed trajectories: which step ruined the task
# --------------------------------------------------------------------------
def _admissible_after(game, actions):
    env = pinned_env(game.game_file, max_steps=60)
    _, info = env.reset()
    for a in actions:
        _, _, _, info = env.step([a])
    cmds = list(info["admissible_commands"][0])
    env.close()
    return cmds


def _wrong_receptacle_loser(game, optimal_plan):
    """optimal plan = go to X, take obj, go to Y, put obj in Y. The loser takes the
    object, carries it to some other receptacle (opening it if needed), puts it
    down there, then idles. Returns the action list and the index of the put."""
    k = next(i for i, a in enumerate(optimal_plan) if a.startswith("take"))
    put = next(a for a in optimal_plan if a.startswith("put") or a.startswith("move"))
    target = put.split(" in/on ")[-1] if " in/on " in put else put.split(" to ")[-1]
    prefix = optimal_plan[:k + 1]
    for goto in [c for c in _admissible_after(game, prefix)
                 if c.startswith("go to") and target not in c]:
        detour = prefix + [goto]
        cmds = _admissible_after(game, detour)
        opener = [c for c in cmds if c.startswith("open")]
        if opener:
            detour = detour + [opener[0]]
            cmds = _admissible_after(game, detour)
        puts = [c for c in cmds
                if (c.startswith("put") or c.startswith("move")) and target not in c
                and not c.endswith(prefix[0][len("go to "):])]     # not back where it came from
        if puts:
            return detour + [puts[0], "look", "look"], len(detour)
    raise AssertionError("no receptacle to misplace the object in")


def test_measured_loser_credit_blames_the_turn_that_made_it_unrecoverable(game, optimal_plan):
    loser, fatal = _wrong_receptacle_loser(game, optimal_plan)
    clean = list(optimal_plan)
    # observations say what the hand holds: a take fills it, a put empties it
    def turns(acts):
        return [{"observation": ("You pick up" if a.startswith("take") else
                                 "You move" if a.startswith(("put", "move")) else "You arrive"),
                 "tool_args": {"action": a}} for a in acts]
    cred = anchor_deltas(game.game_file, [clean, loser], [turns(clean), turns(loser)],
                         budget=50, fill_offpath=False, measure_losers=True)
    row = cred.deltas[1]
    assert row[fatal] == -1.0, row
    assert all(v == 0.0 for i, v in enumerate(row) if i != fatal), row
    # the winner's own sweep is untouched
    assert cred.deltas[0] == leave_one_out(game.game_file, clean, budget=50).deltas
    # one replay per distinct loser prefix + plan, plus one put-down variant per
    # prefix on which the hand is full, on top of the winner sweep and bases
    k = next(i for i, a in enumerate(loser) if a.startswith("take"))
    held_prefixes = fatal - k          # prefixes k+1..fatal end with the object in hand: not replayed
    assert cred.n_ablations == 2 + len(clean) + (len(loser) + 1) - held_prefixes


def test_measured_loser_credit_is_off_by_default_and_harmless_losers_keep_the_fill(game, optimal_plan):
    loser, fatal = _wrong_receptacle_loser(game, optimal_plan)
    idle = optimal_plan[:1] + ["look", "look", "look"]          # wastes time, breaks nothing
    clean = list(optimal_plan)
    obs = [_dummy_turns(clean), _dummy_turns(loser), _dummy_turns(idle)]
    before = anchor_deltas(game.game_file, [clean, loser, idle], obs, budget=50)
    after = anchor_deltas(game.game_file, [clean, loser, idle], obs, budget=50,
                          measure_losers=True)
    assert after.deltas[0] == before.deltas[0]
    # the harmless loser measures all zeros and therefore gets the same fill as before
    assert after.deltas[2] == before.deltas[2]
    # the fatal loser is now measured instead of filled
    assert after.deltas[1][fatal] == -1.0
    assert after.deltas[1] != before.deltas[1]


def test_held_after_reads_the_hand_from_observations():
    from ssc.credit.anchor_cf import _held_after
    acts = ["go to cabinet 2", "take cup 2 from cabinet 2", "take cup 1 from cabinet 2", "move cup 2 to sinkbasin 1"]
    obs = [{"observation": "You arrive at cabinet 2."}, {"observation": "You pick up the cup 2 from the cabinet 2."},
           {"observation": "Nothing happens."}, {"observation": "You move the cup 2 to the sinkbasin 1."}]
    assert _held_after(acts, obs) == [None, None, "cup 2", "cup 2", None]


def test_picking_up_the_wrong_object_is_not_fatal_but_misplacing_the_right_one_is(game, optimal_plan):
    """Holding something else only costs the put-down; the measurement must
    charge the turn that moved the needed object out of the plan's reach."""
    from ssc.env.alfworld_env import pinned_env as _pe
    # a loser that picks up some OTHER object first, then idles
    env = _pe(game.game_file, max_steps=60); _, info = env.reset()
    _, _, _, info = env.step([optimal_plan[0]])
    other = [c for c in info["admissible_commands"][0]
             if c.startswith("take") and c != optimal_plan[1]]
    env.close()
    if not other:
        pytest.skip("only one object at the first receptacle")
    wrong = [optimal_plan[0], other[0], "look", "look"]
    clean = list(optimal_plan)
    turns = lambda acts: [{"observation": ("You pick up" if a.startswith("take") else
                                           "You move" if a.startswith(("put", "move")) else "You arrive"),
                           "tool_args": {"action": a}} for a in acts]
    cred = anchor_deltas(game.game_file, [clean, wrong], [turns(clean), turns(wrong)],
                         budget=50, fill_offpath=False, measure_losers=True)
    assert all(v == 0.0 for v in cred.deltas[1]), cred.deltas[1]
    # the wrong-receptacle loser from the test above is still charged at the put
    loser, fatal = _wrong_receptacle_loser(game, optimal_plan)
    cred = anchor_deltas(game.game_file, [clean, loser], [turns(clean), turns(loser)],
                         budget=50, fill_offpath=False, measure_losers=True)
    assert cred.deltas[1][fatal] == -1.0 and all(v == 0.0 for i, v in enumerate(cred.deltas[1]) if i != fatal), cred.deltas[1]
