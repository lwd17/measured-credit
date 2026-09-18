"""The three ALFWorld arms differ in exactly one place.

Section 27's fairness rule, applied here: the arms share a combination rule
(A_E + omega * A_step) and a normalisation, so any difference between their
training curves is attributable to the CONTENT of the step-level term. These
tests pin that property, because it is the one an experiment cannot recover from
losing -- a difference in scale or in the combination rule would look exactly
like a difference in signal quality.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssc.credit.alf_arms import (  # noqa: E402
    anchor_step_advantage, group_advantage,
    turn_advantages, zscore,
)


def test_grpo_arm_is_the_episode_advantage_repeated():
    """GRPO gives every turn of a rollout the same number; if this drifted, the
    baseline would silently become some other method."""
    adv = turn_advantages("grpo", rewards=[1.0, 0.0], lengths=[3, 2])
    assert adv[0] == [adv[0][0]] * 3
    assert adv[1] == [adv[1][0]] * 2
    assert adv[0][0] > 0 > adv[1][0]


def test_group_advantage_is_zero_mean():
    a = group_advantage([1.0, 0.0, 1.0, 0.0])
    assert abs(sum(a)) < 1e-6


def test_group_advantage_of_a_tied_group_is_zero_not_undefined():
    """Every rollout winning is the common case once a policy is competent, and
    a 0/0 there would put NaN into the gradient rather than no update."""
    a = group_advantage([1.0, 1.0, 1.0])
    assert all(abs(x) < 1e-3 for x in a)
    assert not any(math.isnan(x) for x in a)


def test_anchor_group_of_one_gets_no_step_credit():
    """The standard anchor convention: a state only one rollout visited has
    nothing to compare against."""
    step = anchor_step_advantage([["a", "b"], ["c", "d"]],
                                 [[1.0, 1.0], [0.0, 0.0]])
    assert step == [[0.0, 0.0], [0.0, 0.0]]


def test_anchor_group_with_no_outcome_variance_gets_no_step_credit():
    """Membership is not engagement. This is why a G=2 probe run had 94% anchor
    coverage and an A_S that was identically zero on every turn."""
    step = anchor_step_advantage([["s"], ["s"]], [[1.0], [1.0]])
    assert step == [[0.0], [0.0]]


def test_anchor_group_separates_the_rollout_that_went_on_to_win():
    step = anchor_step_advantage([["s", "x"], ["s", "y"]],
                                 [[1.0, 1.0], [0.0, 0.0]])
    assert step[0][0] > 0 > step[1][0]
    assert step[0][1] == 0.0        # "x" and "y" are singleton anchors










def test_unknown_arm_is_rejected():
    for bad in ("ssc", "", "GRPO"):
        try:
            turn_advantages(bad, [1.0], [1])
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not be accepted as an arm")


def test_missing_inputs_are_rejected_rather_than_silently_zeroed():
    """An arm without its measured signal would train as plain GRPO while
    logging under its own name, which is the failure mode that silently
    invalidates a whole run."""
    for arm, kw in (("anchor_cf", {}), ("anchor_mass", {})):
        try:
            turn_advantages(arm, [1.0, 0.0], [2, 2], **kw)
        except ValueError:
            continue
        raise AssertionError(f"{arm} should refuse to run without its signal")


# --------------------------------------------------------------------------
# Train/inference context match
# --------------------------------------------------------------------------
from ssc.detector.rollout import Outcome, Rollout, Turn  # noqa: E402
from ssc.train.batch import _history  # noqa: E402


def _alf_rollout():
    """A rollout shaped the way the ALFWorld adapter records one."""
    return Rollout(
        task_id="alf_x", question="put a mug in desk",
        turns=[
            Turn(index=0, text="Action: go to desk 1", tool_name="alf_action",
                 tool_args={"action": "go to desk 1",
                            "admissible": ["go to desk 1", "go to drawer 1"],
                            "step_shown": 0, "step_budget": 30,
                            "opening_obs": "You are in the middle of a room."},
                 observation="You arrive at desk 1."),
            Turn(index=1, text="Action: take mug 1 from desk 1",
                 tool_name="alf_action",
                 tool_args={"action": "take mug 1 from desk 1",
                            "admissible": ["take mug 1 from desk 1", "look"],
                            "step_shown": 1, "step_budget": 30},
                 observation="You pick up the mug 1."),
        ],
        outcome=Outcome(success=1, reward=1.0))


def test_training_context_contains_the_admissible_list_the_policy_saw():
    """The list is the bulk of every ALFWorld prompt. Rebuilding the context
    without it would train the policy against a prompt that never existed, on
    every turn -- and nothing downstream would report an error."""
    msgs = _history(_alf_rollout(), upto=1, system_prompt="SYS",
                    env_kind="alfworld")
    joined = "\n".join(m["content"] for m in msgs)
    assert "Admissible actions:" in joined
    assert "go to desk 1" in joined
    assert "(step 0/30)" in joined


def test_training_context_uses_the_state_each_turn_was_generated_from():
    """Turn 1's prompt must carry turn 0's observation and turn 1's own
    admissible list -- not turn 1's observation, which is its own result."""
    msgs = _history(_alf_rollout(), upto=2, system_prompt="SYS",
                    env_kind="alfworld")
    last_user = [m for m in msgs if m["role"] == "user"][-1]["content"]
    assert "You arrive at desk 1." in last_user
    assert "take mug 1 from desk 1" in last_user
    assert "You pick up the mug 1." not in last_user


def test_opening_observation_survives_into_the_first_prompt():
    msgs = _history(_alf_rollout(), upto=0, system_prompt="SYS",
                    env_kind="alfworld")
    assert "You are in the middle of a room." in msgs[1]["content"]
    assert "put a mug in desk" in msgs[1]["content"]


def test_reasoning_is_never_replayed_into_the_context():
    """Only `content` was appended at rollout time; reconstructing the thinking
    would train on a history the policy never conditioned on."""
    r = _alf_rollout()
    r.turns[0].text = "<think>long deliberation</think>\nAction: go to desk 1"
    msgs = _history(r, upto=2, system_prompt="SYS", env_kind="alfworld")
    joined = "\n".join(m["content"] for m in msgs)
    assert "long deliberation" not in joined


# --------------------------------------------------------------------------
# Adapter sync URL handling
# --------------------------------------------------------------------------
from ssc.train.sync import LoRASync  # noqa: E402


def test_sync_normalises_the_openai_style_base_url():
    """The policy's base_url ends in /v1; sync appends its own /v1/... path.

    Keeping both produced /v1/v1/load_lora_adapter, which 404s from a route the
    server does advertise -- so the failure looked like "runtime LoRA updating
    is off" rather than "the URL is wrong".
    """
    assert LoRASync("http://localhost:8100/v1").urls == ["http://localhost:8100"]
    assert LoRASync("http://localhost:8100").urls == ["http://localhost:8100"]
    assert LoRASync("http://a:1/v1, http://b:2/v1").urls == \
        ["http://a:1", "http://b:2"]


# --------------------------------------------------------------------------
# The position arm
# --------------------------------------------------------------------------


import time as _time  # noqa: E402

from ssc.credit.annotate_pool import annotate_batch  # noqa: E402


def test_annotation_returns_empty_for_no_items():
    assert annotate_batch([], gamma=0.95, budget=25, workers=4) == []


def test_annotation_deadline_returns_rather_than_hanging():
    """A single trajectory can hold the planner indefinitely.

    Measured on a training run: one worker sat at 100% CPU for 15 minutes on one
    trajectory while the other 23 finished in seconds, and the arm blocked
    behind it. Fast Downward's search is unbounded -- proving a goal unreachable
    means exhausting the space -- so the pool needs a deadline it can enforce by
    killing workers. An alarm inside the worker would not do: the planner runs
    in a C library through ctypes, and a Python signal handler only runs between
    bytecodes.

    A bogus game file makes every worker fail fast, so this asserts the shape of
    the contract -- one aligned slot per item, no exception, prompt return --
    rather than the timeout path itself, which would cost minutes to provoke.
    """
    items = [("/nonexistent/game.tw-pddl", ["look"]) for _ in range(3)]
    t0 = _time.time()
    out = annotate_batch(items, gamma=0.95, budget=5, workers=2, timeout=45.0)
    assert len(out) == len(items)
    assert all(v is None for v in out)
    assert _time.time() - t0 < 60, "the deadline did not bound the call"




# --------------------------------------------------------------------------
# Gradient accumulation weighting
# --------------------------------------------------------------------------
def test_token_weighted_chunks_reproduce_the_whole_step_token_mean():
    """Accumulating chunk losses must reconstruct the token-mean over the step.

    Each chunk's loss is a mean over ITS OWN actor tokens, so summing
    `loss_c * (tokens_c / tokens_total)` gives the step's token-mean. Dividing
    by the chunk COUNT instead weights every chunk alike -- and chunks built to
    a fixed budget of INPUT tokens hold very different numbers of ACTOR tokens
    (231 to 3497 in a measured step, a 15x spread in per-token weight).

    The distortion is not random: chunks are length-sorted, a turn's context
    grows with its index while its generation does not, so the over-weighted
    chunks are the late-turn ones. That applies a turn-position prior to every
    arm, and turn position alone tracks causal progress at rho = +0.464 -- a
    baseline quietly receiving the signal under test.
    """
    # Three chunks: per-token losses 1.0, 2.0, 3.0 over 100, 400, 500 tokens.
    per_token = [1.0, 2.0, 3.0]
    tokens = [100, 400, 500]
    total = sum(tokens)

    true_mean = sum(p * t for p, t in zip(per_token, tokens)) / total
    token_weighted = sum(p * (t / total) for p, t in zip(per_token, tokens))
    chunk_count = sum(p / len(per_token) for p in per_token)

    assert abs(token_weighted - true_mean) < 1e-12
    assert abs(chunk_count - true_mean) > 0.1, \
        "the count-weighted form must actually differ, or the test proves nothing"


# --------------------------------------------------------------------------
# Anchor counterfactual credit
# --------------------------------------------------------------------------
from ssc.credit.anchor_cf import anchor_key, state_of  # noqa: E402


def test_anchor_key_strips_the_step_counter():
    """Leaving the counter in makes every state unique, which collapses every
    anchor to one member and turns the pooling back into full leave-one-out at
    full price."""
    assert anchor_key("You arrive at desk 1. (step 3/30)") == \
        anchor_key("You arrive at desk 1. (step 17/30)")


def test_state_of_uses_the_observation_the_turn_acted_from():
    """Turn t's own observation is the RESULT of turn t. Keying on it would
    label every action by its own consequence."""
    turns = [{"observation": "result of turn 0",
              "tool_args": {"opening_obs": "the room at the start"}},
             {"observation": "result of turn 1", "tool_args": {}}]
    assert state_of(turns, 0) == "the room at the start"
    assert state_of(turns, 1) == "result of turn 0"


# --------------------------------------------------------------------------
# Centring only where its premise holds
# --------------------------------------------------------------------------
from ssc.credit.alf_arms import signed_scale  # noqa: E402


def test_mixed_outcome_group_is_still_centred():
    """Where the premise holds, the behaviour is unchanged: a group with both
    winners and losers gets the zero-mean signal it always did."""
    a_star = [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    adv = turn_advantages("anchor_cf", rewards=[1.0, 0.0], lengths=[3, 3],
                          a_star=a_star)
    flat = [v for row in adv for v in row]
    assert min(flat) < 0 < max(flat)


def test_signed_scale_maps_uncentred_values_into_unit_range():
    out = signed_scale([0.0, 0.5, 1.0], centred=False)
    assert min(out) == 0.0 and max(out) == 1.0
    assert signed_scale([0.3, 0.3, 0.3], centred=False) == [0.0, 0.0, 0.0]


def test_hidden_action_list_is_not_recorded_into_the_training_context():
    """What training rebuilds must be what the policy acted on.

    The adapter records the admissible list per turn so the encoder can rebuild
    the prompt. Recorded unconditionally, a run configured to HIDE the list
    would still have it written down, and training would then optimise against
    prompts richer than the ones the policy saw -- a mismatch on every turn,
    with nothing to report it.
    """
    from ssc.env.alfworld_env import build_user_message
    shown = build_user_message("You see a desk.", ["go to desk 1"], 0, 30,
                               action_hint="full")
    hidden = build_user_message("You see a desk.", [], 0, 30,
                                action_hint="none")
    assert "Admissible actions:" in shown
    assert "Admissible actions:" not in hidden
    assert "You see a desk." in hidden and "(step 0/30)" in hidden


def test_every_arm_is_zero_mean_on_homogeneous_groups():
    """A constant in the advantage is a uniform push on whatever was sampled.

    `anchor_cf` used to skip centring when the group's outcomes did not vary,
    mapping its deltas to [0, 1]. That is not zero-mean: on a worked example it
    added +0.36 to every turn of every rollout in the group -- including groups
    where every rollout FAILED, where the push reinforces the turns of losing
    trajectories. Over 27 training steps the arm ran -1.9 points against grpo
    and fell -0.43 points per step (t = -2.01), while an otherwise identical
    arm that centred rose. Every arm must be zero-mean here.
    """
    from ssc.credit.alf_arms import turn_advantages

    lengths = [3, 3, 3]
    a_star = [[0.4, 0.1, 0.0], [0.2, 0.0, 0.5], [0.0, 0.3, 0.1]]
    for rewards in ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]):
        for arm in ("anchor_cf",):
            adv = turn_advantages(arm=arm, rewards=rewards, lengths=lengths,
                                  a_star=a_star)
            flat = [v for row in adv for v in row]
            mean = sum(flat) / len(flat)
            assert abs(mean) < 1e-9, (arm, rewards, mean)
            assert any(v < 0 for v in flat), (arm, rewards)












def test_centring_strips_the_trajectory_level_component_and_keeps_the_scale():
    """A(i,t) = A_E(i) + omega * A_step(i,t) double-counts the outcome unless
    the step term is centred inside each trajectory.

    Measured on mixed-outcome groups, the per-trajectory MEANS of the step term
    correlate with A_E at +0.838: most of what the step slot contributes is a
    second copy of the episode advantage. `centre_within_trajectory` leaves that
    job to A_E and keeps only placement, at unit scale so that omega keeps its
    meaning.
    """
    import statistics
    from ssc.credit.alf_arms import group_advantage, turn_advantages

    lengths = [4, 4, 4]
    a_star = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    keys = [["s0", "s1", "s2", "s3"], ["s0", "s1", "s9", "s3"],
            ["s0", "s7", "s2", "s3"]]
    rewards = [1.0, 0.0, 1.0]
    episode = group_advantage(rewards)

    for arm in ("anchor_cf",):
        adv = turn_advantages(arm, rewards=rewards, lengths=lengths,
                              a_star=a_star, anchor_keys=keys)
        step = [[v - e for v in row] for row, e in zip(adv, episode)]
        for row in step:
            assert abs(statistics.mean(row)) < 1e-9, arm
        flat = [v for row in step for v in row]
        assert abs(statistics.pstdev(flat) - 1.0) < 1e-6, arm



def test_centred_signal_still_orders_turns_within_a_trajectory():
    """Centring must remove the outcome copy, not the placement information."""
    from ssc.credit.alf_arms import group_advantage, turn_advantages

    lengths = [4]
    a_star = [[1.0, 0.0, 0.0, 1.0]]
    rewards = [1.0]
    adv = turn_advantages("anchor_cf", rewards=rewards, lengths=lengths,
                          a_star=a_star)
    row = [v - group_advantage(rewards)[0] for v in adv[0]]
    # the two necessary turns must still outrank the two redundant ones
    assert min(row[0], row[3]) > max(row[1], row[2])




def test_an_unmeasured_turn_is_never_pushed_negative():
    """A turn with NO counterfactual to run must stay exactly neutral.

    Centring over the row LENGTH turns every zero into -mu. A WebShop
    trajectory of 13 turns with two non-zero deltas came out as
    [+0.169, -0.031 x 11, +0.169], so every browsing click was pushed down on
    every step; episodes fell from 10.9 turns to 5.1 while GRPO held at 7.3 and
    held-out success fell instead of rising. The measurement licenses no such
    penalty: it says the turn was removable, and in a search task the turns
    whose single deletion changes nothing are the information gathering the task
    requires.
    """
    from ssc.credit.alf_arms import centre_within_trajectory

    row = [0.2] + [0.0] * 11 + [0.2]
    keep = [True] + [False] * 11 + [True]
    out = centre_within_trajectory([row], [keep])[0]
    assert all(abs(out[t]) < 1e-9 for t in range(1, 12)), out
    # the two measured turns carried the same value, so neither outranks the
    # other and the row is left with no placement claim at all.
    assert abs(out[0]) < 1e-9 and abs(out[12]) < 1e-9, out

    # A row where the measurements DIFFER still orders them, and still leaves
    # the unmeasured turns alone.
    row2 = [0.6, 0.0, 0.0, 0.2]
    out2 = centre_within_trajectory([row2], [[True, False, False, True]])[0]
    assert out2[0] > 0 > out2[3], out2
    assert abs(out2[1]) < 1e-9 and abs(out2[2]) < 1e-9, out2


def test_an_unmeasurable_turn_is_neutral_END_TO_END():
    """From raw deltas all the way to the advantage the loss consumes.

    This is deliberately an end-to-end assertion. The same defect was 'fixed'
    twice against intermediate steps and survived both times, because the
    pipeline is z-score -> centre -> rescale and each stage can reintroduce it:
    a zero that is neutral after centring is pushed below the mean by the
    z-score that runs before it. What the loss finally multiplies by the turn's
    tokens is the only place worth asserting.

    Measured cost of not doing this: our arm scored the multi-hop answer turn
    1.324 below the search turns around it (a permutation control on the same
    permuted, scored it +0.176), the policy stopped emitting `answer[...]`
    entirely, and held-out exact match fell 0.442 -> 0.050.
    """
    from ssc.credit.alf_arms import group_advantage, turn_advantages

    lengths = [3, 3, 3]
    rewards = [1.0, 0.0, 1.0]
    # turn 2 of every rollout is the terminal action: no counterfactual exists.
    a_star = [[0.8, 0.1, 0.0], [0.2, 0.0, 0.0], [0.5, 0.4, 0.0]]
    mask = [[True, True, False]] * 3

    adv = turn_advantages("anchor_cf", rewards=rewards, lengths=lengths,
                          a_star=a_star, a_star_mask=mask, omega=1.0)
    episode = group_advantage(rewards)
    for row, e in zip(adv, episode):
        step = [v - e for v in row]
        assert abs(step[2]) < 1e-9, f"terminal turn carried {step[2]:+.3f}"

    # and without the mask the same input DOES push it negative -- the bug this
    # test exists to catch, kept as the contrast rather than described in prose.
    adv_bad = turn_advantages("anchor_cf", rewards=rewards, lengths=lengths,
                              a_star=a_star, omega=1.0)
    worst = min(v - e for row, e in zip(adv_bad, episode) for v in row[2:])
    assert worst < -0.1, worst


def test_anchor_grouping_is_exact_on_alfworld_and_webshop():
    """Anchor grouping matches IDENTICAL states here; similarity is a QA-only
    variant.

    This test replaces one that asserted the opposite. The earlier version read
    the source method's line "we incorporate similarity-based ... threshold of
    0.9" as the general rule and pinned similarity clustering as the default.
    That line sits inside the paragraph on search-augmented QA; the method
    section defines the grouping as "identifying and grouping IDENTICAL states
    across trajectories ... lightweight key-based grouping using hashmaps".

    The mistake was not cosmetic. Coarser grouping drags A_S toward the turn
    position it is meant to beat -- `reports/alf_audit_robustness.json` sweeps
    key coarseness and the correlation with planner progress runs -0.050 at
    0.973 coverage, -0.072 at 0.962, and +0.356 for a degenerate turn-index key
    at 0.984, while similarity clustering measures 0.984. So a WebShop baseline
    trained under similarity is not the published method, and the whole point of
    running it is that it IS the published method.
    """
    from ssc.credit.alf_arms import anchor_step_advantage, similarity_clusters

    a = "Search results page 1: red wool scarf $18.99 blue cotton hat $12.50"
    b = "Search results page 1: red wool scarf $18.99 blue cotton hat $12.55"
    keys, returns = [[a], [b]], [[1.0], [0.0]]

    # Default: two pages one character apart are two different anchors, so each
    # is a group of one and A_S is zero -- the standard anchor convention.
    assert anchor_step_advantage(keys, returns) == [[0.0], [0.0]]

    # The QA variant is still reachable, and multi-hop passes it explicitly.
    got = anchor_step_advantage(keys, returns, sim_threshold=0.9)
    assert got[0][0] > 0 > got[1][0], got
    cl = similarity_clusters([a, b])
    assert cl[a] == cl[b], "the QA variant must still merge near-identical pages"


def test_similarity_clustering_is_deterministic():
    """Cluster ids feed the advantage, so a reordering must not change them."""
    from ssc.credit.alf_arms import similarity_clusters

    keys = ["alpha beta gamma", "alpha beta gamm", "zeta eta theta", "alpha beta gamma"]
    first = similarity_clusters(keys)
    assert similarity_clusters(list(keys)) == first
    assert len(set(first.values())) == 2, first


def test_step_std_off_removes_the_group_amplification():
    """Dividing by a group std set by a handful of values blows them up.

    The step-level baseline literature reports the same: the normalisation
    factor helps on easy tasks and hurts on hard ones, where imbalanced groups
    make the divisor small. Our
    WebShop groups carry 7 distinct non-zero measurements across 279 turns, so
    the divisor is the extreme case. Centring alone keeps the ordering and drops
    the amplification.
    """
    from ssc.credit.alf_arms import masked_zscore

    vals = [0.10, 0.10, 0.10, 0.10, 0.30]     # sd is small, one value stands out
    keep = [True] * 5
    with_std = masked_zscore(vals, keep, divide_by_std=True)
    without = masked_zscore(vals, keep, divide_by_std=False)
    assert max(with_std) > 4 * max(without), (with_std, without)
    # ordering is identical -- only the scale changed
    assert [i for i, _ in sorted(enumerate(with_std), key=lambda x: x[1])] == \
           [i for i, _ in sorted(enumerate(without), key=lambda x: x[1])]
    assert abs(sum(without)) < 1e-9, "still zero-mean"


# --- anchor_mass: the multiplicative form (runbook sections 7/53/54) ---------
#
# The additive arm scored below the untrained base model on WebShop. The
# mechanism is that centring forces a batch of turns in every trajectory to take
# a negative step term: measured, 26-54% of the turns in successful trajectories
# ended with a negative advantage, while plain GRPO flips none of them. The
# multiplicative form rules that out structurally. The tests below are the
# complete list of reasons it differs from the additive arm, asserted one by one.

def _mass_kw():
    return dict(rewards=[1.0, 0.0, 1.0], lengths=[4, 4, 4],
                a_star=[[0.8, 0.0, 0.1, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.2, 0.5, 0.0, 0.3]])


def test_anchor_mass_alpha_one_is_exactly_grpo():
    """At alpha=1 the weights are identically 1, so the arm must equal GRPO
    element by element -- the zero point of the dose knob."""
    from ssc.credit.alf_arms import turn_advantages

    kw = _mass_kw()
    mass = turn_advantages("anchor_mass", alpha=1.0, **kw)
    grpo = turn_advantages("grpo", **kw)
    for rm, rg in zip(mass, grpo):
        for a, b in zip(rm, rg):
            assert abs(a - b) < 1e-9, (mass, grpo)


def test_anchor_mass_preserves_the_advantage_mass_of_each_trajectory():
    """mean_t A(i,t) == A_E(i). The additive arm only guarantees that the step
    term is zero-mean; what is guaranteed here is the total."""
    import statistics

    from ssc.credit.alf_arms import group_advantage, turn_advantages

    kw = _mass_kw()
    adv = turn_advantages("anchor_mass", alpha=0.5, **kw)
    for row, e in zip(adv, group_advantage(kw["rewards"])):
        if e <= 0:                      # apply_to="positive": non-positive rows stay uniform
            assert all(abs(v - e) < 1e-9 for v in row)
        else:
            assert abs(statistics.mean(row) - e) < 1e-6, (row, e)


def test_anchor_mass_never_flips_the_sign_of_a_successful_trajectory():
    """This is the entire reason the arm exists.

    Under the additive form the same input pushes turns of a successful
    trajectory to a negative advantage -- that unlearns behaviour the policy
    already got right, and it is the mechanism behind falling below the base
    model on WebShop. The multiplicative form has q >= 0 and cannot do it. The
    contrast assertion pins the additive arm's behaviour as well, so the two are
    not read as differing only in scale.
    """
    from ssc.credit.alf_arms import group_advantage, turn_advantages

    kw = _mass_kw()
    episode = group_advantage(kw["rewards"])
    mass = turn_advantages("anchor_mass", alpha=0.5, **kw)
    for row, e in zip(mass, episode):
        if e > 0:
            assert all(v >= -1e-12 for v in row), row

    add = turn_advantages("anchor_cf", omega=1.0, **kw)
    flipped = sum(1 for row, e in zip(add, episode) if e > 0
                  for v in row if v < 0)
    assert flipped > 0, \
        "the additive arm flipped no sign on this example, so the contrast is meaningless"


def test_anchor_mass_gives_more_credit_to_the_turn_that_measured_more():
    from ssc.credit.alf_arms import turn_advantages

    adv = turn_advantages("anchor_mass", rewards=[1.0, 0.0], lengths=[3, 3],
                          a_star=[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], alpha=0.5)
    assert adv[0][0] > adv[0][1], adv[0]


# --- anchor_cf_grouped: measured values fed through anchor normalisation -----
#
# This arm exists because of a measurement: our deletion deltas are absolute
# quantities, and on WebShop they correlate with turn position at rho = +0.31 to
# +0.34, with not one of 247 samples negative -- what they capture is structural
# necessity, and in a browsing task that is close to a position prior. A step
# term built from within-anchor comparison instead contrasts different actions
# taken from the SAME state, and centring inside the anchor group removes "how
# hard this state is" and "how late this turn is" together. The test below pins
# that the only difference is where the measurement comes from.


def test_anchor_cf_grouped_needs_both_inputs():
    from ssc.credit.alf_arms import turn_advantages

    import pytest
    kw = dict(rewards=[1.0, 0.0], lengths=[1, 1])
    with pytest.raises(ValueError):
        turn_advantages("anchor_cf_grouped", a_star=[[1.0], [0.0]], **kw)
    with pytest.raises(ValueError):
        turn_advantages("anchor_cf_grouped", anchor_keys=[["s"], ["s"]], **kw)



# --- anneal_dose: use the step term early, withdraw it late -----------------

def test_anneal_dose_walks_from_the_arm_to_grpo_and_stays_there():
    """The endpoint must be exactly GRPO: omega=0 and alpha=1, with no further
    change once anneal_steps has been passed."""
    from ssc.credit.alf_arms import anneal_dose

    assert anneal_dose(0.4, 0.5, 0, 60) == (0.4, 0.5)
    om, al = anneal_dose(0.4, 0.5, 30, 60)
    assert abs(om - 0.2) < 1e-12 and abs(al - 0.75) < 1e-12
    assert anneal_dose(0.4, 0.5, 60, 60) == (0.0, 1.0)
    assert anneal_dose(0.4, 0.5, 99, 60) == (0.0, 1.0), \
        "once the anneal is over it must stay pinned at GRPO"


def test_anneal_dose_is_a_no_op_when_off():
    from ssc.credit.alf_arms import anneal_dose

    for n in (None, 0, -5):
        assert anneal_dose(0.4, 0.5, 37, n) == (0.4, 0.5)


def test_annealed_endpoint_is_exactly_grpo_through_turn_advantages():
    """Feeding the annealed endpoint into turn_advantages, both the additive and
    the multiplicative arm must equal grpo element by element."""
    from ssc.credit.alf_arms import anneal_dose, turn_advantages

    kw = dict(rewards=[1.0, 0.0, 1.0], lengths=[3, 3, 3],
              a_star=[[0.8, 0.0, 0.1], [0.0, 0.0, 0.0], [0.2, 0.5, 0.0]])
    om, al = anneal_dose(0.4, 0.5, 60, 60)
    grpo = turn_advantages("grpo", **kw)
    add = turn_advantages("anchor_cf", omega=om, **kw)
    mul = turn_advantages("anchor_mass", alpha=al, **kw)
    for ref, got in ((grpo, add), (grpo, mul)):
        for r, g in zip(ref, got):
            assert all(abs(x - y) < 1e-9 for x, y in zip(r, g)), (ref, got)





def test_anchor_mass_neutral_unmeasured_keeps_grpo_credit_on_silent_turns():
    """09-12 WebShop probe: turns on which the operator is structurally silent
    (search / navigation) must not be pushed below a weight of 1 just because
    their delta is 0. Under neutral_unmeasured their weight is identically 1
    (they get the GRPO share), and the conservation holds only over the
    measurable turns."""
    from ssc.credit.alf_arms import turn_advantages
    rewards = [1.0, 0.0]; lengths = [4, 4]
    a_star = [[0.0, 1.0, 0.0, 0.2], [0.0] * 4]
    mask = [[False, True, False, True], [False] * 4]     # turns 0 and 2 are structurally unmeasurable
    A = turn_advantages("anchor_mass", rewards, lengths, a_star=a_star, a_star_mask=mask,
                        alpha=0.5, neutral_unmeasured=True)
    e = turn_advantages("grpo", rewards, lengths)[0][0]
    assert A[0][0] == e and A[0][2] == e                 # unmeasurable: identical to GRPO
    q1, q3 = A[0][1] / e, A[0][3] / e
    assert abs((q1 + q3) / 2 - 1.0) < 1e-6 and q1 > 1.0 > q3   # conserved across measurable turns (mean=1); the larger contributor gets more
    assert A[1] == [e2 for e2 in turn_advantages("grpo", rewards, lengths)[1]]   # the failing trajectory is unchanged
    # at alpha=1 it degenerates exactly to GRPO
    A1 = turn_advantages("anchor_mass", rewards, lengths, a_star=a_star, a_star_mask=mask, alpha=1.0, neutral_unmeasured=True)
    assert A1 == turn_advantages("grpo", rewards, lengths)


# ---------------------------------------------------------------------------
# 09-13 ALFWorld 4B gating: nonneg_winners / zero_uniform_groups
# ---------------------------------------------------------------------------

def _gated_mixed_group():
    # 4 trajectories: two successes (4 steps each, only steps 3 and 4 necessary)
    # and two failures (off-path fill values)
    rewards = [1.0, 1.0, 0.0, 0.0]
    lengths = [4, 4, 4, 4]
    a_star = [[0.0, 0.0, 1.0, 1.0],
              [0.0, 1.0, 0.0, 1.0],
              [1.0, 0.75, 0.5, 0.0],
              [1.0, 1.0, 0.0, 0.0]]
    return rewards, lengths, a_star


def test_gated_anchor_cf_with_both_gates_off_is_bitwise_anchor_cf():
    rewards, lengths, a_star = _gated_mixed_group()
    plain = turn_advantages("anchor_cf", rewards, lengths, a_star=a_star)
    same = turn_advantages("anchor_cf", rewards, lengths, a_star=a_star,
                           nonneg_winners=False, zero_uniform_groups=False)
    assert same == plain


def test_gated_anchor_cf_never_pushes_a_winner_turn_below_its_episode_advantage():
    rewards, lengths, a_star = _gated_mixed_group()
    plain = turn_advantages("anchor_cf", rewards, lengths, a_star=a_star)
    gated = turn_advantages("anchor_cf", rewards, lengths, a_star=a_star,
                            nonneg_winners=True)
    episode = group_advantage(rewards)
    # ungated: steps of a successful trajectory whose deletion leaves the outcome
    # unchanged are pushed below A_E (54% of the 4B `go to` steps in the probe)
    assert any(v < episode[0] - 1e-9 for v in plain[0])
    for i in (0, 1):
        for t, v in enumerate(gated[i]):
            if a_star[i][t] > 0:      # necessary steps are still raised
                assert v > episode[i] + 1e-9
            else:                     # redundant steps return to A_E and are no longer penalised
                assert abs(v - episode[i]) < 1e-9
    # the failing trajectories are unchanged element by element: the gate only
    # touches trajectories with a positive advantage
    assert gated[2] == plain[2] and gated[3] == plain[3]


def test_gated_anchor_cf_is_exactly_grpo_on_uniform_outcome_groups():
    rewards = [1.0, 1.0, 1.0]
    lengths = [3, 4, 3]
    a_star = [[0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
    plain = turn_advantages("anchor_cf", rewards, lengths, a_star=a_star)
    gated = turn_advantages("anchor_cf", rewards, lengths, a_star=a_star,
                            zero_uniform_groups=True)
    grpo = turn_advantages("grpo", rewards, lengths)
    # ungated: the step term is still non-zero in an all-success group
    # (the |A| ~= 0.9 position-shaped signal seen for 4B in the probe)
    assert any(abs(v) > 1e-9 for row in plain for v in row)
    assert gated == grpo
    # an all-failure group is zeroed the same way
    zeros = turn_advantages("anchor_cf", [0.0, 0.0], [3, 3],
                            a_star=[[1.0, 0.5, 0.0], [0.2, 0.9, 0.1]],
                            zero_uniform_groups=True)
    assert zeros == turn_advantages("grpo", [0.0, 0.0], [3, 3])
    # mixed groups are unaffected by this switch
    r2, l2, a2 = _gated_mixed_group()
    assert turn_advantages("anchor_cf", r2, l2, a_star=a2, zero_uniform_groups=True) == \
        turn_advantages("anchor_cf", r2, l2, a_star=a2)
