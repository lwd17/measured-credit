"""Properties the counterfactual credit weights must hold.

The dose knob is the whole experimental design, so the two things that would
silently invalidate it are pinned here: alpha = 1 must be EXACTLY GRPO, and
every usable dose must conserve credit mass, or a smaller alpha would also mean
a smaller effective step and the sweep would measure learning rate instead of
dose.
"""
import random

import pytest

from ssc.credit.counterfactual import contribution_weights, shuffled


def test_alpha_one_is_exactly_uniform():
    for deltas in ([0, 5, 0, 3], [1, 1, 1], [0] * 10 + [7]):
        w = contribution_weights(deltas, alpha=1.0)
        assert all(x == 1.0 for x in w), w


@pytest.mark.parametrize("alpha", [0.9, 0.75, 0.5, 0.25, 0.1])
def test_every_usable_dose_conserves_mass(alpha):
    for deltas in ([0, 5, 0, 3], [0] * 20 + [5, 3], [2, 0, 0, 0, 9, 1]):
        w = contribution_weights(deltas, alpha=alpha)
        assert abs(sum(w) / len(w) - 1.0) < 1e-6, (alpha, deltas, sum(w) / len(w))


def test_cap_is_never_exceeded():
    w = contribution_weights([0] * 30 + [10], alpha=0.1, w_max=3.0)
    assert max(w) <= 3.0 + 1e-9


def test_no_measured_contribution_falls_back_to_uniform():
    # A rollout whose replay produced nothing must degrade to GRPO, not to
    # arbitrary credit.
    assert contribution_weights([0, 0, 0], alpha=0.5) == [1.0, 1.0, 1.0]
    assert contribution_weights([], alpha=0.5) == []


def test_weight_is_monotone_in_contribution():
    w = contribution_weights([0, 1, 2, 3], alpha=0.5)
    assert w[0] < w[1] < w[2] < w[3]


def test_shuffle_control_preserves_the_multiset():
    d = [0, 0, 5, 0, 3]
    s = shuffled(d, random.Random(0))
    assert sorted(s) == sorted(d)
    # and therefore removes exactly as much credit, only elsewhere
    a = contribution_weights(d, alpha=0.5)
    b = contribution_weights(s, alpha=0.5)
    assert abs(sum(a) - sum(b)) < 1e-9


def test_lower_alpha_spreads_the_weights_further():
    d = [0, 0, 5, 0, 3]
    spread = [max(contribution_weights(d, alpha=a)) - min(contribution_weights(d, alpha=a))
              for a in (0.9, 0.5, 0.25)]
    assert spread[0] < spread[1] < spread[2]
