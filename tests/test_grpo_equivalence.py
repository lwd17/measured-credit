"""Runbook section 55 Test 1, at the loss level.

The reduction has to hold NUMERICALLY on the actual loss, not just on the
weight vector, because that is the claim reviewers will check: with no
validated REPLACE event, SSC must be GRPO.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssc.credit.grpo_patch import (  # noqa: E402
    grpo_loss, group_normalized_advantage, mass_preserve_batch, ssc_loss,
)
from ssc.credit.survival import build_turn_survival, expand_to_tokens  # noqa: E402
from ssc.detector.schema import SemanticEvent  # noqa: E402


def _batch(B=4, T=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    actor_mask = torch.zeros(B, T, dtype=torch.long)
    for i in range(B):
        actor_mask[i, : T - i] = 1                    # ragged, like real rollouts
    old = torch.randn(B, T, generator=g) * 0.1 - 2.0
    new = old + torch.randn(B, T, generator=g) * 0.05
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    return rewards, actor_mask, old, new


def test1_ssc_equals_grpo_when_no_events():
    rewards, mask, old, new = _batch()
    token_q = torch.ones_like(mask, dtype=torch.float)

    a = grpo_loss(rewards, mask, old, new)
    b = ssc_loss(rewards, token_q, mask, old, new, mass_preserving=True)
    assert torch.allclose(a, b, atol=1e-6), f"GRPO {a.item()} != SSC {b.item()}"


def test1_reduction_also_holds_without_mass_preservation():
    rewards, mask, old, new = _batch(seed=3)
    token_q = torch.ones_like(mask, dtype=torch.float)
    a = grpo_loss(rewards, mask, old, new)
    b = ssc_loss(rewards, token_q, mask, old, new, mass_preserving=False)
    assert torch.allclose(a, b, atol=1e-6)


def test1_reduction_holds_with_groups():
    rewards, mask, old, new = _batch(seed=7)
    groups = torch.tensor([0, 0, 1, 1])
    token_q = torch.ones_like(mask, dtype=torch.float)
    a = grpo_loss(rewards, mask, old, new, group_ids=groups)
    b = ssc_loss(rewards, token_q, mask, old, new, group_ids=groups)
    assert torch.allclose(a, b, atol=1e-6)


def test_ssc_differs_once_a_real_event_exists():
    """Guard against a vacuous reduction test: SSC must actually do something."""
    rewards, mask, old, new = _batch(seed=11)
    token_q = torch.ones_like(mask, dtype=torch.float)
    token_q[0, :4] = 0.0                    # invalidate part of rollout 0
    a = grpo_loss(rewards, mask, old, new)
    b = ssc_loss(rewards, token_q, mask, old, new)
    assert not torch.allclose(a, b, atol=1e-6)


def test_mass_preserve_batch_matches_scalar_implementation():
    from ssc.credit.normalize import mass_preserve

    turn_q = build_turn_survival(
        6, [SemanticEvent("REPLACE", 1, 4, accepted=True, confidence=1.0)])
    tot = [i // 1 for i in range(6)]
    mask_l = [1] * 6
    scalar = mass_preserve(expand_to_tokens(turn_q, tot, mask_l), mask_l)

    batched = mass_preserve_batch(torch.tensor([turn_q]),
                                  torch.tensor([mask_l]))
    assert torch.allclose(batched[0], torch.tensor(scalar), atol=1e-6)


def test_mass_preserve_batch_mean_is_one_per_row():
    q = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                      [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 0]])
    out = mass_preserve_batch(q, mask)
    means = (out * mask).sum(-1) / mask.sum(-1)
    assert torch.allclose(means, torch.ones(2), atol=1e-4)


def test_mass_preserve_batch_degenerate_row_is_uniform():
    q = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    out = mass_preserve_batch(q, mask)
    assert torch.allclose(out[0, :3], torch.ones(3))
    assert out[0, 3] == 0.0


def test_group_normalized_advantage_is_zero_mean_per_group():
    r = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    g = torch.tensor([0, 0, 0, 1, 1, 1])
    a = group_normalized_advantage(r, g)
    for grp in (0, 1):
        assert abs(a[g == grp].mean().item()) < 1e-5


def test_degenerate_group_all_equal_rewards_gives_finite_advantage():
    """All-correct or all-wrong groups are common and must not produce NaN."""
    r = torch.tensor([1.0, 1.0, 1.0])
    a = group_normalized_advantage(r, torch.tensor([0, 0, 0]))
    assert torch.isfinite(a).all() and torch.allclose(a, torch.zeros(3))


def test_observation_tokens_get_no_gradient_weight():
    """Section 55 Test 5 at the loss level."""
    rewards, mask, old, new = _batch(seed=5)
    token_q = torch.ones_like(mask, dtype=torch.float)
    adv_masked = ssc_loss(rewards, token_q, mask, old, new)

    # perturbing logprobs only at masked positions must not change the loss
    new2 = new.clone()
    new2[mask == 0] += 5.0
    adv_masked2 = ssc_loss(rewards, token_q, mask, old, new2)
    assert torch.allclose(adv_masked, adv_masked2, atol=1e-6)


# --- sign semantics: the objective must match the metric it is named for -----

def test_negative_advantage_keeps_its_penalty_by_default():
    """M_inv^+ measures invalidated POSITIVE credit, so the objective must not
    also strip NEGATIVE advantage from invalidated tokens. A repeatedly retried
    failed action in a losing rollout is a plausible cause of the loss; zeroing
    its advantage would let it escape exactly the penalty it should receive."""
    import torch
    from ssc.credit.grpo_patch import ssc_token_advantage
    seq_adv = torch.tensor([1.0, -1.0])
    q = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    mask = torch.ones(2, 2)
    adv = ssc_token_advantage(seq_adv, q, mask, mass_preserving=False)
    assert adv[0].tolist() == [1.0, 0.0], "positive row must be reweighted"
    assert adv[1].tolist() == [-1.0, -1.0], "negative row must keep full penalty"


def test_apply_to_both_restores_the_symmetric_form():
    import torch
    from ssc.credit.grpo_patch import ssc_token_advantage
    adv = ssc_token_advantage(torch.tensor([-1.0]), torch.tensor([[1.0, 0.0]]),
                              torch.ones(1, 2), mass_preserving=False,
                              apply_to="both")
    assert adv[0].tolist() == [-1.0, -0.0]


def test_apply_to_rejects_unknown_mode():
    import pytest
    import torch
    from ssc.credit.grpo_patch import ssc_token_advantage
    with pytest.raises(ValueError):
        ssc_token_advantage(torch.tensor([1.0]), torch.ones(1, 2), torch.ones(1, 2),
                            apply_to="sometimes")


def test_reduction_to_grpo_holds_under_both_sign_modes():
    """Section 55 Test 1 must survive the sign fix, in either mode."""
    import torch
    from ssc.credit.grpo_patch import grpo_loss, ssc_loss
    torch.manual_seed(0)
    rew = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.ones(4, 6)
    q = torch.ones(4, 6)
    old = torch.randn(4, 6)
    new = old + 0.01 * torch.randn(4, 6)
    gid = torch.tensor([0, 0, 1, 1])
    base = grpo_loss(rew, mask, old, new, gid)
    for mode in ("positive", "both"):
        assert torch.allclose(
            base, ssc_loss(rew, q, mask, old, new, gid, mass_preserving=False,
                           apply_to=mode), atol=1e-7), mode


# --- micro-batching must not change the update --------------------------------

def test_single_element_microbatch_self_normalises_to_zero():
    """The failure this guards against: group normalisation inside a
    micro-batch of one gives advantage 0 for every row, so the gradient is
    exactly zero and the run trains nothing while logging normally. Observed as
    |g| = 0.000 on a real training step."""
    import torch
    from ssc.credit.grpo_patch import group_normalized_advantage
    assert float(group_normalized_advantage(torch.tensor([0.7]),
                                            torch.tensor([0]))[0]) == 0.0


def test_microbatched_gradient_equals_full_batch_gradient():
    import torch
    from ssc.credit.grpo_patch import grpo_loss, group_normalized_advantage
    torch.manual_seed(0)
    B, T = 6, 5
    rew = torch.tensor([1.0, 0.0, 0.5, 1.0, 0.0, 0.5])
    gid = torch.tensor([0, 0, 0, 1, 1, 1])
    mask, old = torch.ones(B, T), torch.randn(B, T)
    adv = group_normalized_advantage(rew, gid)

    full = old.clone().requires_grad_(True)
    grpo_loss(rew, mask, old, full, gid, advantages=adv).backward()

    micro = old.clone().requires_grad_(True)
    for i in range(B):
        sl = slice(i, i + 1)
        (grpo_loss(rew[sl], mask[sl], old[sl], micro[sl], gid[sl],
                   advantages=adv[sl]) / B).backward()
    assert torch.allclose(full.grad, micro.grad, atol=1e-6)


def test_supplied_advantages_override_internal_normalisation():
    import torch
    from ssc.credit.grpo_patch import grpo_loss
    loss = grpo_loss(torch.tensor([0.7]), torch.ones(1, 3), torch.zeros(1, 3),
                     torch.zeros(1, 3, requires_grad=True), torch.tensor([0]),
                     advantages=torch.tensor([2.0]))
    assert abs(float(loss) + 2.0) < 1e-6, "at ratio 1 the loss is -advantage"


def test_ssc_also_accepts_supplied_advantages():
    import torch
    from ssc.credit.grpo_patch import ssc_loss
    loss = ssc_loss(torch.tensor([0.7]), torch.ones(1, 3), torch.ones(1, 3),
                    torch.zeros(1, 3), torch.zeros(1, 3, requires_grad=True),
                    torch.tensor([0]), mass_preserving=False,
                    advantages=torch.tensor([2.0]))
    assert abs(float(loss) + 2.0) < 1e-6
