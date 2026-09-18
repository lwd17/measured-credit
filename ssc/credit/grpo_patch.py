"""GRPO integration (runbook sections 8, 54).

The ONLY change relative to baseline GRPO is the token advantage:

    baseline :  token_adv[i, j] = seq_adv[i]
    SSC      :  token_adv[i, j] = seq_adv[i] * q_hat[i, j]

Everything else -- ratio, clipping, masking, reduction -- is untouched, which
is what makes section 55's Test 1 (exact reduction to GRPO when there are no
REPLACE events) a meaningful check rather than a tautology.

Section 27's fairness rule is the reason this file contains no optimizer,
schedule or KL logic: the experimental variable must be credit assignment, not
a retuned trainer.
"""

from __future__ import annotations

import torch

from ssc.credit.normalize import EPS


def group_normalized_advantage(rewards: torch.Tensor,
                               group_ids: torch.Tensor | None = None,
                               eps: float = 1e-6) -> torch.Tensor:
    """A_i = (R_i - mu_G) / (sigma_G + eps)   (section 5).

    With `group_ids`, normalisation is per group (the usual GRPO grouping of
    several rollouts per prompt); without, the whole batch is one group.
    """
    rewards = rewards.float()
    if group_ids is None:
        return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)

    out = torch.zeros_like(rewards)
    for g in torch.unique(group_ids):
        sel = group_ids == g
        r = rewards[sel]
        out[sel] = (r - r.mean()) / (r.std(unbiased=False) + eps)
    return out


def mass_preserve_batch(token_q: torch.Tensor, actor_mask: torch.Tensor,
                        eps: float = EPS) -> torch.Tensor:
    """Batched section 53. Rows whose actor mass is ~0 fall back to uniform."""
    mask = actor_mask.float()
    n_active = mask.sum(dim=-1, keepdim=True)
    total = (token_q * mask).sum(dim=-1, keepdim=True)
    mean_q = total / n_active.clamp(min=1.0)

    scaled = token_q / (mean_q + eps)
    # degenerate rows -> uniform weights, matching normalize.mass_preserve
    degenerate = (mean_q < eps).expand_as(scaled)
    scaled = torch.where(degenerate, torch.ones_like(scaled), scaled)
    return scaled * mask


def ssc_token_advantage(seq_adv: torch.Tensor, token_q: torch.Tensor,
                        actor_mask: torch.Tensor,
                        mass_preserving: bool = True,
                        apply_to: str = "positive") -> torch.Tensor:
    """Section 54's two lines, isolated so the tests can target them.

    `apply_to` fixes a sign inconsistency that was present in the original
    formulation. The quantity the method is named for and measured by is
    M_inv^+ -- invalidated POSITIVE credit mass -- but multiplying every token
    advantage by q also removes NEGATIVE advantage from invalidated tokens.
    For the avoidable-waste criterion that is backwards: a repeatedly-retried
    failed action in a losing rollout is a plausible CAUSE of the loss, and
    zeroing its advantage lets it escape the penalty that should teach the
    policy not to do it.

      "positive" (default) -- reweight only where seq_adv > 0. Objective and
                              metric then measure the same thing.
      "both"               -- the original symmetric form. Defensible for the
                              dataflow criterion, where a turn outside the
                              dependency closure of a failed outcome may
                              genuinely be unrelated to the failure, but it is
                              a claim and must be made explicitly.
    """
    if apply_to not in ("positive", "both"):
        raise ValueError(f"apply_to must be 'positive' or 'both', got {apply_to!r}")
    q = mass_preserve_batch(token_q, actor_mask) if mass_preserving \
        else token_q * actor_mask.float()
    weighted = seq_adv.unsqueeze(-1) * q
    if apply_to == "both":
        return weighted
    # keep negative-advantage rows uniform, still masked to actor positions
    uniform = seq_adv.unsqueeze(-1) * actor_mask.float()
    positive = (seq_adv > 0).unsqueeze(-1)
    return torch.where(positive, weighted, uniform)


def clipped_policy_gradient_loss(token_advantage: torch.Tensor,
                                 actor_mask: torch.Tensor,
                                 old_logprob: torch.Tensor,
                                 new_logprob: torch.Tensor,
                                 clip_eps: float = 0.2) -> torch.Tensor:
    """Standard PPO/GRPO clipped surrogate, token-mean over actor positions."""
    ratio = torch.exp(new_logprob - old_logprob)
    unclipped = ratio * token_advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * token_advantage
    per_token = -torch.min(unclipped, clipped)

    mask = actor_mask.float()
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def grpo_loss(rewards, actor_mask, old_logprob, new_logprob,
              group_ids=None, clip_eps: float = 0.2,
              advantages: torch.Tensor | None = None) -> torch.Tensor:
    """Baseline GRPO: every actor token of a rollout gets the same advantage.

    `advantages` MUST be supplied when a step is split into micro-batches.
    Group normalisation over a single-element micro-batch gives
    (r - r) / (0 + eps) = 0 for every row, so the gradient is exactly zero and
    the run trains nothing while logging normally -- observed as |g| = 0.000 on
    a real step before this parameter existed.
    """
    seq_adv = (advantages if advantages is not None
               else group_normalized_advantage(rewards, group_ids))
    token_adv = seq_adv.unsqueeze(-1).expand_as(actor_mask.float())
    return clipped_policy_gradient_loss(
        token_adv * actor_mask.float(), actor_mask, old_logprob, new_logprob, clip_eps)


def ssc_loss(rewards, token_q, actor_mask, old_logprob, new_logprob,
             group_ids=None, clip_eps: float = 0.2,
             mass_preserving: bool = True,
             apply_to: str = "positive",
             advantages: torch.Tensor | None = None) -> torch.Tensor:
    """SSC-GRPO (section 54). Identical to grpo_loss when token_q is all ones.

    See `grpo_loss` on why `advantages` must be supplied under micro-batching.
    """
    seq_adv = (advantages if advantages is not None
               else group_normalized_advantage(rewards, group_ids))
    token_adv = ssc_token_advantage(seq_adv, token_q, actor_mask, mass_preserving,
                                    apply_to)
    return clipped_policy_gradient_loss(
        token_adv, actor_mask, old_logprob, new_logprob, clip_eps)
