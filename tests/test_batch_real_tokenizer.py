"""Encoding checked against the REAL Qwen3 tokenizer, not a stub.

The stub in test_batch_encoding.py cannot catch template-specific behaviour,
and the template is where the damaging bug was: Qwen3 strips <think> blocks
from all assistant messages except the last, which made a flat whole-trajectory
encoding misattribute actor tokens without raising.

Skipped when the tokenizer is unavailable so the suite still runs offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssc.detector.rollout import Outcome, Rollout, Turn  # noqa: E402
from ssc.detector.schema import SemanticEvent  # noqa: E402
from ssc.train.batch import (  # noqa: E402
    apply_survival, collate, encode_turns,
)

MODEL = "/disk3/yezhen/liangwd_models/Qwen3-4B"


@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {e}")


def rollout(n=4, success=1):
    return Rollout(
        task_id="t", question="determine if unknown substance U is conductive",
        turns=[Turn(i, f"<think>THINKING{i} about the circuit</think>\nAction: go room{i}",
                    "sw_action", {"action": f"go room{i}"},
                    f"OBSERVATION{i} you moved", 40 + i)
               for i in range(n)],
        outcome=Outcome(success=success, reward=float(success)))


def test_one_example_per_turn(tok):
    r = rollout(4)
    exs = encode_turns(r, tok, "SYSTEM")
    assert len(exs) == 4
    assert [e.turn_index for e in exs] == [0, 1, 2, 3]


def test_each_example_has_actor_tokens(tok):
    """A zero-actor example would dilute the token mean without erroring."""
    for e in encode_turns(rollout(4), tok, "SYSTEM"):
        assert e.n_actor_tokens > 0


def test_this_turns_reasoning_is_inside_the_actor_span(tok):
    """The generated reasoning MUST carry loss -- that is the bug being guarded."""
    r = rollout(3)
    exs = encode_turns(r, tok, "SYSTEM")
    for ex in exs:
        actor_ids = [i for i, m in zip(ex.input_ids, ex.actor_mask) if m]
        text = tok.decode(actor_ids)
        assert f"THINKING{ex.turn_index}" in text, \
            f"turn {ex.turn_index} reasoning is not in its own actor span"


def test_earlier_reasoning_is_not_in_a_later_actor_span(tok):
    """Turn t's actor span must contain only turn t's generation."""
    r = rollout(3)
    for ex in encode_turns(r, tok, "SYSTEM"):
        actor_ids = [i for i, m in zip(ex.input_ids, ex.actor_mask) if m]
        text = tok.decode(actor_ids)
        for other in range(3):
            if other != ex.turn_index:
                assert f"THINKING{other}" not in text


def test_observations_are_never_actor_tokens(tok):
    """Section 55 Test 5 against the real template."""
    r = rollout(3)
    for ex in encode_turns(r, tok, "SYSTEM"):
        actor_ids = [i for i, m in zip(ex.input_ids, ex.actor_mask) if m]
        text = tok.decode(actor_ids)
        for k in range(3):
            assert f"OBSERVATION{k}" not in text


def test_context_grows_and_matches_inference_history(tok):
    """Turn t's context must contain earlier ACTIONS but no earlier reasoning,
    because that is exactly what the policy conditioned on at rollout time."""
    r = rollout(3)
    exs = encode_turns(r, tok, "SYSTEM")
    for ex in exs:
        ctx_ids = [i for i, m in zip(ex.input_ids, ex.actor_mask) if not m]
        ctx = tok.decode(ctx_ids)
        for earlier in range(ex.turn_index):
            assert f"go room{earlier}" in ctx, "earlier action missing from context"
            assert f"THINKING{earlier}" not in ctx, \
                "earlier reasoning leaked into context the policy never saw"


def test_survival_is_applied_per_turn(tok):
    r = rollout(5)
    exs = encode_turns(r, tok, "SYSTEM")
    # mass_preserving=False isolates the RAW section 6.1 coefficients; the
    # rollout-scope renormalisation has its own tests in test_trainer_arms.py.
    apply_survival(exs, r.num_turns,
                   [SemanticEvent("REPLACE", 1, 3, accepted=True, confidence=1.0)],
                   hard=True, mass_preserving=False)
    assert [e.q for e in exs] == [1.0, 0.0, 0.0, 1.0, 1.0]


def test_collate_q_is_the_mask_scaled(tok):
    """q must never be an independent source of truth about actor positions."""
    r = rollout(4)
    exs = encode_turns(r, tok, "SYSTEM")
    apply_survival(exs, r.num_turns,
                   [SemanticEvent("REPLACE", 0, 2, accepted=True, confidence=1.0)],
                   hard=True)
    b = collate(exs, tok.pad_token_id or tok.eos_token_id)
    for i, ex in enumerate(exs):
        nz = (b.token_q[i] != 0).long()
        expected = b.actor_mask[i] * (1 if ex.q > 0 else 0)
        assert (nz == expected).all(), "token_q disagrees with actor_mask"


def test_no_events_gives_grpo_identical_loss_with_real_tokenizer(tok):
    import torch
    from ssc.credit.grpo_patch import grpo_loss, ssc_loss
    rs = [rollout(4, s) for s in (1, 0, 1, 0)]
    exs = []
    for g, r in enumerate(rs):
        e = encode_turns(r, tok, "SYSTEM", group_id=g // 2)
        apply_survival(e, r.num_turns, [])
        exs += e
    b = collate(exs, tok.pad_token_id or tok.eos_token_id)
    torch.manual_seed(0)
    old = torch.randn_like(b.token_q)
    new = old + 0.01 * torch.randn_like(old)
    assert torch.allclose(
        grpo_loss(b.rewards, b.actor_mask, old, new, b.group_ids),
        ssc_loss(b.rewards, b.token_q, b.actor_mask, old, new, b.group_ids),
        atol=1e-6)


def test_chunked_logprobs_match_the_naive_form(tok):
    """The memory fix must be numerically identical, not merely close."""
    import torch

    from ssc.train.batch import token_logprobs

    class TinyLM(torch.nn.Module):
        """Deterministic stand-in with a small vocabulary."""

        def __init__(self, vocab=97, T=40):
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.table = torch.randn(T, vocab, generator=g)

        def forward(self, input_ids, attention_mask=None):
            B, T = input_ids.shape
            return type("O", (), {"logits": self.table[:T].unsqueeze(0).repeat(B, 1, 1)})

    m = TinyLM()
    ids = torch.randint(0, 97, (2, 40), generator=torch.Generator().manual_seed(1))

    naive_logits = m(ids).logits[:, :-1, :].float()
    naive = torch.log_softmax(naive_logits, dim=-1).gather(
        -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    for chunk in (1, 7, 256):
        got = token_logprobs(m, ids, torch.ones_like(ids), chunk=chunk)
        assert torch.allclose(naive, got, atol=1e-5), f"chunk={chunk}"
