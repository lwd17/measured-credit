"""The chunked-head path must equal the naive full-vocabulary form.

`token_logprobs` stopped materialising logits so WebShop's 7813-token samples
would fit beside the sampler. That is a memory change, not a maths change, so
the guarantee to keep is exact agreement with `log_softmax(logits).gather(...)`
-- in the value AND in the gradient that reaches the LoRA parameters, since the
gradient is the only thing training consumes.
"""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from ssc.train.batch import _causal_lm_parts, token_logprobs


def _tiny():
    cfg = AutoConfig.for_model(
        "qwen3", vocab_size=997, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128, tie_word_embeddings=False)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()


def _reference(model, ids, mask):
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1, :].float()
    return torch.log_softmax(logits, -1).gather(
        -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def test_matches_naive_log_softmax():
    model = _tiny()
    ids = torch.randint(0, 997, (3, 37))
    mask = torch.ones_like(ids)
    mask[2, :5] = 0
    with torch.no_grad():
        got, want = token_logprobs(model, ids, mask, chunk=8), _reference(model, ids, mask)
    assert got.shape == want.shape == (3, 36)
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()


def test_head_is_actually_split_out():
    """If the split silently failed we would be testing the old path twice."""
    body, head = _causal_lm_parts(_tiny())
    assert head.out_features == 997
    assert not hasattr(body, "lm_head")


def test_gradient_matches_through_checkpoint():
    model = _tiny()
    ids = torch.randint(0, 997, (2, 33))
    mask = torch.ones_like(ids)
    grads = []
    for fn in (token_logprobs, lambda m, i, a, chunk=0: _reference(m, i, a)):
        model.zero_grad(set_to_none=True)
        fn(model, ids, mask, chunk=8).sum().backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()
                                if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], atol=1e-4), \
        (grads[0] - grads[1]).abs().max()


def test_peft_wrapped_model_keeps_adapters_in_the_graph():
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(_tiny(), LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    ids = torch.randint(0, 997, (2, 21))
    mask = torch.ones_like(ids)
    token_logprobs(model, ids, mask, chunk=8).sum().backward()
    lora = [p for n, p in model.named_parameters() if "lora_" in n]
    assert lora and all(p.grad is not None for p in lora)
    assert any(p.grad.abs().sum() > 0 for p in lora)


def test_chunks_bound_the_padded_width_not_the_ragged_sum():
    """A chunk is collated to its longest member, so THAT is what must fit."""
    import types

    from ssc.train.step import plan_chunks

    ex = [types.SimpleNamespace(input_ids=[0] * n)
          for n in [100] * 30 + [900, 1000]]
    for chunk in plan_chunks(ex, token_budget=1000):
        width = max(len(ex[i].input_ids) for i in chunk)
        assert len(chunk) * width <= 1000 or len(chunk) == 1, \
            f"chunk of {len(chunk)} padded to {width} = {len(chunk) * width}"
    assert sum(len(c) for c in plan_chunks(ex, 1000)) == len(ex)


def test_wide_batches_still_match_the_naive_form():
    """`chunk` counts tokens across the batch, so a wide batch slices to one
    row at a time -- the path a WebShop opening-turn micro-batch takes."""
    model = _tiny()
    ids = torch.randint(0, 997, (8, 19))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        got, want = token_logprobs(model, ids, mask, chunk=4), _reference(model, ids, mask)
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()


def test_update_refuses_an_eval_mode_model():
    """The failure this guards is silent: identical gradients, 7x the memory."""
    import pytest

    from ssc.train.step import run_update

    model = _tiny().eval()
    with pytest.raises(RuntimeError, match="eval mode"):
        run_update(model, None, [], [], [[0]], collate=None, pad=0,
                   token_logprobs=None, grpo_loss=None, clip_eps=0.2,
                   max_grad_norm=1.0)


def test_webshop_offpath_fills_only_silent_failures():
    """Deletion is 0-0 on a failure; a winner's pages are what is free to use."""
    from ssc.credit.webshop_cf import _fill_offpath

    # rollout 0 wins, 1 fails and measured nothing, 2 fails but DID measure.
    out = [[0.5, 0.0], [0.0, 0.0], [0.0, 0.3]]
    obs = [["Page A", "Page B"], ["Page A", "Page Z"], ["Page Q", "Page Q"]]
    _fill_offpath(out, obs, bases=[1.0, 0.0, 0.0])
    assert out[0] == [0.5, 0.0], "a winner is never overwritten"
    assert out[2] == [0.0, 0.3], "a real measurement is never overwritten"
    # turn 0 of the silent failure: pages A (on-path) and Z (off) -> 1/2, then 0.
    assert out[1] == [0.5, 0.0], out[1]


def test_webshop_offpath_is_a_noop_without_a_winner():
    from ssc.credit.webshop_cf import _fill_offpath

    out = [[0.0], [0.0]]
    _fill_offpath(out, [["P"], ["P"]], bases=[0.0, 0.0])
    assert out == [[0.0], [0.0]]


def test_structurally_required_turns_are_neutral_not_zero():
    """A turn whose ablation returns exactly 0 has delta == the episode return.

    That is A_E again, sitting on the last turn of every trajectory -- a position
    prior, not a placement measurement. Keeping it taught the WebShop policy to
    buy the first item it opened (7.4 -> 3.0 turns). But zeroing it taught the
    opposite: the step term is centred within the trajectory, so a zero among
    positive deltas is BELOW the row mean and reads as a penalty, and the policy
    stopped buying at all (12.7 turns, score 0.544 -> 0.258). The value that
    centres to exactly zero is the row's own mean over the turns that WERE
    measured.
    """
    import statistics

    from ssc.credit.alf_arms import centre_within_trajectory
    from ssc.credit.webshop_cf import webshop_deltas

    class Worker:
        """Full script scores 0.4; dropping the commit or the search scores 0;
        dropping the option click scores 0.3."""

        def replay(self, session, actions):
            if len(actions) == 3:
                return 0.4
            missing = [a for a in ("search[x]", "click[blue]", "click[buy now]")
                       if a not in actions][0]
            return 0.3 if missing == "click[blue]" else 0.0

    acts = [["search[x]", "click[blue]", "click[buy now]"]]
    obs = [["Search", "Item page", "Item page"]]
    raw = webshop_deltas(Worker(), [0], acts, obs, fill_offpath=False,
                         drop_required=False).deltas[0]
    kept = webshop_deltas(Worker(), [0], acts, obs, fill_offpath=False,
                          drop_required=True).deltas[0]
    assert raw == [0.4, pytest.approx(0.1), 0.4], raw
    # The required turns are MARKED, not given a substitute value: a mask is the
    # only way the difference between "no counterfactual exists" and "the
    # counterfactual returned zero" survives a z-score, and it was that
    # difference that taught two policies to stop terminating.
    cr = webshop_deltas(Worker(), [0], acts, obs, fill_offpath=False,
                        drop_required=True)
    assert cr.mask == [[False, True, False]], cr.mask
    centred = centre_within_trajectory(cr.deltas, cr.mask)[0]
    assert centred[0] == 0.0 and centred[2] == 0.0, centred
    assert statistics.mean([v for v, k in zip(cr.deltas[0], cr.mask[0]) if k]) \
        == pytest.approx(0.1)


def test_webshop_turn_records_the_page_it_acted_from():
    """`state_before` and `clickables` must describe the SAME page.

    The rollout loop overwrites `st` with the result of the step, so reading
    `st["clickables"]` after it recorded the NEXT page's list against this
    page's text. `_webshop_history` pairs those two fields to rebuild the
    prompt, so every WebShop training context was a page shown with the wrong
    clickable list -- a prompt that never existed, which is exactly what that
    function exists to prevent.
    """
    from ssc.detector.rollout import Rollout
    from ssc.train.batch import _webshop_history

    pages = [("Search page", ["search"], True),
             ("Results: A, B", ["back to search", "a", "b"], False),
             ("Item A page", ["back to search", "blue", "buy now"], False)]
    roll = Rollout(task_id="t", question="q", turns=[])
    from ssc.detector.rollout import Turn

    for i, (page, clicks, has) in enumerate(pages):
        roll.turns.append(Turn(
            index=i, text=f"Action: click[x{i}]", tool_name="webshop",
            tool_args={"action": f"click[x{i}]", "state_before": page,
                       "clickables": clicks, "has_search": has,
                       "step_shown": i + 1, "step_budget": 15},
            observation="next"))

    msgs = _webshop_history(roll, upto=len(pages), system_prompt="sys")
    users = [m["content"] for m in msgs if m["role"] == "user"]
    assert len(users) == 3, users
    for (page, clicks, _), text in zip(pages, users):
        assert page in text, (page, text[:200])
        for c in clicks:
            assert c in text, (c, text[:200])
    # the item page's options must not leak into the results page's prompt
    assert "buy now" not in users[1], users[1][:200]
