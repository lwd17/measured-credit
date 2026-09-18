"""Rollout -> training tensors (runbook sections 7, 9, 53, 55).

## Why this is per-TURN and not one flat sequence

Qwen3's chat template strips `<think>` blocks from every assistant message
except the most recent one. A flat encoding of a whole trajectory therefore
cannot represent the training data at all:

* the reasoning tokens the policy actually generated at turn t vanish from the
  sequence for every t except the last, so the actor mask covers a fraction of
  what was generated;
* `Turn.n_generated_tokens` (reported by vLLM, reasoning included) then
  disagrees with the encoded actor-token count, which silently makes M_inv^+
  and the loss measure different quantities.

Neither failure raises. Both were found by rendering the real template rather
than the stub used in unit tests.

The rollout adapters append only `content` (no reasoning) to the running
message list, so at inference turn t conditions on a history with earlier
thinking absent. The faithful training example is therefore

    context    = the message history exactly as turn t saw it
    generation = that turn's <think> + action

which is one example PER TURN. Each carries its rollout's sequence advantage
and its own survival coefficient q_t -- a scalar, since a turn is the unit
sections 5-7 index.

Cost: an n-turn trajectory re-encodes growing prefixes, so O(n^2) tokens. That
is the price of the training sequence matching the inference sequence, and it
is paid deliberately.
"""

from __future__ import annotations

import re

from dataclasses import dataclass

import torch
import torch.utils.checkpoint

from ssc.credit.survival import build_turn_survival
from ssc.detector.rollout import Rollout
from ssc.detector.schema import SemanticEvent


@dataclass
class TurnExample:
    """One (context, generation) pair: the training unit."""

    input_ids: list[int]
    actor_mask: list[int]        # 1 only on tokens this turn generated
    turn_index: int
    reward: float
    group_id: int
    task_id: str
    rollout_index: int
    q: float = 1.0               # survival coefficient for this turn

    @property
    def n_actor_tokens(self) -> int:
        return sum(self.actor_mask)


def _history(rollout: Rollout, upto: int, system_prompt: str,
             env_kind: str) -> list[dict]:
    """Message list as the policy saw it immediately before generating turn `upto`."""
    first_obs = rollout.turns[0].observation if rollout.turns else ""
    if env_kind == "alfworld":
        return _alfworld_history(rollout, upto, system_prompt)
    if env_kind == "webshop":
        return _webshop_history(rollout, upto, system_prompt)
    if env_kind == "multihop":
        return _multihop_history(rollout, upto, system_prompt)
    if env_kind == "scienceworld":
        opening = f"Task: {rollout.question}\n\nObservation:\n{first_obs}"
    else:
        opening = rollout.question

    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": opening}]
    for t in rollout.turns[:upto]:
        # Only `content` was appended at rollout time -- reasoning was never in
        # the context. Reconstructing it here would train on a history the
        # policy never conditioned on.
        msgs.append({"role": "assistant", "content": _content_only(t.text)})
        if t.observation:
            msgs.append({"role": "user", "content": f"Observation:\n{t.observation}"})
    return msgs


def _multihop_history(rollout: Rollout, upto: int, system_prompt: str
                      ) -> list[dict]:
    """Replay the stored messages verbatim.

    Every turn records the user message it was shown, so this reassembles
    nothing and cannot drift from what was displayed. WebShop stored the pieces
    instead and one of them was captured a step late, which made every training
    prompt a page paired with the next page's options -- the failure this
    reconstruction is written to be incapable of.
    """
    msgs = [{"role": "system", "content": system_prompt}]
    for i, t in enumerate(rollout.turns[:max(upto, 0) + 1]):
        args = t.tool_args or {}
        msgs.append({"role": "user", "content": str(args.get("shown", ""))})
        if i < upto:
            msgs.append({"role": "assistant", "content": _content_only(t.text)})
    return msgs


def _webshop_history(rollout: Rollout, upto: int, system_prompt: str) -> list[dict]:
    """WebShop's context, rebuilt from what the policy was actually shown.

    Every WebShop user message lists the clickable items on the page the turn
    was generated from, and on the search page it says a search is available.
    That list is most of the prompt and it changes every turn, so a generic
    reconstruction would train against a prompt that never existed -- the same
    trap `_alfworld_history` exists to avoid, where the omitted admissible list
    silently changed every context in the batch.

    The adapter records the clickables and the page it acted from on each turn,
    so the message is regenerated through the builder the rollout loop used.
    """
    from ssc.env.webshop_env import build_user_message

    def shown(turn, index: int) -> str:
        args = turn.tool_args or {}
        page = str(args.get("state_before") or "")
        clickables = list(args.get("clickables") or [])
        # Prefer the flag the rollout recorded; infer it only for rollouts
        # collected before that field existed. Inferring it from the clickables
        # was doubly wrong while those clickables came from the next page.
        if "has_search" in args:
            has_search = bool(args["has_search"])
        else:
            has_search = ("search" in [c.lower() for c in clickables]
                          or not clickables)
        return build_user_message(page, clickables, has_search,
                                  int(args.get("step_shown", index + 1)) - 1,
                                  int(args.get("step_budget", 15)))

    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": shown(rollout.turns[0], 0)}]
    for i, t in enumerate(rollout.turns[:upto]):
        msgs.append({"role": "assistant", "content": _content_only(t.text)})
        if i + 1 < len(rollout.turns):
            msgs.append({"role": "user",
                         "content": shown(rollout.turns[i + 1], i + 1)})
    return msgs


def _alfworld_history(rollout: Rollout, upto: int, system_prompt: str) -> list[dict]:
    """ALFWorld's context, rebuilt from what the policy was actually shown.

    Every ALFWorld user message carries the admissible-action list for the state
    the turn was generated from -- 18 to 30 lines of it, the bulk of the prompt.
    The generic reconstruction above omits it, so training would optimise the
    policy against a prompt that never existed at rollout time, on every turn.
    The list is recorded per turn by the adapter and replayed here through the
    same builder the rollout loop uses.

    The window is applied exactly as the rollout loop applied it: the opening
    two messages are kept and the tail truncated. A different window would move
    the boundary between what the policy saw and what it did not.
    """
    from ssc.env.alfworld_env import build_user_message

    def shown(turn, index: int) -> str:
        args = turn.tool_args or {}
        prev_obs = (rollout.turns[index - 1].observation if index > 0
                    else _alfworld_opening_observation(rollout))
        return build_user_message(prev_obs, list(args.get("admissible") or []),
                                  int(args.get("step_shown", index)),
                                  _alfworld_budget(rollout))

    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user",
             "content": f"Task: {rollout.question}\n\n" + shown(rollout.turns[0], 0)}]
    for i, t in enumerate(rollout.turns[:upto]):
        msgs.append({"role": "assistant", "content": _content_only(t.text)})
        if i + 1 < len(rollout.turns):
            msgs.append({"role": "user", "content": shown(rollout.turns[i + 1], i + 1)})
    if len(msgs) > 17:
        msgs = msgs[:2] + msgs[-14:]
    return msgs


def _alfworld_opening_observation(rollout: Rollout) -> str:
    """The room description the first turn was generated from.

    Turn 0's `observation` is the RESULT of turn 0's action, so the opening
    state is not recoverable from the turn list; the adapter records it on turn
    0 instead. Rollouts collected before that was added fall back to the empty
    string, which costs the first turn of such a trajectory a faithful prompt.
    """
    if not rollout.turns:
        return ""
    return str((rollout.turns[0].tool_args or {}).get("opening_obs", ""))


def _alfworld_budget(rollout: Rollout) -> int:
    """The `(step k/N)` denominator the policy saw, recovered from the record."""
    for t in rollout.turns:
        args = t.tool_args or {}
        if "step_budget" in args:
            return int(args["step_budget"])
    return max(len(rollout.turns), 1)


def _content_only(text: str) -> str:
    """Strip the reasoning block, leaving what the adapter put in the context."""
    return text.split("</think>")[-1].lstrip("\n")


def _split_think(text: str):
    """Split the "<think>R</think>\nC" text the policy emits back into
    (reasoning, content).

    Returns (None, text) when there is no think block. policy.py uses that
    format to join the reasoning and the body into a single action string; the
    training side must split them again and hand each to the template
    separately, or Qwen3 wraps another empty think block around the whole thing.
    """
    m = re.match(r"\s*<think>(.*?)</think>\s*(.*)\Z", text, re.S)
    if not m:
        return None, text
    return m.group(1).strip("\n"), m.group(2)


def _accepts_thinking(tokenizer) -> bool:
    """Qwen3's chat template takes `enable_thinking`; other templates and the
    test doubles do not, and passing it to them raises."""
    tpl = getattr(tokenizer, "chat_template", None)
    return isinstance(tpl, str) and "enable_thinking" in tpl


def encode_turns(rollout: Rollout, tokenizer, system_prompt: str,
                 env_kind: str = "scienceworld", max_length: int = 16384,
                 group_id: int = 0) -> list[TurnExample]:
    """One TurnExample per action turn.

    `enable_thinking` must match what the sampler used, per turn, or the
    response span stops being the tokens the policy actually generated.
    Qwen3's template inserts an empty `<think>\n\n</think>\n\n` into an
    assistant message that has no think block; with `enable_thinking=True`
    (the tokenizer default) those four tokens land AFTER the generation prompt
    and are counted as generated. WebShop and multi-hop sample with
    `enable_thinking=False`, where vLLM puts that same block INSIDE the prompt
    -- so those runs were giving four phantom tokens per turn the full turn
    advantage, on actions only 10-20 tokens long. ALFWorld samples with
    thinking on and is unaffected: the template re-renders the real think block
    to exactly the sampled text.

    Deciding per turn from the text (rather than a flag) keeps a mixed rollout
    honest and needs no plumbing through three trainers.
    """
    out: list[TurnExample] = []
    reward = float(rollout.outcome.reward or 0.0)

    for t in rollout.turns:
        ctx = _history(rollout, t.index, system_prompt, env_kind)
        reasoning, content = _split_think(t.text or "")
        thinking = reasoning is not None
        # The fake tokenizer used in the tests has no such keywords; only
        # Qwen3's template understands them.
        if _accepts_thinking(tokenizer):
            kw = {"enable_thinking": thinking}
            msg = {"role": "assistant", "content": content}
            if thinking:
                # The template reads the reasoning_content field. Leaving
                # <think> inside content makes it wrap another empty think
                # block, so the training sequence carries 4 tokens more than the
                # sampled one.
                msg["reasoning_content"] = reasoning
        else:
            kw = {}
            msg = {"role": "assistant", "content": t.text}
        ctx_ids = tokenizer.apply_chat_template(
            ctx, tokenize=True, add_generation_prompt=True, **kw)

        # The generation is encoded as the assistant message it was, so the
        # special tokens around it match training-time rendering exactly.
        full_ids = tokenizer.apply_chat_template(
            ctx + [msg], tokenize=True, add_generation_prompt=False, **kw)

        # Sampling stops at <|im_end|>, but the template appends one more "\n".
        # Counting it as part of the response span would make the policy carry
        # the advantage for a token it never generated.
        stops = {getattr(tokenizer, "eos_token_id", None)}
        conv = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(conv):
            try:
                stops.add(conv("<|im_end|>"))
            except Exception:  # noqa: BLE001 -- a test double may not know this token
                pass
        stops.discard(None)
        if stops:
            while len(full_ids) > len(ctx_ids) and full_ids[-1] not in stops:
                full_ids = full_ids[:-1]

        if len(full_ids) <= len(ctx_ids):
            # Template dropped the generation entirely (empty turn); nothing to
            # train on, and silently emitting a zero-actor example would dilute
            # the token mean.
            continue

        input_ids = full_ids
        actor_mask = [0] * len(ctx_ids) + [1] * (len(full_ids) - len(ctx_ids))

        if len(input_ids) > max_length:
            cut = len(input_ids) - max_length
            input_ids = input_ids[cut:]
            actor_mask = actor_mask[cut:]
            if not any(actor_mask):
                continue          # truncation removed the generation itself

        out.append(TurnExample(
            input_ids=input_ids, actor_mask=actor_mask, turn_index=t.index,
            reward=reward, group_id=group_id, task_id=rollout.task_id,
            rollout_index=rollout.rollout_index))
    return out


def apply_survival(examples: list[TurnExample], num_turns: int,
                   events: list[SemanticEvent], hard: bool = False,
                   mass_preserving: bool = True) -> None:
    """Attach q_t to each turn example, in place (sections 6, 52, 53).

    Mass preservation is applied at ROLLOUT scope here, and this is load-bearing.
    The batched helper in grpo_patch normalises PER ROW, which was correct when
    a row was a whole trajectory. With one row per turn, a fully invalidated
    turn has q = 0 across its entire row, the row mean is 0, the degenerate-row
    fallback restores uniform weights -- and SSC silently cancels itself out.

    Normalising across a rollout's turn examples, token-weighted, keeps
    section 53's property (mean actor weight = 1 over the trajectory) while
    preserving the RELATIVE down-weighting that is the whole method.

    Callers must therefore pass `mass_preserving=False` to `ssc_loss`, since the
    normalisation has already happened here.
    """
    turn_q = build_turn_survival(num_turns, events, hard=hard)
    for ex in examples:
        if not 0 <= ex.turn_index < len(turn_q):
            raise ValueError(
                f"turn example index {ex.turn_index} outside 0..{len(turn_q)-1}; "
                "a turn with no survival coefficient would get full credit")
        ex.q = turn_q[ex.turn_index]

    if mass_preserving:
        normalize_rollout_mass(examples)


def normalize_rollout_mass(examples: list[TurnExample], eps: float = 1e-8) -> None:
    """Section 53 at trajectory scope: token-weighted mean actor weight -> 1.

    A trajectory whose every turn was invalidated has no mass to redistribute;
    it falls back to uniform weights, matching `normalize.mass_preserve`.
    """
    total_tokens = sum(e.n_actor_tokens for e in examples)
    if total_tokens == 0:
        return
    weighted = sum(e.q * e.n_actor_tokens for e in examples)
    mean_q = weighted / total_tokens
    if mean_q < eps:
        for e in examples:
            e.q = 1.0
        return
    for e in examples:
        e.q = e.q / mean_q


@dataclass
class TrainBatch:
    input_ids: torch.Tensor      # (B, T)
    attention_mask: torch.Tensor
    actor_mask: torch.Tensor     # (B, T-1) at prediction positions
    token_q: torch.Tensor        # (B, T-1)
    rewards: torch.Tensor        # (B,)
    group_ids: torch.Tensor      # (B,)

    def to(self, device) -> "TrainBatch":
        return TrainBatch(*[getattr(self, f).to(device) for f in
                            ("input_ids", "attention_mask", "actor_mask",
                             "token_q", "rewards", "group_ids")])


def collate(examples: list[TurnExample], pad_token_id: int) -> TrainBatch:
    """Right-pad into a batch, shifting masks to prediction positions.

    Logits at position j predict token j+1, so the weight for generated token j
    lives at index j-1. The shift happens here, once, next to the padding logic
    so the two cannot drift apart.
    """
    B = len(examples)
    T = max(len(e.input_ids) for e in examples)

    input_ids = torch.full((B, T), pad_token_id, dtype=torch.long)
    attention = torch.zeros((B, T), dtype=torch.long)
    a_mask = torch.zeros((B, T - 1), dtype=torch.long)
    q = torch.zeros((B, T - 1), dtype=torch.float)

    for i, e in enumerate(examples):
        n = len(e.input_ids)
        input_ids[i, :n] = torch.tensor(e.input_ids, dtype=torch.long)
        attention[i, :n] = 1
        shifted = e.actor_mask[1:]
        a_mask[i, : n - 1] = torch.tensor(shifted, dtype=torch.long)
        # q is constant across a turn's generated tokens and 0 elsewhere, so it
        # is the mask scaled -- never a separate source of truth about which
        # tokens are actor tokens.
        q[i, : n - 1] = torch.tensor(shifted, dtype=torch.float) * e.q

    return TrainBatch(
        input_ids=input_ids, attention_mask=attention, actor_mask=a_mask,
        token_q=q,
        rewards=torch.tensor([e.reward for e in examples], dtype=torch.float),
        group_ids=torch.tensor([e.group_id for e in examples], dtype=torch.long),
    )


def _causal_lm_parts(model):
    """(body, lm_head) for a possibly PEFT-wrapped causal LM, else None.

    PEFT forwards unknown attributes to the module it wraps, so
    `getattr(peft_model, "model")` hands back the whole ForCausalLM rather than
    the transformer body -- hence `get_base_model()` first, which peft documents
    as the unwrapping entry point. LoRA targets only the attention and MLP
    projections, never `lm_head`, so splitting here leaves every adapter inside
    the body where the gradient still reaches it.
    """
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    body, head = getattr(m, "model", None), getattr(m, "lm_head", None)
    return None if body is None or head is None else (body, head)


def _naive_logprobs(logits, targets, chunk):
    parts = []
    rows = max(1, chunk // max(1, logits.size(0)))
    for i in range(0, logits.size(1), rows):
        lg = logits[:, i:i + rows, :].float()
        tg = targets[:, i:i + rows]
        parts.append(lg.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
                     - lg.logsumexp(dim=-1))
    return torch.cat(parts, dim=1) if parts else logits.new_zeros(
        (logits.size(0), 0))


def token_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                   chunk: int = 1024) -> torch.Tensor:
    """log p(token_j | prefix) at prediction positions -> (B, T-1).

    Computed WITHOUT materialising a full-vocabulary log_softmax. Qwen3's
    vocabulary is 151,936, so at a 10k sequence the naive form allocates

        logits.float()            10k x 152k x 4B  = 6.1 GB
        log_softmax(logits)       another          = 6.1 GB
        + autograd graph for both

    on top of a ~60 GB model, and every arm of the experiment died with
    `torch.OutOfMemoryError` inside exactly these two lines.

    The identity used instead is

        log p(x) = logit[x] - logsumexp(logits)

    which needs only the target logit and a reduction. Slicing that reduction
    was NOT enough by itself: autograd retains every fp32 slice it is asked to
    differentiate, so the chunk loop still held `B x T x V x 4B` until backward
    ran -- 9.5 GB at WebShop's 90th-percentile 7813-token sample, on top of the
    4.75 GB the model's own bf16 logits already cost. ALFWorld's ~2000-token
    turns paid the same tax at a quarter the size and fit; WebShop dropped 144
    of 162 examples per step, and the step blamed a memory error whose real
    driver was the vocabulary dimension, not the batch.

    So the head is applied HERE, one chunk at a time, under a checkpoint. The
    body returns hidden states (B x T x 4096, ~128 MB) and no chunk's logits
    outlive the line consuming them, at the cost of recomputing one projection
    per chunk in backward. Mathematically identical to the naive form, which is
    what the test compares against.

    `chunk` counts TOKENS across the batch, not positions. Counting positions
    made the slice scale with batch width: at a fixed ~12k tokens per
    micro-batch the peak went 23.5 GB at 2 x 6144 to 37.9 GB at 32 x 384,
    because 32 rows x 256 positions is a 4.98 GB fp32 slice where 2 x 256 is
    311 MB. WebShop packs its short opening turns 32 to a micro-batch, so it
    met the worst end of that curve on the very first chunk of every step.
    """
    targets = input_ids[:, 1:]
    parts_of = _causal_lm_parts(model)
    hidden = None
    if parts_of is not None:
        body, head = parts_of
        hidden = getattr(body(input_ids=input_ids, attention_mask=attention_mask),
                         "last_hidden_state", None)
    if hidden is None:
        # A model that does not split this way keeps the previous path, whose
        # peak is the full-logits one: correct, merely larger.
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        return _naive_logprobs(out.logits[:, :-1, :], targets, chunk)

    hidden = hidden[:, :-1, :]

    def _lp(h, tg):
        lg = head(h).float()
        return lg.gather(-1, tg.unsqueeze(-1)).squeeze(-1) - lg.logsumexp(dim=-1)

    parts = []
    rows = max(1, chunk // max(1, hidden.size(0)))
    for i in range(0, hidden.size(1), rows):
        h, tg = hidden[:, i:i + rows, :], targets[:, i:i + rows]
        if torch.is_grad_enabled() and h.requires_grad:
            parts.append(torch.utils.checkpoint.checkpoint(
                _lp, h, tg, use_reentrant=False))
        else:
            parts.append(_lp(h, tg))
    return torch.cat(parts, dim=1) if parts else hidden.new_zeros(
        (input_ids.size(0), 0))
