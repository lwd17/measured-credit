"""The parts of a training step that no environment can change.

Extracted so a second environment does not arrive as a second copy of the
update loop. Everything here was paid for on ALFWorld and the comments record
what it cost:

  token-budget micro-batching   a fixed example count died with
                                `OutOfMemoryError: tried to allocate 5.25 GiB`
                                on the one chunk that drew four long turns.
  actor-token weighting         dividing by chunk COUNT gave chunks 231 to 3497
                                actor tokens the same weight -- a 15x swing,
                                correlated +0.49 with length rank, which
                                applied a turn-position prior to every arm
                                including the baseline. Turn position alone
                                tracks causal progress at rho = +0.464, so the
                                baseline was quietly receiving the signal the
                                experiment exists to isolate.
  per-example OOM retry         an OOM must not end a multi-hour unattended run,
                                and a step that silently trained on half its
                                data must not look like a normal one, so drops
                                are counted and reported.
"""

from __future__ import annotations

import os

import torch


def plan_chunks(examples, token_budget: int) -> list[list[int]]:
    """Group example indices into micro-batches bounded by INPUT tokens.

    Activation memory tracks tokens, not examples, so bounding tokens bounds the
    peak. Sorting by length also keeps padding waste down, since a chunk is
    padded to its longest member.
    """
    order = sorted(range(len(examples)), key=lambda i: len(examples[i].input_ids))
    chunks, cur = [], []
    for i in order:
        n_tok = len(examples[i].input_ids)
        # `order` ascends, so an arriving example is the longest in the chunk
        # and therefore sets the padded width for every member of it. Bounding
        # the SUM of unpadded lengths bounded a quantity that is not allocated:
        # on WebShop the 36 shortest turns summed to the 16384 budget and then
        # collated to 36 x 1069 = 38484, a 2.35x overrun that peaked at 49.5 GB
        # and dropped all 362 examples of the step. ALFWorld's turn lengths sit
        # close enough together that the same overrun stayed under the card and
        # the bug survived every arm of that experiment.
        if cur and (len(cur) + 1) * n_tok > token_budget:
            chunks.append(cur)
            cur = []
        cur.append(i)
    if cur:
        chunks.append(cur)
    return chunks


def run_update(model, optimizer, examples, adv_values, chunks, *,
               collate, pad, token_logprobs, grpo_loss, clip_eps: float,
               max_grad_norm: float) -> tuple[float, float, int]:
    """One optimizer step over all chunks. Returns (loss, grad_norm, n_dropped).

    The per-chunk loss is a token-mean over ITS OWN actor tokens, so the step
    objective is the token-mean over all of them only if each chunk is weighted
    by its share of actor tokens.
    """
    # An eval-mode model disables gradient checkpointing silently: same
    # gradients at lora_dropout=0, 7x the activations, and the only symptom is
    # an OOM that looks like a batch-size problem. WebShop lost every example
    # of every step to it for a day. A third environment will not.
    if not model.training:
        raise RuntimeError(
            "model is in eval mode: gradient checkpointing is inactive and "
            "activations are ~7x larger. Call model.train() after get_peft_model().")
    chunk_tokens = [sum(examples[i].n_actor_tokens for i in idx) for idx in chunks]
    total_tokens = max(sum(chunk_tokens), 1)
    optimizer.zero_grad(set_to_none=True)
    total_loss, n_dropped = 0.0, 0

    def run(indices, weight) -> float:
        b = collate([examples[i] for i in indices], pad).to(model.device)
        adv = torch.tensor([adv_values[i] for i in indices],
                           dtype=torch.float32, device=model.device)
        # One gradient step per batch of rollouts, so the behaviour policy IS
        # the current policy: old_lp equals new_lp at these parameters and the
        # ratio is exactly 1. Recomputing it would be the same number at more
        # cost; this only holds while the loop does a single inner epoch.
        new_lp = token_logprobs(model, b.input_ids, b.attention_mask)
        old_lp = new_lp.detach()
        loss = grpo_loss(b.rewards, b.actor_mask, old_lp, new_lp,
                         b.group_ids, clip_eps, advantages=adv)
        (loss * weight).backward()
        return float(loss.detach())

    first_error: str | None = None
    # A step that drops every chunk names the first failure but not the memory
    # state that produced it, and WebShop reported "tried to allocate 1.76 GiB"
    # on a card whose measured peak for a full chunk is 25 GB -- a gap only a
    # per-chunk trace can close. Off by default; costs a synchronise when on.
    dbg = int(os.environ.get("SSC_MEM_DEBUG", "0"))
    for n_chunk, (idx, n_tok) in enumerate(zip(chunks, chunk_tokens)):
        w = n_tok / total_tokens
        if dbg and n_chunk < dbg:
            lens = [len(examples[i].input_ids) for i in idx]
            print(f"      chunk {n_chunk}: n={len(idx)} max_len={max(lens)} "
                  f"alloc={torch.cuda.memory_allocated()/2**30:.1f}G "
                  f"reserved={torch.cuda.memory_reserved()/2**30:.1f}G "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}G",
                  flush=True)
        try:
            total_loss += run(idx, w) * w
        except Exception as e:  # noqa: BLE001
            # Catching only OutOfMemoryError hid a plain RuntimeError behind a
            # count of "dropped" examples: every chunk failed, |g| came back
            # 0.000, and the log said OOM. Whatever the failure is, the FIRST
            # one is reported, so a broken step names its cause instead of
            # looking like a memory problem.
            if first_error is None:
                first_error = f"{type(e).__name__}: {e}"
            torch.cuda.empty_cache()
            for i in idx:
                try:
                    total_loss += run([i], examples[i].n_actor_tokens / total_tokens) \
                                  * (examples[i].n_actor_tokens / total_tokens)
                except Exception:  # noqa: BLE001
                    torch.cuda.empty_cache()
                    n_dropped += 1
    if first_error and n_dropped:
        print(f"    update failures ({n_dropped} dropped); first: {first_error}",
              flush=True)

    gn = float(torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_grad_norm))
    optimizer.step()
    return total_loss, gn, n_dropped
