"""Rollouts on the multi-hop QA environment.

Each turn stores the EXACT user message the policy was shown, verbatim, rather
than the pieces a reconstruction would have to reassemble. WebShop stored the
pieces and one of them -- the clickable list -- was captured after the step
instead of before, so every training prompt paired a page with the next page's
options: a context that never existed, which cost a full day and voided three
arms. A string that is replayed cannot drift from what was displayed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ssc.detector.rollout import Outcome, Rollout, Turn
from ssc.env.multihop_env import (SYSTEM_PROMPT, WIKI_SYSTEM_PROMPT, MultiHopState,
                                  MultiHopTask, Retriever, WikiRetriever,
                                  exact_match, f1)

ACTION_RE = re.compile(r"Action:\s*(search|answer)\[(.*?)\]\s*$", re.I | re.S)


@dataclass
class MultiHopPolicyConfig:
    model: str = "Qwen3-8B"
    max_turns: int = 5
    max_tokens_per_turn: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    max_format_retries: int = 2
    doc_chars: int = 500
    # "local": the question's own ten paragraphs (HotpotQA distractor); "wiki":
    # BM25 top-k over all of Wikipedia (the Search-R1 setting, which uses top-3 and
    # at most 4 turns).
    retriever: str = "local"
    wiki_url: str = "http://127.0.0.1:8080"
    top_k: int = 2
    # Which score the episode reward uses: the Search-R1 setting uses EM, while the
    # distractor line stays on F1.
    reward: str = "f1"
    system_prompt: str | None = None


def extract_action(text: str) -> tuple[str, str] | None:
    """Last line first: the model sometimes reasons before the action line."""
    for cand in (text.strip().split("\n")[-1], text):
        m = ACTION_RE.search(cand.strip())
        if m:
            return m.group(1).lower(), m.group(2).strip()
    return None


class MultiHopPolicy:
    def __init__(self, cfg: MultiHopPolicyConfig, base_url: str):
        from openai import OpenAI

        self.cfg = cfg
        self.client = OpenAI(base_url=base_url, api_key="x")
        self.wiki = (WikiRetriever(cfg.wiki_url, k=cfg.top_k)
                     if cfg.retriever == "wiki" else None)
        self.system_prompt = cfg.system_prompt or (
            WIKI_SYSTEM_PROMPT if cfg.retriever == "wiki" else SYSTEM_PROMPT)

    def make_state(self, task: MultiHopTask) -> MultiHopState:
        r = self.wiki if self.wiki is not None else Retriever(task, k=self.cfg.top_k)
        return MultiHopState(task=task, retriever=r)

    def _extra(self) -> dict:
        return {"chat_template_kwargs": {"enable_thinking": False}}

    def _gen(self, messages, seed: int) -> str:
        r = self.client.chat.completions.create(
            model=self.cfg.model, messages=messages,
            temperature=self.cfg.temperature, top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_tokens_per_turn, seed=seed,
            extra_body=self._extra())
        return (r.choices[0].message.content or "").strip()

    def results_message(self, state: MultiHopState, ids: list,
                        turn: int | None = None) -> str:
        # The turn budget in the open-domain setting is tight (4), so "turn i of n"
        # goes into the observation: only then can the policy plan when to answer.
        # Running out of turns without answering scores 0, the same for every arm.
        head = ("Results:" if turn is None or self.cfg.retriever != "wiki"
                else f"Results (turn {turn} of {self.cfg.max_turns}):")
        if not ids:
            return f"{head}\n(nothing found)"
        parts = []
        for i in ids:
            title, text = state.retriever.doc(i)
            parts.append(f"[{title}] {text[:self.cfg.doc_chars]}")
        return head + "\n" + "\n\n".join(parts)

    def rollout(self, task: MultiHopTask, rollout_index: int = 0,
                seed: int = 0) -> tuple[Rollout, MultiHopState]:
        state = self.make_state(task)
        roll = Rollout(task_id=task.task_id, question=task.question)
        opening = f"Question: {task.question}"
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": opening}]
        shown = opening
        final: str | None = None

        for step in range(self.cfg.max_turns):
            content, act = "", None
            for attempt in range(self.cfg.max_format_retries + 1):
                try:
                    content = self._gen(messages, seed + 100 * attempt)
                except Exception:  # noqa: BLE001
                    break
                act = extract_action(content)
                if act:
                    break
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content":
                     "Reply with exactly one line: Action: search[...] or "
                     "Action: answer[...]."}]
            if not act:
                roll.stop_reason = "format_failure"
                break

            kind, arg = act
            # `shown` is the message that preceded THIS generation, captured
            # before anything below changes it.
            roll.turns.append(Turn(
                index=len(roll.turns), text=content, tool_name="multihop",
                tool_args={"action": f"{kind}[{arg}]", "kind": kind, "arg": arg,
                           "shown": shown, "step_shown": len(roll.turns),
                           "step_budget": self.cfg.max_turns},
                observation=""))
            if kind == "answer":
                final = arg
                break
            ids = state.retriever.search(arg)
            state.queries.append(arg)
            state.retrieved.append(ids)
            shown = self.results_message(state, ids, turn=len(roll.turns) + 1)
            roll.turns[-1].observation = shown
            messages = messages + [{"role": "assistant", "content": content},
                                   {"role": "user", "content": shown}]

        if final is None:
            # Out of turns without answering: score what the evidence supports
            # rather than dropping the rollout, so a group keeps its variance.
            roll.stop_reason = roll.stop_reason or "no_answer"
        golds = task.all_answers()
        f1v = f1(final or "", golds)
        emv = exact_match(final or "", golds)
        reward = emv if self.cfg.reward == "em" else f1v
        roll.outcome = Outcome(success=int(emv), reward=reward, gold_answer=task.answer)
        roll.final_answer = final or ""
        state.final = final or ""
        state.f1, state.em = f1v, emv
        return roll, state
