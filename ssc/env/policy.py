"""Multi-turn search policy rollout (runbook sections 16, 20, 27).

One ACTION TURN = one policy generation. A turn either calls the search tool
(and receives an observation) or produces the final answer. This is the unit
that section 5 indexes, that section 6 assigns survival coefficients to, and
that the detector numbers -- so it is defined in exactly one place, here.

Section 20 collects DEFAULT rollouts: no SSC, no RL modification, no
counterfactual branches. Nothing in this file may depend on the reference
answer; scoring happens afterwards, in the trainer, never during generation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from openai import OpenAI

from ssc.detector.rollout import Outcome, Rollout, Turn
from ssc.env.search_env import PROMPTS, SEARCH_TOOL, SYSTEM_PROMPT, BM25SearchEnv

ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(.+)$", re.I | re.M)


def extract_answer(content: str) -> tuple[str, bool]:
    """Return (answer, format_ok).

    33% of rollouts do not emit the requested `Answer: X` line. Falling back to
    the WHOLE generation makes exact match fail by construction, which silently
    depresses the success rate -- and M_inv^+ is defined only over successful
    rollouts, so that bug would have propagated straight into Gate A'.

    The fallback is the last non-empty line, which is where a free-form answer
    almost always sits. `format_ok` is returned rather than swallowed so format
    non-compliance stays visible as its own quantity.
    """
    body = content.split("</think>")[-1]
    m = None
    for m in ANSWER_RE.finditer(body):
        pass                      # keep the LAST "Answer:" line, not the first
    if m:
        return m.group(1).strip(), True
    lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    return (lines[-1] if lines else body.strip()), False


@dataclass
class PolicyConfig:
    model: str
    max_turns: int = 12                # section 27: identical across methods
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens_per_turn: int = 2048
    request_timeout: float = 600.0
    max_transient_retries: int = 3
    prompt: str = "default"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class SearchPolicy:
    def __init__(self, env: BM25SearchEnv, cfg: PolicyConfig, base_url: str):
        self.env, self.cfg = env, cfg
        self.client = OpenAI(base_url=base_url, api_key="EMPTY",
                             timeout=cfg.request_timeout)

    def rollout(self, task, rollout_index: int = 0, seed: int = 0) -> Rollout:
        t0 = time.time()
        roll = Rollout(task_id=task.task_id, question=task.question,
                       rollout_index=rollout_index, policy_model=self.cfg.model)
        messages = [{"role": "system", "content": PROMPTS[self.cfg.prompt]},
                    {"role": "user", "content": task.question}]
        attempt = 0

        while len(roll.turns) < self.cfg.max_turns:
            try:
                r = self.client.chat.completions.create(
                    model=self.cfg.model, messages=messages, tools=[SEARCH_TOOL],
                    temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                    max_tokens=self.cfg.max_tokens_per_turn,
                    seed=seed + attempt,
                )
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                if any(k in msg for k in ("context length", "maximum context",
                                          "reduce the length")):
                    roll.stop_reason = "context_limit"
                    break
                # A tool-parser 500 or a transient network fault must not be
                # recorded as the policy choosing to stop; that would make
                # trajectory length partly a measure of infrastructure.
                if attempt < self.cfg.max_transient_retries:
                    attempt += 1
                    continue
                roll.stop_reason = f"api_error:{type(e).__name__}"
                break
            attempt = 0

            msg_obj = r.choices[0].message
            reasoning = getattr(msg_obj, "reasoning_content", None) or ""
            content = msg_obj.content or ""
            calls = msg_obj.tool_calls or []
            n_tok = r.usage.completion_tokens if r.usage else 0

            # The action text is what the policy generated, reasoning included:
            # section 9 weights generated tokens, and the detector needs the
            # stated hypothesis to identify a regime at all.
            action_text = (f"<think>{reasoning}</think>\n{content}"
                           if reasoning else content)

            if not calls:
                roll.turns.append(Turn(index=len(roll.turns), text=action_text,
                                       n_generated_tokens=n_tok))
                ans, fmt_ok = extract_answer(content)
                roll.final_answer = ans[:500]
                roll.format_ok = fmt_ok
                roll.stop_reason = "answered"
                break

            tc = calls[0]
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            queries = args.get("query", [])
            if isinstance(queries, str):
                queries = [queries]

            result = self.env.search(queries)
            observation = self.env.format_observation(result)
            roll.turns.append(Turn(index=len(roll.turns), text=action_text,
                                   tool_name="search", tool_args={"query": queries},
                                   observation=observation, n_generated_tokens=n_tok))

            messages.append({"role": "assistant", "content": content,
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.function.name,
                                                          "arguments": tc.function.arguments}}]})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": observation[:8000]})
        else:
            roll.stop_reason = "max_turns"

        roll.outcome = Outcome()          # filled in by the scorer, not here
        roll.wall_time = round(time.time() - t0, 1)
        return roll
