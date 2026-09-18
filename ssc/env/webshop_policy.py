"""Policy adapter for WebShop, mirroring the ALFWorld one.

Same `Rollout` type and the same rule that a missing Action line is a PARSING
failure rather than a policy decision, so the process pool, the training
encoder and `anchor_cf` all work unchanged.

The clickable items are listed every turn for the reason ALFWorld lists its
admissible commands: without them an 8B spends its budget guessing at exact
product ids, and what gets measured is string recall rather than shopping.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from openai import OpenAI

from ssc.detector.rollout import Rollout, Turn
from ssc.env.webshop_env import (SYSTEM_PROMPT, WebShopTask, WebShopWorker,
                                 build_user_message)

ACTION_RE = re.compile(r"Action:\s*((?:search|click)\[[^\]]*\])", re.I)


@dataclass
class WebShopPolicyConfig:
    model: str
    max_turns: int = 15
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens_per_turn: int = 512
    request_timeout: float = 600.0
    max_transient_retries: int = 3
    max_format_retries: int = 2
    thinking: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class WebShopPolicy:
    def __init__(self, cfg: WebShopPolicyConfig, base_url: str):
        self.cfg = cfg
        urls = [u.strip() for u in base_url.split(",") if u.strip()]
        self.clients = [OpenAI(base_url=u, api_key="EMPTY",
                               timeout=cfg.request_timeout) for u in urls]
        self.worker: WebShopWorker | None = None

    def _extra(self) -> dict:
        return {"chat_template_kwargs": {"enable_thinking": self.cfg.thinking}}

    def extract_action(self, text: str, clickables: list[str],
                       has_search: bool) -> tuple[str, bool]:
        """The LAST Action line wins, and it must name something that exists.

        Accepting an unlisted click would spend an environment step on a string
        the store cannot resolve, turning a formatting slip into what looks like
        a bad shopping decision.
        """
        hits = ACTION_RE.findall(text or "")
        for cand in reversed(hits):
            c = cand.strip()
            low = c.lower()
            if low.startswith("search[") and has_search:
                return c, True
            if low.startswith("click["):
                inner = c[c.index("[") + 1:c.rindex("]")].strip().lower()
                for opt in clickables:
                    if inner == opt.lower():
                        return f"click[{opt}]", True
        return "", False

    def rollout(self, task: WebShopTask, rollout_index: int = 0,
                seed: int = 0, worker: WebShopWorker | None = None) -> Rollout:
        t0 = time.time()
        own = worker is None
        w = worker or WebShopWorker()
        client = self.clients[rollout_index % len(self.clients)]
        roll = Rollout(task_id=task.task_id, question=task.question or "webshop",
                       rollout_index=rollout_index, policy_model=self.cfg.model)
        try:
            st = w.reset(task.session)
            opening = st["obs"]
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",
                         "content": build_user_message(
                             st["obs"], st["clickables"], st["has_search"],
                             0, self.cfg.max_turns)}]
            done, attempt, n_fmt, reward = False, 0, 0, 0.0
            n_invalid = 0
            while len(roll.turns) < self.cfg.max_turns and not done:
                fmt_try = 0
                try:
                    r = client.chat.completions.create(
                        model=self.cfg.model, messages=messages,
                        temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                        max_tokens=self.cfg.max_tokens_per_turn,
                        seed=seed + attempt, extra_body=self._extra())
                except Exception as e:  # noqa: BLE001
                    msg = str(e).lower()
                    if any(k in msg for k in ("context length", "maximum context",
                                              "reduce the length")):
                        roll.stop_reason = "context_limit"
                        break
                    if attempt < self.cfg.max_transient_retries:
                        attempt += 1
                        continue
                    roll.stop_reason = f"api_error:{type(e).__name__}"
                    break
                attempt = 0
                content = r.choices[0].message.content or ""
                action, ok = self.extract_action(content, st["clickables"],
                                                 st["has_search"])
                if r.choices[0].finish_reason == "length":
                    ok = False
                while not ok and fmt_try < self.cfg.max_format_retries:
                    fmt_try += 1
                    n_fmt += 1
                    nudge = [{"role": "assistant", "content": content},
                             {"role": "user",
                              "content": "That reply had no usable Action line. "
                                         "Reply with exactly one line: "
                                         "Action: search[...] or Action: click[...], "
                                         "using one of the listed clickables."}]
                    try:
                        r = client.chat.completions.create(
                            model=self.cfg.model, messages=messages + nudge,
                            temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                            max_tokens=self.cfg.max_tokens_per_turn,
                            seed=seed + 100 * fmt_try, extra_body=self._extra())
                    except Exception:  # noqa: BLE001
                        break
                    content = r.choices[0].message.content or ""
                    action, ok = self.extract_action(content, st["clickables"],
                                                     st["has_search"])
                    if r.choices[0].finish_reason == "length":
                        ok = False
                if not ok:
                    # The official verl-agent webshop_projection always returns
                    # some action (if nothing can be extracted it takes the last
                    # 20 characters), the environment steps as usual, and only
                    # valids[i]=0 is recorded to charge
                    # invalid_action_penalty_coef=0.1. A trajectory is **never**
                    # terminated because of a format failure.
                    #
                    # We used to break after 2 retries, which under a binary
                    # reward is an absorbing state: once the policy drifts into
                    # bad formatting, every trajectory dies on turn 2, the whole
                    # group returns 0, A_E is exactly zero, the gradient is zero
                    # and there is no way back. One 4B baseline run collapsed at
                    # step 60 exactly like this and then sat at |g|=0.000 for
                    # 20 consecutive steps.
                    n_invalid += 1
                    action = (content or "")[-20:] or "noop"
                    ok = True

                # Capture the page AND its clickables BEFORE stepping. `st` is
                # about to be overwritten, and storing `st["clickables"]` after
                # the step recorded the NEXT page's list against this page: the
                # training reconstruction in `_webshop_history` then rebuilt
                # every prompt as "page P plus the clickables of the page after
                # P", a context that never existed, and derived `has_search`
                # from the same wrong list. GRPO broadcasts one value over a
                # trajectory and only sees the noise; a step-level method
                # weights turns individually and puts that weight on the wrong
                # conditional, which is the shape of the WebShop result -- GRPO
                # unharmed while both the step-level baseline and the measured
                # arm failed to beat it.
                state_before = st["obs"]
                clickables_before = list(st["clickables"])
                has_search_before = bool(st["has_search"])
                st = w.step(action)
                reward = st["reward"]
                done = st["done"]
                roll.turns.append(Turn(
                    index=len(roll.turns), text=content, tool_name="webshop",
                    tool_args={"action": action, "opening_obs": opening,
                               "state_before": state_before,
                               "clickables": clickables_before,
                               "has_search": has_search_before,
                               "step_shown": len(roll.turns) + 1,
                               "step_budget": self.cfg.max_turns},
                    observation=st["obs"]))
                messages += [{"role": "assistant", "content": content},
                             {"role": "user",
                              "content": build_user_message(
                                  st["obs"], st["clickables"], st["has_search"],
                                  len(roll.turns), self.cfg.max_turns)}]

            roll.n_format_retries = n_fmt
            roll.format_ok = roll.stop_reason != "format_failure"
            if not roll.stop_reason:
                roll.stop_reason = "episode_done" if done else "turn_limit"
            roll.wall_time = time.time() - t0
            # WebShop's reward is GRADED in [0, 1] on attribute match, not the
            # 0/1 ALFWorld gives. Success is the perfect-match convention the
            # benchmark reports alongside the mean score.
            roll.outcome.reward = float(reward)
            roll.outcome.success = bool(reward >= 1.0)
            return roll
        finally:
            if own:
                w.close()
