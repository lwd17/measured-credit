"""Rollout record and its detector serialization (runbook sections 11, 20, 55).

Section 11 forbids the detector from ever seeing the gold answer, the success
flag, the terminal reward, the GRPO group rank or the benchmark verifier
output. That is a property of a SERIALIZER, not of a convention, so the
outcome fields live in a separate nested object and `to_detector_input()`
builds its result from an explicit allow-list -- never by copying the record
and deleting keys, which silently leaks whatever field is added next.

Section 55 Test 7 asserts this directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

# Section 11's forbidden inputs, plus the obvious spellings of each.
FORBIDDEN_SUBSTRINGS = (
    "reward", "success", "gold", "judge", "score", "correct", "label",
    "is_right", "em_", "_em", "f1", "advantage", "group_rank", "rank",
)


@dataclass
class Turn:
    """One action turn: what the policy generated, and what came back.

    `text` is the generated action (reasoning + tool call, as emitted).
    `observation` is the environment's reply, which the policy did not generate
    and which therefore carries no actor loss (section 9 / Test 5).
    """

    index: int                      # 0-based action-turn index
    text: str
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    observation: str = ""
    n_generated_tokens: int = 0


@dataclass
class Outcome:
    """Everything the detector must never see (section 11).

    Kept in its own object so that leaking it requires reaching into a named
    attribute rather than forgetting to strip a key.
    """

    success: int | None = None
    reward: float | None = None
    gold_answer: str | None = None
    judge_verdict: str | None = None


@dataclass
class Rollout:
    task_id: str
    question: str
    turns: list[Turn] = field(default_factory=list)
    final_answer: str = ""
    rollout_index: int = 0
    policy_model: str = ""
    stop_reason: str = ""
    wall_time: float = 0.0
    format_ok: bool = True
    n_format_retries: int = 0
    outcome: Outcome = field(default_factory=Outcome)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def num_tool_calls(self) -> int:
        return sum(1 for t in self.turns if t.tool_name)

    @property
    def num_generated_tokens(self) -> int:
        return sum(t.n_generated_tokens for t in self.turns)

    # ---- detector-facing ---------------------------------------------------
    def to_detector_input(self, *, include_final_answer: bool = True) -> dict:
        """Section 11's allowed inputs 1-4 and nothing else.

        Built by construction from an allow-list. `include_final_answer` exists
        because section 38 runs a robustness arm with the final answer removed.
        """
        d = {
            "task": self.question,
            "num_turns": self.num_turns,
            "turns": [
                {
                    "turn": t.index + 1,          # detector sees 1-based
                    "action": t.text,
                    "tool": t.tool_name,
                    "tool_input": t.tool_args,
                    "observation": t.observation,
                }
                for t in self.turns
            ],
        }
        if include_final_answer:
            d["final_answer"] = self.final_answer
        return d

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Rollout":
        return cls(
            task_id=d["task_id"], question=d["question"],
            turns=[Turn(**t) for t in d.get("turns", [])],
            final_answer=d.get("final_answer", ""),
            rollout_index=d.get("rollout_index", 0),
            policy_model=d.get("policy_model", ""),
            stop_reason=d.get("stop_reason", ""),
            wall_time=d.get("wall_time", 0.0),
            format_ok=d.get("format_ok", True),
            n_format_retries=d.get("n_format_retries", 0),
            outcome=Outcome(**d.get("outcome", {})),
        )


def assert_reward_blind(payload) -> None:
    """Section 55 Test 7. Raises if any reward-derived field could reach the detector.

    Checks KEYS recursively rather than values: an observation legitimately
    contains the word "score" (a search snippet about a football match), but a
    KEY named `score` is a leak.
    """
    def walk(node, path="") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                kl = str(k).lower()
                for bad in FORBIDDEN_SUBSTRINGS:
                    if bad in kl:
                        raise AssertionError(
                            f"reward-derived key {k!r} at {path or '<root>'} "
                            f"would reach the detector (section 11)")
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(payload)


def detector_payload_json(rollout: Rollout, *, include_final_answer: bool = True) -> str:
    """Serialize for the detector, asserting blindness on the way out."""
    payload = rollout.to_detector_input(include_final_answer=include_final_answer)
    assert_reward_blind(payload)
    return json.dumps(payload, ensure_ascii=False, indent=1)
