"""Core data structures (runbook section 51).

Turn indexing convention, fixed once here because every downstream module
depends on it:

  * A trajectory has `num_turns` ACTION TURNS, indexed 0 .. num_turns-1.
  * The detector sees turns rendered as 1-based ("[turn 1]"), because a model
    reading "[turn 0]" reliably mis-counts. `SemanticEvent` stores 0-based
    indices; conversion happens exactly once, in `from_detector()`.
  * A REPLACE span is HALF-OPEN: [start_turn, replace_turn).

Section 40.2 is the reason the span is half-open. Actions under a wrong
hypothesis are often exactly what produced the evidence that disproved it, so
the action AT the replacement turn is not itself invalidated -- only the span
strictly before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

# ERROR means the event was never judged (validator failure), which is a
# MISSING MEASUREMENT and must never be read as "no event found".
EVENT_TYPES = ("REPLACE", "REFINE", "COMMIT", "STABLE", "ERROR")


@dataclass
class SemanticEvent:
    """One candidate semantic-regime boundary (section 51).

    Only `event_type == "REPLACE"` and `accepted` events ever affect credit
    (section 6.3); REFINE/COMMIT/STABLE are recorded for the audit and for
    ablation A4, never applied.
    """

    event_type: str
    start_turn: int                  # 0-based, inclusive
    replace_turn: int                # 0-based, EXCLUSIVE (half-open span)
    trigger_turn: int | None = None
    old_regime: str = ""
    new_regime: str = ""
    accepted: bool = False
    confidence: float = 0.0
    reason: str = ""
    # Provenance for section 38's robustness experiments.
    proposer_variant: str = ""
    validator_reason: str = ""

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            self.event_type = "STABLE"
        self.confidence = float(min(1.0, max(0.0, self.confidence)))

    @property
    def span_length(self) -> int:
        """D_m = c_m - b_m (section 22)."""
        return self.replace_turn - self.start_turn

    @property
    def is_active(self) -> bool:
        """Whether this event may modify credit at all (section 6.3)."""
        return self.accepted and self.event_type == "REPLACE" and self.span_length > 0

    def turns(self) -> range:
        """The invalidated turns: [b, c)."""
        return range(self.start_turn, self.replace_turn)

    @classmethod
    def from_detector(cls, raw: dict, num_turns: int, *, variant: str = "") -> "SemanticEvent | None":
        """Parse one proposer event. Returns None if it cannot be made valid.

        The detector emits 1-based turns; this is the ONLY place that converts.
        """
        try:
            b = int(raw["start_turn"]) - 1
            c = int(raw["replace_turn"]) - 1
        except (KeyError, TypeError, ValueError):
            return None
        trig = raw.get("trigger_observation_turn", raw.get("trigger_turn"))
        try:
            trig = int(trig) - 1 if trig is not None else None
        except (TypeError, ValueError):
            trig = None

        # Clamp into range rather than discard: a proposer that names the last
        # turn as the replacement is making a boundary error, not a semantic one.
        b = max(0, min(b, num_turns - 1))
        c = max(0, min(c, num_turns))
        if not (0 <= b < c <= num_turns):
            return None
        return cls(
            event_type=str(raw.get("event_type", "REPLACE")).upper(),
            start_turn=b, replace_turn=c, trigger_turn=trig,
            old_regime=str(raw.get("old_regime", ""))[:600],
            new_regime=str(raw.get("new_regime", ""))[:600],
            reason=str(raw.get("reason", ""))[:600],
            confidence=_confidence(raw.get("confidence")),
            proposer_variant=variant,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticEvent":
        """Reload a cached event. Unknown keys are dropped rather than raising,
        so an older cache stays readable after a field is added."""
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})


_WORD_CONFIDENCE = {"high": 0.9, "very high": 0.95, "medium": 0.6,
                    "moderate": 0.6, "low": 0.3, "very low": 0.1, "certain": 1.0}


def _confidence(v) -> float:
    """Detectors return "high" as readily as 0.9; neither may raise."""
    if isinstance(v, (int, float)):
        f = float(v)
        return f / 100.0 if f > 1.0 else f
    s = str(v or "").strip().lower().rstrip("%")
    if s in _WORD_CONFIDENCE:
        return _WORD_CONFIDENCE[s]
    try:
        f = float(s)
    except ValueError:
        return 0.0
    return f / 100.0 if f > 1.0 else f


@dataclass
class RolloutSemanticMetadata:
    """Per-rollout detector output (section 51) plus the section 20 audit row."""

    task_id: str
    num_turns: int
    events: list[SemanticEvent] = field(default_factory=list)
    turn_survival: list[float] = field(default_factory=list)
    semantic_horizon: int = 1
    rollout_index: int = 0

    @property
    def accepted_replace(self) -> list[SemanticEvent]:
        return [e for e in self.events if e.is_active]

    @property
    def num_replace_events(self) -> int:
        """N_replace (section 21)."""
        return len(self.accepted_replace)

    @property
    def invalidated_turns(self) -> int:
        """Turns inside at least one validated REPLACE span (section 22).

        Counted as a UNION, not a sum: overlapping spans must not double-count
        a turn, or M_inv can exceed 1.
        """
        covered: set[int] = set()
        for e in self.accepted_replace:
            covered.update(e.turns())
        return len(covered)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "rollout_index": self.rollout_index,
            "num_turns": self.num_turns,
            "events": [e.to_dict() for e in self.events],
            "turn_survival": self.turn_survival,
            "semantic_horizon": self.semantic_horizon,
            "num_replace_events": self.num_replace_events,
            "invalidated_turns": self.invalidated_turns,
        }
