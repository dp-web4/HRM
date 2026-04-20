"""MetabolicBlock — the model's interoceptive signals.

Per Sprout: "The interoceptive channel — the model's proprioception.
Should be crisp numerical signals, not prose. The model's weights
handle the interpretation."

Carries Nomad's metacog output + ATP/ADP + stuck state + confidence.
Includes swap_recommendations field so metacog can suggest runtime
swaps (e.g., stuck → mechanics:stuck_escape).

Sprint: Phase 5 B7
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sage.context.mrh.base import MRHBlock


@dataclass
class MetacogSignalSummary:
    """One active metacog signal, already extracted from a Metacog instance.

    Block is pure-data per Thor's constraint — the dispatcher calls
    metacog.active_signals() and converts to this form before populating
    the block.
    """
    signal: str                          # e.g., "perseveration"
    severity: float                      # 0..1
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""


@dataclass
class MetabolicBlock(MRHBlock):
    """The model's interoceptive state for the invocation.

    Attributes:
        metacog_signals: active signals from Nomad's Metacog detector.
        metabolic_state: "wake" | "focus" | "rest" | "dream" | "crisis" |
            "active" (gameplay). Compact state label.
        atp_balance: available budget (e.g., steps remaining in game).
        atp_trend: "rising" | "stable" | "falling".
        confidence: overall system/CNN confidence (0..1).
        stuck_duration: consecutive steps with <threshold frame change.
            0 = not stuck.
        actions_since_last_invoke: CNN autonomous steps since LLM's
            last turn (critical for three-party framing).
        swap_recommendations: list of block-swap hints this block wants
            the composer to consider. Format: "{block_kind}:{profile}"
            e.g., "mechanics:stuck_escape". Composer honors or ignores.
    """
    priority: int = 70
    metacog_signals: List[MetacogSignalSummary] = field(default_factory=list)
    metabolic_state: str = "active"
    atp_balance: Optional[float] = None
    atp_trend: str = "stable"
    confidence: Optional[float] = None
    stuck_duration: int = 0
    actions_since_last_invoke: int = 0
    swap_recommendations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.kind = "metabolic"

    def render(self, budget_tokens: int) -> str:
        lines: List[str] = ["## System state"]

        # Crisp numerical status line (Sprout's framing: numbers, not narrative)
        status_parts = [f"metabolic={self.metabolic_state}"]
        if self.atp_balance is not None:
            status_parts.append(f"ATP={self.atp_balance:.0f} ({self.atp_trend})")
        if self.confidence is not None:
            status_parts.append(f"confidence={self.confidence:.2f}")
        status_parts.append(f"stuck={self.stuck_duration}")
        status_parts.append(f"cnn_steps_since_llm={self.actions_since_last_invoke}")
        lines.append(" | ".join(status_parts))

        if self.metacog_signals:
            lines.append("Metacog signals:")
            for s in self.metacog_signals:
                evidence_str = ""
                if s.evidence:
                    kv = [f"{k}={v}" for k, v in list(s.evidence.items())[:3]]
                    evidence_str = f" [{', '.join(kv)}]"
                suffix = f" — {s.suggestion}" if s.suggestion else ""
                lines.append(
                    f"  {s.signal} (severity {s.severity:.1f}){evidence_str}{suffix}"
                )

        # swap_recommendations are for the composer, not the LLM. Skip here.
        return "\n".join(lines)

    def summarize(self, budget_tokens: int) -> str:
        # ADP: single status line
        parts = [self.metabolic_state]
        if self.stuck_duration > 0:
            parts.append(f"stuck={self.stuck_duration}")
        if self.metacog_signals:
            sig_names = ",".join(s.signal for s in self.metacog_signals[:3])
            parts.append(f"signals={sig_names}")
        return f"System: {' | '.join(parts)}."

    def minimal(self) -> str:
        if self.metacog_signals or self.stuck_duration > 0:
            return "System: signals present."
        return "System: nominal."

    def estimate_cost(self) -> int:
        base = 48
        base += len(self.metacog_signals) * 20
        return base

    def get_swap_recommendations(self) -> List[str]:
        """Expose swap hints for the composer to collect (via
        MRHContext.swap_recommendations()).

        Note: the composer doesn't auto-apply these; it's a suggestion
        mechanism. The dispatcher layer (above the composer) decides
        whether to honor.
        """
        return list(self.swap_recommendations)
