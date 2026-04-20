"""
Bridge: Metacog → MetabolicBlock.

Converts live metacog state into the MRH MetabolicBlock that the
Phase 5 composer consumes. The dispatcher calls this once per
invocation to populate the metabolic slot.

Usage:
    from sage.cognition.metacog.mrh_bridge import metacog_to_metabolic_block
    block = metacog_to_metabolic_block(metacog, atp_balance=42.0)
    mrh_context.metabolic = block
"""

from __future__ import annotations

from typing import Optional

from sage.cognition.metacog.core import Metacog
from sage.context.mrh.metabolic import MetabolicBlock, MetacogSignalSummary


def metacog_to_metabolic_block(
    metacog: Metacog,
    *,
    metabolic_state: str = "active",
    atp_balance: Optional[float] = None,
    atp_trend: str = "stable",
    confidence: Optional[float] = None,
    actions_since_last_invoke: int = 0,
) -> MetabolicBlock:
    """Convert live metacog state → MetabolicBlock for MRH composition.

    Extracts active signals, stuck duration, and swap recommendations
    from the metacog instance. Caller supplies the non-metacog fields
    (ATP, metabolic state, confidence) since those come from different
    kernel subsystems.
    """
    # Convert active signals to the block's data-only form
    signal_summaries = [
        MetacogSignalSummary(
            signal=sig.signal,
            severity=sig.severity,
            evidence=dict(sig.evidence),
            suggestion=sig.suggestion,
        )
        for sig in metacog.active_signals()
    ]

    # Derive stuck_duration from perseveration evidence if available
    stuck_duration = 0
    for sig in metacog.active_signals():
        if sig.signal == "perseveration":
            stuck_duration = sig.evidence.get("action_count", 3)
            break

    # Build swap recommendations from critical signals
    swap_recs = []
    for sig in metacog.active_signals():
        if sig.signal == "perseveration" and sig.severity >= 0.7:
            swap_recs.append("mechanics:stuck_escape")
        if sig.signal == "budget_critical":
            swap_recs.append("mechanics:budget_conserve")
        if sig.signal == "exploration_starvation":
            swap_recs.append("mechanics:exploration_hints")

    return MetabolicBlock(
        metacog_signals=signal_summaries,
        metabolic_state=metabolic_state,
        atp_balance=atp_balance,
        atp_trend=atp_trend,
        confidence=confidence,
        stuck_duration=stuck_duration,
        actions_since_last_invoke=actions_since_last_invoke,
        swap_recommendations=swap_recs,
    )
