"""
Lookahead-aware metacog detectors.

Three detectors that observe the PATTERN of lookahead predictions
and signal when the system should change strategy. Separate from
core.py to keep the base metacog clean. Designed as standalone
functions that can be called from observe_tick or post-hoc analysis.

Design doc: shared-context/explorations/metacog-lookahead-integration-2026-04-21/design.md
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .core import MetacogSignal


def detect_prediction_equivalence(
    predictions: Dict[str, int],
    *,
    threshold_ratio: float = 0.2,
    min_equivalent: int = 3,
    severity: float = 0.6,
) -> Optional[MetacogSignal]:
    """Fire when multiple actions produce indistinguishable effects.

    The "LEFT and RIGHT both 201px" problem — the LLM can't choose
    between actions that look the same from pixel-count alone.

    Args:
        predictions: action_name → pixel_change_count (0 = blocked)
        threshold_ratio: how close counts must be to be "equivalent"
        min_equivalent: minimum actions that must be equivalent to fire
    """
    active = {k: v for k, v in predictions.items() if v > 0}
    if len(active) < min_equivalent:
        return None

    max_v = max(active.values())
    if max_v == 0:
        return None

    equivalent = {
        k: v for k, v in active.items()
        if v >= max_v * (1 - threshold_ratio)
    }

    if len(equivalent) < min_equivalent:
        return None

    return MetacogSignal(
        signal="prediction_equivalence",
        severity=severity,
        evidence={
            "equivalent_actions": list(equivalent.keys()),
            "pixel_range": [min(equivalent.values()), max(equivalent.values())],
            "total_active": len(active),
        },
        suggestion="predictions don't distinguish — need spatial or semantic detail",
    )


def detect_lookahead_blind(
    predictions: Dict[str, int],
    *,
    blind_threshold: int = 2,
    severity: float = 0.7,
) -> Optional[MetacogSignal]:
    """Fire when all actions produce negligible or zero effect.

    The "everything blocked" state — directional actions don't apply
    (CLICK game) or the game requires a different interaction mode.

    Args:
        predictions: action_name → pixel_change_count
        blind_threshold: max px to count as "no effect"
    """
    if not predictions:
        return None

    active = {k: v for k, v in predictions.items() if v > blind_threshold}
    if active:
        return None  # at least one action has a real effect

    return MetacogSignal(
        signal="lookahead_blind",
        severity=severity,
        evidence={
            "actions_checked": list(predictions.keys()),
            "max_change": max(predictions.values()) if predictions else 0,
            "blind_threshold": blind_threshold,
        },
        suggestion="all actions blocked — try CLICK at different coordinates or change interaction mode",
    )


def detect_predictions_ignored(
    response: str,
    predictions: Dict[str, str],
    *,
    min_references: float = 0.2,
    severity: float = 0.5,
) -> Optional[MetacogSignal]:
    """Fire when the LLM's response doesn't reference the predictions.

    If the LLM was given prediction data but doesn't use it, the
    prompt framing may need to change (Thor's breakthrough: predictions
    must be the central question, not supplementary context).

    Args:
        response: the LLM's response text
        predictions: action_name → prediction_summary_text
        min_references: fraction of predictions that must be referenced
    """
    if not predictions or not response:
        return None

    response_lower = response.lower()
    referenced = 0

    for action, pred_text in predictions.items():
        # Check if response mentions the action name AND any key
        # fact from the prediction
        action_mentioned = action.lower() in response_lower
        facts = _extract_facts(pred_text)
        fact_mentioned = any(f in response_lower for f in facts)
        if action_mentioned and fact_mentioned:
            referenced += 1

    fraction = referenced / len(predictions)
    if fraction >= min_references:
        return None

    return MetacogSignal(
        signal="predictions_ignored",
        severity=severity,
        evidence={
            "predictions_given": len(predictions),
            "predictions_referenced": referenced,
            "reference_fraction": round(fraction, 3),
        },
        suggestion="LLM not using prediction data — reframe predictions as the central question",
    )


def _extract_facts(pred_text: str) -> List[str]:
    """Pull key facts from a prediction summary for reference checking."""
    facts = []
    text = pred_text.lower()
    # Look for numbers (pixel counts, percentages)
    import re
    for num in re.findall(r'\d+(?:\.\d+)?', text):
        facts.append(num)
    # Look for directional/spatial terms
    for term in ("blocked", "no change", "trivial", "top", "bottom",
                 "left", "right", "center", "red", "blue", "green",
                 "white", "black", "gray"):
        if term in text:
            facts.append(term)
    return facts
