"""
Reward Prediction Error (RPE) — Dopaminergic learning signal for SAGE.

Computes the gap between predicted and actual action outcomes.
Updates action priors that inform the router's selection decisions.

Owner: Legion
Review pair: Sprout (router)
"""

from .core import (
    RewardPredictionError,
    Prediction,
    Outcome,
    RPESignal,
    PriorTable,
    PriorEntry,
    compute_outcome_value,
)

__all__ = [
    "RewardPredictionError",
    "Prediction",
    "Outcome",
    "RPESignal",
    "PriorTable",
    "PriorEntry",
    "compute_outcome_value",
]
