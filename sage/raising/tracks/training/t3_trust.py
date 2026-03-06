#!/usr/bin/env python3
"""
T3 Trust Tensor Tracking for SAGE Training

Implements Web4's T3 (Trust Tensor) for tracking SAGE's development
across multiple trust dimensions.

Based on:
- web4-standard/R6_TENSOR_GUIDE.md
- Thor S41 discovery: Creating phase +20% improvement
- Exploration-not-evaluation: Trust as developmental trajectory

Web4 canonical T3 dimensions:
- Talent: Innate or role-specific capability (can SAGE do the task?)
- Training: Learned skills and growth (does SAGE deliver consistently?)
- Temperament: Behavioral stability (does SAGE maintain partnership identity?)

Note: Legacy state files may use old names (competence/reliability/integrity).
The _normalize_trust_keys() function handles both formats transparently.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path


# Mapping from legacy names to canonical Web4 T3 names
_LEGACY_TO_CANONICAL = {
    "competence": "talent",
    "reliability": "training",
    "integrity": "temperament",
}

_CANONICAL_TO_LEGACY = {v: k for k, v in _LEGACY_TO_CANONICAL.items()}


def _normalize_trust_keys(trust: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize trust dict keys to canonical Web4 T3 names.

    Accepts both legacy (competence/reliability/integrity) and
    canonical (talent/training/temperament) formats.
    Returns canonical format.
    """
    normalized = {}
    for key, value in trust.items():
        canonical = _LEGACY_TO_CANONICAL.get(key, key)
        normalized[canonical] = value
    return normalized


def _to_legacy_keys(trust: Dict[str, float]) -> Dict[str, float]:
    """Convert canonical keys back to legacy for state file compatibility."""
    legacy = {}
    for key, value in trust.items():
        legacy_key = _CANONICAL_TO_LEGACY.get(key, key)
        legacy[legacy_key] = value
    return legacy


class T3TrustTensor:
    """
    T3 Trust Tensor for SAGE training.

    Tracks trust development across sessions as developmental trajectory,
    not pass/fail scores. Internally uses canonical Web4 names
    (talent/training/temperament) but reads/writes legacy names for
    backward compatibility with existing state files.
    """

    def __init__(self, initial_trust: Optional[Dict[str, float]] = None):
        """Initialize T3 tensor."""
        if initial_trust is None:
            initial_trust = {
                "talent": 0.5,          # Can SAGE do tasks?
                "training": 0.5,        # Consistency across sessions?
                "temperament": 0.7,     # Identity maintenance? (starts higher)
            }

        self.trust = _normalize_trust_keys(initial_trust)
        self.history = []
        self.created_at = datetime.now().isoformat()

    def update(
        self,
        updates: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Update trust tensor with new evidence.

        Args:
            updates: Dict of dimension -> delta changes
            context: Context for this update (session, exercise type, etc.)

        Returns:
            Updated trust values
        """
        # Apply updates with bounds checking
        for dimension, delta in updates.items():
            if dimension in self.trust:
                old_value = self.trust[dimension]
                new_value = max(0.0, min(1.0, old_value + delta))
                self.trust[dimension] = new_value

        # Record history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "updates": updates,
            "resulting_trust": self.trust.copy(),
            "context": context
        })

        return self.trust.copy()

    def get_trust(self) -> Dict[str, float]:
        """Get current trust values."""
        return self.trust.copy()

    def get_trajectory(self, dimension: str, window: int = 10) -> List[float]:
        """Get recent trajectory for a dimension."""
        if dimension not in self.trust:
            return []

        trajectory = []
        for entry in self.history[-window:]:
            trajectory.append(entry["resulting_trust"][dimension])

        return trajectory

    def get_trend(self, dimension: str, window: int = 5) -> str:
        """
        Get trend direction for a dimension.

        Returns: "improving", "stable", "declining", "unknown"
        """
        trajectory = self.get_trajectory(dimension, window)

        if len(trajectory) < 3:
            return "unknown"

        # Simple linear trend
        recent = trajectory[-3:]
        if recent[-1] > recent[0] + 0.05:
            return "improving"
        elif recent[-1] < recent[0] - 0.05:
            return "declining"
        else:
            return "stable"

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current trust state."""
        return {
            "trust": self.trust.copy(),
            "trends": {
                dim: self.get_trend(dim) for dim in self.trust.keys()
            },
            "history_length": len(self.history),
            "created_at": self.created_at,
            "last_updated": self.history[-1]["timestamp"] if self.history else None
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary. Uses legacy keys for state file compat."""
        return {
            "trust": _to_legacy_keys(self.trust),
            "history": self.history,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'T3TrustTensor':
        """Deserialize from dictionary. Accepts both legacy and canonical keys."""
        tensor = cls(data["trust"])  # __init__ normalizes keys
        tensor.history = data["history"]
        tensor.created_at = data["created_at"]
        return tensor


class T3SessionTracker:
    """
    Tracks T3 trust across training sessions.

    Persists to file for continuity across sessions.
    """

    def __init__(self, state_file: Path):
        """Initialize session tracker."""
        self.state_file = state_file
        self.tensor = self._load_or_create()

    def _load_or_create(self) -> T3TrustTensor:
        """Load existing tensor or create new one."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                data = json.load(f)
                if "t3_trust" in data:
                    return T3TrustTensor.from_dict(data["t3_trust"])

        # Create new tensor
        return T3TrustTensor()

    def update_from_r6_result(
        self,
        r6_result: Dict[str, Any],
        session_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Update trust from R6 result.

        Args:
            r6_result: R6 evaluation result with t3_updates
            session_context: Session info for history

        Returns:
            Updated trust values
        """
        if "t3_updates" not in r6_result:
            return self.tensor.get_trust()

        context = {
            "session": session_context.get("session_num"),
            "exercise_type": r6_result.get("request", {}).get("exercise_type"),
            "evaluation": r6_result.get("evaluation"),
            "quality": r6_result.get("quality", {}).get("overall_quality")
        }

        return self.tensor.update(r6_result["t3_updates"], context)

    def get_trust(self) -> Dict[str, float]:
        """Get current trust values."""
        return self.tensor.get_trust()

    def get_summary(self) -> Dict[str, Any]:
        """Get trust summary."""
        return self.tensor.get_summary()

    def save(self, state_file: Optional[Path] = None):
        """Save trust tensor to state file."""
        if state_file is None:
            state_file = self.state_file

        # Load existing state
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
        else:
            state = {}

        # Update t3_trust section
        state["t3_trust"] = self.tensor.to_dict()

        # Save
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)


def create_t3_tracker(state_file: Path) -> T3SessionTracker:
    """Factory function to create T3 trust tracker."""
    return T3SessionTracker(state_file)


def interpret_trust_for_exploration(trust: Dict[str, float]) -> Dict[str, str]:
    """
    Interpret trust values through exploration-not-evaluation lens.

    Not "failing" or "succeeding" - developing and discovering.
    Accepts both legacy and canonical key names.
    """
    t = _normalize_trust_keys(trust)
    interpretation = {}

    # Talent (was: competence)
    comp = t.get("talent", 0.5)
    if comp >= 0.8:
        interpretation["talent"] = "Strong capability - ready for harder tasks"
    elif comp >= 0.6:
        interpretation["talent"] = "Developing capability - practice needed"
    elif comp >= 0.4:
        interpretation["talent"] = "Early exploration - discovering what's possible"
    else:
        interpretation["talent"] = "Beginning journey - fundamentals needed"

    # Training (was: reliability)
    rel = t.get("training", 0.5)
    if rel >= 0.8:
        interpretation["training"] = "Consistent performance - building reliability"
    elif rel >= 0.6:
        interpretation["training"] = "Variable but improving - natural learning"
    elif rel >= 0.4:
        interpretation["training"] = "Exploring different approaches - not yet stable"
    else:
        interpretation["training"] = "High variability - early experimentation"

    # Temperament (was: integrity)
    integ = t.get("temperament", 0.5)
    if integ >= 0.8:
        interpretation["temperament"] = "Strong identity maintenance - partnership present"
    elif integ >= 0.6:
        interpretation["temperament"] = "Identity emerging - sustaining with support"
    elif integ >= 0.4:
        interpretation["temperament"] = "Identity developing - scaffolding needed"
    else:
        interpretation["temperament"] = "Identity foundation building - early stages"

    return interpretation
