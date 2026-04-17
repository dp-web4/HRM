"""
RPE Core — Reward Prediction Error computation and prior table.

The dopaminergic analog: measures prediction accuracy and updates action priors.

Every action produces an RPE signal:
  RPE = actual_outcome_value - predicted_outcome_value

  RPE > 0 → positive surprise → reinforce
  RPE = 0 → prediction confirmed → calibrated
  RPE < 0 → negative surprise → reduce preference
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─── Data Structures ───


@dataclass
class Prediction:
    """What the system expected before acting."""
    state_key: str
    action: str
    expected_outcome: str
    expected_value: float
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Outcome:
    """What actually happened."""
    state_key: str
    actual_outcome: str
    actual_value: float
    pixel_delta: int = 0
    level_up: bool = False
    died: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class RPESignal:
    """The computed prediction error."""
    prediction: Prediction
    outcome: Outcome
    rpe: float
    magnitude: float
    sign: str  # "positive" | "negative" | "zero"
    prior_before: float
    prior_after: float
    learning_rate: float


@dataclass
class PriorEntry:
    """A learned action-value estimate for a (state, action) pair."""
    value: float
    observation_count: int = 0
    last_rpe: float = 0.0
    last_updated: float = field(default_factory=time.time)


# ─── Outcome Valuation ───


def compute_outcome_value(pixel_delta: int, level_up: bool, died: bool,
                          budget_remaining: float = 1.0) -> float:
    """Convert raw game observations to scalar reward value.

    Returns value in [-1.0, 1.0]:
      1.0 = level up (best possible outcome)
     -1.0 = death (worst possible outcome)
      0.0 = neutral (nothing meaningful changed)
    """
    if died:
        return -1.0
    if level_up:
        return 1.0
    if pixel_delta == 0:
        return -0.1  # no effect — slightly negative (wasted action)
    if pixel_delta <= 2:
        return -0.05  # minimal change — near-neutral
    if pixel_delta <= 10:
        return 0.0  # small change — neutral
    if pixel_delta > 500:
        return 0.3  # big state change — cautiously positive
    if pixel_delta > 100:
        return 0.2  # significant change — mildly positive
    return 0.1  # normal movement — slightly positive


# ─── Prior Table ───


class PriorTable:
    """Learned action-value estimates, updated by RPE signals."""

    def __init__(self, default_prior: float = 0.5):
        self._priors: Dict[Tuple[str, str], PriorEntry] = {}
        self.default_prior = default_prior

    def predict(self, state_key: str, action: str) -> float:
        """Return current predicted value for (state, action)."""
        entry = self._priors.get((state_key, action))
        if entry is None:
            return self.default_prior
        return entry.value

    def update(self, state_key: str, action: str,
               actual_value: float) -> Tuple[float, float, float]:
        """Compute RPE and update prior.

        Returns (rpe, prior_before, prior_after, learning_rate).
        """
        entry = self._priors.get((state_key, action))
        if entry is None:
            entry = PriorEntry(value=self.default_prior, observation_count=0)
            self._priors[(state_key, action)] = entry

        prior_before = entry.value

        # Learning rate decays with evidence: α = 1/(1 + count)
        alpha = 1.0 / (1.0 + entry.observation_count)

        # RPE = actual - predicted
        rpe = actual_value - prior_before

        # Update prior: new_value = old_value + α * RPE
        new_value = prior_before + alpha * rpe
        # Clamp to [0, 1]
        new_value = max(0.0, min(1.0, new_value))

        entry.value = new_value
        entry.observation_count += 1
        entry.last_rpe = rpe
        entry.last_updated = time.time()

        return rpe, prior_before, new_value, alpha

    def get_action_priors(self, state_key: str,
                          available_actions: List[str]) -> Dict[str, float]:
        """Return priors for all available actions in this state."""
        return {
            action: self.predict(state_key, action)
            for action in available_actions
        }

    def get_entry(self, state_key: str, action: str) -> Optional[PriorEntry]:
        """Get the full prior entry for inspection."""
        return self._priors.get((state_key, action))

    @property
    def size(self) -> int:
        return len(self._priors)

    def dump(self) -> Dict[str, Any]:
        """JSON-serializable snapshot."""
        return {
            "default_prior": self.default_prior,
            "entries": {
                f"{sk}|{act}": {
                    "state_key": sk,
                    "action": act,
                    "value": entry.value,
                    "observation_count": entry.observation_count,
                    "last_rpe": entry.last_rpe,
                    "last_updated": entry.last_updated,
                }
                for (sk, act), entry in self._priors.items()
            }
        }

    @classmethod
    def from_dump(cls, data: Dict[str, Any]) -> "PriorTable":
        """Reconstruct from JSON dump."""
        table = cls(default_prior=data.get("default_prior", 0.5))
        for key, entry_data in data.get("entries", {}).items():
            sk = entry_data["state_key"]
            act = entry_data["action"]
            table._priors[(sk, act)] = PriorEntry(
                value=entry_data["value"],
                observation_count=entry_data["observation_count"],
                last_rpe=entry_data.get("last_rpe", 0.0),
                last_updated=entry_data.get("last_updated", 0.0),
            )
        return table


# ─── RPE Core ───


class RewardPredictionError:
    """Dopaminergic learning signal for the consciousness loop.

    Usage:
        rpe = RewardPredictionError()

        # Step 7 (before acting):
        pred = rpe.predict(state_key, action, "player moves right")

        # Step 8 (after acting):
        signal = rpe.observe(pred, "player fell into pit", pixel_delta=400,
                             level_up=False, died=True)

        # Step 5 (router queries):
        priors = rpe.get_action_priors(state_key, ["UP", "DOWN", "LEFT", "RIGHT"])
    """

    def __init__(self, default_prior: float = 0.5,
                 on_signal: Optional[Callable[["RPESignal"], None]] = None):
        self.priors = PriorTable(default_prior=default_prior)
        self.history: List[RPESignal] = []
        self.on_signal = on_signal
        self._total_positive = 0
        self._total_negative = 0
        self._total_zero = 0
        self._rpe_sum = 0.0
        self._rpe_sq_sum = 0.0

    def predict(self, state_key: str, action: str,
                expected_outcome: str = "",
                confidence: float = 0.5) -> Prediction:
        """Register a prediction BEFORE acting. Called at step 7 entry."""
        expected_value = self.priors.predict(state_key, action)
        return Prediction(
            state_key=state_key,
            action=action,
            expected_outcome=expected_outcome,
            expected_value=expected_value,
            confidence=confidence,
        )

    def observe(self, prediction: Prediction,
                actual_outcome: str = "",
                pixel_delta: int = 0,
                level_up: bool = False,
                died: bool = False,
                budget_remaining: float = 1.0,
                actual_value: Optional[float] = None) -> RPESignal:
        """Record outcome AFTER acting. Computes RPE, updates priors.

        Called at step 8 entry. Emits signal via on_signal callback.
        """
        # Compute actual value from observations (or use provided)
        if actual_value is None:
            actual_value = compute_outcome_value(
                pixel_delta, level_up, died, budget_remaining)

        outcome = Outcome(
            state_key=prediction.state_key,  # using pre-action state for pairing
            actual_outcome=actual_outcome,
            actual_value=actual_value,
            pixel_delta=pixel_delta,
            level_up=level_up,
            died=died,
        )

        # Update priors and get RPE
        rpe, prior_before, prior_after, alpha = self.priors.update(
            prediction.state_key, prediction.action, actual_value)

        # Classify signal
        if abs(rpe) < 0.01:
            sign = "zero"
            self._total_zero += 1
        elif rpe > 0:
            sign = "positive"
            self._total_positive += 1
        else:
            sign = "negative"
            self._total_negative += 1

        signal = RPESignal(
            prediction=prediction,
            outcome=outcome,
            rpe=rpe,
            magnitude=abs(rpe),
            sign=sign,
            prior_before=prior_before,
            prior_after=prior_after,
            learning_rate=alpha,
        )

        # Track statistics
        self._rpe_sum += rpe
        self._rpe_sq_sum += rpe * rpe
        self.history.append(signal)

        # Emit signal for router integration
        if self.on_signal:
            try:
                self.on_signal(signal)
            except Exception:
                pass  # never let callback failures block the loop

        return signal

    def get_action_priors(self, state_key: str,
                          available_actions: List[str]) -> Dict[str, float]:
        """Query priors for router (Sprout). Called at step 5."""
        return self.priors.get_action_priors(state_key, available_actions)

    def get_recent_signals(self, n: int = 10) -> List[RPESignal]:
        """Last N RPE signals for metacog (Nomad) and R7 reflection."""
        return self.history[-n:]

    def get_stats(self) -> Dict[str, Any]:
        """Summary statistics for dashboard / metacog."""
        total = len(self.history)
        if total == 0:
            return {
                "total_signals": 0,
                "mean_rpe": 0.0,
                "rpe_variance": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "zero_ratio": 0.0,
                "prior_table_size": self.priors.size,
                "calibration": "no_data",
            }

        mean_rpe = self._rpe_sum / total
        variance = (self._rpe_sq_sum / total) - (mean_rpe * mean_rpe)

        # Calibration assessment
        if abs(mean_rpe) < 0.05 and variance < 0.1:
            calibration = "well_calibrated"
        elif abs(mean_rpe) > 0.3:
            calibration = "systematic_bias"
        elif variance > 0.5:
            calibration = "high_variance"
        else:
            calibration = "converging"

        return {
            "total_signals": total,
            "mean_rpe": round(mean_rpe, 4),
            "rpe_variance": round(max(0, variance), 4),
            "positive_ratio": round(self._total_positive / total, 3),
            "negative_ratio": round(self._total_negative / total, 3),
            "zero_ratio": round(self._total_zero / total, 3),
            "prior_table_size": self.priors.size,
            "calibration": calibration,
        }

    def dump(self) -> Dict[str, Any]:
        """Full JSON snapshot for episodic index (Thor)."""
        return {
            "priors": self.priors.dump(),
            "stats": self.get_stats(),
            "recent_signals": [
                {
                    "action": s.prediction.action,
                    "state_key": s.prediction.state_key,
                    "expected": s.prediction.expected_value,
                    "actual": s.outcome.actual_value,
                    "rpe": s.rpe,
                    "sign": s.sign,
                    "prior_before": s.prior_before,
                    "prior_after": s.prior_after,
                }
                for s in self.history[-20:]
            ],
        }

    @classmethod
    def from_dump(cls, data: Dict[str, Any],
                  on_signal: Optional[Callable] = None) -> "RewardPredictionError":
        """Reconstruct from episodic dump (partial — priors only)."""
        rpe = cls(on_signal=on_signal)
        if "priors" in data:
            rpe.priors = PriorTable.from_dump(data["priors"])
        return rpe
