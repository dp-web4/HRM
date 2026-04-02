"""
Surprise Detector - Prediction Error

Detects deviation from expected sensor values.
Learns to predict next sensor state, measures surprise as prediction error.
"""

import time as _time

import torch
import numpy as np
from typing import Dict, Any, Optional
from collections import deque

from sage.services.snarc.temporal import TimestampedDeque, DEFAULT_HALF_LIVES


class SimplePredictorEMA:
    """
    Simple Exponential Moving Average predictor

    Predicts next value as weighted average of past observations.
    Good baseline for smooth time series.
    """

    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Smoothing factor (0-1). Higher = more weight to recent.
        """
        self.alpha = alpha
        self.prediction: Optional[Any] = None

    def predict(self) -> Optional[Any]:
        """Get current prediction"""
        return self.prediction

    def update(self, observation: Any):
        """Update predictor with new observation"""
        if self.prediction is None:
            # First observation - just store it
            self.prediction = observation
        else:
            # EMA update
            if isinstance(observation, torch.Tensor):
                self.prediction = (
                    self.alpha * observation +
                    (1 - self.alpha) * self.prediction
                )
            elif isinstance(observation, np.ndarray):
                self.prediction = (
                    self.alpha * observation +
                    (1 - self.alpha) * self.prediction
                )
            elif isinstance(observation, (int, float)):
                self.prediction = (
                    self.alpha * observation +
                    (1 - self.alpha) * self.prediction
                )
            else:
                # Unknown type - just store
                self.prediction = observation


class SurpriseDetector:
    """
    Detect deviation from prediction

    Maintains predictor per sensor, computes surprise as
    normalized distance between prediction and actual.
    """

    def __init__(self, alpha: float = 0.3, history_size: int = 100,
                 half_life: float = DEFAULT_HALF_LIVES['surprise']):
        """
        Args:
            alpha: Smoothing factor for EMA predictor
            history_size: How many past surprise scores to keep
            half_life: Time-decay half-life in seconds for normalization weighting
        """
        self.alpha = alpha
        self.history_size = history_size
        self.half_life = half_life

        # Predictor per sensor
        self.predictors: Dict[str, SimplePredictorEMA] = {}

        # Surprise history per sensor (for normalization) — original deque kept for compat
        self.surprise_history: Dict[str, deque] = {}
        # Time-aware history for weighted normalization
        self._timed_history: Dict[str, TimestampedDeque] = {}

    def compute(self, sensor_output: Any, sensor_id: str) -> float:
        """
        Compute surprise for sensor observation

        Args:
            sensor_output: Current sensor reading
            sensor_id: Unique sensor identifier

        Returns:
            Surprise score (0.0-1.0)
                0.0 = perfectly predicted
                1.0 = maximally surprising
        """
        # Get or create predictor for this sensor
        if sensor_id not in self.predictors:
            self.predictors[sensor_id] = SimplePredictorEMA(self.alpha)
            self.surprise_history[sensor_id] = deque(maxlen=self.history_size)
            self._timed_history[sensor_id] = TimestampedDeque(maxlen=self.history_size)

        predictor = self.predictors[sensor_id]
        prediction = predictor.predict()

        # If first observation, no surprise yet
        if prediction is None:
            predictor.update(sensor_output)
            return 0.0

        # Compute prediction error
        error = self._compute_distance(sensor_output, prediction)

        # Normalize using history
        normalized_surprise = self._normalize_surprise(error, sensor_id)

        # Update predictor and history
        now = _time.time()
        predictor.update(sensor_output)
        self.surprise_history[sensor_id].append(error)
        self._timed_history[sensor_id].append(error, now)

        return normalized_surprise

    def _compute_distance(self, actual: Any, predicted: Any) -> float:
        """
        Compute distance between actual and predicted

        Handles different data types (tensors, arrays, scalars)
        """
        if isinstance(actual, torch.Tensor) and isinstance(predicted, torch.Tensor):
            # L2 distance for tensors
            dist = torch.norm(actual - predicted).item()
            return dist

        elif isinstance(actual, np.ndarray) and isinstance(predicted, np.ndarray):
            # L2 distance for arrays
            dist = np.linalg.norm(actual - predicted)
            return float(dist)

        elif isinstance(actual, (int, float)) and isinstance(predicted, (int, float)):
            # Absolute difference for scalars
            return abs(actual - predicted)

        else:
            # Unknown types - no meaningful distance
            return 0.0

    def _normalize_surprise(self, error: float, sensor_id: str) -> float:
        """
        Normalize surprise using historical distribution with time-decay weighting.

        Recent observations contribute more to the normalization distribution
        than old ones. Uses percentile-based normalization:
        - If error is at 50th percentile → surprise = 0.5
        - If error is at 95th percentile → surprise = 0.95
        """
        timed = self._timed_history.get(sensor_id)

        if timed is None or len(timed) < 10:
            return min(error / 10.0, 1.0)

        return timed.weighted_percentile_of(error, self.half_life)

    def reset_sensor(self, sensor_id: str):
        """Reset predictor for specific sensor"""
        if sensor_id in self.predictors:
            del self.predictors[sensor_id]
        if sensor_id in self.surprise_history:
            del self.surprise_history[sensor_id]
        if sensor_id in self._timed_history:
            del self._timed_history[sensor_id]

    def reset_all(self):
        """Reset all predictors"""
        self.predictors.clear()
        self.surprise_history.clear()
        self._timed_history.clear()
