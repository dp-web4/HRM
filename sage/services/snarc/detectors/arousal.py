"""
Arousal Detector - Signal Magnitude

Detects intensity/urgency of sensor signals.
High magnitude = high arousal (demands immediate attention).
"""

import torch
import numpy as np
from typing import Any, Dict
from collections import deque


class ArousalDetector:
    """
    Detect intensity/urgency of signals

    Computes magnitude of sensor output, normalized by
    historical distribution to account for sensor-specific ranges.
    """

    def __init__(self, history_size: int = 100):
        """
        Args:
            history_size: How many past magnitudes to keep for normalization
        """
        self.history_size = history_size

        # Magnitude history per sensor (for normalization)
        self.magnitude_history: Dict[str, deque] = {}

    def compute(self, sensor_output: Any, sensor_id: str) -> float:
        """
        Compute arousal from signal magnitude

        Args:
            sensor_output: Current sensor reading
            sensor_id: Unique sensor identifier

        Returns:
            Arousal score (0.0-1.0)
                0.0 = low intensity/calm
                1.0 = high intensity/urgent
        """
        # Get or create history for this sensor
        if sensor_id not in self.magnitude_history:
            self.magnitude_history[sensor_id] = deque(maxlen=self.history_size)

        # Compute raw magnitude
        magnitude = self._compute_magnitude(sensor_output)

        # Normalize using history
        normalized_arousal = self._normalize_arousal(magnitude, sensor_id)

        # Store magnitude
        self.magnitude_history[sensor_id].append(magnitude)

        return normalized_arousal

    def _compute_magnitude(self, data: Any) -> float:
        """
        Compute magnitude of sensor signal

        Different computation based on data type
        """
        if isinstance(data, torch.Tensor):
            # L2 norm for tensors
            return float(torch.norm(data).item())

        elif isinstance(data, np.ndarray):
            # L2 norm for arrays
            return float(np.linalg.norm(data))

        elif isinstance(data, (int, float)):
            # Absolute value for scalars
            return abs(float(data))

        elif isinstance(data, (list, tuple)):
            # L2 norm for sequences
            arr = np.array(data)
            return float(np.linalg.norm(arr))

        else:
            # Unknown type - return moderate arousal
            return 0.5

    def _normalize_arousal(self, magnitude: float, sensor_id: str) -> float:
        """
        Normalize arousal using historical magnitudes

        Uses percentile-based normalization like SurpriseDetector
        """
        history = self.magnitude_history[sensor_id]

        if len(history) < 10:
            # Not enough history - simple normalization
            # Assume typical range is 0-10
            return min(magnitude / 10.0, 1.0)

        # Compute percentile
        sorted_history = sorted(history)
        rank = sum(1 for h in sorted_history if h < magnitude)
        percentile = rank / len(sorted_history)

        return percentile

    def get_statistics(self, sensor_id: str) -> Dict[str, float]:
        """
        Get arousal statistics for sensor

        Returns:
            Dict with mean, std, min, max of historical magnitudes
        """
        if sensor_id not in self.magnitude_history:
            return {}

        history = list(self.magnitude_history[sensor_id])
        if not history:
            return {}

        return {
            'mean': float(np.mean(history)),
            'std': float(np.std(history)),
            'min': float(np.min(history)),
            'max': float(np.max(history)),
            'current': history[-1] if history else 0.0
        }

    def reset_sensor(self, sensor_id: str):
        """Reset history for specific sensor"""
        if sensor_id in self.magnitude_history:
            del self.magnitude_history[sensor_id]

    def reset_all(self):
        """Reset all history"""
        self.magnitude_history.clear()
