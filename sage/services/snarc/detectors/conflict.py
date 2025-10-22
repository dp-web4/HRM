"""
Conflict Detector - Cross-Sensor Disagreement

Detects when different sensors provide inconsistent information.
High conflict = suspicious, might need verification.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict


class ConflictDetector:
    """
    Detect cross-sensor disagreement

    Looks for inconsistencies across sensor modalities.
    Example: Vision says "object approaching" but audio says "silence"

    Simple version: looks for sudden changes in correlation patterns.
    """

    def __init__(self, correlation_window: int = 20):
        """
        Args:
            correlation_window: How many recent observations to use
                               for correlation estimation
        """
        self.correlation_window = correlation_window

        # Recent observations per sensor
        self.sensor_buffer: Dict[str, List[Any]] = defaultdict(list)

        # Expected correlation patterns (learned)
        # Maps (sensor1, sensor2) -> expected_correlation
        self.expected_correlations: Dict[Tuple[str, str], float] = {}

    def compute(self, all_sensor_outputs: Dict[str, Any], sensor_id: str) -> float:
        """
        Compute conflict for current observation

        Args:
            all_sensor_outputs: Outputs from ALL sensors at this timestep
                               {sensor_id: sensor_output}
            sensor_id: Which sensor to compute conflict for

        Returns:
            Conflict score (0.0-1.0)
                0.0 = no conflict (all sensors agree)
                1.0 = high conflict (sensor disagrees with others)
        """
        # Update buffer for all sensors
        for sid, output in all_sensor_outputs.items():
            buffer = self.sensor_buffer[sid]
            buffer.append(output)
            # Keep only recent observations
            if len(buffer) > self.correlation_window:
                self.sensor_buffer[sid] = buffer[-self.correlation_window:]

        # Need at least 2 sensors to have conflict
        if len(all_sensor_outputs) < 2:
            return 0.0

        # Compute conflict for target sensor
        conflicts = []

        for other_id, other_output in all_sensor_outputs.items():
            if other_id == sensor_id:
                continue

            # Check if we have enough history
            if (len(self.sensor_buffer[sensor_id]) < 5 or
                len(self.sensor_buffer[other_id]) < 5):
                continue

            # Compute current correlation
            current_corr = self._compute_correlation(
                sensor_id,
                other_id,
                self.sensor_buffer[sensor_id],
                self.sensor_buffer[other_id]
            )

            # Get expected correlation
            expected_corr = self._get_expected_correlation(sensor_id, other_id)

            # Conflict = deviation from expected
            conflict = abs(current_corr - expected_corr)
            conflicts.append(conflict)

            # Update expected correlation (learn over time)
            self._update_expected_correlation(sensor_id, other_id, current_corr)

        if not conflicts:
            return 0.0

        # Return max conflict with any other sensor
        return float(np.clip(max(conflicts), 0.0, 1.0))

    def _compute_correlation(
        self,
        sensor1_id: str,
        sensor2_id: str,
        buffer1: List[Any],
        buffer2: List[Any]
    ) -> float:
        """
        Compute correlation between two sensor streams

        Returns value -1.0 to 1.0
        """
        # Convert to numeric sequences if possible
        seq1 = self._to_numeric_sequence(buffer1)
        seq2 = self._to_numeric_sequence(buffer2)

        if seq1 is None or seq2 is None:
            return 0.0  # Can't compute correlation

        if len(seq1) != len(seq2):
            return 0.0

        # Compute Pearson correlation
        try:
            corr = np.corrcoef(seq1, seq2)[0, 1]
            if np.isnan(corr):
                return 0.0
            return float(corr)
        except:
            return 0.0

    def _to_numeric_sequence(self, buffer: List[Any]) -> np.ndarray:
        """
        Convert sensor buffer to numeric sequence

        Takes magnitude/norm of each observation
        """
        try:
            numeric = []
            for obs in buffer:
                if isinstance(obs, torch.Tensor):
                    numeric.append(float(torch.norm(obs).item()))
                elif isinstance(obs, np.ndarray):
                    numeric.append(float(np.linalg.norm(obs)))
                elif isinstance(obs, (int, float)):
                    numeric.append(float(obs))
                else:
                    return None  # Can't convert

            return np.array(numeric)
        except:
            return None

    def _get_expected_correlation(self, sensor1_id: str, sensor2_id: str) -> float:
        """Get expected correlation between sensors"""
        # Order-independent lookup
        key1 = (sensor1_id, sensor2_id)
        key2 = (sensor2_id, sensor1_id)

        if key1 in self.expected_correlations:
            return self.expected_correlations[key1]
        elif key2 in self.expected_correlations:
            return self.expected_correlations[key2]
        else:
            return 0.0  # No expectation yet

    def _update_expected_correlation(
        self,
        sensor1_id: str,
        sensor2_id: str,
        observed_corr: float,
        alpha: float = 0.1
    ):
        """
        Update expected correlation using exponential moving average

        Args:
            alpha: Learning rate (higher = adapt faster to new patterns)
        """
        # Order-independent storage
        key = tuple(sorted([sensor1_id, sensor2_id]))

        if key in self.expected_correlations:
            # EMA update
            old_expected = self.expected_correlations[key]
            new_expected = alpha * observed_corr + (1 - alpha) * old_expected
            self.expected_correlations[key] = new_expected
        else:
            # First observation
            self.expected_correlations[key] = observed_corr

    def get_correlation_matrix(self, sensor_ids: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Get expected correlations between all sensor pairs

        Returns:
            Dict mapping (sensor1, sensor2) -> correlation
        """
        matrix = {}
        for i, sid1 in enumerate(sensor_ids):
            for sid2 in sensor_ids[i+1:]:
                corr = self._get_expected_correlation(sid1, sid2)
                matrix[(sid1, sid2)] = corr
        return matrix

    def reset_sensor(self, sensor_id: str):
        """Reset buffer and correlations for specific sensor"""
        if sensor_id in self.sensor_buffer:
            del self.sensor_buffer[sensor_id]

        # Remove correlations involving this sensor
        keys_to_remove = [
            k for k in self.expected_correlations.keys()
            if sensor_id in k
        ]
        for key in keys_to_remove:
            del self.expected_correlations[key]

    def reset_all(self):
        """Reset everything"""
        self.sensor_buffer.clear()
        self.expected_correlations.clear()
