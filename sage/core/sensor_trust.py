#!/usr/bin/env python3
"""
Sensor Trust System - Multi-Sensor Confidence and Reliability Tracking
=======================================================================

Implements Track 1 of Jetson Nano deployment roadmap:
- Per-sensor trust metrics (confidence scoring 0.0-1.0)
- Historical accuracy tracking
- Drift detection
- Adaptive trust adjustment
- Integration with SNARC salience for trust-weighted attention

Design Principles:
- Algorithmic trust computation (no learning required)
- Per-sensor trust tracking (not global)
- Temporal awareness (trust evolves over time)
- Graceful degradation (handle sensor failures)
- Lightweight (suitable for Nano constraints)

Trust Dimensions:
1. **Consistency**: How stable are sensor readings over time?
2. **Reliability**: Does the sensor produce valid data?
3. **Accuracy**: How well do predictions match reality?
4. **Quality**: Self-reported sensor quality scores
5. **Recency**: Recent performance weighted more heavily

Trust is computed algorithmically from:
- Prediction error history (consistency)
- Failure rate tracking (reliability)
- Cross-sensor validation (accuracy, computed at fusion level)
- Sensor-reported quality scores
- Exponential decay for recency weighting
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class TrustMetrics:
    """Trust metrics for a single sensor"""
    sensor_name: str

    # Core trust scores (0.0 - 1.0)
    consistency: float = 1.0  # Stability of readings
    reliability: float = 1.0  # Valid data rate
    accuracy: float = 1.0     # Prediction vs reality
    quality: float = 1.0      # Self-reported quality

    # Combined trust score
    trust_score: float = 1.0

    # Performance tracking
    total_readings: int = 0
    failed_readings: int = 0
    high_quality_readings: int = 0

    # Statistics
    avg_prediction_error: float = 0.0
    variance_score: float = 0.0

    # Temporal
    last_update: float = 0.0
    confidence_trend: str = "stable"  # "increasing", "stable", "decreasing"

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            'sensor_name': self.sensor_name,
            'trust_score': self.trust_score,
            'consistency': self.consistency,
            'reliability': self.reliability,
            'accuracy': self.accuracy,
            'quality': self.quality,
            'total_readings': self.total_readings,
            'failed_readings': self.failed_readings,
            'confidence_trend': self.confidence_trend
        }


@dataclass
class SensorReading:
    """Record of a single sensor reading for trust computation"""
    observation: torch.Tensor
    quality: float
    timestamp: float
    prediction_error: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class SensorTrustTracker:
    """
    Tracks trust metrics for a single sensor over time.

    Computes trust algorithmically from:
    - Reading consistency (low variance = high trust)
    - Failure rate (fewer failures = high trust)
    - Prediction accuracy (low error = high trust)
    - Self-reported quality (sensor's own confidence)
    """

    def __init__(
        self,
        sensor_name: str,
        memory_size: int = 1000,
        consistency_window: int = 10,
        decay_rate: float = 0.95,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trust tracker for a sensor.

        Args:
            sensor_name: Identifier for the sensor
            memory_size: How many readings to remember
            consistency_window: Window size for consistency computation
            decay_rate: Exponential decay for recency weighting (0-1)
            device: Torch device for computations
        """
        self.sensor_name = sensor_name
        self.memory_size = memory_size
        self.consistency_window = consistency_window
        self.decay_rate = decay_rate
        self.device = device or torch.device('cpu')

        # Reading history
        self.reading_history = deque(maxlen=memory_size)

        # Trust metrics
        self.metrics = TrustMetrics(sensor_name=sensor_name)

        # Weights for trust dimensions (can be tuned per sensor type)
        self.weights = {
            'consistency': 0.3,
            'reliability': 0.3,
            'accuracy': 0.2,
            'quality': 0.2
        }

        # Performance tracking
        self._last_trust_scores = deque(maxlen=20)  # For trend detection

    def update(
        self,
        observation: torch.Tensor,
        quality: float = 1.0,
        prediction: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None
    ) -> TrustMetrics:
        """
        Update trust metrics with a new sensor reading.

        Args:
            observation: Sensor reading (any shape)
            quality: Self-reported quality (0-1)
            prediction: Optional prediction for this reading (for accuracy)
            metadata: Optional metadata

        Returns:
            Updated TrustMetrics
        """
        timestamp = time.time()

        # Ensure observation on correct device
        if observation.device != self.device:
            observation = observation.to(self.device)

        # Compute prediction error if prediction available
        prediction_error = None
        if prediction is not None:
            if prediction.device != self.device:
                prediction = prediction.to(self.device)
            prediction_error = float(torch.mean((observation - prediction) ** 2).item())

        # Create reading record
        reading = SensorReading(
            observation=observation.clone().detach().cpu(),
            quality=quality,
            timestamp=timestamp,
            prediction_error=prediction_error,
            metadata=metadata or {}
        )

        # Store reading
        self.reading_history.append(reading)

        # Update metrics
        self.metrics.total_readings += 1
        self.metrics.last_update = timestamp

        # Track quality
        if quality >= 0.7:
            self.metrics.high_quality_readings += 1
        elif quality < 0.3:
            self.metrics.failed_readings += 1

        # Recompute trust scores
        self._compute_trust()

        return self.metrics

    def report_failure(self):
        """Report a sensor failure (invalid/missing data)"""
        self.metrics.failed_readings += 1
        self.metrics.total_readings += 1
        self.metrics.last_update = time.time()
        self._compute_trust()

    def _compute_trust(self):
        """Recompute all trust dimensions"""
        if len(self.reading_history) == 0:
            # No data yet - default to moderate trust
            self.metrics.trust_score = 0.7
            return

        # 1. CONSISTENCY: Variance in recent readings
        self.metrics.consistency = self._compute_consistency()

        # 2. RELIABILITY: Failure rate
        self.metrics.reliability = self._compute_reliability()

        # 3. ACCURACY: Prediction error (if available)
        self.metrics.accuracy = self._compute_accuracy()

        # 4. QUALITY: Average self-reported quality
        self.metrics.quality = self._compute_quality()

        # Combined trust score (weighted)
        self.metrics.trust_score = (
            self.weights['consistency'] * self.metrics.consistency +
            self.weights['reliability'] * self.metrics.reliability +
            self.weights['accuracy'] * self.metrics.accuracy +
            self.weights['quality'] * self.metrics.quality
        )

        # Apply recency weighting (recent trust matters more)
        self.metrics.trust_score = self._apply_recency_weighting(
            self.metrics.trust_score
        )

        # Track trust trend
        self._update_trend()

        # Clamp to [0, 1]
        self.metrics.trust_score = max(0.0, min(1.0, self.metrics.trust_score))

    def _compute_consistency(self) -> float:
        """
        Compute consistency from reading variance.
        Low variance = high consistency.
        """
        if len(self.reading_history) < self.consistency_window:
            return 1.0  # Not enough data - assume consistent

        # Get recent readings
        recent = list(self.reading_history)[-self.consistency_window:]

        # Stack observations
        observations = [r.observation.flatten() for r in recent]
        obs_stack = torch.stack(observations)

        # Compute coefficient of variation (std / mean)
        # Lower CV = higher consistency
        std = torch.std(obs_stack, dim=0).mean()
        mean = torch.abs(obs_stack).mean()

        if mean < 1e-6:
            # Near-zero signal - variance irrelevant
            return 1.0

        cv = float(std / (mean + 1e-6))

        # Convert CV to consistency score (0-1)
        # CV of 0 = perfect consistency (1.0)
        # CV of 1+ = very inconsistent (approaches 0)
        consistency = float(np.exp(-cv))

        self.metrics.variance_score = cv
        return consistency

    def _compute_reliability(self) -> float:
        """
        Compute reliability from failure rate.
        Fewer failures = higher reliability.
        """
        if self.metrics.total_readings == 0:
            return 1.0

        failure_rate = self.metrics.failed_readings / self.metrics.total_readings

        # Convert failure rate to reliability
        # 0% failures = 1.0 reliability
        # 50%+ failures = 0.0 reliability
        reliability = 1.0 - min(1.0, 2.0 * failure_rate)

        return reliability

    def _compute_accuracy(self) -> float:
        """
        Compute accuracy from prediction errors.
        Lower error = higher accuracy.
        """
        # Get readings with prediction errors
        errors = [r.prediction_error for r in self.reading_history
                  if r.prediction_error is not None]

        if len(errors) == 0:
            # No prediction data - assume moderate accuracy
            return 0.8

        # Average prediction error (recent weighted more)
        weights = np.array([self.decay_rate ** (len(errors) - i - 1)
                           for i in range(len(errors))])
        weights = weights / weights.sum()

        avg_error = float(np.average(errors, weights=weights))
        self.metrics.avg_prediction_error = avg_error

        # Convert error to accuracy score
        # Error of 0 = perfect accuracy (1.0)
        # Error of 1+ = poor accuracy (approaches 0)
        accuracy = float(np.exp(-avg_error))

        return accuracy

    def _compute_quality(self) -> float:
        """
        Compute average quality from self-reported scores.
        """
        if len(self.reading_history) == 0:
            return 1.0

        # Recent quality scores (exponentially weighted)
        qualities = [r.quality for r in self.reading_history]

        weights = np.array([self.decay_rate ** (len(qualities) - i - 1)
                           for i in range(len(qualities))])
        weights = weights / weights.sum()

        avg_quality = float(np.average(qualities, weights=weights))

        return avg_quality

    def _apply_recency_weighting(self, trust_score: float) -> float:
        """
        Apply recency weighting to trust score.
        Recent performance matters more than distant past.
        """
        if len(self.reading_history) < 5:
            return trust_score

        # Compute recent vs overall performance
        recent_5 = list(self.reading_history)[-5:]
        recent_quality = np.mean([r.quality for r in recent_5])

        overall_quality = np.mean([r.quality for r in self.reading_history])

        # If recent quality much worse, reduce trust
        if recent_quality < overall_quality - 0.2:
            trust_score *= 0.8
        # If recent quality much better, increase trust
        elif recent_quality > overall_quality + 0.2:
            trust_score *= 1.1

        return trust_score

    def _update_trend(self):
        """Update confidence trend indicator"""
        self._last_trust_scores.append(self.metrics.trust_score)

        if len(self._last_trust_scores) < 10:
            self.metrics.confidence_trend = "stable"
            return

        recent_10 = list(self._last_trust_scores)
        first_half = np.mean(recent_10[:5])
        second_half = np.mean(recent_10[5:])

        diff = second_half - first_half

        if diff > 0.1:
            self.metrics.confidence_trend = "increasing"
        elif diff < -0.1:
            self.metrics.confidence_trend = "decreasing"
        else:
            self.metrics.confidence_trend = "stable"

    def get_trust_score(self) -> float:
        """Get current trust score (0-1)"""
        return self.metrics.trust_score

    def get_metrics(self) -> TrustMetrics:
        """Get full trust metrics"""
        return self.metrics

    def reset(self):
        """Reset trust tracker (useful for testing)"""
        self.reading_history.clear()
        self.metrics = TrustMetrics(sensor_name=self.sensor_name)
        self._last_trust_scores.clear()


class MultiSensorTrustSystem:
    """
    Central trust management for multiple sensors.

    Tracks trust for each sensor independently and provides
    aggregated trust information for sensor fusion.
    """

    def __init__(
        self,
        memory_size: int = 1000,
        consistency_window: int = 10,
        decay_rate: float = 0.95,
        device: Optional[torch.device] = None
    ):
        """
        Initialize multi-sensor trust system.

        Args:
            memory_size: Readings to remember per sensor
            consistency_window: Window for consistency computation
            decay_rate: Recency weighting decay
            device: Torch device
        """
        self.memory_size = memory_size
        self.consistency_window = consistency_window
        self.decay_rate = decay_rate
        self.device = device or torch.device('cpu')

        # Per-sensor trust trackers
        self.trackers: Dict[str, SensorTrustTracker] = {}

    def register_sensor(self, sensor_name: str):
        """Register a new sensor for trust tracking"""
        if sensor_name not in self.trackers:
            self.trackers[sensor_name] = SensorTrustTracker(
                sensor_name=sensor_name,
                memory_size=self.memory_size,
                consistency_window=self.consistency_window,
                decay_rate=self.decay_rate,
                device=self.device
            )

    def update(
        self,
        sensor_name: str,
        observation: torch.Tensor,
        quality: float = 1.0,
        prediction: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None
    ) -> TrustMetrics:
        """
        Update trust for a sensor.

        Auto-registers sensor if not already registered.
        """
        if sensor_name not in self.trackers:
            self.register_sensor(sensor_name)

        return self.trackers[sensor_name].update(
            observation=observation,
            quality=quality,
            prediction=prediction,
            metadata=metadata
        )

    def report_failure(self, sensor_name: str):
        """Report a sensor failure"""
        if sensor_name not in self.trackers:
            self.register_sensor(sensor_name)

        self.trackers[sensor_name].report_failure()

    def get_trust_score(self, sensor_name: str) -> float:
        """Get trust score for a sensor (0-1)"""
        if sensor_name not in self.trackers:
            return 0.5  # Unknown sensor - moderate trust

        return self.trackers[sensor_name].get_trust_score()

    def get_all_trust_scores(self) -> Dict[str, float]:
        """Get trust scores for all sensors"""
        return {
            name: tracker.get_trust_score()
            for name, tracker in self.trackers.items()
        }

    def get_metrics(self, sensor_name: str) -> Optional[TrustMetrics]:
        """Get full metrics for a sensor"""
        if sensor_name not in self.trackers:
            return None

        return self.trackers[sensor_name].get_metrics()

    def get_all_metrics(self) -> Dict[str, TrustMetrics]:
        """Get metrics for all sensors"""
        return {
            name: tracker.get_metrics()
            for name, tracker in self.trackers.items()
        }

    def get_most_trusted_sensor(self) -> Optional[Tuple[str, float]]:
        """Get the most trusted sensor"""
        if len(self.trackers) == 0:
            return None

        trust_scores = self.get_all_trust_scores()
        best_sensor = max(trust_scores.items(), key=lambda x: x[1])

        return best_sensor

    def get_degraded_sensors(self, threshold: float = 0.5) -> List[str]:
        """Get list of sensors with trust below threshold"""
        return [
            name for name, score in self.get_all_trust_scores().items()
            if score < threshold
        ]

    def get_stats(self) -> Dict:
        """Get system-wide statistics"""
        if len(self.trackers) == 0:
            return {
                'num_sensors': 0,
                'avg_trust': 0.0,
                'total_readings': 0
            }

        trust_scores = self.get_all_trust_scores()
        all_metrics = self.get_all_metrics()

        return {
            'num_sensors': len(self.trackers),
            'avg_trust': np.mean(list(trust_scores.values())),
            'min_trust': min(trust_scores.values()),
            'max_trust': max(trust_scores.values()),
            'total_readings': sum(m.total_readings for m in all_metrics.values()),
            'total_failures': sum(m.failed_readings for m in all_metrics.values()),
            'degraded_sensors': self.get_degraded_sensors(),
            'most_trusted': self.get_most_trusted_sensor()
        }


if __name__ == "__main__":
    # Test sensor trust system
    print("Testing Sensor Trust System\n")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trust_system = MultiSensorTrustSystem(device=device)

    # Simulate two sensors: one reliable, one degrading
    print("\n1. Simulating reliable 'vision' sensor...")
    for i in range(50):
        # Consistent readings with high quality
        obs = torch.randn(10, device=device) + 5.0
        trust_system.update('vision', obs, quality=0.9)

    print("\n2. Simulating degrading 'audio' sensor...")
    for i in range(50):
        # Increasingly noisy readings
        noise_level = 0.1 * (i / 50)  # Increasing noise
        quality = 1.0 - noise_level
        obs = torch.randn(10, device=device) * (1 + noise_level)
        trust_system.update('audio', obs, quality=quality)

    print("\n3. Simulating unreliable 'proprioception' sensor...")
    for i in range(50):
        if i % 5 == 0:
            # Every 5th reading fails
            trust_system.report_failure('proprioception')
        else:
            obs = torch.randn(14, device=device)
            trust_system.update('proprioception', obs, quality=0.7)

    # Get trust scores
    print("\n" + "="*60)
    print("TRUST SCORES:")
    print("="*60)

    for sensor_name, trust_score in trust_system.get_all_trust_scores().items():
        metrics = trust_system.get_metrics(sensor_name)
        print(f"\n{sensor_name.upper()}:")
        print(f"  Trust Score: {trust_score:.3f}")
        print(f"  Consistency: {metrics.consistency:.3f}")
        print(f"  Reliability: {metrics.reliability:.3f}")
        print(f"  Accuracy: {metrics.accuracy:.3f}")
        print(f"  Quality: {metrics.quality:.3f}")
        print(f"  Trend: {metrics.confidence_trend}")
        print(f"  Readings: {metrics.total_readings} (failures: {metrics.failed_readings})")

    # System stats
    print("\n" + "="*60)
    print("SYSTEM STATISTICS:")
    print("="*60)
    stats = trust_system.get_stats()
    print(f"  Sensors: {stats['num_sensors']}")
    print(f"  Average Trust: {stats['avg_trust']:.3f}")
    print(f"  Min Trust: {stats['min_trust']:.3f}")
    print(f"  Max Trust: {stats['max_trust']:.3f}")
    print(f"  Total Readings: {stats['total_readings']}")
    print(f"  Total Failures: {stats['total_failures']}")
    print(f"  Most Trusted: {stats['most_trusted'][0]} ({stats['most_trusted'][1]:.3f})")
    print(f"  Degraded Sensors: {stats['degraded_sensors']}")

    print("\n" + "="*60)
    print("✓ Sensor Trust System test complete!")
    print("="*60)
