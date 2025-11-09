#!/usr/bin/env python3
"""
Sensor Fusion Engine - Trust-Weighted Multi-Modal Integration
===============================================================

Implements Track 1 of Jetson Nano deployment roadmap:
- Trust-weighted sensor combination
- Conflict detection between sensors
- Fallback strategies for sensor failures
- Cross-modal validation
- Integration with SNARC for conflict scoring

Design Principles:
- Trust-weighted fusion (high-trust sensors weighted more)
- Conflict detection (sensors disagree)
- Graceful degradation (handle sensor failures)
- Modality-aware (different sensor types handled appropriately)
- Lightweight (Nano-compatible)

Fusion Strategies:
1. **Weighted Average**: Combine sensors weighted by trust
2. **Conflict Resolution**: Detect and resolve disagreements
3. **Fallback Selection**: Use most-trusted sensor when conflicts unresolvable
4. **Cross-Validation**: Sensors validate each other

Integration with SNARC:
- Conflict dimension computed from sensor disagreement
- Trust scores modulate attention allocation
- Failed sensors generate high salience (need attention)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time

# Import from sensor_trust (same directory)
try:
    from sage.core.sensor_trust import (
        MultiSensorTrustSystem,
        TrustMetrics
    )
except ModuleNotFoundError:
    # Relative import for standalone testing
    from sensor_trust import (
        MultiSensorTrustSystem,
        TrustMetrics
    )


@dataclass
class FusionResult:
    """Result of multi-sensor fusion"""
    # Fused data
    fused_observation: Optional[torch.Tensor] = None

    # Trust and conflict
    fusion_confidence: float = 0.0  # Overall confidence in fusion (0-1)
    conflict_score: float = 0.0      # Degree of sensor disagreement (0-1)

    # Sensor contributions
    sensor_weights: Dict[str, float] = field(default_factory=dict)
    sensors_used: List[str] = field(default_factory=list)
    sensors_failed: List[str] = field(default_factory=list)

    # Strategy used
    fusion_strategy: str = "none"  # "weighted_average", "fallback", "majority"

    # Metadata
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            'fusion_confidence': self.fusion_confidence,
            'conflict_score': self.conflict_score,
            'sensor_weights': self.sensor_weights,
            'sensors_used': self.sensors_used,
            'sensors_failed': self.sensors_failed,
            'fusion_strategy': self.fusion_strategy,
            'timestamp': self.timestamp
        }


class SensorFusionEngine:
    """
    Multi-sensor fusion engine with trust-weighted combination.

    Combines observations from multiple sensors weighted by their
    trust scores. Detects conflicts and handles sensor failures gracefully.
    """

    def __init__(
        self,
        trust_system: MultiSensorTrustSystem,
        conflict_threshold: float = 0.3,
        min_confidence: float = 0.4,
        device: Optional[torch.device] = None
    ):
        """
        Initialize sensor fusion engine.

        Args:
            trust_system: Trust tracking system for sensors
            conflict_threshold: Threshold for detecting conflicts (0-1)
            min_confidence: Minimum confidence to trust fusion result
            device: Torch device
        """
        self.trust_system = trust_system
        self.conflict_threshold = conflict_threshold
        self.min_confidence = min_confidence
        self.device = device or torch.device('cpu')

        # Statistics
        self._fusion_count = 0
        self._conflict_count = 0
        self._fallback_count = 0

    def fuse(
        self,
        observations: Dict[str, torch.Tensor],
        qualities: Optional[Dict[str, float]] = None,
        modality_group: Optional[str] = None
    ) -> FusionResult:
        """
        Fuse observations from multiple sensors.

        Args:
            observations: Dict mapping sensor_name -> observation tensor
            qualities: Optional dict mapping sensor_name -> quality score
            modality_group: Optional grouping (e.g., "vision", "audio")

        Returns:
            FusionResult with fused observation and metadata
        """
        self._fusion_count += 1
        timestamp = time.time()

        if len(observations) == 0:
            # No sensors - empty result
            return FusionResult(
                fusion_confidence=0.0,
                fusion_strategy="none",
                timestamp=timestamp
            )

        # Ensure qualities dict exists
        qualities = qualities or {}

        # Single sensor - no fusion needed
        if len(observations) == 1:
            sensor_name = list(observations.keys())[0]
            obs = observations[sensor_name]
            quality = qualities.get(sensor_name, 1.0)

            # Update trust
            self.trust_system.update(sensor_name, obs, quality=quality)

            return FusionResult(
                fused_observation=obs,
                fusion_confidence=quality,
                conflict_score=0.0,
                sensor_weights={sensor_name: 1.0},
                sensors_used=[sensor_name],
                fusion_strategy="single_sensor",
                timestamp=timestamp
            )

        # Multi-sensor fusion
        return self._fuse_multi_sensor(observations, qualities, timestamp)

    def _fuse_multi_sensor(
        self,
        observations: Dict[str, torch.Tensor],
        qualities: Dict[str, float],
        timestamp: float
    ) -> FusionResult:
        """
        Fuse multiple sensors with trust-weighted combination.
        """
        # Update trust for all sensors
        for sensor_name, obs in observations.items():
            quality = qualities.get(sensor_name, 1.0)
            self.trust_system.update(sensor_name, obs, quality=quality)

        # Get trust scores
        trust_scores = {
            name: self.trust_system.get_trust_score(name)
            for name in observations.keys()
        }

        # Filter out failed sensors (very low trust)
        valid_sensors = {
            name: obs for name, obs in observations.items()
            if trust_scores[name] > 0.2
        }

        sensors_failed = [
            name for name in observations.keys()
            if name not in valid_sensors
        ]

        # If no valid sensors, use fallback
        if len(valid_sensors) == 0:
            return self._fallback_fusion(observations, trust_scores, timestamp)

        # Detect conflicts
        conflict_score = self._compute_conflict(valid_sensors)

        # If high conflict, use conflict resolution
        if conflict_score > self.conflict_threshold:
            self._conflict_count += 1
            return self._resolve_conflict(
                valid_sensors,
                trust_scores,
                conflict_score,
                sensors_failed,
                timestamp
            )

        # Normal weighted fusion
        fused_obs, weights, confidence = self._weighted_fusion(
            valid_sensors,
            trust_scores
        )

        return FusionResult(
            fused_observation=fused_obs,
            fusion_confidence=confidence,
            conflict_score=conflict_score,
            sensor_weights=weights,
            sensors_used=list(valid_sensors.keys()),
            sensors_failed=sensors_failed,
            fusion_strategy="weighted_average",
            timestamp=timestamp,
            metadata={
                'num_sensors': len(valid_sensors),
                'trust_variance': float(np.var(list(trust_scores.values())))
            }
        )

    def _weighted_fusion(
        self,
        observations: Dict[str, torch.Tensor],
        trust_scores: Dict[str, float]
    ) -> Tuple[torch.Tensor, Dict[str, float], float]:
        """
        Weighted average fusion based on trust scores.

        Returns:
            (fused_observation, weights, confidence)
        """
        # Normalize trust scores to weights
        total_trust = sum(trust_scores[name] for name in observations.keys())
        if total_trust < 1e-6:
            # All sensors have zero trust - equal weights
            total_trust = len(observations)
            weights = {name: 1.0 / total_trust for name in observations.keys()}
        else:
            weights = {
                name: trust_scores[name] / total_trust
                for name in observations.keys()
            }

        # Ensure all observations are on correct device and same shape
        observations_aligned = self._align_observations(observations)

        # Weighted sum
        fused = None
        for name, obs in observations_aligned.items():
            weighted_obs = obs * weights[name]
            if fused is None:
                fused = weighted_obs
            else:
                fused = fused + weighted_obs

        # Confidence is weighted average of trust scores
        confidence = float(sum(weights[name] * trust_scores[name]
                              for name in observations.keys()))

        return fused, weights, confidence

    def _compute_conflict(
        self,
        observations: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute conflict score from sensor disagreement.

        Conflict = variance in observations (high variance = high conflict)

        Returns:
            Conflict score (0-1)
        """
        if len(observations) < 2:
            return 0.0

        # Align observations to same shape
        observations_aligned = self._align_observations(observations)

        # Stack observations
        obs_list = [obs.flatten() for obs in observations_aligned.values()]
        obs_stack = torch.stack(obs_list)

        # Compute coefficient of variation
        std = torch.std(obs_stack, dim=0).mean()
        mean = torch.abs(obs_stack).mean()

        if mean < 1e-6:
            return 0.0  # Near-zero signal - no conflict

        cv = float(std / (mean + 1e-6))

        # Convert to [0, 1] conflict score
        # CV of 0 = no conflict
        # CV of 1+ = high conflict
        conflict = min(1.0, cv)

        return conflict

    def _resolve_conflict(
        self,
        observations: Dict[str, torch.Tensor],
        trust_scores: Dict[str, float],
        conflict_score: float,
        sensors_failed: List[str],
        timestamp: float
    ) -> FusionResult:
        """
        Resolve conflict between sensors.

        Strategy: Use most-trusted sensor's observation.
        """
        # Find most-trusted sensor
        best_sensor = max(trust_scores.items(), key=lambda x: x[1])
        sensor_name, trust = best_sensor

        return FusionResult(
            fused_observation=observations[sensor_name],
            fusion_confidence=trust,
            conflict_score=conflict_score,
            sensor_weights={sensor_name: 1.0},
            sensors_used=[sensor_name],
            sensors_failed=sensors_failed,
            fusion_strategy="conflict_resolution_fallback",
            timestamp=timestamp,
            metadata={
                'conflict_reason': 'high_disagreement',
                'fallback_sensor': sensor_name,
                'num_conflicting': len(observations)
            }
        )

    def _fallback_fusion(
        self,
        observations: Dict[str, torch.Tensor],
        trust_scores: Dict[str, float],
        timestamp: float
    ) -> FusionResult:
        """
        Fallback fusion when all sensors have low trust.

        Strategy: Use least-bad sensor.
        """
        self._fallback_count += 1

        # Find least-bad sensor
        best_sensor = max(trust_scores.items(), key=lambda x: x[1])
        sensor_name, trust = best_sensor

        return FusionResult(
            fused_observation=observations[sensor_name],
            fusion_confidence=trust,
            conflict_score=0.0,
            sensor_weights={sensor_name: 1.0},
            sensors_used=[sensor_name],
            sensors_failed=[name for name in observations.keys() if name != sensor_name],
            fusion_strategy="fallback_degraded",
            timestamp=timestamp,
            metadata={
                'fallback_reason': 'all_sensors_low_trust',
                'fallback_sensor': sensor_name
            }
        )

    def _align_observations(
        self,
        observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Align observations to same shape and device.

        Handles different sensor dimensionalities by:
        1. Flattening all to 1D
        2. Padding to max length
        3. Moving to correct device
        """
        aligned = {}

        # Find max dimension
        max_dim = max(obs.numel() for obs in observations.values())

        for name, obs in observations.items():
            # Move to correct device
            if obs.device != self.device:
                obs = obs.to(self.device)

            # Flatten
            obs_flat = obs.flatten()

            # Pad if needed
            if obs_flat.numel() < max_dim:
                padding = torch.zeros(
                    max_dim - obs_flat.numel(),
                    device=self.device,
                    dtype=obs_flat.dtype
                )
                obs_flat = torch.cat([obs_flat, padding])

            aligned[name] = obs_flat

        return aligned

    def get_stats(self) -> Dict:
        """Get fusion statistics"""
        return {
            'total_fusions': self._fusion_count,
            'conflicts_detected': self._conflict_count,
            'fallbacks': self._fallback_count,
            'conflict_rate': self._conflict_count / max(1, self._fusion_count),
            'fallback_rate': self._fallback_count / max(1, self._fusion_count)
        }


class CrossModalValidator:
    """
    Cross-modal validation between sensors.

    Uses one modality to validate another (e.g., vision validates proprioception).
    """

    def __init__(
        self,
        trust_system: MultiSensorTrustSystem,
        device: Optional[torch.device] = None
    ):
        """
        Initialize cross-modal validator.

        Args:
            trust_system: Trust system for sensors
            device: Torch device
        """
        self.trust_system = trust_system
        self.device = device or torch.device('cpu')

        # Validation history
        self._validation_history = []

    def validate(
        self,
        sensor_a: str,
        observation_a: torch.Tensor,
        sensor_b: str,
        observation_b: torch.Tensor,
        correlation_expected: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Validate sensor_a using sensor_b.

        Args:
            sensor_a: Sensor to validate
            observation_a: Observation from sensor A
            sensor_b: Reference sensor
            observation_b: Observation from sensor B
            correlation_expected: Expected correlation (0-1)

        Returns:
            (validation_passed, confidence)
        """
        # Compute correlation
        if observation_a.numel() != observation_b.numel():
            # Different sizes - align
            min_size = min(observation_a.numel(), observation_b.numel())
            obs_a = observation_a.flatten()[:min_size]
            obs_b = observation_b.flatten()[:min_size]
        else:
            obs_a = observation_a.flatten()
            obs_b = observation_b.flatten()

        # Correlation
        correlation = float(torch.corrcoef(torch.stack([obs_a, obs_b]))[0, 1])

        # Validation passes if correlation close to expected
        validation_passed = abs(correlation - correlation_expected) < 0.3

        # Confidence based on sensor B's trust
        confidence = self.trust_system.get_trust_score(sensor_b)

        # Store validation
        self._validation_history.append({
            'sensor_a': sensor_a,
            'sensor_b': sensor_b,
            'correlation': correlation,
            'passed': validation_passed,
            'confidence': confidence,
            'timestamp': time.time()
        })

        return validation_passed, confidence


if __name__ == "__main__":
    # Test sensor fusion system
    print("Testing Sensor Fusion Engine\n")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create trust system and fusion engine
    trust_system = MultiSensorTrustSystem(device=device)
    fusion_engine = SensorFusionEngine(trust_system, device=device)

    # Test 1: Single sensor fusion
    print("\n1. Single sensor fusion...")
    result = fusion_engine.fuse({
        'vision': torch.randn(10, device=device)
    })
    print(f"   Strategy: {result.fusion_strategy}")
    print(f"   Confidence: {result.fusion_confidence:.3f}")
    print(f"   Conflict: {result.conflict_score:.3f}")

    # Test 2: Multi-sensor fusion with agreement
    print("\n2. Multi-sensor fusion (agreement)...")
    base = torch.randn(10, device=device)
    observations = {
        'vision': base + torch.randn(10, device=device) * 0.1,
        'proprioception': base + torch.randn(10, device=device) * 0.1
    }
    qualities = {'vision': 0.9, 'proprioception': 0.8}
    result = fusion_engine.fuse(observations, qualities)
    print(f"   Strategy: {result.fusion_strategy}")
    print(f"   Confidence: {result.fusion_confidence:.3f}")
    print(f"   Conflict: {result.conflict_score:.3f}")
    print(f"   Sensors: {result.sensors_used}")
    print(f"   Weights: {result.sensor_weights}")

    # Test 3: Multi-sensor fusion with conflict
    print("\n3. Multi-sensor fusion (conflict)...")
    observations = {
        'vision': torch.randn(10, device=device) + 5.0,
        'proprioception': torch.randn(10, device=device) - 5.0
    }
    qualities = {'vision': 0.9, 'proprioception': 0.7}
    result = fusion_engine.fuse(observations, qualities)
    print(f"   Strategy: {result.fusion_strategy}")
    print(f"   Confidence: {result.fusion_confidence:.3f}")
    print(f"   Conflict: {result.conflict_score:.3f} ⚠️")
    print(f"   Sensors used: {result.sensors_used}")

    # Test 4: Graceful degradation
    print("\n4. Graceful degradation (sensor failure)...")
    # Build trust first
    for i in range(20):
        fusion_engine.fuse({
            'vision': torch.randn(10, device=device),
            'audio': torch.randn(10, device=device)
        }, qualities={'vision': 0.9, 'audio': 0.9})

    # Now fail audio sensor repeatedly
    for i in range(10):
        trust_system.report_failure('audio')

    # Fusion should degrade gracefully
    observations = {
        'vision': torch.randn(10, device=device),
        'audio': torch.randn(10, device=device)
    }
    qualities = {'vision': 0.9, 'audio': 0.1}
    result = fusion_engine.fuse(observations, qualities)
    print(f"   Strategy: {result.fusion_strategy}")
    print(f"   Sensors used: {result.sensors_used}")
    print(f"   Sensors failed: {result.sensors_failed}")
    print(f"   Confidence: {result.fusion_confidence:.3f}")

    # Test 5: Cross-modal validation
    print("\n5. Cross-modal validation...")
    validator = CrossModalValidator(trust_system, device=device)

    # Correlated observations (should validate)
    base = torch.randn(10, device=device)
    obs_a = base + torch.randn(10, device=device) * 0.1
    obs_b = base + torch.randn(10, device=device) * 0.1

    passed, confidence = validator.validate('vision', obs_a, 'proprioception', obs_b, correlation_expected=0.8)
    print(f"   Validation passed: {passed} ✓" if passed else f"   Validation passed: {passed} ✗")
    print(f"   Confidence: {confidence:.3f}")

    # Statistics
    print("\n" + "="*60)
    print("FUSION STATISTICS:")
    print("="*60)
    stats = fusion_engine.get_stats()
    print(f"  Total fusions: {stats['total_fusions']}")
    print(f"  Conflicts detected: {stats['conflicts_detected']}")
    print(f"  Fallbacks: {stats['fallbacks']}")
    print(f"  Conflict rate: {stats['conflict_rate']:.1%}")

    trust_stats = trust_system.get_stats()
    print(f"\n  Registered sensors: {trust_stats['num_sensors']}")
    print(f"  Average trust: {trust_stats['avg_trust']:.3f}")
    print(f"  Most trusted: {trust_stats['most_trusted'][0]} ({trust_stats['most_trusted'][1]:.3f})")

    print("\n" + "="*60)
    print("✓ Sensor Fusion Engine test complete!")
    print("="*60)
