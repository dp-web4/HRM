"""
SNARC Compression Module
========================

Implements compression-action-threshold pattern for SAGE attention allocation.

**Pattern**:
```
Multi-dimensional sensor inputs (high-D)
  ↓ (SNARC compression function)
Scalar salience score [0, 1]
  ↓ (metabolic-state-dependent threshold)
Binary attention decision (attend or ignore)
```

**SNARC Dimensions**:
- **S**urprise: Unexpected deviation from prediction
- **N**ovelty: Haven't seen this before
- **A**rousal: Physiological activation (computational load, errors)
- **R**eward: Positive valence (goal achievement, progress)
- **C**onflict: Competing hypotheses, uncertainty

**References**:
- ATTENTION_COMPRESSION_DESIGN.md (Dec 5, 2025)
- Compression-Action-Threshold universal pattern
- Synchronism coherence function (tanh compression)

**Session**: Thor Autonomous Research (2025-12-05)
**Author**: Claude (guest) on Thor via claude-code
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from enum import Enum
import math
from collections import deque


# ============================================================================
# SNARC Compression Components
# ============================================================================

@dataclass
class SNARCDimensions:
    """Five SNARC dimensions for attention allocation"""
    surprise: float = 0.0    # 0-1: prediction error magnitude
    novelty: float = 0.0     # 0-1: memory mismatch
    arousal: float = 0.0     # 0-1: computational activation
    reward: float = 0.0      # 0-1: goal proximity/achievement
    conflict: float = 0.0    # 0-1: hypothesis uncertainty

    def __post_init__(self):
        """Validate all dimensions are in [0, 1]"""
        for name, value in self.__dict__.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"SNARC dimension {name}={value} outside [0, 1]")


@dataclass
class SNARCWeights:
    """Learned weights for SNARC compression"""
    surprise: float = 0.25
    novelty: float = 0.20
    arousal: float = 0.20
    reward: float = 0.20
    conflict: float = 0.15

    def normalize(self):
        """Ensure weights sum to 1.0"""
        total = sum([self.surprise, self.novelty, self.arousal, self.reward, self.conflict])
        if total > 0:
            self.surprise /= total
            self.novelty /= total
            self.arousal /= total
            self.reward /= total
            self.conflict /= total

    def as_dict(self) -> Dict[str, float]:
        return {
            'surprise': self.surprise,
            'novelty': self.novelty,
            'arousal': self.arousal,
            'reward': self.reward,
            'conflict': self.conflict
        }


class CompressionMode(Enum):
    """Compression function type"""
    LINEAR = "linear"          # Direct weighted sum
    SATURATING = "saturating"  # tanh-based compression (outlier resistant)


# ============================================================================
# SNARC Compressor
# ============================================================================

class SNARCCompressor:
    """
    Compresses high-dimensional sensor data to scalar salience via SNARC.

    Implements compression-action-threshold pattern:
    - Input: Multi-dimensional sensor readings
    - Compression: SNARC dimensions → scalar salience
    - Output: Salience ∈ [0, 1]
    """

    def __init__(
        self,
        weights: Optional[SNARCWeights] = None,
        compression_mode: CompressionMode = CompressionMode.LINEAR,
        saturation_gain: float = 3.0
    ):
        """
        Initialize SNARC compressor.

        Args:
            weights: SNARC dimension weights (default: balanced)
            compression_mode: Linear or saturating compression
            saturation_gain: Gain for tanh saturation (higher = steeper)
        """
        self.weights = weights or SNARCWeights()
        self.weights.normalize()
        self.compression_mode = compression_mode
        self.saturation_gain = saturation_gain

        # History for learning
        self.salience_history: deque = deque(maxlen=100)
        self.outcome_history: deque = deque(maxlen=100)

    def compute_surprise(self, sensor_data: Dict) -> float:
        """
        Surprise: Unexpected deviation from prediction.

        High surprise when:
        - Urgent conditions arise unexpectedly
        - Values deviate from recent averages
        - Rare events occur
        """
        surprise = 0.0

        # Urgent conditions are surprising
        if sensor_data.get('urgent_count', 0) > 0:
            surprise += 0.5

        # Check novelty score if available (indicates unexpectedness)
        novelty_score = sensor_data.get('novelty_score', 0.0)
        if novelty_score > 0.5:
            surprise += 0.3 * novelty_score

        # Large deviations are surprising
        if sensor_data.get('deviation', 0.0) > 0.7:
            surprise += 0.3

        return min(1.0, surprise)

    def compute_novelty(self, sensor_data: Dict) -> float:
        """
        Novelty: Haven't seen this before.

        High novelty when:
        - New patterns in sensor data
        - First occurrence of condition
        - Memory retrieval fails (no similar past)
        """
        novelty = 0.0

        # Novelty score from sensor
        novelty_score = sensor_data.get('novelty_score', 0.0)
        novelty += novelty_score

        # First-time conditions
        if sensor_data.get('first_seen', False):
            novelty += 0.4

        # Memory mismatch
        if sensor_data.get('memory_match', 1.0) < 0.3:
            novelty += 0.3

        return min(1.0, novelty)

    def compute_arousal(self, sensor_data: Dict) -> float:
        """
        Arousal: Computational/physiological activation.

        High arousal when:
        - System load is high
        - Error conditions present
        - Resource pressure (ATP, memory)
        """
        arousal = 0.0

        # Urgent conditions trigger arousal
        if sensor_data.get('urgent_count', 0) > 0:
            arousal += 0.4

        # Resource pressure
        atp_util = sensor_data.get('atp_utilization', 0.0)
        if atp_util > 0.7:
            arousal += 0.3

        # Error conditions
        if sensor_data.get('error_count', 0) > 0:
            arousal += 0.4

        # High count indicates activation
        count = sensor_data.get('count', 0)
        if count > 10:
            arousal += 0.2
        elif count > 5:
            arousal += 0.1

        return min(1.0, arousal)

    def compute_reward(self, sensor_data: Dict) -> float:
        """
        Reward: Positive valence, goal achievement.

        High reward when:
        - Goal progress detected
        - Task completion signals
        - Positive feedback
        """
        reward = 0.0

        # Explicit reward signal
        reward_signal = sensor_data.get('reward', 0.0)
        reward += reward_signal

        # Goal proximity
        goal_proximity = sensor_data.get('goal_proximity', 0.0)
        reward += 0.3 * goal_proximity

        # Positive feedback
        if sensor_data.get('positive_feedback', False):
            reward += 0.4

        # Success indicators
        if sensor_data.get('success', False):
            reward += 0.5

        return min(1.0, reward)

    def compute_conflict(self, sensor_data: Dict) -> float:
        """
        Conflict: Competing hypotheses, uncertainty.

        High conflict when:
        - Multiple interpretations possible
        - Uncertainty in measurements
        - Contradictory signals
        """
        conflict = 0.0

        # Explicit uncertainty
        uncertainty = sensor_data.get('uncertainty', 0.0)
        conflict += uncertainty

        # Conflict count
        conflict_count = sensor_data.get('conflict_count', 0)
        if conflict_count > 0:
            conflict += 0.4

        # Variance indicators
        variance = sensor_data.get('variance', 0.0)
        conflict += 0.3 * variance

        return min(1.0, conflict)

    def compute_snarc_dimensions(self, sensor_data: Dict) -> SNARCDimensions:
        """
        Compute all five SNARC dimensions from sensor data.

        Args:
            sensor_data: Dictionary of sensor readings

        Returns:
            SNARCDimensions with all five dimensions computed
        """
        return SNARCDimensions(
            surprise=self.compute_surprise(sensor_data),
            novelty=self.compute_novelty(sensor_data),
            arousal=self.compute_arousal(sensor_data),
            reward=self.compute_reward(sensor_data),
            conflict=self.compute_conflict(sensor_data)
        )

    def compress_to_salience(self, dimensions: SNARCDimensions) -> float:
        """
        Compress SNARC dimensions to scalar salience.

        Implements compression function:
        - Linear: weighted sum
        - Saturating: tanh-based compression (outlier resistant)

        Args:
            dimensions: Computed SNARC dimensions

        Returns:
            Salience ∈ [0, 1]
        """
        # Weighted combination
        weighted_sum = (
            self.weights.surprise * dimensions.surprise +
            self.weights.novelty * dimensions.novelty +
            self.weights.arousal * dimensions.arousal +
            self.weights.reward * dimensions.reward +
            self.weights.conflict * dimensions.conflict
        )

        # Apply compression
        if self.compression_mode == CompressionMode.LINEAR:
            # Linear compression (simple weighted sum)
            salience = weighted_sum

        elif self.compression_mode == CompressionMode.SATURATING:
            # Saturating compression (tanh-based, outlier resistant)
            # Center at 0.5 (neutral salience)
            centered = weighted_sum - 0.5

            # tanh compression with gain
            compressed = math.tanh(self.saturation_gain * centered)

            # Shift back to [0, 1]
            salience = 0.5 + 0.5 * compressed

        else:
            raise ValueError(f"Unknown compression mode: {self.compression_mode}")

        # Record for learning
        self.salience_history.append(salience)

        return salience

    def compute_salience(self, sensor_data: Dict) -> Tuple[float, SNARCDimensions]:
        """
        Full pipeline: sensor data → SNARC dimensions → scalar salience.

        Args:
            sensor_data: Dictionary of sensor readings

        Returns:
            (salience, dimensions) - salience score and breakdown
        """
        dimensions = self.compute_snarc_dimensions(sensor_data)
        salience = self.compress_to_salience(dimensions)
        return salience, dimensions

    def update_weights(self, outcome_quality: float, learning_rate: float = 0.01):
        """
        Adapt SNARC weights based on outcome quality.

        Higher outcome quality → strengthen weights of dimensions that were high.

        Args:
            outcome_quality: 0-1, quality of attention decision outcome
            learning_rate: Learning rate for weight updates
        """
        if len(self.salience_history) == 0:
            return

        self.outcome_history.append(outcome_quality)

        # Simple gradient: increase weights for dimensions that correlated with good outcomes
        # (This is a placeholder - real implementation would use proper gradient descent)

        # For now, just track statistics
        if len(self.outcome_history) >= 10:
            avg_outcome = sum(list(self.outcome_history)[-10:]) / 10
            # Could adjust weights here based on avg_outcome
            pass

    def get_statistics(self) -> Dict:
        """Get compression statistics for monitoring"""
        stats = {
            'weights': self.weights.as_dict(),
            'compression_mode': self.compression_mode.value,
            'salience_history_len': len(self.salience_history),
        }

        if self.salience_history:
            stats['avg_salience'] = sum(self.salience_history) / len(self.salience_history)
            stats['min_salience'] = min(self.salience_history)
            stats['max_salience'] = max(self.salience_history)

        return stats


# ============================================================================
# Demo & Testing
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("SNARC COMPRESSION MODULE - DEMO")
    print("="*80)
    print()

    # Create compressor
    compressor = SNARCCompressor(compression_mode=CompressionMode.LINEAR)

    print("Testing SNARC compression on sample sensor data:")
    print()

    # Test case 1: Normal operation
    normal_data = {
        'urgent_count': 0,
        'novelty_score': 0.3,
        'count': 3,
        'atp_utilization': 0.4,
        'reward': 0.5,
    }

    salience, dims = compressor.compute_salience(normal_data)
    print("1. Normal operation:")
    print(f"   Dimensions: S={dims.surprise:.2f}, N={dims.novelty:.2f}, A={dims.arousal:.2f}, R={dims.reward:.2f}, C={dims.conflict:.2f}")
    print(f"   Salience: {salience:.3f}")
    print()

    # Test case 2: Urgent condition
    urgent_data = {
        'urgent_count': 2,
        'novelty_score': 0.7,
        'count': 15,
        'atp_utilization': 0.8,
        'error_count': 1,
    }

    salience, dims = compressor.compute_salience(urgent_data)
    print("2. Urgent condition:")
    print(f"   Dimensions: S={dims.surprise:.2f}, N={dims.novelty:.2f}, A={dims.arousal:.2f}, R={dims.reward:.2f}, C={dims.conflict:.2f}")
    print(f"   Salience: {salience:.3f}")
    print()

    # Test case 3: Goal achievement
    goal_data = {
        'urgent_count': 0,
        'novelty_score': 0.1,
        'count': 1,
        'reward': 0.9,
        'success': True,
        'goal_proximity': 0.95,
    }

    salience, dims = compressor.compute_salience(goal_data)
    print("3. Goal achievement:")
    print(f"   Dimensions: S={dims.surprise:.2f}, N={dims.novelty:.2f}, A={dims.arousal:.2f}, R={dims.reward:.2f}, C={dims.conflict:.2f}")
    print(f"   Salience: {salience:.3f}")
    print()

    # Test case 4: High conflict/uncertainty
    conflict_data = {
        'urgent_count': 0,
        'novelty_score': 0.4,
        'uncertainty': 0.8,
        'conflict_count': 2,
        'variance': 0.6,
    }

    salience, dims = compressor.compute_salience(conflict_data)
    print("4. High conflict/uncertainty:")
    print(f"   Dimensions: S={dims.surprise:.2f}, N={dims.novelty:.2f}, A={dims.arousal:.2f}, R={dims.reward:.2f}, C={dims.conflict:.2f}")
    print(f"   Salience: {salience:.3f}")
    print()

    # Compare compression modes
    print("Comparing compression modes:")
    print()

    linear_compressor = SNARCCompressor(compression_mode=CompressionMode.LINEAR)
    saturating_compressor = SNARCCompressor(compression_mode=CompressionMode.SATURATING, saturation_gain=3.0)

    test_data = urgent_data
    sal_linear, _ = linear_compressor.compute_salience(test_data)
    sal_saturating, _ = saturating_compressor.compute_salience(test_data)

    print(f"  Urgent data:")
    print(f"    Linear compression: {sal_linear:.3f}")
    print(f"    Saturating compression: {sal_saturating:.3f}")
    print()

    # Statistics
    print("Compressor statistics:")
    stats = compressor.get_statistics()
    print(f"  Weights: {stats['weights']}")
    print(f"  Mode: {stats['compression_mode']}")
    print(f"  Avg salience: {stats.get('avg_salience', 0):.3f}")
    print()

    print("="*80)
    print("DEMO COMPLETE")
    print("="*80)
