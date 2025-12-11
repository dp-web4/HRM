"""
Adaptive Objective Weighting for Multi-Objective Temporal Adaptation

Session 28: Context-Aware Adaptive Weighting

Dynamically adjusts objective weights (coverage, quality, energy) based on
operational context to optimize for the current situation. Replaces static
50/30/20 weighting with context-aware adaptation.

Context Dimensions:
1. **ATP Level**: High ATP → emphasize quality, Low ATP → emphasize coverage
2. **Attention Rate**: High attention → emphasize energy, Low → emphasize coverage
3. **Quality Trend**: Improving → maintain, Declining → emphasize quality
4. **Coverage Stability**: Stable → can emphasize quality, Unstable → coverage priority

Design Principles:
- Smooth weight transitions (avoid oscillation)
- Weights always sum to 1.0
- Constraints: Each weight ∈ [0.1, 0.7]
- Default baseline: coverage=0.5, quality=0.3, energy=0.2

Hardware: Jetson AGX Thor (generalizes to all platforms)
Based on: Session 26 (multi-objective), Session 27 (quality metrics)
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import statistics


@dataclass
class ObjectiveWeights:
    """
    Multi-objective optimization weights.

    Attributes:
        coverage: Weight for coverage objective [0.1, 0.7]
        quality: Weight for quality objective [0.1, 0.7]
        energy: Weight for energy efficiency objective [0.1, 0.7]

    Invariant: coverage + quality + energy = 1.0
    """
    coverage: float
    quality: float
    energy: float

    def __post_init__(self):
        """Validate weights sum to 1.0 and are in valid range"""
        total = self.coverage + self.quality + self.energy
        if not (0.999 <= total <= 1.001):  # Allow floating point tolerance
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")

        for name, value in [('coverage', self.coverage),
                           ('quality', self.quality),
                           ('energy', self.energy)]:
            if not (0.1 <= value <= 0.7):
                raise ValueError(f"{name} weight {value:.2f} outside range [0.1, 0.7]")

    def to_tuple(self) -> Tuple[float, float, float]:
        """Return weights as tuple for compatibility"""
        return (self.coverage, self.quality, self.energy)

    def to_dict(self) -> Dict[str, float]:
        """Return weights as dictionary"""
        return {
            'coverage': self.coverage,
            'quality': self.quality,
            'energy': self.energy
        }


@dataclass
class OperatingContext:
    """
    Current operating context for adaptive weighting.

    Attributes:
        atp_level: Current ATP level (0-1, normalized)
        attention_rate: Recent attention allocation rate (0-1)
        coverage: Recent coverage score (0-1)
        quality: Recent quality score (0-1)
        energy_efficiency: Recent energy efficiency (0-1)
    """
    atp_level: float
    attention_rate: float
    coverage: float
    quality: float
    energy_efficiency: float

    def __post_init__(self):
        """Validate all metrics are in [0, 1]"""
        for name, value in [('atp_level', self.atp_level),
                           ('attention_rate', self.attention_rate),
                           ('coverage', self.coverage),
                           ('quality', self.quality),
                           ('energy_efficiency', self.energy_efficiency)]:
            if value is not None and not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} = {value:.2f} outside range [0, 1]")


class AdaptiveWeightCalculator:
    """
    Calculates context-aware objective weights for multi-objective optimization.

    Implements smooth weight adaptation based on operating context to optimize
    for the current situation while avoiding oscillation.

    Strategy:
    1. Start with baseline weights (50/30/20)
    2. Apply context-based adjustments
    3. Smooth transitions using exponential moving average
    4. Ensure weights sum to 1.0 and respect constraints
    """

    # Baseline weights (Session 26 validated)
    BASELINE_COVERAGE = 0.5
    BASELINE_QUALITY = 0.3
    BASELINE_ENERGY = 0.2

    # Weight constraints
    MIN_WEIGHT = 0.1
    MAX_WEIGHT = 0.7

    # Smoothing parameter (0 = no smoothing, 1 = max smoothing)
    SMOOTHING_ALPHA = 0.3  # 30% new, 70% old

    def __init__(self):
        """Initialize adaptive weight calculator"""
        # Current smoothed weights
        self.current_weights = ObjectiveWeights(
            coverage=self.BASELINE_COVERAGE,
            quality=self.BASELINE_QUALITY,
            energy=self.BASELINE_ENERGY
        )

        # Statistics
        self.total_updates = 0
        self.total_adjustments = 0  # Times weights actually changed

    def calculate_weights(self, context: OperatingContext) -> ObjectiveWeights:
        """
        Calculate adaptive objective weights based on operating context.

        Args:
            context: Current operating context

        Returns:
            ObjectiveWeights with context-adjusted weights

        Strategy:
            1. Start with baseline
            2. Adjust based on ATP level (high → quality, low → coverage)
            3. Adjust based on attention rate (high → energy, low → coverage)
            4. Adjust based on performance trends
            5. Normalize and smooth
        """
        self.total_updates += 1

        # Start with baseline
        coverage_weight = self.BASELINE_COVERAGE
        quality_weight = self.BASELINE_QUALITY
        energy_weight = self.BASELINE_ENERGY

        # === ATP-BASED ADJUSTMENT ===
        # High ATP (> 0.7): Can afford quality → increase quality weight
        # Low ATP (< 0.3): Need coverage → increase coverage weight
        if context.atp_level > 0.7:
            # High ATP: shift 10% from coverage to quality
            shift = 0.10
            coverage_weight -= shift
            quality_weight += shift
        elif context.atp_level < 0.3:
            # Low ATP: shift 10% from quality to coverage
            shift = 0.10
            quality_weight -= shift
            coverage_weight += shift

        # === ATTENTION RATE ADJUSTMENT ===
        # High attention (> 0.8): Spending a lot → emphasize energy
        # Low attention (< 0.2): Not attending enough → emphasize coverage
        if context.attention_rate > 0.8:
            # High attention: shift 5% to energy efficiency
            shift = 0.05
            coverage_weight -= shift * 0.6  # 3% from coverage
            quality_weight -= shift * 0.4   # 2% from quality
            energy_weight += shift
        elif context.attention_rate < 0.2:
            # Low attention: shift 5% to coverage
            shift = 0.05
            quality_weight -= shift * 0.6   # 3% from quality
            energy_weight -= shift * 0.4    # 2% from energy
            coverage_weight += shift

        # === PERFORMANCE-BASED ADJUSTMENT ===
        # If coverage is low (< 0.85), prioritize it
        if context.coverage < 0.85:
            shift = 0.10
            quality_weight -= shift * 0.7
            energy_weight -= shift * 0.3
            coverage_weight += shift

        # If quality is declining (< 0.6), emphasize it
        if context.quality < 0.6:
            shift = 0.05
            coverage_weight -= shift * 0.6
            energy_weight -= shift * 0.4
            quality_weight += shift

        # === NORMALIZE AND CONSTRAIN ===
        raw_weights = self._normalize_weights(
            coverage_weight,
            quality_weight,
            energy_weight
        )

        # === SMOOTH TRANSITIONS ===
        smoothed_weights = self._smooth_weights(raw_weights)

        # Check if weights actually changed
        if self._weights_changed(smoothed_weights):
            self.total_adjustments += 1

        # Update current weights
        self.current_weights = smoothed_weights

        return smoothed_weights

    def _normalize_weights(
        self,
        coverage: float,
        quality: float,
        energy: float
    ) -> ObjectiveWeights:
        """
        Normalize weights to sum to 1.0 and respect constraints.

        Args:
            coverage, quality, energy: Unnormalized weights

        Returns:
            Normalized ObjectiveWeights
        """
        # Apply min/max constraints
        coverage = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, coverage))
        quality = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, quality))
        energy = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, energy))

        # Normalize to sum to 1.0
        total = coverage + quality + energy
        coverage /= total
        quality /= total
        energy /= total

        return ObjectiveWeights(
            coverage=coverage,
            quality=quality,
            energy=energy
        )

    def _smooth_weights(self, new_weights: ObjectiveWeights) -> ObjectiveWeights:
        """
        Smooth weight transitions using exponential moving average.

        Prevents oscillation by blending new weights with current weights.

        Args:
            new_weights: Newly calculated weights

        Returns:
            Smoothed ObjectiveWeights
        """
        alpha = self.SMOOTHING_ALPHA

        smoothed_coverage = (alpha * new_weights.coverage +
                            (1 - alpha) * self.current_weights.coverage)
        smoothed_quality = (alpha * new_weights.quality +
                           (1 - alpha) * self.current_weights.quality)
        smoothed_energy = (alpha * new_weights.energy +
                          (1 - alpha) * self.current_weights.energy)

        # Renormalize (smoothing might introduce small errors)
        return self._normalize_weights(
            smoothed_coverage,
            smoothed_quality,
            smoothed_energy
        )

    def _weights_changed(self, new_weights: ObjectiveWeights) -> bool:
        """Check if weights changed significantly (> 1%)"""
        coverage_delta = abs(new_weights.coverage - self.current_weights.coverage)
        quality_delta = abs(new_weights.quality - self.current_weights.quality)
        energy_delta = abs(new_weights.energy - self.current_weights.energy)

        return (coverage_delta > 0.01 or
                quality_delta > 0.01 or
                energy_delta > 0.01)

    def get_stats(self) -> Dict[str, any]:
        """Get calculator statistics"""
        return {
            'total_updates': self.total_updates,
            'total_adjustments': self.total_adjustments,
            'adjustment_rate': (self.total_adjustments / self.total_updates
                              if self.total_updates > 0 else 0.0),
            'current_weights': self.current_weights.to_dict()
        }

    def reset(self):
        """Reset to baseline weights"""
        self.current_weights = ObjectiveWeights(
            coverage=self.BASELINE_COVERAGE,
            quality=self.BASELINE_QUALITY,
            energy=self.BASELINE_ENERGY
        )
        self.total_updates = 0
        self.total_adjustments = 0


# Convenience function for quick weight calculation
def calculate_adaptive_weights(
    atp_level: float,
    attention_rate: float,
    coverage: float,
    quality: float,
    energy_efficiency: float
) -> Tuple[float, float, float]:
    """
    Quick function to calculate adaptive weights without maintaining state.

    Args:
        atp_level: Current ATP level (0-1)
        attention_rate: Recent attention rate (0-1)
        coverage: Recent coverage score (0-1)
        quality: Recent quality score (0-1)
        energy_efficiency: Recent energy efficiency (0-1)

    Returns:
        (coverage_weight, quality_weight, energy_weight) tuple
    """
    calculator = AdaptiveWeightCalculator()
    context = OperatingContext(
        atp_level=atp_level,
        attention_rate=attention_rate,
        coverage=coverage,
        quality=quality,
        energy_efficiency=energy_efficiency
    )
    weights = calculator.calculate_weights(context)
    return weights.to_tuple()


if __name__ == "__main__":
    # Quick validation
    print("=" * 70)
    print("Adaptive Objective Weighting - Session 28")
    print("=" * 70)
    print()

    calculator = AdaptiveWeightCalculator()

    # Test scenarios
    scenarios = [
        {
            'name': 'Baseline (normal operation)',
            'context': OperatingContext(
                atp_level=0.5,
                attention_rate=0.5,
                coverage=0.95,
                quality=0.8,
                energy_efficiency=0.75
            )
        },
        {
            'name': 'High ATP (can afford quality)',
            'context': OperatingContext(
                atp_level=0.9,
                attention_rate=0.6,
                coverage=0.95,
                quality=0.8,
                energy_efficiency=0.75
            )
        },
        {
            'name': 'Low ATP (need coverage)',
            'context': OperatingContext(
                atp_level=0.2,
                attention_rate=0.3,
                coverage=0.85,
                quality=0.7,
                energy_efficiency=0.60
            )
        },
        {
            'name': 'High attention (emphasize energy)',
            'context': OperatingContext(
                atp_level=0.6,
                attention_rate=0.9,
                coverage=0.95,
                quality=0.8,
                energy_efficiency=0.50
            )
        },
        {
            'name': 'Low coverage (prioritize coverage)',
            'context': OperatingContext(
                atp_level=0.5,
                attention_rate=0.5,
                coverage=0.75,
                quality=0.8,
                energy_efficiency=0.75
            )
        },
        {
            'name': 'Low quality (prioritize quality)',
            'context': OperatingContext(
                atp_level=0.6,
                attention_rate=0.5,
                coverage=0.95,
                quality=0.5,
                energy_efficiency=0.75
            )
        }
    ]

    for scenario in scenarios:
        weights = calculator.calculate_weights(scenario['context'])
        print(f"{scenario['name']}:")
        print(f"  Context: ATP={scenario['context'].atp_level:.1f}, " +
              f"Attn={scenario['context'].attention_rate:.1f}, " +
              f"Cov={scenario['context'].coverage:.2f}, " +
              f"Qual={scenario['context'].quality:.2f}")
        print(f"  Weights: Coverage={weights.coverage:.1%}, " +
              f"Quality={weights.quality:.1%}, " +
              f"Energy={weights.energy:.1%}")
        print()

    stats = calculator.get_stats()
    print("=" * 70)
    print("Statistics:")
    print(f"  Total updates: {stats['total_updates']}")
    print(f"  Total adjustments: {stats['total_adjustments']}")
    print(f"  Adjustment rate: {stats['adjustment_rate']:.1%}")
    print("=" * 70)
