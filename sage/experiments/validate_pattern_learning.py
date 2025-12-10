#!/usr/bin/env python3
"""
Session 22: Pattern Learning Validation for Temporal Adaptation

Validates the pattern learning capability of temporal adaptation framework.
Tests whether the system can learn time-of-day patterns and predictively
adjust ATP parameters for optimal performance.

Research Questions:
1. Can temporal adapter learn recurring time-of-day patterns?
2. Does pattern learning improve performance compared to reactive adaptation?
3. How quickly does the system learn patterns?
4. What is the accuracy of predictive parameter adjustment?

Expected Results:
- Pattern detection within 2-3 occurrences
- Predictive adjustments reduce adaptations by 50%+
- Coverage maintained or improved with learning
- Confidence increases asymptotically to 0.99

Hardware: Jetson AGX Thor
Session: 22 (autonomous)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import random
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# Production temporal adaptation
from core.temporal_adaptation import (
    TemporalAdapter,
    TemporalPattern,
    create_responsive_adapter,
    AdaptationTrigger
)

# Experimental validation infrastructure
from experiments.validate_atp_on_real_consciousness import (
    ATPTunedConsciousness,
    SensorObservation
)


def get_time_of_day_pattern(hour: int) -> Tuple[str, float]:
    """
    Return expected workload pattern for given hour.

    Simulates realistic conversation workload patterns:
    - Early morning (0-6h): Low activity (quiet/sleep)
    - Morning (6-12h): Rising activity (waking up)
    - Midday (12-14h): Peak activity (lunch conversations)
    - Afternoon (14-18h): Moderate activity (work/tasks)
    - Evening (18-22h): Medium-high activity (social time)
    - Night (22-24h): Declining activity (winding down)

    Returns:
        (pattern_name, expected_salience)
    """
    patterns = {
        (0, 6): ("early_morning_quiet", 0.25),
        (6, 9): ("morning_wakeup", 0.50),
        (9, 12): ("morning_active", 0.65),
        (12, 14): ("midday_peak", 0.80),
        (14, 18): ("afternoon_moderate", 0.55),
        (18, 22): ("evening_social", 0.70),
        (22, 24): ("night_winddown", 0.40)
    }

    for (start, end), (name, salience) in patterns.items():
        if start <= hour < end:
            return name, salience

    return "unknown", 0.5


def generate_realistic_workload(
    num_observations: int,
    current_hour: int,
    noise: float = 0.1
) -> List[SensorObservation]:
    """
    Generate observations matching time-of-day pattern.

    Args:
        num_observations: Number of observations to generate
        current_hour: Current hour (0-23)
        noise: Random noise to add to salience (0-1)

    Returns:
        List of sensor observations with realistic salience distribution
    """
    pattern_name, base_salience = get_time_of_day_pattern(current_hour)

    observations = []
    for i in range(num_observations):
        # Add noise to base salience
        salience = max(0.0, min(1.0, base_salience + random.uniform(-noise, noise)))

        observations.append(
            SensorObservation(
                sensor_name=f"sensor_{i % 10}",
                salience=salience,
                data=str({
                    "hour": current_hour,
                    "pattern": pattern_name,
                    "value": random.random()
                })
            )
        )

    return observations


class PatternLearningValidator:
    """
    Validates pattern learning in temporal adaptation.

    Simulates multiple days of realistic workload patterns and
    tracks how quickly the system learns and adapts.
    """

    def __init__(self, adapter: TemporalAdapter, start_suboptimal: bool = False):
        """
        Initialize validator with temporal adapter.

        Args:
            adapter: Temporal adapter instance
            start_suboptimal: Start with suboptimal parameters to force learning
        """
        self.adapter = adapter

        # Optionally start with suboptimal parameters to test learning
        if start_suboptimal:
            initial_cost = 0.03  # Too high (wastes ATP)
            initial_recovery = 0.02  # Too low (slow recovery)
            adapter.current_cost = initial_cost
            adapter.current_recovery = initial_recovery
        else:
            initial_cost = adapter.current_cost
            initial_recovery = adapter.current_recovery

        self.consciousness = ATPTunedConsciousness(
            identity_name="pattern_learning_validation",
            attention_cost=initial_cost,
            rest_recovery=initial_recovery
        )

        # Tracking
        self.total_cycles = 0
        self.total_adaptations = 0
        self.pattern_history: List[Dict] = []
        self.performance_by_hour: Dict[int, List[float]] = {h: [] for h in range(24)}

        # Ground truth optimal parameters for each hour (for comparison)
        self.optimal_params: Dict[int, Tuple[float, float]] = {}

    def simulate_hour(self, hour: int, observations_per_hour: int = 1000):
        """
        Simulate one hour of consciousness operation.

        Args:
            hour: Hour of day (0-23)
            observations_per_hour: Number of observations to process
        """
        pattern_name, expected_salience = get_time_of_day_pattern(hour)

        # Generate realistic workload for this hour
        observations = generate_realistic_workload(observations_per_hour, hour)

        # Track metrics for this hour
        high_salience_count = 0
        attended_high_salience = 0
        hour_adaptations = 0

        for obs in observations:
            # Check if would be attended
            threshold = self.consciousness.attention_cost * (1.0 - self.consciousness.atp_level)
            attended = obs.salience > threshold

            # Track high-salience observations
            if obs.salience > 0.7:
                high_salience_count += 1
                if attended:
                    attended_high_salience += 1

            # Update temporal adapter
            result = self.adapter.update(
                attended=attended,
                salience=obs.salience,
                atp_level=self.consciousness.atp_level,
                high_salience_count=high_salience_count,
                attended_high_salience=attended_high_salience
            )

            # If adaptation occurred, update consciousness
            if result is not None:
                new_cost, new_recovery = result
                self.consciousness.attention_cost = new_cost
                self.consciousness.rest_recovery = new_recovery
                hour_adaptations += 1
                self.total_adaptations += 1

            self.total_cycles += 1

        # Process observations through consciousness
        self.consciousness.process_cycle(observations)

        # Calculate hour performance
        coverage = attended_high_salience / high_salience_count if high_salience_count > 0 else 0.0
        self.performance_by_hour[hour].append(coverage)

        return {
            'hour': hour,
            'pattern': pattern_name,
            'observations': observations_per_hour,
            'adaptations': hour_adaptations,
            'coverage': coverage,
            'final_cost': self.consciousness.attention_cost,
            'final_recovery': self.consciousness.rest_recovery
        }

    def simulate_day(self, day_number: int, observations_per_hour: int = 1000):
        """
        Simulate 24 hours of operation.

        Args:
            day_number: Day number (for tracking multi-day learning)
            observations_per_hour: Observations to process per hour
        """
        print(f"\n{'='*70}")
        print(f"Day {day_number}: Simulating 24 hours of pattern learning")
        print(f"{'='*70}")

        day_results = []

        for hour in range(24):
            result = self.simulate_hour(hour, observations_per_hour)
            day_results.append(result)

            # Show progress every 4 hours
            if (hour + 1) % 4 == 0:
                print(f"  Hour {hour:02d}: pattern={result['pattern']:<20} "
                      f"adaptations={result['adaptations']} "
                      f"coverage={result['coverage']:.1%}")

        # Day summary
        total_day_adaptations = sum(r['adaptations'] for r in day_results)
        avg_coverage = statistics.mean(r['coverage'] for r in day_results)

        print(f"\nDay {day_number} Summary:")
        print(f"  Total adaptations: {total_day_adaptations}")
        print(f"  Average coverage: {avg_coverage:.1%}")
        print(f"  Total cycles: {self.total_cycles:,}")

        return day_results

    def get_learning_statistics(self) -> Dict:
        """Get pattern learning statistics"""
        stats = self.adapter.get_statistics()

        # Add pattern-specific metrics
        if self.adapter.enable_pattern_learning:
            stats['learned_patterns'] = {}
            for pattern_name, pattern in self.adapter.learned_patterns.items():
                stats['learned_patterns'][pattern_name] = {
                    'pattern_type': pattern.pattern_type,
                    'time_range': pattern.time_range,
                    'optimal_cost': pattern.optimal_cost,
                    'optimal_recovery': pattern.optimal_recovery,
                    'confidence': pattern.confidence,
                    'observations': pattern.observations
                }

        # Add performance by hour
        stats['avg_coverage_by_hour'] = {
            h: statistics.mean(coverages) if coverages else 0.0
            for h, coverages in self.performance_by_hour.items()
        }

        return stats


def test_pattern_learning_basic():
    """
    Test 1: Basic pattern learning over 3 days.

    Validates that patterns are detected and learned.
    """
    print("\n" + "="*70)
    print("TEST 1: Basic Pattern Learning (3 days)")
    print("="*70)

    # Create adapter with pattern learning enabled
    adapter = create_responsive_adapter()  # Responsive mode enables pattern learning
    validator = PatternLearningValidator(adapter, start_suboptimal=True)

    # Simulate 3 days
    all_results = []
    for day in range(1, 4):
        day_results = validator.simulate_day(day, observations_per_hour=500)
        all_results.extend(day_results)

    # Analyze learning
    stats = validator.get_learning_statistics()

    print("\n" + "="*70)
    print("Pattern Learning Results")
    print("="*70)

    if stats.get('learned_patterns'):
        print(f"\nLearned {len(stats['learned_patterns'])} patterns:")
        for name, pattern in stats['learned_patterns'].items():
            print(f"\n  Pattern: {pattern['pattern_type']}")
            print(f"    Time range: {pattern['time_range'][0]:02d}:00 - {pattern['time_range'][1]:02d}:00")
            print(f"    Optimal cost: {pattern['optimal_cost']:.4f}")
            print(f"    Optimal recovery: {pattern['optimal_recovery']:.4f}")
            print(f"    Confidence: {pattern['confidence']:.2%}")
            print(f"    Observations: {pattern['observations']}")
    else:
        print("\n  ⚠️  No patterns learned (pattern learning may not be fully implemented)")

    print(f"\nOverall Statistics:")
    print(f"  Total cycles: {validator.total_cycles:,}")
    print(f"  Total adaptations: {validator.total_adaptations}")
    print(f"  Adaptations/day: {validator.total_adaptations / 3:.1f}")

    return validator, stats


def test_pattern_learning_benefit():
    """
    Test 2: Compare pattern learning vs reactive-only adaptation.

    Measures improvement from pattern learning.
    """
    print("\n" + "="*70)
    print("TEST 2: Pattern Learning vs Reactive-Only (2 days each)")
    print("="*70)

    # Test with pattern learning
    print("\n--- WITH Pattern Learning ---")
    adapter_learning = create_responsive_adapter()  # Pattern learning enabled
    validator_learning = PatternLearningValidator(adapter_learning, start_suboptimal=True)

    for day in range(1, 3):
        validator_learning.simulate_day(day, observations_per_hour=500)

    stats_learning = validator_learning.get_learning_statistics()

    # Test without pattern learning
    print("\n--- WITHOUT Pattern Learning (Reactive Only) ---")
    from core.temporal_adaptation import create_production_adapter
    adapter_reactive = create_production_adapter()  # Pattern learning disabled
    validator_reactive = PatternLearningValidator(adapter_reactive, start_suboptimal=True)

    for day in range(1, 3):
        validator_reactive.simulate_day(day, observations_per_hour=500)

    stats_reactive = validator_reactive.get_learning_statistics()

    # Compare results
    print("\n" + "="*70)
    print("Comparison: Learning vs Reactive")
    print("="*70)

    print(f"\n{'Metric':<30} {'With Learning':<20} {'Reactive Only':<20} {'Improvement':<15}")
    print("-" * 85)

    adaptations_learning = validator_learning.total_adaptations
    adaptations_reactive = validator_reactive.total_adaptations
    improvement_adaptations = ((adaptations_reactive - adaptations_learning) / adaptations_reactive * 100) if adaptations_reactive > 0 else 0

    print(f"{'Total Adaptations':<30} {adaptations_learning:<20} {adaptations_reactive:<20} {improvement_adaptations:>13.1f}%")

    # Coverage comparison
    avg_cov_learning = statistics.mean([c for coverages in stats_learning['avg_coverage_by_hour'].values() for c in ([coverages] if isinstance(coverages, float) else coverages) if c > 0])
    avg_cov_reactive = statistics.mean([c for coverages in stats_reactive['avg_coverage_by_hour'].values() for c in ([coverages] if isinstance(coverages, float) else coverages) if c > 0])
    improvement_coverage = ((avg_cov_learning - avg_cov_reactive) / avg_cov_reactive * 100) if avg_cov_reactive > 0 else 0

    print(f"{'Average Coverage':<30} {avg_cov_learning:<19.1%} {avg_cov_reactive:<19.1%} {improvement_coverage:>13.1f}%")

    print(f"\nConclusion:")
    if improvement_adaptations > 20:
        print(f"  ✅ Pattern learning reduces adaptations by {improvement_adaptations:.1f}%")
    elif improvement_adaptations > 0:
        print(f"  ⚠️  Modest adaptation reduction ({improvement_adaptations:.1f}%)")
    else:
        print(f"  ❌ No adaptation reduction (pattern learning needs implementation)")

    if improvement_coverage > 0:
        print(f"  ✅ Coverage improved by {improvement_coverage:.1f}%")
    elif improvement_coverage > -5:
        print(f"  ✅ Coverage maintained (within 5%)")
    else:
        print(f"  ⚠️  Coverage decreased by {abs(improvement_coverage):.1f}%")


def main():
    """Run Session 22 pattern learning validation"""
    print("=" * 80)
    print(" " * 20 + "Session 22: Pattern Learning Validation")
    print("=" * 80)
    print("\nValidating temporal adaptation pattern learning capability")
    print("Testing time-of-day pattern detection and predictive parameter adjustment")
    print()

    # Test 1: Basic learning
    validator, stats = test_pattern_learning_basic()

    # Test 2: Benefit analysis
    test_pattern_learning_benefit()

    # Final summary
    print("\n" + "="*80)
    print("Session 22 Complete")
    print("="*80)

    print("\nKey Findings:")
    if stats.get('learned_patterns'):
        print(f"1. Pattern learning functional - {len(stats['learned_patterns'])} patterns detected")
        print("2. Time-of-day patterns successfully learned and applied")
    else:
        print("1. Pattern learning scaffolded but needs implementation")
        print("2. TemporalPattern dataclass and API ready for learning logic")

    print("\nNext Steps:")
    print("- Implement pattern detection logic in TemporalAdapter.update()")
    print("- Add predictive parameter application based on time-of-day")
    print("- Test with real conversation workloads")
    print("- Measure long-term pattern stability (multi-week learning)")


if __name__ == "__main__":
    main()
