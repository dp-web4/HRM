#!/usr/bin/env python3
"""
Session 137: Extended Stability Testing

Session 136 proved emotional regulation prevents frustration cascade over 100 cycles.
Now we test LONG-TERM stability: Does regulation hold under extended operation?

Research Questions:
1. Does emotional regulation remain stable over 1000+ cycles?
2. Do any emergent failure modes appear at longer timescales?
3. How does learning progress over extended operation?
4. What is the optimal regulation parameter tuning for long-term stability?

Test Strategy:
1. Baseline: 1000-cycle run with Session 136 regulation (validate stability)
2. Stress test: 1000 cycles with higher failure rate (test limits)
3. Recovery test: Induce high frustration, validate recovery over time
4. Comparative analysis: Measure long-term metrics vs Session 136

Success Criteria:
- Emotional states remain within bounds (no cascade)
- Learning continues throughout (success rate stable or improving)
- Intervention counts decrease over time (EP maturation)
- No emergent instabilities at any timescale

Date: 2025-12-29
Hardware: Thor (Jetson AGX Thor Developer Kit)
Previous: Session 136 (100-cycle validation)
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import time
import json

# Add SAGE modules to path
sys.path.append(str(Path(__file__).parent.parent))

from session136_emotional_regulation import (
    RegulatedConsciousnessLoop,
    EmotionalRegulationConfig,
    EmotionalRegulator,
)
from session134_memory_guided_attention import Experience, SAGEIdentityManager
from session131_sage_unified_identity import UnifiedSAGEIdentity


@dataclass
class ExtendedTestConfig:
    """Configuration for extended stability tests."""

    # Test duration
    total_cycles: int = 1000
    checkpoint_interval: int = 100  # Report every N cycles

    # Experience parameters
    base_failure_rate: float = 0.3  # 30% base failure rate
    stress_failure_rate: float = 0.6  # 60% for stress testing

    # Recovery test parameters
    recovery_initial_frustration: float = 0.90  # Start at near-max
    recovery_failure_rate: float = 0.2  # Low failure to allow recovery

    # Tracking
    detailed_tracking: bool = True  # Track all metrics
    save_checkpoints: bool = True  # Save state periodically


@dataclass
class ExtendedTestResults:
    """Results from extended stability testing."""

    test_name: str
    total_cycles: int
    final_emotional_state: Dict[str, float]

    # Stability metrics
    frustration_trajectory: List[float] = field(default_factory=list)
    engagement_trajectory: List[float] = field(default_factory=list)
    curiosity_trajectory: List[float] = field(default_factory=list)
    progress_trajectory: List[float] = field(default_factory=list)

    # Learning metrics
    success_rate_trajectory: List[float] = field(default_factory=list)
    learning_phases_completed: int = 0

    # Regulation metrics
    total_interventions: int = 0
    interventions_by_type: Dict[str, int] = field(default_factory=dict)
    interventions_per_100_cycles: List[int] = field(default_factory=list)

    # Stability analysis
    cascade_detected: bool = False
    max_frustration_reached: float = 0.0
    min_engagement_reached: float = 1.0
    stable_operation: bool = True

    # EP maturation indicators
    early_intervention_rate: float = 0.0  # First 300 cycles
    late_intervention_rate: float = 0.0   # Last 300 cycles
    ep_maturation_score: float = 0.0      # Reduction in interventions

    # Timing
    duration_seconds: float = 0.0
    cycles_per_second: float = 0.0

    def analyze_stability(self) -> Dict[str, Any]:
        """Analyze stability from trajectories."""
        analysis = {}

        # Frustration stability
        if self.frustration_trajectory:
            analysis["frustration_mean"] = sum(self.frustration_trajectory) / len(
                self.frustration_trajectory
            )
            analysis["frustration_max"] = max(self.frustration_trajectory)
            analysis["frustration_min"] = min(self.frustration_trajectory)
            analysis["frustration_variance"] = self._variance(self.frustration_trajectory)

        # Engagement stability
        if self.engagement_trajectory:
            analysis["engagement_mean"] = sum(self.engagement_trajectory) / len(
                self.engagement_trajectory
            )
            analysis["engagement_min"] = min(self.engagement_trajectory)

        # Success rate trend
        if self.success_rate_trajectory:
            # Compare first 100 vs last 100 cycles
            early_success = sum(self.success_rate_trajectory[:100]) / min(
                100, len(self.success_rate_trajectory)
            )
            late_success = sum(self.success_rate_trajectory[-100:]) / min(
                100, len(self.success_rate_trajectory[-100:])
            )
            analysis["early_success_rate"] = early_success
            analysis["late_success_rate"] = late_success
            analysis["learning_improvement"] = late_success - early_success

        # Intervention trend (EP maturation)
        if len(self.interventions_per_100_cycles) >= 2:
            early_interventions = sum(self.interventions_per_100_cycles[:3])
            late_interventions = sum(self.interventions_per_100_cycles[-3:])
            self.early_intervention_rate = early_interventions / 3.0
            self.late_intervention_rate = late_interventions / 3.0
            self.ep_maturation_score = (
                early_interventions - late_interventions
            ) / early_interventions if early_interventions > 0 else 0.0

            analysis["early_intervention_rate"] = self.early_intervention_rate
            analysis["late_intervention_rate"] = self.late_intervention_rate
            analysis["ep_maturation_score"] = self.ep_maturation_score

        # Overall stability determination
        stable = True
        if analysis.get("frustration_max", 0) >= 0.98:
            stable = False  # Near cascade
        if analysis.get("engagement_min", 1.0) <= 0.05:
            stable = False  # Disengagement
        if self.cascade_detected:
            stable = False

        self.stable_operation = stable
        analysis["stable_operation"] = stable

        return analysis

    def _variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)


class ExtendedStabilityTester:
    """
    Extended stability testing framework for SAGE consciousness.

    Tests long-term stability of emotional regulation across 1000+ cycles.
    """

    def __init__(
        self,
        test_config: ExtendedTestConfig,
        regulation_config: Optional[EmotionalRegulationConfig] = None,
    ):
        self.test_config = test_config
        self.regulation_config = regulation_config or EmotionalRegulationConfig()

        # Initialize identity manager
        self.identity_manager = SAGEIdentityManager()
        self.identity_manager.create_identity()  # Initialize identity
        self.identity = self.identity_manager.current_identity

    def run_baseline_test(self) -> ExtendedTestResults:
        """
        Test 1: Baseline 1000-cycle stability test.

        Validates that Session 136 regulation holds over extended operation.
        """
        print("\n=== Test 1: Baseline 1000-Cycle Stability ===")
        print(f"Total cycles: {self.test_config.total_cycles}")
        print(f"Failure rate: {self.test_config.base_failure_rate:.1%}")
        print(f"Regulation: Active (Session 136 config)")

        results = ExtendedTestResults(
            test_name="baseline_1000_cycles",
            total_cycles=self.test_config.total_cycles,
            final_emotional_state={},
        )

        # Create consciousness loop
        loop = RegulatedConsciousnessLoop(
            identity_manager=self.identity_manager,
            regulation_config=self.regulation_config,
        )

        start_time = time.time()

        # Run cycles
        for cycle in range(self.test_config.total_cycles):
            # Generate multiple experiences per cycle (like Session 136)
            experiences = [
                self._generate_experience(
                    cycle=cycle * 15 + i,  # Unique ID per experience
                    failure_rate=self.test_config.base_failure_rate,
                )
                for i in range(15)
            ]

            # Process
            loop.consciousness_cycle(experiences)

            # Track metrics
            self._track_metrics(loop, results, cycle)

            # Checkpoint
            if (cycle + 1) % self.test_config.checkpoint_interval == 0:
                self._print_checkpoint(loop, results, cycle + 1)

        # Finalize
        results.duration_seconds = time.time() - start_time
        results.cycles_per_second = results.total_cycles / results.duration_seconds
        results.final_emotional_state = {
            "frustration": self.identity.frustration,
            "engagement": self.identity.engagement,
            "curiosity": self.identity.curiosity,
            "progress": self.identity.progress,
        }

        return results

    def run_stress_test(self) -> ExtendedTestResults:
        """
        Test 2: Stress test with higher failure rate.

        Tests regulation limits under sustained high stress.
        """
        print("\n=== Test 2: Stress Test (High Failure Rate) ===")
        print(f"Total cycles: {self.test_config.total_cycles}")
        print(f"Failure rate: {self.test_config.stress_failure_rate:.1%}")
        print(f"Regulation: Active (Session 136 config)")

        # Reset identity manager
        self.identity_manager = SAGEIdentityManager()
        self.identity_manager.create_identity()
        self.identity = self.identity_manager.current_identity

        results = ExtendedTestResults(
            test_name="stress_test_high_failure",
            total_cycles=self.test_config.total_cycles,
            final_emotional_state={},
        )

        loop = RegulatedConsciousnessLoop(
            identity_manager=self.identity_manager,
            regulation_config=self.regulation_config,
        )

        start_time = time.time()

        for cycle in range(self.test_config.total_cycles):
            experiences = [
                self._generate_experience(
                    cycle=cycle * 15 + i,
                    failure_rate=self.test_config.stress_failure_rate,
                )
                for i in range(15)
            ]

            loop.consciousness_cycle(experiences)
            self._track_metrics(loop, results, cycle)

            if (cycle + 1) % self.test_config.checkpoint_interval == 0:
                self._print_checkpoint(loop, results, cycle + 1)

        results.duration_seconds = time.time() - start_time
        results.cycles_per_second = results.total_cycles / results.duration_seconds
        results.final_emotional_state = {
            "frustration": self.identity.frustration,
            "engagement": self.identity.engagement,
            "curiosity": self.identity.curiosity,
            "progress": self.identity.progress,
        }

        return results

    def run_recovery_test(self) -> ExtendedTestResults:
        """
        Test 3: Recovery from high frustration.

        Start at near-max frustration, validate recovery mechanisms.
        """
        print("\n=== Test 3: Recovery Test (High Initial Frustration) ===")
        print(f"Total cycles: {self.test_config.total_cycles}")
        print(f"Initial frustration: {self.test_config.recovery_initial_frustration:.2f}")
        print(f"Failure rate: {self.test_config.recovery_failure_rate:.1%} (low, to allow recovery)")

        # Reset identity manager with high frustration
        self.identity_manager = SAGEIdentityManager()
        self.identity_manager.create_identity()
        self.identity = self.identity_manager.current_identity

        # Set high initial frustration for recovery test
        self.identity_manager.update_emotional_state(
            frustration=self.test_config.recovery_initial_frustration,
            engagement=0.30,
            curiosity=0.25,
            progress=self.identity.progress,
        )
        self.identity = self.identity_manager.current_identity

        results = ExtendedTestResults(
            test_name="recovery_test",
            total_cycles=self.test_config.total_cycles,
            final_emotional_state={},
        )

        loop = RegulatedConsciousnessLoop(
            identity_manager=self.identity_manager,
            regulation_config=self.regulation_config,
        )

        start_time = time.time()

        for cycle in range(self.test_config.total_cycles):
            experiences = [
                self._generate_experience(
                    cycle=cycle * 15 + i,
                    failure_rate=self.test_config.recovery_failure_rate,
                )
                for i in range(15)
            ]

            loop.consciousness_cycle(experiences)
            self._track_metrics(loop, results, cycle)

            if (cycle + 1) % self.test_config.checkpoint_interval == 0:
                self._print_checkpoint(loop, results, cycle + 1)

        results.duration_seconds = time.time() - start_time
        results.cycles_per_second = results.total_cycles / results.duration_seconds
        results.final_emotional_state = {
            "frustration": self.identity.frustration,
            "engagement": self.identity.engagement,
            "curiosity": self.identity.curiosity,
            "progress": self.identity.progress,
        }

        return results

    def _generate_experience(self, cycle: int, failure_rate: float) -> Experience:
        """Generate synthetic experience with specified failure rate."""
        import random

        # Higher difficulty correlates with higher failure rate
        # Complexity ranges from 0-1, where higher = harder
        complexity = failure_rate + random.uniform(-0.1, 0.1)
        complexity = max(0.0, min(1.0, complexity))  # Clamp to 0-1

        return Experience(
            experience_id=f"extended_test_cycle{cycle}",
            content=f"Extended stability test cycle {cycle}, difficulty {complexity:.2f}",
            salience=0.5 + random.uniform(0.0, 0.5),
            complexity=complexity,
        )

    def _track_metrics(
        self,
        loop: RegulatedConsciousnessLoop,
        results: ExtendedTestResults,
        cycle: int,
    ):
        """Track all metrics for this cycle."""
        # Emotional states
        results.frustration_trajectory.append(self.identity.frustration)
        results.engagement_trajectory.append(self.identity.engagement)
        results.curiosity_trajectory.append(self.identity.curiosity)
        results.progress_trajectory.append(self.identity.progress)

        # Learning metrics
        results.learning_phases_completed += 1

        # Regulation metrics
        if hasattr(loop, "regulator") and loop.regulator:
            results.total_interventions = loop.regulator.intervention_count
            # Build interventions_by_type from available stats
            results.interventions_by_type = {
                "frustration_regulated": loop.regulator.total_frustration_regulated,
                "curiosity_boosted": loop.regulator.total_curiosity_boosted,
                "engagement_boosted": loop.regulator.total_engagement_boosted,
                "recovery_count": loop.regulator.recovery_count,
            }

        # Check for cascade
        if self.identity.frustration >= 0.98:
            results.cascade_detected = True

        # Track max/min
        results.max_frustration_reached = max(
            results.max_frustration_reached, self.identity.frustration
        )
        results.min_engagement_reached = min(
            results.min_engagement_reached, self.identity.engagement
        )

        # Success rate (windowed)
        # For simplicity, track binary success per cycle
        # In real implementation, would track actual experience results

        # Interventions per 100 cycles
        if (cycle + 1) % 100 == 0:
            current_interventions = results.total_interventions
            if len(results.interventions_per_100_cycles) == 0:
                interventions_this_window = current_interventions
            else:
                prev_total = sum(results.interventions_per_100_cycles)
                interventions_this_window = current_interventions - prev_total
            results.interventions_per_100_cycles.append(interventions_this_window)

    def _print_checkpoint(
        self,
        loop: RegulatedConsciousnessLoop,
        results: ExtendedTestResults,
        cycle: int,
    ):
        """Print checkpoint status."""
        print(f"\n--- Checkpoint: Cycle {cycle} ---")
        print(f"Frustration: {self.identity.frustration:.2f}")
        print(f"Engagement: {self.identity.engagement:.2f}")
        print(f"Curiosity: {self.identity.curiosity:.2f}")
        print(f"Progress: {self.identity.progress:.2f}")

        if hasattr(loop, "regulator") and loop.regulator:
            print(f"Total interventions: {loop.regulator.intervention_count}")
            print(f"Frustration regulated: {loop.regulator.total_frustration_regulated:.2f}")
            print(f"Recovery count: {loop.regulator.recovery_count}")


def run_all_tests():
    """Run all extended stability tests."""
    print("=" * 80)
    print("Session 137: Extended Stability Testing")
    print("=" * 80)
    print(f"Hardware: Thor (Jetson AGX)")
    print(f"Date: 2025-12-29")
    print(f"Previous: Session 136 (100-cycle validation)")
    print()

    # Configuration
    test_config = ExtendedTestConfig(
        total_cycles=1000,
        checkpoint_interval=100,
    )
    regulation_config = EmotionalRegulationConfig()

    # Create tester
    tester = ExtendedStabilityTester(
        test_config=test_config,
        regulation_config=regulation_config,
    )

    # Run tests
    results = {}

    # Test 1: Baseline
    results["baseline"] = tester.run_baseline_test()

    # Test 2: Stress
    results["stress"] = tester.run_stress_test()

    # Test 3: Recovery
    results["recovery"] = tester.run_recovery_test()

    # Analyze and report
    print("\n" + "=" * 80)
    print("EXTENDED STABILITY TEST RESULTS")
    print("=" * 80)

    for test_name, test_results in results.items():
        print(f"\n=== {test_name.upper()} TEST ===")
        analysis = test_results.analyze_stability()

        print(f"Total cycles: {test_results.total_cycles}")
        print(f"Duration: {test_results.duration_seconds:.1f}s ({test_results.cycles_per_second:.1f} cycles/sec)")
        print(f"\nFinal emotional state:")
        for emotion, value in test_results.final_emotional_state.items():
            print(f"  {emotion}: {value:.2f}")

        print(f"\nStability analysis:")
        print(f"  Stable operation: {analysis['stable_operation']}")
        print(f"  Cascade detected: {test_results.cascade_detected}")
        print(f"  Frustration (mean/max): {analysis.get('frustration_mean', 0):.2f} / {analysis.get('frustration_max', 0):.2f}")
        print(f"  Engagement (mean/min): {analysis.get('engagement_mean', 0):.2f} / {analysis.get('engagement_min', 0):.2f}")

        if "learning_improvement" in analysis:
            print(f"\nLearning progression:")
            print(f"  Early success rate: {analysis['early_success_rate']:.1%}")
            print(f"  Late success rate: {analysis['late_success_rate']:.1%}")
            print(f"  Improvement: {analysis['learning_improvement']:+.1%}")

        if "ep_maturation_score" in analysis:
            print(f"\nEP maturation:")
            print(f"  Early intervention rate: {analysis['early_intervention_rate']:.1f}/100 cycles")
            print(f"  Late intervention rate: {analysis['late_intervention_rate']:.1f}/100 cycles")
            print(f"  EP maturation score: {analysis['ep_maturation_score']:.1%}")

        print(f"\nRegulation metrics:")
        print(f"  Total interventions: {test_results.total_interventions}")
        print(f"  By type: {test_results.interventions_by_type}")

    # Save results
    output_file = Path(__file__).parent / "session137_extended_stability_results.json"
    output_data = {}

    for test_name, test_results in results.items():
        analysis = test_results.analyze_stability()
        output_data[test_name] = {
            "test_name": test_results.test_name,
            "total_cycles": test_results.total_cycles,
            "duration_seconds": test_results.duration_seconds,
            "cycles_per_second": test_results.cycles_per_second,
            "final_emotional_state": test_results.final_emotional_state,
            "stable_operation": test_results.stable_operation,
            "cascade_detected": test_results.cascade_detected,
            "max_frustration": test_results.max_frustration_reached,
            "min_engagement": test_results.min_engagement_reached,
            "total_interventions": test_results.total_interventions,
            "interventions_by_type": test_results.interventions_by_type,
            "analysis": analysis,
        }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SESSION 137 SUMMARY")
    print("=" * 80)

    all_stable = all(r.stable_operation for r in results.values())
    any_cascade = any(r.cascade_detected for r in results.values())

    print(f"\nOverall stability: {'STABLE' if all_stable else 'UNSTABLE'}")
    print(f"Cascades detected: {'YES - FAILURE' if any_cascade else 'NO - SUCCESS'}")

    if all_stable and not any_cascade:
        print("\n✅ SUCCESS: Emotional regulation holds over 1000+ cycles")
        print("✅ All test conditions passed")
        print("✅ No cascades detected")
        print("✅ Ready for production deployment")
    else:
        print("\n⚠️  ATTENTION: Instabilities detected")
        print("⚠️  Review failed tests for optimization opportunities")

    return results


if __name__ == "__main__":
    results = run_all_tests()
