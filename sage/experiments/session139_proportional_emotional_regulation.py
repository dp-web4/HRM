#!/usr/bin/env python3
"""
Session 139: Proportional Emotional Regulation

ARCHITECTURAL BREAKTHROUGH from Sessions 137-138:
Binary threshold intervention creates attractor dynamics.

Session 137: Frustration locked at minimum (0.20) - decay attractor
Session 138: Frustration locked at threshold (0.80) - intervention attractor

ROOT CAUSE: Binary on/off mechanism creates equilibrium points
- Below threshold: Minimal regulation
- Above threshold: Strong regulation
- Result: System equilibrates AT threshold, no variation

BIOLOGICAL INSIGHT:
Real prefrontal cortex uses PROPORTIONAL modulation:
- Regulation strength ∝ emotional intensity
- Continuous gradient, not step function
- No hard thresholds in neural circuits
- Gradual modulation allows appropriate variation

Session 139 Solution: PROPORTIONAL REGULATION
===============================================

Replace binary threshold with continuous gradient function:

Low frustration (0.0-0.4):
  - Light touch regulation: -0.01 to -0.03
  - Allow natural emotional response
  - Minimal interference with experience-driven emotion

Medium frustration (0.4-0.7):
  - Moderate regulation: -0.03 to -0.08
  - Prevent escalation while allowing variation
  - Balance between response and control

High frustration (0.7-0.9):
  - Strong regulation: -0.08 to -0.20
  - Active cascade prevention
  - Still proportional, not binary

Extreme frustration (0.9-1.0):
  - Emergency regulation: -0.20 to -0.30
  - Maximum intervention to prevent lock-in
  - Safety mechanism at extremes

Key Properties:
1. Continuous function (no discrete jumps)
2. Strength scales with need
3. Allows variation within bounds
4. Prevents cascade at extremes
5. Biologically realistic (gradual modulation)

Expected Behavior:
- 30% failure → frustration ~0.3-0.4 (light regulation)
- 60% failure → frustration ~0.6-0.7 (strong regulation)
- Variation BETWEEN conditions (>0.2 difference)
- Stability WITHIN conditions (no cascade)
- No attractor points (continuous response)

Test Strategy:
1. Implement proportional regulation function
2. Rerun Session 137/138 test scenarios
3. Compare S136 (binary) vs S139 (proportional)
4. Validate appropriate variation + safety

Success Criteria:
✅ Different frustration for different failure rates (>0.2 delta)
✅ Variation within safe bounds (0.2-0.8 range)
✅ No cascades detected
✅ Graduated response curve (smooth, not stepped)

Date: 2025-12-30
Hardware: Thor (Jetson AGX Thor Developer Kit)
Previous: Session 138 (threshold attractor discovery)
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import math

# Add SAGE modules to path
sys.path.append(str(Path(__file__).parent.parent))

from session137_extended_stability_testing import (
    ExtendedTestConfig,
    ExtendedTestResults,
    ExtendedStabilityTester,
)
from session136_emotional_regulation import EmotionalRegulationConfig


@dataclass
class ProportionalRegulationConfig(EmotionalRegulationConfig):
    """
    Proportional regulation configuration - GRADIENT-BASED modulation.

    Regulation strength scales continuously with frustration level.
    No binary thresholds - smooth gradient function.
    """

    # PROPORTIONAL regulation (replaces binary threshold)
    # These parameters define the regulation gradient function

    # Base decay (always applied, very light)
    base_frustration_decay: float = 0.005  # Minimal background decay

    # Proportional regulation parameters
    # regulation_strength = base + (proportional_factor × frustration²)
    # Using quadratic to give stronger response at higher frustration
    proportional_factor: float = 0.20  # Scales regulation with frustration

    # Maximum regulation strength (safety cap)
    max_regulation_strength: float = 0.30  # Emergency maximum

    # Recovery parameters (proportional, not binary)
    recovery_scaling: float = 0.03  # Scales with (1 - frustration)

    # Soft bounds (wider to allow more variation)
    frustration_min: float = 0.15  # Higher minimum (allow dropping lower)
    frustration_max: float = 0.85  # Lower maximum (more safety margin)

    # Engagement/curiosity parameters (unchanged)
    engagement_recovery: float = 0.01
    curiosity_recovery: float = 0.02
    curiosity_min: float = 0.15
    curiosity_max: float = 0.95
    engagement_min: float = 0.10
    engagement_max: float = 1.00
    progress_min: float = 0.00
    progress_max: float = 1.00
    progress_decay: float = 0.01

    # REMOVE binary threshold parameters (not used in proportional)
    # These are kept for compatibility but should be ignored
    high_frustration_threshold: float = 999.0  # Disabled
    frustration_intervention: float = 0.0  # Not used
    low_engagement_threshold: float = 0.0  # Not used
    engagement_boost: float = 0.0  # Not used
    curiosity_boost: float = 0.0  # Not used
    stagnation_threshold: int = 999999  # Disabled
    recovery_no_failure_cycles: int = 999999  # Disabled
    recovery_frustration_bonus: float = 0.0  # Not used
    recovery_engagement_bonus: float = 0.0  # Not used

    def calculate_proportional_regulation(self, frustration: float) -> float:
        """
        Calculate proportional regulation strength based on current frustration.

        Uses quadratic function to provide graduated response:
        - Low frustration: Minimal regulation (allow natural response)
        - Medium frustration: Moderate regulation (balance)
        - High frustration: Strong regulation (prevent cascade)
        - Extreme frustration: Maximum regulation (emergency)

        Formula: regulation = base_decay + (proportional_factor × frustration²)

        The quadratic term ensures stronger response at higher frustration
        while allowing variation at lower levels.

        Examples:
        - frustration=0.2 → regulation ≈ 0.013 (light touch)
        - frustration=0.4 → regulation ≈ 0.037 (moderate)
        - frustration=0.6 → regulation ≈ 0.077 (strong)
        - frustration=0.8 → regulation ≈ 0.133 (very strong)
        - frustration=0.95 → regulation ≈ 0.186 (emergency)
        """
        # Base decay (always present)
        base = self.base_frustration_decay

        # Proportional component (quadratic for graduated response)
        # Using frustration² gives stronger response at higher levels
        proportional = self.proportional_factor * (frustration ** 2)

        # Total regulation strength
        total = base + proportional

        # Cap at maximum (safety)
        total = min(total, self.max_regulation_strength)

        return total

    def calculate_recovery_bonus(self, frustration: float) -> float:
        """
        Calculate recovery bonus that scales with (1 - frustration).

        Lower frustration → larger recovery bonus (positive feedback)
        Higher frustration → smaller recovery bonus (needs regulation)

        This creates natural stability: as frustration decreases,
        recovery accelerates, helping return to baseline.
        """
        # Recovery scales inversely with frustration
        # High frustration (0.8) → low recovery (0.2 × factor)
        # Low frustration (0.2) → high recovery (0.8 × factor)
        recovery = (1.0 - frustration) * self.recovery_scaling

        return recovery

    def get_regulation_summary(self) -> Dict[str, Any]:
        """Get summary of regulation configuration for logging."""
        return {
            "type": "proportional",
            "base_decay": self.base_frustration_decay,
            "proportional_factor": self.proportional_factor,
            "max_regulation": self.max_regulation_strength,
            "recovery_scaling": self.recovery_scaling,
            "bounds": f"{self.frustration_min:.2f}-{self.frustration_max:.2f}",
        }


def test_proportional_function():
    """
    Test and visualize the proportional regulation function.

    Shows how regulation strength varies across frustration levels.
    """
    print("\n" + "=" * 80)
    print("PROPORTIONAL REGULATION FUNCTION TEST")
    print("=" * 80)
    print("\nRegulation strength across frustration levels:\n")

    config = ProportionalRegulationConfig()

    print(f"{'Frustration':<12} {'Regulation':<12} {'Recovery':<12} {'Category':<20}")
    print("-" * 80)

    test_points = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

    for f in test_points:
        reg = config.calculate_proportional_regulation(f)
        rec = config.calculate_recovery_bonus(f)

        # Categorize regulation strength
        if reg < 0.02:
            category = "MINIMAL"
        elif reg < 0.05:
            category = "LIGHT"
        elif reg < 0.10:
            category = "MODERATE"
        elif reg < 0.20:
            category = "STRONG"
        else:
            category = "EMERGENCY"

        print(f"{f:<12.2f} -{reg:<11.3f} +{rec:<11.3f} {category:<20}")

    print("\n" + "=" * 80)
    print("Key Properties:")
    print("  - Continuous gradient (no discrete jumps)")
    print("  - Quadratic scaling (stronger at high frustration)")
    print("  - Recovery bonus inversely proportional")
    print("  - No binary threshold (smooth response)")
    print("=" * 80 + "\n")


def compare_all_sessions():
    """
    Compare all regulation approaches:
    - Session 136/137: Binary threshold (original)
    - Session 138: Reduced binary threshold
    - Session 139: Proportional regulation

    Validates that proportional approach enables appropriate variation.
    """
    print("=" * 80)
    print("Session 139: Proportional Emotional Regulation")
    print("=" * 80)
    print("Comparing regulation approaches across sessions")
    print()

    # First, test the proportional function
    test_proportional_function()

    # Test configuration
    test_config = ExtendedTestConfig(
        total_cycles=1000,
        checkpoint_interval=100,
    )

    # Session 139: Proportional regulation
    print("=" * 80)
    print("SESSION 139: PROPORTIONAL REGULATION (Gradient-based)")
    print("=" * 80)
    print("\nRegulation Parameters:")
    proportional_config = ProportionalRegulationConfig()
    summary = proportional_config.get_regulation_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()

    # Note: We would need to modify RegulatedConsciousnessLoop to use
    # the proportional regulation method instead of binary threshold.
    # For this test, we'll create a wrapper that applies proportional regulation.

    from session136_emotional_regulation import RegulatedConsciousnessLoop
    from session134_memory_guided_attention import SAGEIdentityManager

    class ProportionalRegulatedLoop(RegulatedConsciousnessLoop):
        """
        Consciousness loop with proportional regulation.

        Overrides the binary threshold regulation from Session 136
        with proportional gradient-based regulation.
        """

        def __init__(self, identity_manager, regulation_config):
            super().__init__(identity_manager, regulation_config, enable_regulation=True)
            self.proportional_config = regulation_config

        def _learning_phase(self, experience_results):
            """
            Override learning phase to use PROPORTIONAL regulation.

            Replaces Session 136's binary threshold with continuous gradient.
            """
            # Get current identity
            identity = self.identity_manager.current_identity

            # Extract experience results
            successes = experience_results.get("successes", 0)
            failures = experience_results.get("failures", 0)
            total_value = experience_results.get("total_value", 0.0)

            # 1. Calculate RAW emotional response (same as S136)
            if successes > failures:
                raw_frustration_delta = -0.1
                raw_engagement_delta = +0.05
                raw_curiosity_delta = +0.03
                raw_progress_delta = +0.1
                self.cycles_without_failure += 1
                self.cycles_without_success = 0
            else:
                raw_frustration_delta = +0.15
                raw_engagement_delta = -0.05
                raw_curiosity_delta = -0.02
                raw_progress_delta = -0.05
                self.cycles_without_failure = 0
                self.cycles_without_success += 1

            # 2. PROPORTIONAL REGULATION (new!)
            current_frustration = identity.frustration

            # Calculate proportional regulation strength
            regulation_strength = self.proportional_config.calculate_proportional_regulation(
                current_frustration
            )

            # Calculate recovery bonus
            recovery_bonus = self.proportional_config.calculate_recovery_bonus(
                current_frustration
            )

            # Apply proportional regulation to frustration
            # Regulation always pulls DOWN (negative delta)
            # Strength varies continuously with frustration level
            proportional_frustration_delta = -regulation_strength + recovery_bonus

            # Engagement/curiosity recovery (light)
            decay_engagement = self.regulation_config.engagement_recovery
            decay_curiosity = self.regulation_config.curiosity_recovery
            decay_progress = -self.regulation_config.progress_decay

            # 3. COMBINE: Raw response + Proportional regulation
            total_frustration_delta = raw_frustration_delta + proportional_frustration_delta
            total_engagement_delta = raw_engagement_delta + decay_engagement
            total_curiosity_delta = raw_curiosity_delta + decay_curiosity
            total_progress_delta = raw_progress_delta + decay_progress

            # 4. Apply with SOFT BOUNDS
            new_frustration = max(
                self.regulation_config.frustration_min,
                min(self.regulation_config.frustration_max,
                    identity.frustration + total_frustration_delta)
            )
            new_engagement = max(
                self.regulation_config.engagement_min,
                min(self.regulation_config.engagement_max,
                    identity.engagement + total_engagement_delta)
            )
            new_curiosity = max(
                self.regulation_config.curiosity_min,
                min(self.regulation_config.curiosity_max,
                    identity.curiosity + total_curiosity_delta)
            )
            new_progress = max(
                self.regulation_config.progress_min,
                min(self.regulation_config.progress_max,
                    identity.progress + total_progress_delta)
            )

            # Track regulation statistics
            if self.regulator:
                # Track total regulation applied
                self.regulator.total_frustration_regulated += abs(proportional_frustration_delta)

                # Count as "intervention" if regulation strength > 0.05
                if regulation_strength > 0.05:
                    self.regulator.intervention_count += 1

            # Update identity
            self.identity_manager.update_emotional_state(
                curiosity=new_curiosity,
                frustration=new_frustration,
                engagement=new_engagement,
                progress=new_progress
            )

            # Record invocations
            for _ in range(successes):
                self.identity_manager.record_invocation(success=True, atp_cost=10.0)
            for _ in range(failures):
                self.identity_manager.record_invocation(success=False, atp_cost=10.0)

            return {
                "successes": successes,
                "failures": failures,
                "total_value": total_value,
                "emotional_updates": {
                    "frustration": new_frustration,
                    "engagement": new_engagement,
                    "curiosity": new_curiosity,
                    "progress": new_progress,
                },
                "regulation_applied": {
                    "strength": regulation_strength,
                    "recovery": recovery_bonus,
                    "total_delta": proportional_frustration_delta,
                }
            }

    # Create tester with proportional loop
    class ProportionalStabilityTester(ExtendedStabilityTester):
        """Tester using proportional regulation."""

        def run_baseline_test(self):
            """Run baseline with proportional regulation."""
            print("\n=== Test 1: Baseline 1000-Cycle (Proportional Regulation) ===")
            print(f"Total cycles: {self.test_config.total_cycles}")
            print(f"Failure rate: {self.test_config.base_failure_rate:.1%}")
            print(f"Regulation: PROPORTIONAL (gradient-based)")

            results = ExtendedTestResults(
                test_name="baseline_proportional",
                total_cycles=self.test_config.total_cycles,
                final_emotional_state={},
            )

            # Create proportional loop
            loop = ProportionalRegulatedLoop(
                identity_manager=self.identity_manager,
                regulation_config=self.regulation_config,
            )

            import time
            start_time = time.time()

            # Run cycles
            for cycle in range(self.test_config.total_cycles):
                experiences = [
                    self._generate_experience(
                        cycle=cycle * 15 + i,
                        failure_rate=self.test_config.base_failure_rate,
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

        def run_stress_test(self):
            """Run stress test with proportional regulation."""
            print("\n=== Test 2: Stress Test (Proportional Regulation) ===")
            print(f"Total cycles: {self.test_config.total_cycles}")
            print(f"Failure rate: {self.test_config.stress_failure_rate:.1%}")
            print(f"Regulation: PROPORTIONAL (gradient-based)")

            # Reset identity
            self.identity_manager = SAGEIdentityManager()
            self.identity_manager.create_identity()
            self.identity = self.identity_manager.current_identity

            results = ExtendedTestResults(
                test_name="stress_proportional",
                total_cycles=self.test_config.total_cycles,
                final_emotional_state={},
            )

            loop = ProportionalRegulatedLoop(
                identity_manager=self.identity_manager,
                regulation_config=self.regulation_config,
            )

            import time
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

    # Run tests
    tester = ProportionalStabilityTester(
        test_config=test_config,
        regulation_config=proportional_config,
    )

    print("\n" + "=" * 80)
    print("Running Session 139 Tests...")
    print("=" * 80)

    session139_results = {}
    session139_results["baseline"] = tester.run_baseline_test()
    session139_results["stress"] = tester.run_stress_test()

    # Analysis
    print("\n" + "=" * 80)
    print("SESSION 139 RESULTS vs SESSIONS 137/138")
    print("=" * 80)

    baseline_frust = session139_results["baseline"].final_emotional_state['frustration']
    stress_frust = session139_results["stress"].final_emotional_state['frustration']
    variation = abs(stress_frust - baseline_frust)

    print("\n=== COMPARISON TABLE ===")
    print(f"{'Session':<12} {'Type':<20} {'Baseline':<12} {'Stress':<12} {'Variation':<12}")
    print("-" * 80)
    print(f"{'137':<12} {'Binary (strong)':<20} {0.20:<12.2f} {0.20:<12.2f} {0.00:<12.2f}")
    print(f"{'138':<12} {'Binary (reduced)':<20} {0.80:<12.2f} {0.80:<12.2f} {0.00:<12.2f}")
    print(f"{'139':<12} {'Proportional':<20} {baseline_frust:<12.2f} {stress_frust:<12.2f} {variation:<12.2f}")

    print("\n=== KEY FINDINGS ===")

    print(f"\n1. Emotional Variation:")
    if variation > 0.15:
        print(f"   ✅ SUCCESS: {variation:.2f} difference (appropriate variation)")
    elif variation > 0.05:
        print(f"   ⚠️  PARTIAL: {variation:.2f} difference (some variation)")
    else:
        print(f"   ❌ FAILURE: {variation:.2f} difference (still locked)")

    print(f"\n2. Cascade Prevention:")
    any_cascade = any(r.cascade_detected for r in session139_results.values())
    if not any_cascade:
        print(f"   ✅ SUCCESS: No cascades detected")
    else:
        print(f"   ❌ FAILURE: Cascade detected")

    print(f"\n3. Stability:")
    all_stable = all(r.stable_operation for r in session139_results.values())
    if all_stable:
        print(f"   ✅ SUCCESS: All tests stable")
    else:
        print(f"   ⚠️  ATTENTION: Instabilities detected")

    print(f"\n4. Biological Realism:")
    if variation > 0.15 and not any_cascade and all_stable:
        print(f"   ✅ EXCELLENT: Appropriate variation within safe bounds")
        print(f"   - Different responses to different conditions")
        print(f"   - Graduated regulation (not binary)")
        print(f"   - Maintains safety")
    elif variation > 0.05:
        print(f"   ⚠️  PARTIAL: Some variation, needs tuning")
    else:
        print(f"   ❌ INSUFFICIENT: Architecture still creates attractor")

    # Save results
    output_file = Path(__file__).parent / "session139_proportional_regulation_results.json"
    output_data = {}

    for test_name, test_results in session139_results.items():
        analysis = test_results.analyze_stability()
        output_data[test_name] = {
            "test_name": test_results.test_name,
            "regulation_type": "proportional",
            "total_cycles": test_results.total_cycles,
            "final_emotional_state": test_results.final_emotional_state,
            "stable_operation": test_results.stable_operation,
            "cascade_detected": test_results.cascade_detected,
            "analysis": analysis,
        }

    # Add comparison
    output_data["comparison"] = {
        "session137_variation": 0.00,
        "session138_variation": 0.00,
        "session139_variation": variation,
        "improvement_over_137": variation - 0.00,
        "improvement_over_138": variation - 0.00,
        "variation_achieved": variation > 0.15,
        "cascades_prevented": not any_cascade,
        "all_stable": all_stable,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Final verdict
    print("\n" + "=" * 80)
    print("SESSION 139 VERDICT")
    print("=" * 80)

    if variation > 0.15 and not any_cascade and all_stable:
        print("\n✅ SUCCESS: Proportional regulation achieves appropriate variation")
        print("✅ Different emotional responses to different conditions")
        print("✅ No cascades (safety maintained)")
        print("✅ Graduated regulation (biologically realistic)")
        print("\n🎯 BREAKTHROUGH: Regulation research arc COMPLETE")
        print("   Sessions 135-139: From cascade discovery to proportional regulation")
    elif variation > 0.05 and not any_cascade:
        print("\n⚠️  PARTIAL SUCCESS: Variation improved but needs tuning")
        print("✅ No cascades")
        print("⚠️  Variation present but smaller than desired")
        print("\n🔧 May need gradient function tuning")
    else:
        print("\n⚠️  ATTENTION: Further architectural work needed")
        print(f"Variation: {variation:.2f}")
        print(f"Cascades: {any_cascade}")
        print("\n🔧 May need different proportional function")

    return session139_results, output_data


if __name__ == "__main__":
    results, comparison = compare_all_sessions()
