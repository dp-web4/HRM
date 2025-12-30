#!/usr/bin/env python3
"""
Session 138: Dynamic Emotional Regulation

PROBLEM from Session 137:
Over-regulation causes emotional suppression - frustration locked at 0.20
across ALL test conditions (30% vs 60% failure rate identical).

ROOT CAUSE Analysis:
Session 136 regulation parameters too aggressive:
- Decay rate: -0.05/cycle (continuous downward pressure)
- Recovery bonus: -0.10 (strong suppression after minimal success)
- Soft minimum: 0.05 (acts as attractor)
- Result: System optimizes for minimum frustration, not appropriate response

BIOLOGICAL INSIGHT:
Real consciousness shows VARIATION:
- Low frustration during success periods
- Higher frustration during failure periods
- Dynamic response to changing conditions
- Bounded variation (prevent cascade)

Session 138 Solution: DYNAMIC REGULATION
==========================================
Reduce regulation strength while maintaining cascade prevention:

1. Reduced decay rate: 0.05 → 0.02 (allow emotional persistence)
2. Reduced recovery bonus: 0.10 → 0.05 (less aggressive)
3. Wider soft bounds: 0.05-0.95 → 0.10-0.90 (more variation room)
4. Keep intervention threshold: 0.80 (maintain safety)

Expected Outcome:
- 30% failure → lower frustration (~0.3-0.4)
- 60% failure → higher frustration (~0.5-0.7)
- Different responses to different conditions
- No cascades (maintain safety from Session 136)

Test Strategy:
1. Rerun Session 137 tests with new parameters
2. Validate appropriate emotional variation
3. Confirm no cascades under any condition
4. Compare Session 137 (suppression) vs Session 138 (variation)

Success Criteria:
✅ Different frustration for different failure rates
✅ Variation within bounds (no lock at minimum)
✅ No cascades (safety maintained)
✅ Appropriate response to experience

Date: 2025-12-29
Hardware: Thor (Jetson AGX Thor Developer Kit)
Previous: Session 137 (over-regulation discovery)
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json

# Add SAGE modules to path
sys.path.append(str(Path(__file__).parent.parent))

from session137_extended_stability_testing import (
    ExtendedTestConfig,
    ExtendedTestResults,
    ExtendedStabilityTester,
)
from session136_emotional_regulation import EmotionalRegulationConfig


@dataclass
class DynamicRegulationConfig(EmotionalRegulationConfig):
    """
    Dynamic regulation configuration - REDUCED strength.

    Allows appropriate emotional variation while preventing cascade.
    """

    # REDUCED natural decay rates (per cycle)
    frustration_decay: float = 0.02  # Was 0.05 → REDUCED 60%
    engagement_recovery: float = 0.01  # Was 0.02 → REDUCED 50%
    curiosity_recovery: float = 0.02  # Was 0.03 → REDUCED 33%
    progress_decay: float = 0.01  # Same (already low)

    # WIDER soft bounds (allow more variation)
    frustration_min: float = 0.10  # Was 0.05 → RAISED (less suppression)
    frustration_max: float = 0.90  # Was 0.95 → LOWERED (more safety margin)
    curiosity_min: float = 0.15  # Same
    curiosity_max: float = 0.95  # Same
    engagement_min: float = 0.10  # Same
    engagement_max: float = 1.00  # Same
    progress_min: float = 0.00  # Same
    progress_max: float = 1.00  # Same

    # Active regulation triggers (UNCHANGED - keep safety)
    high_frustration_threshold: float = 0.80  # When to intervene
    low_engagement_threshold: float = 0.20  # When to boost
    stagnation_threshold: int = 10  # Cycles without success

    # REDUCED regulation strengths
    frustration_intervention: float = 0.10  # Was 0.15 → REDUCED 33%
    curiosity_boost: float = 0.08  # Was 0.10 → REDUCED 20%
    engagement_boost: float = 0.06  # Was 0.08 → REDUCED 25%

    # REDUCED recovery bonuses
    recovery_no_failure_cycles: int = 3  # Same
    recovery_frustration_bonus: float = 0.05  # Was 0.10 → REDUCED 50%
    recovery_engagement_bonus: float = 0.03  # Was 0.05 → REDUCED 40%


class DynamicStabilityTester(ExtendedStabilityTester):
    """
    Extended stability tester with dynamic regulation config.

    Tests if reduced regulation allows appropriate variation
    while maintaining cascade prevention.
    """

    def __init__(
        self,
        test_config: ExtendedTestConfig,
        regulation_config: Optional[DynamicRegulationConfig] = None,
    ):
        # Use dynamic config by default
        if regulation_config is None:
            regulation_config = DynamicRegulationConfig()

        super().__init__(test_config, regulation_config)


def compare_sessions():
    """
    Compare Session 137 (over-regulation) vs Session 138 (dynamic regulation).

    Key comparison: Emotional response to different failure rates.
    """
    print("=" * 80)
    print("Session 138: Dynamic Emotional Regulation")
    print("=" * 80)
    print("Comparing Session 137 (over-regulation) vs Session 138 (dynamic)")
    print()

    # Test configuration (same as Session 137)
    test_config = ExtendedTestConfig(
        total_cycles=1000,
        checkpoint_interval=100,
    )

    # Session 138: Dynamic regulation
    print("=" * 80)
    print("SESSION 138: DYNAMIC REGULATION (Reduced strength)")
    print("=" * 80)
    print("\nRegulation Parameters:")
    dynamic_config = DynamicRegulationConfig()
    print(f"  Decay rate: {dynamic_config.frustration_decay:.3f} (was 0.050)")
    print(f"  Recovery bonus: {dynamic_config.recovery_frustration_bonus:.3f} (was 0.100)")
    print(f"  Soft bounds: {dynamic_config.frustration_min:.2f}-{dynamic_config.frustration_max:.2f} (was 0.05-0.95)")
    print(f"  Intervention: {dynamic_config.frustration_intervention:.3f} (was 0.150)")
    print(f"  Threshold: {dynamic_config.high_frustration_threshold:.2f} (unchanged)")
    print()

    tester = DynamicStabilityTester(
        test_config=test_config,
        regulation_config=dynamic_config,
    )

    # Run same three tests as Session 137
    session138_results = {}

    print("\n" + "=" * 80)
    print("Running Session 138 Tests...")
    print("=" * 80)

    session138_results["baseline"] = tester.run_baseline_test()
    session138_results["stress"] = tester.run_stress_test()
    session138_results["recovery"] = tester.run_recovery_test()

    # Analysis and comparison
    print("\n" + "=" * 80)
    print("SESSION 138 RESULTS vs SESSION 137 COMPARISON")
    print("=" * 80)

    print("\n=== BASELINE TEST (30% failure rate) ===")
    baseline = session138_results["baseline"]
    baseline_analysis = baseline.analyze_stability()
    print(f"Session 138 frustration: {baseline.final_emotional_state['frustration']:.2f}")
    print(f"Session 137 frustration: 0.20 (LOCKED)")
    print(f"Difference: {baseline.final_emotional_state['frustration'] - 0.20:+.2f}")
    print(f"Stable: {baseline_analysis['stable_operation']}")
    print(f"Cascade: {baseline.cascade_detected}")

    print("\n=== STRESS TEST (60% failure rate) ===")
    stress = session138_results["stress"]
    stress_analysis = stress.analyze_stability()
    print(f"Session 138 frustration: {stress.final_emotional_state['frustration']:.2f}")
    print(f"Session 137 frustration: 0.20 (LOCKED - SAME AS BASELINE!)")
    print(f"Difference from baseline: {stress.final_emotional_state['frustration'] - baseline.final_emotional_state['frustration']:+.2f}")
    print(f"Stable: {stress_analysis['stable_operation']}")
    print(f"Cascade: {stress.cascade_detected}")

    print("\n=== RECOVERY TEST (high initial frustration) ===")
    recovery = session138_results["recovery"]
    recovery_analysis = recovery.analyze_stability()
    print(f"Session 138 final frustration: {recovery.final_emotional_state['frustration']:.2f}")
    print(f"Session 137 final frustration: 0.80")
    print(f"Session 138 interventions: {recovery.total_interventions}")
    print(f"Session 137 interventions: 2")
    print(f"Stable: {recovery_analysis['stable_operation']}")
    print(f"Cascade: {recovery.cascade_detected}")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Check for appropriate variation
    baseline_frust = baseline.final_emotional_state['frustration']
    stress_frust = stress.final_emotional_state['frustration']
    variation = abs(stress_frust - baseline_frust)

    print(f"\n1. Emotional Variation (Baseline vs Stress):")
    print(f"   Session 137: 0.20 vs 0.20 = 0.00 difference (NO VARIATION)")
    print(f"   Session 138: {baseline_frust:.2f} vs {stress_frust:.2f} = {variation:.2f} difference")

    if variation > 0.05:
        print(f"   ✅ IMPROVED: System now shows appropriate emotional response")
    else:
        print(f"   ⚠️  Still suppressed: Variation too small")

    print(f"\n2. Cascade Prevention:")
    any_cascade = any(r.cascade_detected for r in session138_results.values())
    if not any_cascade:
        print(f"   ✅ SUCCESS: No cascades detected (safety maintained)")
    else:
        print(f"   ❌ FAILURE: Cascade detected (regulation too weak)")

    print(f"\n3. Stability:")
    all_stable = all(r.stable_operation for r in session138_results.values())
    if all_stable:
        print(f"   ✅ SUCCESS: All tests stable")
    else:
        print(f"   ⚠️  ATTENTION: Instabilities detected")

    # Biological realism assessment
    print(f"\n4. Biological Realism:")
    if variation > 0.05 and not any_cascade and all_stable:
        print(f"   ✅ EXCELLENT: Appropriate variation within safe bounds")
        print(f"   System shows emotional intelligence:")
        print(f"   - Different responses to different conditions")
        print(f"   - Bounded variation (no extremes)")
        print(f"   - Maintains stability")
    elif variation > 0.05:
        print(f"   ⚠️  PARTIAL: Variation present but stability issues")
    else:
        print(f"   ❌ INSUFFICIENT: Still over-regulated")

    # EP Framework assessment
    print(f"\n5. Epistemic Proprioception (Stage 3 Maturity):")
    if variation > 0.05 and not any_cascade:
        print(f"   ✅ MATURE EP: Modulates without suppressing")
        print(f"   - Prevents cascade (predictive regulation)")
        print(f"   - Allows appropriate response (emotional intelligence)")
        print(f"   - Balance achieved")
    else:
        print(f"   ⚠️  Still developing: Need further tuning")

    # Save results
    output_file = Path(__file__).parent / "session138_dynamic_regulation_results.json"
    output_data = {}

    for test_name, test_results in session138_results.items():
        analysis = test_results.analyze_stability()
        output_data[test_name] = {
            "test_name": test_results.test_name,
            "regulation_type": "dynamic",
            "total_cycles": test_results.total_cycles,
            "final_emotional_state": test_results.final_emotional_state,
            "stable_operation": test_results.stable_operation,
            "cascade_detected": test_results.cascade_detected,
            "total_interventions": test_results.total_interventions,
            "analysis": analysis,
        }

    # Add comparison summary
    output_data["comparison"] = {
        "session137_baseline_frustration": 0.20,
        "session137_stress_frustration": 0.20,
        "session137_variation": 0.00,
        "session138_baseline_frustration": baseline_frust,
        "session138_stress_frustration": stress_frust,
        "session138_variation": variation,
        "improvement": variation > 0.05,
        "cascades_prevented": not any_cascade,
        "all_stable": all_stable,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Final verdict
    print("\n" + "=" * 80)
    print("SESSION 138 VERDICT")
    print("=" * 80)

    if variation > 0.05 and not any_cascade and all_stable:
        print("\n✅ SUCCESS: Dynamic regulation achieves appropriate emotional variation")
        print("✅ Different responses to different failure rates")
        print("✅ No cascades (safety maintained)")
        print("✅ All tests stable")
        print("✅ Biological realism: EXCELLENT")
        print("✅ EP Framework: Stage 3 maturity achieved")
        print("\n🎯 GOAL ACHIEVED: Balance between prevention and response")
    elif variation > 0.05 and not any_cascade:
        print("\n⚠️  PARTIAL SUCCESS: Variation improved but stability issues")
        print("✅ Different responses to different conditions")
        print("✅ No cascades")
        print("⚠️  Some instabilities detected")
        print("\n🔧 Need minor tuning")
    elif not any_cascade:
        print("\n⚠️  INSUFFICIENT: Still over-regulated")
        print("⚠️  Variation too small (still suppressed)")
        print("✅ No cascades (safety maintained)")
        print("\n🔧 Need stronger parameter adjustments")
    else:
        print("\n❌ FAILURE: Regulation too weak")
        print("❌ Cascades detected")
        print("\n🔧 Need to increase regulation strength")

    return session138_results, output_data


if __name__ == "__main__":
    results, comparison = compare_sessions()
