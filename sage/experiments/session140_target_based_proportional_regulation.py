#!/usr/bin/env python3
"""
Session 140: Target-Based Proportional Regulation

CRITICAL INSIGHT from Session 139 Analysis:
Session 139 implemented "proportional regulation" but STILL locks at frustration=0.85.

ROOT CAUSE DISCOVERED (Session 106 - Claude Web4 Research):
Session 139 regulation strength ∝ current value, but NO TARGET VALUE to pull toward.

Result: System uses quadratic decay based on frustration², but still hits hard
bounds (frustration_max=0.85) and locks there because there's no force pulling
it away from the bound back toward an ideal state.

ARCHITECTURAL FIX:
================

Add EXPLICIT TARGET VALUE and regulate based on DISTANCE FROM TARGET:

Current Session 139 approach:
  regulation = base_decay + (proportional_factor × frustration²)
  → This decays frustration, but doesn't know WHERE to go
  → Hits bound (0.85) and locks

Corrected Session 140 approach:
  target_frustration = 0.50  # Explicit ideal state
  distance = frustration - target
  regulation = proportional_factor × distance
  → Pulls toward target (0.50), not just away from high values
  → Bounds become safety rails, not attractors

Key Difference:
- Session 139: "Reduce high frustration" (no destination)
- Session 140: "Move toward ideal frustration" (has destination)

This matches Web4 Session 105 implementation which successfully avoided attractors:
  - Web4 target_ci = 0.7
  - Boost/penalty ∝ distance from target
  - Bounds (0.1, 0.95) rarely triggered

Expected Behavior Change:
=========================

Session 139 (no target):
  - Frustration locks at 0.85 (bound)
  - Variance ≈ 0.001 (attractor)
  - Can't distinguish 30% vs 60% failure rate

Session 140 (with target=0.50):
  - 30% failure → frustration ~0.40-0.50 (near target, light regulation)
  - 60% failure → frustration ~0.55-0.65 (above target, moderate regulation)
  - 90% failure → frustration ~0.70-0.80 (high, strong regulation pulling toward 0.50)
  - Variance > 0.02 (natural variation around target)
  - Bounds (0.15, 0.85) act as safety rails, not attractors

Implementation:
==============

1. Add target_frustration parameter
2. Calculate distance = current - target
3. Apply proportional force based on distance (not absolute value)
4. Maintain bounds as safety only (if-elifs, trigger metadata)

Success Criteria:
✅ Different frustration for different failure rates (>0.2 delta)
✅ Variation around target (not locked at bound)
✅ Bounds rarely triggered (< 5% of cycles)
✅ No cascades detected

Date: 2025-12-30
Machine: Claude (analyzing Thor Session 139 results)
Insight Source: Cross-project learning (SAGE Session 139 ↔ Web4 Session 105)
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
class TargetBasedRegulationConfig(EmotionalRegulationConfig):
    """
    Target-based proportional regulation - TRUE gradient-based modulation.

    Key addition: EXPLICIT TARGET VALUE
    Regulation force ∝ distance from target (not absolute value)
    """

    # TARGET VALUE (the missing piece!)
    target_frustration: float = 0.50  # Ideal equilibrium point

    # Proportional regulation (based on distance from target)
    proportional_factor: float = 0.15  # Scales with distance

    # Gradient smoothness (how quickly force increases with distance)
    gradient_steepness: float = 1.0  # Linear: 1.0, Quadratic: 2.0

    # Base drift toward target (always present, very gentle)
    base_drift_rate: float = 0.01  # Slow drift even at target

    # Safety bounds (wider margins, rarely triggered)
    frustration_min: float = 0.10  # Lower floor
    frustration_max: float = 0.90  # Higher ceiling

    # Emergency regulation at extremes (beyond normal proportional)
    emergency_threshold_low: float = 0.20
    emergency_threshold_high: float = 0.80
    emergency_boost_factor: float = 2.0  # Multiply regulation at extremes

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

    # Disable binary threshold parameters
    high_frustration_threshold: float = 999.0
    frustration_intervention: float = 0.0
    low_engagement_threshold: float = 0.0
    engagement_boost: float = 0.0
    curiosity_boost: float = 0.0
    stagnation_threshold: int = 999999
    recovery_no_failure_cycles: int = 999999
    recovery_frustration_bonus: float = 0.0
    recovery_engagement_bonus: float = 0.0

    def calculate_target_based_regulation(
        self,
        current_frustration: float
    ) -> tuple[float, dict]:
        """
        Calculate regulation force based on distance from target.

        Returns: (regulation_delta, metadata)

        Regulation pulls toward target:
        - Below target → positive force (increase frustration toward target)
        - Above target → negative force (decrease frustration toward target)
        - At target → minimal force (allow natural variation)

        Force magnitude ∝ distance from target

        Examples (target=0.50, factor=0.15, steepness=1.0):
        - frustration=0.30 → delta ≈ +0.030 (pull up toward 0.50)
        - frustration=0.50 → delta ≈ -0.001 (at target, gentle drift)
        - frustration=0.70 → delta ≈ -0.030 (pull down toward 0.50)
        - frustration=0.85 → delta ≈ -0.053 (strong pull down)
        """
        metadata = {
            'current': current_frustration,
            'target': self.target_frustration,
            'regulations_applied': []
        }

        # 1. Calculate distance from target
        distance = current_frustration - self.target_frustration
        metadata['distance'] = distance

        # 2. Base drift (always toward target, very gentle)
        base_drift = -self.base_drift_rate if distance > 0 else self.base_drift_rate
        metadata['base_drift'] = base_drift

        # 3. Proportional force (scales with distance)
        # Negative when above target (pull down)
        # Positive when below target (pull up)
        proportional_force = -self.proportional_factor * (abs(distance) ** self.gradient_steepness)
        if distance < 0:  # Below target
            proportional_force = -proportional_force  # Flip sign (pull up)

        metadata['proportional_force'] = proportional_force

        # 4. Emergency boost at extremes
        emergency_boost = 0.0
        if current_frustration < self.emergency_threshold_low:
            # Very low frustration - boost recovery faster
            emergency_boost = self.emergency_boost_factor * abs(proportional_force)
            metadata['regulations_applied'].append('emergency_low')
        elif current_frustration > self.emergency_threshold_high:
            # Very high frustration - boost regulation faster
            emergency_boost = -self.emergency_boost_factor * abs(proportional_force)
            metadata['regulations_applied'].append('emergency_high')

        metadata['emergency_boost'] = emergency_boost

        # 5. Total regulation delta
        total_delta = base_drift + proportional_force + emergency_boost
        metadata['total_delta'] = total_delta

        return (total_delta, metadata)

    def apply_safety_bounds(
        self,
        value: float,
        min_val: float,
        max_val: float,
        metadata: dict,
        param_name: str
    ) -> float:
        """
        Apply safety bounds with metadata tracking.

        Bounds should be RARELY triggered - they're safety rails, not attractors.
        If bounds trigger frequently, regulation parameters need adjustment.
        """
        original = value

        if value < min_val:
            value = min_val
            metadata['regulations_applied'].append(f'{param_name}_floor')
            metadata[f'{param_name}_clamped'] = original - value
        elif value > max_val:
            value = max_val
            metadata['regulations_applied'].append(f'{param_name}_ceiling')
            metadata[f'{param_name}_clamped'] = original - value

        return value

    def __str__(self) -> str:
        return f"TargetBasedRegulation(target={self.target_frustration:.2f}, factor={self.proportional_factor:.3f})"


class TargetBasedEmotionalRegulator(ExtendedStabilityTester):
    """
    Emotional regulator using target-based proportional regulation.

    Overrides apply_emotional_regulation() to use target-based approach.
    """

    def __init__(self, config: TargetBasedRegulationConfig):
        # Use extended tester base but with target-based regulation
        test_config = ExtendedTestConfig(
            base_failure_rate=0.3,
            total_cycles=1000
        )
        super().__init__(test_config)
        self.regulation_config = config

    def apply_emotional_regulation(self, identity, failure_occurred: bool) -> tuple:
        """
        Apply target-based proportional emotional regulation.

        Returns: (new_identity, metadata)
        """
        metadata = {
            'regulation_type': 'target_based_proportional',
            'cycle': self.current_cycle,
            'failure': failure_occurred,
            'regulations': []
        }

        # 1. Calculate experience-driven changes (unchanged)
        if failure_occurred:
            frustration_increase = 0.05
            engagement_decrease = 0.02
            curiosity_decrease = 0.01
        else:
            frustration_increase = -0.02  # Success reduces frustration
            engagement_decrease = -0.01  # Success increases engagement
            curiosity_decrease = -0.005  # Success increases curiosity

        # 2. Calculate TARGET-BASED regulation for frustration
        reg_delta, reg_metadata = self.regulation_config.calculate_target_based_regulation(
            identity.frustration
        )

        metadata['frustration_regulation'] = reg_metadata
        metadata['regulations'].extend(reg_metadata['regulations_applied'])

        # 3. Combine experience + regulation
        total_frustration_delta = frustration_increase + reg_delta

        # 4. Apply frustration change with safety bounds
        new_frustration = identity.frustration + total_frustration_delta
        new_frustration = self.regulation_config.apply_safety_bounds(
            new_frustration,
            self.regulation_config.frustration_min,
            self.regulation_config.frustration_max,
            metadata,
            'frustration'
        )

        # 5. Apply engagement/curiosity changes (simple, no target-based yet)
        new_engagement = max(
            self.regulation_config.engagement_min,
            min(self.regulation_config.engagement_max,
                identity.engagement + engagement_decrease + self.regulation_config.engagement_recovery)
        )

        new_curiosity = max(
            self.regulation_config.curiosity_min,
            min(self.regulation_config.curiosity_max,
                identity.curiosity + curiosity_decrease + self.regulation_config.curiosity_recovery)
        )

        # 6. Update identity
        identity.frustration = new_frustration
        identity.engagement = new_engagement
        identity.curiosity = new_curiosity

        return (identity, metadata)


def run_target_based_tests():
    """
    Run Session 140 tests with target-based proportional regulation.

    Compare against Session 139 results.
    """
    print("="*70)
    print("Session 140: Target-Based Proportional Regulation")
    print("="*70)

    config = TargetBasedRegulationConfig(
        target_frustration=0.50,
        proportional_factor=0.15,
        gradient_steepness=1.0,
        base_drift_rate=0.01,
        frustration_min=0.10,
        frustration_max=0.90
    )

    print(f"\nConfiguration:")
    print(f"  Target frustration: {config.target_frustration}")
    print(f"  Proportional factor: {config.proportional_factor}")
    print(f"  Gradient steepness: {config.gradient_steepness}")
    print(f"  Bounds: [{config.frustration_min}, {config.frustration_max}]")

    # Test 1: Baseline (30% failure)
    print(f"\n{'='*70}")
    print("Test 1: Baseline (30% failure rate)")
    print(f"{'='*70}")

    regulator_baseline = TargetBasedEmotionalRegulator(config)
    results_baseline = regulator_baseline.run_baseline_test()

    print(f"\nResults:")
    print(f"  Final frustration: {results_baseline.final_emotional_state['frustration']:.3f}")
    print(f"  Mean frustration: {results_baseline.analysis['frustration_mean']:.3f}")
    print(f"  Frustration variance: {results_baseline.analysis['frustration_variance']:.4f}")
    print(f"  Frustration range: [{results_baseline.analysis['frustration_min']:.3f}, {results_baseline.analysis['frustration_max']:.3f}]")
    print(f"  Stable operation: {results_baseline.stable_operation}")

    # Test 2: Stress (60% failure)
    print(f"\n{'='*70}")
    print("Test 2: Stress (60% failure rate)")
    print(f"{'='*70}")

    regulator_stress = TargetBasedEmotionalRegulator(config)
    results_stress = regulator_stress.run_stress_test()

    print(f"\nResults:")
    print(f"  Final frustration: {results_stress.final_emotional_state['frustration']:.3f}")
    print(f"  Mean frustration: {results_stress.analysis['frustration_mean']:.3f}")
    print(f"  Frustration variance: {results_stress.analysis['frustration_variance']:.4f}")
    print(f"  Frustration range: [{results_stress.analysis['frustration_min']:.3f}, {results_stress.analysis['frustration_max']:.3f}]")
    print(f"  Stable operation: {results_stress.stable_operation}")

    # Comparison
    print(f"\n{'='*70}")
    print("Comparison: Session 139 vs Session 140")
    print(f"{'='*70}")

    delta_frustration = results_stress.analysis['frustration_mean'] - results_baseline.analysis['frustration_mean']
    variance_ratio = results_baseline.analysis['frustration_variance'] / 0.001  # S139 variance

    print(f"\nSession 139 (no target):")
    print(f"  Both tests locked at frustration=0.85")
    print(f"  Variance ≈ 0.001 (attractor)")
    print(f"  No differentiation between conditions")

    print(f"\nSession 140 (with target=0.50):")
    print(f"  Baseline (30%): frustration ≈ {results_baseline.analysis['frustration_mean']:.3f}")
    print(f"  Stress (60%): frustration ≈ {results_stress.analysis['frustration_mean']:.3f}")
    print(f"  Delta: {delta_frustration:.3f} (target: >0.2)")
    print(f"  Variance increase: {variance_ratio:.1f}x")

    success = delta_frustration > 0.2 and variance_ratio > 10
    print(f"\n{'✓' if success else '✗'} Success: {success}")

    # Save results
    results_dict = {
        'baseline': {
            'test_name': 'baseline_target_based',
            'regulation_type': 'target_based_proportional',
            'total_cycles': results_baseline.total_cycles,
            'final_emotional_state': results_baseline.final_emotional_state,
            'stable_operation': results_baseline.stable_operation,
            'cascade_detected': results_baseline.cascade_detected,
            'analysis': results_baseline.analysis
        },
        'stress': {
            'test_name': 'stress_target_based',
            'regulation_type': 'target_based_proportional',
            'total_cycles': results_stress.total_cycles,
            'final_emotional_state': results_stress.final_emotional_state,
            'stable_operation': results_stress.stable_operation,
            'cascade_detected': results_stress.cascade_detected,
            'analysis': results_stress.analysis
        },
        'comparison': {
            'session139_variance': 0.001,
            'session140_baseline_variance': results_baseline.analysis['frustration_variance'],
            'session140_stress_variance': results_stress.analysis['frustration_variance'],
            'variance_improvement': variance_ratio,
            'frustration_differentiation': delta_frustration,
            'target_based_success': success
        },
        'config': {
            'target_frustration': config.target_frustration,
            'proportional_factor': config.proportional_factor,
            'gradient_steepness': config.gradient_steepness,
            'bounds': [config.frustration_min, config.frustration_max]
        }
    }

    output_file = Path(__file__).parent / 'session140_target_based_results.json'
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results_dict


if __name__ == "__main__":
    results = run_target_based_tests()
