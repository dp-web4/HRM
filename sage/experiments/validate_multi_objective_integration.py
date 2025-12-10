#!/usr/bin/env python3
"""
Session 24: Multi-Objective Integration Validation

Validates that Session 24 multi-objective optimization integration works correctly
in the production temporal_adaptation.py module.

Tests:
1. Backward compatibility (existing code still works without quality scores)
2. Multi-objective tracking (quality and energy metrics computed correctly)
3. Factory function (create_multi_objective_adapter() works as expected)
4. Configuration options (weights and enable_multi_objective flag)

Hardware: Jetson AGX Thor
Session: 24 (autonomous)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import statistics
from typing import List

# Production temporal adaptation with Session 24 integration
from core.temporal_adaptation import (
    create_production_adapter,
    create_multi_objective_adapter,
    TemporalAdapter
)

# Experimental validation infrastructure
from experiments.validate_atp_on_real_consciousness import (
    ATPTunedConsciousness,
    SensorObservation
)


def simulate_quality_score(salience: float, attended: bool, atp_level: float) -> float:
    """
    Simulate response quality based on attention and ATP state.

    Quality depends on:
    - Whether observation was attended
    - ATP level (higher ATP = better quality)
    - Salience (higher salience observations expect better quality)
    """
    if not attended:
        return 0.0

    base_quality = 0.5 + (atp_level * 0.5)
    salience_factor = min(1.0, salience / 0.7)
    noise = random.uniform(-0.1, 0.1)

    quality = base_quality * salience_factor + noise
    return max(0.0, min(1.0, quality))


def test_backward_compatibility():
    """
    Test 1: Verify backward compatibility.

    Existing code that doesn't provide quality scores should continue working.
    """
    print("\n" + "="*70)
    print("TEST 1: Backward Compatibility")
    print("="*70)

    adapter = create_production_adapter()
    consciousness = ATPTunedConsciousness(
        identity_name="backward_compat_test",
        attention_cost=adapter.current_cost,
        rest_recovery=adapter.current_recovery
    )

    print(f"\nTesting production adapter WITHOUT quality scores...")
    print(f"Initial parameters: cost={adapter.current_cost:.4f}, recovery={adapter.current_recovery:.4f}")

    # Run cycles without quality scores (old API)
    for i in range(1000):
        salience = random.betavariate(2, 3)
        obs = SensorObservation(
            sensor_name=f"sensor_{i % 10}",
            salience=salience,
            data=str({"value": random.random()})
        )

        threshold = consciousness.attention_cost * (1.0 - consciousness.atp_level)
        attended = salience > threshold

        # OLD API: No quality_score parameter
        result = adapter.update(
            attended=attended,
            salience=salience,
            atp_level=consciousness.atp_level,
            high_salience_count=100 if i % 100 == 0 else 0,
            attended_high_salience=50 if i % 100 == 0 else 0
        )

        consciousness.process_cycle([obs])

    metrics = adapter.current_window.get_metrics()

    print(f"\n✅ Backward compatibility verified!")
    print(f"   Cycles processed: 1,000")
    print(f"   Coverage: {metrics.get('coverage', 0):.1%}")
    print(f"   Quality: {metrics.get('quality', 0):.1%} (should be 0% without scores)")
    print(f"   Energy: {metrics.get('energy_efficiency', 0):.1%}")
    print(f"   Weighted fitness: {metrics.get('weighted_fitness', 0):.3f}")

    assert metrics.get('quality', 0) == 0.0, "Quality should be 0 without quality scores"
    assert 'weighted_fitness' in metrics, "Weighted fitness should still be computed"

    return adapter


def test_multi_objective_tracking():
    """
    Test 2: Verify multi-objective tracking works correctly.

    When quality scores are provided, they should be tracked and included in metrics.
    """
    print("\n" + "="*70)
    print("TEST 2: Multi-Objective Tracking")
    print("="*70)

    adapter = create_multi_objective_adapter()
    consciousness = ATPTunedConsciousness(
        identity_name="multi_objective_test",
        attention_cost=adapter.current_cost,
        rest_recovery=adapter.current_recovery
    )

    print(f"\nTesting multi-objective adapter WITH quality scores...")
    print(f"Initial parameters: cost={adapter.current_cost:.4f}, recovery={adapter.current_recovery:.4f}")
    print(f"Multi-objective enabled: {adapter.enable_multi_objective}")
    print(f"Weights: coverage={adapter.coverage_weight:.1%}, quality={adapter.quality_weight:.1%}, energy={adapter.energy_weight:.1%}")

    quality_scores_provided = []

    # Run cycles with quality scores (new API)
    for i in range(1000):
        salience = random.betavariate(2, 3)
        obs = SensorObservation(
            sensor_name=f"sensor_{i % 10}",
            salience=salience,
            data=str({"value": random.random()})
        )

        threshold = consciousness.attention_cost * (1.0 - consciousness.atp_level)
        attended = salience > threshold

        # Simulate quality score
        quality = simulate_quality_score(salience, attended, consciousness.atp_level)
        if attended:
            quality_scores_provided.append(quality)

        # NEW API: With quality_score parameter
        result = adapter.update(
            attended=attended,
            salience=salience,
            atp_level=consciousness.atp_level,
            high_salience_count=100 if i % 100 == 0 else 0,
            attended_high_salience=50 if i % 100 == 0 else 0,
            quality_score=quality,
            attention_cost=adapter.current_cost
        )

        consciousness.process_cycle([obs])

    metrics = adapter.current_window.get_metrics()

    print(f"\n✅ Multi-objective tracking verified!")
    print(f"   Cycles processed: 1,000")
    print(f"   Quality scores provided: {len(quality_scores_provided)}")
    print(f"   Coverage: {metrics.get('coverage', 0):.1%}")
    print(f"   Quality: {metrics.get('quality', 0):.1%} (should be >0% with scores)")
    print(f"   Energy Efficiency: {metrics.get('energy_efficiency', 0):.1%}")
    print(f"   Weighted Fitness: {metrics.get('weighted_fitness', 0):.3f}")

    # Verify quality was tracked
    assert metrics.get('quality', 0) > 0.0, "Quality should be >0 with quality scores"
    assert len(quality_scores_provided) > 0, "Should have provided quality scores"

    # Verify energy was tracked
    assert 'energy_efficiency' in metrics, "Energy efficiency should be tracked"

    # Compare tracked quality to actual provided scores
    expected_avg_quality = statistics.mean(quality_scores_provided)
    tracked_quality = metrics.get('quality', 0)
    quality_diff = abs(expected_avg_quality - tracked_quality)

    print(f"\n   Quality validation:")
    print(f"   - Expected avg: {expected_avg_quality:.1%}")
    print(f"   - Tracked avg: {tracked_quality:.1%}")
    print(f"   - Difference: {quality_diff:.1%}")

    assert quality_diff < 0.05, f"Quality tracking error too large: {quality_diff:.1%}"

    return adapter


def test_factory_function():
    """
    Test 3: Verify factory function creates correct configuration.
    """
    print("\n" + "="*70)
    print("TEST 3: Factory Function Configuration")
    print("="*70)

    # Test multi-objective factory
    adapter = create_multi_objective_adapter()

    print(f"\nMulti-objective adapter configuration:")
    print(f"   Initial cost: {adapter.current_cost:.4f} (expected 0.005)")
    print(f"   Initial recovery: {adapter.current_recovery:.4f} (expected 0.080)")
    print(f"   Multi-objective enabled: {adapter.enable_multi_objective}")
    print(f"   Pattern learning enabled: {adapter.enable_pattern_learning}")
    print(f"   Coverage weight: {adapter.coverage_weight:.1%}")
    print(f"   Quality weight: {adapter.quality_weight:.1%}")
    print(f"   Energy weight: {adapter.energy_weight:.1%}")

    # Verify Pareto-optimal parameters from Session 23
    assert adapter.current_cost == 0.005, "Should use Pareto-optimal cost"
    assert adapter.current_recovery == 0.080, "Should use Pareto-optimal recovery"
    assert adapter.enable_multi_objective == True, "Should enable multi-objective"
    assert adapter.coverage_weight == 0.5, "Default coverage weight should be 0.5"
    assert adapter.quality_weight == 0.3, "Default quality weight should be 0.3"
    assert adapter.energy_weight == 0.2, "Default energy weight should be 0.2"

    # Test custom weights
    custom = create_multi_objective_adapter(
        coverage_weight=0.7,
        quality_weight=0.2,
        energy_weight=0.1
    )

    print(f"\n   Custom weights:")
    print(f"   Coverage: {custom.coverage_weight:.1%}")
    print(f"   Quality: {custom.quality_weight:.1%}")
    print(f"   Energy: {custom.energy_weight:.1%}")

    assert custom.coverage_weight == 0.7, "Should use custom coverage weight"
    assert custom.quality_weight == 0.2, "Should use custom quality weight"
    assert custom.energy_weight == 0.1, "Should use custom energy weight"

    print(f"\n✅ Factory function configuration verified!")


def test_weighted_fitness_computation():
    """
    Test 4: Verify weighted fitness is computed correctly.
    """
    print("\n" + "="*70)
    print("TEST 4: Weighted Fitness Computation")
    print("="*70)

    adapter = create_multi_objective_adapter()

    # Manually set metrics in window
    adapter.current_window.coverage_scores.append(0.95)
    adapter.current_window.quality_scores.append(0.60)
    # Energy will be computed from atp_spent

    # Add some ATP spending
    for i in range(100):
        adapter.current_window.atp_spent.append(0.005)
        adapter.current_window.cycle_count += 1

    metrics = adapter.current_window.get_metrics()

    coverage = metrics.get('coverage', 0)
    quality = metrics.get('quality', 0)
    energy = metrics.get('energy_efficiency', 0)
    weighted = metrics.get('weighted_fitness', 0)

    # Manually compute expected weighted fitness
    expected_weighted = (
        adapter.coverage_weight * coverage +
        adapter.quality_weight * quality +
        adapter.energy_weight * energy
    )

    print(f"\nWeighted fitness computation:")
    print(f"   Coverage: {coverage:.1%} × {adapter.coverage_weight:.1%} = {adapter.coverage_weight * coverage:.3f}")
    print(f"   Quality: {quality:.1%} × {adapter.quality_weight:.1%} = {adapter.quality_weight * quality:.3f}")
    print(f"   Energy: {energy:.1%} × {adapter.energy_weight:.1%} = {adapter.energy_weight * energy:.3f}")
    print(f"   Weighted sum: {weighted:.3f}")
    print(f"   Expected: {expected_weighted:.3f}")

    diff = abs(weighted - expected_weighted)
    assert diff < 0.001, f"Weighted fitness computation error: {diff}"

    print(f"\n✅ Weighted fitness computation verified!")


def main():
    """Run Session 24 integration validation"""
    print("=" * 80)
    print(" " * 15 + "Session 24: Multi-Objective Integration Validation")
    print("=" * 80)
    print("\nValidating multi-objective optimization integration into temporal_adaptation.py")
    print()

    # Run tests
    test_backward_compatibility()
    test_multi_objective_tracking()
    test_factory_function()
    test_weighted_fitness_computation()

    # Final summary
    print("\n" + "="*80)
    print("Session 24 Integration Validation Complete")
    print("="*80)

    print("\nAll Tests Passed:")
    print("  ✅ Backward compatibility maintained")
    print("  ✅ Multi-objective tracking functional")
    print("  ✅ Factory function creates correct configuration")
    print("  ✅ Weighted fitness computed correctly")

    print("\nIntegration Summary:")
    print("  - Added 161 LOC to temporal_adaptation.py")
    print("  - Quality and energy tracking in TemporalWindow")
    print("  - Multi-objective fitness calculation")
    print("  - New factory: create_multi_objective_adapter()")
    print("  - Fully backward compatible (optional quality scores)")

    print("\nNext Steps:")
    print("  - Test with real SAGE conversation workloads")
    print("  - Compare single-objective vs multi-objective performance")
    print("  - Deploy to production and monitor quality improvements")


if __name__ == "__main__":
    main()
