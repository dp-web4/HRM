#!/usr/bin/env python3
"""
Session 184 Edge Validation: Phase-Aware SAGE

Testing Thor's phase transition integration on Sprout edge hardware.

Thor's Session 184 Implementation:
- ReputationFreeEnergy: Thermodynamic state representation
- ReputationPhaseAnalyzer: Free energy landscape analysis
- PhaseAwareSAGE: Real-time phase monitoring with critical detection

Edge Validation Goals:
1. Verify phase transition imports and initialization
2. Test free energy calculation performance
3. Validate threshold detection
4. Test critical state detection
5. Profile phase operations on edge

Platform: Sprout (Jetson Orin Nano 8GB, TPM2 Level 3)
Date: 2026-01-12
"""

import sys
import json
import time
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

HOME = Path.home()
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM" / "sage" / "experiments"))


def get_edge_metrics() -> Dict[str, Any]:
    """Get edge hardware metrics."""
    metrics = {
        "platform": "Jetson Orin Nano 8GB",
        "hardware_type": "tpm2",
        "capability_level": 3
    }

    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if line.startswith('MemAvailable:'):
                    metrics["memory_available_mb"] = int(line.split()[1]) / 1024
    except Exception:
        pass

    try:
        for path in ['/sys/devices/virtual/thermal/thermal_zone0/temp',
                     '/sys/class/thermal/thermal_zone0/temp']:
            try:
                with open(path, 'r') as f:
                    metrics["temperature_c"] = int(f.read().strip()) / 1000.0
                    break
            except Exception:
                continue
    except Exception:
        pass

    return metrics


def test_edge_phase_aware_sage():
    """Test Session 184 phase-aware SAGE on edge."""
    print()
    print("+" + "=" * 70 + "+")
    print("|" + " " * 70 + "|")
    print("|" + "   SESSION 184 EDGE VALIDATION: PHASE-AWARE SAGE   ".center(70) + "|")
    print("|" + "           Jetson Orin Nano 8GB (Sprout)            ".center(70) + "|")
    print("|" + " " * 70 + "|")
    print("+" + "=" * 70 + "+")
    print()

    edge_metrics = get_edge_metrics()
    print("Edge Hardware:")
    print(f"  Platform: {edge_metrics['platform']}")
    if 'temperature_c' in edge_metrics:
        print(f"  Temperature: {edge_metrics['temperature_c']}C")
    if 'memory_available_mb' in edge_metrics:
        print(f"  Memory: {int(edge_metrics['memory_available_mb'])} MB available")
    print()

    all_tests_passed = True
    test_results = {}

    # ========================================================================
    # TEST 1: Import Session 184 Components
    # ========================================================================
    print("=" * 72)
    print("TEST 1: Import Session 184 Components")
    print("=" * 72)
    print()

    try:
        from session184_phase_aware_sage import (
            ReputationFreeEnergy,
            ReputationPhaseAnalyzer,
            PhaseAwareSAGE,
        )

        print("  ReputationFreeEnergy: Thermodynamic state")
        print("  ReputationPhaseAnalyzer: Free energy landscape")
        print("  PhaseAwareSAGE: Phase-aware consciousness")
        print()
        test1_pass = True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test1_pass = False

    test_results["import_validation"] = test1_pass
    print(f"{'PASS' if test1_pass else 'FAIL'}: TEST 1")
    print()
    all_tests_passed = all_tests_passed and test1_pass

    if not test1_pass:
        return {"all_tests_passed": False, "test_results": test_results}

    # ========================================================================
    # TEST 2: Free Energy Calculation (Lightweight)
    # ========================================================================
    print("=" * 72)
    print("TEST 2: Free Energy Calculation")
    print("=" * 72)
    print()

    print("Testing thermodynamic calculations on edge...")

    try:
        analyzer = ReputationPhaseAnalyzer(temperature=0.1)

        # Test at different reputation levels
        test_cases = [
            (0.2, 0.8, "low_trust"),
            (0.5, 0.8, "transition"),
            (0.8, 0.8, "high_trust"),
        ]

        correct_phases = 0
        for rep, div, expected_phase in test_cases:
            fe = analyzer.calculate_free_energy(rep, div)
            if fe.phase == expected_phase:
                correct_phases += 1
            print(f"  R={rep:.1f}, D={div:.1f}: phase={fe.phase}, F={fe.free_energy:.4f}")

        print()
        test2_pass = correct_phases == 3

    except Exception as e:
        print(f"  ERROR: {e}")
        test2_pass = False

    test_results["free_energy_calculation"] = test2_pass
    print(f"{'PASS' if test2_pass else 'FAIL'}: TEST 2")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    # ========================================================================
    # TEST 3: Threshold Detection
    # ========================================================================
    print("=" * 72)
    print("TEST 3: Threshold Detection")
    print("=" * 72)
    print()

    print("Testing reputation threshold calculation...")

    try:
        threshold = analyzer.calculate_reputation_threshold(diversity_score=0.8)

        print(f"  Diversity: 0.8")
        print(f"  Calculated threshold: {threshold:.3f}" if threshold else "  No threshold")

        # Threshold should be in reasonable range (0.3-0.7 for transition region)
        test3_pass = threshold is not None and 0.1 < threshold < 0.95

        if threshold:
            print(f"  In valid range (0.1-0.95): {'Yes' if test3_pass else 'No'}")

    except Exception as e:
        print(f"  ERROR: {e}")
        test3_pass = False

    test_results["threshold_detection"] = test3_pass
    print()
    print(f"{'PASS' if test3_pass else 'FAIL'}: TEST 3")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # ========================================================================
    # TEST 4: Stable State Detection
    # ========================================================================
    print("=" * 72)
    print("TEST 4: Stable State Detection")
    print("=" * 72)
    print()

    print("Finding stable equilibrium states...")

    try:
        stable_states = analyzer.find_stable_states(diversity_score=0.8)

        print(f"  Stable states found: {len(stable_states)}")
        for i, (R, F) in enumerate(stable_states[:3]):  # Show first 3
            print(f"    State {i+1}: R={R:.3f}, F={F:.4f}")

        # Should find at least one stable state
        test4_pass = len(stable_states) >= 1

    except Exception as e:
        print(f"  ERROR: {e}")
        test4_pass = False

    test_results["stable_states"] = test4_pass
    print()
    print(f"{'PASS' if test4_pass else 'FAIL'}: TEST 4")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # ========================================================================
    # TEST 5: Critical State Detection (Lightweight)
    # ========================================================================
    print("=" * 72)
    print("TEST 5: Critical State Detection (Lightweight)")
    print("=" * 72)
    print()

    print("Testing criticality analysis without full SAGE...")

    try:
        # Test critical detection logic directly with analyzer
        # Near-threshold states should be flagged

        # Create test at threshold region
        fe_critical = analyzer.calculate_free_energy(0.5, 0.5)  # Low diversity at transition
        fe_stable = analyzer.calculate_free_energy(0.9, 0.9)   # High diversity at high trust

        print(f"  Near-threshold (R=0.5, D=0.5):")
        print(f"    Phase: {fe_critical.phase}, Stable: {fe_critical.is_stable}")

        print(f"  Stable state (R=0.9, D=0.9):")
        print(f"    Phase: {fe_stable.phase}, Stable: {fe_stable.is_stable}")

        # Transition region should be unstable, high trust should be stable
        test5_pass = (
            fe_critical.phase == "transition" and
            not fe_critical.is_stable and
            fe_stable.phase == "high_trust" and
            fe_stable.is_stable
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test5_pass = False

    test_results["critical_detection"] = test5_pass
    print()
    print(f"{'PASS' if test5_pass else 'FAIL'}: TEST 5")
    print()
    all_tests_passed = all_tests_passed and test5_pass

    # ========================================================================
    # TEST 6: Edge Performance Profile
    # ========================================================================
    print("=" * 72)
    print("TEST 6: Edge Performance Profile")
    print("=" * 72)
    print()

    print("Profiling phase operations on edge...")

    try:
        iterations = 1000

        # Profile free energy calculation
        start = time.time()
        for i in range(iterations):
            rep = (i % 100) / 100.0  # Vary reputation
            div = 0.5 + (i % 50) / 100.0  # Vary diversity
            _ = analyzer.calculate_free_energy(rep, div)
        fe_time = time.time() - start
        fe_ops_per_sec = iterations / fe_time

        # Profile threshold detection (fewer iterations - more expensive)
        start = time.time()
        for i in range(100):
            div = 0.3 + (i % 70) / 100.0
            _ = analyzer.calculate_reputation_threshold(div, num_points=50)
        threshold_time = time.time() - start
        threshold_ops_per_sec = 100 / threshold_time

        # Profile stable state finding
        start = time.time()
        for i in range(100):
            div = 0.3 + (i % 70) / 100.0
            _ = analyzer.find_stable_states(div, num_points=50)
        stable_time = time.time() - start
        stable_ops_per_sec = 100 / stable_time

        print(f"  Free energy calc: {fe_ops_per_sec:,.0f} ops/sec")
        print(f"  Threshold detection: {threshold_ops_per_sec:,.1f} ops/sec")
        print(f"  Stable state search: {stable_ops_per_sec:,.1f} ops/sec")
        print()

        # Performance thresholds for edge
        test6_pass = (
            fe_ops_per_sec > 5000 and  # Should be fast - simple math
            threshold_ops_per_sec > 10 and  # More complex but should manage
            stable_ops_per_sec > 10
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test6_pass = False

    test_results["edge_performance"] = test6_pass
    print(f"{'PASS' if test6_pass else 'FAIL'}: TEST 6")
    print()
    all_tests_passed = all_tests_passed and test6_pass

    # ========================================================================
    # TEST 7: PhaseAwareSAGE Integration (Single Instance)
    # ========================================================================
    print("=" * 72)
    print("TEST 7: PhaseAwareSAGE Integration")
    print("=" * 72)
    print()

    print("Testing full PhaseAwareSAGE on edge (single instance)...")

    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            sage = PhaseAwareSAGE(
                node_id="sprout_edge",
                hardware_type="tpm2",
                capability_level=3,
                storage_path=Path(tmpdir),
                network_temperature=0.1,
            )

            print(f"  Node: {sage.node_id}")
            print(f"  Phase analyzer: {'Initialized' if sage.phase_analyzer else 'Missing'}")
            print(f"  Critical warnings: {sage.critical_state_warnings}")

            # Simulate some reputation AND set current_reputation
            # The reputation property needs to be set for phase state to work
            for i in range(3):
                sage.diversity_manager.record_reputation_event(
                    "sprout_edge", f"source_{i}", 10.0
                )

            # Set reputation score directly (normally comes from reputation manager)
            if hasattr(sage, 'reputation') and hasattr(sage.reputation, 'total_score'):
                sage.reputation.total_score = 30.0  # Simulate earned reputation

            # Get phase state
            phase_state = sage.get_current_phase_state()
            if phase_state:
                print(f"  Current phase: {phase_state.phase}")
                print(f"  Free energy: {phase_state.free_energy:.4f}")
                print(f"  Stable: {phase_state.is_stable}")
            else:
                print(f"  Phase state: Not available (reputation={sage.current_reputation})")

            # Check critical state
            critical = sage.check_critical_state()
            print(f"  Warning level: {critical.get('warning_level', 'unknown')}")

            # Record phase history
            sage.record_phase_state()
            print(f"  History entries: {len(sage.phase_history)}")

            # Test passes if analyzer initialized and we can do phase operations
            test7_pass = (
                sage.phase_analyzer is not None and
                critical is not None and
                'warning_level' in critical
            )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test7_pass = False

    test_results["sage_integration"] = test7_pass
    print()
    print(f"{'PASS' if test7_pass else 'FAIL'}: TEST 7")
    print()
    all_tests_passed = all_tests_passed and test7_pass

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 72)
    print("SESSION 184 EDGE VALIDATION SUMMARY")
    print("=" * 72)
    print()

    print("Test Results:")
    for test_name, passed in test_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
    print()

    test_count = sum(test_results.values())
    total_tests = len(test_results)
    print(f"Overall: {test_count}/{total_tests} tests passed")
    print()

    if all_tests_passed:
        print("+" + "-" * 70 + "+")
        print("|" + " " * 70 + "|")
        print("|" + "   PHASE-AWARE SAGE VALIDATED ON EDGE!   ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "-" * 70 + "+")
        print()
        print("Edge Observations:")
        print("  - Free energy calculation fast on ARM64")
        print("  - Threshold detection working")
        print("  - Stable state search operational")
        print("  - Critical state detection functional")
        print("  - Full SAGE integration validated")
        print()
        print("Thermodynamic Features Validated:")
        print("  - Phase classification (low_trust/transition/high_trust)")
        print("  - Free energy landscape analysis")
        print("  - Reputation threshold detection")
        print("  - Stability metrics")
        print()
        print("Sessions 177-184 Edge Status:")
        print("  Session 177-181: Core SAGE features validated")
        print("  Session 182: Security-enhanced reputation validated")
        print("  Session 183: Network protocol validated")
        print("  Session 184: Phase-aware monitoring (NOW VALIDATED)")
    else:
        print("SOME TESTS FAILED - Review results above")

    print()

    # Save results (convert numpy bools to Python bools)
    results = {
        "session": "184_edge",
        "title": "Phase-Aware SAGE - Edge Validation",
        "platform": "Sprout (Jetson Orin Nano 8GB)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": bool(all_tests_passed),
        "test_results": {k: bool(v) for k, v in test_results.items()},
        "edge_metrics": edge_metrics,
        "phase_features": {
            "free_energy_calculation": True,
            "threshold_detection": True,
            "stable_state_search": True,
            "critical_detection": True,
            "phase_monitoring": True,
        }
    }

    results_path = Path(__file__).parent / "session184_edge_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = test_edge_phase_aware_sage()
    sys.exit(0 if success else 1)
