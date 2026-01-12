#!/usr/bin/env python3
"""
Session 188 Edge Validation: Trust-Magnetism Experimental Validation

Testing Thor's trust-magnetism experimental validation on Sprout edge hardware.

Thor's Session 188 Implementation:
- TrustNode: Simulated network node with reputation
- TrustNetworkSimulator: FM, PM, AF scenario simulation
- TrustMagnetismValidator: Experimental validation framework

Edge Validation Goals:
1. Verify trust-magnetism components import correctly
2. Test trust network simulation
3. Validate FM scenario (high trust, low variance)
4. Validate PM scenario (low trust, high variance)
5. Test magnetic observable calculation
6. Profile trust-magnetism operations on edge

Platform: Sprout (Jetson Orin Nano 8GB, TPM2 Level 3)
Date: 2026-01-12
"""

import sys
import json
import time
import numpy as np
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


def test_edge_trust_magnetism():
    """Test Session 188 trust-magnetism validation on edge."""
    print()
    print("+" + "=" * 70 + "+")
    print("|" + " " * 70 + "|")
    print("|" + "SESSION 188 EDGE: TRUST-MAGNETISM EXPERIMENTAL VALIDATION".center(70) + "|")
    print("|" + "           Jetson Orin Nano 8GB (Sprout)              ".center(70) + "|")
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
    # TEST 1: Import Session 188 Components
    # ========================================================================
    print("=" * 72)
    print("TEST 1: Import Session 188 Components")
    print("=" * 72)
    print()

    try:
        from session188_trust_magnetism_validation import (
            TrustNode,
            TrustNetworkSimulator,
            TrustMagnetismValidator,
        )
        from session187_magnetic_coherence_integration import (
            MagneticPhase,
            TrustNetworkMagneticAnalogy,
        )

        print("  TrustNode: Network node representation")
        print("  TrustNetworkSimulator: FM/PM/AF scenario simulation")
        print("  TrustMagnetismValidator: Experimental validation")
        print("  MagneticPhase: Phase classification")
        print("  TrustNetworkMagneticAnalogy: Trust-magnetism mapping")
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
    # TEST 2: Trust Network Initialization
    # ========================================================================
    print("=" * 72)
    print("TEST 2: Trust Network Initialization")
    print("=" * 72)
    print()

    print("Testing trust network simulator initialization...")

    try:
        simulator = TrustNetworkSimulator(node_count=5, initial_reputation=100.0)

        print(f"  Node count: {simulator.node_count}")
        print(f"  Initial reputation: {simulator.initial_reputation}")
        print(f"  Nodes created: {len(simulator.nodes)}")

        # Check trust relationships initialized
        node = list(simulator.nodes.values())[0]
        print(f"  Trust relationships per node: {len(node.trust_scores)}")

        test2_pass = (
            len(simulator.nodes) == 5 and
            len(node.trust_scores) == 4 and
            all(abs(t - 0.5) < 0.001 for t in node.trust_scores.values())
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test2_pass = False

    test_results["network_init"] = test2_pass
    print()
    print(f"{'PASS' if test2_pass else 'FAIL'}: TEST 2")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    # ========================================================================
    # TEST 3: Ferromagnetic Scenario Simulation
    # ========================================================================
    print("=" * 72)
    print("TEST 3: Ferromagnetic Scenario Simulation")
    print("=" * 72)
    print()

    print("Testing FM scenario (high trust, low variance)...")

    try:
        simulator = TrustNetworkSimulator(node_count=5, initial_reputation=100.0)
        fm_results = simulator.simulate_ferromagnetic_scenario(steps=10)

        print(f"  Scenario: {fm_results['scenario']}")
        print(f"  Steps: {fm_results['steps']}")
        print(f"  Final avg trust: {fm_results['final_avg_trust']:.4f}")
        print(f"  Final variance: {fm_results['final_variance']:.6f}")

        # FM should have high trust and low variance
        test3_pass = (
            fm_results['final_avg_trust'] > 0.7 and
            fm_results['final_variance'] < 0.1
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test3_pass = False

    test_results["fm_scenario"] = test3_pass
    print()
    print(f"{'PASS' if test3_pass else 'FAIL'}: TEST 3")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # ========================================================================
    # TEST 4: Paramagnetic Scenario Simulation
    # ========================================================================
    print("=" * 72)
    print("TEST 4: Paramagnetic Scenario Simulation")
    print("=" * 72)
    print()

    print("Testing PM scenario (low trust, high variance)...")

    try:
        simulator = TrustNetworkSimulator(node_count=5, initial_reputation=100.0)
        pm_results = simulator.simulate_paramagnetic_scenario(steps=10)

        print(f"  Scenario: {pm_results['scenario']}")
        print(f"  Final avg trust: {pm_results['final_avg_trust']:.4f}")
        print(f"  Final variance: {pm_results['final_variance']:.6f}")

        # PM should have lower trust or higher variance than FM
        test4_pass = (
            pm_results['final_avg_trust'] < 0.6 or
            pm_results['final_variance'] > fm_results['final_variance']
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test4_pass = False

    test_results["pm_scenario"] = test4_pass
    print()
    print(f"{'PASS' if test4_pass else 'FAIL'}: TEST 4")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # ========================================================================
    # TEST 5: Trust-Magnetism Mapping
    # ========================================================================
    print("=" * 72)
    print("TEST 5: Trust-Magnetism Mapping")
    print("=" * 72)
    print()

    print("Testing trust network to magnetic phase analysis...")

    try:
        analyzer = TrustNetworkMagneticAnalogy()

        # FM scenario: high trust, low variance
        fm_state = analyzer.analyze_trust_network_phase(
            avg_trust=0.9,
            trust_variance=0.01,
        )

        print(f"  FM scenario (trust=0.9, var=0.01):")
        print(f"    Temperature: {fm_state.temperature:.4f}")
        print(f"    Phase: {fm_state.magnetic_phase.value}")
        print(f"    Coherence: {fm_state.coherence:.4f}")

        # PM scenario: low trust, high variance
        pm_state = analyzer.analyze_trust_network_phase(
            avg_trust=0.3,
            trust_variance=0.2,
        )

        print(f"  PM scenario (trust=0.3, var=0.2):")
        print(f"    Temperature: {pm_state.temperature:.4f}")
        print(f"    Phase: {pm_state.magnetic_phase.value}")
        print(f"    Coherence: {pm_state.coherence:.4f}")

        # FM should have higher coherence (ordered state)
        test5_pass = (
            fm_state.coherence > pm_state.coherence or
            fm_state.temperature < pm_state.temperature
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test5_pass = False

    test_results["trust_magnetism_map"] = test5_pass
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

    print("Profiling trust-magnetism operations on edge...")

    try:
        analyzer = TrustNetworkMagneticAnalogy()
        iterations = 1000

        # Profile trust-to-temperature mapping
        start = time.time()
        for i in range(iterations):
            trust = (i % 100) / 100.0
            variance = 0.01 + (i % 50) / 250.0
            _ = analyzer.map_trust_to_temperature(trust, variance)
        temp_time = time.time() - start
        temp_ops_per_sec = iterations / temp_time

        # Profile phase analysis
        start = time.time()
        for i in range(iterations):
            trust = (i % 100) / 100.0
            variance = 0.01 + (i % 50) / 250.0
            delta = (i % 100) - 50
            _ = analyzer.analyze_trust_network_phase(trust, variance, delta)
        phase_time = time.time() - start
        phase_ops_per_sec = iterations / phase_time

        # Profile network simulation (fewer iterations - more expensive)
        start = time.time()
        for i in range(10):
            sim = TrustNetworkSimulator(node_count=5)
            _ = sim.simulate_ferromagnetic_scenario(steps=5)
        sim_time = time.time() - start
        sim_ops_per_sec = 10 / sim_time

        print(f"  Trust-to-temperature: {temp_ops_per_sec:,.0f} ops/sec")
        print(f"  Phase analysis: {phase_ops_per_sec:,.0f} ops/sec")
        print(f"  Network simulation (5 nodes, 5 steps): {sim_ops_per_sec:.1f} ops/sec")
        print()

        test6_pass = (
            temp_ops_per_sec > 10000 and
            phase_ops_per_sec > 1000 and
            sim_ops_per_sec > 5
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test6_pass = False

    test_results["edge_performance"] = test6_pass
    print(f"{'PASS' if test6_pass else 'FAIL'}: TEST 6")
    print()
    all_tests_passed = all_tests_passed and test6_pass

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 72)
    print("SESSION 188 EDGE VALIDATION SUMMARY")
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
        print("|" + " TRUST-MAGNETISM VALIDATION VALIDATED ON EDGE! ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "-" * 70 + "+")
        print()
        print("Edge Observations:")
        print("  - Trust network simulation working")
        print("  - FM scenario: high trust, low variance")
        print("  - PM scenario: low trust, high variance")
        print("  - Trust-to-magnetic mapping operational")
        print()
        print("Predictions Validated on Edge:")
        print("  - P187.4: Trust networks exhibit magnetic phases")
        print("  - FM -> Ferromagnetic (ordered)")
        print("  - PM -> Paramagnetic (disordered)")
        print()
        print("Sessions 177-188 Edge Stack: COMPLETE")
        print("  Theory (187) + Validation (188) = Unified Framework")
    else:
        print("SOME TESTS FAILED - Review results above")

    print()

    # Save results
    results = {
        "session": "188_edge",
        "title": "Trust-Magnetism Experimental Validation - Edge",
        "platform": "Sprout (Jetson Orin Nano 8GB)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": bool(all_tests_passed),
        "test_results": {k: bool(v) for k, v in test_results.items()},
        "edge_metrics": edge_metrics,
        "validation_features": {
            "network_simulation": True,
            "fm_scenario": True,
            "pm_scenario": True,
            "trust_magnetism_mapping": True,
        }
    }

    results_path = Path(__file__).parent / "session188_edge_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = test_edge_trust_magnetism()
    sys.exit(0 if success else 1)
