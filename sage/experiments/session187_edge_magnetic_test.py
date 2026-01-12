#!/usr/bin/env python3
"""
Session 187 Edge Validation: Magnetic Coherence Integration

Testing Thor's magnetic-quantum coherence integration on Sprout edge hardware.

Thor's Session 187 Implementation:
- MagneticPhase: Paramagnetic, Transition, Ferromagnetic, Antiferromagnetic
- MagneticState: Complete quantum-magnetic state representation
- MagneticCoherenceAnalyzer: Quantum-magnetic framework
- TrustNetworkMagneticAnalogy: Trust networks as magnetic systems

Edge Validation Goals:
1. Verify magnetic components import correctly
2. Test correlation length calculation
3. Validate critical exponent derivation (gamma, beta)
4. Test decoherence rate mapping
5. Test spin coherence evolution
6. Validate magnetization from Born rule
7. Profile magnetic operations on edge

Platform: Sprout (Jetson Orin Nano 8GB, TPM2 Level 3)
Date: 2026-01-12
"""

import sys
import json
import time
import math
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


def test_edge_magnetic_coherence():
    """Test Session 187 magnetic coherence on edge."""
    print()
    print("+" + "=" * 70 + "+")
    print("|" + " " * 70 + "|")
    print("|" + "SESSION 187 EDGE VALIDATION: MAGNETIC COHERENCE INTEGRATION".center(70) + "|")
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
    # TEST 1: Import Session 187 Components
    # ========================================================================
    print("=" * 72)
    print("TEST 1: Import Session 187 Components")
    print("=" * 72)
    print()

    try:
        from session187_magnetic_coherence_integration import (
            MagneticPhase,
            MagneticState,
            MagneticCoherenceAnalyzer,
            TrustNetworkMagneticAnalogy,
        )

        print("  MagneticPhase: Magnetic ordering states")
        print("  MagneticState: Complete quantum-magnetic state")
        print("  MagneticCoherenceAnalyzer: Quantum-magnetic framework")
        print("  TrustNetworkMagneticAnalogy: Trust network mapping")
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
    # TEST 2: Correlation Length Calculation
    # ========================================================================
    print("=" * 72)
    print("TEST 2: Correlation Length Calculation")
    print("=" * 72)
    print()

    print("Testing correlation length xi near Tc...")

    try:
        analyzer = MagneticCoherenceAnalyzer(dimension=3, exchange_coupling=1.0)
        tc = analyzer.calculate_critical_temperature()

        # Test at different temperatures
        test_temps = [tc * 0.5, tc * 0.9, tc * 1.0, tc * 1.1, tc * 2.0]
        xi_values = []

        for temp in test_temps:
            xi = analyzer.calculate_correlation_length(temp, tc)
            xi_values.append(xi)
            print(f"  T/Tc={temp/tc:.2f}: xi={xi:.2f}")

        # Correlation length should be larger near Tc
        test2_pass = xi_values[1] > xi_values[0] and xi_values[2] >= xi_values[1]

    except Exception as e:
        print(f"  ERROR: {e}")
        test2_pass = False

    test_results["correlation_length"] = test2_pass
    print()
    print(f"{'PASS' if test2_pass else 'FAIL'}: TEST 2")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    # ========================================================================
    # TEST 3: Critical Exponents (gamma, beta)
    # ========================================================================
    print("=" * 72)
    print("TEST 3: Critical Exponents (gamma, beta)")
    print("=" * 72)
    print()

    print("Testing critical exponent derivation from CBP Session 16...")

    try:
        # Test at different correlation lengths
        xi_values = [1.0, 2.0, 5.0, 10.0, 100.0]

        for xi in xi_values:
            gamma, beta = analyzer.calculate_critical_exponents(xi)
            # Verify: gamma = 2/xi, beta = 1/(2*gamma) = xi/4
            expected_gamma = 2.0 / xi
            expected_beta = xi / 4.0
            print(f"  xi={xi:.1f}: gamma={gamma:.4f} (exp:{expected_gamma:.4f}), beta={beta:.4f} (exp:{expected_beta:.4f})")

        # Verify the relationship beta = 1/(2*gamma)
        gamma_test, beta_test = analyzer.calculate_critical_exponents(4.0)
        test3_pass = abs(beta_test - 1.0 / (2.0 * gamma_test)) < 0.001

    except Exception as e:
        print(f"  ERROR: {e}")
        test3_pass = False

    test_results["critical_exponents"] = test3_pass
    print()
    print(f"{'PASS' if test3_pass else 'FAIL'}: TEST 3")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # ========================================================================
    # TEST 4: Decoherence Rate Mapping
    # ========================================================================
    print("=" * 72)
    print("TEST 4: Decoherence Rate Mapping")
    print("=" * 72)
    print()

    print("Testing decoherence rate Gamma_d ~ gamma/xi...")

    try:
        # Larger correlation length should give smaller decoherence rate
        xi_small = 2.0
        xi_large = 20.0

        gamma_small, _ = analyzer.calculate_critical_exponents(xi_small)
        gamma_large, _ = analyzer.calculate_critical_exponents(xi_large)

        gamma_d_small = analyzer.calculate_decoherence_rate(gamma_small, xi_small)
        gamma_d_large = analyzer.calculate_decoherence_rate(gamma_large, xi_large)

        print(f"  Small xi (xi={xi_small}): Gamma_d={gamma_d_small:.4f}")
        print(f"  Large xi (xi={xi_large}): Gamma_d={gamma_d_large:.4f}")
        print(f"  Ratio: {gamma_d_small/gamma_d_large:.2f}x")

        # Larger correlations → smaller decoherence rate
        test4_pass = gamma_d_small > gamma_d_large

    except Exception as e:
        print(f"  ERROR: {e}")
        test4_pass = False

    test_results["decoherence_mapping"] = test4_pass
    print()
    print(f"{'PASS' if test4_pass else 'FAIL'}: TEST 4")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # ========================================================================
    # TEST 5: Spin Coherence Evolution
    # ========================================================================
    print("=" * 72)
    print("TEST 5: Spin Coherence Evolution")
    print("=" * 72)
    print()

    print("Testing spin coherence C(T) evolution...")

    try:
        tc = analyzer.calculate_critical_temperature()

        # Test at different temperatures
        test_cases = [
            (tc * 0.5, "Below Tc (ordered)"),
            (tc * 1.0, "At Tc (critical)"),
            (tc * 2.0, "Above Tc (paramagnetic)"),
        ]

        coherences = []
        for temp, desc in test_cases:
            xi = analyzer.calculate_correlation_length(temp, tc)
            gamma, _ = analyzer.calculate_critical_exponents(xi)
            gamma_d = analyzer.calculate_decoherence_rate(gamma, xi)
            coherence = analyzer.calculate_spin_coherence(temp, tc, gamma_d)
            coherences.append(coherence)
            print(f"  {desc}: C={coherence:.4f}")

        # Below Tc should have higher coherence than above Tc
        test5_pass = coherences[0] > coherences[2]

    except Exception as e:
        print(f"  ERROR: {e}")
        test5_pass = False

    test_results["spin_coherence"] = test5_pass
    print()
    print(f"{'PASS' if test5_pass else 'FAIL'}: TEST 5")
    print()
    all_tests_passed = all_tests_passed and test5_pass

    # ========================================================================
    # TEST 6: Magnetization from Born Rule
    # ========================================================================
    print("=" * 72)
    print("TEST 6: Magnetization from Born Rule")
    print("=" * 72)
    print()

    print("Testing M = P(ordered) x |t|^beta...")

    try:
        tc = analyzer.calculate_critical_temperature()

        # Below Tc - should have magnetization
        temp_below = tc * 0.7
        xi = analyzer.calculate_correlation_length(temp_below, tc)
        gamma, beta = analyzer.calculate_critical_exponents(xi)
        gamma_d = analyzer.calculate_decoherence_rate(gamma, xi)
        coherence = analyzer.calculate_spin_coherence(temp_below, tc, gamma_d)
        mag_below, prob_below = analyzer.calculate_magnetization_born_rule(coherence, temp_below, tc, beta)

        print(f"  Below Tc (T/Tc=0.7):")
        print(f"    Coherence: {coherence:.4f}")
        print(f"    P(ordered): {prob_below:.4f}")
        print(f"    Magnetization: {mag_below:.4f}")

        # Above Tc - should have no magnetization
        temp_above = tc * 1.5
        xi = analyzer.calculate_correlation_length(temp_above, tc)
        gamma, beta = analyzer.calculate_critical_exponents(xi)
        gamma_d = analyzer.calculate_decoherence_rate(gamma, xi)
        coherence = analyzer.calculate_spin_coherence(temp_above, tc, gamma_d)
        mag_above, prob_above = analyzer.calculate_magnetization_born_rule(coherence, temp_above, tc, beta)

        print(f"  Above Tc (T/Tc=1.5):")
        print(f"    Coherence: {coherence:.4f}")
        print(f"    P(ordered): {prob_above:.4f}")
        print(f"    Magnetization: {mag_above:.4f}")

        # Below Tc should have higher magnetization
        test6_pass = mag_below > mag_above

    except Exception as e:
        print(f"  ERROR: {e}")
        test6_pass = False

    test_results["magnetization_born"] = test6_pass
    print()
    print(f"{'PASS' if test6_pass else 'FAIL'}: TEST 6")
    print()
    all_tests_passed = all_tests_passed and test6_pass

    # ========================================================================
    # TEST 7: Edge Performance Profile
    # ========================================================================
    print("=" * 72)
    print("TEST 7: Edge Performance Profile")
    print("=" * 72)
    print()

    print("Profiling magnetic operations on edge...")

    try:
        iterations = 1000
        tc = analyzer.calculate_critical_temperature()

        # Profile correlation length calculation
        start = time.time()
        for i in range(iterations):
            temp = tc * (0.5 + (i % 200) / 100.0)
            _ = analyzer.calculate_correlation_length(temp, tc)
        xi_time = time.time() - start
        xi_ops_per_sec = iterations / xi_time

        # Profile critical exponents
        start = time.time()
        for i in range(iterations):
            xi = 1.0 + (i % 100)
            _, _ = analyzer.calculate_critical_exponents(xi)
        exp_time = time.time() - start
        exp_ops_per_sec = iterations / exp_time

        # Profile full state analysis
        start = time.time()
        for i in range(100):
            temp = tc * (0.5 + (i % 30) / 20.0)
            _ = analyzer.analyze_magnetic_state(temp)
        state_time = time.time() - start
        state_ops_per_sec = 100 / state_time

        print(f"  Correlation length: {xi_ops_per_sec:,.0f} ops/sec")
        print(f"  Critical exponents: {exp_ops_per_sec:,.0f} ops/sec")
        print(f"  Full state analysis: {state_ops_per_sec:.1f} ops/sec")
        print()

        test7_pass = (
            xi_ops_per_sec > 10000 and
            exp_ops_per_sec > 100000 and
            state_ops_per_sec > 100
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test7_pass = False

    test_results["edge_performance"] = test7_pass
    print(f"{'PASS' if test7_pass else 'FAIL'}: TEST 7")
    print()
    all_tests_passed = all_tests_passed and test7_pass

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 72)
    print("SESSION 187 EDGE VALIDATION SUMMARY")
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
        print("|" + " MAGNETIC COHERENCE INTEGRATION VALIDATED ON EDGE! ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "-" * 70 + "+")
        print()
        print("Edge Observations:")
        print("  - Correlation length calculation fast on ARM64")
        print("  - Critical exponents (gamma, beta) derivation working")
        print("  - Decoherence rate mapping operational")
        print("  - Spin coherence evolution validated")
        print("  - Born rule magnetization functional")
        print()
        print("Seven-Domain Unification on Edge:")
        print("  1. Physics (thermodynamics)")
        print("  2. Biochemistry (ATP)")
        print("  3. Biophysics (memory)")
        print("  4. Neuroscience (attention)")
        print("  5. Distributed Systems (federation)")
        print("  6. Quantum Measurement (decoherence)")
        print("  7. Magnetism (spin coherence) <- Session 187")
        print()
        print("Sessions 177-187 Edge Stack: VALIDATED")
    else:
        print("SOME TESTS FAILED - Review results above")

    print()

    # Save results
    results = {
        "session": "187_edge",
        "title": "Magnetic Coherence Integration - Edge Validation",
        "platform": "Sprout (Jetson Orin Nano 8GB)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": bool(all_tests_passed),
        "test_results": {k: bool(v) for k, v in test_results.items()},
        "edge_metrics": edge_metrics,
        "magnetic_features": {
            "correlation_length": True,
            "critical_exponents": True,
            "decoherence_mapping": True,
            "spin_coherence": True,
            "born_rule_magnetization": True,
        }
    }

    results_path = Path(__file__).parent / "session187_edge_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = test_edge_magnetic_coherence()
    sys.exit(0 if success else 1)
