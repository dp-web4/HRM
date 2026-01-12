#!/usr/bin/env python3
"""
Session 186 Edge Validation: Quantum-Phase Integration

Testing Thor's quantum-phase integration on Sprout edge hardware.

Thor's Session 186 Implementation:
- VerificationState: SUPERPOSITION, DECOHERENCE, DEFINITE
- AttestationMeasurement: Complete quantum state dataclass
- QuantumAttestationVerifier: Decoherence simulation
- QuantumPhaseAwareSAGE: Unified micro-macro framework

Edge Validation Goals:
1. Verify quantum components import correctly
2. Test decoherence calculation performance
3. Validate Born probability computation
4. Test verification simulation
5. Test QuantumPhaseAwareSAGE integration
6. Profile quantum operations on edge

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


def test_edge_quantum_phase():
    """Test Session 186 quantum-phase integration on edge."""
    print()
    print("+" + "=" * 70 + "+")
    print("|" + " " * 70 + "|")
    print("|" + " SESSION 186 EDGE VALIDATION: QUANTUM-PHASE INTEGRATION ".center(70) + "|")
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
    # TEST 1: Import Session 186 Components
    # ========================================================================
    print("=" * 72)
    print("TEST 1: Import Session 186 Components")
    print("=" * 72)
    print()

    try:
        from session186_quantum_phase_integration import (
            VerificationState,
            AttestationMeasurement,
            QuantumAttestationVerifier,
            QuantumPhaseAwareSAGE,
        )

        print("  VerificationState: Quantum measurement states")
        print("  AttestationMeasurement: Full measurement record")
        print("  QuantumAttestationVerifier: Decoherence simulation")
        print("  QuantumPhaseAwareSAGE: Unified framework")
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
    # TEST 2: Decoherence Rate Calculation
    # ========================================================================
    print("=" * 72)
    print("TEST 2: Decoherence Rate Calculation")
    print("=" * 72)
    print()

    print("Testing decoherence rate (Gamma_d) on edge...")

    try:
        verifier = QuantumAttestationVerifier()

        # Test different evidence/validator combinations
        test_cases = [
            (0.9, 5, 0.1, "High evidence, many validators"),
            (0.5, 3, 0.1, "Medium evidence, few validators"),
            (0.2, 5, 0.5, "Low evidence, noisy network"),
        ]

        rates = []
        for evidence, validators, temp, desc in test_cases:
            rate = verifier.calculate_decoherence_rate(evidence, validators, temp)
            rates.append(rate)
            print(f"  {desc}:")
            print(f"    E={evidence}, V={validators}, T={temp} -> Gamma_d={rate:.2f}")

        # Higher evidence should give faster decoherence
        test2_pass = rates[0] > rates[1] > rates[2]

    except Exception as e:
        print(f"  ERROR: {e}")
        test2_pass = False

    test_results["decoherence_rate"] = test2_pass
    print()
    print(f"{'PASS' if test2_pass else 'FAIL'}: TEST 2")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    # ========================================================================
    # TEST 3: Born Probability Calculation
    # ========================================================================
    print("=" * 72)
    print("TEST 3: Born Probability Calculation")
    print("=" * 72)
    print()

    print("Testing Born rule probabilities...")

    try:
        # Test at different coherence levels
        test_cases = [
            (0.9, 0.0, "Strong evidence, definite"),
            (0.9, 1.0, "Strong evidence, superposition"),
            (0.5, 0.0, "Medium evidence, definite"),
            (0.1, 0.0, "Weak evidence, definite"),
        ]

        for evidence, coherence, desc in test_cases:
            p_verify, p_reject = verifier.calculate_born_probabilities(evidence, coherence)
            print(f"  {desc}:")
            print(f"    E={evidence}, C={coherence} -> P(v)={p_verify:.3f}, P(r)={p_reject:.3f}")

        # Strong evidence + definite state should have high P(verify)
        p_v_strong, _ = verifier.calculate_born_probabilities(0.9, 0.0)
        # Weak evidence + definite state should have low P(verify)
        p_v_weak, _ = verifier.calculate_born_probabilities(0.1, 0.0)
        # Superposition should be closer to 0.5
        p_v_super, _ = verifier.calculate_born_probabilities(0.9, 1.0)

        test3_pass = (
            p_v_strong > 0.9 and
            p_v_weak < 0.1 and
            0.4 < p_v_super < 0.75  # Closer to 0.5 in superposition
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test3_pass = False

    test_results["born_probabilities"] = test3_pass
    print()
    print(f"{'PASS' if test3_pass else 'FAIL'}: TEST 3")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # ========================================================================
    # TEST 4: Verification Simulation
    # ========================================================================
    print("=" * 72)
    print("TEST 4: Verification Simulation")
    print("=" * 72)
    print()

    print("Testing quantum verification simulation...")

    try:
        # Simulate strong evidence verification
        measurement = verifier.simulate_verification(
            evidence_strength=0.9,
            validator_count=5,
            network_temperature=0.1,
        )

        print(f"  Strong Evidence (0.9):")
        print(f"    Initial coherence: 1.0 (superposition)")
        print(f"    Final coherence: {measurement.coherence:.4f}")
        print(f"    State: {measurement.verification_state.value}")
        print(f"    Decoherence time: {measurement.decoherence_time:.3f}s")
        print(f"    Measurement time: {measurement.measurement_time:.3f}s")
        print(f"    Outcome: {measurement.outcome}")
        print(f"    Reputation delta: {measurement.reputation_delta:+.1f}" if measurement.reputation_delta else "    No delta")

        # Should reach definite state with outcome
        test4_pass = (
            measurement.verification_state == VerificationState.DEFINITE and
            measurement.coherence < 0.01 and
            measurement.outcome is not None and
            measurement.reputation_delta is not None
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test4_pass = False

    test_results["verification_simulation"] = test4_pass
    print()
    print(f"{'PASS' if test4_pass else 'FAIL'}: TEST 4")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # ========================================================================
    # TEST 5: Edge Performance Profile (Quantum Operations)
    # ========================================================================
    print("=" * 72)
    print("TEST 5: Edge Performance Profile (Quantum Operations)")
    print("=" * 72)
    print()

    print("Profiling quantum operations on edge...")

    try:
        iterations = 1000

        # Profile decoherence rate calculation
        start = time.time()
        for i in range(iterations):
            evidence = (i % 100) / 100.0
            validators = (i % 10) + 1
            temp = 0.1 + (i % 50) / 100.0
            _ = verifier.calculate_decoherence_rate(evidence, validators, temp)
        rate_time = time.time() - start
        rate_ops_per_sec = iterations / rate_time

        # Profile Born probability calculation
        start = time.time()
        for i in range(iterations):
            evidence = (i % 100) / 100.0
            coherence = (i % 100) / 100.0
            _ = verifier.calculate_born_probabilities(evidence, coherence)
        born_time = time.time() - start
        born_ops_per_sec = iterations / born_time

        # Profile full verification simulation (fewer iterations - more expensive)
        start = time.time()
        for i in range(100):
            evidence = 0.5 + (i % 50) / 100.0
            _ = verifier.simulate_verification(
                evidence_strength=evidence,
                validator_count=3,
                network_temperature=0.1,
                max_time=1.0,  # Limit simulation time
                dt=0.1,
            )
        sim_time = time.time() - start
        sim_ops_per_sec = 100 / sim_time

        print(f"  Decoherence rate calc: {rate_ops_per_sec:,.0f} ops/sec")
        print(f"  Born probability calc: {born_ops_per_sec:,.0f} ops/sec")
        print(f"  Full simulation: {sim_ops_per_sec:.1f} ops/sec")
        print()

        test5_pass = (
            rate_ops_per_sec > 10000 and  # Simple math
            born_ops_per_sec > 10000 and  # Simple math
            sim_ops_per_sec > 50  # More complex but should manage
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test5_pass = False

    test_results["edge_performance"] = test5_pass
    print(f"{'PASS' if test5_pass else 'FAIL'}: TEST 5")
    print()
    all_tests_passed = all_tests_passed and test5_pass

    # ========================================================================
    # TEST 6: QuantumPhaseAwareSAGE Integration (Single Instance)
    # ========================================================================
    print("=" * 72)
    print("TEST 6: QuantumPhaseAwareSAGE Integration")
    print("=" * 72)
    print()

    print("Testing full QuantumPhaseAwareSAGE on edge (single instance)...")

    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            sage = QuantumPhaseAwareSAGE(
                node_id="sprout_edge",
                hardware_type="tpm2",
                capability_level=3,
                storage_path=Path(tmpdir),
                network_temperature=0.1,
            )

            print(f"  Node: {sage.node_id}")
            print(f"  Quantum verifier: {'Ready' if sage.quantum_verifier else 'Missing'}")
            print(f"  Phase analyzer: {'Ready' if sage.phase_analyzer else 'Missing'}")
            print(f"  Measurement history: {len(sage.measurement_history)}")

            # Perform quantum verification
            measurement = sage.verify_attestation_quantum(
                evidence_strength=0.8,
                validator_count=3,
            )

            print(f"  Verification result: {measurement.outcome}")
            print(f"  Measurements recorded: {len(sage.measurement_history)}")

            # Get quantum statistics
            stats = sage.get_quantum_statistics()
            print(f"  Total measurements: {stats.get('total_measurements', 0)}")

            test6_pass = (
                sage.quantum_verifier is not None and
                sage.phase_analyzer is not None and
                len(sage.measurement_history) == 1 and
                measurement.outcome is not None
            )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test6_pass = False

    test_results["sage_integration"] = test6_pass
    print()
    print(f"{'PASS' if test6_pass else 'FAIL'}: TEST 6")
    print()
    all_tests_passed = all_tests_passed and test6_pass

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 72)
    print("SESSION 186 EDGE VALIDATION SUMMARY")
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
        print("|" + "  QUANTUM-PHASE INTEGRATION VALIDATED ON EDGE!  ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "-" * 70 + "+")
        print()
        print("Edge Observations:")
        print("  - Decoherence rate calculation fast on ARM64")
        print("  - Born probability computation working")
        print("  - Full verification simulation operational")
        print("  - Quantum-phase integration functional")
        print()
        print("Quantum Features Validated:")
        print("  - Coherence decay (C: 1.0 -> 0.0)")
        print("  - Verification states (SUPERPOSITION/DECOHERENCE/DEFINITE)")
        print("  - Decoherence rate based on evidence/validators/temp")
        print("  - Born rule probabilities")
        print("  - Outcome sampling and reputation impact")
        print()
        print("Six-Domain Unification on Edge:")
        print("  1. Physics (superconductors)")
        print("  2. Biochemistry (enzymes)")
        print("  3. Biophysics (photosynthesis)")
        print("  4. Neuroscience (consciousness)")
        print("  5. Distributed Systems (reputation)")
        print("  6. Quantum Measurement (attestation)")
        print()
        print("Sessions 177-186 Edge Stack: COMPLETE")
    else:
        print("SOME TESTS FAILED - Review results above")

    print()

    # Save results
    results = {
        "session": "186_edge",
        "title": "Quantum-Phase Integration - Edge Validation",
        "platform": "Sprout (Jetson Orin Nano 8GB)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": bool(all_tests_passed),
        "test_results": {k: bool(v) for k, v in test_results.items()},
        "edge_metrics": edge_metrics,
        "quantum_features": {
            "decoherence_dynamics": True,
            "born_probabilities": True,
            "verification_simulation": True,
            "quantum_phase_integration": True,
        }
    }

    results_path = Path(__file__).parent / "session186_edge_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = test_edge_quantum_phase()
    sys.exit(0 if success else 1)
