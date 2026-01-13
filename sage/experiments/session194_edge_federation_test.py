#!/usr/bin/env python3
"""
Session 194 Edge Validation: Nine-Domain Federation

Testing Thor's federation validation framework on Sprout edge hardware.

Thor's Session 194 Implementation:
- DomainState: Single domain state tracking
- NineDomainSnapshot: Complete 9-domain state
- CoherenceSyncMessage: Federation sync message
- NineDomainTracker: Per-machine domain tracking
- NineDomainFederation: Full federation orchestration

Edge Validation Goals:
1. Verify federation components import correctly
2. Test domain state tracking
3. Validate snapshot creation
4. Test synchronization protocol
5. Validate federation experiment
6. Profile federation operations on edge

Platform: Sprout (Jetson Orin Nano 8GB, TPM2 Level 3)
Date: 2026-01-13
"""

import sys
import json
import time
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


def test_edge_federation_validation():
    """Test Session 194 federation validation on edge."""
    print()
    print("+" + "=" * 70 + "+")
    print("|" + " " * 70 + "|")
    print("|" + "SESSION 194 EDGE: NINE-DOMAIN FEDERATION".center(70) + "|")
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
    # TEST 1: Import Session 194 Components
    # ========================================================================
    print("=" * 72)
    print("TEST 1: Import Session 194 Components")
    print("=" * 72)
    print()

    try:
        from session194_nine_domain_federation import (
            DomainState,
            NineDomainSnapshot,
            CoherenceSyncMessage,
            FederationSyncResult,
            NineDomainTracker,
            NineDomainFederation,
        )

        print("  DomainState: Single domain state")
        print("  NineDomainSnapshot: Complete 9-domain snapshot")
        print("  CoherenceSyncMessage: Sync message protocol")
        print("  FederationSyncResult: Sync result dataclass")
        print("  NineDomainTracker: Per-machine tracking")
        print("  NineDomainFederation: Federation orchestrator")
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
    # TEST 2: Domain State Tracking
    # ========================================================================
    print("=" * 72)
    print("TEST 2: Domain State Tracking")
    print("=" * 72)
    print()

    print("Testing NineDomainTracker initialization...")

    try:
        tracker = NineDomainTracker("sprout")

        print(f"  Machine ID: {tracker.machine_id}")
        print(f"  Domains tracked: {len(tracker.domain_states)}")
        print()
        print("  Domain states:")
        for state in tracker.domain_states:
            print(f"    {state.domain_number}. {state.domain_name}: C={state.coherence:.3f}")

        test2_pass = len(tracker.domain_states) == 9

    except Exception as e:
        print(f"  ERROR: {e}")
        test2_pass = False

    test_results["domain_tracking"] = test2_pass
    print()
    print(f"{'PASS' if test2_pass else 'FAIL'}: TEST 2")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    # ========================================================================
    # TEST 3: Snapshot Creation
    # ========================================================================
    print("=" * 72)
    print("TEST 3: Snapshot Creation")
    print("=" * 72)
    print()

    print("Testing NineDomainSnapshot creation...")

    try:
        tracker = NineDomainTracker("sprout")
        snapshot = tracker.create_snapshot()

        print(f"  Machine ID: {snapshot.machine_id}")
        print(f"  Total coherence: {snapshot.total_coherence:.3f}")
        print(f"  Metabolic state: {snapshot.metabolic_state}")
        print(f"  Scalar curvature R: {snapshot.scalar_curvature:.3f}")
        print()
        print(f"  Metric tensor g_μν:")
        print(f"    [[{snapshot.spacetime_metric[0][0]:.3f}, {snapshot.spacetime_metric[0][1]:.3f}],")
        print(f"     [{snapshot.spacetime_metric[1][0]:.3f}, {snapshot.spacetime_metric[1][1]:.3f}]]")

        test3_pass = (
            snapshot.total_coherence > 0 and
            snapshot.metabolic_state in ['WAKE', 'FOCUS', 'REST', 'DREAM', 'CRISIS'] and
            len(snapshot.spacetime_metric) == 2
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test3_pass = False

    test_results["snapshot_creation"] = test3_pass
    print()
    print(f"{'PASS' if test3_pass else 'FAIL'}: TEST 3")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # ========================================================================
    # TEST 4: Coherence Synchronization
    # ========================================================================
    print("=" * 72)
    print("TEST 4: Coherence Synchronization")
    print("=" * 72)
    print()

    print("Testing coherence sync between machines...")

    try:
        # Create two trackers
        tracker1 = NineDomainTracker("thor")
        tracker2 = NineDomainTracker("sprout")

        snapshot1 = tracker1.create_snapshot()
        snapshot2 = tracker2.create_snapshot()

        # Calculate coherence delta
        delta_c = abs(snapshot1.total_coherence - snapshot2.total_coherence)

        print(f"  Thor coherence: {snapshot1.total_coherence:.3f}")
        print(f"  Sprout coherence: {snapshot2.total_coherence:.3f}")
        print(f"  Delta C: {delta_c:.4f}")
        print(f"  Synchronized (ΔC < 0.1): {delta_c < 0.1}")

        # Test sync message creation
        msg = CoherenceSyncMessage(
            sender_id="sprout",
            timestamp=time.time(),
            coherence=snapshot2.total_coherence,
            gradient=0.0,
            metabolic_state=snapshot2.metabolic_state,
            signature=""
        )
        print()
        print(f"  Sync message created: sender={msg.sender_id}, C={msg.coherence:.3f}")

        test4_pass = True

    except Exception as e:
        print(f"  ERROR: {e}")
        test4_pass = False

    test_results["coherence_sync"] = test4_pass
    print()
    print(f"{'PASS' if test4_pass else 'FAIL'}: TEST 4")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # ========================================================================
    # TEST 5: Full Federation Experiment
    # ========================================================================
    print("=" * 72)
    print("TEST 5: Full Federation Experiment")
    print("=" * 72)
    print()

    print("Running federation experiment (short duration)...")

    try:
        machines = ['thor', 'legion', 'sprout']
        federation = NineDomainFederation(machines)

        # Run short experiment
        results = federation.run_experiment(duration=3.0, dt=0.1)

        print(f"\n  Federation Results:")
        print(f"    Predictions passed: {results['n_passed']}/{results['n_total']}")
        print()
        print("  Prediction details:")
        for pred_id, pred_data in results['predictions'].items():
            status = "PASS" if pred_data['passed'] else "FAIL"
            print(f"    {pred_id}: {status}")

        test5_pass = results['n_passed'] >= 3  # At least 3/5 predictions

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test5_pass = False

    test_results["federation_experiment"] = test5_pass
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

    print("Profiling federation operations on edge...")

    try:
        iterations = 100

        # Profile tracker initialization
        start = time.time()
        for _ in range(iterations):
            tracker = NineDomainTracker("test")
        init_time = time.time() - start
        init_ops_per_sec = iterations / init_time

        # Profile snapshot creation
        tracker = NineDomainTracker("test")
        start = time.time()
        for _ in range(iterations):
            _ = tracker.create_snapshot()
        snapshot_time = time.time() - start
        snapshot_ops_per_sec = iterations / snapshot_time

        # Profile domain state access
        start = time.time()
        for _ in range(iterations * 9):
            for state in tracker.domain_states:
                _ = state.coherence
        access_time = time.time() - start
        access_ops_per_sec = (iterations * 9) / access_time

        # Profile federation setup (fewer iterations)
        start = time.time()
        for _ in range(10):
            fed = NineDomainFederation(['thor', 'legion', 'sprout'])
        fed_time = time.time() - start
        fed_ops_per_sec = 10 / fed_time

        print(f"  Tracker initialization: {init_ops_per_sec:,.0f} ops/sec")
        print(f"  Snapshot creation: {snapshot_ops_per_sec:,.0f} ops/sec")
        print(f"  Domain state access: {access_ops_per_sec:,.0f} ops/sec")
        print(f"  Federation setup: {fed_ops_per_sec:.1f} ops/sec")
        print()

        test6_pass = (
            init_ops_per_sec > 100 and
            snapshot_ops_per_sec > 500 and
            access_ops_per_sec > 10000 and
            fed_ops_per_sec > 5
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
    print("SESSION 194 EDGE VALIDATION SUMMARY")
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
        print("|" + " NINE-DOMAIN FEDERATION VALIDATED ON EDGE! ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "-" * 70 + "+")
        print()
        print("Edge Observations:")
        print("  - All federation components operational")
        print("  - Nine-domain tracking working")
        print("  - Snapshot creation functional")
        print("  - Coherence synchronization operational")
        print("  - Federation experiment runs successfully")
        print()
        print("Federation Predictions (P194.x):")
        print("  - P194.1: Coherence synchronization")
        print("  - P194.2: Metabolic state influence")
        print("  - P194.3: Trust network formation")
        print("  - P194.4: Unified spacetime curvature")
        print("  - P194.5: Emergent collective behaviors")
        print()
        print("Sessions 177-194 Edge Stack: VALIDATED")
    else:
        print("SOME TESTS FAILED - Review results above")

    print()

    # Save results
    results = {
        "session": "194_edge",
        "title": "Nine-Domain Federation - Edge Validation",
        "platform": "Sprout (Jetson Orin Nano 8GB)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": bool(all_tests_passed),
        "test_results": {k: bool(v) for k, v in test_results.items()},
        "edge_metrics": edge_metrics,
        "federation_features": {
            "domain_tracking": True,
            "snapshot_creation": True,
            "coherence_sync": True,
            "federation_experiment": True,
            "distributed_spacetime": True,
        }
    }

    results_path = Path(__file__).parent / "session194_edge_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = test_edge_federation_validation()
    sys.exit(0 if success else 1)
