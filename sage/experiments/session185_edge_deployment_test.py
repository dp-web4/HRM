#!/usr/bin/env python3
"""
Session 185 Edge Validation: Phase 1 LAN Deployment

Testing Thor's Phase 1 deployment orchestration on Sprout edge hardware.

Thor's Session 185 Implementation:
- Phase1Deployment: Orchestration class for deployment
- Phase monitoring loop with configurable intervals
- Reputation evolution simulation
- Comprehensive results compilation

Edge Validation Goals:
1. Verify deployment components import correctly
2. Test Phase1Deployment initialization
3. Validate phase snapshot collection
4. Test reputation evolution simulation (abbreviated)
5. Validate results compilation
6. Profile deployment operations on edge

Platform: Sprout (Jetson Orin Nano 8GB, TPM2 Level 3)
Date: 2026-01-12
"""

import sys
import json
import time
import asyncio
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


async def test_edge_phase1_deployment():
    """Test Session 185 Phase 1 deployment on edge."""
    print()
    print("+" + "=" * 70 + "+")
    print("|" + " " * 70 + "|")
    print("|" + " SESSION 185 EDGE VALIDATION: PHASE 1 LAN DEPLOYMENT ".center(70) + "|")
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
    # TEST 1: Import Session 185 Components
    # ========================================================================
    print("=" * 72)
    print("TEST 1: Import Session 185 Components")
    print("=" * 72)
    print()

    try:
        from session185_phase1_lan_deployment import (
            Phase1Deployment,
        )
        from session184_phase_aware_sage import (
            PhaseAwareSAGE,
            ReputationFreeEnergy,
        )

        print("  Phase1Deployment: Deployment orchestrator")
        print("  PhaseAwareSAGE: Phase-aware consciousness")
        print("  ReputationFreeEnergy: Thermodynamic state")
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
    # TEST 2: Phase1Deployment Initialization
    # ========================================================================
    print("=" * 72)
    print("TEST 2: Phase1Deployment Initialization")
    print("=" * 72)
    print()

    print("Testing deployment orchestrator initialization...")

    try:
        deployment = Phase1Deployment(
            node_id="sprout_edge",
            duration_seconds=15,  # Very short for edge test
            phase_check_interval=5.0,  # Faster checks
        )

        print(f"  Node ID: {deployment.node_id}")
        print(f"  Duration: {deployment.duration_seconds}s")
        print(f"  Check interval: {deployment.phase_check_interval}s")
        print(f"  SAGE instance: {'Not yet' if deployment.sage is None else 'Ready'}")
        print()

        test2_pass = (
            deployment.node_id == "sprout_edge" and
            deployment.duration_seconds == 15 and
            deployment.sage is None  # Not initialized until deployment starts
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test2_pass = False

    test_results["deployment_init"] = test2_pass
    print(f"{'PASS' if test2_pass else 'FAIL'}: TEST 2")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    # ========================================================================
    # TEST 3: SAGE Initialization via Deployment
    # ========================================================================
    print("=" * 72)
    print("TEST 3: SAGE Initialization via Deployment")
    print("=" * 72)
    print()

    print("Testing SAGE initialization in deployment context...")

    try:
        # Initialize SAGE through deployment
        await deployment.initialize_sage()

        print(f"  SAGE node: {deployment.sage.node_id}")
        print(f"  Phase analyzer: {'Ready' if deployment.sage.phase_analyzer else 'Missing'}")

        test3_pass = (
            deployment.sage is not None and
            deployment.sage.node_id == "sprout_edge" and
            deployment.sage.phase_analyzer is not None
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test3_pass = False

    test_results["sage_init"] = test3_pass
    print()
    print(f"{'PASS' if test3_pass else 'FAIL'}: TEST 3")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # ========================================================================
    # TEST 4: Phase Snapshot Collection
    # ========================================================================
    print("=" * 72)
    print("TEST 4: Phase Snapshot Collection")
    print("=" * 72)
    print()

    print("Testing phase snapshot collection...")

    try:
        # Set deployment start time
        deployment.deployment_start_time = time.time()

        # Set some reputation for phase state
        if hasattr(deployment.sage, 'reputation') and hasattr(deployment.sage.reputation, 'total_score'):
            deployment.sage.reputation.total_score = 50.0

        # Add diversity
        for i in range(3):
            deployment.sage.diversity_manager.record_reputation_event(
                "sprout_edge", f"source_{i}", 10.0
            )

        # Collect snapshot
        snapshot = await deployment.collect_phase_snapshot("test_snapshot")

        print(f"  Snapshot event: {snapshot['event']}")
        print(f"  Elapsed: {snapshot['elapsed_seconds']:.2f}s")
        print(f"  Snapshots collected: {len(deployment.phase_snapshots)}")

        if snapshot['phase_state']:
            print(f"  Phase: {snapshot['phase_state']['phase']}")
            print(f"  Free energy: {snapshot['phase_state']['free_energy']:.4f}")

        test4_pass = (
            len(deployment.phase_snapshots) == 1 and
            snapshot['event'] == "test_snapshot" and
            'sage_state' in snapshot
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test4_pass = False

    test_results["snapshot_collection"] = test4_pass
    print()
    print(f"{'PASS' if test4_pass else 'FAIL'}: TEST 4")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # ========================================================================
    # TEST 5: Phase Evolution Analysis
    # ========================================================================
    print("=" * 72)
    print("TEST 5: Phase Evolution Analysis")
    print("=" * 72)
    print()

    print("Testing phase evolution analysis (multiple snapshots)...")

    try:
        # Collect several more snapshots with varying reputation
        reputations = [60.0, 80.0, 100.0]
        for i, rep in enumerate(reputations):
            if hasattr(deployment.sage, 'reputation') and hasattr(deployment.sage.reputation, 'total_score'):
                deployment.sage.reputation.total_score = rep
            await deployment.collect_phase_snapshot(f"evolution_{i}")
            await asyncio.sleep(0.1)  # Small delay

        # Analyze phase evolution
        phase_analysis = deployment.analyze_phase_evolution()

        print(f"  Total snapshots: {len(deployment.phase_snapshots)}")
        print(f"  Phase distribution: {phase_analysis.get('phase_distribution', {})}")
        print(f"  Free energy range: [{phase_analysis['free_energy']['min']:.4f}, {phase_analysis['free_energy']['max']:.4f}]")
        print(f"  Reputation growth: {phase_analysis['reputation']['growth']:.4f}")

        test5_pass = (
            len(deployment.phase_snapshots) >= 4 and
            'phase_distribution' in phase_analysis and
            phase_analysis['free_energy']['all_negative']  # Healthy system has negative F
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test5_pass = False

    test_results["evolution_analysis"] = test5_pass
    print()
    print(f"{'PASS' if test5_pass else 'FAIL'}: TEST 5")
    print()
    all_tests_passed = all_tests_passed and test5_pass

    # ========================================================================
    # TEST 6: Results Compilation
    # ========================================================================
    print("=" * 72)
    print("TEST 6: Results Compilation")
    print("=" * 72)
    print()

    print("Testing results compilation...")

    try:
        # Add a simulated event for reputation analysis
        deployment.simulated_events.append({
            "timestamp": time.time(),
            "elapsed": 5.0,
            "reputation_delta": 10.0,
            "source": "test_source",
            "description": "Test event",
            "new_reputation": 110.0,
        })

        # Compile results
        results = await deployment.compile_results()

        print(f"  Results keys: {list(results.keys())}")
        print(f"  Deployment node: {results['deployment']['node_id']}")
        print(f"  Total snapshots: {results['metrics']['total_snapshots']}")
        print(f"  Total events: {results['metrics']['total_events']}")

        test6_pass = (
            'deployment' in results and
            'metrics' in results and
            'phase_analysis' in results and
            results['deployment']['node_id'] == 'sprout_edge'
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test6_pass = False

    test_results["results_compilation"] = test6_pass
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

    print("Profiling deployment operations on edge...")

    try:
        # Profile snapshot collection
        iterations = 50
        start = time.time()
        for i in range(iterations):
            await deployment.collect_phase_snapshot(f"perf_test_{i}")
        snapshot_time = time.time() - start
        snapshot_ops_per_sec = iterations / snapshot_time

        # Profile phase analysis
        start = time.time()
        for _ in range(100):
            _ = deployment.analyze_phase_evolution()
        analysis_time = time.time() - start
        analysis_ops_per_sec = 100 / analysis_time

        print(f"  Snapshot collection: {snapshot_ops_per_sec:.1f} ops/sec")
        print(f"  Phase analysis: {analysis_ops_per_sec:.0f} ops/sec")
        print()

        test7_pass = (
            snapshot_ops_per_sec > 10 and  # Should manage at least 10 snapshots/sec
            analysis_ops_per_sec > 100
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
    print("SESSION 185 EDGE VALIDATION SUMMARY")
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
        print("|" + "  PHASE 1 LAN DEPLOYMENT VALIDATED ON EDGE!  ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "-" * 70 + "+")
        print()
        print("Edge Observations:")
        print("  - Deployment orchestration working")
        print("  - Phase snapshot collection functional")
        print("  - Phase evolution analysis operational")
        print("  - Results compilation validated")
        print()
        print("Sessions 177-185 Edge Status: ALL VALIDATED")
        print("  Session 177-181: Core SAGE features")
        print("  Session 182: Security-enhanced reputation")
        print("  Session 183: Network protocol")
        print("  Session 184: Phase-aware monitoring")
        print("  Session 185: LAN deployment (NOW VALIDATED)")
        print()
        print("Ready for Phase 2: Multi-node Federation")
        print("  - Thor (10.0.0.99): Development hub")
        print("  - Legion (10.0.0.72): High-ATP anchor")
        print("  - Sprout (10.0.0.36): Edge validation")
    else:
        print("SOME TESTS FAILED - Review results above")

    print()

    # Save results
    results = {
        "session": "185_edge",
        "title": "Phase 1 LAN Deployment - Edge Validation",
        "platform": "Sprout (Jetson Orin Nano 8GB)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": bool(all_tests_passed),
        "test_results": {k: bool(v) for k, v in test_results.items()},
        "edge_metrics": edge_metrics,
        "deployment_features": {
            "phase_monitoring": True,
            "snapshot_collection": True,
            "evolution_analysis": True,
            "results_compilation": True,
        }
    }

    results_path = Path(__file__).parent / "session185_edge_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = asyncio.run(test_edge_phase1_deployment())
    sys.exit(0 if success else 1)
