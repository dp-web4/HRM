#!/usr/bin/env python3
"""
Sessions 179-180 Edge Validation: Reputation-Aware Depth & Persistence

Testing Thor's reputation-based cognitive depth and persistence on Sprout.

Thor's Implementations:
- Session 179: Reputation acts as "cognitive credit" - high-reputation nodes
  can operate at lighter depths while maintaining trust
- Session 180: Persistent reputation storage anchored to hardware identity,
  survives across sessions

Edge Validation Goals:
1. Verify reputation multiplier calculations on edge
2. Test cognitive credit effects on depth selection
3. Validate persistent storage I/O on ARM64
4. Profile reputation system performance
5. Test cross-session reputation persistence

Platform: Sprout (Jetson Orin Nano 8GB, TPM2 Level 3)
Session: Autonomous Edge Validation - Sessions 179-180
Date: 2026-01-10
"""

import sys
import json
import time
import os
import tempfile
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


def test_edge_reputation_systems():
    """Test Sessions 179-180 reputation systems on edge."""
    print()
    print("+" + "=" * 70 + "+")
    print("|" + " " * 70 + "|")
    print("|" + "  SESSIONS 179-180 EDGE VALIDATION: REPUTATION SYSTEMS  ".center(70) + "|")
    print("|" + "           Jetson Orin Nano 8GB (Sprout)                 ".center(70) + "|")
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
    # TEST 1: Import Session 179-180 Components
    # ========================================================================
    print("=" * 72)
    print("TEST 1: Import Session 179-180 Components")
    print("=" * 72)
    print()

    try:
        from session179_reputation_aware_depth import (
            SimpleReputation,
            ReputationAwareAdaptiveSAGE,
        )
        from session180_persistent_reputation import (
            ReputationEvent,
            PersistentReputationScore,
            PersistentReputationManager,
        )
        from session178_federated_sage_verification import CognitiveDepth

        print("  Session 179: SimpleReputation, ReputationAwareAdaptiveSAGE")
        print("  Session 180: ReputationEvent, PersistentReputationScore, Manager")
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
    # TEST 2: Session 179 - Reputation Multiplier Logic
    # ========================================================================
    print("=" * 72)
    print("TEST 2: Session 179 - Reputation Multiplier Logic")
    print("=" * 72)
    print()

    print("Testing cognitive credit multipliers...")

    try:
        test_cases = [
            (60, "excellent", 0.7),   # 30% bonus
            (30, "good", 0.85),       # 15% bonus
            (10, "neutral", 1.0),     # No adjustment
            (-10, "poor", 1.15),      # 15% penalty
            (-30, "untrusted", 1.3),  # 30% penalty
        ]

        passed = 0
        for score, expected_level, expected_mult in test_cases:
            rep = SimpleReputation(
                node_id="test_node",
                total_score=score,
                event_count=10,
                positive_events=5,
                negative_events=5
            )

            level_match = rep.reputation_level == expected_level
            mult_match = abs(rep.reputation_multiplier - expected_mult) < 0.01

            if level_match and mult_match:
                passed += 1
                print(f"    Score {score:+3d}: {rep.reputation_level} ({rep.reputation_multiplier:.2f}x)")
            else:
                print(f"    FAIL: Score {score} -> {rep.reputation_level}/{rep.reputation_multiplier}")

        print()
        test2_pass = passed == len(test_cases)

    except Exception as e:
        print(f"  ERROR: {e}")
        test2_pass = False

    test_results["reputation_multiplier"] = test2_pass
    print(f"{'PASS' if test2_pass else 'FAIL'}: TEST 2")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    # ========================================================================
    # TEST 3: Session 179 - Reputation Event Recording
    # ========================================================================
    print("=" * 72)
    print("TEST 3: Session 179 - Reputation Event Recording")
    print("=" * 72)
    print()

    print("Testing reputation accumulation...")

    try:
        rep = SimpleReputation(
            node_id="accumulation_test",
            total_score=0.0,
            event_count=0,
            positive_events=0,
            negative_events=0
        )

        # Record positive events
        for _ in range(5):
            rep.record_event(0.8)  # High quality

        # Record negative events
        for _ in range(2):
            rep.record_event(-0.5)  # Low quality

        print(f"  After 5 positive (+0.8) and 2 negative (-0.5) events:")
        print(f"    Total score: {rep.total_score:.1f}")
        print(f"    Event count: {rep.event_count}")
        print(f"    Positive: {rep.positive_events}, Negative: {rep.negative_events}")
        print(f"    Level: {rep.reputation_level}")
        print(f"    Multiplier: {rep.reputation_multiplier:.2f}x")
        print()

        # Expected: 5*8 - 2*5 = 40 - 10 = 30 (good reputation)
        test3_pass = (
            rep.total_score == 30.0 and
            rep.event_count == 7 and
            rep.positive_events == 5 and
            rep.negative_events == 2 and
            rep.reputation_level == "good"
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        test3_pass = False

    test_results["reputation_accumulation"] = test3_pass
    print(f"{'PASS' if test3_pass else 'FAIL'}: TEST 3")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # ========================================================================
    # TEST 4: Session 180 - Persistent Storage I/O
    # ========================================================================
    print("=" * 72)
    print("TEST 4: Session 180 - Persistent Storage I/O")
    print("=" * 72)
    print()

    print("Testing persistent reputation storage on edge...")

    try:
        # Create temp directory for test storage
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PersistentReputationManager(
                storage_path=Path(tmpdir),
                node_id="edge_manager"
            )

            # Record events
            start = time.time()
            for i in range(10):
                manager.record_event(
                    node_id="edge_test_node",
                    event_type="quality_event",
                    impact=0.7 if i % 2 == 0 else -0.3,
                    context={"test": i}
                )
            write_time = time.time() - start

            # Get reputation
            start = time.time()
            score = manager.get_score("edge_test_node")
            read_time = time.time() - start

            print(f"  10 events written in {write_time*1000:.2f}ms")
            print(f"  Reputation read in {read_time*1000:.2f}ms")
            print(f"  Score: {score.total_score:.1f}")
            print(f"  Level: {score.reputation_level}")
            print()

            # Verify persistence - reload manager
            manager2 = PersistentReputationManager(
                storage_path=Path(tmpdir),
                node_id="edge_manager"
            )
            score2 = manager2.get_score("edge_test_node")

            print(f"  After reload:")
            print(f"    Score: {score2.total_score:.1f}")
            print(f"    Events: {score2.event_count}")
            print()

            test4_pass = (
                score.total_score == score2.total_score and
                score.event_count == score2.event_count and
                write_time < 1.0  # Should be fast
            )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test4_pass = False

    test_results["persistent_storage"] = test4_pass
    print(f"{'PASS' if test4_pass else 'FAIL'}: TEST 4")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # ========================================================================
    # TEST 5: Session 180 - Cross-Session Persistence
    # ========================================================================
    print("=" * 72)
    print("TEST 5: Session 180 - Cross-Session Persistence")
    print("=" * 72)
    print()

    print("Testing reputation persistence across sessions...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Session 1: Build reputation
            mgr1 = PersistentReputationManager(
                storage_path=Path(tmpdir),
                node_id="session_mgr"
            )

            for i in range(20):
                mgr1.record_event(
                    node_id="persistent_node",
                    event_type="quality_event",
                    impact=0.9,  # High quality
                    context={"session": 1, "event": i}
                )

            score1 = mgr1.get_score("persistent_node")
            print(f"  Session 1: {score1.event_count} events, score {score1.total_score:.1f}")

            # Session 2: Continue reputation (simulating restart)
            mgr2 = PersistentReputationManager(
                storage_path=Path(tmpdir),
                node_id="session_mgr"
            )

            # Verify existing reputation loaded
            score2_loaded = mgr2.get_score("persistent_node")
            # Capture values before mutation (get_score returns same object)
            loaded_score_val = score2_loaded.total_score
            loaded_event_count = score2_loaded.event_count
            print(f"  Session 2 (loaded): {loaded_event_count} events, score {loaded_score_val:.1f}")

            # Add more events
            for i in range(10):
                mgr2.record_event(
                    node_id="persistent_node",
                    event_type="quality_event",
                    impact=0.8,
                    context={"session": 2, "event": i}
                )

            score2_final = mgr2.get_score("persistent_node")
            print(f"  Session 2 (final): {score2_final.event_count} events, score {score2_final.total_score:.1f}")
            print()

            # Verify accumulation (compare captured value, not mutated object)
            test5_pass = (
                abs(loaded_score_val - score1.total_score) < 0.01 and
                score2_final.event_count == 30 and
                score2_final.total_score > score1.total_score
            )

    except Exception as e:
        print(f"  ERROR: {e}")
        test5_pass = False

    test_results["cross_session_persistence"] = test5_pass
    print(f"{'PASS' if test5_pass else 'FAIL'}: TEST 5")
    print()
    all_tests_passed = all_tests_passed and test5_pass

    # ========================================================================
    # TEST 6: Cognitive Credit Effect
    # ========================================================================
    print("=" * 72)
    print("TEST 6: Cognitive Credit Effect (Reputation -> ATP Efficiency)")
    print("=" * 72)
    print()

    print("Testing reputation-based cognitive efficiency...")

    try:
        # High reputation node - gets bonus
        high_rep = SimpleReputation("high_rep", 60, 50, 45, 5)  # excellent

        # Low reputation node - gets penalty
        low_rep = SimpleReputation("low_rep", -30, 50, 10, 40)  # untrusted

        # Same ATP level for both
        base_atp = 80.0

        # Effective ATP = actual ATP / multiplier
        high_effective = base_atp / high_rep.reputation_multiplier
        low_effective = base_atp / low_rep.reputation_multiplier

        print(f"  High reputation node (score: {high_rep.total_score}):")
        print(f"    Multiplier: {high_rep.reputation_multiplier:.2f}x")
        print(f"    Effective ATP: {high_effective:.1f} (from {base_atp})")
        print()
        print(f"  Low reputation node (score: {low_rep.total_score}):")
        print(f"    Multiplier: {low_rep.reputation_multiplier:.2f}x")
        print(f"    Effective ATP: {low_effective:.1f} (from {base_atp})")
        print()

        # High-rep should have higher effective ATP
        efficiency_gap = high_effective - low_effective
        print(f"  Cognitive credit gap: {efficiency_gap:.1f} ATP")
        print()

        test6_pass = efficiency_gap > 30  # Significant efficiency gain

    except Exception as e:
        print(f"  ERROR: {e}")
        test6_pass = False

    test_results["cognitive_credit_effect"] = test6_pass
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

    print("Profiling reputation operations on edge...")

    try:
        iterations = 1000

        # Test reputation calculations
        rep = SimpleReputation("perf_test", 50, 100, 80, 20)

        start = time.time()
        for _ in range(iterations):
            _ = rep.reputation_level
            _ = rep.reputation_multiplier
        calc_time = time.time() - start

        calc_ops_per_sec = iterations * 2 / calc_time

        # Test event recording
        start = time.time()
        for i in range(iterations):
            rep.record_event(0.5 if i % 2 == 0 else -0.3)
        record_time = time.time() - start

        record_ops_per_sec = iterations / record_time

        print(f"  Reputation calculations: {calc_ops_per_sec:,.0f} ops/sec")
        print(f"  Event recording: {record_ops_per_sec:,.0f} ops/sec")
        print()

        test7_pass = calc_ops_per_sec > 100000 and record_ops_per_sec > 100000

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
    print("SESSIONS 179-180 EDGE VALIDATION SUMMARY")
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
        print("|" + "  REPUTATION SYSTEMS VALIDATED ON EDGE!  ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "-" * 70 + "+")
        print()
        print("Edge Observations:")
        print("  - Reputation multiplier logic correct (0.7x - 1.3x)")
        print("  - Cognitive credit creates significant efficiency gap")
        print("  - Persistent storage I/O fast on ARM64")
        print("  - Cross-session reputation accumulation working")
        print("  - High performance on edge hardware")
    else:
        print("SOME TESTS FAILED - Review results above")

    print()

    # Save results
    results = {
        "session": "179_180_edge",
        "title": "Reputation Systems - Edge Validation",
        "platform": "Sprout (Jetson Orin Nano 8GB)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": all_tests_passed,
        "test_results": test_results,
        "edge_metrics": edge_metrics,
        "convergent_research": {
            "session_179": "Reputation-Aware Adaptive Depth",
            "session_180": "Persistent Reputation Storage",
            "edge_validation": "Cognitive credit and persistence validated"
        }
    }

    results_path = Path(__file__).parent / "session179_180_edge_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = test_edge_reputation_systems()
    sys.exit(0 if success else 1)
