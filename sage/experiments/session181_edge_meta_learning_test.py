#!/usr/bin/env python3
"""
Session 181 Edge Validation: Meta-Learning Adaptive Depth

Testing Thor's meta-learning depth selection on Sprout edge hardware.

Thor's Session 181 Implementation:
- System learns optimal depth from verification history
- Pattern persistence across sessions
- Combines ATP + reputation + network + learning signals
- Self-optimizing consciousness architecture

Edge Validation Goals:
1. Verify pattern storage I/O on ARM64
2. Test learning insight extraction
3. Validate cross-session learning persistence
4. Profile meta-learning performance
5. Test learned depth preference selection

Platform: Sprout (Jetson Orin Nano 8GB, TPM2 Level 3)
Session: Autonomous Edge Validation - Session 181
Date: 2026-01-10
"""

import sys
import json
import time
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


def test_edge_meta_learning():
    """Test Session 181 meta-learning on edge."""
    print()
    print("+" + "=" * 70 + "+")
    print("|" + " " * 70 + "|")
    print("|" + "  SESSION 181 EDGE VALIDATION: META-LEARNING ADAPTIVE DEPTH  ".center(70) + "|")
    print("|" + "             Jetson Orin Nano 8GB (Sprout)                   ".center(70) + "|")
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
    # TEST 1: Import Session 181 Components
    # ========================================================================
    print("=" * 72)
    print("TEST 1: Import Session 181 Components")
    print("=" * 72)
    print()

    try:
        from session181_meta_learning_adaptive_depth import (
            DepthVerificationPattern,
            LearningInsight,
            PersistentMetaLearningManager,
            MetaLearningAdaptiveSAGE,
        )
        from session178_federated_sage_verification import CognitiveDepth

        print("  DepthVerificationPattern: Pattern data class")
        print("  LearningInsight: Insight data class")
        print("  PersistentMetaLearningManager: Storage manager")
        print("  MetaLearningAdaptiveSAGE: Full integration")
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
    # TEST 2: Pattern Storage I/O
    # ========================================================================
    print("=" * 72)
    print("TEST 2: Pattern Storage I/O on Edge")
    print("=" * 72)
    print()

    print("Testing pattern storage performance...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PersistentMetaLearningManager(
                storage_path=Path(tmpdir),
                node_id="edge_test"
            )

            # Record patterns
            start = time.time()
            for i in range(20):
                manager.record_pattern(
                    depth_used=CognitiveDepth.STANDARD,
                    atp_before=100.0,
                    atp_after=80.0,
                    reputation_before=25.0,
                    reputation_after=26.0,
                    quality_achieved=0.75,
                    success=True,
                    context={"iteration": i}
                )
            write_time = time.time() - start

            # Analyze patterns
            start = time.time()
            insights = manager.analyze_patterns()
            analyze_time = time.time() - start

            print(f"  20 patterns written in {write_time*1000:.2f}ms")
            print(f"  Pattern analysis in {analyze_time*1000:.2f}ms")
            print(f"  Insights generated: {len(insights)}")
            print()

            # Verify persistence - reload manager
            manager2 = PersistentMetaLearningManager(
                storage_path=Path(tmpdir),
                node_id="edge_test"
            )

            print(f"  After reload:")
            print(f"    Patterns recovered: {len(manager2.patterns)}")
            print()

            test2_pass = (
                len(manager2.patterns) == 20 and
                len(insights) >= 1 and
                write_time < 1.0
            )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test2_pass = False

    test_results["pattern_storage"] = test2_pass
    print(f"{'PASS' if test2_pass else 'FAIL'}: TEST 2")
    print()
    all_tests_passed = all_tests_passed and test2_pass

    # ========================================================================
    # TEST 3: Learning Insight Extraction
    # ========================================================================
    print("=" * 72)
    print("TEST 3: Learning Insight Extraction")
    print("=" * 72)
    print()

    print("Testing learning from different depth outcomes...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PersistentMetaLearningManager(
                storage_path=Path(tmpdir),
                node_id="insight_test"
            )

            # DEEP produces best quality
            for i in range(5):
                manager.record_pattern(
                    depth_used=CognitiveDepth.DEEP,
                    atp_before=100.0, atp_after=65.0,
                    reputation_before=20.0, reputation_after=22.0,
                    quality_achieved=0.9,
                    success=True,
                    context={"depth_group": "deep"}
                )

            # STANDARD produces medium quality
            for i in range(5):
                manager.record_pattern(
                    depth_used=CognitiveDepth.STANDARD,
                    atp_before=100.0, atp_after=80.0,
                    reputation_before=20.0, reputation_after=20.5,
                    quality_achieved=0.7,
                    success=True,
                    context={"depth_group": "standard"}
                )

            # LIGHT produces lower quality
            for i in range(5):
                manager.record_pattern(
                    depth_used=CognitiveDepth.LIGHT,
                    atp_before=100.0, atp_after=91.0,
                    reputation_before=20.0, reputation_after=19.0,
                    quality_achieved=0.5,
                    success=False,
                    context={"depth_group": "light"}
                )

            # Analyze patterns
            insights = manager.analyze_patterns()

            print(f"  Recorded 15 patterns across 3 depths")
            print(f"  Generated {len(insights)} insights:")
            print()

            for insight in insights:
                print(f"    {insight.insight_type}:")
                print(f"      {insight.description}")
                print(f"      Confidence: {insight.confidence:.1%}")
                print()

            # Verify insights reflect data
            # Should learn that DEEP produces highest quality
            quality_insight = next(
                (i for i in insights if i.insight_type == "optimal_quality_depth"),
                None
            )

            test3_pass = (
                quality_insight is not None and
                "DEEP" in quality_insight.description
            )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test3_pass = False

    test_results["insight_extraction"] = test3_pass
    print(f"{'PASS' if test3_pass else 'FAIL'}: TEST 3")
    print()
    all_tests_passed = all_tests_passed and test3_pass

    # ========================================================================
    # TEST 4: Learned Depth Preference
    # ========================================================================
    print("=" * 72)
    print("TEST 4: Learned Depth Preference Selection")
    print("=" * 72)
    print()

    print("Testing learned depth preference...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PersistentMetaLearningManager(
                storage_path=Path(tmpdir),
                node_id="preference_test"
            )

            # STANDARD consistently best at high ATP
            for i in range(10):
                manager.record_pattern(
                    depth_used=CognitiveDepth.STANDARD,
                    atp_before=100.0, atp_after=80.0,
                    reputation_before=20.0, reputation_after=21.0,
                    quality_achieved=0.85,
                    success=True,
                    context={}
                )

            # LIGHT decent but lower quality
            for i in range(5):
                manager.record_pattern(
                    depth_used=CognitiveDepth.LIGHT,
                    atp_before=100.0, atp_after=91.0,
                    reputation_before=20.0, reputation_after=19.5,
                    quality_achieved=0.6,
                    success=True,
                    context={}
                )

            # Get learned preference at different ATP levels
            pref_high = manager.get_learned_depth_preference(current_atp=100.0)
            pref_low = manager.get_learned_depth_preference(current_atp=10.0)

            print(f"  At 100 ATP: Learned preference = {pref_high.name if pref_high else 'None'}")
            print(f"  At 10 ATP: Learned preference = {pref_low.name if pref_low else 'None'}")
            print()

            # Should prefer STANDARD at high ATP (better quality)
            test4_pass = (
                pref_high == CognitiveDepth.STANDARD and
                (pref_low is None or pref_low == CognitiveDepth.LIGHT)
            )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        test4_pass = False

    test_results["learned_preference"] = test4_pass
    print(f"{'PASS' if test4_pass else 'FAIL'}: TEST 4")
    print()
    all_tests_passed = all_tests_passed and test4_pass

    # ========================================================================
    # TEST 5: Cross-Session Learning Persistence
    # ========================================================================
    print("=" * 72)
    print("TEST 5: Cross-Session Learning Persistence")
    print("=" * 72)
    print()

    print("Testing learning persistence across sessions...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Session 1: Record patterns
            mgr1 = PersistentMetaLearningManager(
                storage_path=Path(tmpdir),
                node_id="persist_test"
            )

            for i in range(8):
                mgr1.record_pattern(
                    depth_used=CognitiveDepth.DEEP,
                    atp_before=100.0, atp_after=65.0,
                    reputation_before=30.0, reputation_after=32.0,
                    quality_achieved=0.88,
                    success=True,
                    context={"session": 1}
                )

            patterns_s1 = len(mgr1.patterns)
            insights_s1 = len(mgr1.analyze_patterns())
            print(f"  Session 1: {patterns_s1} patterns, {insights_s1} insights")

            # Session 2: Reload and continue
            mgr2 = PersistentMetaLearningManager(
                storage_path=Path(tmpdir),
                node_id="persist_test"
            )

            patterns_s2_loaded = len(mgr2.patterns)

            for i in range(4):
                mgr2.record_pattern(
                    depth_used=CognitiveDepth.STANDARD,
                    atp_before=100.0, atp_after=80.0,
                    reputation_before=32.0, reputation_after=32.5,
                    quality_achieved=0.72,
                    success=True,
                    context={"session": 2}
                )

            patterns_s2_final = len(mgr2.patterns)
            print(f"  Session 2 (loaded): {patterns_s2_loaded} patterns")
            print(f"  Session 2 (after new): {patterns_s2_final} patterns")

            # Session 3: Verify cumulative
            mgr3 = PersistentMetaLearningManager(
                storage_path=Path(tmpdir),
                node_id="persist_test"
            )

            patterns_s3 = len(mgr3.patterns)
            print(f"  Session 3 (recovered): {patterns_s3} patterns")
            print()

            test5_pass = (
                patterns_s2_loaded == 8 and
                patterns_s3 == 12 and
                patterns_s2_final == 12
            )

    except Exception as e:
        print(f"  ERROR: {e}")
        test5_pass = False

    test_results["cross_session_persistence"] = test5_pass
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

    print("Profiling meta-learning operations on edge...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PersistentMetaLearningManager(
                storage_path=Path(tmpdir),
                node_id="perf_test"
            )

            iterations = 100

            # Pattern recording performance
            start = time.time()
            for i in range(iterations):
                manager.record_pattern(
                    depth_used=CognitiveDepth.STANDARD,
                    atp_before=100.0, atp_after=80.0,
                    reputation_before=20.0, reputation_after=21.0,
                    quality_achieved=0.75,
                    success=True,
                    context={"i": i}
                )
            record_time = time.time() - start

            record_ops_per_sec = iterations / record_time

            # Preference calculation performance
            start = time.time()
            for _ in range(iterations):
                _ = manager.get_learned_depth_preference(current_atp=100.0)
            pref_time = time.time() - start

            pref_ops_per_sec = iterations / pref_time

            # Analysis performance
            start = time.time()
            for _ in range(10):
                _ = manager.analyze_patterns()
            analyze_time = time.time() - start

            analyze_per_sec = 10 / analyze_time

            print(f"  Pattern recording: {record_ops_per_sec:,.0f} ops/sec")
            print(f"  Preference lookup: {pref_ops_per_sec:,.0f} ops/sec")
            print(f"  Pattern analysis: {analyze_per_sec:,.1f} ops/sec")
            print()

            # Edge should handle at least 100 pattern records/sec
            test6_pass = record_ops_per_sec > 100 and pref_ops_per_sec > 1000

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
    print("SESSION 181 EDGE VALIDATION SUMMARY")
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
        print("|" + "  META-LEARNING ADAPTIVE DEPTH VALIDATED ON EDGE!  ".center(70) + "|")
        print("|" + " " * 70 + "|")
        print("+" + "-" * 70 + "+")
        print()
        print("Edge Observations:")
        print("  - Pattern storage I/O fast on ARM64")
        print("  - Learning insights correctly extracted from patterns")
        print("  - Learned depth preference reflects training data")
        print("  - Cross-session learning persistence working")
        print("  - Performance adequate for edge deployment")
        print()
        print("Architecture Stack Validated:")
        print("  Session 177: ATP-adaptive depth")
        print("  Session 178: Federated coordination")
        print("  Session 179: Reputation-aware depth")
        print("  Session 180: Persistent reputation")
        print("  Session 181: Meta-learning depth (NOW VALIDATED)")
    else:
        print("SOME TESTS FAILED - Review results above")

    print()

    # Save results
    results = {
        "session": "181_edge",
        "title": "Meta-Learning Adaptive Depth - Edge Validation",
        "platform": "Sprout (Jetson Orin Nano 8GB)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "all_tests_passed": all_tests_passed,
        "test_results": test_results,
        "edge_metrics": edge_metrics,
        "convergent_research": {
            "session_181": "Meta-Learning Adaptive Depth",
            "legion_session_160": "Meta-Learning Patterns",
            "edge_validation": "Self-optimizing consciousness validated"
        }
    }

    results_path = Path(__file__).parent / "session181_edge_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return all_tests_passed


if __name__ == "__main__":
    success = test_edge_meta_learning()
    sys.exit(0 if success else 1)
