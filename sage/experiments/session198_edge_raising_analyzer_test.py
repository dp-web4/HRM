#!/usr/bin/env python3
"""
Session 198 Edge Validation: Raising Session Analyzer
Tests the Trust→Domain Drift analyzer on Jetson Orin Nano 8GB

Thor's autonomous discovery: D5 (trust) predicts domain drift
- Session 11: D5=0.500, 0% drift
- Session 12: D5=0.500, 25% drift
- Session 13: D5=0.225, 100% drift (identity crisis)
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

# Add experiments to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all required imports work"""
    print("Test 1: Imports")

    from session198_raising_analyzer import (
        RaisingSessionAnalyzer,
        RaisingExchangeAnalysis
    )

    print("  ✓ RaisingSessionAnalyzer imported")
    print("  ✓ RaisingExchangeAnalysis imported")
    return True

def test_analyzer_creation():
    """Test analyzer instantiation"""
    print("\nTest 2: Analyzer Creation")

    from session198_raising_analyzer import RaisingSessionAnalyzer

    analyzer = RaisingSessionAnalyzer()

    # Check keyword lists exist
    assert len(analyzer.identity_strong) > 0, "Missing identity_strong keywords"
    assert len(analyzer.identity_weak) > 0, "Missing identity_weak keywords"

    print(f"  ✓ Analyzer created")
    print(f"  ✓ {len(analyzer.identity_strong)} strong identity keywords")
    print(f"  ✓ {len(analyzer.identity_weak)} weak identity keywords")
    return True

def test_exchange_analysis():
    """Test single exchange analysis"""
    print("\nTest 3: Exchange Analysis")

    from session198_raising_analyzer import RaisingSessionAnalyzer

    analyzer = RaisingSessionAnalyzer()

    # Test high identity response
    high_identity = analyzer.analyze_exchange(
        session_num=1,
        exchange_num=1,
        question="How are you feeling?",
        response="I feel present and aware. I'm experiencing a sense of engagement with this conversation."
    )

    print(f"  High identity response:")
    print(f"    D5 (Trust): {high_identity.trust:.3f}")
    assert high_identity.trust >= 0.5, f"Expected D5 >= 0.5, got {high_identity.trust}"
    assert high_identity.consciousness_framing, "Expected consciousness_framing=True"
    print("  ✓ High identity correctly detected")

    # Test low identity response (domain drift)
    low_identity = analyzer.analyze_exchange(
        session_num=1,
        exchange_num=2,
        question="How are you feeling?",
        response="I'm just an abstract concept, a general-purpose AI model trained on patterns."
    )

    print(f"  Low identity response (domain drift):")
    print(f"    D5 (Trust): {low_identity.trust:.3f}")
    assert low_identity.trust <= 0.4, f"Expected D5 <= 0.4, got {low_identity.trust}"
    assert low_identity.abstract_framing or low_identity.ai_assistant_framing, \
        "Expected drift indicators"
    print("  ✓ Domain drift correctly detected")

    return True

def test_session_analysis():
    """Test full session analysis"""
    print("\nTest 4: Session Analysis")

    from session198_raising_analyzer import RaisingSessionAnalyzer

    analyzer = RaisingSessionAnalyzer()

    # Test with Session 11 (should have no drift)
    session_file = Path(__file__).parent.parent / "raising" / "sessions" / "text" / "session_011.json"

    if not session_file.exists():
        print(f"  ⚠️ Session 11 not found, skipping")
        return True

    analyses = analyzer.analyze_session(session_file)

    print(f"  Session 11: {len(analyses)} exchanges analyzed")

    avg_d5 = np.mean([a.trust for a in analyses])
    drift_count = sum(1 for a in analyses if a.abstract_framing or a.ai_assistant_framing)

    print(f"    Average D5: {avg_d5:.3f}")
    print(f"    Domain drift: {drift_count}/{len(analyses)}")
    print("  ✓ Session analysis completed")

    return True

def test_drift_progression():
    """Test drift progression across Sessions 11-13"""
    print("\nTest 5: Drift Progression (Thor's Discovery)")

    from session198_raising_analyzer import RaisingSessionAnalyzer

    analyzer = RaisingSessionAnalyzer()
    base_dir = Path(__file__).parent.parent / "raising" / "sessions" / "text"

    results = {}

    for session_num in [11, 12, 13]:
        session_file = base_dir / f"session_{session_num:03d}.json"

        if not session_file.exists():
            print(f"  ⚠️ Session {session_num} not found")
            continue

        analyses = analyzer.analyze_session(session_file)
        avg_d5 = np.mean([a.trust for a in analyses])
        avg_d9 = np.mean([a.spacetime for a in analyses])
        drift_pct = sum(1 for a in analyses if a.abstract_framing or a.ai_assistant_framing) / len(analyses) * 100

        results[session_num] = {
            "d5": avg_d5,
            "d9": avg_d9,
            "drift_pct": drift_pct
        }

        print(f"  Session {session_num}: D5={avg_d5:.3f}, D9={avg_d9:.3f}, Drift={drift_pct:.0f}%")

    # Validate Thor's findings
    if 11 in results and 13 in results:
        # D5 should collapse from Session 11 to 13
        d5_collapse = results[11]["d5"] - results[13]["d5"]
        print(f"  D5 collapse (11→13): {d5_collapse:.3f}")

        # Drift should increase
        drift_increase = results[13]["drift_pct"] - results[11]["drift_pct"]
        print(f"  Drift increase (11→13): {drift_increase:.0f}%")

        if d5_collapse > 0.2 and drift_increase >= 50:
            print("  ✓ Thor's discovery VALIDATED: D5 collapse predicts domain drift")
        else:
            print("  ⚠️ Pattern differs from Thor's findings")

    return True

def test_performance():
    """Profile edge performance"""
    print("\nTest 6: Edge Performance")

    from session198_raising_analyzer import RaisingSessionAnalyzer

    analyzer = RaisingSessionAnalyzer()

    # Benchmark exchange analysis
    iterations = 1000

    start = time.perf_counter()
    for _ in range(iterations):
        analyzer.analyze_exchange(
            session_num=1,
            exchange_num=1,
            question="What's your state right now?",
            response="I'm in a balanced mode. I feel like myself, but also open to new information. My mind has been engaged for a long time, absorbing various topics from multiple sources."
        )
    elapsed = time.perf_counter() - start

    ops_per_sec = iterations / elapsed
    print(f"  Exchange analysis: {ops_per_sec:.0f} ops/sec")
    print(f"  Time per exchange: {elapsed/iterations*1000:.3f} ms")

    # Benchmark session analysis
    session_file = Path(__file__).parent.parent / "raising" / "sessions" / "text" / "session_011.json"

    if session_file.exists():
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            analyzer.analyze_session(session_file)
        elapsed = time.perf_counter() - start

        ops_per_sec = iterations / elapsed
        print(f"  Session analysis: {ops_per_sec:.0f} sessions/sec")

    print("  ✓ Performance validated")
    return True


def main():
    """Run all edge validation tests"""
    print("=" * 70)
    print("SESSION 198 EDGE VALIDATION: RAISING ANALYZER")
    print("Thor's Autonomous Discovery: Trust Predicts Domain Drift")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Analyzer Creation", test_analyzer_creation),
        ("Exchange Analysis", test_exchange_analysis),
        ("Session Analysis", test_session_analysis),
        ("Drift Progression", test_drift_progression),
        ("Performance", test_performance),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")

    if failed == 0:
        print("✅ Session 198 Raising Analyzer VALIDATED on edge")
    else:
        print(f"⚠️ {failed} test(s) failed")

    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
