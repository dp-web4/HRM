#!/usr/bin/env python3
"""
MRH Inference Testing - Edge Validation

Test Thor's improved MRH inference with temporal keyword detection on ARM64.

Validates:
1. Temporal keyword detection (yesterday, this month, etc.)
2. Spatial auto-adjustment from temporal
3. Complexity inference from query structure
4. Edge-specific performance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from core.mrh_utils import infer_situation_mrh, format_mrh


# Test queries spanning different horizons
TEST_QUERIES = [
    # Session horizon (ephemeral/session)
    ("Hello!", {"expected_T": "ephemeral", "expected_R": "local", "expected_C": "simple"}),
    ("What did we just discuss?", {"expected_T": "session", "expected_R": "local", "expected_C": "simple"}),
    ("What are we doing?", {"expected_T": "session", "expected_R": "local", "expected_C": "agent-scale"}),

    # Day horizon (day, regional)
    ("What did we accomplish yesterday?", {"expected_T": "day", "expected_R": "regional", "expected_C": "agent-scale"}),
    ("What happened earlier today?", {"expected_T": "day", "expected_R": "regional", "expected_C": "agent-scale"}),
    ("What did we complete this morning?", {"expected_T": "day", "expected_R": "regional", "expected_C": "agent-scale"}),
    ("Show me what I did last night", {"expected_T": "day", "expected_R": "regional", "expected_C": "agent-scale"}),

    # Epoch horizon (epoch, regional)
    ("What patterns emerged this month?", {"expected_T": "epoch", "expected_R": "regional", "expected_C": "society-scale"}),
    ("What have we learned over time?", {"expected_T": "epoch", "expected_R": "regional", "expected_C": "society-scale"}),
    ("Show me long-term trends", {"expected_T": "epoch", "expected_R": "regional", "expected_C": "society-scale"}),
    ("What patterns emerged this year?", {"expected_T": "epoch", "expected_R": "regional", "expected_C": "society-scale"}),

    # Global spatial extent
    ("Search the web for latest developments", {"expected_T": "session", "expected_R": "global", "expected_C": "agent-scale"}),
    ("Find online information about X", {"expected_T": "ephemeral", "expected_R": "global", "expected_C": "simple"}),
]


def test_temporal_keyword_detection():
    """Test that temporal keywords correctly identify horizons"""
    print("="*80)
    print("TEST 1: Temporal Keyword Detection")
    print("="*80)

    passed = 0
    failed = 0

    for query, expected in TEST_QUERIES:
        start = time.perf_counter()
        mrh = infer_situation_mrh(query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Check temporal match
        temporal_match = mrh['deltaT'] == expected['expected_T']
        spatial_match = mrh['deltaR'] == expected['expected_R']
        complexity_match = mrh['deltaC'] == expected['expected_C']

        all_match = temporal_match and spatial_match and complexity_match

        if all_match:
            passed += 1
            status = "✓"
        else:
            failed += 1
            status = "✗"

        print(f"\n{status} Query: '{query[:60]}'")
        print(f"  Expected: {format_mrh(expected)}")
        print(f"  Inferred: {format_mrh(mrh)}")
        print(f"  Time: {elapsed_ms:.3f} ms")

        if not temporal_match:
            print(f"  ⚠ Temporal mismatch: expected {expected['expected_T']}, got {mrh['deltaT']}")
        if not spatial_match:
            print(f"  ⚠ Spatial mismatch: expected {expected['expected_R']}, got {mrh['deltaR']}")
        if not complexity_match:
            print(f"  ⚠ Complexity mismatch: expected {expected['expected_C']}, got {mrh['deltaC']}")

    print(f"\n{'='*80}")
    print(f"RESULTS: {passed}/{len(TEST_QUERIES)} passed ({passed/len(TEST_QUERIES)*100:.1f}%)")
    print(f"{'='*80}")

    return passed, failed


def test_spatial_auto_adjustment():
    """Test that day/epoch queries auto-adjust spatial extent to regional"""
    print("\n" + "="*80)
    print("TEST 2: Spatial Auto-Adjustment (Day/Epoch → Regional)")
    print("="*80)

    test_cases = [
        ("What did we do yesterday?", "day", "regional"),
        ("Show me this month's patterns", "epoch", "regional"),
        ("What happened today?", "day", "regional"),
        ("Long-term trends in the data", "epoch", "regional"),
    ]

    passed = 0
    failed = 0

    for query, expected_t, expected_r in test_cases:
        mrh = infer_situation_mrh(query)

        temporal_match = mrh['deltaT'] == expected_t
        spatial_match = mrh['deltaR'] == expected_r

        if temporal_match and spatial_match:
            passed += 1
            status = "✓"
        else:
            failed += 1
            status = "✗"

        print(f"\n{status} Query: '{query}'")
        print(f"  Expected: ΔT={expected_t}, ΔR={expected_r}")
        print(f"  Inferred: ΔT={mrh['deltaT']}, ΔR={mrh['deltaR']}")

        if not spatial_match:
            print(f"  ⚠ Auto-adjustment failed: {expected_t} should→{expected_r}, got {mrh['deltaR']}")

    print(f"\n{'='*80}")
    print(f"RESULTS: {passed}/{len(test_cases)} passed ({passed/len(test_cases)*100:.1f}%)")
    print(f"{'='*80}")

    return passed, failed


def test_edge_performance():
    """Test inference performance on ARM64"""
    print("\n" + "="*80)
    print("TEST 3: Edge Performance (ARM64)")
    print("="*80)

    # Benchmark on all test queries
    times = []

    for query, _ in TEST_QUERIES:
        start = time.perf_counter()
        mrh = infer_situation_mrh(query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\nInference Performance ({len(TEST_QUERIES)} queries):")
    print(f"  Average: {avg_time:.6f} ms")
    print(f"  Min:     {min_time:.6f} ms")
    print(f"  Max:     {max_time:.6f} ms")

    # Compare to Session 10 baseline (14 μs = 0.014 ms)
    baseline_ms = 0.014
    slowdown = avg_time / baseline_ms

    print(f"\nComparison to Session 10 baseline:")
    print(f"  Baseline (simple queries): {baseline_ms:.6f} ms")
    print(f"  Current (with keywords):   {avg_time:.6f} ms")
    print(f"  Slowdown factor:           {slowdown:.2f}×")

    # Verdict
    if avg_time < 0.1:
        print(f"\n  ✓ PASS: Still very fast (<100 μs)")
    elif avg_time < 1.0:
        print(f"\n  ✓ PASS: Still fast (<1 ms)")
    else:
        print(f"\n  ⚠ WARNING: Slower than expected")

    return avg_time


def test_keyword_coverage():
    """Test coverage of temporal keywords"""
    print("\n" + "="*80)
    print("TEST 4: Temporal Keyword Coverage")
    print("="*80)

    keyword_tests = {
        "Day keywords": [
            "yesterday",
            "earlier today",
            "this morning",
            "last night",
            "today",
        ],
        "Epoch keywords": [
            "this month",
            "this year",
            "over time",
            "patterns emerged",
            "long-term trends",
        ],
        "Session keywords": [
            "just now",
            "what we discussed",
            "current conversation",
        ]
    }

    for category, keywords in keyword_tests.items():
        print(f"\n{category}:")
        for keyword in keywords:
            query = f"Show me {keyword}"
            mrh = infer_situation_mrh(query)

            expected_t = "day" if "Day" in category else ("epoch" if "Epoch" in category else "session")
            match = mrh['deltaT'] == expected_t

            status = "✓" if match else "✗"
            print(f"  {status} '{keyword}' → ΔT={mrh['deltaT']} (expected: {expected_t})")


def main():
    """Run all MRH inference tests"""
    print("="*80)
    print("MRH INFERENCE TESTING - EDGE VALIDATION")
    print("="*80)
    print("\nHardware: Jetson Orin Nano 8GB (ARM64)")
    print("Purpose: Validate Thor's improved MRH inference on edge")
    print(f"\nTest suite: {len(TEST_QUERIES)} queries across 3 temporal horizons")

    # Run tests
    passed_1, failed_1 = test_temporal_keyword_detection()
    passed_2, failed_2 = test_spatial_auto_adjustment()
    avg_time = test_edge_performance()
    test_keyword_coverage()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: MRH INFERENCE ON EDGE")
    print("="*80)

    total_passed = passed_1 + passed_2
    total_tests = len(TEST_QUERIES) + 4  # 4 spatial auto-adjustment tests

    print(f"\nAccuracy: {passed_1}/{len(TEST_QUERIES)} queries correct ({passed_1/len(TEST_QUERIES)*100:.1f}%)")
    print(f"Spatial auto-adjust: {passed_2}/4 correct ({passed_2/4*100:.1f}%)")
    print(f"Average inference time: {avg_time:.6f} ms")

    # Verdict
    print("\n" + "="*80)
    print("EDGE VALIDATION VERDICT")
    print("="*80)

    if passed_1 == len(TEST_QUERIES) and avg_time < 0.1:
        print("\n✓ MRH inference works PERFECTLY on edge")
        print("  Temporal keywords detected correctly")
        print("  Spatial auto-adjustment working")
        print("  Performance excellent (<100 μs)")
    elif passed_1 >= len(TEST_QUERIES) * 0.8:
        print("\n✓ MRH inference works WELL on edge")
        print(f"  {passed_1}/{len(TEST_QUERIES)} queries correct")
        print(f"  Performance: {avg_time:.3f} ms")
        print("  Minor tuning may improve accuracy")
    else:
        print("\n⚠ MRH inference needs TUNING for edge")
        print(f"  Only {passed_1}/{len(TEST_QUERIES)} queries correct")
        print("  Temporal keyword detection may need adjustment")

    print("\n" + "="*80)

    return passed_1, failed_1, avg_time


if __name__ == '__main__':
    main()
