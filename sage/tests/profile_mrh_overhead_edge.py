#!/usr/bin/env python3
"""
MRH Overhead Profiling - Edge Hardware

Test MRH computation overhead on Jetson Orin Nano 8GB.

Questions:
1. Does MRH inference add measurable latency on edge hardware?
2. Is MRH similarity computation fast enough for real-time selection?
3. Does ARM64 affect MRH computation performance?

Hypothesis: MRH overhead is negligible (<1ms) even on edge hardware.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from typing import Dict, List
from statistics import mean, stdev

from core.mrh_utils import (
    compute_mrh_similarity,
    compute_mrh_distance,
    infer_situation_mrh,
    select_plugin_with_mrh,
    format_mrh
)


def benchmark_function(func, *args, iterations: int = 1000):
    """Benchmark function execution time"""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return {
        'mean': mean(times),
        'stdev': stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
        'result': result
    }


def test_mrh_similarity_computation():
    """Test MRH similarity computation speed"""
    print("\n" + "="*80)
    print("TEST 1: MRH Similarity Computation")
    print("="*80)

    # Test profiles
    mrh1 = {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}
    mrh2 = {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}

    stats = benchmark_function(compute_mrh_similarity, mrh1, mrh2, iterations=10000)

    print(f"\nSimilarity Computation (10,000 iterations):")
    print(f"  Mean:   {stats['mean']:.6f} ms")
    print(f"  Stdev:  {stats['stdev']:.6f} ms")
    print(f"  Min:    {stats['min']:.6f} ms")
    print(f"  Max:    {stats['max']:.6f} ms")
    print(f"  Result: {stats['result']:.2f}")

    # Threshold check
    if stats['mean'] < 0.01:  # <10 microseconds
        print("  ✓ PASS: Similarity computation is negligible (<10 μs)")
    elif stats['mean'] < 0.1:  # <100 microseconds
        print("  ✓ PASS: Similarity computation is very fast (<100 μs)")
    elif stats['mean'] < 1.0:  # <1ms
        print("  ⚠ ACCEPTABLE: Similarity computation is fast (<1 ms)")
    else:
        print("  ✗ SLOW: Similarity computation may impact real-time performance")

    return stats


def test_mrh_distance_computation():
    """Test MRH distance computation speed"""
    print("\n" + "="*80)
    print("TEST 2: MRH Distance Computation")
    print("="*80)

    # Test profiles with different distances
    mrh1 = {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}
    mrh2 = {'deltaR': 'global', 'deltaT': 'epoch', 'deltaC': 'society-scale'}

    stats = benchmark_function(compute_mrh_distance, mrh1, mrh2, iterations=10000)

    print(f"\nDistance Computation (10,000 iterations):")
    print(f"  Mean:   {stats['mean']:.6f} ms")
    print(f"  Stdev:  {stats['stdev']:.6f} ms")
    print(f"  Min:    {stats['min']:.6f} ms")
    print(f"  Max:    {stats['max']:.6f} ms")
    print(f"  Result: {stats['result']:.2f}")

    # Threshold check
    if stats['mean'] < 0.01:
        print("  ✓ PASS: Distance computation is negligible (<10 μs)")
    elif stats['mean'] < 0.1:
        print("  ✓ PASS: Distance computation is very fast (<100 μs)")
    elif stats['mean'] < 1.0:
        print("  ⚠ ACCEPTABLE: Distance computation is fast (<1 ms)")
    else:
        print("  ✗ SLOW: Distance computation may impact real-time performance")

    return stats


def test_mrh_inference():
    """Test MRH inference from query text"""
    print("\n" + "="*80)
    print("TEST 3: MRH Inference from Query Text")
    print("="*80)

    queries = [
        "hello",
        "what are we doing?",
        "analyze the relationship between consciousness and emergence",
        "search the web for latest developments",
        "what did we discuss yesterday?"
    ]

    total_time = 0
    results = []

    for query in queries:
        start = time.perf_counter()
        mrh = infer_situation_mrh(query)
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        total_time += elapsed_ms

        results.append({
            'query': query,
            'mrh': mrh,
            'time_ms': elapsed_ms
        })

        print(f"\nQuery: '{query[:50]}...'")
        print(f"  Inferred MRH: {format_mrh(mrh)}")
        print(f"  Time: {elapsed_ms:.6f} ms")

    avg_time = total_time / len(queries)
    print(f"\nAverage inference time: {avg_time:.6f} ms")

    # Threshold check
    if avg_time < 0.1:
        print("  ✓ PASS: MRH inference is very fast (<100 μs)")
    elif avg_time < 1.0:
        print("  ✓ PASS: MRH inference is fast (<1 ms)")
    elif avg_time < 10.0:
        print("  ⚠ ACCEPTABLE: MRH inference is acceptable (<10 ms)")
    else:
        print("  ✗ SLOW: MRH inference may impact conversation latency")

    return results


def test_plugin_selection_overhead():
    """Test MRH-aware plugin selection overhead"""
    print("\n" + "="*80)
    print("TEST 4: Plugin Selection Overhead")
    print("="*80)

    # Mock plugins
    class MockPlugin:
        def __init__(self, name, mrh, trust, cost):
            self.name = name
            self.mrh = mrh
            self.trust = trust
            self.cost = cost

        def get_mrh_profile(self):
            return self.mrh

        def get_trust_score(self):
            return self.trust

        def get_atp_cost(self):
            return self.cost

    plugins = [
        ('pattern', MockPlugin('pattern', {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}, 0.95, 1)),
        ('qwen-0.5b', MockPlugin('qwen-0.5b', {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}, 0.75, 30)),
        ('gpt-4', MockPlugin('gpt-4', {'deltaR': 'global', 'deltaT': 'session', 'deltaC': 'agent-scale'}, 0.90, 100))
    ]

    trust_scores = {name: p.trust for name, p in plugins}
    atp_costs = {name: p.cost for name, p in plugins}

    # Test simple query
    simple_query = "hello"

    # Baseline: trust-only selection
    start_baseline = time.perf_counter()
    for _ in range(1000):
        best = max(plugins, key=lambda item: item[1].trust / item[1].cost)
    end_baseline = time.perf_counter()
    baseline_time = (end_baseline - start_baseline) / 1000 * 1000  # ms per iteration

    # MRH-aware selection
    start_mrh = time.perf_counter()
    for _ in range(1000):
        best = select_plugin_with_mrh(simple_query, plugins, trust_scores, atp_costs)
    end_mrh = time.perf_counter()
    mrh_time = (end_mrh - start_mrh) / 1000 * 1000  # ms per iteration

    overhead = mrh_time - baseline_time
    overhead_pct = (overhead / baseline_time) * 100 if baseline_time > 0 else 0

    print(f"\nPlugin Selection (1,000 iterations):")
    print(f"  Baseline (trust/cost):     {baseline_time:.6f} ms")
    print(f"  MRH-aware:                 {mrh_time:.6f} ms")
    print(f"  Overhead:                  {overhead:.6f} ms ({overhead_pct:.1f}%)")

    # Threshold check
    if overhead < 0.01:
        print("  ✓ PASS: MRH overhead negligible (<10 μs)")
    elif overhead < 0.1:
        print("  ✓ PASS: MRH overhead very small (<100 μs)")
    elif overhead < 1.0:
        print("  ✓ PASS: MRH overhead acceptable (<1 ms)")
    else:
        print("  ⚠ WARNING: MRH overhead may impact plugin switching latency")

    return {
        'baseline_ms': baseline_time,
        'mrh_ms': mrh_time,
        'overhead_ms': overhead,
        'overhead_pct': overhead_pct
    }


def test_conversation_throughput():
    """Test MRH impact on conversation throughput"""
    print("\n" + "="*80)
    print("TEST 5: Conversation Throughput Impact")
    print("="*80)

    # Simulate 100-turn conversation
    queries = [
        "hello",
        "how are you?",
        "what can you do?",
        "tell me about consciousness",
        "explain IRP"
    ] * 20  # 100 queries

    # Baseline: direct plugin selection (no MRH)
    start_baseline = time.perf_counter()
    for query in queries:
        # Simulate simple pattern check
        is_simple = len(query.split()) < 5
    end_baseline = time.perf_counter()
    baseline_time = (end_baseline - start_baseline) * 1000

    # MRH-aware: infer MRH for each query
    start_mrh = time.perf_counter()
    for query in queries:
        mrh = infer_situation_mrh(query)
        # Simulate simple comparison
        is_simple = mrh['deltaC'] == 'simple'
    end_mrh = time.perf_counter()
    mrh_time = (end_mrh - start_baseline) * 1000

    overhead_total = mrh_time - baseline_time
    overhead_per_turn = overhead_total / len(queries)

    print(f"\n100-Turn Conversation Simulation:")
    print(f"  Baseline total time:       {baseline_time:.3f} ms")
    print(f"  MRH-aware total time:      {mrh_time:.3f} ms")
    print(f"  Total overhead:            {overhead_total:.3f} ms")
    print(f"  Overhead per turn:         {overhead_per_turn:.6f} ms")

    # In context of typical IRP time (30s on Jetson)
    irp_time_ms = 30000  # 30 seconds
    overhead_vs_irp = (overhead_per_turn / irp_time_ms) * 100

    print(f"\nContext (vs typical 30s IRP on Jetson):")
    print(f"  MRH overhead: {overhead_per_turn:.6f} ms")
    print(f"  IRP time:     {irp_time_ms} ms")
    print(f"  Overhead:     {overhead_vs_irp:.6f}% of IRP time")

    if overhead_vs_irp < 0.001:
        print("  ✓ PASS: MRH overhead completely negligible")
    elif overhead_vs_irp < 0.01:
        print("  ✓ PASS: MRH overhead negligible")
    elif overhead_vs_irp < 0.1:
        print("  ✓ PASS: MRH overhead very small")
    else:
        print("  ⚠ WARNING: MRH overhead may be noticeable")

    return {
        'baseline_ms': baseline_time,
        'mrh_ms': mrh_time,
        'overhead_total_ms': overhead_total,
        'overhead_per_turn_ms': overhead_per_turn,
        'overhead_vs_irp_pct': overhead_vs_irp
    }


def main():
    """Run all MRH overhead profiling tests"""
    print("="*80)
    print("MRH OVERHEAD PROFILING - EDGE HARDWARE")
    print("="*80)
    print("\nHardware: Jetson Orin Nano 8GB (ARM64)")
    print("Platform: Linux")
    print("Purpose: Validate MRH computation performance on edge")

    results = {}

    # Run tests
    results['similarity'] = test_mrh_similarity_computation()
    results['distance'] = test_mrh_distance_computation()
    results['inference'] = test_mrh_inference()
    results['selection'] = test_plugin_selection_overhead()
    results['throughput'] = test_conversation_throughput()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: MRH PERFORMANCE ON EDGE")
    print("="*80)

    print(f"\nMRH Similarity:     {results['similarity']['mean']:.6f} ms (mean)")
    print(f"MRH Distance:       {results['distance']['mean']:.6f} ms (mean)")
    print(f"Plugin Selection:   +{results['selection']['overhead_ms']:.6f} ms overhead")
    print(f"Per-turn overhead:  {results['throughput']['overhead_per_turn_ms']:.6f} ms")
    print(f"vs IRP time (30s):  {results['throughput']['overhead_vs_irp_pct']:.6f}%")

    # Final verdict
    print("\n" + "="*80)
    print("EDGE VALIDATION VERDICT")
    print("="*80)

    total_overhead = results['selection']['overhead_ms']

    if total_overhead < 0.1:
        print("\n✓ MRH overhead is NEGLIGIBLE on edge hardware")
        print("  Safe for real-time conversation use")
        print("  No performance concerns for Jetson deployment")
    elif total_overhead < 1.0:
        print("\n✓ MRH overhead is ACCEPTABLE on edge hardware")
        print("  Minimal impact on conversation latency")
        print("  Safe for production deployment")
    else:
        print("\n⚠ MRH overhead is MEASURABLE on edge hardware")
        print("  May impact plugin switching latency")
        print("  Consider optimization for production")

    print("\n" + "="*80)

    return results


if __name__ == '__main__':
    main()
