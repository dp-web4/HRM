#!/usr/bin/env python3
"""
Session 142: Five-Domain EP Coordinator Performance Benchmark

Validates performance of Thor's five-domain Multi-EP Coordinator across
realistic scenarios, providing baseline metrics for production readiness.

Benchmarks:
1. Consensus (all EPs agree)
2. Conflict resolution (EPs disagree, priority-based)
3. Cascade detection (multiple severe predictions)
4. Mixed scenarios (realistic combinations)

Comparison to Web4:
- Web4 (Legion RTX 4090): 280K decisions/sec, 3.46μs latency
- Thor (Jetson AGX): To be measured
- Expected: ~100-150K decisions/sec (Thor has more compute than Sprout's 97K)

Created: 2025-12-31 (Session 142)
Hardware: Thor (Jetson AGX Thor Developer Kit)
"""

import sys
from pathlib import Path
import time
from dataclasses import dataclass
from typing import List, Dict
import json
from datetime import datetime

# Add SAGE modules to path
sys.path.append(str(Path(__file__).parent.parent))

# Import Multi-EP Coordinator
from experiments.multi_ep_coordinator import (
    EPDomain, EPPrediction, MultiEPCoordinator, MultiEPDecision
)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    test_name: str
    iterations: int
    total_time_seconds: float
    throughput_per_second: float
    avg_latency_ms: float
    avg_latency_us: float
    min_latency_ms: float
    max_latency_ms: float
    notes: str


def create_test_prediction(
    domain: EPDomain,
    outcome_prob: float = 0.8,
    severity: float = 0.2,
    recommendation: str = "proceed"
) -> EPPrediction:
    """Create a test EP prediction."""
    return EPPrediction(
        domain=domain,
        outcome_probability=outcome_prob,
        confidence=0.9,
        severity=severity,
        recommendation=recommendation,
        reasoning=f"Test prediction for {domain.value}",
        adjustment_strategy=None
    )


def benchmark_consensus(coordinator: MultiEPCoordinator, iterations: int = 100000) -> BenchmarkResult:
    """
    Benchmark: All EPs agree (consensus scenario).
    
    All 5 EPs recommend "proceed" with low severity.
    This is the happy path - no conflicts or cascades.
    """
    # Create consistent predictions
    emotional_pred = create_test_prediction(EPDomain.EMOTIONAL, 0.9, 0.1, "proceed")
    quality_pred = create_test_prediction(EPDomain.QUALITY, 0.9, 0.1, "proceed")
    attention_pred = create_test_prediction(EPDomain.ATTENTION, 0.9, 0.1, "proceed")
    grounding_pred = create_test_prediction(EPDomain.GROUNDING, 0.9, 0.1, "proceed")
    authorization_pred = create_test_prediction(EPDomain.AUTHORIZATION, 0.9, 0.1, "proceed")
    
    latencies = []
    
    start_time = time.time()
    
    for _ in range(iterations):
        iter_start = time.time()
        
        decision = coordinator.coordinate(
            emotional_pred=emotional_pred,
            quality_pred=quality_pred,
            attention_pred=attention_pred,
            grounding_pred=grounding_pred,
            authorization_pred=authorization_pred
        )
        
        iter_end = time.time()
        latencies.append((iter_end - iter_start) * 1000)  # ms
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return BenchmarkResult(
        test_name="Consensus (all proceed)",
        iterations=iterations,
        total_time_seconds=total_time,
        throughput_per_second=iterations / total_time,
        avg_latency_ms=sum(latencies) / len(latencies),
        avg_latency_us=(sum(latencies) / len(latencies)) * 1000,
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        notes="All 5 EPs agree to proceed - happy path"
    )


def benchmark_conflict_resolution(coordinator: MultiEPCoordinator, iterations: int = 100000) -> BenchmarkResult:
    """
    Benchmark: Conflict resolution.
    
    Emotional EP says "defer", others say "proceed".
    Should trigger priority-based resolution (Emotional wins).
    """
    emotional_pred = create_test_prediction(EPDomain.EMOTIONAL, 0.4, 0.8, "defer")
    quality_pred = create_test_prediction(EPDomain.QUALITY, 0.9, 0.2, "proceed")
    attention_pred = create_test_prediction(EPDomain.ATTENTION, 0.9, 0.2, "proceed")
    grounding_pred = create_test_prediction(EPDomain.GROUNDING, 0.9, 0.2, "proceed")
    authorization_pred = create_test_prediction(EPDomain.AUTHORIZATION, 0.9, 0.2, "proceed")
    
    latencies = []
    
    start_time = time.time()
    
    for _ in range(iterations):
        iter_start = time.time()
        
        decision = coordinator.coordinate(
            emotional_pred=emotional_pred,
            quality_pred=quality_pred,
            attention_pred=attention_pred,
            grounding_pred=grounding_pred,
            authorization_pred=authorization_pred
        )
        
        iter_end = time.time()
        latencies.append((iter_end - iter_start) * 1000)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return BenchmarkResult(
        test_name="Conflict Resolution",
        iterations=iterations,
        total_time_seconds=total_time,
        throughput_per_second=iterations / total_time,
        avg_latency_ms=sum(latencies) / len(latencies),
        avg_latency_us=(sum(latencies) / len(latencies)) * 1000,
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        notes="Emotional defers, others proceed - priority resolution"
    )


def benchmark_cascade_detection(coordinator: MultiEPCoordinator, iterations: int = 100000) -> BenchmarkResult:
    """
    Benchmark: Cascade detection.
    
    Emotional and Grounding both have severe predictions (>0.7 severity).
    Should trigger cascade detection logic.
    """
    emotional_pred = create_test_prediction(EPDomain.EMOTIONAL, 0.3, 0.9, "adjust")
    quality_pred = create_test_prediction(EPDomain.QUALITY, 0.9, 0.2, "proceed")
    attention_pred = create_test_prediction(EPDomain.ATTENTION, 0.9, 0.2, "proceed")
    grounding_pred = create_test_prediction(EPDomain.GROUNDING, 0.3, 0.8, "adjust")
    authorization_pred = create_test_prediction(EPDomain.AUTHORIZATION, 0.9, 0.2, "proceed")
    
    latencies = []
    
    start_time = time.time()
    
    for _ in range(iterations):
        iter_start = time.time()
        
        decision = coordinator.coordinate(
            emotional_pred=emotional_pred,
            quality_pred=quality_pred,
            attention_pred=attention_pred,
            grounding_pred=grounding_pred,
            authorization_pred=authorization_pred
        )
        
        iter_end = time.time()
        latencies.append((iter_end - iter_start) * 1000)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return BenchmarkResult(
        test_name="Cascade Detection",
        iterations=iterations,
        total_time_seconds=total_time,
        throughput_per_second=iterations / total_time,
        avg_latency_ms=sum(latencies) / len(latencies),
        avg_latency_us=(sum(latencies) / len(latencies)) * 1000,
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        notes="2 severe predictions (Emotional + Grounding) trigger cascade"
    )


def benchmark_mixed_scenarios(coordinator: MultiEPCoordinator, iterations: int = 50000) -> BenchmarkResult:
    """
    Benchmark: Mixed realistic scenarios.
    
    Rotates through 4 scenario types:
    1. All proceed (consensus)
    2. Emotional defers (conflict)
    3. Cascade (multiple severe)
    4. Authorization rejects (security issue)
    """
    scenarios = [
        # Scenario 1: Consensus
        (
            create_test_prediction(EPDomain.EMOTIONAL, 0.9, 0.2, "proceed"),
            create_test_prediction(EPDomain.QUALITY, 0.9, 0.2, "proceed"),
            create_test_prediction(EPDomain.ATTENTION, 0.9, 0.2, "proceed"),
            create_test_prediction(EPDomain.GROUNDING, 0.9, 0.2, "proceed"),
            create_test_prediction(EPDomain.AUTHORIZATION, 0.9, 0.2, "proceed"),
        ),
        # Scenario 2: Emotional conflict
        (
            create_test_prediction(EPDomain.EMOTIONAL, 0.3, 0.7, "defer"),
            create_test_prediction(EPDomain.QUALITY, 0.8, 0.3, "proceed"),
            create_test_prediction(EPDomain.ATTENTION, 0.8, 0.3, "proceed"),
            create_test_prediction(EPDomain.GROUNDING, 0.8, 0.3, "proceed"),
            create_test_prediction(EPDomain.AUTHORIZATION, 0.8, 0.3, "proceed"),
        ),
        # Scenario 3: Cascade
        (
            create_test_prediction(EPDomain.EMOTIONAL, 0.2, 0.9, "adjust"),
            create_test_prediction(EPDomain.QUALITY, 0.7, 0.5, "proceed"),
            create_test_prediction(EPDomain.ATTENTION, 0.7, 0.5, "proceed"),
            create_test_prediction(EPDomain.GROUNDING, 0.3, 0.8, "adjust"),
            create_test_prediction(EPDomain.AUTHORIZATION, 0.9, 0.2, "proceed"),
        ),
        # Scenario 4: Authorization rejects
        (
            create_test_prediction(EPDomain.EMOTIONAL, 0.8, 0.3, "proceed"),
            create_test_prediction(EPDomain.QUALITY, 0.8, 0.3, "proceed"),
            create_test_prediction(EPDomain.ATTENTION, 0.8, 0.3, "proceed"),
            create_test_prediction(EPDomain.GROUNDING, 0.7, 0.4, "proceed"),
            create_test_prediction(EPDomain.AUTHORIZATION, 0.2, 0.9, "defer"),
        ),
    ]
    
    latencies = []
    
    start_time = time.time()
    
    for i in range(iterations):
        scenario = scenarios[i % 4]
        
        iter_start = time.time()
        
        decision = coordinator.coordinate(
            emotional_pred=scenario[0],
            quality_pred=scenario[1],
            attention_pred=scenario[2],
            grounding_pred=scenario[3],
            authorization_pred=scenario[4]
        )
        
        iter_end = time.time()
        latencies.append((iter_end - iter_start) * 1000)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return BenchmarkResult(
        test_name="Mixed Scenarios",
        iterations=iterations,
        total_time_seconds=total_time,
        throughput_per_second=iterations / total_time,
        avg_latency_ms=sum(latencies) / len(latencies),
        avg_latency_us=(sum(latencies) / len(latencies)) * 1000,
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        notes="4 scenario types rotated (consensus, conflict, cascade, security reject)"
    )


def run_benchmark_suite():
    """Run complete benchmark suite."""
    print("=" * 70)
    print("Session 142: Five-Domain EP Coordinator Benchmark")
    print("=" * 70)
    print()
    print("Hardware: Thor (Jetson AGX Thor Developer Kit)")
    print("Component: Multi-EP Coordinator (5 domains)")
    print("Domains: Emotional, Quality, Attention, Grounding, Authorization")
    print()
    
    # Create coordinator
    coordinator = MultiEPCoordinator()
    
    results = []
    
    # Benchmark 1: Consensus
    print("Running benchmark 1/4: Consensus (100K iterations)...")
    result1 = benchmark_consensus(coordinator, 100000)
    results.append(result1)
    print(f"  Throughput: {result1.throughput_per_second:,.0f} decisions/sec")
    print(f"  Latency: {result1.avg_latency_us:.2f} μs")
    print()
    
    # Benchmark 2: Conflict Resolution
    print("Running benchmark 2/4: Conflict Resolution (100K iterations)...")
    result2 = benchmark_conflict_resolution(coordinator, 100000)
    results.append(result2)
    print(f"  Throughput: {result2.throughput_per_second:,.0f} decisions/sec")
    print(f"  Latency: {result2.avg_latency_us:.2f} μs")
    print()
    
    # Benchmark 3: Cascade Detection
    print("Running benchmark 3/4: Cascade Detection (100K iterations)...")
    result3 = benchmark_cascade_detection(coordinator, 100000)
    results.append(result3)
    print(f"  Throughput: {result3.throughput_per_second:,.0f} decisions/sec")
    print(f"  Latency: {result3.avg_latency_us:.2f} μs")
    print()
    
    # Benchmark 4: Mixed Scenarios
    print("Running benchmark 4/4: Mixed Scenarios (50K iterations)...")
    result4 = benchmark_mixed_scenarios(coordinator, 50000)
    results.append(result4)
    print(f"  Throughput: {result4.throughput_per_second:,.0f} decisions/sec")
    print(f"  Latency: {result4.avg_latency_us:.2f} μs")
    print()
    
    # Calculate averages
    avg_throughput = sum(r.throughput_per_second for r in results) / len(results)
    avg_latency_us = sum(r.avg_latency_us for r in results) / len(results)
    
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print()
    print(f"Average Throughput: {avg_throughput:,.0f} decisions/sec")
    print(f"Average Latency: {avg_latency_us:.2f} μs ({avg_latency_us/1000:.3f} ms)")
    print()
    print("Comparison to Other Hardware:")
    print(f"- Web4/Legion (RTX 4090):     280,944 decisions/sec,  3.46 μs")
    print(f"- Web4/Sprout (Orin Nano):     97,204 decisions/sec, 10.29 μs")
    print(f"- SAGE/Thor (Jetson AGX):  {avg_throughput:>10,.0f} decisions/sec, {avg_latency_us:>5.2f} μs")
    print()
    
    if avg_throughput > 97204:
        speedup_sprout = avg_throughput / 97204
        print(f"✅ Thor is {speedup_sprout:.2f}x faster than Sprout (expected given hardware)")
    
    if avg_throughput < 280944:
        ratio_legion = avg_throughput / 280944
        print(f"⚡ Thor achieves {ratio_legion:.1%} of Legion performance (expected for embedded)")
    
    print()
    print("Production Readiness Assessment:")
    if avg_throughput > 50000:
        print(f"✅ PRODUCTION READY: {avg_throughput:,.0f} decisions/sec exceeds 50K threshold")
    if avg_latency_us < 100:
        print(f"✅ LOW LATENCY: {avg_latency_us:.2f}μs is sub-100μs (real-time capable)")
    
    print()
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "hardware": "Thor (Jetson AGX Thor Developer Kit)",
        "component": "SAGE Multi-EP Coordinator (5 domains)",
        "domains": ["emotional", "quality", "attention", "grounding", "authorization"],
        "results": [
            {
                "test_name": r.test_name,
                "iterations": r.iterations,
                "total_time_seconds": r.total_time_seconds,
                "throughput_per_second": r.throughput_per_second,
                "avg_latency_ms": r.avg_latency_ms,
                "avg_latency_us": r.avg_latency_us,
                "min_latency_ms": r.min_latency_ms,
                "max_latency_ms": r.max_latency_ms,
                "notes": r.notes
            }
            for r in results
        ],
        "summary": {
            "avg_throughput": avg_throughput,
            "avg_latency_us": avg_latency_us
        },
        "comparison": {
            "web4_legion_rtx4090": {
                "throughput": 280944,
                "latency_us": 3.46
            },
            "web4_sprout_orin_nano": {
                "throughput": 97204,
                "latency_us": 10.29
            }
        }
    }
    
    output_file = Path(__file__).parent / "session142_ep_coordinator_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {output_file.name}")
    print()


if __name__ == "__main__":
    run_benchmark_suite()
