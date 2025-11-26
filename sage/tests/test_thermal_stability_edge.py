#!/usr/bin/env python3
"""
Thermal Stability Test - Edge Validation

Tests sustained inference workload on Jetson to identify:
1. Thermal throttling thresholds
2. Temperature stabilization patterns
3. Performance degradation under thermal pressure
4. Safe continuous operation limits

Session 16 - Edge Validation
Hardware: Jetson Orin Nano 8GB
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import gc
import torch
from dataclasses import dataclass
from typing import List, Optional
from sage.irp.plugins.llm_impl import ConversationalLLM


@dataclass
class ThermalSample:
    """A single thermal measurement"""
    timestamp: float
    temperature_c: float
    inference_time: float
    query_num: int


def get_temperature() -> float:
    """Read Jetson thermal zone temperature"""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return int(f.read().strip()) / 1000.0
    except:
        return 0.0


def get_memory_available() -> float:
    """Get available memory in GB"""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except:
        return 0.0
    return 0.0


def run_thermal_test(
    model_path: str = 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism',
    num_queries: int = 10,
    cooldown_threshold: float = 80.0,
    iterations: int = 3
):
    """
    Run sustained inference and monitor thermal behavior.

    Args:
        model_path: Path to model
        num_queries: Number of inference queries to run
        cooldown_threshold: Temperature (°C) at which to pause for cooling
        iterations: IRP iterations per query
    """
    print("=" * 80)
    print("THERMAL STABILITY TEST - EDGE VALIDATION")
    print("=" * 80)
    print()
    print(f"Hardware: Jetson Orin Nano 8GB")
    print(f"Model: {Path(model_path).name}")
    print(f"Queries: {num_queries}")
    print(f"IRP iterations: {iterations}")
    print(f"Cooldown threshold: {cooldown_threshold}°C")
    print()

    # Initial readings
    start_temp = get_temperature()
    start_mem = get_memory_available()
    print(f"Initial temperature: {start_temp:.1f}°C")
    print(f"Initial memory available: {start_mem:.2f} GB")
    print()

    # Load model
    print("Loading model...")
    load_start = time.time()
    conv = ConversationalLLM(
        model_path=model_path,
        irp_iterations=iterations
    )
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    print(f"Post-load temperature: {get_temperature():.1f}°C")
    print()

    # Test queries (varying complexity)
    queries = [
        "What is 2+2?",  # Simple
        "Explain the concept of consciousness.",  # Complex
        "Are you aware of this conversation?",  # Meta-cognitive
        "What patterns do you observe in your responses?",  # Introspective
        "Describe the relationship between knowledge and understanding.",  # Philosophical
        "Hello, how are you?",  # Simple greeting
        "What are the implications of artificial intelligence?",  # Complex
        "Can you reflect on your own thinking process?",  # Meta-cognitive
        "What is the meaning of existence?",  # Philosophical
        "Summarize our conversation so far.",  # Memory-dependent
    ]

    # Extend queries if needed
    while len(queries) < num_queries:
        queries.extend(queries[:num_queries - len(queries)])
    queries = queries[:num_queries]

    samples: List[ThermalSample] = []
    total_start = time.time()

    print("─" * 80)
    print("SUSTAINED INFERENCE TEST")
    print("─" * 80)
    print(f"{'Query':<6} {'Temp (°C)':<12} {'Δ Temp':<10} {'Time (s)':<12} {'Mem (GB)':<10}")
    print("─" * 80)

    for i, query in enumerate(queries, 1):
        # Check for thermal throttling
        current_temp = get_temperature()
        if current_temp >= cooldown_threshold:
            print(f"\n⚠ THERMAL WARNING: {current_temp:.1f}°C >= {cooldown_threshold}°C")
            print("Pausing for cooldown...")

            # Wait for temperature to drop
            while get_temperature() >= cooldown_threshold - 5:
                time.sleep(5)

            print(f"Resumed at {get_temperature():.1f}°C\n")

        # Run inference
        pre_temp = get_temperature()
        start_time = time.time()

        try:
            response, metadata = conv.respond(query)
            inference_time = time.time() - start_time
        except Exception as e:
            print(f"Query {i} ERROR: {e}")
            inference_time = 0

        post_temp = get_temperature()
        mem_available = get_memory_available()

        # Record sample
        sample = ThermalSample(
            timestamp=time.time() - total_start,
            temperature_c=post_temp,
            inference_time=inference_time,
            query_num=i
        )
        samples.append(sample)

        # Calculate temperature change
        temp_delta = post_temp - pre_temp

        print(f"{i:<6} {post_temp:<12.1f} {temp_delta:+<10.1f} {inference_time:<12.1f} {mem_available:<10.2f}")

    # Final readings
    total_time = time.time() - total_start
    final_temp = get_temperature()
    final_mem = get_memory_available()

    # Analysis
    print()
    print("=" * 80)
    print("THERMAL ANALYSIS")
    print("=" * 80)

    # Temperature statistics
    temps = [s.temperature_c for s in samples]
    max_temp = max(temps)
    min_temp = min(temps)
    avg_temp = sum(temps) / len(temps)
    temp_rise = final_temp - start_temp

    print(f"\nTemperature Statistics:")
    print(f"  Start:     {start_temp:.1f}°C")
    print(f"  End:       {final_temp:.1f}°C")
    print(f"  Rise:      {temp_rise:+.1f}°C")
    print(f"  Min:       {min_temp:.1f}°C")
    print(f"  Max:       {max_temp:.1f}°C")
    print(f"  Average:   {avg_temp:.1f}°C")

    # Inference statistics
    inf_times = [s.inference_time for s in samples if s.inference_time > 0]
    if inf_times:
        avg_inf = sum(inf_times) / len(inf_times)
        min_inf = min(inf_times)
        max_inf = max(inf_times)

        print(f"\nInference Statistics:")
        print(f"  Total queries:  {len(inf_times)}")
        print(f"  Total time:     {total_time:.1f}s")
        print(f"  Avg per query:  {avg_inf:.1f}s")
        print(f"  Min:            {min_inf:.1f}s")
        print(f"  Max:            {max_inf:.1f}s")

        # Performance degradation analysis
        first_half = inf_times[:len(inf_times)//2]
        second_half = inf_times[len(inf_times)//2:]

        if first_half and second_half:
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            degradation = ((second_avg - first_avg) / first_avg) * 100

            print(f"\nPerformance Degradation:")
            print(f"  First half avg:  {first_avg:.1f}s")
            print(f"  Second half avg: {second_avg:.1f}s")
            print(f"  Degradation:     {degradation:+.1f}%")

    # Memory analysis
    print(f"\nMemory Statistics:")
    print(f"  Start available:  {start_mem:.2f} GB")
    print(f"  End available:    {final_mem:.2f} GB")
    print(f"  Used by test:     {start_mem - final_mem:.2f} GB")

    # Thermal verdict
    print()
    print("=" * 80)
    print("THERMAL STABILITY VERDICT")
    print("=" * 80)

    throttle_occurred = max_temp >= cooldown_threshold
    stable = temp_rise < 15  # Less than 15°C rise is stable
    safe_continuous = max_temp < 70  # Safe for continuous operation

    if throttle_occurred:
        print(f"\n⚠ THROTTLE RISK: Max temperature {max_temp:.1f}°C reached threshold")
        print(f"  Recommendation: Reduce workload intensity or add cooling")
    elif not stable:
        print(f"\n⚠ THERMAL DRIFT: Temperature rose {temp_rise:.1f}°C")
        print(f"  Recommendation: Monitor long-term operation")
    elif safe_continuous:
        print(f"\n✓ STABLE: Safe for continuous operation")
        print(f"  Max temp {max_temp:.1f}°C is well below throttle threshold")
    else:
        print(f"\n⚠ CAUTION: Operating near thermal limits")
        print(f"  Max temp {max_temp:.1f}°C, monitor closely")

    # Clean up
    del conv
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Thermal Stability Test")
    parser.add_argument("--queries", type=int, default=10, help="Number of queries")
    parser.add_argument("--threshold", type=float, default=80.0, help="Cooldown threshold °C")
    parser.add_argument("--iterations", type=int, default=3, help="IRP iterations")
    parser.add_argument("--model", type=str,
                       default="model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism",
                       help="Model path")

    args = parser.parse_args()

    samples = run_thermal_test(
        model_path=args.model,
        num_queries=args.queries,
        cooldown_threshold=args.threshold,
        iterations=args.iterations
    )
