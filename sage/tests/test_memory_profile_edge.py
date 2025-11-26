#!/usr/bin/env python3
"""
Memory Profile Test - Edge Validation

Tests memory usage patterns on Jetson for:
1. Model loading/unloading
2. Model swapping (switching between models)
3. Memory fragmentation
4. GPU memory management

Session 16 - Edge Validation
Hardware: Jetson Orin Nano 8GB (unified memory)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import gc
import torch
from typing import List, Dict, Any
from sage.irp.plugins.llm_impl import ConversationalLLM


def get_memory_stats() -> Dict[str, float]:
    """Get comprehensive memory statistics"""
    stats = {}

    # System memory from /proc/meminfo
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    stats['total_gb'] = int(line.split()[1]) / (1024 * 1024)
                elif line.startswith('MemAvailable:'):
                    stats['available_gb'] = int(line.split()[1]) / (1024 * 1024)
                elif line.startswith('MemFree:'):
                    stats['free_gb'] = int(line.split()[1]) / (1024 * 1024)
                elif line.startswith('Buffers:'):
                    stats['buffers_gb'] = int(line.split()[1]) / (1024 * 1024)
                elif line.startswith('Cached:'):
                    if 'Cached:' == line.split(':')[0]:
                        stats['cached_gb'] = int(line.split()[1]) / (1024 * 1024)
    except:
        pass

    # GPU memory (CUDA)
    if torch.cuda.is_available():
        stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
        stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        stats['gpu_max_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)

    return stats


def print_memory_stats(label: str, stats: Dict[str, float]):
    """Print formatted memory statistics"""
    print(f"\n  [{label}]")
    print(f"    System: {stats.get('available_gb', 0):.2f} GB available of {stats.get('total_gb', 0):.2f} GB")
    if 'gpu_allocated_gb' in stats:
        print(f"    GPU: {stats.get('gpu_allocated_gb', 0):.3f} GB allocated, {stats.get('gpu_reserved_gb', 0):.3f} GB reserved")


def test_model_load_unload():
    """Test single model load/unload cycle"""
    print("=" * 80)
    print("TEST 1: MODEL LOAD/UNLOAD CYCLE")
    print("=" * 80)

    model_path = 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism'

    # Initial state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    initial = get_memory_stats()
    print_memory_stats("Initial", initial)

    # Load model
    print(f"\n  Loading: {Path(model_path).name}...")
    start = time.time()
    conv = ConversationalLLM(model_path=model_path, irp_iterations=3)
    load_time = time.time() - start

    loaded = get_memory_stats()
    print_memory_stats(f"After load ({load_time:.2f}s)", loaded)

    # Run inference
    print("\n  Running inference...")
    response, metadata = conv.respond("What is 2+2?")

    after_inference = get_memory_stats()
    print_memory_stats("After inference", after_inference)

    # Unload model
    print("\n  Unloading model...")
    del conv
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(1)  # Allow cleanup

    final = get_memory_stats()
    print_memory_stats("After unload", final)

    # Analysis
    memory_used = initial['available_gb'] - loaded['available_gb']
    memory_recovered = final['available_gb'] - loaded['available_gb']
    leak = initial['available_gb'] - final['available_gb']

    print(f"\n  Analysis:")
    print(f"    Memory used by model: {memory_used:.2f} GB")
    print(f"    Memory recovered: {memory_recovered:.2f} GB")
    print(f"    Potential leak: {leak:.2f} GB")

    return leak < 0.1  # Success if leak is minimal


def test_model_swapping():
    """Test swapping between multiple models"""
    print("\n" + "=" * 80)
    print("TEST 2: MODEL SWAPPING")
    print("=" * 80)

    models = [
        ('epistemic-pragmatism', 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism'),
        ('introspective-merged', 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged'),
    ]

    # Initial state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    initial = get_memory_stats()
    print_memory_stats("Initial", initial)

    memory_usage = []

    for i, (name, path) in enumerate(models, 1):
        print(f"\n  Swap {i}: Loading {name}...")

        start = time.time()
        conv = ConversationalLLM(model_path=path, irp_iterations=3)
        load_time = time.time() - start

        after_load = get_memory_stats()
        print_memory_stats(f"Loaded {name} ({load_time:.2f}s)", after_load)

        # Run inference
        response, _ = conv.respond("Hello")

        after_inf = get_memory_stats()

        memory_usage.append({
            'model': name,
            'load_time': load_time,
            'memory_after_load': initial['available_gb'] - after_load['available_gb'],
            'memory_after_inference': initial['available_gb'] - after_inf['available_gb']
        })

        # Unload
        print(f"  Unloading {name}...")
        del conv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1)

        after_unload = get_memory_stats()
        print_memory_stats(f"After unloading {name}", after_unload)

    # Final state
    final = get_memory_stats()
    print_memory_stats("Final (all unloaded)", final)

    # Analysis
    total_leak = initial['available_gb'] - final['available_gb']

    print(f"\n  Memory Usage Summary:")
    print(f"    {'Model':<25} {'Load Time':<12} {'Mem Used':<12}")
    print(f"    {'-'*49}")
    for m in memory_usage:
        print(f"    {m['model']:<25} {m['load_time']:<12.2f} {m['memory_after_load']:<12.2f}")

    print(f"\n  Total memory leak after swaps: {total_leak:.3f} GB")

    return total_leak < 0.2  # Success if leak is acceptable


def test_consecutive_loads():
    """Test loading the same model multiple times"""
    print("\n" + "=" * 80)
    print("TEST 3: CONSECUTIVE LOADS (FRAGMENTATION TEST)")
    print("=" * 80)

    model_path = 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism'
    num_cycles = 3

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    initial = get_memory_stats()
    print_memory_stats("Initial", initial)

    for cycle in range(1, num_cycles + 1):
        print(f"\n  Cycle {cycle}/{num_cycles}")

        # Load
        conv = ConversationalLLM(model_path=model_path, irp_iterations=3)
        after_load = get_memory_stats()

        # Inference
        conv.respond("Quick test")

        # Unload
        del conv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.5)

        after_cycle = get_memory_stats()
        cycle_leak = initial['available_gb'] - after_cycle['available_gb']
        print(f"    Available after cycle: {after_cycle['available_gb']:.2f} GB (leak: {cycle_leak:.3f} GB)")

    final = get_memory_stats()
    total_leak = initial['available_gb'] - final['available_gb']

    print(f"\n  Total leak after {num_cycles} cycles: {total_leak:.3f} GB")
    print(f"  Per-cycle average leak: {total_leak/num_cycles:.3f} GB")

    return total_leak < 0.3


def run_memory_profile():
    """Run all memory profile tests"""
    print("=" * 80)
    print("MEMORY PROFILE TEST - EDGE VALIDATION")
    print("=" * 80)
    print()
    print("Hardware: Jetson Orin Nano 8GB (unified CPU/GPU memory)")
    print("Purpose: Profile memory usage patterns for edge deployment")
    print()

    results = {}

    # Test 1: Load/Unload
    results['load_unload'] = test_model_load_unload()

    # Test 2: Model Swapping
    results['swapping'] = test_model_swapping()

    # Test 3: Consecutive Loads
    results['consecutive'] = test_consecutive_loads()

    # Summary
    print("\n" + "=" * 80)
    print("MEMORY PROFILE SUMMARY")
    print("=" * 80)

    print(f"\n  {'Test':<30} {'Result':<10}")
    print(f"  {'-'*40}")
    for test, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {test:<30} {status}")

    all_passed = all(results.values())

    print()
    if all_passed:
        print("✓ All memory tests passed - safe for production edge deployment")
    else:
        print("⚠ Some memory tests failed - investigate before production")

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = run_memory_profile()
