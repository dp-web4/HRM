#!/usr/bin/env python3
"""
Compare Nemotron 4B vs Q3-Omni 30B for SAGE integration.

This benchmarks:
1. Memory usage
2. Inference speed (tokens/sec)
3. Generation quality
4. Integration complexity

Purpose: Determine if Nemotron 4B is a viable drop-in replacement for
Q3-Omni 30B in SAGE's language reasoning role.
"""

import sys
import torch
import time
from pathlib import Path

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from irp.plugins.nemotron_irp import NemotronIRPPlugin


def benchmark_nemotron():
    """Benchmark Nemotron 4B performance."""
    print("=" * 80)
    print("Nemotron 4B Benchmark")
    print("=" * 80)
    print()

    # Configuration
    config = {
        'lazy_load': False,
        'max_length': 512,
    }

    print("🚀 Loading Nemotron...")
    start_load = time.time()

    plugin = NemotronIRPPlugin(config)

    load_time = time.time() - start_load

    # Memory stats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)

        print(f"✅ Loaded in {load_time:.1f}s")
        print(f"📊 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        print()

    # Test cases
    test_cases = [
        {
            "name": "Simple Question",
            "prompt": "What is consciousness?",
            "max_tokens": 50,
        },
        {
            "name": "Code Generation",
            "prompt": "Write a Python function to check if a number is prime.",
            "max_tokens": 100,
        },
        {
            "name": "Reasoning Task",
            "prompt": "If all A are B, and all B are C, then are all A also C? Explain your reasoning.",
            "max_tokens": 80,
        },
        {
            "name": "Creative Writing",
            "prompt": "Write a short story about a robot learning to feel emotions.",
            "max_tokens": 150,
        },
    ]

    results = []

    print("📝 Running Test Cases...")
    print("=" * 80)
    print()

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: {test['name']}")
        print("-" * 80)

        # Initialize state
        state = plugin.init_state(test['prompt'])

        # Generate
        start = time.time()
        history = []

        for step in range(test['max_tokens']):
            state = plugin.step(state)
            e = plugin.energy(state)
            history.append(e)

            # Check halt
            if plugin.halt(history):
                break

        gen_time = time.time() - start

        # Get output
        output = plugin.get_output(state)
        tokens_generated = state.x.shape[1] - state.metadata['input_length']

        # Calculate speed
        tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0

        # Store results
        results.append({
            'name': test['name'],
            'prompt': test['prompt'],
            'output': output,
            'tokens': tokens_generated,
            'time': gen_time,
            'tokens_per_sec': tokens_per_sec,
            'steps': len(history),
        })

        print(f"Prompt: {test['prompt']}")
        print(f"Output: {output[:200]}{'...' if len(output) > 200 else ''}")
        print(f"Stats: {tokens_generated} tokens in {gen_time:.2f}s ({tokens_per_sec:.2f} tok/s)")
        print()

    # Summary
    print("=" * 80)
    print("📊 Benchmark Summary")
    print("=" * 80)
    print()

    avg_tokens_per_sec = sum(r['tokens_per_sec'] for r in results) / len(results)
    total_tokens = sum(r['tokens'] for r in results)
    total_time = sum(r['time'] for r in results)

    print(f"Model: Nemotron 4B")
    print(f"Test cases: {len(results)}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average speed: {avg_tokens_per_sec:.2f} tok/s")
    print(f"Load time: {load_time:.1f}s")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak GPU memory: {peak_mem:.2f} GB")

    print()

    # Metrics
    metrics = plugin.get_metrics()
    print("Plugin Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print()

    return results


def compare_with_q3omni():
    """
    Compare with Q3-Omni baseline (from previous validation).

    Q3-Omni 30B baseline (from sage/quantization validation):
    - Memory: ~65.72 GB
    - Speed: ~1.34 tok/s
    - Quality: Working (dragon story generation)
    """
    print("=" * 80)
    print("Comparison: Nemotron 4B vs Q3-Omni 30B")
    print("=" * 80)
    print()

    # Run Nemotron benchmark
    nemotron_results = benchmark_nemotron()

    # Q3-Omni baseline (from validation)
    q3omni_baseline = {
        'model_size': '30B parameters',
        'memory_gb': 65.72,
        'tokens_per_sec': 1.34,
        'load_time': 180.0,  # ~3 minutes
        'status': 'Working (validated)',
    }

    # Nemotron stats
    avg_speed = sum(r['tokens_per_sec'] for r in nemotron_results) / len(nemotron_results)

    if torch.cuda.is_available():
        nemotron_mem = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        nemotron_mem = 8.0  # Estimated

    print("📊 Side-by-Side Comparison")
    print("=" * 80)
    print()
    print(f"{'Metric':<25} {'Nemotron 4B':<20} {'Q3-Omni 30B':<20} {'Improvement':<15}")
    print("-" * 80)
    print(f"{'Model Size':<25} {'4B params':<20} {q3omni_baseline['model_size']:<20} {'7.5x smaller':<15}")
    print(f"{'GPU Memory':<25} {f'{nemotron_mem:.2f} GB':<20} {f\"{q3omni_baseline['memory_gb']:.2f} GB\":<20} {f\"{q3omni_baseline['memory_gb']/nemotron_mem:.1f}x less\":<15}")
    print(f"{'Tokens/sec':<25} {f'{avg_speed:.2f}':<20} {f\"{q3omni_baseline['tokens_per_sec']:.2f}\":<20} {f\"{avg_speed/q3omni_baseline['tokens_per_sec']:.1f}x faster\":<15}")
    print(f"{'Load Time':<25} {f'{nemotron_results[0] if nemotron_results else 0:.1f}s':<20} {f\"{q3omni_baseline['load_time']:.1f}s\":<20} {'-':<15}")
    print()

    print("✅ Advantages of Nemotron 4B:")
    print("  - 7.5x smaller model (4B vs 30B params)")
    print(f"  - {q3omni_baseline['memory_gb']/nemotron_mem:.1f}x less GPU memory")
    print(f"  - {avg_speed/q3omni_baseline['tokens_per_sec']:.1f}x faster inference")
    print("  - Easier to deploy on edge devices")
    print("  - Can run alongside other SAGE plugins")
    print()

    print("⚠️  Trade-offs:")
    print("  - Smaller model may have lower reasoning capability")
    print("  - Need to validate quality on SAGE tasks")
    print("  - Text-only (Q3-Omni has audio)")
    print()

    print("🎯 Recommendation:")
    print("  Nemotron 4B is ideal for SAGE integration when:")
    print("  - Memory is constrained (edge devices)")
    print("  - Speed is critical (real-time orchestration)")
    print("  - Running multiple plugins simultaneously")
    print("  - Text-only language reasoning needed")
    print()

    return nemotron_results


if __name__ == "__main__":
    try:
        compare_with_q3omni()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
