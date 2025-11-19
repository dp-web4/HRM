"""
Track 9: LLM IRP Pipeline Profiling

Profile the complete LLM IRP pipeline to identify bottlenecks:
- Model loading time
- Per-iteration inference time
- SNARC scoring time
- Memory usage patterns
- Opportunities for optimization

Target: Understand Sprout's 55s inference time and optimize
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import tracemalloc
from contextlib import contextmanager
from sage.irp.plugins.llm_impl import ConversationalLLM
from sage.irp.plugins.llm_snarc_integration import ConversationalMemory


@contextmanager
def profile_section(name):
    """Profile a section of code."""
    start_time = time.time()
    start_mem = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB

    yield

    end_time = time.time()
    end_mem = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB

    elapsed = end_time - start_time
    mem_delta = end_mem - start_mem

    print(f"  [{name}]")
    print(f"    Time: {elapsed:.3f}s")
    print(f"    Memory: {end_mem:.1f}MB (Δ {mem_delta:+.1f}MB)")


def profile_llm_pipeline(model_path="Qwen/Qwen2.5-0.5B-Instruct", num_questions=3):
    """Profile complete LLM IRP pipeline."""

    print("="*80)
    print("TRACK 9: LLM IRP PIPELINE PROFILING")
    print("="*80)
    print(f"\nModel: {model_path}")
    print(f"Questions: {num_questions}")
    print(f"IRP iterations: 5\n")

    # Start memory tracking
    tracemalloc.start()

    # Phase 1: Model Loading
    print("Phase 1: Model Loading")
    with profile_section("Model initialization"):
        conv = ConversationalLLM(
            model_path=model_path,
            irp_iterations=5
        )

    with profile_section("Memory initialization"):
        memory = ConversationalMemory(salience_threshold=0.15)

    # Phase 2: Inference Pipeline
    print("\nPhase 2: Inference Pipeline")

    questions = [
        "What is the difference between knowledge and understanding?",
        "Are you aware of this conversation?",
        "What is 2+2?"
    ][:num_questions]

    total_inference_time = 0
    total_snarc_time = 0
    iteration_times = []

    for i, question in enumerate(questions, 1):
        print(f"\n  Question {i}: {question[:50]}...")

        # Inference with IRP
        with profile_section(f"Q{i} - IRP inference"):
            start = time.time()
            response, irp_info = conv.respond(question, use_irp=True)
            inference_time = time.time() - start
            total_inference_time += inference_time

        # Track per-iteration timing
        if 'all_energies' in irp_info:
            avg_per_iter = inference_time / irp_info['iterations']
            iteration_times.append(avg_per_iter)
            print(f"    Per-iteration: {avg_per_iter:.3f}s")

        # SNARC scoring
        with profile_section(f"Q{i} - SNARC scoring"):
            start = time.time()
            is_salient, scores = memory.record_exchange(question, response, irp_info)
            snarc_time = time.time() - start
            total_snarc_time += snarc_time

        print(f"    Salience: {scores['total_salience']:.3f} {'✓' if is_salient else '✗'}")

    # Phase 3: Summary Statistics
    print("\n" + "="*80)
    print("PROFILING SUMMARY")
    print("="*80)

    current, peak = tracemalloc.get_traced_memory()

    print(f"\n📊 Time Breakdown:")
    print(f"  Total inference: {total_inference_time:.2f}s ({num_questions} questions)")
    print(f"  Avg per question: {total_inference_time/num_questions:.2f}s")
    if iteration_times:
        avg_iter = sum(iteration_times) / len(iteration_times)
        print(f"  Avg per iteration: {avg_iter:.3f}s")
    print(f"  Total SNARC: {total_snarc_time:.3f}s")
    print(f"  Avg SNARC per question: {total_snarc_time/num_questions:.3f}s")

    print(f"\n💾 Memory Usage:")
    print(f"  Current: {current / 1024 / 1024:.1f}MB")
    print(f"  Peak: {peak / 1024 / 1024:.1f}MB")

    print(f"\n⚡ Performance Analysis:")
    inference_pct = (total_inference_time / (total_inference_time + total_snarc_time)) * 100
    snarc_pct = (total_snarc_time / (total_inference_time + total_snarc_time)) * 100
    print(f"  Inference: {inference_pct:.1f}% of pipeline time")
    print(f"  SNARC: {snarc_pct:.1f}% of pipeline time")

    if iteration_times:
        print(f"\n🎯 Optimization Opportunities:")
        print(f"  1. Per-iteration time: {avg_iter:.3f}s")
        print(f"     - 5 iterations × {avg_iter:.3f}s = {avg_iter * 5:.2f}s")
        print(f"     - Reducing to 3 iterations: {avg_iter * 3:.2f}s (save {avg_iter * 2:.2f}s)")

        print(f"  2. SNARC overhead: {total_snarc_time:.3f}s total")
        print(f"     - {total_snarc_time/num_questions:.3f}s per question")
        print(f"     - Negligible vs inference ({snarc_pct:.1f}%)")

        print(f"  3. Model loading: One-time cost")
        print(f"     - Reuse model across sessions")
        print(f"     - Keep-alive pattern for production")

    tracemalloc.stop()

    return {
        'total_time': total_inference_time + total_snarc_time,
        'inference_time': total_inference_time,
        'snarc_time': total_snarc_time,
        'avg_per_question': total_inference_time / num_questions,
        'avg_per_iteration': sum(iteration_times) / len(iteration_times) if iteration_times else None,
        'peak_memory_mb': peak / 1024 / 1024,
        'num_questions': num_questions
    }


def compare_configurations():
    """Compare different IRP configurations."""
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)

    configs = [
        ("3 iterations", 3),
        ("5 iterations (default)", 5),
        ("7 iterations", 7),
    ]

    results = []

    for name, iterations in configs:
        print(f"\n\nTesting: {name}")
        print("-" * 80)

        tracemalloc.start()

        conv = ConversationalLLM(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            irp_iterations=iterations
        )

        # Single test question
        question = "What is the difference between knowledge and understanding?"

        start = time.time()
        response, irp_info = conv.respond(question, use_irp=True)
        elapsed = time.time() - start

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append({
            'name': name,
            'iterations': iterations,
            'time': elapsed,
            'per_iteration': elapsed / iterations,
            'energy': irp_info['final_energy'],
            'converged': irp_info['converged'],
            'memory_mb': peak / 1024 / 1024
        })

        print(f"  Time: {elapsed:.2f}s ({elapsed/iterations:.3f}s per iteration)")
        print(f"  Energy: {irp_info['final_energy']:.3f}")
        print(f"  Converged: {irp_info['converged']}")
        print(f"  Memory: {peak / 1024 / 1024:.1f}MB")

    # Summary comparison
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)

    print(f"\n{'Configuration':<25} {'Time':<10} {'Per-Iter':<12} {'Energy':<10} {'Converged'}")
    print("-" * 80)
    for r in results:
        converged = "✓" if r['converged'] else "✗"
        print(f"{r['name']:<25} {r['time']:<10.2f} {r['per_iteration']:<12.3f} {r['energy']:<10.3f} {converged}")

    print(f"\n💡 Recommendations:")
    fastest = min(results, key=lambda x: x['time'])
    print(f"  Fastest: {fastest['name']} ({fastest['time']:.2f}s)")

    best_quality = min(results, key=lambda x: x['energy'])
    print(f"  Best quality: {best_quality['name']} (energy: {best_quality['energy']:.3f})")

    # Trade-off analysis
    baseline = next(r for r in results if r['iterations'] == 5)
    for r in results:
        if r['iterations'] != 5:
            time_diff = ((r['time'] - baseline['time']) / baseline['time']) * 100
            energy_diff = ((r['energy'] - baseline['energy']) / baseline['energy']) * 100
            print(f"  {r['name']} vs baseline: {time_diff:+.1f}% time, {energy_diff:+.1f}% energy")


if __name__ == "__main__":
    print("Track 9: Real-Time Optimization - Pipeline Profiling")
    print("Target: Optimize for edge deployment (Sprout's 55s → faster)\n")

    # Full pipeline profile
    metrics = profile_llm_pipeline(num_questions=3)

    # Configuration comparison
    compare_configurations()  # Test different iteration counts

    print("\n" + "="*80)
    print("✓ Profiling complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Compare with Sprout's edge metrics (55s avg)")
    print("  2. Identify platform differences (Thor vs Sprout)")
    print("  3. Implement edge-optimized configurations")
    print("  4. Re-profile with optimizations")
