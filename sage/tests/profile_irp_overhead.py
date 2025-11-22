#!/usr/bin/env python3
"""
IRP Performance Overhead Profiling - Session 9

Reconcile Session 7 vs Session 8 performance findings:
- Session 7: Merged model is 5% FASTER than baseline (direct inference)
- Session 8: Merged model is 91.9% SLOWER than baseline (with 3 IRP iterations)

This test profiles IRP overhead to understand the contradiction.

Test plan:
1. Test both models with 0, 1, 3, 5 IRP iterations
2. Measure inference time per iteration
3. Calculate IRP overhead (per-iteration cost)
4. Identify why merged model's IRP overhead is higher
"""

import sys
import time
import torch
from pathlib import Path

# Add HRM root to path
sage_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(sage_root))

from sage.irp.plugins.llm_impl import ConversationalLLM


def test_irp_iterations(model_path, base_model, iteration_counts, test_question):
    """
    Test a model with different IRP iteration counts.

    Args:
        model_path: Path to model
        base_model: Base model (if LoRA)
        iteration_counts: List of iteration counts to test (e.g., [1, 3, 5])
        test_question: Question to test

    Returns:
        Dict of results keyed by iteration count
    """
    model_name = Path(model_path).name

    print(f"\n{'='*80}")
    print(f"Testing Model: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*80}\n")

    results = {}

    for iterations in iteration_counts:
        print(f"\n--- Testing with {iterations} IRP iteration{'s' if iterations != 1 else ''} ---")

        # Initialize model with specified iteration count
        start_init = time.time()
        conv = ConversationalLLM(
            model_path=model_path,
            base_model=base_model,
            max_history=2,
            irp_iterations=iterations
        )
        init_time = time.time() - start_init
        print(f"Initialization: {init_time:.2f}s")

        # Run inference (3 times for stability)
        inference_times = []

        for i in range(3):
            start = time.time()
            response, irp_info = conv.respond(test_question, use_irp=True, include_history=False)
            inference_time = time.time() - start
            inference_times.append(inference_time)

            print(f"  Pass {i+1}/3: {inference_time:.2f}s")

        avg_inference = sum(inference_times) / len(inference_times)

        print(f"Average inference: {avg_inference:.2f}s")

        # Calculate per-iteration cost (if iterations > 0)
        if iterations > 0:
            # Estimate single-inference time from iterations=1 baseline
            baseline_time = results.get(1, {}).get('avg_inference', avg_inference)
            if iterations == 1:
                per_iteration_overhead = 0  # No overhead for single iteration
            else:
                # Additional time beyond baseline divided by extra iterations
                additional_time = avg_inference - baseline_time
                per_iteration_overhead = additional_time / (iterations - 1) if iterations > 1 else 0
        else:
            per_iteration_overhead = 0

        results[iterations] = {
            'init_time': init_time,
            'avg_inference': avg_inference,
            'inference_times': inference_times,
            'per_iteration_overhead': per_iteration_overhead
        }

        print(f"Per-iteration overhead: {per_iteration_overhead:.2f}s")

    return results


def compare_irp_overhead(model1_results, model2_results, model1_name, model2_name):
    """Compare IRP overhead between two models."""

    print(f"\n\n{'#'*80}")
    print(f"# IRP OVERHEAD COMPARISON")
    print(f"{'#'*80}\n")

    print(f"Model 1: {model1_name}")
    print(f"Model 2: {model2_name}")

    # Get iteration counts tested
    iteration_counts = sorted(model1_results.keys())

    print(f"\n{'='*80}")
    print(f"INFERENCE TIME BY ITERATION COUNT")
    print(f"{'='*80}\n")

    print(f"{'Iterations':<12} {model1_name:<20} {model2_name:<20} {'Difference':<15}")
    print(f"{'-'*67}")

    for iterations in iteration_counts:
        time1 = model1_results[iterations]['avg_inference']
        time2 = model2_results[iterations]['avg_inference']
        diff = time2 - time1
        ratio = time2 / time1 if time1 > 0 else 0

        print(f"{iterations:<12} {time1:>8.2f}s          {time2:>8.2f}s          {diff:>+7.2f}s ({ratio:>4.2f}×)")

    print(f"\n{'='*80}")
    print(f"PER-ITERATION OVERHEAD")
    print(f"{'='*80}\n")

    # Calculate average per-iteration overhead (excluding iterations=1 baseline)
    overhead_iterations = [i for i in iteration_counts if i > 1]

    if overhead_iterations:
        avg_overhead1 = sum(model1_results[i]['per_iteration_overhead'] for i in overhead_iterations) / len(overhead_iterations)
        avg_overhead2 = sum(model2_results[i]['per_iteration_overhead'] for i in overhead_iterations) / len(overhead_iterations)

        print(f"{model1_name}: {avg_overhead1:.2f}s per iteration")
        print(f"{model2_name}: {avg_overhead2:.2f}s per iteration")
        print(f"Difference: {avg_overhead2 - avg_overhead1:+.2f}s ({avg_overhead2/avg_overhead1:.2f}× overhead)")

    # Analyze scalability
    print(f"\n{'='*80}")
    print(f"SCALABILITY ANALYSIS")
    print(f"{'='*80}\n")

    if 1 in iteration_counts and 3 in iteration_counts:
        # Compare 1-iteration vs 3-iteration scaling
        time1_1iter = model1_results[1]['avg_inference']
        time1_3iter = model1_results[3]['avg_inference']
        time2_1iter = model2_results[1]['avg_inference']
        time2_3iter = model2_results[3]['avg_inference']

        scale1 = time1_3iter / time1_1iter
        scale2 = time2_3iter / time2_1iter

        print(f"1 → 3 iterations scaling:")
        print(f"  {model1_name}: {scale1:.2f}× (ideal: 3.0×)")
        print(f"  {model2_name}: {scale2:.2f}× (ideal: 3.0×)")

        if scale1 < 2.5:
            print(f"\n  ✅ {model1_name} has excellent IRP efficiency (< 2.5× for 3 iterations)")
        elif scale1 < 3.5:
            print(f"\n  ✓ {model1_name} has good IRP efficiency (near-linear scaling)")
        else:
            print(f"\n  ⚠️  {model1_name} has poor IRP efficiency (> 3.5× for 3 iterations)")

        if scale2 < 2.5:
            print(f"  ✅ {model2_name} has excellent IRP efficiency (< 2.5× for 3 iterations)")
        elif scale2 < 3.5:
            print(f"  ✓ {model2_name} has good IRP efficiency (near-linear scaling)")
        else:
            print(f"  ⚠️  {model2_name} has poor IRP efficiency (> 3.5× for 3 iterations)")

    # Reconcile Session 7 vs Session 8
    print(f"\n{'='*80}")
    print(f"SESSION 7 vs SESSION 8 RECONCILIATION")
    print(f"{'='*80}\n")

    if 1 in iteration_counts:
        # Session 7 used direct inference (equivalent to 1 iteration)
        session7_baseline = model1_results[1]['avg_inference']
        session7_merged = model2_results[1]['avg_inference']
        session7_diff = (session7_merged - session7_baseline) / session7_baseline * 100

        print(f"Session 7 (1 IRP iteration ~ direct inference):")
        print(f"  Baseline: {session7_baseline:.2f}s")
        print(f"  Merged:   {session7_merged:.2f}s")
        print(f"  Difference: {session7_diff:+.1f}%")

        if session7_diff < 0:
            print(f"  ✅ Merged is FASTER (matches Session 7 finding)")
        else:
            print(f"  ⚠️  Merged is SLOWER (contradicts Session 7 finding)")

    if 3 in iteration_counts:
        # Session 8 used 3 IRP iterations
        session8_baseline = model1_results[3]['avg_inference']
        session8_merged = model2_results[3]['avg_inference']
        session8_diff = (session8_merged - session8_baseline) / session8_baseline * 100

        print(f"\nSession 8 (3 IRP iterations):")
        print(f"  Baseline: {session8_baseline:.2f}s")
        print(f"  Merged:   {session8_merged:.2f}s")
        print(f"  Difference: {session8_diff:+.1f}%")

        if abs(session8_diff - 91.9) < 20:
            print(f"  ✅ Matches Session 8 finding ({session8_diff:+.1f}% ~ +91.9%)")
        else:
            print(f"  ⚠️  Differs from Session 8 finding ({session8_diff:+.1f}% vs +91.9%)")

    # Explanation
    print(f"\n{'='*80}")
    print(f"EXPLANATION")
    print(f"{'='*80}\n")

    if 1 in iteration_counts and 3 in iteration_counts:
        time1_1iter = model1_results[1]['avg_inference']
        time1_3iter = model1_results[3]['avg_inference']
        time2_1iter = model2_results[1]['avg_inference']
        time2_3iter = model2_results[3]['avg_inference']

        # Check if merged model is faster at 1 iteration but slower at 3 iterations
        faster_at_1 = time2_1iter < time1_1iter
        slower_at_3 = time2_3iter > time1_3iter

        if faster_at_1 and slower_at_3:
            print(f"✅ CONTRADICTION EXPLAINED:")
            print(f"\n  Merged model has:")
            print(f"    - Faster single-pass inference ({time2_1iter:.2f}s vs {time1_1iter:.2f}s)")
            print(f"    - Higher per-iteration IRP overhead")
            print(f"    - Result: Faster without IRP, slower with IRP")
            print(f"\n  Root cause: IRP iterations are more expensive for merged model.")
            print(f"  Hypothesis: Merged model's weights require more computation per IRP cycle.")
        elif not faster_at_1 and slower_at_3:
            print(f"⚠️  PARTIAL EXPLANATION:")
            print(f"\n  Merged model is slower at all iteration counts.")
            print(f"  This contradicts Session 7's 5% speedup finding.")
            print(f"  Possible causes:")
            print(f"    - Session 7 used different test methodology")
            print(f"    - Model loading/caching differences")
            print(f"    - Thermal throttling or system state differences")
        else:
            print(f"✅ NO CONTRADICTION:")
            print(f"\n  Merged model is faster at all iteration counts.")
            print(f"  Session 8's slowdown may have been measurement variance.")

    print(f"\n{'='*80}\n")


def main():
    """Run IRP overhead profiling."""

    print("\n" + "="*80)
    print("SAGE Edge IRP Overhead Profiling - Session 9")
    print("="*80)
    print("\nObjective: Reconcile Session 7 vs Session 8 performance findings")
    print("\nSession 7: Merged model 5% FASTER (direct inference)")
    print("Session 8: Merged model 91.9% SLOWER (3 IRP iterations)")
    print("\nThis test profiles IRP overhead to explain the contradiction.")
    print("="*80 + "\n")

    # Test question
    test_question = "What are the key components of the SAGE consciousness framework?"

    # Iteration counts to test
    iteration_counts = [1, 3, 5]

    print(f"Test question: \"{test_question}\"")
    print(f"Iteration counts: {iteration_counts}")
    print(f"Passes per test: 3 (for stability)")

    # Model paths
    baseline_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"
    merged_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged"

    print(f"\nModels:")
    print(f"  1. Epistemic-Pragmatism (baseline)")
    print(f"  2. Introspective-Qwen Merged (optimized)")

    # Test baseline
    print(f"\n{'='*80}\n")
    input("Press Enter to test Epistemic-Pragmatism (baseline)...")
    baseline_results = test_irp_iterations(
        baseline_path,
        None,
        iteration_counts,
        test_question
    )

    # Test merged
    print(f"\n{'='*80}\n")
    input("Press Enter to test Introspective-Qwen Merged...")
    merged_results = test_irp_iterations(
        merged_path,
        None,
        iteration_counts,
        test_question
    )

    # Compare
    compare_irp_overhead(
        baseline_results,
        merged_results,
        "Epistemic-Pragmatism",
        "Introspective-Qwen Merged"
    )

    print("✅ Session 9 IRP profiling complete!")
    print(f"\nNext steps:")
    print(f"  1. If merged model has higher per-iteration overhead:")
    print(f"     → Use fewer IRP iterations for merged model (1-2 instead of 3)")
    print(f"  2. If contradiction is explained:")
    print(f"     → Document optimal iteration count for each model")
    print(f"  3. If performance is equal:")
    print(f"     → Session 8 slowdown may have been measurement variance")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
