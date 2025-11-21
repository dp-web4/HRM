#!/usr/bin/env python3
"""
Three-Model Performance Comparison - Session 7

Compare all three models:
1. Epistemic-Pragmatism (baseline, 10.10s)
2. Introspective-Qwen LoRA (regression, 24.31s)
3. Introspective-Qwen Merged (optimized, 16.23s expected)

Goal: Understand why merged model is faster than LoRA but slower than baseline.
"""

import sys
import time
import torch
from pathlib import Path

# Add HRM root to path
sage_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(sage_root))

from sage.irp.plugins.llm_impl import LLMIRPPlugin


def profile_model_quick(model_path, base_model=None, test_question=None):
    """Quick 1-pass profiling for comparison."""

    if test_question is None:
        test_question = "What are the key components of the SAGE consciousness framework?"

    model_name = Path(model_path).name

    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")

    # Initialize
    print(f"Initializing...")
    start = time.time()
    plugin = LLMIRPPlugin(
        model_path=model_path,
        base_model=base_model,
        device="cuda",
        max_tokens=150
    )
    init_time = time.time() - start
    print(f"✅ Initialized in {init_time:.2f}s")

    # Inference
    print(f"Running inference...")
    state = plugin.init_state(test_question)
    inputs = plugin.tokenizer(
        state['prompt'],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(plugin.device)

    start = time.time()
    with torch.no_grad():
        outputs = plugin.model.generate(
            **inputs,
            max_new_tokens=plugin.max_tokens,
            temperature=state['temperature'],
            do_sample=True,
            pad_token_id=plugin.tokenizer.eos_token_id
        )
    inference_time = time.time() - start

    response = plugin.tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"✅ Inference completed in {inference_time:.2f}s")

    return {
        'name': model_name,
        'path': model_path,
        'base_model': base_model,
        'init_time': init_time,
        'inference_time': inference_time,
        'response': response
    }


def compare_three_models(results):
    """Compare results from all three models."""

    print(f"\n\n{'#'*80}")
    print(f"# THREE-MODEL PERFORMANCE COMPARISON")
    print(f"{'#'*80}")

    # Extract results
    baseline = results[0]
    lora = results[1]
    merged = results[2]

    print(f"\n{'-'*80}")
    print(f"{'Model':<30} {'Init Time':<15} {'Inference Time':<15} {'Total':<15}")
    print(f"{'-'*80}")

    for r in results:
        total = r['init_time'] + r['inference_time']
        print(f"{r['name']:<30} {r['init_time']:>8.2f}s      {r['inference_time']:>8.2f}s      {total:>8.2f}s")

    print(f"{'-'*80}")

    # Ratios
    print(f"\n{'='*80}")
    print(f"PERFORMANCE RATIOS (vs Baseline)")
    print(f"{'='*80}")

    baseline_inf = baseline['inference_time']
    lora_inf = lora['inference_time']
    merged_inf = merged['inference_time']

    lora_ratio = lora_inf / baseline_inf
    merged_ratio = merged_inf / baseline_inf

    print(f"\nInference Time:")
    print(f"  Baseline (Epistemic-Pragmatism): {baseline_inf:.2f}s  (1.00×)")
    print(f"  LoRA (Introspective-Qwen):       {lora_inf:.2f}s  ({lora_ratio:.2f}× SLOWER)")
    print(f"  Merged (Introspective-Qwen):     {merged_inf:.2f}s  ({merged_ratio:.2f}× SLOWER)")

    # Improvement from LoRA to merged
    improvement = (lora_inf - merged_inf) / lora_inf * 100
    speedup = lora_inf / merged_inf

    print(f"\n{'='*80}")
    print(f"MERGE EFFECTIVENESS")
    print(f"{'='*80}")
    print(f"LoRA → Merged improvement: {improvement:+.1f}% ({speedup:.2f}× faster)")
    print(f"Remaining gap to baseline: {merged_inf - baseline_inf:+.2f}s")

    if merged_ratio < 1.3:
        print(f"\n✅ MERGE SUCCESS: Merged model within 30% of baseline!")
    elif merged_ratio < 1.5:
        print(f"\n⚠️  PARTIAL SUCCESS: Merged model 30-50% slower than baseline")
    else:
        print(f"\n❌ MERGE INSUFFICIENT: Merged model >50% slower than baseline")

    # Hypothesis for remaining gap
    print(f"\n{'='*80}")
    print(f"HYPOTHESIS: Why is merged model still slower?")
    print(f"{'='*80}")

    if merged_ratio > 1.2:
        print(f"Possible causes:")
        print(f"  1. Model architecture difference (sleep-learned weights)")
        print(f"  2. Fine-tuning changes computation graph")
        print(f"  3. ARM64 optimization difference")
        print(f"  4. Memory access patterns differ")
        print(f"\nConclusion: Introspective-Qwen's weights (not just LoRA overhead)")
        print(f"            cause slower inference. This is inherent to the model.")

    print(f"{'='*80}")


def main():
    """Run three-model comparison."""

    print("\n" + "="*80)
    print("SAGE Edge Three-Model Comparison - Session 7")
    print("="*80)
    print("\nComparing:")
    print("  1. Epistemic-Pragmatism (baseline)")
    print("  2. Introspective-Qwen LoRA (regression)")
    print("  3. Introspective-Qwen Merged (optimized)")
    print("="*80 + "\n")

    test_question = "What are the key components of the SAGE consciousness framework?"

    # Model paths
    models = [
        {
            'path': 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism',
            'base': None,
            'name': 'Baseline'
        },
        {
            'path': 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model',
            'base': 'Qwen/Qwen2.5-0.5B-Instruct',
            'name': 'LoRA'
        },
        {
            'path': 'model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged',
            'base': None,
            'name': 'Merged'
        }
    ]

    results = []

    for i, model in enumerate(models):
        input(f"Press Enter to profile model {i+1}/3 ({model['name']})...")
        result = profile_model_quick(model['path'], model['base'], test_question)
        results.append(result)

    # Compare
    compare_three_models(results)

    print("\n✅ Three-model comparison complete!\n")


if __name__ == "__main__":
    main()
