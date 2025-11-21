#!/usr/bin/env python3
"""
Performance Regression Profiling - Session 7

Profile the 2× performance regression in Introspective-Qwen vs Epistemic-Pragmatism.

Session 6 found:
- Epistemic-Pragmatism: 51.18s avg inference
- Introspective-Qwen: 102.66s avg inference (2× slower!)

This test breaks down timing into stages:
1. Model initialization (loading base model + LoRA adapter)
2. Input tokenization
3. Forward pass (inference)
4. Output decoding
5. IRP iterations

Goal: Identify which stage causes the 2× slowdown.
"""

import sys
import time
import torch
from pathlib import Path

# Add HRM root to path
sage_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(sage_root))

from sage.irp.plugins.llm_impl import LLMIRPPlugin


def profile_model_init(model_path, base_model=None):
    """Profile model initialization time."""
    print(f"\n{'='*80}")
    print(f"Profiling Model Initialization: {Path(model_path).name}")
    print(f"{'='*80}")

    start = time.time()

    # Stage 1: Base model loading
    stage1_start = time.time()
    plugin = LLMIRPPlugin(
        model_path=model_path,
        base_model=base_model,
        device="cuda",
        max_tokens=150
    )
    stage1_end = time.time()

    total_time = stage1_end - start

    print(f"\n✅ Initialization complete: {total_time:.2f}s")
    print(f"   - Base model + adapter load: {stage1_end - stage1_start:.2f}s")

    return plugin, total_time


def profile_single_inference(plugin, question, iteration=0):
    """Profile a single inference pass with detailed timing."""

    # Stage 1: Initialize state
    stage1_start = time.time()
    state = plugin.init_state(question)
    stage1_end = time.time()

    # Stage 2: Tokenization
    stage2_start = time.time()
    inputs = plugin.tokenizer(
        state['prompt'],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(plugin.device)
    stage2_end = time.time()

    # Stage 3: Forward pass (inference)
    stage3_start = time.time()
    with torch.no_grad():
        outputs = plugin.model.generate(
            **inputs,
            max_new_tokens=plugin.max_tokens,
            temperature=state['temperature'],
            do_sample=True,
            pad_token_id=plugin.tokenizer.eos_token_id
        )
    stage3_end = time.time()

    # Stage 4: Decoding
    stage4_start = time.time()
    response = plugin.tokenizer.decode(outputs[0], skip_special_tokens=True)
    stage4_end = time.time()

    # Calculate timings
    timings = {
        'init_state': stage1_end - stage1_start,
        'tokenization': stage2_end - stage2_start,
        'forward_pass': stage3_end - stage3_start,
        'decoding': stage4_end - stage4_start,
        'total': stage4_end - stage1_start
    }

    return response, timings


def profile_model(model_path, base_model=None, test_question=None):
    """Complete profiling of a model: init + inference breakdown."""

    if test_question is None:
        test_question = "What are the key components of the SAGE consciousness framework?"

    print(f"\n{'#'*80}")
    print(f"# Model: {Path(model_path).name}")
    print(f"# Base: {base_model if base_model else 'N/A (full model)'}")
    print(f"{'#'*80}")

    # Profile initialization
    plugin, init_time = profile_model_init(model_path, base_model)

    # Profile 3 inference passes (to get stable measurements)
    print(f"\n{'-'*80}")
    print(f"Profiling Inference (3 passes for stability)")
    print(f"{'-'*80}")

    all_timings = []

    for i in range(3):
        print(f"\n--- Pass {i+1}/3 ---")
        response, timings = profile_single_inference(plugin, test_question, iteration=i)
        all_timings.append(timings)

        print(f"  Init state:    {timings['init_state']*1000:>8.2f}ms")
        print(f"  Tokenization:  {timings['tokenization']*1000:>8.2f}ms")
        print(f"  Forward pass:  {timings['forward_pass']:>8.2f}s  <-- Main inference")
        print(f"  Decoding:      {timings['decoding']*1000:>8.2f}ms")
        print(f"  Total:         {timings['total']:>8.2f}s")

    # Calculate averages
    avg_timings = {
        key: sum(t[key] for t in all_timings) / len(all_timings)
        for key in all_timings[0].keys()
    }

    print(f"\n{'='*80}")
    print(f"AVERAGE TIMINGS (3 passes)")
    print(f"{'='*80}")
    print(f"  Init state:    {avg_timings['init_state']*1000:>8.2f}ms")
    print(f"  Tokenization:  {avg_timings['tokenization']*1000:>8.2f}ms")
    print(f"  Forward pass:  {avg_timings['forward_pass']:>8.2f}s  <-- 🎯 CRITICAL PATH")
    print(f"  Decoding:      {avg_timings['decoding']*1000:>8.2f}ms")
    print(f"  ────────────────────────────")
    print(f"  Total inference: {avg_timings['total']:>6.2f}s")

    # Percentage breakdown
    print(f"\n  Breakdown:")
    print(f"    Forward pass: {avg_timings['forward_pass']/avg_timings['total']*100:>5.1f}%")
    print(f"    Tokenization: {avg_timings['tokenization']/avg_timings['total']*100:>5.1f}%")
    print(f"    Decoding:     {avg_timings['decoding']/avg_timings['total']*100:>5.1f}%")
    print(f"    Init state:   {avg_timings['init_state']/avg_timings['total']*100:>5.1f}%")

    return {
        'model_path': model_path,
        'base_model': base_model,
        'init_time': init_time,
        'avg_timings': avg_timings,
        'all_timings': all_timings
    }


def compare_models(results1, results2):
    """Compare profiling results between two models."""

    print(f"\n\n{'#'*80}")
    print(f"# PERFORMANCE COMPARISON")
    print(f"{'#'*80}")

    model1_name = Path(results1['model_path']).name
    model2_name = Path(results2['model_path']).name

    print(f"\nModel 1: {model1_name}")
    print(f"Model 2: {model2_name}")

    print(f"\n{'-'*80}")
    print(f"{'Stage':<20} {'Model 1':<15} {'Model 2':<15} {'Difference':<15} {'Ratio':<10}")
    print(f"{'-'*80}")

    # Compare initialization
    init1 = results1['init_time']
    init2 = results2['init_time']
    diff = init2 - init1
    ratio = init2 / init1 if init1 > 0 else 0
    print(f"{'Initialization':<20} {init1:>8.2f}s      {init2:>8.2f}s      {diff:>+8.2f}s      {ratio:>5.2f}×")

    print(f"\n{'-'*80}")

    # Compare inference stages
    avg1 = results1['avg_timings']
    avg2 = results2['avg_timings']

    for stage in ['init_state', 'tokenization', 'forward_pass', 'decoding', 'total']:
        t1 = avg1[stage]
        t2 = avg2[stage]
        diff = t2 - t1
        ratio = t2 / t1 if t1 > 0 else 0

        # Format timing (ms for small values, s for large)
        if stage in ['init_state', 'tokenization', 'decoding']:
            t1_str = f"{t1*1000:>8.2f}ms"
            t2_str = f"{t2*1000:>8.2f}ms"
            diff_str = f"{diff*1000:>+8.2f}ms"
        else:
            t1_str = f"{t1:>8.2f}s"
            t2_str = f"{t2:>8.2f}s"
            diff_str = f"{diff:>+8.2f}s"

        stage_name = stage.replace('_', ' ').title()
        marker = "🎯" if stage == 'forward_pass' else "  "

        print(f"{marker} {stage_name:<18} {t1_str:<14} {t2_str:<14} {diff_str:<14} {ratio:>5.2f}×")

    print(f"{'-'*80}")

    # Summary
    total1 = avg1['total']
    total2 = avg2['total']
    total_diff = total2 - total1
    total_ratio = total2 / total1

    print(f"\n{'='*80}")
    print(f"REGRESSION ANALYSIS")
    print(f"{'='*80}")
    print(f"Model 1 total: {total1:.2f}s")
    print(f"Model 2 total: {total2:.2f}s")
    print(f"Difference:    {total_diff:+.2f}s ({(total_ratio-1)*100:+.1f}%)")
    print(f"Ratio:         {total_ratio:.2f}× {'SLOWER' if total_ratio > 1 else 'FASTER'}")

    # Identify bottleneck
    print(f"\n🔍 BOTTLENECK IDENTIFICATION:")
    forward_ratio = avg2['forward_pass'] / avg1['forward_pass']
    token_ratio = avg2['tokenization'] / avg1['tokenization']
    decode_ratio = avg2['decoding'] / avg1['decoding']

    print(f"   Forward pass regression: {forward_ratio:.2f}× ({(forward_ratio-1)*100:+.1f}%)")
    print(f"   Tokenization regression: {token_ratio:.2f}× ({(token_ratio-1)*100:+.1f}%)")
    print(f"   Decoding regression:     {decode_ratio:.2f}× ({(decode_ratio-1)*100:+.1f}%)")

    if forward_ratio > 1.5:
        print(f"\n   ⚠️  PRIMARY BOTTLENECK: Forward pass (inference)")
        print(f"       Model 2's forward pass is {forward_ratio:.2f}× slower")
        print(f"       This accounts for {(avg2['forward_pass']-avg1['forward_pass'])/total_diff*100:.1f}% of total regression")

    print(f"\n{'='*80}")


def main():
    """Run performance profiling comparison."""

    print("\n" + "="*80)
    print("SAGE Edge Performance Regression Profiling - Session 7")
    print("="*80)
    print("\nObjective: Identify root cause of 2× slowdown in Introspective-Qwen")
    print("\nSession 6 baseline:")
    print("  - Epistemic-Pragmatism: 51.18s avg")
    print("  - Introspective-Qwen:   102.66s avg (2× slower!)")
    print("\nThis test profiles each stage to find the bottleneck.")

    # Test question (analytical, from Session 6)
    test_question = "What are the key components of the SAGE consciousness framework?"

    # Model 1: Epistemic-Pragmatism (baseline)
    model1_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"
    model1_base = None  # Full model

    # Model 2: Introspective-Qwen (regression)
    model2_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model"
    model2_base = "Qwen/Qwen2.5-0.5B-Instruct"  # LoRA adapter

    print(f"\nTest question: \"{test_question}\"")
    print(f"\n{'='*80}\n")

    # Profile both models
    input("Press Enter to start profiling Epistemic-Pragmatism (baseline)...")
    results1 = profile_model(model1_path, model1_base, test_question)

    print(f"\n\n{'='*80}\n")
    input("Press Enter to start profiling Introspective-Qwen (regression)...")
    results2 = profile_model(model2_path, model2_base, test_question)

    # Compare results
    compare_models(results1, results2)

    print("\n✅ Profiling complete!")
    print(f"\nNext steps based on findings:")
    print(f"  1. If forward pass is bottleneck → Investigate LoRA merge overhead")
    print(f"  2. If tokenization is bottleneck → Check base model tokenizer loading")
    print(f"  3. If initialization is slow → Check base model caching")
    print(f"\nSession 7 profiling data will inform optimization strategy.\n")


if __name__ == "__main__":
    main()
