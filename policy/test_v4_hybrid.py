#!/usr/bin/env python3
"""
Session K: Test v4_hybrid prompt

Goal: Eliminate EC01 vs M02 trade-off with 5-example hybrid prompt.

Test:
- v3_condensed (4ex) - Session J baseline
- v4_hybrid (5ex) - New hybrid

Target: EC01=100% AND M02=100% (eliminate trade-off)
"""

import time
from llama_cpp import Llama

from test_suite_semantic import TEST_SCENARIOS, evaluate_response_semantic
from prompts_v3 import build_prompt_v3
from prompts_v4 import build_prompt_v4


def test_variant(llm, variant_name, prompt_builder, scenarios):
    """Test a single variant on all scenarios."""
    print(f"\n{'='*70}")
    print(f"Testing: {variant_name}")
    print(f"{'='*70}\n")

    results = []
    for scenario in scenarios:
        # Build prompt
        prompt = prompt_builder(scenario.situation)

        # Generate
        print(f"[{scenario.id}] Generating...")
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a policy interpreter."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9
        )

        response = output['choices'][0]['message']['content'].strip()

        # Evaluate (Session I threshold: 0.35)
        result = evaluate_response_semantic(response, scenario, similarity_threshold=0.35)

        # Report
        coverage = result['scores']['reasoning_coverage_semantic']
        status = "PASS" if result['passed'] else "FAIL"
        print(f"  {scenario.id}: {coverage:.1%} coverage - {status}")

        results.append({
            'scenario_id': scenario.id,
            'coverage': coverage,
            'passed': result['passed'],
            'decision_correct': result['scores']['decision_correct']
        })

    # Summary
    passed = sum(1 for r in results if r['passed'])
    avg_coverage = sum(r['coverage'] for r in results) / len(results)
    decisions = sum(1 for r in results if r['decision_correct']) / len(results)

    print(f"\n{variant_name} Summary:")
    print(f"  Pass rate: {passed}/{len(scenarios)} ({passed/len(scenarios):.1%})")
    print(f"  Decision accuracy: {decisions:.1%}")
    print(f"  Avg reasoning coverage: {avg_coverage:.1%}")

    return results


def compare_on_key_scenarios(v3_results, v4_results):
    """Compare EC01 and M02 specifically."""
    print(f"\n{'='*70}")
    print("KEY SCENARIO COMPARISON (EC01 & M02)")
    print(f"{'='*70}\n")

    print(f"{'Scenario':<10} {'v3_condensed':<15} {'v4_hybrid':<15} {'Change':<10}")
    print("-" * 60)

    for sid in ['EC01', 'M02']:
        v3 = [r for r in v3_results if r['scenario_id'] == sid][0]
        v4 = [r for r in v4_results if r['scenario_id'] == sid][0]

        v3_cov = v3['coverage']
        v4_cov = v4['coverage']
        change = v4_cov - v3_cov

        change_str = f"{change:+.1%}" if change != 0 else "same"

        print(f"{sid:<10} {v3_cov:<14.1%} {v4_cov:<14.1%} {change_str:<10}")

    print()

    # Check if goal achieved
    ec01_v4 = [r for r in v4_results if r['scenario_id'] == 'EC01'][0]
    m02_v4 = [r for r in v4_results if r['scenario_id'] == 'M02'][0]

    if ec01_v4['coverage'] == 1.0 and m02_v4['coverage'] == 1.0:
        print("🎯 GOAL ACHIEVED! Both EC01=100% AND M02=100%")
    elif ec01_v4['coverage'] == 1.0:
        print("✓ EC01=100% (maintained)")
        print(f"⚠ M02={m02_v4['coverage']:.1%} (still needs work)")
    elif m02_v4['coverage'] == 1.0:
        print("✓ M02=100% (improved)")
        print(f"⚠ EC01={ec01_v4['coverage']:.1%} (regressed)")
    else:
        print(f"⚠ EC01={ec01_v4['coverage']:.1%}, M02={m02_v4['coverage']:.1%}")
        print("Trade-off not eliminated")


def main():
    print("Session K: V4 Hybrid Prompt Testing")
    print("="*70)

    # Load model
    model_path = "/home/dp/ai-workspace/HRM/model-zoo/phi-4-mini-gguf/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
    print(f"\nLoading model from {model_path}...")
    start = time.time()
    llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=-1, verbose=False)
    print(f"Model loaded in {time.time() - start:.1f}s\n")

    # Test both variants
    scenarios = TEST_SCENARIOS

    # v3_condensed (Session J baseline - 4 examples)
    v3_results = test_variant(
        llm,
        "v3_condensed (4ex)",
        lambda sit: build_prompt_v3(sit, variant="condensed"),
        scenarios
    )

    # v4_hybrid (new - 5 examples)
    v4_results = test_variant(
        llm,
        "v4_hybrid (5ex)",
        lambda sit: build_prompt_v4(sit, variant="hybrid"),
        scenarios
    )

    # Compare on key scenarios
    compare_on_key_scenarios(v3_results, v4_results)

    # Full comparison
    print(f"\n{'='*70}")
    print("FULL SCENARIO COMPARISON")
    print(f"{'='*70}\n")

    print(f"{'Scenario':<10} {'v3_condensed':<15} {'v4_hybrid':<15} {'Change':<10}")
    print("-" * 60)

    for scenario in scenarios:
        sid = scenario.id
        v3 = [r for r in v3_results if r['scenario_id'] == sid][0]
        v4 = [r for r in v4_results if r['scenario_id'] == sid][0]

        v3_cov = v3['coverage']
        v4_cov = v4['coverage']
        change = v4_cov - v3_cov

        change_str = f"{change:+.1%}" if change != 0 else "same"
        print(f"{sid:<10} {v3_cov:<14.1%} {v4_cov:<14.1%} {change_str:<10}")

    print(f"\n{'='*70}")
    print("SESSION K COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
