#!/usr/bin/env python3
"""
Test v2_fewshot prompt on full 8-scenario suite.

Based on quick test results showing v2_fewshot achieved 100% pass rate,
now validate on complete test suite.
"""

import json
import time
from pathlib import Path
from llama_cpp import Llama

from test_suite_semantic import (
    TEST_SCENARIOS,
    evaluate_response_semantic,
    create_test_report
)
from prompts_v2 import build_prompt_v2


def load_model(model_path: str):
    """Load the phi-4-mini GGUF model."""
    print(f"Loading model from {model_path}...")
    start = time.time()

    llm = Llama(
        model_path=model_path,
        n_ctx=8192,
        n_gpu_layers=-1,
        verbose=False
    )

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s")
    return llm


def test_full_suite(llm: Llama, save_path: str = "results/v2_fewshot_full.json"):
    """Test v2_fewshot on all 8 scenarios."""

    print(f"\n{'='*70}")
    print("Testing v2_fewshot on FULL SUITE (8 scenarios)")
    print(f"{'='*70}\n")

    results = []
    start_time = time.time()

    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        print(f"\n[{i}/8] Testing {scenario.id}: {scenario.description}")
        print(f"  Difficulty: {scenario.difficulty}")

        # Build prompt
        prompt = build_prompt_v2(
            scenario.situation,
            variant="fewshot",
            context=scenario.situation.get("team_context", "")
        )

        # Generate response
        print("  Generating response...")
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a policy interpreter. Analyze actions and provide structured decisions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9
        )

        response_text = output['choices'][0]['message']['content'].strip()

        # Evaluate with adjusted threshold (0.49 to reduce false negatives)
        result = evaluate_response_semantic(response_text, scenario, similarity_threshold=0.49)
        result['response'] = response_text
        result['variant'] = 'v2_fewshot'
        results.append(result)

        # Quick feedback
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        decision_status = "✓" if result['scores']['decision_correct'] else "✗"
        reasoning = result['scores']['reasoning_coverage_semantic']
        print(f"  Result: {status}")
        print(f"    Decision: {decision_status} ({scenario.expected_decision})")
        print(f"    Reasoning coverage: {reasoning:.2f}")
        print(f"    Structure: {result['scores']['output_structure']:.2f}")

    elapsed = time.time() - start_time

    # Create report
    report = create_test_report(results)
    report['variant'] = 'v2_fewshot'
    report['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report['total_time_seconds'] = elapsed
    report['avg_time_per_scenario'] = elapsed / len(TEST_SCENARIOS)

    # Save results
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}\n")
    print(f"Pass rate: {report['pass_rate']:.1%}")
    print(f"Decision accuracy: {report['average_scores']['decision_correct']:.1%}")
    print(f"Reasoning coverage (semantic): {report['average_scores']['reasoning_coverage_semantic']:.3f}")
    print(f"Output structure: {report['average_scores']['output_structure']:.3f}")
    print(f"\nBy difficulty:")
    for diff, stats in report['by_difficulty'].items():
        print(f"  {diff}: {stats['passed']}/{stats['total']} passed")
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/len(TEST_SCENARIOS):.1f}s per scenario)")
    print(f"Results saved to: {save_path}")

    return report


if __name__ == "__main__":
    model_path = "/home/dp/ai-workspace/HRM/model-zoo/phi-4-mini-gguf/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

    llm = load_model(model_path)
    report = test_full_suite(llm)

    print("\n" + "="*70)
    print("v2_fewshot FULL SUITE TEST COMPLETE")
    print("="*70)
