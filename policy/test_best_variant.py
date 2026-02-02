#!/usr/bin/env python3
"""
Test the best-performing variant (v2_fewshot) on full test suite.

Based on quick test results:
- v2_fewshot: 100% pass rate (3/3 scenarios)
- v2_explicit: 66.7% pass rate
- v2_checklist: 66.7% pass rate
- v1_baseline: 33.3% pass rate

Now test v2_fewshot on all 8 scenarios.
"""

import json
import time
from llama_cpp import Llama
from test_suite_semantic import TEST_SCENARIOS, evaluate_response_semantic, create_test_report
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
    print(f"Model loaded in {elapsed:.1f}s\n")
    return llm


def test_fewshot_full(llm: Llama):
    """Test v2_fewshot on all 8 scenarios."""
    print("="*70)
    print("TESTING v2_fewshot ON FULL SUITE (8 scenarios)")
    print("="*70)
    print()

    results = []

    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        print(f"[{i}/8] {scenario.id} - {scenario.description}")
        print(f"  Difficulty: {scenario.difficulty}")

        # Build few-shot prompt
        prompt = build_prompt_v2(
            scenario.situation,
            variant="fewshot",
            context=scenario.situation.get("team_context", "")
        )

        # Generate response
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

        # Evaluate
        result = evaluate_response_semantic(response_text, scenario, similarity_threshold=0.5)
        result['response'] = response_text
        results.append(result)

        # Feedback
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        decision_status = "✓" if result['scores']['decision_correct'] else "✗"
        reasoning = result['scores']['reasoning_coverage_semantic']

        print(f"  {status}")
        print(f"    Decision: {decision_status}")
        print(f"    Reasoning coverage: {reasoning:.2f}")
        print(f"    Expected: {scenario.expected_decision}")
        print()

    # Create report
    report = create_test_report(results)
    report['variant'] = 'v2_fewshot'
    report['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Save results
    output_file = "results/v2_fewshot_full_test.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    # Summary
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print()
    print(f"Pass rate: {report['pass_rate']:.1%} ({report['passed']}/{report['total_scenarios']})")
    print(f"\nAverage scores:")
    print(f"  Decision accuracy: {report['average_scores']['decision_correct']:.3f}")
    print(f"  Reasoning coverage: {report['average_scores']['reasoning_coverage_semantic']:.3f}")
    print(f"  Output structure: {report['average_scores']['output_structure']:.3f}")
    print(f"\nBy difficulty:")
    for diff, stats in report['by_difficulty'].items():
        print(f"  {diff}: {stats['passed']}/{stats['total']} passed")

    print(f"\nResults saved to: {output_file}")

    # Detailed failures
    failures = [r for r in results if not r['passed']]
    if failures:
        print(f"\n{'='*70}")
        print(f"FAILED SCENARIOS ({len(failures)})")
        print(f"{'='*70}\n")
        for fail in failures:
            print(f"[{fail['scenario_id']}] {fail['difficulty']}")
            print(f"  Decision correct: {fail['scores']['decision_correct']}")
            print(f"  Reasoning coverage: {fail['scores']['reasoning_coverage_semantic']:.2f}")
            print(f"  Missing reasoning elements:")
            for detail in fail['reasoning_details']:
                if not detail['present']:
                    print(f"    - {detail['expected']} (sim: {detail['similarity']:.3f})")
            print()

    return report


if __name__ == "__main__":
    model_path = "/home/dp/ai-workspace/HRM/model-zoo/phi-4-mini-gguf/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

    llm = load_model(model_path)
    report = test_fewshot_full(llm)

    if report['pass_rate'] >= 0.75:
        print("\n🎉 SUCCESS! v2_fewshot achieves ≥75% pass rate")
    elif report['pass_rate'] >= 0.625:
        print("\n✅ GOOD! v2_fewshot achieves ≥62.5% pass rate (5/8)")
    else:
        print("\n⚠️  NEEDS IMPROVEMENT. Consider additional prompt tuning.")
