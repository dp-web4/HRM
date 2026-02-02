#!/usr/bin/env python3
"""
Phase 1: Baseline Capability Assessment (GGUF version for Sprout)

Tests phi-4-mini Q4_K_M GGUF model on policy interpretation tasks.
Uses llama-cpp-python instead of transformers.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

from llama_cpp import Llama

from test_suite import TEST_SCENARIOS, format_scenario_for_llm, evaluate_response, create_test_report


def load_phi4_mini_gguf(model_path: str):
    """Load phi-4-mini GGUF model via llama-cpp."""
    print(f"Loading GGUF model from {model_path}...")

    model = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,  # Use all GPU layers
        verbose=False
    )

    print("Model loaded successfully (GGUF via llama-cpp)")
    return model


def generate_response(model, prompt: str, max_tokens: int = 512) -> str:
    """Generate response from GGUF model."""
    output = model.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a policy interpreter. Analyze actions and provide structured decisions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9
    )

    return output['choices'][0]['message']['content']


def run_baseline_test(model_path: str, num_scenarios: int = None, save_path: str = None):
    """Run baseline capability test."""

    # Load model
    model = load_phi4_mini_gguf(model_path)

    # Select scenarios to test
    scenarios = TEST_SCENARIOS[:num_scenarios] if num_scenarios else TEST_SCENARIOS

    print(f"\nRunning baseline test on {len(scenarios)} scenarios...")
    print("=" * 60)

    results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Testing {scenario.id}: {scenario.description}")
        print(f"Difficulty: {scenario.difficulty}")

        # Format prompt
        prompt = format_scenario_for_llm(scenario)

        # Generate response
        print("Generating response...")
        response = generate_response(model, prompt)

        # Evaluate
        eval_result = evaluate_response(response, scenario)
        eval_result["response"] = response
        results.append(eval_result)

        # Print result
        status = "✓ PASS" if eval_result["passed"] else "✗ FAIL"
        print(f"Result: {status}")
        print(f"  Decision correct: {eval_result['scores']['decision_correct']}")
        print(f"  Reasoning coverage: {eval_result['scores']['reasoning_coverage']:.2f}")
        print(f"  Output structure: {eval_result['scores']['output_structure']:.2f}")

        if not eval_result["passed"]:
            print(f"\n  Expected decision: {scenario.expected_decision}")
            print(f"  Response snippet: {response[:200]}...")

    # Create report
    print("\n" + "=" * 60)
    print("BASELINE TEST REPORT (GGUF - Sprout)")
    print("=" * 60)

    report = create_test_report(results)

    print(f"\nOverall Performance:")
    print(f"  Scenarios tested: {report['total_scenarios']}")
    print(f"  Passed: {report['passed']}")
    print(f"  Failed: {report['failed']}")
    print(f"  Pass rate: {report['pass_rate']:.1%}")

    print(f"\nPerformance by difficulty:")
    for diff, stats in sorted(report['by_difficulty'].items()):
        rate = stats['passed'] / stats['total']
        print(f"  {diff}: {stats['passed']}/{stats['total']} ({rate:.1%})")

    print(f"\nAverage Scores:")
    for metric, score in report['average_scores'].items():
        print(f"  {metric}: {score:.1%}")

    # Save results
    if save_path:
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "model_type": "GGUF Q4_K_M",
            "machine": "Sprout (Jetson Orin Nano)",
            "num_scenarios": len(scenarios),
            "report": report,
            "results": results
        }

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\nResults saved to: {save_path}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline policy interpretation test (GGUF)")
    parser.add_argument(
        "--model",
        default="/home/sprout/ai-workspace/HRM/model-zoo/phi-4-mini-instruct-gguf/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        help="Path to phi-4-mini GGUF model"
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        help="Number of scenarios to test (default: all)"
    )
    parser.add_argument(
        "--save",
        default="/home/sprout/ai-workspace/HRM/policy/results/baseline_test_sprout.json",
        help="Path to save results"
    )

    args = parser.parse_args()

    run_baseline_test(
        model_path=args.model,
        num_scenarios=args.num_scenarios,
        save_path=args.save
    )
