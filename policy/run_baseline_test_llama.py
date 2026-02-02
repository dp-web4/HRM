#!/usr/bin/env python3
"""
Phase 1: Baseline Capability Assessment (llama-cpp version)

Tests phi-4-mini base model on policy interpretation tasks using llama-cpp.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

from llama_cpp import Llama

from test_suite import TEST_SCENARIOS, format_scenario_for_llm, evaluate_response, create_test_report


def load_phi4_mini_gguf(model_path: str, n_ctx: int = 2048, n_threads: int = None):
    """Load phi-4-mini GGUF model with llama-cpp."""
    print(f"Loading GGUF model from {model_path}...")

    # Auto-detect threads if not specified
    if n_threads is None:
        import os
        n_threads = os.cpu_count() or 4

    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=-1,  # Use GPU if available (llama-cpp will handle gracefully)
        verbose=False
    )

    print(f"Model loaded successfully")
    print(f"  Context size: {n_ctx}")
    print(f"  Threads: {n_threads}")

    return model


def generate_response(model, prompt: str, max_tokens: int = 512) -> str:
    """Generate response from model."""
    output = model(
        prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        stop=["</s>", "\n\n\n"],  # Stop sequences
        echo=False
    )

    response = output['choices'][0]['text'].strip()
    return response


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
    print("BASELINE TEST REPORT")
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
            "model_type": "llama-cpp GGUF",
            "num_scenarios": len(scenarios),
            "report": report
        }

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\nResults saved to: {save_path}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline policy interpretation test (llama-cpp)")
    parser.add_argument(
        "--model",
        default="/home/dp/ai-workspace/HRM/model-zoo/phi-4-mini-gguf/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        help="Number of scenarios to test (default: all)"
    )
    parser.add_argument(
        "--save",
        default="/home/dp/ai-workspace/HRM/policy/results/baseline_test_llama.json",
        help="Path to save results"
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=2048,
        help="Context size (default: 2048)"
    )

    args = parser.parse_args()

    run_baseline_test(
        model_path=args.model,
        num_scenarios=args.num_scenarios,
        save_path=args.save
    )
