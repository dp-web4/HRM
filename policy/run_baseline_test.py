#!/usr/bin/env python3
"""
Phase 1: Baseline Capability Assessment

Tests phi-4-mini base model on policy interpretation tasks.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from test_suite import TEST_SCENARIOS, format_scenario_for_llm, evaluate_response, create_test_report


def load_phi4_mini(model_path: str):
    """Load phi-4-mini model."""
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    if device == "cpu":
        model = model.to(device)

    print(f"Model loaded successfully on {device}")
    return tokenizer, model, device


def generate_response(tokenizer, model, device, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the model's response (after the prompt)
    if prompt in response:
        response = response.split(prompt, 1)[1].strip()

    return response


def run_baseline_test(model_path: str, num_scenarios: int = None, save_path: str = None):
    """Run baseline capability test."""

    # Load model
    tokenizer, model, device = load_phi4_mini(model_path)

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
        response = generate_response(tokenizer, model, device, prompt)

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
            "device": device,
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

    parser = argparse.ArgumentParser(description="Run baseline policy interpretation test")
    parser.add_argument(
        "--model",
        default="/home/dp/ai-workspace/HRM/model-zoo/phi-4-mini",
        help="Path to phi-4-mini model"
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        help="Number of scenarios to test (default: all)"
    )
    parser.add_argument(
        "--save",
        default="/home/dp/ai-workspace/HRM/policy/results/baseline_test.json",
        help="Path to save results"
    )

    args = parser.parse_args()

    run_baseline_test(
        model_path=args.model,
        num_scenarios=args.num_scenarios,
        save_path=args.save
    )
