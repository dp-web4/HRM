#!/usr/bin/env python3
"""
Test different prompt variants to find optimal approach.

Compares:
- v1 (baseline): Original prompts from prompts.py
- v2_explicit: Step-by-step checking prompts
- v2_fewshot: Example-based prompts
- v2_checklist: Checkbox-style prompts

Measures improvement in reasoning coverage and pass rate.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
from llama_cpp import Llama

from test_suite_semantic import (
    TEST_SCENARIOS,
    evaluate_response_semantic,
    create_test_report
)
from prompts import build_hardbound_prompt
from prompts_v2 import build_prompt_v2


def load_model(model_path: str):
    """Load the phi-4-mini GGUF model."""
    print(f"Loading model from {model_path}...")
    start = time.time()

    llm = Llama(
        model_path=model_path,
        n_ctx=8192,  # Context window
        n_gpu_layers=-1,  # Use all GPU layers if available
        verbose=False
    )

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s")
    return llm


def test_scenario_with_prompt(
    llm: Llama,
    scenario,
    prompt_builder,
    variant_name: str,
    temperature: float = 0.7,
    max_tokens: int = 512
) -> Dict[str, Any]:
    """
    Test a single scenario with a specific prompt variant.

    Args:
        llm: Loaded model
        scenario: PolicyScenario to test
        prompt_builder: Function that builds prompt from situation
        variant_name: Name of this variant (for logging)
        temperature: Sampling temperature
        max_tokens: Max response length

    Returns:
        Evaluation result dict
    """
    # Build prompt
    prompt = prompt_builder(scenario.situation, scenario.situation.get("team_context", ""))

    # Generate response using chat completion API
    print(f"  [{scenario.id}] Generating response with {variant_name}...")
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a policy interpreter. Analyze actions and provide structured decisions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9
    )

    response_text = output['choices'][0]['message']['content'].strip()

    # Evaluate
    result = evaluate_response_semantic(response_text, scenario, similarity_threshold=0.5)
    result['response'] = response_text
    result['variant'] = variant_name

    return result


def test_variant(
    llm: Llama,
    variant_name: str,
    prompt_builder,
    scenarios: List = None,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Test a prompt variant on all scenarios.

    Args:
        llm: Loaded model
        variant_name: Name of this variant
        prompt_builder: Function to build prompts
        scenarios: List of scenarios to test (default: all TEST_SCENARIOS)
        temperature: Sampling temperature

    Returns:
        Test report with results
    """
    if scenarios is None:
        scenarios = TEST_SCENARIOS

    print(f"\n{'='*70}")
    print(f"Testing variant: {variant_name}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"{'='*70}\n")

    results = []
    for scenario in scenarios:
        result = test_scenario_with_prompt(
            llm, scenario, prompt_builder, variant_name, temperature
        )
        results.append(result)

        # Quick feedback
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        decision_status = "✓" if result['scores']['decision_correct'] else "✗"
        reasoning = result['scores']['reasoning_coverage_semantic']
        print(f"    {status} - Decision: {decision_status}, Reasoning: {reasoning:.2f}")

    # Create report
    report = create_test_report(results)
    report['variant'] = variant_name
    report['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")

    return report


def compare_variants(
    llm: Llama,
    scenarios: List = None,
    save_dir: str = "results/prompt_variants"
):
    """
    Compare all prompt variants systematically.

    Args:
        llm: Loaded model
        scenarios: Scenarios to test (default: first 3 for quick test)
        save_dir: Directory to save results
    """
    if scenarios is None:
        # Quick test: first 3 scenarios
        scenarios = TEST_SCENARIOS[:3]

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    variants = {
        "v1_baseline": lambda sit, ctx: build_hardbound_prompt(sit, ctx),
        "v2_explicit": lambda sit, ctx: build_prompt_v2(sit, variant="explicit", context=ctx),
        "v2_fewshot": lambda sit, ctx: build_prompt_v2(sit, variant="fewshot", context=ctx),
        "v2_checklist": lambda sit, ctx: build_prompt_v2(sit, variant="checklist", context=ctx),
    }

    all_reports = {}

    for variant_name, prompt_builder in variants.items():
        report = test_variant(llm, variant_name, prompt_builder, scenarios)
        all_reports[variant_name] = report

        # Save individual variant results
        output_file = save_path / f"{variant_name}.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved {variant_name} results to {output_file}")

    # Comparison summary
    print(f"\n{'='*70}")
    print("VARIANT COMPARISON SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Variant':<20} {'Pass Rate':<12} {'Decision':<12} {'Reasoning':<12} {'Structure':<12}")
    print("-" * 70)

    for variant_name in variants.keys():
        report = all_reports[variant_name]
        pass_rate = report['pass_rate']
        decision = report['average_scores']['decision_correct']
        reasoning = report['average_scores']['reasoning_coverage_semantic']
        structure = report['average_scores']['output_structure']

        print(f"{variant_name:<20} {pass_rate:<12.1%} {decision:<12.3f} {reasoning:<12.3f} {structure:<12.3f}")

    # Save comparison
    comparison_file = save_path / "comparison_summary.json"
    with open(comparison_file, 'w') as f:
        json.dump({
            "variants": list(variants.keys()),
            "num_scenarios": len(scenarios),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reports": all_reports
        }, f, indent=2)

    print(f"\nSaved comparison to {comparison_file}")

    # Detailed analysis
    print(f"\n{'='*70}")
    print("DETAILED ANALYSIS")
    print(f"{'='*70}\n")

    for variant_name, report in all_reports.items():
        print(f"\n{variant_name}:")
        print(f"  Pass rate: {report['pass_rate']:.1%}")
        print(f"  By difficulty:")
        for diff, stats in report['by_difficulty'].items():
            print(f"    {diff}: {stats['passed']}/{stats['total']} passed")

    return all_reports


if __name__ == "__main__":
    import sys

    # Model path
    model_path = "/home/dp/ai-workspace/HRM/model-zoo/phi-4-mini-gguf/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

    # Parse args
    quick = "--quick" in sys.argv
    full = "--full" in sys.argv

    if full:
        print("Running FULL test (8 scenarios)...")
        scenarios = TEST_SCENARIOS
    else:
        print("Running QUICK test (3 scenarios)...")
        print("Use --full for all 8 scenarios")
        scenarios = TEST_SCENARIOS[:3]

    # Load model once
    llm = load_model(model_path)

    # Compare all variants
    print("\nStarting prompt variant comparison...")
    results = compare_variants(llm, scenarios=scenarios)

    print("\n" + "="*70)
    print("PROMPT VARIANT TESTING COMPLETE")
    print("="*70)
    print("\nResults saved to: results/prompt_variants/")
    print("\nRecommendation: Choose variant with highest reasoning coverage + pass rate")
