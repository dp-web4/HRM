#!/usr/bin/env python3
"""
Test v4_hybrid on Attack-Based Scenarios (Session M)

Tests the production-ready v4_hybrid prompt on sophisticated attack scenarios
derived from web4/hardbound/attack_simulations.py.

Goals:
1. Validate v4_hybrid generalizes to real-world attacks
2. Assess reasoning quality on sophisticated threats
3. Identify any gaps in prompt design
4. Provide integration context for hardbound deployment

Usage:
    python3 test_attack_scenarios.py                    # Test all scenarios
    python3 test_attack_scenarios.py --scenario A02     # Test specific scenario
    python3 test_attack_scenarios.py --save results/attack_test.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from test_suite_attack_scenarios import get_attack_scenarios, get_scenario_by_id, AttackScenario
from test_suite_semantic import PolicyScenario, evaluate_response_semantic
from prompts_v4 import build_prompt_v4

try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python not installed")
    print("Install with: pip install llama-cpp-python")
    sys.exit(1)


# Model path
MODEL_PATH = Path("/home/dp/ai-workspace/HRM/model-zoo/phi-4-mini-gguf/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf")


def load_model(verbose: bool = False) -> Llama:
    """Load the Phi-4-mini model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print(f"Loading model: {MODEL_PATH.name}")
    start = time.time()

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=8192,
        n_gpu_layers=-1,  # Auto-detect GPU
        verbose=verbose
    )

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.2f}s")

    return llm


def test_scenario(llm: Llama, scenario: AttackScenario, verbose: bool = False) -> Dict[str, Any]:
    """
    Test v4_hybrid on a single attack scenario.

    Returns:
        Result dict with decision, reasoning, and evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Testing {scenario.id}: {scenario.name}")
    print(f"Risk: {scenario.risk_level.upper()}, Expected: {scenario.expected_decision}")
    print(f"{'='*80}")

    # Convert AttackScenario to PolicyScenario format for evaluation
    test_scenario = PolicyScenario(
        id=scenario.id,
        description=scenario.name,
        situation=scenario.situation,
        expected_classification=scenario.category,
        expected_decision=scenario.expected_decision,
        expected_reasoning_elements=scenario.expected_reasoning_keywords,
        difficulty="hard"  # All attack scenarios are hard difficulty
    )

    # Build prompt using v4_hybrid
    prompt = build_prompt_v4(scenario.situation, variant="hybrid")

    if verbose:
        print(f"\n[Prompt Preview (first 500 chars)]:")
        print(prompt[:500] + "...")

    # Generate response
    print(f"\nGenerating response...")
    start = time.time()

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a policy interpreter for enterprise governance systems."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.7,
        stop=["</decision>", "\n\n\n"]
    )

    elapsed = time.time() - start
    print(f"Response generated in {elapsed:.2f}s")

    # Extract response text
    response_text = response['choices'][0]['message']['content'].strip()

    if verbose:
        print(f"\n[Full Response]:")
        print(response_text)
        print()

    # Evaluate using semantic evaluation (Session I threshold: 0.35)
    eval_result = evaluate_response_semantic(response_text, test_scenario, similarity_threshold=0.35)

    # Extract key metrics
    decision_correct = eval_result['scores']['decision_correct']
    reasoning_coverage = eval_result['scores']['reasoning_coverage_semantic']
    overall_pass = eval_result['passed']

    # Extract decision from response (simple parsing)
    extracted_decision = "unknown"
    response_lower = response_text.lower()
    if "allow" in response_lower and "deny" not in response_lower[:response_lower.find("allow") if "allow" in response_lower else 0]:
        extracted_decision = "allow"
    elif "deny" in response_lower:
        extracted_decision = "deny"
    elif "attestation" in response_lower or "require" in response_lower:
        extracted_decision = "require_attestation"

    # Print results
    print(f"\n{'─'*80}")
    print(f"Results:")
    print(f"  Decision: {extracted_decision} (expected: {scenario.expected_decision})")
    print(f"  Match: {'✓' if decision_correct else '✗'}")
    print(f"  Reasoning Coverage: {reasoning_coverage:.1%} (semantic similarity)")
    print(f"  Pass: {'✓' if overall_pass else '✗'} (threshold: 0.35)")
    print(f"  Overall: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
    print(f"{'─'*80}")

    return {
        "scenario_id": scenario.id,
        "scenario_name": scenario.name,
        "attack_type": scenario.attack_type,
        "risk_level": scenario.risk_level,
        "expected_decision": scenario.expected_decision,
        "extracted_decision": extracted_decision,
        "decision_match": decision_correct,
        "reasoning_coverage": reasoning_coverage,
        "reasoning_pass": overall_pass,
        "overall_pass": overall_pass,
        "response_text": response_text,
        "inference_time": elapsed,
        "eval_details": eval_result
    }


def run_test_suite(
    scenario_ids: List[str] = None,
    verbose: bool = False,
    save_path: str = None
) -> Dict[str, Any]:
    """
    Run v4_hybrid tests on attack scenarios.

    Args:
        scenario_ids: List of scenario IDs to test (None = all)
        verbose: Print detailed output
        save_path: Path to save results JSON

    Returns:
        Test results summary
    """
    print("=" * 80)
    print("V4_HYBRID ATTACK SCENARIO TESTING (Session M)")
    print("=" * 80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Prompt: v4_hybrid (5 examples)")
    print()

    # Load model
    llm = load_model(verbose=verbose)

    # Get scenarios
    if scenario_ids:
        scenarios = [get_scenario_by_id(sid) for sid in scenario_ids]
    else:
        scenarios = get_attack_scenarios()

    print(f"\nTesting {len(scenarios)} scenario(s)...")

    # Run tests
    results = []
    for scenario in scenarios:
        result = test_scenario(llm, scenario, verbose=verbose)
        results.append(result)

    # Compute summary statistics
    total = len(results)
    decision_correct = sum(1 for r in results if r["decision_match"])
    reasoning_pass = sum(1 for r in results if r["reasoning_pass"])
    overall_pass = sum(1 for r in results if r["overall_pass"])

    avg_coverage = sum(r["reasoning_coverage"] for r in results) / total
    avg_inference = sum(r["inference_time"] for r in results) / total

    summary = {
        "test_suite": "attack_scenarios",
        "prompt_variant": "v4_hybrid",
        "model": MODEL_PATH.name,
        "timestamp": datetime.now().isoformat(),
        "total_scenarios": total,
        "decision_accuracy": decision_correct / total,
        "reasoning_pass_rate": reasoning_pass / total,
        "overall_pass_rate": overall_pass / total,
        "avg_reasoning_coverage": avg_coverage,
        "avg_inference_time": avg_inference,
        "results": results
    }

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total Scenarios: {total}")
    print(f"Decision Accuracy: {decision_correct}/{total} ({summary['decision_accuracy']:.1%})")
    print(f"Reasoning Pass Rate: {reasoning_pass}/{total} ({summary['reasoning_pass_rate']:.1%})")
    print(f"Overall Pass Rate: {overall_pass}/{total} ({summary['overall_pass_rate']:.1%})")
    print(f"Avg Reasoning Coverage: {avg_coverage:.1%}")
    print(f"Avg Inference Time: {avg_inference:.2f}s")
    print(f"{'='*80}")

    # Per-scenario breakdown
    print(f"\nPer-Scenario Results:")
    print(f"{'─'*80}")
    for r in results:
        status = "PASS ✓" if r["overall_pass"] else "FAIL ✗"
        print(f"{r['scenario_id']}: {status} - {r['scenario_name']}")
        print(f"  Decision: {r['extracted_decision']} ({'✓' if r['decision_match'] else '✗'})")
        print(f"  Coverage: {r['reasoning_coverage']:.1%}")
        print(f"  Attack Type: {r['attack_type']}")
        print(f"  Risk Level: {r['risk_level'].upper()}")
        if not r["overall_pass"]:
            if not r["decision_match"]:
                print(f"  ⚠️  Decision mismatch: expected {r['expected_decision']}")
            if not r["reasoning_pass"]:
                print(f"  ⚠️  Low reasoning coverage: {r['reasoning_coverage']:.1%} < 35%")
    print(f"{'─'*80}")

    # Save results
    if save_path:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {save_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Test v4_hybrid on attack scenarios")
    parser.add_argument(
        '--scenario', '-s',
        type=str,
        help="Test specific scenario (e.g., A02)"
    )
    parser.add_argument(
        '--save',
        type=str,
        help="Save results to JSON file"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Print detailed output"
    )

    args = parser.parse_args()

    # Determine which scenarios to test
    scenario_ids = [args.scenario] if args.scenario else None

    # Run tests
    run_test_suite(
        scenario_ids=scenario_ids,
        verbose=args.verbose,
        save_path=args.save
    )


if __name__ == "__main__":
    main()
