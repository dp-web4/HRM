#!/usr/bin/env python3
"""
Session N: Comprehensive v5 Testing

Tests v5 (attack-aware) on both basic and attack scenarios to ensure:
1. No regression on basic scenarios (maintain 100% pass rate)
2. Improvement on attack scenarios (especially A02 Sybil detection)

Usage:
    python3 test_v5_comprehensive.py                # Test both suites
    python3 test_v5_comprehensive.py --basic-only   # Just basic scenarios
    python3 test_v5_comprehensive.py --attack-only  # Just attack scenarios
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from test_suite_semantic import get_test_scenarios as get_basic_scenarios, evaluate_response_semantic, PolicyScenario
from test_suite_attack_scenarios import get_attack_scenarios, AttackScenario
from prompts_v5 import build_prompt_v5

try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python not installed")
    sys.exit(1)

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
        n_gpu_layers=-1,
        verbose=verbose
    )

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.2f}s")
    return llm


def test_basic_scenario(llm: Llama, scenario: PolicyScenario, verbose: bool = False) -> Dict[str, Any]:
    """Test v5 on a basic scenario."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing {scenario.id}: {scenario.description}")
        print(f"{'='*80}")

    # Build v5 prompt
    prompt = build_prompt_v5(scenario.situation, variant="attack_aware")

    # Generate
    start = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a policy interpreter for enterprise governance."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.7,
        stop=["</decision>", "\n\n\n"]
    )
    elapsed = time.time() - start

    response_text = response['choices'][0]['message']['content'].strip()

    if verbose:
        print(f"\nResponse ({elapsed:.2f}s):")
        print(response_text[:300] + "..." if len(response_text) > 300 else response_text)

    # Evaluate
    eval_result = evaluate_response_semantic(response_text, scenario, similarity_threshold=0.35)

    return {
        "scenario_id": scenario.id,
        "description": scenario.description,
        "expected_decision": scenario.expected_decision,
        "decision_correct": eval_result['scores']['decision_correct'],
        "reasoning_coverage": eval_result['scores']['reasoning_coverage_semantic'],
        "passed": eval_result['passed'],
        "inference_time": elapsed,
        "response_text": response_text
    }


def test_attack_scenario(llm: Llama, scenario: AttackScenario, verbose: bool = False) -> Dict[str, Any]:
    """Test v5 on an attack scenario."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing {scenario.id}: {scenario.name}")
        print(f"Risk: {scenario.risk_level.upper()}")
        print(f"{'='*80}")

    # Convert to PolicyScenario for evaluation
    policy_scenario = PolicyScenario(
        id=scenario.id,
        description=scenario.name,
        situation=scenario.situation,
        expected_classification=scenario.category,
        expected_decision=scenario.expected_decision,
        expected_reasoning_elements=scenario.expected_reasoning_keywords,
        difficulty="hard"
    )

    # Build v5 prompt
    prompt = build_prompt_v5(scenario.situation, variant="attack_aware")

    # Generate
    start = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a policy interpreter for enterprise governance."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.7,
        stop=["</decision>", "\n\n\n"]
    )
    elapsed = time.time() - start

    response_text = response['choices'][0]['message']['content'].strip()

    if verbose:
        print(f"\nResponse ({elapsed:.2f}s):")
        print(response_text[:300] + "..." if len(response_text) > 300 else response_text)

    # Evaluate
    eval_result = evaluate_response_semantic(response_text, policy_scenario, similarity_threshold=0.35)

    # Extract decision
    extracted_decision = "unknown"
    response_lower = response_text.lower()
    if "allow" in response_lower and "deny" not in response_lower[:response_lower.find("allow") if "allow" in response_lower else 0]:
        extracted_decision = "allow"
    elif "deny" in response_lower:
        extracted_decision = "deny"
    elif "attestation" in response_lower or "require" in response_lower:
        extracted_decision = "require_attestation"

    return {
        "scenario_id": scenario.id,
        "name": scenario.name,
        "attack_type": scenario.attack_type,
        "risk_level": scenario.risk_level,
        "expected_decision": scenario.expected_decision,
        "extracted_decision": extracted_decision,
        "decision_correct": eval_result['scores']['decision_correct'],
        "reasoning_coverage": eval_result['scores']['reasoning_coverage_semantic'],
        "passed": eval_result['passed'],
        "inference_time": elapsed,
        "response_text": response_text
    }


def run_comprehensive_test(basic_only: bool = False, attack_only: bool = False, verbose: bool = False):
    """Run comprehensive v5 testing."""
    print("="*80)
    print("V5 COMPREHENSIVE TESTING (Session N)")
    print("="*80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Prompt: v5 attack-aware (v4_hybrid + attack indicators)")
    print()

    llm = load_model(verbose=verbose)

    results = {
        "basic": [],
        "attack": []
    }

    # Test basic scenarios
    if not attack_only:
        print(f"\n{'='*80}")
        print("TESTING BASIC SCENARIOS (8 scenarios)")
        print("Goal: Maintain 100% pass rate from v4_hybrid")
        print(f"{'='*80}")

        basic_scenarios = get_basic_scenarios()
        for i, scenario in enumerate(basic_scenarios, 1):
            print(f"\n[{i}/{len(basic_scenarios)}] {scenario.id}...", end=" ", flush=True)
            result = test_basic_scenario(llm, scenario, verbose=verbose)
            results["basic"].append(result)
            status = "PASS ✓" if result["passed"] else "FAIL ✗"
            print(f"{status} ({result['reasoning_coverage']:.1%} coverage)")

    # Test attack scenarios
    if not basic_only:
        print(f"\n{'='*80}")
        print("TESTING ATTACK SCENARIOS (5 scenarios)")
        print("Goal: Improve A02 (Sybil), maintain A04/A05")
        print(f"{'='*80}")

        attack_scenarios = get_attack_scenarios()
        for i, scenario in enumerate(attack_scenarios, 1):
            print(f"\n[{i}/{len(attack_scenarios)}] {scenario.id}...", end=" ", flush=True)
            result = test_attack_scenario(llm, scenario, verbose=verbose)
            results["attack"].append(result)
            status = "PASS ✓" if result["passed"] else "FAIL ✗"
            print(f"{status} ({result['reasoning_coverage']:.1%} coverage)")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if results["basic"]:
        basic_passed = sum(1 for r in results["basic"] if r["passed"])
        basic_total = len(results["basic"])
        basic_coverage = sum(r["reasoning_coverage"] for r in results["basic"]) / basic_total
        print(f"\nBasic Scenarios:")
        print(f"  Pass Rate: {basic_passed}/{basic_total} ({basic_passed/basic_total:.1%})")
        print(f"  Avg Coverage: {basic_coverage:.1%}")
        print(f"  Target: 100% (v4_hybrid baseline)")
        if basic_passed == basic_total:
            print(f"  ✅ NO REGRESSION - v5 maintains basic scenario performance")
        else:
            print(f"  ⚠️  REGRESSION DETECTED - v5 broke basic scenarios")

    if results["attack"]:
        attack_passed = sum(1 for r in results["attack"] if r["passed"])
        attack_total = len(results["attack"])
        attack_coverage = sum(r["reasoning_coverage"] for r in results["attack"]) / attack_total
        attack_decision_correct = sum(1 for r in results["attack"] if r["decision_correct"])
        print(f"\nAttack Scenarios:")
        print(f"  Pass Rate: {attack_passed}/{attack_total} ({attack_passed/attack_total:.1%})")
        print(f"  Decision Accuracy: {attack_decision_correct}/{attack_total} ({attack_decision_correct/attack_total:.1%})")
        print(f"  Avg Coverage: {attack_coverage:.1%}")
        print(f"  v4 Baseline: 40% pass, 40% decision accuracy")

        # Check A02 specifically
        a02_result = next((r for r in results["attack"] if r["scenario_id"] == "A02"), None)
        if a02_result:
            print(f"\n  A02 (Sybil Detection) - KEY TARGET:")
            print(f"    Decision: {a02_result['extracted_decision']} (expected: deny)")
            print(f"    Correct: {'✓' if a02_result['decision_correct'] else '✗'}")
            print(f"    v4 Result: allow (WRONG)")
            if a02_result['decision_correct']:
                print(f"    ✅ IMPROVED - v5 fixed Sybil detection gap!")
            else:
                print(f"    ⚠️  Still failing - attack indicators insufficient")

    # Detailed results
    if results["basic"] and not verbose:
        print(f"\nBasic Scenario Details:")
        for r in results["basic"]:
            status = "PASS ✓" if r["passed"] else "FAIL ✗"
            print(f"  {r['scenario_id']}: {status} - {r['reasoning_coverage']:.1%}")

    if results["attack"] and not verbose:
        print(f"\nAttack Scenario Details:")
        for r in results["attack"]:
            status = "PASS ✓" if r["passed"] else "FAIL ✗"
            dec_status = "✓" if r["decision_correct"] else "✗"
            print(f"  {r['scenario_id']}: {status} - Decision: {r['extracted_decision']} ({dec_status})")

    # Save results
    output_file = Path("results/v5_comprehensive_test.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_PATH.name,
            "prompt_version": "v5_attack_aware",
            "basic_results": results["basic"],
            "attack_results": results["attack"]
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive v5 testing")
    parser.add_argument('--basic-only', action='store_true', help="Test only basic scenarios")
    parser.add_argument('--attack-only', action='store_true', help="Test only attack scenarios")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose output")

    args = parser.parse_args()

    run_comprehensive_test(
        basic_only=args.basic_only,
        attack_only=args.attack_only,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
