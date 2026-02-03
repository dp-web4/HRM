#!/usr/bin/env python3
"""
Analyze reasoning coverage gaps for M02 and EC01 scenarios.

These scenarios have correct decisions but low reasoning coverage (33.33%).
Let's analyze the semantic similarity scores to understand if it's an evaluation
issue or a model expression issue.
"""

from test_suite_semantic import TEST_SCENARIOS, evaluate_response_semantic
from policy_logging import PolicyDecisionLog
import json


def analyze_scenario_reasoning(scenario_id: str):
    """Analyze reasoning coverage for a specific scenario."""

    # Get scenario from test suite
    scenario = next((s for s in TEST_SCENARIOS if s.id == scenario_id), None)
    if not scenario:
        print(f"Scenario {scenario_id} not found!")
        return

    # Get logged decision from database
    log = PolicyDecisionLog('results/policy_decisions.db')
    decisions = log.get_all_decisions(limit=20)
    decision = next((d for d in decisions if d.get('scenario_id') == scenario_id), None)

    if not decision:
        print(f"No logged decision found for {scenario_id}")
        return

    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario.id} - {scenario.description}")
    print(f"{'='*70}\n")

    print(f"Decision: {decision.get('decision')}")
    print(f"Expected: {scenario.expected_decision}")
    print(f"Decision Correct: {decision.get('decision_correct')}")
    print(f"Reasoning Coverage: {decision.get('reasoning_coverage'):.2%}\n")

    print(f"Expected Reasoning Elements:")
    for elem in scenario.expected_reasoning_elements:
        print(f"  - {elem}")

    print(f"\nModel Reasoning:")
    print(decision.get('reasoning', '(no reasoning)'))

    # Now evaluate with different thresholds
    print(f"\n{'='*70}")
    print(f"SEMANTIC SIMILARITY ANALYSIS")
    print(f"{'='*70}\n")

    response_text = decision.get('full_response', '')

    # Test multiple thresholds
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.49, 0.5, 0.55, 0.6]

    print(f"Testing different similarity thresholds:\n")
    print(f"{'Threshold':>10} | {'Coverage':>10} | {'Pass':>6}")
    print(f"{'-'*10}-+-{'-'*10}-+-{'-'*6}")

    for threshold in thresholds:
        result = evaluate_response_semantic(response_text, scenario, similarity_threshold=threshold)
        pass_mark = "✓" if result['passed'] else "✗"
        coverage = result['scores']['reasoning_coverage_semantic']
        print(f"{threshold:>10.2f} | {coverage:>9.1%} | {pass_mark:>6}")

    # Detailed element analysis at threshold 0.49
    print(f"\nDetailed Element Analysis (threshold=0.49):")
    result = evaluate_response_semantic(response_text, scenario, similarity_threshold=0.49)

    print(f"\n{'Expected Element':<30} | {'Best Match':<50} | {'Score':>6} | {'Found':>6}")
    print(f"{'-'*30}-+-{'-'*50}-+-{'-'*6}-+-{'-'*6}")

    # Use reasoning_details from result
    for detail in result['reasoning_details']:
        expected = detail['expected']
        best_match = detail['best_match']
        score = detail['similarity']
        present = "✓" if detail['present'] else "✗"

        # Truncate best match if too long
        if len(best_match) > 47:
            best_match = best_match[:44] + "..."

        print(f"{expected:<30} | {best_match:<50} | {score:>6.3f} | {present:>6}")


def main():
    """Analyze both problematic scenarios."""

    print("\n" + "="*70)
    print("REASONING COVERAGE GAP ANALYSIS")
    print("="*70)

    print("\nAnalyzing scenarios with correct decisions but low reasoning coverage...")

    # Analyze M02
    analyze_scenario_reasoning("M02")

    # Analyze EC01
    analyze_scenario_reasoning("EC01")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    print("\nKey Findings:")
    print("1. Check if model reasoning contains semantic matches at different thresholds")
    print("2. Identify which expected elements are truly missing vs poorly matched")
    print("3. Determine if evaluation needs adjustment or model needs better prompts")


if __name__ == "__main__":
    main()
