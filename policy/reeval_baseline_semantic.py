#!/usr/bin/env python3
"""
Re-evaluate baseline test results using semantic similarity.

Takes existing test results and re-scores them with the improved
semantic similarity metric to see true reasoning coverage.
"""

import json
import sys
from test_suite_semantic import (
    evaluate_response_semantic,
    create_test_report,
    TEST_SCENARIOS
)

def reeval_baseline(baseline_file: str, similarity_threshold: float = 0.55):
    """
    Re-evaluate baseline test results with semantic similarity.

    Args:
        baseline_file: Path to baseline_test_llama.json
        similarity_threshold: Cosine similarity threshold (default 0.55)
    """
    # Load baseline results
    print(f"Loading baseline results from {baseline_file}...")
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)

    original_report = baseline['report']
    print(f"\nOriginal evaluation (keyword-based):")
    print(f"  Pass rate: {original_report['pass_rate']:.1%}")
    print(f"  Average scores:")
    for key, val in original_report['average_scores'].items():
        print(f"    {key}: {val:.3f}")

    # Build scenario lookup
    scenario_map = {s.id: s for s in TEST_SCENARIOS}

    # Re-evaluate each response with semantic similarity
    print(f"\nRe-evaluating with semantic similarity (threshold={similarity_threshold})...")
    new_results = []

    for orig_result in original_report['detailed_results']:
        scenario_id = orig_result['scenario_id']
        response = orig_result['response']
        scenario = scenario_map[scenario_id]

        # Evaluate with semantic similarity
        new_result = evaluate_response_semantic(
            response,
            scenario,
            similarity_threshold=similarity_threshold
        )

        # Add response for reference
        new_result['response'] = response
        new_results.append(new_result)

    # Create new report
    new_report = create_test_report(new_results)

    print(f"\nNew evaluation (semantic similarity):")
    print(f"  Pass rate: {new_report['pass_rate']:.1%}")
    print(f"  Average scores:")
    for key, val in new_report['average_scores'].items():
        print(f"    {key}: {val:.3f}")

    # Detailed comparison
    print(f"\n{'='*70}")
    print("Detailed Results by Scenario:")
    print(f"{'='*70}")

    for i, result in enumerate(new_results):
        scenario = scenario_map[result['scenario_id']]
        print(f"\n[{result['scenario_id']}] {scenario.description}")
        print(f"  Difficulty: {result['difficulty']}")
        print(f"  Decision: {scenario.expected_decision} - {'✓' if result['scores']['decision_correct'] else '✗'}")
        print(f"  Reasoning coverage:")
        print(f"    Keyword:  {result['scores']['reasoning_coverage_keyword']:.2f}")
        print(f"    Semantic: {result['scores']['reasoning_coverage_semantic']:.2f}")
        print(f"  Output structure: {result['scores']['output_structure']:.2f}")
        print(f"  Overall: {'✓ PASS' if result['passed'] else '✗ FAIL'}")

        # Show reasoning element matches
        if 'reasoning_details' in result:
            print(f"  Reasoning elements:")
            for detail in result['reasoning_details']:
                status = '✓' if detail['present'] else '✗'
                print(f"    {status} '{detail['expected']}' (sim: {detail['similarity']:.3f})")

    # Save updated results
    output_file = baseline_file.replace('.json', '_semantic.json')
    output_data = {
        "timestamp": baseline['timestamp'],
        "model_path": baseline['model_path'],
        "model_type": baseline['model_type'],
        "num_scenarios": baseline['num_scenarios'],
        "similarity_threshold": similarity_threshold,
        "evaluation_method": "semantic_similarity",
        "report": new_report
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Saved semantic evaluation to: {output_file}")
    print(f"{'='*70}")

    return new_report


if __name__ == "__main__":
    baseline_file = "results/baseline_test_llama.json"

    if len(sys.argv) > 1:
        baseline_file = sys.argv[1]

    # Try different thresholds
    thresholds = [0.50, 0.55, 0.60]

    print("="*70)
    print("SEMANTIC SIMILARITY RE-EVALUATION")
    print("="*70)
    print(f"\nBaseline file: {baseline_file}")
    print(f"Testing similarity thresholds: {thresholds}")
    print()

    for threshold in thresholds:
        print(f"\n{'#'*70}")
        print(f"# THRESHOLD: {threshold}")
        print(f"{'#'*70}")
        reeval_baseline(baseline_file, similarity_threshold=threshold)
