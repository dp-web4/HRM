#!/usr/bin/env python3
"""
Re-evaluate existing test results with adjusted threshold.

Quick validation of threshold change without re-running expensive model inference.
"""

import json
from test_suite_semantic import TEST_SCENARIOS, evaluate_response_semantic, create_test_report

def reeval_with_threshold(results_file: str, new_threshold: float = 0.49):
    """Re-evaluate existing results with new threshold."""

    print(f"Loading results from {results_file}...")
    with open(results_file) as f:
        data = json.load(f)

    # Build scenario lookup
    scenario_map = {s.id: s for s in TEST_SCENARIOS}

    # Re-evaluate each response
    print(f"\nRe-evaluating with threshold={new_threshold}...")
    new_results = []

    for orig_result in data['detailed_results']:
        scenario_id = orig_result['scenario_id']
        response = orig_result['response']
        scenario = scenario_map[scenario_id]

        # Evaluate with new threshold
        new_result = evaluate_response_semantic(response, scenario, similarity_threshold=new_threshold)
        new_result['response'] = response
        new_result['variant'] = 'v2_fewshot'
        new_results.append(new_result)

    # Create new report
    new_report = create_test_report(new_results)

    # Compare
    print(f"\n{'='*70}")
    print("THRESHOLD COMPARISON")
    print(f"{'='*70}\n")

    print(f"Original (threshold=0.5):")
    print(f"  Pass rate: {data['pass_rate']:.1%}")
    print(f"  Decision accuracy: {data['average_scores']['decision_correct']:.1%}")
    print(f"  Reasoning coverage: {data['average_scores']['reasoning_coverage_semantic']:.3f}")

    print(f"\nNew (threshold={new_threshold}):")
    print(f"  Pass rate: {new_report['pass_rate']:.1%}")
    print(f"  Decision accuracy: {new_report['average_scores']['decision_correct']:.1%}")
    print(f"  Reasoning coverage: {new_report['average_scores']['reasoning_coverage_semantic']:.3f}")

    print(f"\nChange:")
    pass_rate_change = new_report['pass_rate'] - data['pass_rate']
    print(f"  Pass rate: {pass_rate_change:+.1%}")

    # Show which scenarios changed
    print(f"\n{'='*70}")
    print("SCENARIO CHANGES")
    print(f"{'='*70}\n")

    for i, (orig, new) in enumerate(zip(data['detailed_results'], new_results)):
        if orig['passed'] != new['passed']:
            status = "✗ → ✓" if new['passed'] else "✓ → ✗"
            print(f"{orig['scenario_id']}: {status}")
            print(f"  Reasoning: {orig['scores']['reasoning_coverage_semantic']:.3f} → {new['scores']['reasoning_coverage_semantic']:.3f}")

    # Save new results
    output_file = results_file.replace('.json', '_threshold049.json')
    output_data = {
        **new_report,
        "timestamp": data.get('timestamp'),
        "similarity_threshold": new_threshold,
        "total_time_seconds": data.get('total_time_seconds'),
        "avg_time_per_scenario": data.get('avg_time_per_scenario')
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to: {output_file}")

    return new_report


if __name__ == "__main__":
    results_file = "results/v2_fewshot_full.json"
    reeval_with_threshold(results_file, new_threshold=0.49)
