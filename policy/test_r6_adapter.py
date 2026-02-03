#!/usr/bin/env python3
"""
Test R6Request Adapter with Full HRM Test Suite

Converts all 8 HRM test scenarios to R6Request format and validates them.
This demonstrates integration readiness with hardbound PolicyModel.
"""

import json
from r6_adapter import hrm_to_r6, validate_r6_request
from test_suite_semantic import TEST_SCENARIOS


def test_all_scenarios():
    """Convert all test scenarios to R6Request format."""

    print("="*70)
    print("R6REQUEST ADAPTER - FULL TEST SUITE CONVERSION")
    print("="*70)
    print(f"\nConverting {len(TEST_SCENARIOS)} HRM test scenarios to R6Request format\n")

    results = {
        'total_scenarios': len(TEST_SCENARIOS),
        'successful_conversions': 0,
        'validation_errors': 0,
        'conversions': []
    }

    for scenario in TEST_SCENARIOS:
        print(f"\n{'='*70}")
        print(f"Scenario {scenario.id}: {scenario.description}")
        print(f"Difficulty: {scenario.difficulty}")
        print(f"{'='*70}\n")

        # Convert to R6Request
        r6_request = hrm_to_r6(
            scenario.situation,
            scenario_id=scenario.id
        )

        # Validate
        errors = validate_r6_request(r6_request)

        conversion_result = {
            'scenario_id': scenario.id,
            'description': scenario.description,
            'difficulty': scenario.difficulty,
            'r6_request': r6_request,
            'validation_errors': errors,
            'valid': len(errors) == 0
        }

        if len(errors) == 0:
            print(f"✅ Conversion successful")
            results['successful_conversions'] += 1
        else:
            print(f"❌ Validation errors: {errors}")
            results['validation_errors'] += 1

        # Display key fields
        print(f"\nR6Request summary:")
        print(f"  Request ID: {r6_request['requestId']}")
        print(f"  Actor: {r6_request['actorId']}")
        print(f"  Action: {r6_request['action']['type']} on {r6_request['action']['target']}")
        print(f"  Risk Assessment: {r6_request['context']['callerRiskAssessment']}")
        print(f"  Trust: C={r6_request['trustState']['competence']:.2f}, "
              f"R={r6_request['trustState']['reliability']:.2f}, "
              f"I={r6_request['trustState']['integrity']:.2f}")

        if 'coherenceState' in r6_request:
            cs = r6_request['coherenceState']
            print(f"  Coherence: d9={cs['d9Score']:.2f}, coupling={cs.get('couplingState', 'N/A')}")

        results['conversions'].append(conversion_result)

    # Summary
    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print(f"{'='*70}\n")

    print(f"Total scenarios: {results['total_scenarios']}")
    print(f"Successful conversions: {results['successful_conversions']}")
    print(f"Validation errors: {results['validation_errors']}")

    if results['validation_errors'] == 0:
        print(f"\n✅ All scenarios converted successfully to R6Request format!")
        print(f"\n🎯 Ready for hardbound PolicyModel integration testing")
    else:
        print(f"\n⚠️  Some scenarios had validation errors - review above")

    # Save results
    output_file = "results/r6_adapter_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_file}")

    return results


def analyze_risk_distribution():
    """Analyze risk assessment distribution across scenarios."""

    print(f"\n{'='*70}")
    print("RISK ASSESSMENT DISTRIBUTION")
    print(f"{'='*70}\n")

    risk_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}

    for scenario in TEST_SCENARIOS:
        r6 = hrm_to_r6(scenario.situation, scenario_id=scenario.id)
        risk = r6['context']['callerRiskAssessment']
        risk_counts[risk] += 1

        print(f"{scenario.id}: {risk:8s} - {scenario.description}")

    print(f"\nRisk distribution:")
    for risk, count in risk_counts.items():
        pct = (count / len(TEST_SCENARIOS)) * 100
        print(f"  {risk:8s}: {count} ({pct:.1f}%)")


def show_example_r6_requests():
    """Show example R6Requests for documentation."""

    print(f"\n{'='*70}")
    print("EXAMPLE R6REQUESTS")
    print(f"{'='*70}\n")

    # Show one example from each difficulty level
    examples = {
        'easy': 'E01',
        'medium': 'M01',
        'hard': 'H01',
        'edge_case': 'EC01'
    }

    for difficulty, scenario_id in examples.items():
        scenario = next(s for s in TEST_SCENARIOS if s.id == scenario_id)

        print(f"\n{'-'*70}")
        print(f"{difficulty.upper()}: {scenario.id} - {scenario.description}")
        print(f"{'-'*70}\n")

        r6 = hrm_to_r6(scenario.situation, scenario_id=scenario.id)
        print(json.dumps(r6, indent=2))


if __name__ == "__main__":
    # Run full test suite conversion
    results = test_all_scenarios()

    # Analyze risk distribution
    analyze_risk_distribution()

    # Show example R6Requests
    show_example_r6_requests()

    # Exit with appropriate code
    exit(0 if results['validation_errors'] == 0 else 1)
