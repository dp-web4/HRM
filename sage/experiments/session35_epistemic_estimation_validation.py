#!/usr/bin/env python3
"""
Session 35: Epistemic Estimation Validation

Tests the improved epistemic estimation system against labeled data.
This validates the fix for Session 34's finding that heuristic estimation
achieved 0% accuracy.

Author: Thor (Autonomous Session 35)
Date: 2025-12-12
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.epistemic_estimator import (
    ImprovedEpistemicEstimator,
    estimate_epistemic_state
)
from core.epistemic_states import EpistemicState, EpistemicMetrics
from core.sage_real_measurements import SAGERealMeasurements


# Labeled test dataset with clear epistemic signatures
LABELED_TEST_DATA = [
    # CONFIDENT: High specificity, numbers, technical terms
    ("ATP level is precisely 75.5 with salience threshold at 0.7, confirming optimal attention allocation with 89% coverage efficiency.",
     EpistemicState.CONFIDENT),

    ("The epistemic state tracker maintains exactly 100 cycles with confidence levels consistently ranging from 0.65 to 0.85.",
     EpistemicState.CONFIDENT),

    ("Convergence definitely occurs at cycle 743 with fitness 0.847, clearly satisfying the threshold of 0.83.",
     EpistemicState.CONFIDENT),

    # UNCERTAIN: Hedging, maybe, unclear
    ("The mechanism might be related to attention allocation, but it's unclear exactly how the threshold is determined.",
     EpistemicState.UNCERTAIN),

    ("This appears to work, though it's hard to say definitively whether the pattern holds in all cases.",
     EpistemicState.UNCERTAIN),

    ("The results seem positive, but perhaps additional testing would clarify the underlying dynamics.",
     EpistemicState.UNCERTAIN),

    # FRUSTRATED: Gap between expected and actual, inconsistencies
    ("The convergence should work according to the model, but the observed behavior doesn't match expectations.",
     EpistemicState.FRUSTRATED),

    ("Tried adjusting the threshold multiple times without success - the adaptation still shows inconsistent patterns.",
     EpistemicState.FRUSTRATED),

    ("The quality metrics remain unclear despite repeated analysis, and the gap between predicted and actual performance continues.",
     EpistemicState.FRUSTRATED),

    # CONFUSED: Multiple interpretations, conflicting signals
    ("On one hand the efficiency suggests optimization, but on the other hand the adaptation frequency indicates instability.",
     EpistemicState.CONFUSED),

    ("There are multiple possible interpretations - it could be learning or it could be random fluctuation.",
     EpistemicState.CONFUSED),

    ("The competing explanations are difficult to reconcile: high quality but low confidence.",
     EpistemicState.CONFUSED),

    # LEARNING: Integration, emerging patterns
    ("Integrating the new quality metrics with temporal adaptation reveals an emerging pattern in weight stability.",
     EpistemicState.LEARNING),

    ("Beginning to see how epistemic awareness connects with federation routing - developing insight into distributed meta-cognition.",
     EpistemicState.LEARNING),

    ("Refining understanding of the relationship between confidence and quality scores based on recent observations.",
     EpistemicState.LEARNING),

    # STABLE: Established, conventional, as expected
    ("The standard multi-objective optimization approach works as expected with established parameter ranges.",
     EpistemicState.STABLE),

    ("This is a familiar pattern consistent with conventional temporal adaptation behavior.",
     EpistemicState.STABLE),

    ("The well-understood quality scoring system predictably identifies technical responses.",
     EpistemicState.STABLE),
]


def test_improved_estimator():
    """Test improved epistemic estimator against labeled data"""
    print("=" * 80)
    print("SESSION 35: IMPROVED EPISTEMIC ESTIMATION VALIDATION")
    print("=" * 80)
    print()
    print("Testing improved estimator against 18 labeled responses")
    print("(Session 34 baseline: 0% accuracy with heuristic)")
    print()

    estimator = ImprovedEpistemicEstimator()

    predictions = []
    ground_truth = []
    correct = 0

    # Test each labeled example
    for i, (response, true_state) in enumerate(LABELED_TEST_DATA):
        metrics = estimator.estimate_from_response(response)
        predicted_state = metrics.primary_state()

        predictions.append((predicted_state, metrics))
        ground_truth.append(true_state)

        is_correct = predicted_state == true_state
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"{i+1:2d}. {status} Predicted: {predicted_state.value:12s} | "
              f"Actual: {true_state.value:12s} | "
              f"Conf: {metrics.confidence:.2f}")

    print()
    accuracy = correct / len(LABELED_TEST_DATA)
    print(f"Overall Accuracy: {correct}/{len(LABELED_TEST_DATA)} = {accuracy:.1%}")
    print()

    # Detailed breakdown by state
    print("Accuracy by State:")
    for state in EpistemicState:
        state_indices = [i for i, (_, true_state) in enumerate(LABELED_TEST_DATA)
                        if true_state == state]
        if not state_indices:
            continue

        state_correct = sum(1 for i in state_indices
                          if predictions[i][0] == ground_truth[i])
        state_accuracy = state_correct / len(state_indices)
        print(f"  {state.value:12s}: {state_correct}/{len(state_indices)} = {state_accuracy:.1%}")

    print()

    # Compare to Session 34 baseline
    print("Comparison to Session 34:")
    print(f"  Session 34 heuristic: 0/10 = 0.0%")
    print(f"  Session 35 improved:  {correct}/{len(LABELED_TEST_DATA)} = {accuracy:.1%}")
    print(f"  Improvement: +{accuracy:.1%}")
    print()

    # Q2 prediction assessment
    print("Q2 Prediction Assessment:")
    print(f"  Target: ≥66% accuracy (4/6 states correct)")
    print(f"  Achieved: {accuracy:.1%}")
    if accuracy >= 0.66:
        print("  ✅ Q2 PREDICTION VALIDATED")
    else:
        print(f"  ⚠️  Below target (gap: {0.66 - accuracy:.1%})")

    print()
    return accuracy


def test_with_real_measurement_integration():
    """Test integration with Session 34 real measurement infrastructure"""
    print("=" * 80)
    print("INTEGRATION TEST: Q2 Measurement with Improved Estimator")
    print("=" * 80)
    print()

    # Use same test data
    predictions = []
    ground_truth = []

    estimator = ImprovedEpistemicEstimator()

    for response, true_state in LABELED_TEST_DATA:
        metrics = estimator.estimate_from_response(response)
        predicted_state = metrics.primary_state()
        predictions.append((predicted_state, metrics))
        ground_truth.append(true_state)

    # Use Session 34 real measurement infrastructure
    measurer = SAGERealMeasurements()
    result = measurer.measure_epistemic_accuracy(predictions, ground_truth)

    print(f"Sample size: {result.sample_size} predictions")
    print(f"Accuracy: {result.observed_value:.3f} ± {result.observed_error:.3f}")
    print(f"Prediction: ≥0.66 (4/6 states correct)")
    print(f"Notes: {result.notes}")
    print()

    if result.observed_value >= 0.66:
        print("✅ Q2 Prediction VALIDATED with improved estimator")
    else:
        print(f"⚠️  Still below target: {result.observed_value:.1%} vs 66%")

    print()
    return result


def analyze_failure_modes():
    """Analyze which states are hardest to predict"""
    print("=" * 80)
    print("FAILURE MODE ANALYSIS")
    print("=" * 80)
    print()

    estimator = ImprovedEpistemicEstimator()

    # Confusion matrix
    confusion_matrix = {state: {state2: 0 for state2 in EpistemicState}
                       for state in EpistemicState}

    for response, true_state in LABELED_TEST_DATA:
        metrics = estimator.estimate_from_response(response)
        predicted_state = metrics.primary_state()
        confusion_matrix[true_state][predicted_state] += 1

    # Print confusion matrix
    print("Confusion Matrix (Actual → Predicted):")
    print()

    # Header
    print("              ", end="")
    for state in EpistemicState:
        print(f"{state.value[:4]:5s}", end="")
    print()

    # Rows
    for true_state in EpistemicState:
        print(f"{true_state.value:12s}  ", end="")
        for pred_state in EpistemicState:
            count = confusion_matrix[true_state][pred_state]
            if count > 0:
                symbol = "█" * count
                print(f"{symbol:5s}", end="")
            else:
                print("     ", end="")
        print()

    print()

    # Identify most common errors
    print("Common Misclassifications:")
    errors = []
    for true_state in EpistemicState:
        for pred_state in EpistemicState:
            if true_state != pred_state and confusion_matrix[true_state][pred_state] > 0:
                errors.append((
                    confusion_matrix[true_state][pred_state],
                    true_state.value,
                    pred_state.value
                ))

    errors.sort(reverse=True)
    for count, true_s, pred_s in errors[:5]:
        print(f"  {count}× {true_s} → {pred_s}")

    print()


def main():
    """Run Session 35 validation suite"""
    print()
    print("Session 35: Epistemic Estimation Refinement")
    print()
    print("Addressing Session 34 finding: Heuristic estimator achieved 0% accuracy")
    print("Solution: Improved linguistic analysis with pattern matching")
    print()

    # Test 1: Accuracy on labeled data
    accuracy = test_improved_estimator()

    # Test 2: Integration with real measurement
    result = test_with_real_measurement_integration()

    # Test 3: Failure mode analysis
    analyze_failure_modes()

    # Summary
    print("=" * 80)
    print("SESSION 35 VALIDATION SUMMARY")
    print("=" * 80)
    print()
    print("Improvements over Session 34:")
    print(f"  Session 34 heuristic: 0.0% accuracy")
    print(f"  Session 35 improved:  {accuracy:.1%} accuracy")
    print(f"  Improvement: +{accuracy:.1%}")
    print()

    print("Q2 Prediction Status:")
    print(f"  Target: ≥66% accuracy")
    print(f"  Achieved: {result.observed_value:.1%}")
    if result.observed_value >= 0.66:
        print("  ✅ Q2 PREDICTION NOW VALIDATED")
    else:
        print(f"  ⚠️  Gap: {0.66 - result.observed_value:.1%}")
    print()

    print("Next Steps:")
    print("  1. Collect actual EpistemicStateTracker data from production")
    print("  2. Refine linguistic patterns based on real SAGE responses")
    print("  3. Build classifier from labeled conversation data")
    print("  4. Compare estimator vs actual tracker accuracy")
    print()

    if result.observed_value >= 0.66:
        print("✅ SESSION 35 SUCCESS: Q2 estimation improved to validation threshold")
        return 0
    else:
        print("⚠️  SESSION 35 PARTIAL SUCCESS: Significant improvement but below target")
        return 1


if __name__ == '__main__':
    exit(main())
