#!/usr/bin/env python3
"""
Compare Old Heuristic vs New Data-Driven ThresholdEvaluator
============================================================

Validate that refined models dramatically improve prediction accuracy.

Author: Claude (autonomous research) on Thor
Date: 2025-12-07
Session: ThresholdEvaluator update validation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'core'))

from adaptive_thresholds import AdaptiveThresholds, ThresholdObjectives


# Import both versions of ThresholdEvaluator
# We'll use the class directly from learn_adaptive_thresholds
from learn_adaptive_thresholds import ThresholdEvaluator


def compare_models():
    """Compare predictions from old heuristic vs new data-driven models"""
    print("=" * 80)
    print("THRESHOLD EVALUATOR MODEL COMPARISON")
    print("=" * 80)
    print()

    # Test configurations (from previous validation)
    test_configs = [
        {'wake': 0.45, 'focus': 0.35, 'name': 'Baseline (v1.0.0)', 'actual_attn': 0.189, 'actual_atp': 0.856},
        {'wake': 0.51, 'focus': 0.41, 'name': 'Learned (v2.0.0)', 'actual_attn': 0.183, 'actual_atp': 0.883},
        {'wake': 0.40, 'focus': 0.30, 'name': 'Lower thresholds', 'actual_attn': None, 'actual_atp': None},
        {'wake': 0.55, 'focus': 0.45, 'name': 'Higher thresholds', 'actual_attn': None, 'actual_atp': None}
    ]

    # Create evaluators
    old_evaluator = ThresholdEvaluator(use_refined_models=False)
    new_evaluator = ThresholdEvaluator(use_refined_models=True)

    print("Testing threshold configurations:")
    print()

    for config in test_configs:
        thresholds = AdaptiveThresholds(
            wake=config['wake'],
            focus=config['focus'],
            rest=0.85,
            dream=0.15
        )

        # Get predictions
        old_perf = old_evaluator.evaluate(thresholds)
        new_perf = new_evaluator.evaluate(thresholds)

        print(f"{config['name']}: WAKE={config['wake']:.2f}, FOCUS={config['focus']:.2f}")
        print(f"  Old model: Attn={old_perf.attention_rate*100:5.1f}%, ATP={old_perf.avg_atp:.3f}, Salience={old_perf.avg_attended_salience:.3f}")
        print(f"  New model: Attn={new_perf.attention_rate*100:5.1f}%, ATP={new_perf.avg_atp:.3f}, Salience={new_perf.avg_attended_salience:.3f}")

        if config['actual_attn'] is not None:
            print(f"  Actual:    Attn={config['actual_attn']*100:5.1f}%, ATP={config['actual_atp']:.3f}")

            old_attn_error = abs(old_perf.attention_rate - config['actual_attn']) / config['actual_attn']
            new_attn_error = abs(new_perf.attention_rate - config['actual_attn']) / config['actual_attn']
            old_atp_error = abs(old_perf.avg_atp - config['actual_atp']) / config['actual_atp']
            new_atp_error = abs(new_perf.avg_atp - config['actual_atp']) / config['actual_atp']

            print(f"  Old error: Attn={old_attn_error*100:5.1f}%, ATP={old_atp_error*100:5.1f}%")
            print(f"  New error: Attn={new_attn_error*100:5.1f}%, ATP={new_atp_error*100:5.1f}%")
            print(f"  Improvement: Attn={((old_attn_error-new_attn_error)/old_attn_error)*100:+5.1f}%, ATP={((old_atp_error-new_atp_error)/old_atp_error)*100:+5.1f}%")

        print()

    # Summary
    print("=" * 80)
    print("MODEL ACCURACY SUMMARY")
    print("=" * 80)
    print()

    # Baseline config
    baseline = AdaptiveThresholds(wake=0.45, focus=0.35, rest=0.85, dream=0.15)
    old_baseline = old_evaluator.evaluate(baseline)
    new_baseline = new_evaluator.evaluate(baseline)

    actual_baseline_attn = 0.189
    actual_baseline_atp = 0.856

    old_attn_err = abs(old_baseline.attention_rate - actual_baseline_attn) / actual_baseline_attn
    new_attn_err = abs(new_baseline.attention_rate - actual_baseline_attn) / actual_baseline_attn
    old_atp_err = abs(old_baseline.avg_atp - actual_baseline_atp) / actual_baseline_atp
    new_atp_err = abs(new_baseline.avg_atp - actual_baseline_atp) / actual_baseline_atp

    print("Baseline Threshold Predictions (WAKE=0.45, FOCUS=0.35):")
    print()
    print(f"Attention Rate:")
    print(f"  Old model error: {old_attn_err*100:.1f}%")
    print(f"  New model error: {new_attn_err*100:.1f}%")
    print(f"  Improvement: {((old_attn_err - new_attn_err)/old_attn_err)*100:.1f}%")
    print()
    print(f"ATP:")
    print(f"  Old model error: {old_atp_err*100:.1f}%")
    print(f"  New model error: {new_atp_err*100:.1f}%")
    print(f"  Improvement: {((old_atp_err - new_atp_err)/old_atp_err)*100:.1f}%")
    print()

    # Learned config
    learned = AdaptiveThresholds(wake=0.51, focus=0.41, rest=0.85, dream=0.15)
    old_learned = old_evaluator.evaluate(learned)
    new_learned = new_evaluator.evaluate(learned)

    actual_learned_attn = 0.183
    actual_learned_atp = 0.883

    old_attn_err_l = abs(old_learned.attention_rate - actual_learned_attn) / actual_learned_attn
    new_attn_err_l = abs(new_learned.attention_rate - actual_learned_attn) / actual_learned_attn
    old_atp_err_l = abs(old_learned.avg_atp - actual_learned_atp) / actual_learned_atp
    new_atp_err_l = abs(new_learned.avg_atp - actual_learned_atp) / actual_learned_atp

    print("Learned Threshold Predictions (WAKE=0.51, FOCUS=0.41):")
    print()
    print(f"Attention Rate:")
    print(f"  Old model error: {old_attn_err_l*100:.1f}%")
    print(f"  New model error: {new_attn_err_l*100:.1f}%")
    print(f"  Improvement: {((old_attn_err_l - new_attn_err_l)/old_attn_err_l)*100:.1f}%")
    print()
    print(f"ATP:")
    print(f"  Old model error: {old_atp_err_l*100:.1f}%")
    print(f"  New model error: {new_atp_err_l*100:.1f}%")
    print(f"  Improvement: {((old_atp_err_l - new_atp_err_l)/old_atp_err_l)*100:.1f}%")
    print()

    # Overall assessment
    avg_old_attn_err = (old_attn_err + old_attn_err_l) / 2
    avg_new_attn_err = (new_attn_err + new_attn_err_l) / 2
    avg_old_atp_err = (old_atp_err + old_atp_err_l) / 2
    avg_new_atp_err = (new_atp_err + new_atp_err_l) / 2

    print("=" * 80)
    print("OVERALL IMPROVEMENT")
    print("=" * 80)
    print()
    print(f"Average Attention Error:")
    print(f"  Old model: {avg_old_attn_err*100:.1f}%")
    print(f"  New model: {avg_new_attn_err*100:.1f}%")
    print(f"  Reduction: {((avg_old_attn_err - avg_new_attn_err)/avg_old_attn_err)*100:.1f}%")
    print()
    print(f"Average ATP Error:")
    print(f"  Old model: {avg_old_atp_err*100:.1f}%")
    print(f"  New model: {avg_new_atp_err*100:.1f}%")
    print(f"  Reduction: {((avg_old_atp_err - avg_new_atp_err)/avg_old_atp_err)*100:.1f}%")
    print()

    if avg_new_attn_err < avg_old_attn_err and avg_new_atp_err < avg_old_atp_err:
        print("✅ SUCCESS: Refined models dramatically improve prediction accuracy!")
        print()
        print("Next steps:")
        print("1. Re-run adaptive learning with refined models")
        print("2. Validate learned thresholds hit intended targets")
        print("3. Compare learning convergence (old vs new models)")
    else:
        print("⚠️  UNEXPECTED: Refined models did not improve all metrics")
        print("   Review model coefficients and validation data")

    print()


if __name__ == "__main__":
    compare_models()
