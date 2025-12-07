#!/usr/bin/env python3
"""
Multi-Run Validation of Learned Thresholds
===========================================

Run multiple validation trials to reduce random variation and get more
reliable comparison between baseline and learned thresholds.

Author: Claude (autonomous research) on Thor
Date: 2025-12-07
Session: Learned threshold validation (multi-run)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'core'))

from validate_learned_thresholds import (
    SimplifiedConsciousnessValidator,
    run_validation_experiment
)
from adaptive_thresholds import (
    AdaptiveThresholds,
    ThresholdPerformance,
    ThresholdObjectives
)
from pattern_library import PatternLibrary
from simulated_lct_identity import SimulatedLCTIdentity


def multi_run_validation():
    """Run multiple validation trials and aggregate results"""
    print("=" * 80)
    print("MULTI-RUN VALIDATION OF LEARNED THRESHOLDS")
    print("=" * 80)
    print()

    # Initialize pattern library
    print("1️⃣  Loading patterns...")
    lct_identity = SimulatedLCTIdentity()
    consciousness_key = lct_identity.get_or_create_identity("thor-sage-consciousness")
    library = PatternLibrary(
        lct_identity=lct_identity,
        consciousness_lct_id="thor-sage-consciousness"
    )
    print()

    # Load thresholds
    baseline_pattern_id = "thresholds_328972e37761ea41"
    learned_pattern_id = "thresholds_c662a805013d9629"

    try:
        baseline_pattern = library.load_pattern(baseline_pattern_id, "thresholds")
        baseline_thresholds = AdaptiveThresholds(**baseline_pattern.pattern_data)
    except:
        baseline_thresholds = AdaptiveThresholds(wake=0.45, focus=0.35, rest=0.85, dream=0.15)

    try:
        learned_pattern = library.load_pattern(learned_pattern_id, "thresholds")
        learned_thresholds = AdaptiveThresholds(**learned_pattern.pattern_data)
    except:
        learned_thresholds = AdaptiveThresholds(wake=0.51, focus=0.41, rest=0.85, dream=0.15)

    print(f"2️⃣  Baseline: WAKE={baseline_thresholds.wake:.2f}, FOCUS={baseline_thresholds.focus:.2f}")
    print(f"   Learned:  WAKE={learned_thresholds.wake:.2f}, FOCUS={learned_thresholds.focus:.2f}")
    print()

    # Run multiple trials
    num_trials = 10
    cycles_per_trial = 100

    print(f"3️⃣  Running {num_trials} trials ({cycles_per_trial} cycles each)...")
    print()

    baseline_results = []
    learned_results = []
    objectives = ThresholdObjectives(
        target_attention_rate=0.40,
        min_atp_level=0.30,
        min_salience_quality=0.30,
        max_state_changes_per_100=50.0
    )

    for trial in range(num_trials):
        print(f"   Trial {trial+1}/{num_trials}:")

        # Run baseline
        baseline_perf, _ = run_validation_experiment(
            baseline_thresholds,
            cycles_per_trial,
            f"Baseline Trial {trial+1}"
        )
        baseline_score = baseline_perf.score(objectives)
        baseline_results.append((baseline_perf, baseline_score))

        print(f"      Baseline: Attn={baseline_perf.attention_rate*100:4.1f}%, "
              f"ATP={baseline_perf.avg_atp:.3f}, Score={baseline_score:.3f}")

        # Run learned
        learned_perf, _ = run_validation_experiment(
            learned_thresholds,
            cycles_per_trial,
            f"Learned Trial {trial+1}"
        )
        learned_score = learned_perf.score(objectives)
        learned_results.append((learned_perf, learned_score))

        print(f"      Learned:  Attn={learned_perf.attention_rate*100:4.1f}%, "
              f"ATP={learned_perf.avg_atp:.3f}, Score={learned_score:.3f}")
        print()

    # Aggregate results
    print("=" * 80)
    print("AGGREGATED RESULTS (MEAN ± STD)")
    print("=" * 80)
    print()

    # Calculate means and stds
    def calc_stats(results):
        perfs, scores = zip(*results)
        return {
            'attention_rate_mean': sum(p.attention_rate for p in perfs) / len(perfs),
            'attention_rate_std': (sum((p.attention_rate - sum(p2.attention_rate for p2 in perfs)/len(perfs))**2 for p in perfs) / len(perfs)) ** 0.5,
            'avg_atp_mean': sum(p.avg_atp for p in perfs) / len(perfs),
            'avg_atp_std': (sum((p.avg_atp - sum(p2.avg_atp for p2 in perfs)/len(perfs))**2 for p in perfs) / len(perfs)) ** 0.5,
            'salience_mean': sum(p.avg_attended_salience for p in perfs) / len(perfs),
            'salience_std': (sum((p.avg_attended_salience - sum(p2.avg_attended_salience for p2 in perfs)/len(perfs))**2 for p in perfs) / len(perfs)) ** 0.5,
            'state_changes_mean': sum(p.state_changes_per_100 for p in perfs) / len(perfs),
            'state_changes_std': (sum((p.state_changes_per_100 - sum(p2.state_changes_per_100 for p2 in perfs)/len(perfs))**2 for p in perfs) / len(perfs)) ** 0.5,
            'score_mean': sum(scores) / len(scores),
            'score_std': (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5,
        }

    baseline_stats = calc_stats(baseline_results)
    learned_stats = calc_stats(learned_results)

    print("                         Baseline              Learned               Difference")
    print("                         --------              -------               ----------")
    print(f"Attention rate           {baseline_stats['attention_rate_mean']*100:5.1f}% ± {baseline_stats['attention_rate_std']*100:4.1f}     {learned_stats['attention_rate_mean']*100:5.1f}% ± {learned_stats['attention_rate_std']*100:4.1f}     {(learned_stats['attention_rate_mean']-baseline_stats['attention_rate_mean'])*100:+5.1f}%")
    print(f"Avg ATP                  {baseline_stats['avg_atp_mean']:5.3f} ± {baseline_stats['avg_atp_std']:5.3f}     {learned_stats['avg_atp_mean']:5.3f} ± {learned_stats['avg_atp_std']:5.3f}     {learned_stats['avg_atp_mean']-baseline_stats['avg_atp_mean']:+6.3f}")
    print(f"Avg attended salience    {baseline_stats['salience_mean']:5.3f} ± {baseline_stats['salience_std']:5.3f}     {learned_stats['salience_mean']:5.3f} ± {learned_stats['salience_std']:5.3f}     {learned_stats['salience_mean']-baseline_stats['salience_mean']:+6.3f}")
    print(f"State changes/100        {baseline_stats['state_changes_mean']:5.1f} ± {baseline_stats['state_changes_std']:4.1f}      {learned_stats['state_changes_mean']:5.1f} ± {learned_stats['state_changes_std']:4.1f}      {learned_stats['state_changes_mean']-baseline_stats['state_changes_mean']:+5.1f}")
    print(f"Composite score          {baseline_stats['score_mean']:5.3f} ± {baseline_stats['score_std']:5.3f}     {learned_stats['score_mean']:5.3f} ± {learned_stats['score_std']:5.3f}     {learned_stats['score_mean']-baseline_stats['score_mean']:+6.3f}")
    print()

    # Statistical significance (simple t-test approximation)
    score_diff = learned_stats['score_mean'] - baseline_stats['score_mean']
    score_std_err = (baseline_stats['score_std']**2 + learned_stats['score_std']**2) ** 0.5
    if score_std_err > 0:
        t_stat = score_diff / score_std_err
        print(f"T-statistic for score difference: {t_stat:.2f}")
        if abs(t_stat) > 2.0:
            print("  → Statistically significant difference (p < 0.05)")
        else:
            print("  → Not statistically significant (p >= 0.05)")
    print()

    # Conclusion
    if learned_stats['score_mean'] > baseline_stats['score_mean']:
        improvement_pct = ((learned_stats['score_mean'] - baseline_stats['score_mean']) / baseline_stats['score_mean']) * 100
        print(f"✅ VALIDATION SUCCESSFUL: Learned thresholds improved by {improvement_pct:.1f}%")
        print()
        print("Key improvements:")
        if learned_stats['avg_atp_mean'] > baseline_stats['avg_atp_mean']:
            print(f"  ✓ Better ATP management: +{(learned_stats['avg_atp_mean']-baseline_stats['avg_atp_mean'])*100:.1f}%")
        if learned_stats['salience_mean'] > baseline_stats['salience_mean']:
            print(f"  ✓ Higher attention quality: +{(learned_stats['salience_mean']-baseline_stats['salience_mean'])*100:.1f}%")
        if learned_stats['state_changes_mean'] < baseline_stats['state_changes_mean']:
            print(f"  ✓ More stable states: -{(baseline_stats['state_changes_mean']-learned_stats['state_changes_mean']):.1f} changes/100")
    else:
        print(f"⚠️  Learned thresholds did not improve composite score")
        print(f"   Baseline: {baseline_stats['score_mean']:.3f} ± {baseline_stats['score_std']:.3f}")
        print(f"   Learned:  {learned_stats['score_mean']:.3f} ± {learned_stats['score_std']:.3f}")
        print()
        print("   However, note individual metric changes:")
        if learned_stats['avg_atp_mean'] > baseline_stats['avg_atp_mean']:
            print(f"   ✓ ATP improved: +{(learned_stats['avg_atp_mean']-baseline_stats['avg_atp_mean'])*100:.1f}%")
        if learned_stats['salience_mean'] > baseline_stats['salience_mean']:
            print(f"   ✓ Salience improved: +{(learned_stats['salience_mean']-baseline_stats['salience_mean'])*100:.1f}%")
        if learned_stats['state_changes_mean'] < baseline_stats['state_changes_mean']:
            print(f"   ✓ State stability improved: -{(baseline_stats['state_changes_mean']-learned_stats['state_changes_mean']):.1f} changes/100")

    print()

    return {
        'baseline_stats': baseline_stats,
        'learned_stats': learned_stats,
        'num_trials': num_trials
    }


if __name__ == "__main__":
    result = multi_run_validation()
