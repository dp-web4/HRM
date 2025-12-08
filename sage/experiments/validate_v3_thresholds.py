#!/usr/bin/env python3
"""
Validate v3.0.0 Learned Thresholds (Refined Model Learning)
=============================================================

Compare:
- Baseline v1.0.0: WAKE=0.45, FOCUS=0.35 (heuristic)
- Learned v2.0.0: WAKE=0.51, FOCUS=0.41 (learned with heuristic model)
- Learned v3.0.0: WAKE=0.39, FOCUS=0.29 (learned with refined model)

Expected finding: v3.0.0 should have higher attention rate (closer to 40% target)
because learning correctly reduced thresholds to increase attention.

Author: Claude (autonomous research) on Thor
Date: 2025-12-07
Session: Validation of refined model learning
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'core'))

from validate_learned_thresholds import SimplifiedConsciousnessValidator
from adaptive_thresholds import AdaptiveThresholds


def run_multi_trial_validation(thresholds: AdaptiveThresholds, trials: int = 10, cycles: int = 100):
    """Run multiple trials and aggregate results"""
    all_perfs = []

    for trial in range(trials):
        validator = SimplifiedConsciousnessValidator(thresholds)
        for _ in range(cycles):
            validator.cycle()
        perf = validator.get_performance()
        all_perfs.append(perf)

    # Calculate statistics
    attention_mean = sum(p.attention_rate for p in all_perfs) / len(all_perfs)
    attention_std = (sum((p.attention_rate - attention_mean)**2 for p in all_perfs) / len(all_perfs)) ** 0.5

    atp_mean = sum(p.avg_atp for p in all_perfs) / len(all_perfs)
    atp_std = (sum((p.avg_atp - atp_mean)**2 for p in all_perfs) / len(all_perfs)) ** 0.5

    salience_mean = sum(p.avg_attended_salience for p in all_perfs) / len(all_perfs)
    salience_std = (sum((p.avg_attended_salience - salience_mean)**2 for p in all_perfs) / len(all_perfs)) ** 0.5

    state_changes_mean = sum(p.state_changes_per_100 for p in all_perfs) / len(all_perfs)
    state_changes_std = (sum((p.state_changes_per_100 - state_changes_mean)**2 for p in all_perfs) / len(all_perfs)) ** 0.5

    return {
        'attention_rate': (attention_mean, attention_std),
        'avg_atp': (atp_mean, atp_std),
        'avg_attended_salience': (salience_mean, salience_std),
        'state_changes_per_100': (state_changes_mean, state_changes_std)
    }


def main():
    print("=" * 80)
    print("VALIDATION OF v3.0.0 THRESHOLDS (REFINED MODEL LEARNING)")
    print("=" * 80)
    print()

    # Three threshold configurations
    configs = [
        {
            'name': 'Baseline v1.0.0',
            'thresholds': AdaptiveThresholds(wake=0.45, focus=0.35, rest=0.85, dream=0.15),
            'description': 'Original heuristic baseline'
        },
        {
            'name': 'Learned v2.0.0',
            'thresholds': AdaptiveThresholds(wake=0.51, focus=0.41, rest=0.85, dream=0.15),
            'description': 'Learned with old heuristic model (increased thresholds)'
        },
        {
            'name': 'Learned v3.0.0',
            'thresholds': AdaptiveThresholds(wake=0.39, focus=0.29, rest=0.85, dream=0.15),
            'description': 'Learned with refined model (decreased thresholds)'
        }
    ]

    print("Running 10 trials × 100 cycles per configuration...\n")

    results = {}
    for config in configs:
        print(f"Testing {config['name']}: {config['description']}")
        print(f"  WAKE={config['thresholds'].wake:.2f}, FOCUS={config['thresholds'].focus:.2f}")

        stats = run_multi_trial_validation(config['thresholds'], trials=10, cycles=100)
        results[config['name']] = stats

        print(f"  Attention: {stats['attention_rate'][0]*100:5.1f}% ± {stats['attention_rate'][1]*100:4.1f}%")
        print(f"  ATP: {stats['avg_atp'][0]:.3f} ± {stats['avg_atp'][1]:.3f}")
        print(f"  Salience: {stats['avg_attended_salience'][0]:.3f} ± {stats['avg_attended_salience'][1]:.3f}")
        print()

    # Summary comparison
    print("=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print()

    baseline = results['Baseline v1.0.0']
    v2 = results['Learned v2.0.0']
    v3 = results['Learned v3.0.0']

    print("Attention Rate:")
    print(f"  Baseline v1.0.0: {baseline['attention_rate'][0]*100:5.1f}%")
    print(f"  Learned v2.0.0:  {v2['attention_rate'][0]*100:5.1f}% ({(v2['attention_rate'][0] - baseline['attention_rate'][0])*100:+5.1f}%)")
    print(f"  Learned v3.0.0:  {v3['attention_rate'][0]*100:5.1f}% ({(v3['attention_rate'][0] - baseline['attention_rate'][0])*100:+5.1f}%)")
    print()

    print("ATP:")
    print(f"  Baseline v1.0.0: {baseline['avg_atp'][0]:.3f}")
    print(f"  Learned v2.0.0:  {v2['avg_atp'][0]:.3f} ({(v2['avg_atp'][0] - baseline['avg_atp'][0])*100:+5.1f}%)")
    print(f"  Learned v3.0.0:  {v3['avg_atp'][0]:.3f} ({(v3['avg_atp'][0] - baseline['avg_atp'][0])*100:+5.1f}%)")
    print()

    print("Attended Salience:")
    print(f"  Baseline v1.0.0: {baseline['avg_attended_salience'][0]:.3f}")
    print(f"  Learned v2.0.0:  {v2['avg_attended_salience'][0]:.3f} ({(v2['avg_attended_salience'][0] - baseline['avg_attended_salience'][0])*100:+5.1f}%)")
    print(f"  Learned v3.0.0:  {v3['avg_attended_salience'][0]:.3f} ({(v3['avg_attended_salience'][0] - baseline['avg_attended_salience'][0])*100:+5.1f}%)")
    print()

    # Key findings
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    # Check if v3 has higher attention than baseline
    if v3['attention_rate'][0] > baseline['attention_rate'][0]:
        print("✅ v3.0.0 has HIGHER attention than baseline (correct direction)")
        print(f"   {baseline['attention_rate'][0]*100:.1f}% → {v3['attention_rate'][0]*100:.1f}% (+{(v3['attention_rate'][0] - baseline['attention_rate'][0])*100:.1f}%)")
    else:
        print("⚠️  v3.0.0 has lower or equal attention to baseline")
    print()

    # Check if v3 has lower attention than v2
    if v3['attention_rate'][0] < v2['attention_rate'][0]:
        print("⚠️  v3.0.0 has LOWER attention than v2.0.0")
        print(f"   This is unexpected - lower thresholds should increase attention")
    else:
        print("✅ v3.0.0 has higher attention than v2.0.0 (consistent with lower thresholds)")
    print()

    # Check attention proximity to 40% target
    target = 0.40
    v3_error = abs(v3['attention_rate'][0] - target) / target
    baseline_error = abs(baseline['attention_rate'][0] - target) / target
    v2_error = abs(v2['attention_rate'][0] - target) / target

    print(f"Distance from 40% attention target:")
    print(f"  Baseline v1.0.0: {baseline_error*100:5.1f}% error")
    print(f"  Learned v2.0.0:  {v2_error*100:5.1f}% error")
    print(f"  Learned v3.0.0:  {v3_error*100:5.1f}% error")
    print()

    if v3_error < baseline_error:
        print(f"✅ v3.0.0 is closer to target (improved by {(baseline_error - v3_error)*100:.1f}%)")
    else:
        print(f"⚠️  v3.0.0 is farther from target than baseline")
    print()

    # Overall assessment
    print("=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    print()

    print("Model Refinement Impact:")
    print("- Old heuristic model: Predicted 49% attention → Learned to INCREASE thresholds")
    print("- Refined model: Predicted 17% attention → Learned to DECREASE thresholds")
    print()
    print("This demonstrates:")
    print("1. Model accuracy determines learning direction")
    print("2. Inaccurate models lead to counterproductive learning")
    print("3. Refined models enable correct optimization")
    print()

    # Check if max attention is ~25% (as empirical data suggested)
    max_attention = max(baseline['attention_rate'][0], v2['attention_rate'][0], v3['attention_rate'][0])
    print(f"Maximum observed attention: {max_attention*100:.1f}%")
    if max_attention < 0.25:
        print("⚠️  Attention appears capped below 25% - may indicate fundamental limit")
        print("    40% target may be unachievable with current consciousness architecture")
    print()


if __name__ == "__main__":
    main()
