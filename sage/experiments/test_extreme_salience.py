#!/usr/bin/env python3
"""
Test Extreme Salience Distributions - Can We Reach 40% Attention?
===================================================================

**Context**: Session 9 achieved 34% attention with Beta(5,2) and High Priority
[0.5, 1.0] distributions. This is only 6% from the 40% target.

**Question**: Can even more extreme salience distributions push us to 40%?

**Hypothesis**: Real-world environments with emergencies, novelty, and high-priority
events likely have extreme salience distributions with heavy high-end tails.

**Experiments**:
1. Extreme High Priority [0.6, 1.0] - Only very high salience events
2. Beta(8,2) - Even steeper high-salience skew than Beta(5,2)
3. Beta(10,2) - Maximum high-salience concentration
4. Bimodal - 80% routine [0.2, 0.4] + 20% critical [0.8, 1.0]

**Expected**: One or more distributions should achieve or exceed 40% attention.

Author: Claude (autonomous research) on Thor
Date: 2025-12-08
Session: Testing extreme salience distributions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'core'))

import random
from typing import Dict, List
from salience_distribution_experiments import (
    SalienceGenerator,
    ConsciousnessSimulator,
    run_experiment
)
from adaptive_thresholds import AdaptiveThresholds


class BimodalSalienceGenerator:
    """Generate salience from bimodal distribution (routine + critical events)"""

    def __init__(self, routine_min=0.2, routine_max=0.4, critical_min=0.8, critical_max=1.0, critical_prob=0.2):
        self.routine_min = routine_min
        self.routine_max = routine_max
        self.critical_min = critical_min
        self.critical_max = critical_max
        self.critical_prob = critical_prob

    def generate(self) -> float:
        """Generate salience - mostly routine, occasionally critical"""
        if random.random() < self.critical_prob:
            # Critical event (high salience)
            return self.critical_min + random.random() * (self.critical_max - self.critical_min)
        else:
            # Routine event (low salience)
            return self.routine_min + random.random() * (self.routine_max - self.routine_min)

    def describe(self) -> str:
        return f"Bimodal({int((1-self.critical_prob)*100)}% [{self.routine_min:.1f},{self.routine_max:.1f}] + {int(self.critical_prob*100)}% [{self.critical_min:.1f},{self.critical_max:.1f}])"


def main():
    print("=" * 80)
    print("TESTING EXTREME SALIENCE DISTRIBUTIONS")
    print("=" * 80)
    print()
    print("Question: Can extreme salience distributions achieve the 40% attention target?")
    print()
    print("Context from Session 9:")
    print("  - Beta(5,2): 30.9% attention (Session 7), 34% with learning (Session 9)")
    print("  - High Priority [0.5, 1.0]: 31.0% (Session 7), 34% with learning (Session 9)")
    print("  - Gap to 40% target: 6 percentage points")
    print()
    print("Hypothesis: More extreme distributions → Higher attention")
    print()

    # Fixed baseline thresholds for comparison
    thresholds = AdaptiveThresholds(wake=0.45, focus=0.35, rest=0.85, dream=0.15)
    print(f"Using fixed thresholds: WAKE={thresholds.wake:.2f}, FOCUS={thresholds.focus:.2f}\n")

    # Define extreme distributions
    experiments = [
        {
            'name': 'Baseline (Session 9)',
            'salience': SalienceGenerator('beta', alpha=5, beta=2),
            'expected': '34% (Session 9 result)',
            'description': 'Control - validated in Session 9'
        },
        {
            'name': 'Extreme High Priority',
            'salience': SalienceGenerator('uniform', min=0.6, max=1.0),
            'expected': '36-38%',
            'description': 'Only very high salience events (emergencies, crises)'
        },
        {
            'name': 'Beta(8,2)',
            'salience': SalienceGenerator('beta', alpha=8, beta=2),
            'expected': '36-39%',
            'description': 'Steeper high-salience skew than Beta(5,2)'
        },
        {
            'name': 'Beta(10,2)',
            'salience': SalienceGenerator('beta', alpha=10, beta=2),
            'expected': '38-41%',
            'description': 'Maximum high-salience concentration'
        },
        {
            'name': 'Bimodal Critical',
            'salience': BimodalSalienceGenerator(
                routine_min=0.2, routine_max=0.4,
                critical_min=0.8, critical_max=1.0,
                critical_prob=0.2
            ),
            'expected': '30-35%',
            'description': 'Realistic: 80% routine, 20% critical events'
        },
        {
            'name': 'Bimodal High Critical',
            'salience': BimodalSalienceGenerator(
                routine_min=0.2, routine_max=0.4,
                critical_min=0.8, critical_max=1.0,
                critical_prob=0.3
            ),
            'expected': '32-37%',
            'description': 'High-stress environment: 30% critical events'
        },
    ]

    print("Running experiments (10 trials × 1000 cycles each)...\n")
    print("=" * 80)

    results = []
    for exp in experiments:
        print(f"\nExperiment: {exp['name']}")
        print(f"  Distribution: {exp['salience'].describe()}")
        print(f"  {exp['description']}")
        print(f"  Expected: {exp['expected']}")

        result = run_experiment(thresholds, exp['salience'], cycles=1000, trials=10)
        results.append({
            'name': exp['name'],
            'salience_gen': exp['salience'],
            'result': result,
            'expected': exp['expected']
        })

        # Display results
        attn = result['attention']
        atp_res = result['atp']
        sal = result['salience']
        sal_dist = result['salience_dist']

        print(f"  → Attention: {attn[0]*100:5.1f}% ± {attn[1]*100:4.1f}%")
        print(f"  → ATP: {atp_res[0]:.3f} ± {atp_res[1]:.3f}")
        print(f"  → Attended salience: {sal[0]:.3f} ± {sal[1]:.3f}")
        print(f"  → Salience dist: mean={sal_dist['mean']:.2f}, range=[{sal_dist['min']:.2f}, {sal_dist['max']:.2f}]")

    # Summary analysis
    print("\n" + "=" * 80)
    print("SUMMARY: CAN WE REACH 40% ATTENTION?")
    print("=" * 80)
    print()

    # Sort by attention
    results_sorted = sorted(results, key=lambda r: r['result']['attention'][0], reverse=True)

    print("Attention Rates (Highest to Lowest):\n")
    for i, res in enumerate(results_sorted, 1):
        attn = res['result']['attention'][0]
        sal_mean = res['result']['salience_dist']['mean']
        distance_from_40 = abs(attn - 0.40)

        status = ""
        if attn >= 0.40:
            status = "✅ TARGET ACHIEVED!"
        elif attn >= 0.38:
            status = "⭐ Very close!"
        elif attn >= 0.35:
            status = "↗ Promising"

        print(f"{i}. {res['name']:25s}: {attn*100:5.1f}% attention  "
              f"(mean salience={sal_mean:.2f}, gap={distance_from_40*100:4.1f}%)  {status}")

    print()

    # Find best result
    best = results_sorted[0]
    best_attn = best['result']['attention'][0]

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    if best_attn >= 0.40:
        print("✅ SUCCESS: 40% ATTENTION TARGET ACHIEVED!")
        print()
        print(f"Best distribution: {best['name']}")
        print(f"Attention achieved: {best_attn*100:.1f}%")
        print(f"Mean salience: {best['result']['salience_dist']['mean']:.2f}")
        print()
        print("Implications:")
        print("1. 40% target IS achievable with extreme salience distributions")
        print("2. Architecture is NOT fundamentally limited to <40%")
        print("3. Real-world environments with crises/emergencies can reach 40%+")
        print("4. Validates complete research arc (Sessions 6-9-10)")
        print()
        print("Next steps:")
        print("- Measure real-world salience from sensors")
        print("- Test if camera/microphone produce extreme salience")
        print("- Deploy on Sprout for hardware validation")

    elif best_attn >= 0.38:
        print("⭐ VERY CLOSE: Within 2% of 40% target")
        print()
        print(f"Best distribution: {best['name']}")
        print(f"Attention achieved: {best_attn*100:.1f}%")
        print(f"Gap to target: {(0.40 - best_attn)*100:.1f}%")
        print()
        print("Implications:")
        print("1. 40% target appears to be near architectural ceiling")
        print("2. May need even MORE extreme distributions")
        print("3. Or: Learning + extreme salience might close the gap")
        print("4. Or: 38-39% is realistic maximum, adjust target")
        print()
        print("Next steps:")
        print("- Test Beta(15,2) or Beta(20,2)")
        print("- Run learning with extreme distributions")
        print("- Consider 38% ± 2% as realistic target range")

    elif best_attn >= 0.35:
        print("↗ IMPROVED: But 40% target remains elusive")
        print()
        print(f"Best distribution: {best['name']}")
        print(f"Attention achieved: {best_attn*100:.1f}%")
        print(f"Gap to target: {(0.40 - best_attn)*100:.1f}%")
        print()
        print("Analysis:")
        print(f"- Session 9 achieved: 34%")
        print(f"- Session 10 best: {best_attn*100:.1f}%")
        print(f"- Improvement: {(best_attn - 0.34)*100:+.1f}%")
        print()
        print("Implications:")
        print("1. Ceiling appears to be around 35-36% even with extreme salience")
        print("2. 40% may require architecture changes, not just salience")
        print("3. Or: Realistic target should be 35% ± 5%")
        print()
        print("Next steps:")
        print("- Accept 35% as validated ceiling")
        print("- Focus on real-world deployment")
        print("- Optimize for 35%, not 40%")

    else:
        print("⚠️  NO SIGNIFICANT IMPROVEMENT")
        print()
        print("Extreme distributions did not improve beyond Session 9 results.")
        print("34-35% appears to be fundamental ceiling.")

    print()

    # Salience distribution analysis
    print("=" * 80)
    print("SALIENCE DISTRIBUTION IMPACT")
    print("=" * 80)
    print()

    print("Mean Salience vs Attention Rate:")
    for res in results_sorted:
        sal_mean = res['result']['salience_dist']['mean']
        attn = res['result']['attention'][0]
        print(f"  {sal_mean:.2f} → {attn*100:5.1f}%  ({res['name']})")

    print()

    # Calculate correlation
    sal_means = [r['result']['salience_dist']['mean'] for r in results]
    attentions = [r['result']['attention'][0] for r in results]

    mean_sal = sum(sal_means) / len(sal_means)
    mean_attn = sum(attentions) / len(attentions)

    cov = sum((s - mean_sal) * (a - mean_attn) for s, a in zip(sal_means, attentions)) / len(sal_means)
    var_sal = sum((s - mean_sal)**2 for s in sal_means) / len(sal_means)
    var_attn = sum((a - mean_attn)**2 for a in attentions) / len(attentions)

    if var_sal > 0 and var_attn > 0:
        correlation = cov / (var_sal**0.5 * var_attn**0.5)
        print(f"Correlation (mean salience vs attention): r = {correlation:+.3f}")

        if abs(correlation) > 0.7:
            print("  ✅ Strong correlation maintained")
        else:
            print("  ⚠️  Correlation weakening - may be hitting ceiling")

    print()


if __name__ == "__main__":
    main()
