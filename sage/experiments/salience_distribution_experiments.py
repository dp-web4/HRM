#!/usr/bin/env python3
"""
Salience Distribution Experiments - Validate Attention Ceiling Hypothesis
===========================================================================

**Hypothesis**: The ~17% attention ceiling is caused by salience distribution
[0.2, 0.6] uniform random. Changing the distribution should move the ceiling.

**Experiments**:
1. Baseline: [0.2, 0.6] uniform (current architecture)
2. Wider ranges: [0.3, 0.7], [0.4, 0.8], [0.1, 0.9]
3. Non-uniform: Beta(2,5), Beta(5,2), Normal(0.5, 0.15)
4. High-salience: [0.5, 1.0] uniform (simulated high-priority events)

**Expected Results**:
- Wider ranges → Higher ceiling (more high-salience events)
- Higher minimum → Higher ceiling (everything more salient)
- Ceiling should scale with salience distribution parameters

**Validation**: If ceiling moves proportionally with distribution, hypothesis
confirmed. If ceiling stays constant, other factors dominate.

Author: Claude (autonomous research) on Thor
Date: 2025-12-07
Session: Salience distribution hypothesis validation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'core'))

import random
from collections import Counter
from adaptive_thresholds import AdaptiveThresholds, ThresholdPerformance


class SalienceGenerator:
    """Generate salience values from different distributions"""

    def __init__(self, distribution_type: str, **params):
        self.distribution_type = distribution_type
        self.params = params

    def generate(self) -> float:
        """Generate a salience value"""
        if self.distribution_type == 'uniform':
            min_val = self.params.get('min', 0.2)
            max_val = self.params.get('max', 0.6)
            return min_val + random.random() * (max_val - min_val)

        elif self.distribution_type == 'beta':
            alpha = self.params.get('alpha', 2)
            beta = self.params.get('beta', 5)
            return random.betavariate(alpha, beta)

        elif self.distribution_type == 'normal':
            mean = self.params.get('mean', 0.5)
            std = self.params.get('std', 0.15)
            # Clip to [0, 1] range
            val = random.gauss(mean, std)
            return max(0.0, min(1.0, val))

        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

    def describe(self) -> str:
        """Human-readable description"""
        if self.distribution_type == 'uniform':
            return f"Uniform[{self.params.get('min', 0.2):.1f}, {self.params.get('max', 0.6):.1f}]"
        elif self.distribution_type == 'beta':
            return f"Beta(α={self.params.get('alpha', 2)}, β={self.params.get('beta', 5)})"
        elif self.distribution_type == 'normal':
            return f"Normal(μ={self.params.get('mean', 0.5):.1f}, σ={self.params.get('std', 0.15):.2f})"
        else:
            return self.distribution_type


class ConsciousnessSimulator:
    """Simplified consciousness simulator with configurable salience"""

    def __init__(self, thresholds: AdaptiveThresholds, salience_gen: SalienceGenerator):
        self.thresholds = thresholds
        self.salience_gen = salience_gen

        # Metabolic state
        self.state = 'REST'
        self.atp = 0.9

        # Tracking
        self.total_cycles = 0
        self.attended_count = 0
        self.attended_salience_sum = 0
        self.state_changes = 0
        self.last_state = 'REST'

        self.atp_sum = 0
        self.atp_min = 1.0

        self.state_history = []

    def cycle(self):
        """Execute one consciousness cycle"""
        # Generate observation with configurable salience
        salience = self.salience_gen.generate()

        # State-dependent attention
        attended = False
        if self.state == 'WAKE' and salience > self.thresholds.wake:
            attended = True
            self.atp -= 0.05  # Attention costs ATP
        elif self.state == 'FOCUS' and salience > self.thresholds.focus:
            attended = True
            self.atp -= 0.05

        # Track attended observations
        if attended:
            self.attended_count += 1
            self.attended_salience_sum += salience

        # ATP dynamics
        if self.state == 'REST':
            self.atp = min(1.0, self.atp + 0.02)  # Faster recovery in REST
        elif self.state == 'DREAM':
            self.atp = min(1.0, self.atp + 0.01)  # Slow recovery in DREAM
        else:
            self.atp = min(1.0, self.atp + 0.005)  # Minimal recovery in WAKE/FOCUS

        # State transitions
        new_state = self._transition_state()
        if new_state != self.last_state:
            self.state_changes += 1
        self.last_state = new_state
        self.state = new_state

        # Track metrics
        self.total_cycles += 1
        self.state_history.append(self.state)
        self.atp_sum += self.atp
        self.atp_min = min(self.atp_min, self.atp)

    def _transition_state(self) -> str:
        """Determine next metabolic state"""
        # ATP-driven transitions
        if self.atp < 0.25:
            return 'REST'

        # State machine
        if self.state == 'REST':
            if self.atp > self.thresholds.rest:
                return 'WAKE'
            return 'REST'

        elif self.state == 'WAKE':
            if self.atp < 0.4:
                return 'REST'
            if random.random() < 0.05:  # 5% chance to FOCUS
                return 'FOCUS'
            return 'WAKE'

        elif self.state == 'FOCUS':
            if self.atp < 0.5:
                return 'WAKE'
            if random.random() < 0.1:  # 10% chance to exit FOCUS
                return 'WAKE'
            return 'FOCUS'

        elif self.state == 'DREAM':
            if self.atp > self.thresholds.dream:
                return 'WAKE'
            return 'DREAM'

        return self.state

    def get_performance(self) -> ThresholdPerformance:
        """Get performance metrics"""
        attention_rate = self.attended_count / self.total_cycles if self.total_cycles > 0 else 0
        avg_atp = self.atp_sum / self.total_cycles if self.total_cycles > 0 else 0
        avg_attended_salience = (self.attended_salience_sum / self.attended_count
                                  if self.attended_count > 0 else 0)
        state_changes_per_100 = (self.state_changes / self.total_cycles) * 100 if self.total_cycles > 0 else 0

        return ThresholdPerformance(
            attention_rate=attention_rate,
            avg_atp=avg_atp,
            min_atp=self.atp_min,
            avg_attended_salience=avg_attended_salience,
            state_changes_per_100=state_changes_per_100,
            cycles_evaluated=self.total_cycles
        )


def run_experiment(thresholds: AdaptiveThresholds,
                   salience_gen: SalienceGenerator,
                   cycles: int = 1000,
                   trials: int = 10):
    """Run multiple trials with given salience distribution"""

    all_perfs = []
    all_saliences = []

    for trial in range(trials):
        sim = ConsciousnessSimulator(thresholds, salience_gen)

        # Collect salience samples for analysis
        trial_saliences = []
        for _ in range(cycles):
            trial_saliences.append(salience_gen.generate())
            sim.cycle()

        all_saliences.extend(trial_saliences)
        all_perfs.append(sim.get_performance())

    # Calculate statistics
    attention_mean = sum(p.attention_rate for p in all_perfs) / len(all_perfs)
    attention_std = (sum((p.attention_rate - attention_mean)**2 for p in all_perfs) / len(all_perfs)) ** 0.5

    atp_mean = sum(p.avg_atp for p in all_perfs) / len(all_perfs)
    atp_std = (sum((p.avg_atp - atp_mean)**2 for p in all_perfs) / len(all_perfs)) ** 0.5

    salience_mean = sum(p.avg_attended_salience for p in all_perfs) / len(all_perfs)
    salience_std = (sum((p.avg_attended_salience - salience_mean)**2 for p in all_perfs) / len(all_perfs)) ** 0.5

    # Salience distribution statistics
    all_saliences.sort()
    sal_min = all_saliences[0]
    sal_max = all_saliences[-1]
    sal_mean = sum(all_saliences) / len(all_saliences)
    sal_median = all_saliences[len(all_saliences) // 2]
    sal_p25 = all_saliences[len(all_saliences) // 4]
    sal_p75 = all_saliences[3 * len(all_saliences) // 4]

    return {
        'attention': (attention_mean, attention_std),
        'atp': (atp_mean, atp_std),
        'salience': (salience_mean, salience_std),
        'salience_dist': {
            'min': sal_min,
            'p25': sal_p25,
            'median': sal_median,
            'mean': sal_mean,
            'p75': sal_p75,
            'max': sal_max
        }
    }


def main():
    print("=" * 80)
    print("SALIENCE DISTRIBUTION EXPERIMENTS")
    print("=" * 80)
    print()
    print("Hypothesis: Attention ceiling caused by salience distribution [0.2, 0.6]")
    print("Prediction: Changing distribution should move the ceiling")
    print()

    # Use baseline thresholds for all experiments
    thresholds = AdaptiveThresholds(wake=0.45, focus=0.35, rest=0.85, dream=0.15)
    print(f"Fixed thresholds: WAKE={thresholds.wake:.2f}, FOCUS={thresholds.focus:.2f}")
    print()
    print("Running 10 trials × 1000 cycles per distribution...\n")

    # Define experiments
    experiments = [
        # Baseline
        {
            'name': 'Baseline',
            'salience': SalienceGenerator('uniform', min=0.2, max=0.6),
            'description': 'Current architecture (control)'
        },

        # Wider ranges
        {
            'name': 'Wider Range 1',
            'salience': SalienceGenerator('uniform', min=0.1, max=0.7),
            'description': 'Wider range (±0.1)'
        },
        {
            'name': 'Wider Range 2',
            'salience': SalienceGenerator('uniform', min=0.0, max=0.8),
            'description': 'Much wider range (2× spread)'
        },
        {
            'name': 'Wide Range',
            'salience': SalienceGenerator('uniform', min=0.1, max=0.9),
            'description': 'Very wide range (full spectrum)'
        },

        # Shifted ranges
        {
            'name': 'Higher Salience',
            'salience': SalienceGenerator('uniform', min=0.4, max=0.8),
            'description': 'Shifted higher (+0.2)'
        },
        {
            'name': 'High Priority',
            'salience': SalienceGenerator('uniform', min=0.5, max=1.0),
            'description': 'High-priority events only'
        },

        # Lower salience
        {
            'name': 'Lower Salience',
            'salience': SalienceGenerator('uniform', min=0.1, max=0.5),
            'description': 'Shifted lower (-0.1)'
        },

        # Non-uniform distributions
        {
            'name': 'Beta(2,5)',
            'salience': SalienceGenerator('beta', alpha=2, beta=5),
            'description': 'Skewed toward low salience (long tail)'
        },
        {
            'name': 'Beta(5,2)',
            'salience': SalienceGenerator('beta', alpha=5, beta=2),
            'description': 'Skewed toward high salience'
        },
        {
            'name': 'Normal',
            'salience': SalienceGenerator('normal', mean=0.5, std=0.15),
            'description': 'Normal distribution (common events)'
        },
    ]

    results = []

    print("=" * 80)
    print()

    for exp in experiments:
        print(f"Testing: {exp['name']}")
        print(f"  Distribution: {exp['salience'].describe()}")
        print(f"  {exp['description']}")

        result = run_experiment(thresholds, exp['salience'], cycles=1000, trials=10)
        results.append({
            'name': exp['name'],
            'salience_gen': exp['salience'],
            'result': result
        })

        # Display results
        attn = result['attention']
        atp_res = result['atp']
        sal = result['salience']
        sal_dist = result['salience_dist']

        print(f"  Attention: {attn[0]*100:5.1f}% ± {attn[1]*100:4.1f}%")
        print(f"  ATP: {atp_res[0]:.3f} ± {atp_res[1]:.3f}")
        print(f"  Attended salience: {sal[0]:.3f} ± {sal[1]:.3f}")
        print(f"  Salience distribution: [{sal_dist['min']:.2f}, {sal_dist['max']:.2f}], "
              f"mean={sal_dist['mean']:.2f}, median={sal_dist['median']:.2f}")
        print()

    # Summary analysis
    print("=" * 80)
    print("SUMMARY: ATTENTION CEILING vs SALIENCE DISTRIBUTION")
    print("=" * 80)
    print()

    # Sort by attention rate
    results_sorted = sorted(results, key=lambda r: r['result']['attention'][0], reverse=True)

    print("Attention Rate Rankings:")
    print()
    for i, res in enumerate(results_sorted, 1):
        attn = res['result']['attention'][0]
        sal_dist = res['result']['salience_dist']
        print(f"{i:2d}. {res['name']:20s}: {attn*100:5.1f}% attention  "
              f"(salience: mean={sal_dist['mean']:.2f}, range=[{sal_dist['min']:.2f}, {sal_dist['max']:.2f}])")

    print()
    print("=" * 80)
    print("HYPOTHESIS VALIDATION")
    print("=" * 80)
    print()

    # Compare baseline to experiments
    baseline_result = next(r for r in results if r['name'] == 'Baseline')
    baseline_attn = baseline_result['result']['attention'][0]

    print(f"Baseline attention: {baseline_attn*100:.1f}%")
    print()

    # Check if any distribution significantly changes ceiling
    significant_changes = []
    for res in results:
        if res['name'] == 'Baseline':
            continue

        attn = res['result']['attention'][0]
        diff = attn - baseline_attn
        pct_change = (diff / baseline_attn) * 100

        if abs(pct_change) > 10:  # >10% change is significant
            significant_changes.append({
                'name': res['name'],
                'attention': attn,
                'diff': diff,
                'pct_change': pct_change
            })

    if significant_changes:
        print(f"✅ HYPOTHESIS CONFIRMED: {len(significant_changes)} distributions show >10% change")
        print()
        for change in significant_changes:
            print(f"  {change['name']:20s}: {change['attention']*100:5.1f}% "
                  f"({change['diff']*100:+5.1f}%, {change['pct_change']:+5.1f}%)")
        print()
        print("Conclusion: Salience distribution DOES control attention ceiling")
        print("Implication: Can achieve higher attention with realistic salience")
    else:
        print("⚠️  HYPOTHESIS REJECTED: No significant ceiling movement")
        print()
        print("Conclusion: Salience distribution NOT the limiting factor")
        print("Implication: Other factors (ATP, state machine) dominate")

    print()

    # Correlation analysis
    print("=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    print()

    print("Testing if attention correlates with salience distribution parameters:")
    print()

    # Extract data for correlation
    sal_means = [r['result']['salience_dist']['mean'] for r in results]
    sal_maxes = [r['result']['salience_dist']['max'] for r in results]
    attentions = [r['result']['attention'][0] for r in results]

    # Simple correlation: attention vs mean salience
    mean_sal = sum(sal_means) / len(sal_means)
    mean_attn = sum(attentions) / len(attentions)

    cov = sum((s - mean_sal) * (a - mean_attn) for s, a in zip(sal_means, attentions)) / len(sal_means)
    var_sal = sum((s - mean_sal)**2 for s in sal_means) / len(sal_means)
    var_attn = sum((a - mean_attn)**2 for a in attentions) / len(attentions)

    correlation = cov / (var_sal**0.5 * var_attn**0.5) if var_sal > 0 and var_attn > 0 else 0

    print(f"Attention vs Mean Salience:")
    print(f"  Correlation: {correlation:+.3f}")
    if abs(correlation) > 0.7:
        print(f"  ✅ Strong correlation - salience distribution controls attention")
    elif abs(correlation) > 0.4:
        print(f"  ⚠️  Moderate correlation - salience is one factor")
    else:
        print(f"  ❌ Weak correlation - salience not dominant factor")
    print()

    # Maximum attention achieved
    max_result = results_sorted[0]
    max_attn = max_result['result']['attention'][0]
    max_sal_dist = max_result['result']['salience_dist']

    print(f"Maximum Attention Achieved:")
    print(f"  Distribution: {max_result['name']}")
    print(f"  Attention: {max_attn*100:.1f}%")
    print(f"  Salience: mean={max_sal_dist['mean']:.2f}, max={max_sal_dist['max']:.2f}")
    print(f"  Improvement over baseline: {((max_attn / baseline_attn) - 1)*100:+.1f}%")
    print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    if correlation > 0.7:
        print("1. Salience distribution is the dominant factor controlling attention")
        print("2. To achieve higher attention rates:")
        print(f"   - Use distributions with higher mean (>{baseline_result['result']['salience_dist']['mean']:.2f})")
        print(f"   - Ensure high-salience tail (max >{baseline_result['result']['salience_dist']['max']:.2f})")
        print("3. Current [0.2, 0.6] range is limiting - real-world events likely have:")
        print("   - Wider range (emergencies, novelty can exceed 0.6)")
        print("   - Heavier high-salience tail (power law, not uniform)")
        print("4. Next steps:")
        print("   - Study real sensor data to measure actual salience distributions")
        print("   - Implement realistic salience model based on empirical data")
    else:
        print("1. Salience distribution is not the only limiting factor")
        print("2. ATP dynamics and state machine also contribute to ceiling")
        print("3. Need multi-factor optimization:")
        print("   - Adjust ATP recovery/consumption rates")
        print("   - Modify state transition probabilities")
        print("   - AND use realistic salience distribution")
        print("4. Architecture-level redesign may be needed for >25% attention")

    print()


if __name__ == "__main__":
    main()
