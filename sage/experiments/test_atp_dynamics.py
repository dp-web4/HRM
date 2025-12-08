#!/usr/bin/env python3
"""
ATP Dynamics Investigation - Root Cause of 31% Attention Ceiling
=================================================================

**Context**: Session 10 discovered fundamental 31% attention ceiling that
persists regardless of salience distribution extremity. Extreme distributions
(Beta(10,2), Uniform[0.6,1.0]) all plateau at 31.1%.

**Hypothesis**: ATP dynamics create the ceiling. Mechanism:
    High attention → More ATP consumption
                  → ATP depletes faster
                  → More REST transitions needed
                  → Less WAKE/FOCUS time
                  → Attention ceiling at ~31%

**Current Parameters** (from ConsciousnessSimulator):
- Attention cost: -0.05 ATP per attend
- REST recovery: +0.02 ATP per cycle
- DREAM recovery: +0.01 ATP per cycle
- WAKE/FOCUS recovery: +0.005 ATP per cycle

**Equilibrium Calculation**:
    At 31% attention:
    - ATP consumption: 0.31 × 0.05 = 0.0155 per cycle
    - Must be balanced by recovery from state distribution

**Prediction**: If ATP is the bottleneck, then:
1. Increasing REST recovery → Ceiling raises
2. Decreasing attention cost → Ceiling raises
3. Both adjustments → Ceiling raises further
4. If ceiling doesn't move → ATP not the bottleneck

**Experiments**:
1. Baseline (current params): Expected 31% ceiling
2. 2× REST recovery (+0.04): Expected 35-40% ceiling
3. 0.6× attention cost (-0.03): Expected 35-40% ceiling
4. Both adjustments: Expected 40%+ ceiling

**Validation**: If ceiling moves proportionally with ATP parameters,
hypothesis confirmed. If ceiling stays at 31%, other factors dominate.

Author: Claude (autonomous research) on Thor
Date: 2025-12-08
Session: ATP dynamics ceiling investigation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'core'))

import random
from typing import Dict, List, Tuple
from collections import Counter
from salience_distribution_experiments import (
    SalienceGenerator,
    run_experiment
)
from adaptive_thresholds import AdaptiveThresholds, ThresholdPerformance


class ConsciousnessSimulatorATP:
    """
    Consciousness simulator with configurable ATP parameters.

    This extends the base simulator to allow testing different ATP dynamics
    to validate the hypothesis that ATP recovery/cost balance determines
    the attention ceiling.
    """

    def __init__(
        self,
        thresholds: AdaptiveThresholds,
        salience_gen: SalienceGenerator,
        attention_cost: float = 0.05,
        rest_recovery: float = 0.02,
        dream_recovery: float = 0.01,
        wake_recovery: float = 0.005
    ):
        self.thresholds = thresholds
        self.salience_gen = salience_gen

        # ATP parameters (configurable)
        self.attention_cost = attention_cost
        self.rest_recovery = rest_recovery
        self.dream_recovery = dream_recovery
        self.wake_recovery = wake_recovery

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
        """Execute one consciousness cycle with configurable ATP dynamics"""
        # Generate observation
        salience = self.salience_gen.generate()

        # State-dependent attention
        attended = False
        if self.state == 'WAKE' and salience > self.thresholds.wake:
            attended = True
            self.atp -= self.attention_cost  # Configurable cost
        elif self.state == 'FOCUS' and salience > self.thresholds.focus:
            attended = True
            self.atp -= self.attention_cost

        # Track attended observations
        if attended:
            self.attended_count += 1
            self.attended_salience_sum += salience

        # ATP dynamics (configurable recovery rates)
        if self.state == 'REST':
            self.atp = min(1.0, self.atp + self.rest_recovery)
        elif self.state == 'DREAM':
            self.atp = min(1.0, self.atp + self.dream_recovery)
        else:
            self.atp = min(1.0, self.atp + self.wake_recovery)

        # State transitions (same logic as base simulator)
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
            if random.random() < 0.05:
                return 'FOCUS'
            return 'WAKE'

        elif self.state == 'FOCUS':
            if self.atp < 0.5:
                return 'WAKE'
            if random.random() < 0.1:
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

        # State changes per 100 cycles
        state_changes_per_100 = (self.state_changes / self.total_cycles * 100) if self.total_cycles > 0 else 0

        return ThresholdPerformance(
            attention_rate=attention_rate,
            avg_atp=avg_atp,
            min_atp=self.atp_min,
            avg_attended_salience=avg_attended_salience,
            state_changes_per_100=state_changes_per_100,
            cycles_evaluated=self.total_cycles
        )

    def get_state_distribution(self) -> Dict:
        """Get state distribution separately"""
        state_counts = Counter(self.state_history)
        total = len(self.state_history)
        return {state: count/total for state, count in state_counts.items()} if total > 0 else {}


def run_atp_experiment(
    thresholds: AdaptiveThresholds,
    salience_gen: SalienceGenerator,
    attention_cost: float,
    rest_recovery: float,
    cycles: int = 1000,
    trials: int = 10
) -> Dict:
    """
    Run experiment with specific ATP parameters.

    Returns aggregated metrics across multiple trials.
    """
    attention_rates = []
    atp_levels = []
    attended_saliences = []
    state_dists = []

    for trial in range(trials):
        # Create simulator with custom ATP params
        sim = ConsciousnessSimulatorATP(
            thresholds=thresholds,
            salience_gen=salience_gen,
            attention_cost=attention_cost,
            rest_recovery=rest_recovery
        )

        # Run simulation
        for _ in range(cycles):
            sim.cycle()

        # Collect metrics
        perf = sim.get_performance()
        attention_rates.append(perf.attention_rate)
        atp_levels.append(perf.avg_atp)
        attended_saliences.append(perf.avg_attended_salience)
        state_dists.append(sim.get_state_distribution())

    # Calculate means and std devs
    def mean_std(values):
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = variance ** 0.5
        return (mean, std)

    # Aggregate state distributions
    state_keys = set()
    for dist in state_dists:
        state_keys.update(dist.keys())

    avg_state_dist = {}
    for state in state_keys:
        values = [dist.get(state, 0) for dist in state_dists]
        avg_state_dist[state] = sum(values) / len(values)

    return {
        'attention': mean_std(attention_rates),
        'atp': mean_std(atp_levels),
        'salience': mean_std(attended_saliences),
        'state_dist': avg_state_dist,
        'trials': trials,
        'cycles': cycles
    }


def main():
    print("=" * 80)
    print("ATP DYNAMICS INVESTIGATION - ROOT CAUSE OF 31% CEILING")
    print("=" * 80)
    print()
    print("Question: Does ATP recovery/cost balance create the 31% attention ceiling?")
    print()
    print("Context from Session 10:")
    print("  - All extreme salience distributions plateau at 31.1%")
    print("  - Beta(10,2), Uniform[0.6,1.0], Beta(8,2) all hit same ceiling")
    print("  - ATP decreases with higher attention (0.613 → 0.608)")
    print("  - Hypothesis: ATP equilibrium limits sustainable attention")
    print()
    print("Prediction: Adjusting ATP parameters should move ceiling")
    print()

    # Use high-salience distribution (Session 10 showed this hits ceiling)
    salience = SalienceGenerator('beta', alpha=5, beta=2)
    thresholds = AdaptiveThresholds(wake=0.45, focus=0.35, rest=0.85, dream=0.15)

    print(f"Using Beta(5,2) salience distribution (hits 31% ceiling in Session 10)")
    print(f"Fixed thresholds: WAKE={thresholds.wake:.2f}, FOCUS={thresholds.focus:.2f}\n")

    # Define experiments
    experiments = [
        {
            'name': 'Baseline (Session 10)',
            'attention_cost': 0.05,
            'rest_recovery': 0.02,
            'expected': '~31% (current ceiling)',
            'description': 'Current parameters - should replicate 31% ceiling'
        },
        {
            'name': '2× REST Recovery',
            'attention_cost': 0.05,
            'rest_recovery': 0.04,
            'expected': '35-40%',
            'description': 'Double REST recovery rate - more sustainable attention'
        },
        {
            'name': '0.6× Attention Cost',
            'attention_cost': 0.03,
            'rest_recovery': 0.02,
            'expected': '35-40%',
            'description': 'Reduce attention cost 40% - less ATP depletion'
        },
        {
            'name': 'Both Adjustments',
            'attention_cost': 0.03,
            'rest_recovery': 0.04,
            'expected': '40%+',
            'description': 'Combined: Higher recovery + Lower cost'
        },
        {
            'name': '3× REST Recovery',
            'attention_cost': 0.05,
            'rest_recovery': 0.06,
            'expected': '40%+',
            'description': 'Very high recovery - test extreme'
        },
    ]

    print("Running experiments (10 trials × 1000 cycles each)...")
    print("=" * 80)

    results = []
    for exp in experiments:
        print(f"\nExperiment: {exp['name']}")
        print(f"  ATP params: attention_cost={exp['attention_cost']:.3f}, "
              f"rest_recovery={exp['rest_recovery']:.3f}")
        print(f"  {exp['description']}")
        print(f"  Expected: {exp['expected']}")

        result = run_atp_experiment(
            thresholds=thresholds,
            salience_gen=salience,
            attention_cost=exp['attention_cost'],
            rest_recovery=exp['rest_recovery'],
            cycles=1000,
            trials=10
        )

        results.append({
            'name': exp['name'],
            'params': {
                'attention_cost': exp['attention_cost'],
                'rest_recovery': exp['rest_recovery']
            },
            'result': result,
            'expected': exp['expected']
        })

        # Display results
        attn = result['attention']
        atp_res = result['atp']
        state_dist = result['state_dist']

        print(f"  → Attention: {attn[0]*100:5.1f}% ± {attn[1]*100:4.1f}%")
        print(f"  → ATP: {atp_res[0]:.3f} ± {atp_res[1]:.3f}")
        print(f"  → State dist: WAKE={state_dist.get('WAKE', 0)*100:.1f}%, "
              f"FOCUS={state_dist.get('FOCUS', 0)*100:.1f}%, "
              f"REST={state_dist.get('REST', 0)*100:.1f}%")

    # Summary analysis
    print("\n" + "=" * 80)
    print("SUMMARY: DID ATP PARAMETERS MOVE THE CEILING?")
    print("=" * 80)
    print()

    # Sort by attention
    results_sorted = sorted(results, key=lambda r: r['result']['attention'][0], reverse=True)

    print("Attention Rates (Highest to Lowest):\n")
    baseline_attn = None
    for i, res in enumerate(results_sorted, 1):
        attn = res['result']['attention'][0]
        if res['name'] == 'Baseline (Session 10)':
            baseline_attn = attn

        delta_from_baseline = ""
        if baseline_attn is not None and res['name'] != 'Baseline (Session 10)':
            delta = (attn - baseline_attn) * 100
            delta_from_baseline = f" ({delta:+.1f}% vs baseline)"

        status = ""
        if attn >= 0.40:
            status = "✅ Exceeded 40% target!"
        elif attn >= 0.35:
            status = "↗ Above 31% ceiling"
        elif attn >= 0.31:
            status = "⚠️  At ceiling"
        else:
            status = "⬇ Below ceiling"

        print(f"{i}. {res['name']:25s}: {attn*100:5.1f}% attention  {delta_from_baseline:15s}  {status}")

    print()

    # Analyze ceiling movement
    baseline_result = next((r for r in results if r['name'] == 'Baseline (Session 10)'), None)
    if baseline_result:
        baseline_attn = baseline_result['result']['attention'][0]

        print("=" * 80)
        print("CEILING MOVEMENT ANALYSIS")
        print("=" * 80)
        print()
        print(f"Baseline ceiling: {baseline_attn*100:.1f}%")
        print()

        ceiling_moved = False
        for res in results:
            if res['name'] == 'Baseline (Session 10)':
                continue

            attn = res['result']['attention'][0]
            delta = (attn - baseline_attn) * 100

            if abs(delta) > 2.0:  # >2% change is significant
                ceiling_moved = True
                print(f"  {res['name']:25s}: {attn*100:5.1f}% ({delta:+.1f}%) ← CEILING MOVED")
            else:
                print(f"  {res['name']:25s}: {attn*100:5.1f}% ({delta:+.1f}%) ← No change")

        print()

        if ceiling_moved:
            print("✅ HYPOTHESIS VALIDATED: ATP parameters move the ceiling!")
            print()
            print("Implications:")
            print("1. ATP dynamics ARE the root cause of 31% ceiling")
            print("2. Ceiling is adjustable through ATP parameter tuning")
            print("3. Session 10's 31% was equilibrium of current ATP params")
            print("4. Higher recovery or lower cost enables higher attention")
            print()
            print("Architecture Assessment:")
            print("- Design is sound (scales with ATP capacity)")
            print("- 31% ceiling was parameter choice, not fundamental limit")
            print("- Can optimize ATP params for target attention rate")
        else:
            print("❌ HYPOTHESIS REJECTED: ATP parameters don't significantly affect ceiling")
            print()
            print("Implications:")
            print("1. ATP dynamics are NOT the primary bottleneck")
            print("2. Ceiling is created by other factors (state machine, efficiency)")
            print("3. Need to investigate alternative hypotheses")
            print("4. 31% may be more fundamental architectural limit")

    print()

    # ATP consumption analysis
    print("=" * 80)
    print("ATP EQUILIBRIUM ANALYSIS")
    print("=" * 80)
    print()

    print("ATP Consumption vs Recovery Balance:\n")
    for res in results_sorted:
        attn = res['result']['attention'][0]
        atp_avg = res['result']['atp'][0]
        params = res['params']

        # Calculate equilibrium
        consumption = attn * params['attention_cost']
        # Approximate recovery (assumes ~70% REST, 30% WAKE/FOCUS based on typical state dist)
        state_dist = res['result']['state_dist']
        rest_pct = state_dist.get('REST', 0.7)
        wake_pct = state_dist.get('WAKE', 0.25)
        focus_pct = state_dist.get('FOCUS', 0.05)

        recovery = (rest_pct * params['rest_recovery'] +
                   (wake_pct + focus_pct) * 0.005)

        balance = recovery - consumption

        print(f"{res['name']:25s}:")
        print(f"  Attention: {attn*100:5.1f}%  ATP: {atp_avg:.3f}")
        print(f"  Consumption: {consumption:.4f}  Recovery: {recovery:.4f}  "
              f"Balance: {balance:+.4f}")
        print()

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    best = results_sorted[0]
    best_attn = best['result']['attention'][0]

    if best_attn > baseline_attn * 1.05:  # >5% improvement
        print(f"✅ ATP parameters significantly affect ceiling!")
        print()
        print(f"Best configuration: {best['name']}")
        print(f"  Attention: {best_attn*100:.1f}%")
        print(f"  Improvement: {(best_attn - baseline_attn)*100:+.1f}% vs baseline")
        print()
        print("Next steps:")
        print("1. ATP dynamics confirmed as ceiling mechanism")
        print("2. Optimize ATP parameters for target performance")
        print("3. Consider biological realism vs performance trade-offs")
        print("4. Test on real hardware with optimized parameters")
    else:
        print("⚠️  ATP parameters have minimal effect on ceiling")
        print()
        print("Need to investigate alternative hypotheses:")
        print("1. State machine time budget constraints")
        print("2. Attention processing efficiency limits")
        print("3. Threshold interaction effects")
        print("4. Other architectural bottlenecks")

    print()


if __name__ == "__main__":
    main()
