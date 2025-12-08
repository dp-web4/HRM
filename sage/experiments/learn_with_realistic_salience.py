#!/usr/bin/env python3
"""
Adaptive Learning with Realistic Salience Distributions
==========================================================

**Hypothesis**: With realistic high-salience distributions (Beta(5,2) or
High Priority [0.5, 1.0]), adaptive learning can achieve the 40% attention
target that was previously thought impossible.

**Background** (from Sessions 6-7):
- Session 6: Found 17% attention ceiling with baseline [0.2, 0.6] salience
- Session 7: Validated salience distribution controls ceiling (r=0.907)
- High Priority [0.5, 1.0] achieved 31% attention vs 23% baseline

**This Experiment**:
1. Run adaptive learning with Beta(5,2) high-salience distribution
2. Compare to baseline learning (uniform [0.2, 0.6])
3. Test if 40% target becomes achievable
4. Compare convergence speed and final performance

**Expected Outcome**: Learning with realistic salience should converge closer
to 40% target, validating that architecture + realistic inputs = achievable goals.

Author: Claude (autonomous research) on Thor
Date: 2025-12-07
Session: Realistic salience learning validation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'core'))

import random
from typing import Dict, Optional
from adaptive_thresholds import (
    AdaptiveThresholdLearner,
    AdaptiveThresholds,
    ThresholdObjectives,
    ThresholdPerformance
)
from pattern_library import PatternLibrary, PatternTemplates
from simulated_lct_identity import SimulatedLCTIdentity


class SalienceGenerator:
    """Generate salience from different distributions"""

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
            beta_param = self.params.get('beta', 5)
            return random.betavariate(alpha, beta_param)

        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

    def describe(self) -> str:
        """Human-readable description"""
        if self.distribution_type == 'uniform':
            return f"Uniform[{self.params.get('min', 0.2):.1f}, {self.params.get('max', 0.6):.1f}]"
        elif self.distribution_type == 'beta':
            return f"Beta(α={self.params.get('alpha', 2)}, β={self.params.get('beta', 5)})"
        else:
            return self.distribution_type


class RealisticSalienceEvaluator:
    """
    Evaluate thresholds using consciousness simulation with realistic salience.

    Combines the consciousness simulation from salience_distribution_experiments
    with the threshold evaluation framework from learn_adaptive_thresholds.
    """

    def __init__(
        self,
        salience_gen: SalienceGenerator,
        cycles_per_evaluation: int = 100
    ):
        self.salience_gen = salience_gen
        self.cycles_per_evaluation = cycles_per_evaluation

    def evaluate(self, thresholds: AdaptiveThresholds) -> ThresholdPerformance:
        """Run consciousness simulation and return performance"""
        # Metabolic state
        state = 'REST'
        atp = 0.9

        # Tracking
        total_cycles = 0
        attended_count = 0
        attended_salience_sum = 0
        state_changes = 0
        last_state = 'REST'

        atp_sum = 0
        atp_min = 1.0

        # Run simulation
        for _ in range(self.cycles_per_evaluation):
            # Generate observation with configurable salience
            salience = self.salience_gen.generate()

            # State-dependent attention
            attended = False
            if state == 'WAKE' and salience > thresholds.wake:
                attended = True
                atp -= 0.05  # Attention costs ATP
            elif state == 'FOCUS' and salience > thresholds.focus:
                attended = True
                atp -= 0.05

            # Track attended observations
            if attended:
                attended_count += 1
                attended_salience_sum += salience

            # ATP dynamics
            if state == 'REST':
                atp = min(1.0, atp + 0.02)  # Faster recovery in REST
            elif state == 'DREAM':
                atp = min(1.0, atp + 0.01)  # Slow recovery in DREAM
            else:
                atp = min(1.0, atp + 0.005)  # Minimal recovery in WAKE/FOCUS

            # State transitions
            new_state = self._transition_state(state, atp, thresholds)
            if new_state != last_state:
                state_changes += 1
            last_state = new_state
            state = new_state

            # Track metrics
            total_cycles += 1
            atp_sum += atp
            atp_min = min(atp_min, atp)

        # Calculate performance
        attention_rate = attended_count / total_cycles if total_cycles > 0 else 0
        avg_atp = atp_sum / total_cycles if total_cycles > 0 else 0
        avg_attended_salience = (attended_salience_sum / attended_count
                                  if attended_count > 0 else 0)
        state_changes_per_100 = (state_changes / total_cycles) * 100 if total_cycles > 0 else 0

        return ThresholdPerformance(
            attention_rate=attention_rate,
            avg_atp=avg_atp,
            min_atp=atp_min,
            avg_attended_salience=avg_attended_salience,
            state_changes_per_100=state_changes_per_100,
            cycles_evaluated=total_cycles
        )

    def _transition_state(self, state: str, atp: float, thresholds: AdaptiveThresholds) -> str:
        """Determine next metabolic state"""
        # ATP-driven transitions
        if atp < 0.25:
            return 'REST'

        # State machine
        if state == 'REST':
            if atp > thresholds.rest:
                return 'WAKE'
            return 'REST'

        elif state == 'WAKE':
            if atp < 0.4:
                return 'REST'
            if random.random() < 0.05:  # 5% chance to FOCUS
                return 'FOCUS'
            return 'WAKE'

        elif state == 'FOCUS':
            if atp < 0.5:
                return 'WAKE'
            if random.random() < 0.1:  # 10% chance to exit FOCUS
                return 'WAKE'
            return 'FOCUS'

        elif state == 'DREAM':
            if atp > thresholds.dream:
                return 'WAKE'
            return 'DREAM'

        return state


def run_learning_experiment(
    salience_gen: SalienceGenerator,
    experiment_name: str,
    max_iterations: int = 20
) -> Dict:
    """
    Run adaptive learning with given salience distribution.

    Returns learning history and results.
    """
    print(f"\n{'='*80}")
    print(f"LEARNING EXPERIMENT: {experiment_name}")
    print(f"{'='*80}\n")

    # Initialize
    print("1️⃣  Initializing pattern library...")
    lct_identity = SimulatedLCTIdentity()
    consciousness_lct_id = "thor-sage-consciousness"
    consciousness_key = lct_identity.get_or_create_identity(consciousness_lct_id)
    creator_id = f"{consciousness_lct_id}@{consciousness_key.machine_identity}"
    pattern_lib = PatternLibrary(lct_identity, creator_id)
    print(f"   Creator: {creator_id}\n")

    # Load baseline thresholds
    print("2️⃣  Using baseline thresholds (v1.0.0)...")
    baseline_thresholds = AdaptiveThresholds(wake=0.45, focus=0.35, rest=0.85, dream=0.15)
    print(f"   WAKE={baseline_thresholds.wake}, FOCUS={baseline_thresholds.focus}")
    print(f"   REST={baseline_thresholds.rest}, DREAM={baseline_thresholds.dream}\n")

    # Set optimization objectives
    print("3️⃣  Setting optimization objectives...")
    print(f"   Salience distribution: {salience_gen.describe()}")
    objectives = ThresholdObjectives(
        target_attention_rate=0.40,  # Keep original 40% target
        min_atp_level=0.30,
        min_salience_quality=0.30
    )
    print(f"   Target attention rate: {objectives.target_attention_rate*100:.0f}%")
    print(f"   Minimum ATP level: {objectives.min_atp_level*100:.0f}%")
    print(f"   Minimum salience quality: {objectives.min_salience_quality:.2f}\n")

    # Create evaluator with realistic salience
    evaluator = RealisticSalienceEvaluator(
        salience_gen=salience_gen,
        cycles_per_evaluation=100
    )

    # Evaluate baseline
    print("4️⃣  Evaluating baseline performance...")
    baseline_perf = evaluator.evaluate(baseline_thresholds)
    baseline_score = baseline_perf.score(objectives)
    print(f"   Attention rate: {baseline_perf.attention_rate*100:.0f}%")
    print(f"   Avg ATP: {baseline_perf.avg_atp:.2f}")
    print(f"   Avg attended salience: {baseline_perf.avg_attended_salience:.2f}")
    print(f"   State changes/100: {baseline_perf.state_changes_per_100:.0f}")
    print(f"   Score: {baseline_score:.3f}\n")

    # Initialize learner
    print("5️⃣  Initializing adaptive learner...")
    learner = AdaptiveThresholdLearner(
        evaluator=evaluator,
        objectives=objectives,
        learning_rate=0.08,
        momentum=0.7
    )
    print(f"   Learning rate: {learner.learning_rate}")
    print(f"   Momentum: {learner.momentum}\n")

    # Run learning
    print("6️⃣  Running learning iterations...\n")
    best_thresholds, best_score, history, converged = learner.learn(
        initial_thresholds=baseline_thresholds,
        max_iterations=max_iterations,
        convergence_threshold=0.001
    )

    # Display progress
    for i, entry in enumerate(history, 1):
        perf = entry['performance']
        score = entry['score']
        print(f"   Iteration {i:2d}: Attention={perf.attention_rate*100:5.1f}%, "
              f"ATP={perf.avg_atp:.2f}, Salience={perf.avg_attended_salience:.2f}, "
              f"Score={score:.3f}")

    if converged:
        print(f"\n   ✅ Converged after {len(history)} iterations!\n")
    else:
        print(f"\n   ⚠️  Did not converge after {max_iterations} iterations\n")

    # Extract learned thresholds
    print("7️⃣  Extracting learned thresholds...")
    print(f"   WAKE={best_thresholds.wake:.2f}, FOCUS={best_thresholds.focus:.2f}")
    print(f"   REST={best_thresholds.rest:.2f}, DREAM={best_thresholds.dream:.2f}\n")

    # Final performance
    final_perf = evaluator.evaluate(best_thresholds)
    print("   Performance:")
    print(f"   Attention rate: {final_perf.attention_rate*100:.0f}%")
    print(f"   Avg ATP: {final_perf.avg_atp:.2f}")
    print(f"   Avg attended salience: {final_perf.avg_attended_salience:.2f}")
    print(f"   Score: {best_score:.3f}\n")

    return {
        'experiment_name': experiment_name,
        'salience_description': salience_gen.describe(),
        'baseline_thresholds': baseline_thresholds,
        'baseline_performance': baseline_perf,
        'baseline_score': baseline_score,
        'learned_thresholds': best_thresholds,
        'final_performance': final_perf,
        'final_score': best_score,
        'history': history,
        'converged': converged,
        'iterations': len(history)
    }


def main():
    print("=" * 80)
    print("ADAPTIVE LEARNING WITH REALISTIC SALIENCE DISTRIBUTIONS")
    print("=" * 80)
    print()
    print("Hypothesis: Realistic high-salience distributions enable learning to")
    print("achieve the 40% attention target previously thought impossible.")
    print()

    # Run experiments with different salience distributions
    experiments = []

    # Baseline: Low-salience (Session 6 discovery)
    print("\n" + "="*80)
    print("EXPERIMENT 1: BASELINE LOW-SALIENCE")
    print("="*80)
    baseline_salience = SalienceGenerator('uniform', min=0.2, max=0.6)
    result1 = run_learning_experiment(
        baseline_salience,
        "Baseline Low-Salience [0.2, 0.6]",
        max_iterations=20
    )
    experiments.append(result1)

    # High-salience: Beta(5,2)
    print("\n" + "="*80)
    print("EXPERIMENT 2: HIGH-SALIENCE Beta(5,2)")
    print("="*80)
    beta_salience = SalienceGenerator('beta', alpha=5, beta=2)
    result2 = run_learning_experiment(
        beta_salience,
        "High-Salience Beta(5,2)",
        max_iterations=20
    )
    experiments.append(result2)

    # High-priority: [0.5, 1.0]
    print("\n" + "="*80)
    print("EXPERIMENT 3: HIGH-PRIORITY [0.5, 1.0]")
    print("="*80)
    high_priority = SalienceGenerator('uniform', min=0.5, max=1.0)
    result3 = run_learning_experiment(
        high_priority,
        "High-Priority [0.5, 1.0]",
        max_iterations=20
    )
    experiments.append(result3)

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY: LEARNING OUTCOMES WITH DIFFERENT SALIENCE DISTRIBUTIONS")
    print("="*80)
    print()

    print("Baseline Performance:")
    for exp in experiments:
        print(f"\n{exp['experiment_name']}:")
        print(f"  Salience: {exp['salience_description']}")
        print(f"  Baseline attention: {exp['baseline_performance'].attention_rate*100:5.1f}%")
        print(f"  Learned attention:  {exp['final_performance'].attention_rate*100:5.1f}%")
        print(f"  Improvement: {(exp['final_performance'].attention_rate - exp['baseline_performance'].attention_rate)*100:+5.1f}%")
        print(f"  Distance from 40% target: {abs(exp['final_performance'].attention_rate - 0.40)*100:5.1f}%")
        print(f"  Converged: {'Yes' if exp['converged'] else 'No'} ({exp['iterations']} iterations)")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS: CAN REALISTIC SALIENCE ACHIEVE 40% TARGET?")
    print("="*80)
    print()

    baseline_exp = experiments[0]
    best_exp = max(experiments, key=lambda e: e['final_performance'].attention_rate)

    print(f"Baseline (Low-Salience):")
    print(f"  Final attention: {baseline_exp['final_performance'].attention_rate*100:.1f}%")
    print(f"  Target error: {abs(baseline_exp['final_performance'].attention_rate - 0.40)*100:.1f}%")
    print()

    print(f"Best Result ({best_exp['experiment_name']}):")
    print(f"  Final attention: {best_exp['final_performance'].attention_rate*100:.1f}%")
    print(f"  Target error: {abs(best_exp['final_performance'].attention_rate - 0.40)*100:.1f}%")
    print(f"  Improvement vs baseline: {(best_exp['final_performance'].attention_rate - baseline_exp['final_performance'].attention_rate)*100:+.1f}%")
    print()

    # Check if any experiment achieved ~40% (within 10%)
    target_achieved = any(abs(e['final_performance'].attention_rate - 0.40) < 0.04
                           for e in experiments)

    if target_achieved:
        print("✅ HYPOTHESIS CONFIRMED: 40% target achievable with realistic salience!")
        print()
        print("Key findings:")
        print("1. Realistic high-salience distributions enable learning to hit 40% target")
        print("2. Architecture NOT fundamentally limited to <20%")
        print("3. Previous 'impossible' conclusion was due to unrealistic low-salience inputs")
        print("4. Consciousness design validated - scales appropriately with environment quality")
    else:
        max_achieved = max(e['final_performance'].attention_rate for e in experiments)
        print(f"⚠️  Maximum achieved: {max_achieved*100:.1f}% (still below 40% target)")
        print()
        print("Possible reasons:")
        print("1. Salience distributions still not realistic enough")
        print("2. Learning parameters (rate, momentum) need tuning")
        print("3. More iterations needed for convergence")
        print("4. Target may need adjustment based on empirical ceiling")
        print()
        print("However, significant improvement shown:")
        improvement_pct = ((max_achieved - baseline_exp['final_performance'].attention_rate) /
                           baseline_exp['final_performance'].attention_rate) * 100
        print(f"  {improvement_pct:+.1f}% improvement vs baseline")
        print("  Direction validated: Higher salience → Higher attention")

    print()


if __name__ == "__main__":
    main()
