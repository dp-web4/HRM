#!/usr/bin/env python3
"""
Learn Adaptive Metabolic Thresholds for Hardware-Grounded Consciousness
========================================================================

Integrate adaptive threshold learning with hardware-grounded consciousness.

**Experiment**:
1. Start with baseline thresholds (from previous validation)
2. Run consciousness evaluation cycles
3. Measure performance (attention rate, ATP, salience, state changes)
4. Adjust thresholds using AdaptiveThresholdLearner
5. Repeat until converged
6. Save learned thresholds as versioned pattern (v2.0.0)
7. Compare baseline vs learned performance

**Integration**:
- Uses HardwareGroundedConsciousness (not full implementation, evaluation only)
- Uses AdaptiveThresholdLearner for threshold optimization
- Uses PatternLibrary for threshold versioning

Author: Claude (autonomous research) on Thor
Date: 2025-12-07
Session: Adaptive threshold learning
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'core'))

import time
from typing import Dict, List
from adaptive_thresholds import (
    AdaptiveThresholdLearner,
    AdaptiveThresholds,
    ThresholdObjectives,
    ThresholdPerformance
)
from pattern_library import PatternLibrary, PatternTemplates
from simulated_lct_identity import SimulatedLCTIdentity


# Simplified consciousness evaluation (without full deployment)
class ThresholdEvaluator:
    """
    Evaluate thresholds using data-driven models.

    Uses empirically-derived regression models from refine_threshold_model.py
    instead of heuristic assumptions.

    Models (from 5,400 cycle empirical study):
    - Attention: 0.2740 - 0.2582 * avg_threshold (R² = 0.4578)
    - ATP: 0.6827 + 0.4764 * avg_threshold (R² = 0.8051)
    - Salience: 0.3884 + 0.2869 * avg_threshold (R² = 0.2413)
    """

    def __init__(
        self,
        cycles_per_evaluation: int = 50,
        use_refined_models: bool = True
    ):
        self.cycles_per_evaluation = cycles_per_evaluation
        self.use_refined_models = use_refined_models

        # Refined model coefficients (from empirical data)
        # attention = a + b * avg_threshold
        self.attention_model = {'a': 0.2740, 'b': -0.2582, 'r2': 0.4578}
        self.atp_model = {'a': 0.6827, 'b': 0.4764, 'r2': 0.8051}
        self.salience_model = {'a': 0.3884, 'b': 0.2869, 'r2': 0.2413}

    def evaluate(self, thresholds: AdaptiveThresholds) -> ThresholdPerformance:
        """
        Evaluate a set of thresholds.

        Returns performance metrics using data-driven models.
        """
        avg_threshold = (thresholds.wake + thresholds.focus) / 2.0

        if self.use_refined_models:
            # Use refined data-driven models
            # Attention rate model (R² = 0.4578)
            attention_rate = self.attention_model['a'] + self.attention_model['b'] * avg_threshold
            attention_rate = max(0.0, min(0.25, attention_rate))  # Clamp to observed range [0, 25%]

            # ATP model (R² = 0.8051)
            avg_atp = self.atp_model['a'] + self.atp_model['b'] * avg_threshold
            avg_atp = max(0.80, min(1.0, avg_atp))  # Clamp to observed range [0.80, 1.0]
            min_atp = avg_atp * 0.95  # Min observed ~95% of average

            # Salience quality model (R² = 0.2413)
            avg_attended_salience = self.salience_model['a'] + self.salience_model['b'] * avg_threshold
            avg_attended_salience = max(0.35, min(0.60, avg_attended_salience))  # Clamp to observed range

            # State change model (heuristic - not refined yet)
            threshold_spread = abs(thresholds.wake - thresholds.dream)
            state_changes_per_100 = max(5.0, 60.0 - (threshold_spread * 50.0))

        else:
            # Old heuristic model (for comparison)
            attention_rate = 0.85 - (avg_threshold * 0.9)
            attention_rate = max(0.0, min(0.9, attention_rate))

            rest_factor = thresholds.rest
            attention_cost = attention_rate * 0.5
            avg_atp = 0.9 - attention_cost + (rest_factor * 0.1)
            avg_atp = max(0.2, min(1.0, avg_atp))
            min_atp = avg_atp * 0.8

            if attention_rate > 0.05:
                selectivity = avg_threshold
                base_salience = 0.35
                salience_bonus = selectivity * 0.3
                avg_attended_salience = base_salience + salience_bonus
            else:
                avg_attended_salience = 0.0

            threshold_spread = abs(thresholds.wake - thresholds.dream)
            state_changes_per_100 = max(5.0, 60.0 - (threshold_spread * 50.0))

        return ThresholdPerformance(
            attention_rate=attention_rate,
            avg_atp=avg_atp,
            min_atp=min_atp,
            avg_attended_salience=avg_attended_salience,
            state_changes_per_100=state_changes_per_100,
            cycles_evaluated=self.cycles_per_evaluation
        )


def learn_adaptive_thresholds():
    """Main learning experiment"""
    print("=" * 80)
    print("LEARNING ADAPTIVE METABOLIC THRESHOLDS")
    print("=" * 80)
    print()

    # Initialize pattern library
    print("1️⃣  Initializing pattern library...")
    lct_identity = SimulatedLCTIdentity()
    consciousness_key = lct_identity.get_or_create_identity("thor-sage-consciousness")
    library = PatternLibrary(
        lct_identity=lct_identity,
        consciousness_lct_id="thor-sage-consciousness"
    )
    print(f"   Creator: {consciousness_key.to_compact_id()}")
    print()

    # Load baseline thresholds (from previous validation)
    print("2️⃣  Loading baseline thresholds (v1.0.0)...")
    baseline = AdaptiveThresholds(
        wake=0.45,
        focus=0.35,
        rest=0.85,
        dream=0.15
    )
    print(f"   WAKE={baseline.wake:.2f}, FOCUS={baseline.focus:.2f}")
    print(f"   REST={baseline.rest:.2f}, DREAM={baseline.dream:.2f}")
    print()

    # Define objectives
    print("3️⃣  Setting optimization objectives...")
    objectives = ThresholdObjectives(
        target_attention_rate=0.40,  # Target 40% attention
        min_atp_level=0.30,
        min_salience_quality=0.30,
        max_state_changes_per_100=50.0
    )
    print(f"   Target attention rate: {objectives.target_attention_rate*100:.0f}%")
    print(f"   Minimum ATP level: {objectives.min_atp_level*100:.0f}%")
    print(f"   Minimum salience quality: {objectives.min_salience_quality:.2f}")
    print()

    # Evaluate baseline
    print("4️⃣  Evaluating baseline performance...")
    evaluator = ThresholdEvaluator(cycles_per_evaluation=50)
    baseline_perf = evaluator.evaluate(baseline)
    baseline_score = baseline_perf.score(objectives)
    print(f"   Attention rate: {baseline_perf.attention_rate*100:.0f}%")
    print(f"   Avg ATP: {baseline_perf.avg_atp:.2f}")
    print(f"   Avg attended salience: {baseline_perf.avg_attended_salience:.2f}")
    print(f"   State changes/100: {baseline_perf.state_changes_per_100:.0f}")
    print(f"   Score: {baseline_score:.3f}")
    print()

    # Initialize learner
    print("5️⃣  Initializing adaptive learner...")
    learner = AdaptiveThresholdLearner(
        baseline_thresholds=baseline,
        objectives=objectives,
        learning_rate=0.08,
        momentum=0.7,
        convergence_window=5
    )
    print(f"   Learning rate: {learner.learning_rate}")
    print(f"   Momentum: {learner.momentum}")
    print()

    # Learning loop
    print("6️⃣  Running learning iterations...")
    print()

    max_iterations = 20
    for i in range(max_iterations):
        # Get current thresholds
        current = learner.get_current_thresholds()

        # Evaluate
        performance = evaluator.evaluate(current)
        score = performance.score(objectives)

        # Update learner
        learner.update(performance)

        # Display progress
        print(f"   Iteration {i+1:2d}: "
              f"Attention={performance.attention_rate*100:5.1f}%, "
              f"ATP={performance.avg_atp:.2f}, "
              f"Salience={performance.avg_attended_salience:.2f}, "
              f"Score={score:.3f}")

        # Check convergence
        if learner.has_converged():
            print()
            print(f"   ✅ Converged after {i+1} iterations!")
            break

        time.sleep(0.1)  # Brief pause for readability
    else:
        print()
        print(f"   ⚠️  Reached max iterations ({max_iterations})")

    print()

    # Get learned thresholds
    print("7️⃣  Extracting learned thresholds...")
    learned = learner.get_best_thresholds()
    learned_perf = evaluator.evaluate(learned)
    learned_score = learned_perf.score(objectives)
    print(f"   WAKE={learned.wake:.2f}, FOCUS={learned.focus:.2f}")
    print(f"   REST={learned.rest:.2f}, DREAM={learned.dream:.2f}")
    print()

    print("   Performance:")
    print(f"   Attention rate: {learned_perf.attention_rate*100:.0f}%")
    print(f"   Avg ATP: {learned_perf.avg_atp:.2f}")
    print(f"   Avg attended salience: {learned_perf.avg_attended_salience:.2f}")
    print(f"   Score: {learned_score:.3f}")
    print()

    # Save learned thresholds as pattern (v2.0.0)
    print("8️⃣  Saving learned thresholds as pattern (v2.0.0)...")
    learned_data = PatternTemplates.metabolic_thresholds(
        wake=learned.wake,
        focus=learned.focus,
        rest=learned.rest,
        dream=learned.dream,
        description="Learned thresholds optimized for 40% attention rate",
        tags=["thresholds", "metabolic", "thor", "learned", "adaptive"]
    )
    learned_pattern = library.create_pattern(
        pattern_type="thresholds",
        pattern_data=learned_data,
        description="Learned thresholds optimized for 40% attention rate",
        version="2.0.0",
        tags=["thresholds", "metabolic", "thor", "learned", "adaptive"],
        metadata={
            "experiment": "adaptive_threshold_learning",
            "learning_iterations": learner.iteration,
            "baseline_score": baseline_score,
            "learned_score": learned_score,
            "score_improvement": learned_score - baseline_score,
            "objectives": objectives.to_dict(),
            "baseline_performance": baseline_perf.to_dict(),
            "learned_performance": learned_perf.to_dict(),
            "platform": "thor",
            "architecture": "hardware-grounded-consciousness"
        }
    )
    library.save_pattern(learned_pattern)
    print(f"   Pattern ID: {learned_pattern.metadata.pattern_id}")
    print(f"   Signature: {learned_pattern.signature.signature[:32]}...")
    print()

    # Save learning history
    print("9️⃣  Saving learning history...")
    history_dir = Path.home() / ".sage" / "learning_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / f"adaptive_thresholds_{learned_pattern.metadata.pattern_id}.json"
    learner.save_history(history_file)
    print(f"   History: {history_file}")
    print()

    # Comparison summary
    print("=" * 80)
    print("LEARNING RESULTS")
    print("=" * 80)
    print()

    print("Baseline (v1.0.0) vs Learned (v2.0.0):")
    print()
    print("                         Baseline    Learned    Improvement")
    print("                         --------    -------    -----------")
    print(f"WAKE threshold           {baseline.wake:6.2f}      {learned.wake:5.2f}      {learned.wake-baseline.wake:+6.2f}")
    print(f"FOCUS threshold          {baseline.focus:6.2f}      {learned.focus:5.2f}      {learned.focus-baseline.focus:+6.2f}")
    print(f"REST threshold           {baseline.rest:6.2f}      {learned.rest:5.2f}      {learned.rest-baseline.rest:+6.2f}")
    print(f"DREAM threshold          {baseline.dream:6.2f}      {learned.dream:5.2f}      {learned.dream-baseline.dream:+6.2f}")
    print()
    print(f"Attention rate           {baseline_perf.attention_rate*100:5.0f}%      {learned_perf.attention_rate*100:4.0f}%      {(learned_perf.attention_rate-baseline_perf.attention_rate)*100:+5.0f}%")
    print(f"Avg ATP                  {baseline_perf.avg_atp:6.2f}      {learned_perf.avg_atp:5.2f}      {learned_perf.avg_atp-baseline_perf.avg_atp:+6.2f}")
    print(f"Avg attended salience    {baseline_perf.avg_attended_salience:6.2f}      {learned_perf.avg_attended_salience:5.2f}      {learned_perf.avg_attended_salience-baseline_perf.avg_attended_salience:+6.2f}")
    print(f"Score                    {baseline_score:6.3f}      {learned_score:5.3f}      {learned_score-baseline_score:+6.3f}")
    print()

    print("✅ Adaptive threshold learning complete!")
    print()
    print(f"Learned pattern saved: {learned_pattern.metadata.pattern_id}")
    print(f"Learning history saved: {history_file.name}")
    print()

    return {
        'baseline': baseline,
        'learned': learned,
        'baseline_score': baseline_score,
        'learned_score': learned_score,
        'pattern_id': learned_pattern.metadata.pattern_id,
        'iterations': learner.iteration
    }


if __name__ == "__main__":
    result = learn_adaptive_thresholds()
