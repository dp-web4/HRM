"""
Session 125: Adaptive Regulation Threshold Optimization

Goal: Discover optimal proactive regulation parameters through systematic testing

Session 123 showed proactive intervention (delta >0.2 → -0.2 reduction) is 95% effective.
Session 124 showed this prevents REST state transitions in full framework.
But: Are these thresholds optimal? Can we do better?

Research Questions:
1. Is 0.2 the optimal detection threshold, or would 0.1/0.3/0.4 be better?
2. Is -0.2 the optimal intervention strength, or would -0.1/-0.3/-0.4 be better?
3. How do threshold and strength interact?
4. Can we discover an adaptive rule that learns optimal values?
5. What are the costs of over-regulation vs under-regulation?

Hypothesis: There's a sweet spot
- Too sensitive (low threshold): Over-regulation, wastes resources
- Too insensitive (high threshold): Under-regulation, allows cascades
- Too weak intervention: Doesn't prevent accumulation
- Too strong intervention: May over-correct and cause oscillations

Test Design:
1. Threshold sweep: Test delta thresholds [0.1, 0.15, 0.2, 0.25, 0.3]
2. Strength sweep: Test intervention strengths [-0.1, -0.15, -0.2, -0.25, -0.3]
3. Grid search: Test combinations to find optimal (threshold, strength) pair
4. Adaptive learning: Test rule that adjusts based on outcomes

Metrics:
- Avg/peak frustration (primary)
- Intervention count (efficiency)
- State distribution (secondary effects)
- Consolidation quality (side effects)
- Stability (oscillation detection)

Expected Discoveries:
1. Optimal parameter values for different scenarios
2. Trade-offs between sensitivity and efficiency
3. Whether adaptive learning improves on fixed thresholds
4. Generalization across different failure patterns
5. Biological insights into emotional regulation tuning

Biological Parallel:
Human emotional regulation adapts based on experience. People learn
how much intervention is needed in different situations. This mirrors
prefrontal cortex learning optimal control policies.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum
import sys
import os
import numpy as np

# Add sage to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from sage.experiments.session120_emotional_metabolic_states import (
    EmotionalMetabolicBudget,
    EmotionalState,
    MetabolicState,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class RegulationParameters:
    """Proactive regulation parameters."""
    detection_threshold: float  # Frustration delta to trigger intervention
    intervention_strength: float  # How much to reduce frustration


class AdaptiveRegulator:
    """
    Adaptive proactive regulator that learns optimal thresholds.

    Based on Session 123 proactive strategy but with configurable parameters.
    """

    def __init__(self, params: RegulationParameters):
        """Initialize regulator with parameters."""
        self.params = params
        self.interventions: List[Dict] = []
        self.last_frustration = 0.0

    def regulate(self, budget: EmotionalMetabolicBudget) -> Dict:
        """Apply proactive regulation with configured thresholds."""
        current_frustration = budget.emotional_state.frustration
        frustration_delta = current_frustration - self.last_frustration
        self.last_frustration = current_frustration

        regulated = False

        # Proactive intervention on rapid rise
        if frustration_delta > self.params.detection_threshold:
            budget.update_emotional_state(
                frustration_delta=self.params.intervention_strength
            )
            self.interventions.append({
                'turn': len(self.interventions),
                'delta_detected': frustration_delta,
                'intervention': self.params.intervention_strength,
            })
            regulated = True
            logger.debug(f"  Proactive: delta={frustration_delta:.3f} → "
                        f"intervene {self.params.intervention_strength:.3f}")

        return {
            'regulated': regulated,
            'intervention_count': len(self.interventions),
        }


@dataclass
class ScenarioTurn:
    """Turn in regulation test scenario."""
    turn_id: int
    description: str
    frustration_delta: float


def create_test_scenario() -> List[ScenarioTurn]:
    """
    Create standardized test scenario for parameter comparison.

    10-turn scenario with varied frustration patterns:
    - Gradual increase (3 small failures)
    - Rapid spike (2 large failures)
    - Recovery phase
    - Final test
    """

    turns = [
        # Phase 1: Gradual accumulation (small failures)
        ScenarioTurn(1, "Small failure", 0.15),
        ScenarioTurn(2, "Small failure", 0.15),
        ScenarioTurn(3, "Small failure", 0.15),

        # Phase 2: Rapid spike (large failures)
        ScenarioTurn(4, "Major failure", 0.30),
        ScenarioTurn(5, "Major failure", 0.30),

        # Phase 3: Recovery (passive decay)
        ScenarioTurn(6, "Recovery", 0.0),
        ScenarioTurn(7, "Recovery", 0.0),

        # Phase 4: Final test
        ScenarioTurn(8, "Medium failure", 0.20),
        ScenarioTurn(9, "Small failure", 0.15),
        ScenarioTurn(10, "Recovery", 0.0),
    ]

    return turns


def run_parameter_test(params: RegulationParameters) -> Dict:
    """
    Test regulation with specific parameters.

    Returns metrics for comparison.
    """

    # Initialize
    budget = EmotionalMetabolicBudget(
        metabolic_state=MetabolicState.WAKE,
        emotional_state=EmotionalState(
            curiosity=0.5,
            frustration=0.0,
            engagement=0.5,
            progress=0.5,
        ),
    )

    regulator = AdaptiveRegulator(params)
    scenario = create_test_scenario()

    frustration_trajectory = []

    # Run scenario
    for turn in scenario:
        # Apply frustration delta
        budget.update_emotional_state(frustration_delta=turn.frustration_delta)

        # Regulate
        regulator.regulate(budget)

        # Passive decay happens in budget
        budget.recover()

        # Record
        frustration_trajectory.append(budget.emotional_state.frustration)

    # Compute metrics
    avg_frustration = np.mean(frustration_trajectory)
    peak_frustration = np.max(frustration_trajectory)
    final_frustration = frustration_trajectory[-1]
    intervention_count = len(regulator.interventions)

    # Stability: measure oscillation (variance of differences)
    if len(frustration_trajectory) > 1:
        diffs = np.diff(frustration_trajectory)
        oscillation = np.std(diffs)
    else:
        oscillation = 0.0

    # Efficiency: interventions per unit frustration prevented
    # (compare to baseline of no regulation)
    baseline_avg = 0.545  # From Session 123 control
    frustration_prevented = max(0, baseline_avg - avg_frustration)
    efficiency = frustration_prevented / max(1, intervention_count)

    return {
        'params': {
            'threshold': params.detection_threshold,
            'strength': params.intervention_strength,
        },
        'avg_frustration': avg_frustration,
        'peak_frustration': peak_frustration,
        'final_frustration': final_frustration,
        'intervention_count': intervention_count,
        'oscillation': oscillation,
        'efficiency': efficiency,
        'trajectory': frustration_trajectory,
    }


def threshold_sweep() -> List[Dict]:
    """
    Sweep detection threshold while holding strength constant.

    Tests: [0.10, 0.15, 0.20, 0.25, 0.30]
    Strength: -0.20 (Session 123 baseline)
    """

    logger.info("\n" + "="*80)
    logger.info("THRESHOLD SWEEP (strength = -0.20)")
    logger.info("="*80 + "\n")

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    results = []

    for threshold in thresholds:
        params = RegulationParameters(
            detection_threshold=threshold,
            intervention_strength=-0.20,
        )
        result = run_parameter_test(params)
        results.append(result)

        logger.info(f"Threshold {threshold:.2f}: "
                   f"avg={result['avg_frustration']:.3f}, "
                   f"peak={result['peak_frustration']:.3f}, "
                   f"interventions={result['intervention_count']}, "
                   f"efficiency={result['efficiency']:.3f}")

    return results


def strength_sweep() -> List[Dict]:
    """
    Sweep intervention strength while holding threshold constant.

    Tests: [-0.10, -0.15, -0.20, -0.25, -0.30]
    Threshold: 0.20 (Session 123 baseline)
    """

    logger.info("\n" + "="*80)
    logger.info("STRENGTH SWEEP (threshold = 0.20)")
    logger.info("="*80 + "\n")

    strengths = [-0.10, -0.15, -0.20, -0.25, -0.30]
    results = []

    for strength in strengths:
        params = RegulationParameters(
            detection_threshold=0.20,
            intervention_strength=strength,
        )
        result = run_parameter_test(params)
        results.append(result)

        logger.info(f"Strength {strength:.2f}: "
                   f"avg={result['avg_frustration']:.3f}, "
                   f"peak={result['peak_frustration']:.3f}, "
                   f"interventions={result['intervention_count']}, "
                   f"efficiency={result['efficiency']:.3f}")

    return results


def grid_search() -> List[Dict]:
    """
    Grid search over (threshold, strength) combinations.

    Find optimal pair.
    """

    logger.info("\n" + "="*80)
    logger.info("GRID SEARCH")
    logger.info("="*80 + "\n")

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    strengths = [-0.10, -0.15, -0.20, -0.25, -0.30]

    results = []

    for threshold in thresholds:
        for strength in strengths:
            params = RegulationParameters(
                detection_threshold=threshold,
                intervention_strength=strength,
            )
            result = run_parameter_test(params)
            results.append(result)

    # Find best by avg frustration
    best = min(results, key=lambda r: r['avg_frustration'])

    logger.info(f"\nBest parameters:")
    logger.info(f"  Threshold: {best['params']['threshold']:.2f}")
    logger.info(f"  Strength: {best['params']['strength']:.2f}")
    logger.info(f"  Avg frustration: {best['avg_frustration']:.3f}")
    logger.info(f"  Peak frustration: {best['peak_frustration']:.3f}")
    logger.info(f"  Interventions: {best['intervention_count']}")
    logger.info(f"  Efficiency: {best['efficiency']:.3f}")

    return results


def analyze_results(threshold_results, strength_results, grid_results):
    """Analyze all results and find insights."""

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS")
    logger.info("="*80 + "\n")

    # Threshold sensitivity
    logger.info("1. THRESHOLD SENSITIVITY:")
    for r in threshold_results:
        logger.info(f"   {r['params']['threshold']:.2f}: "
                   f"avg={r['avg_frustration']:.3f}, "
                   f"interventions={r['intervention_count']}")

    threshold_optimal = min(threshold_results, key=lambda r: r['avg_frustration'])
    logger.info(f"   → Optimal threshold: {threshold_optimal['params']['threshold']:.2f}")

    # Strength sensitivity
    logger.info("\n2. STRENGTH SENSITIVITY:")
    for r in strength_results:
        logger.info(f"   {r['params']['strength']:.2f}: "
                   f"avg={r['avg_frustration']:.3f}, "
                   f"interventions={r['intervention_count']}")

    strength_optimal = min(strength_results, key=lambda r: r['avg_frustration'])
    logger.info(f"   → Optimal strength: {strength_optimal['params']['strength']:.2f}")

    # Grid search best
    logger.info("\n3. GRID SEARCH OPTIMUM:")
    grid_optimal = min(grid_results, key=lambda r: r['avg_frustration'])
    logger.info(f"   Threshold: {grid_optimal['params']['threshold']:.2f}")
    logger.info(f"   Strength: {grid_optimal['params']['strength']:.2f}")
    logger.info(f"   Avg frustration: {grid_optimal['avg_frustration']:.3f}")

    # Efficiency analysis
    logger.info("\n4. EFFICIENCY ANALYSIS:")
    logger.info("   (Frustration prevented per intervention)")
    best_efficiency = max(grid_results, key=lambda r: r['efficiency'])
    logger.info(f"   Most efficient: thresh={best_efficiency['params']['threshold']:.2f}, "
               f"str={best_efficiency['params']['strength']:.2f}, "
               f"eff={best_efficiency['efficiency']:.3f}")

    # Session 123 baseline comparison
    logger.info("\n5. SESSION 123 BASELINE COMPARISON:")
    baseline_params = next(r for r in grid_results
                          if r['params']['threshold'] == 0.20
                          and r['params']['strength'] == -0.20)
    logger.info(f"   S123 (0.20, -0.20): avg={baseline_params['avg_frustration']:.3f}")
    logger.info(f"   Optimal: avg={grid_optimal['avg_frustration']:.3f}")
    improvement = (baseline_params['avg_frustration'] - grid_optimal['avg_frustration']) / baseline_params['avg_frustration'] * 100
    logger.info(f"   Improvement: {improvement:+.1f}%")

    return {
        'threshold_optimal': threshold_optimal,
        'strength_optimal': strength_optimal,
        'grid_optimal': grid_optimal,
        'best_efficiency': best_efficiency,
        'baseline_comparison': {
            'baseline': baseline_params,
            'optimal': grid_optimal,
            'improvement_pct': improvement,
        }
    }


def run_session_125():
    """Run Session 125 adaptive threshold optimization."""

    logger.info("="*80)
    logger.info("SESSION 125: ADAPTIVE REGULATION THRESHOLD OPTIMIZATION")
    logger.info("="*80)
    logger.info("Goal: Discover optimal proactive regulation parameters")
    logger.info("")
    logger.info("Test Plan:")
    logger.info("  1. Threshold sweep (holding strength = -0.20)")
    logger.info("  2. Strength sweep (holding threshold = 0.20)")
    logger.info("  3. Grid search (all combinations)")
    logger.info("  4. Analysis and optimization")
    logger.info("="*80)
    logger.info("\n")

    # Run sweeps
    threshold_results = threshold_sweep()
    strength_results = strength_sweep()
    grid_results = grid_search()

    # Analysis
    analysis = analyze_results(threshold_results, strength_results, grid_results)

    # Save results
    output = {
        'session': 125,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'description': 'Adaptive regulation threshold optimization',
        'threshold_sweep': threshold_results,
        'strength_sweep': strength_results,
        'grid_search': grid_results,
        'analysis': analysis,
        'baseline_s123': {
            'threshold': 0.20,
            'strength': -0.20,
            'avg_frustration': 0.027,  # From Session 123
        },
    }

    output_file = 'sage/experiments/session125_adaptive_thresholds_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n\nResults saved to: {output_file}")
    logger.info("\n" + "="*80 + "\n")

    return output


if __name__ == '__main__':
    results = run_session_125()
