"""
Session 123: Emotional Regulation Strategies

Goal: Implement and test emotional regulation mechanisms for sustained emotional states

Sessions 120-122 built emotional/metabolic framework that generates and tracks emotions.
However, the system has no active regulation - emotions only decay passively (10% per turn).
This session implements active regulation strategies inspired by cognitive neuroscience.

Problem: Sustained frustration (from repeated failures) accumulates and triggers REST state,
reducing productivity. Biological systems have active regulation mechanisms (cognitive control,
prefrontal cortex modulation). Can we implement analogous strategies?

Biological Emotional Regulation:
1. **Cognitive Reappraisal**: Reinterpret situation (PFC-amygdala regulation)
2. **Attentional Control**: Shift focus away from stressor (dorsolateral PFC)
3. **Physiological Regulation**: Breathing, relaxation (autonomic modulation)
4. **Context Switching**: Change environment/task (behavioral strategy)
5. **Social Support**: Seek help (interpersonal regulation)

Computational Strategies:
1. **Cognitive Reappraisal**: Reframe frustration as learning opportunity
   - Trigger: frustration >0.5
   - Effect: -0.3 frustration, +0.1 progress (reframing)

2. **Engagement Reallocation**: Shift attention to curiosity
   - Trigger: frustration >0.6
   - Effect: +0.2 curiosity, +0.1 engagement (attention shift)

3. **State-Based Intervention**: Use REST state productively
   - Trigger: Enters REST state
   - Effect: Enhanced frustration decay in REST (2x decay rate)

4. **Proactive Prevention**: Detect rising frustration early
   - Trigger: frustration delta >0.2 in single turn
   - Effect: Immediate -0.2 frustration (early intervention)

5. **Metabolic Reset**: Force WAKE state to reset emotional baseline
   - Trigger: frustration >0.7
   - Effect: Transition to WAKE, emotional reset toward neutral

Test Design:
- **Control condition**: No regulation (passive decay only)
- **Regulation conditions**: Each strategy tested individually
- **Combined condition**: All strategies active
- **Measure**: Frustration trajectory, task performance, state transitions

Expected Discoveries:
1. Which regulation strategies are most effective?
2. Optimal intervention timing (early vs late)?
3. Do strategies interact synergistically?
4. Emergent regulation behaviors?
5. Biological realism of computational strategies?

Biological Parallels:
- Cognitive reappraisal ↔ PFC-amygdala top-down regulation
- Attentional control ↔ Dorsolateral PFC executive control
- State intervention ↔ Autonomic nervous system regulation
- Proactive prevention ↔ Predictive emotion regulation
- Metabolic reset ↔ Homeostatic emotional regulation
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timezone
from enum import Enum
import sys
import os

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


class RegulationStrategy(Enum):
    """Emotional regulation strategies."""
    NONE = "none"  # Control - passive decay only
    REAPPRAISAL = "reappraisal"  # Cognitive reframing
    ATTENTION = "attention"  # Shift focus
    STATE_BASED = "state_based"  # Use metabolic state
    PROACTIVE = "proactive"  # Early intervention
    METABOLIC_RESET = "metabolic_reset"  # Force state change
    COMBINED = "combined"  # All strategies


class EmotionalRegulator:
    """
    Emotional regulation system.

    Monitors emotional state and applies regulation strategies when needed.
    """

    def __init__(self, strategy: RegulationStrategy = RegulationStrategy.NONE):
        """Initialize regulator with strategy."""
        self.strategy = strategy
        self.interventions = []
        self.previous_frustration = 0.0

    def regulate(
        self,
        budget: EmotionalMetabolicBudget,
        context: str = ""
    ) -> Dict:
        """
        Apply regulation strategy if needed.

        Returns dict with:
        - regulated: bool (was regulation applied?)
        - strategy_used: str (which strategy?)
        - emotional_delta: dict (changes made)
        """
        result = {
            'regulated': False,
            'strategy_used': 'none',
            'emotional_delta': {},
        }

        if self.strategy == RegulationStrategy.NONE:
            return result

        frustration = budget.emotional_state.frustration
        frustration_delta = frustration - self.previous_frustration

        # Apply regulation strategies
        if self.strategy == RegulationStrategy.COMBINED:
            # Try all strategies in order
            regulated = self._try_all_strategies(budget, frustration, frustration_delta)
            if regulated:
                result['regulated'] = True
                result['strategy_used'] = 'combined'
        else:
            # Single strategy
            regulated = self._apply_single_strategy(
                budget, frustration, frustration_delta, self.strategy
            )
            if regulated:
                result['regulated'] = True
                result['strategy_used'] = self.strategy.value

        # Update tracking
        self.previous_frustration = budget.emotional_state.frustration

        if result['regulated']:
            self.interventions.append({
                'turn': len(self.interventions) + 1,
                'strategy': result['strategy_used'],
                'context': context,
            })
            logger.info(f"  [REGULATION] Strategy: {result['strategy_used']}")

        return result

    def _apply_single_strategy(
        self,
        budget: EmotionalMetabolicBudget,
        frustration: float,
        frustration_delta: float,
        strategy: RegulationStrategy,
    ) -> bool:
        """Apply a single regulation strategy."""
        if strategy == RegulationStrategy.REAPPRAISAL and frustration > 0.5:
            # Cognitive reappraisal: reframe as learning
            budget.update_emotional_state(
                frustration_delta=-0.3,
                progress_delta=0.1,  # Reframe: failure → learning
            )
            logger.info(f"    Reappraisal: -0.3 frustration, +0.1 progress")
            return True

        elif strategy == RegulationStrategy.ATTENTION and frustration > 0.6:
            # Attentional control: shift to curiosity
            budget.update_emotional_state(
                curiosity_delta=0.2,
                engagement_delta=0.1,
            )
            logger.info(f"    Attention shift: +0.2 curiosity, +0.1 engagement")
            return True

        elif strategy == RegulationStrategy.STATE_BASED and budget.metabolic_state == MetabolicState.REST:
            # Enhanced decay in REST state
            budget.update_emotional_state(
                frustration_delta=-0.3,  # 2x normal decay
            )
            logger.info(f"    State-based: Enhanced decay in REST (-0.3)")
            return True

        elif strategy == RegulationStrategy.PROACTIVE and frustration_delta > 0.2:
            # Early intervention on rapid rise
            budget.update_emotional_state(
                frustration_delta=-0.2,
            )
            logger.info(f"    Proactive: Early intervention (-0.2)")
            return True

        elif strategy == RegulationStrategy.METABOLIC_RESET and frustration > 0.7:
            # Force WAKE state and emotional reset
            # This would require modifying transition logic, simplified here
            budget.update_emotional_state(
                frustration_delta=-0.4,
                engagement_delta=0.1,
            )
            logger.info(f"    Metabolic reset: Major intervention (-0.4 frustration)")
            return True

        return False

    def _try_all_strategies(
        self,
        budget: EmotionalMetabolicBudget,
        frustration: float,
        frustration_delta: float,
    ) -> bool:
        """Try all strategies (for COMBINED mode)."""
        regulated = False

        # Proactive (highest priority - prevent escalation)
        if self._apply_single_strategy(budget, frustration, frustration_delta, RegulationStrategy.PROACTIVE):
            regulated = True

        # Metabolic reset (severe frustration)
        elif self._apply_single_strategy(budget, frustration, frustration_delta, RegulationStrategy.METABOLIC_RESET):
            regulated = True

        # Attention (high frustration)
        elif self._apply_single_strategy(budget, frustration, frustration_delta, RegulationStrategy.ATTENTION):
            regulated = True

        # Reappraisal (moderate frustration)
        elif self._apply_single_strategy(budget, frustration, frustration_delta, RegulationStrategy.REAPPRAISAL):
            regulated = True

        # State-based (in REST)
        elif self._apply_single_strategy(budget, frustration, frustration_delta, RegulationStrategy.STATE_BASED):
            regulated = True

        return regulated


def create_frustration_scenario() -> List[Dict]:
    """
    Create sustained frustration scenario.

    Simulates repeated failures to induce frustration accumulation.
    """
    turns = [
        {'id': 1, 'event': 'normal', 'frustration_delta': 0.0, 'desc': 'Baseline'},
        {'id': 2, 'event': 'failure', 'frustration_delta': 0.25, 'desc': 'First failure'},
        {'id': 3, 'event': 'failure', 'frustration_delta': 0.25, 'desc': 'Second failure'},
        {'id': 4, 'event': 'failure', 'frustration_delta': 0.25, 'desc': 'Third failure'},
        {'id': 5, 'event': 'failure', 'frustration_delta': 0.25, 'desc': 'Fourth failure'},
        {'id': 6, 'event': 'partial_success', 'frustration_delta': -0.1, 'desc': 'Small win'},
        {'id': 7, 'event': 'failure', 'frustration_delta': 0.25, 'desc': 'Fifth failure'},
        {'id': 8, 'event': 'normal', 'frustration_delta': 0.0, 'desc': 'Pause'},
        {'id': 9, 'event': 'failure', 'frustration_delta': 0.25, 'desc': 'Sixth failure'},
        {'id': 10, 'event': 'success', 'frustration_delta': -0.3, 'desc': 'Final success'},
    ]
    return turns


def run_session_123():
    """Run Session 123 emotional regulation testing."""

    logger.info("="*80)
    logger.info("SESSION 123: EMOTIONAL REGULATION STRATEGIES")
    logger.info("="*80)
    logger.info("Goal: Test emotional regulation strategies for sustained frustration")
    logger.info("")
    logger.info("Strategies:")
    logger.info("  1. NONE: Passive decay only (control)")
    logger.info("  2. REAPPRAISAL: Cognitive reframing (frustration → learning)")
    logger.info("  3. ATTENTION: Shift focus to curiosity")
    logger.info("  4. STATE_BASED: Enhanced decay in REST state")
    logger.info("  5. PROACTIVE: Early intervention on rapid rise")
    logger.info("  6. METABOLIC_RESET: Force state change")
    logger.info("  7. COMBINED: All strategies")
    logger.info("="*80)
    logger.info("\n")

    # Test all strategies
    strategies = [
        RegulationStrategy.NONE,
        RegulationStrategy.REAPPRAISAL,
        RegulationStrategy.ATTENTION,
        RegulationStrategy.PROACTIVE,
        RegulationStrategy.COMBINED,
    ]

    results = {}

    for strategy in strategies:
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING STRATEGY: {strategy.value.upper()}")
        logger.info(f"{'='*80}\n")

        # Create budget and regulator
        budget = EmotionalMetabolicBudget(
            metabolic_state=MetabolicState.WAKE,
            emotional_state=EmotionalState(
                curiosity=0.5,
                frustration=0.0,
                engagement=0.5,
                progress=0.5,
            ),
        )

        regulator = EmotionalRegulator(strategy=strategy)

        # Run scenario
        turns = create_frustration_scenario()
        frustration_trajectory = []
        intervention_count = 0

        for turn in turns:
            logger.info(f"Turn {turn['id']}: {turn['desc']}")

            # Apply frustration from event
            budget.update_emotional_state(
                frustration_delta=turn['frustration_delta']
            )

            # Apply regulation
            reg_result = regulator.regulate(budget, context=turn['desc'])
            if reg_result['regulated']:
                intervention_count += 1

            # Check metabolic state
            budget.transition_metabolic_state('normal')

            # Record
            frustration_trajectory.append({
                'turn': turn['id'],
                'frustration': budget.emotional_state.frustration,
                'state': budget.metabolic_state.value,
                'regulated': reg_result['regulated'],
            })

            logger.info(f"  Frustration: {budget.emotional_state.frustration:.2f}, "
                       f"State: {budget.metabolic_state.value}")

            # Recovery
            budget.recover()

        # Analyze
        avg_frustration = sum(t['frustration'] for t in frustration_trajectory) / len(frustration_trajectory)
        peak_frustration = max(t['frustration'] for t in frustration_trajectory)
        final_frustration = frustration_trajectory[-1]['frustration']

        results[strategy.value] = {
            'avg_frustration': avg_frustration,
            'peak_frustration': peak_frustration,
            'final_frustration': final_frustration,
            'intervention_count': intervention_count,
            'trajectory': frustration_trajectory,
        }

        logger.info(f"\nResults for {strategy.value}:")
        logger.info(f"  Avg frustration: {avg_frustration:.3f}")
        logger.info(f"  Peak frustration: {peak_frustration:.3f}")
        logger.info(f"  Final frustration: {final_frustration:.3f}")
        logger.info(f"  Interventions: {intervention_count}")

    # Comparison
    logger.info(f"\n\n{'='*80}")
    logger.info("STRATEGY COMPARISON")
    logger.info(f"{'='*80}\n")

    control = results['none']
    logger.info("Strategy           | Avg Frust | Peak Frust | Final Frust | Interventions | Improvement")
    logger.info("-"*90)
    for strategy_name, data in results.items():
        improvement = (control['avg_frustration'] - data['avg_frustration']) / control['avg_frustration'] * 100
        logger.info(f"{strategy_name:18} | {data['avg_frustration']:.3f}     | "
                   f"{data['peak_frustration']:.3f}      | {data['final_frustration']:.3f}       | "
                   f"{data['intervention_count']:13} | {improvement:+.1f}%")

    # Save results
    output = {
        'session': 123,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'description': 'Emotional regulation strategies',
        'results': results,
    }

    output_file = 'sage/experiments/session123_emotional_regulation_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n\nResults saved to: {output_file}")
    logger.info("\n" + "="*80 + "\n")

    return output


if __name__ == '__main__':
    results = run_session_123()
