"""
Session 120: Emotional/Metabolic State Integration

Goal: Extend multi-resource framework with emotional and metabolic dimensions

This session adds emotional intelligence and metabolic state management to the
multi-resource consciousness architecture developed in Sessions 107-119.

Emotional States (0-1 continuous):
- Curiosity: Drives exploration, increases attention allocation
- Frustration: Reduces reasoning quality, triggers coping strategies
- Engagement: Affects sustained attention, memory consolidation
- Progress: Positive feedback, increases confidence and risk tolerance

Metabolic States (discrete modes):
- WAKE: Normal operation baseline (standard resource budgets)
- FOCUS: High engagement (increased budgets, reduced recovery)
- REST: Lower activity (reduced budgets, increased recovery)
- DREAM: Background processing (consolidation emphasis)
- CRISIS: High stress (essential functions only, minimal budgets)

Integration Approach:
1. Emotional states modulate resource budgets (multiplicative factors)
2. Metabolic states set baseline budgets and recovery rates
3. Combined effects create rich behavioral dynamics
4. Emotions drive state transitions (engagement → FOCUS, frustration → REST)

Biological Parallels:
- Curiosity increases dopamine → enhanced learning/attention (PFC activation)
- Frustration increases cortisol → reduced reasoning (PFC inhibition)
- Engagement modulates arousal → sustained performance (LC-NE system)
- Progress triggers reward → confidence/risk-taking (VTA-NAcc pathway)

Expected Discoveries:
1. Emotion-resource coupling creates emergent regulation
2. Metabolic states gate cognitive capabilities
3. Emotional dynamics affect component priority (from S119)
4. Natural coping strategies emerge (frustration → simpler strategies)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime, timezone
import sys
import os
import numpy as np

# Add sage to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from sage.experiments.session110_crisis_mode_integration import (
    MultiResourceBudget,
    OperationalMode,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MetabolicState(Enum):
    """Metabolic states affecting baseline resource availability."""
    WAKE = "wake"       # Normal operation
    FOCUS = "focus"     # High engagement, increased resources
    REST = "rest"       # Lower activity, increased recovery
    DREAM = "dream"     # Background consolidation
    CRISIS = "crisis"   # Emergency mode, minimal resources


@dataclass
class EmotionalState:
    """Emotional state vector affecting cognitive performance."""
    curiosity: float = 0.5      # 0-1, drives exploration/attention
    frustration: float = 0.0    # 0-1, reduces reasoning quality
    engagement: float = 0.5     # 0-1, affects sustained attention
    progress: float = 0.5       # 0-1, increases confidence/risk tolerance

    def __post_init__(self):
        """Ensure all values are in [0, 1]."""
        self.curiosity = np.clip(self.curiosity, 0, 1)
        self.frustration = np.clip(self.frustration, 0, 1)
        self.engagement = np.clip(self.engagement, 0, 1)
        self.progress = np.clip(self.progress, 0, 1)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'curiosity': float(self.curiosity),
            'frustration': float(self.frustration),
            'engagement': float(self.engagement),
            'progress': float(self.progress),
        }


class EmotionalMetabolicBudget:
    """
    Extended multi-resource budget with emotional and metabolic modulation.

    Combines:
    - MultiResourceBudget (compute, memory, tool ATP)
    - EmotionalState (curiosity, frustration, engagement, progress)
    - MetabolicState (WAKE, FOCUS, REST, DREAM, CRISIS)

    Emotional modulation:
    - Curiosity increases attention budget (1.0 → 1.5x)
    - Frustration decreases reasoning budget (1.0 → 0.5x)
    - Engagement affects sustained attention and memory
    - Progress increases risk tolerance

    Metabolic baselines:
    - WAKE: 100 ATP, 2.4 ATP/cycle recovery (standard)
    - FOCUS: 150 ATP, 1.5 ATP/cycle recovery (high output, slower recovery)
    - REST: 60 ATP, 4.0 ATP/cycle recovery (low output, fast recovery)
    - DREAM: 40 ATP, 3.0 ATP/cycle recovery (consolidation only)
    - CRISIS: 30 ATP, 1.0 ATP/cycle recovery (minimal, emergency)
    """

    def __init__(
        self,
        metabolic_state: MetabolicState = MetabolicState.WAKE,
        emotional_state: Optional[EmotionalState] = None,
    ):
        """Initialize emotional/metabolic budget."""
        self.metabolic_state = metabolic_state
        self.emotional_state = emotional_state if emotional_state else EmotionalState()

        # Set baseline budgets from metabolic state
        params = self._get_metabolic_parameters(metabolic_state)

        # Create underlying multi-resource budget
        self.resource_budget = MultiResourceBudget(
            compute_atp=params['base_atp'],
            memory_atp=params['base_atp'],
            tool_atp=params['base_atp'],
            compute_recovery=params['compute_recovery'],
            memory_recovery=params['memory_recovery'],
            tool_recovery=params['tool_recovery'],
        )

        # State transition history
        self.state_history: List[Dict] = []
        self.emotion_history: List[Dict] = []

    def _get_metabolic_parameters(self, state: MetabolicState) -> Dict:
        """Get baseline ATP and recovery rates for metabolic state."""
        if state == MetabolicState.WAKE:
            return {
                'base_atp': 100.0,
                'compute_recovery': 2.4,
                'memory_recovery': 1.2,
                'tool_recovery': 12.0,
            }
        elif state == MetabolicState.FOCUS:
            return {
                'base_atp': 150.0,
                'compute_recovery': 1.5,
                'memory_recovery': 0.8,
                'tool_recovery': 8.0,
            }
        elif state == MetabolicState.REST:
            return {
                'base_atp': 60.0,
                'compute_recovery': 4.0,
                'memory_recovery': 2.0,
                'tool_recovery': 16.0,
            }
        elif state == MetabolicState.DREAM:
            return {
                'base_atp': 40.0,
                'compute_recovery': 3.0,
                'memory_recovery': 3.5,
                'tool_recovery': 5.0,
            }
        else:  # CRISIS
            return {
                'base_atp': 30.0,
                'compute_recovery': 1.0,
                'memory_recovery': 0.5,
                'tool_recovery': 3.0,
            }

    def update_emotional_state(
        self,
        curiosity_delta: float = 0.0,
        frustration_delta: float = 0.0,
        engagement_delta: float = 0.0,
        progress_delta: float = 0.0,
    ):
        """
        Update emotional state with deltas.

        Emotions decay toward neutral (0.5) over time with decay rate 0.1.
        """
        # Apply deltas
        self.emotional_state.curiosity += curiosity_delta
        self.emotional_state.frustration += frustration_delta
        self.emotional_state.engagement += engagement_delta
        self.emotional_state.progress += progress_delta

        # Decay toward neutral (0.5)
        decay_rate = 0.1
        self.emotional_state.curiosity += (0.5 - self.emotional_state.curiosity) * decay_rate
        self.emotional_state.frustration += (0.0 - self.emotional_state.frustration) * decay_rate  # Decay to 0
        self.emotional_state.engagement += (0.5 - self.emotional_state.engagement) * decay_rate
        self.emotional_state.progress += (0.5 - self.emotional_state.progress) * decay_rate

        # Clip to [0, 1]
        self.emotional_state.curiosity = np.clip(self.emotional_state.curiosity, 0, 1)
        self.emotional_state.frustration = np.clip(self.emotional_state.frustration, 0, 1)
        self.emotional_state.engagement = np.clip(self.emotional_state.engagement, 0, 1)
        self.emotional_state.progress = np.clip(self.emotional_state.progress, 0, 1)

        # Record
        self.emotion_history.append(self.emotional_state.to_dict())

    def transition_metabolic_state(self, event: str):
        """
        Transition metabolic state based on event.

        Transition logic:
        - High engagement (>0.7) → FOCUS
        - High frustration (>0.6) → REST
        - Low engagement (<0.3) and high consolidation need → DREAM
        - Crisis event → CRISIS
        - Normal operation → WAKE
        """
        old_state = self.metabolic_state

        if event == "crisis":
            new_state = MetabolicState.CRISIS
        elif self.emotional_state.frustration > 0.6:
            new_state = MetabolicState.REST
        elif self.emotional_state.engagement > 0.7:
            new_state = MetabolicState.FOCUS
        elif self.emotional_state.engagement < 0.3:
            new_state = MetabolicState.DREAM
        else:
            new_state = MetabolicState.WAKE

        if new_state != old_state:
            self.metabolic_state = new_state

            # Update resource budgets
            params = self._get_metabolic_parameters(new_state)
            self.resource_budget.compute_atp = params['base_atp']
            self.resource_budget.memory_atp = params['base_atp']
            self.resource_budget.tool_atp = params['base_atp']
            self.resource_budget.compute_recovery = params['compute_recovery']
            self.resource_budget.memory_recovery = params['memory_recovery']
            self.resource_budget.tool_recovery = params['tool_recovery']

            logger.info(f"  [METABOLIC] State transition: {old_state.value} → {new_state.value}")
            logger.info(f"    ATP: {params['base_atp']:.1f}, "
                       f"Recovery: compute={params['compute_recovery']:.1f}, "
                       f"memory={params['memory_recovery']:.1f}")

            # Record
            self.state_history.append({
                'old_state': old_state.value,
                'new_state': new_state.value,
                'event': event,
                'emotional_state': self.emotional_state.to_dict(),
            })

    def get_emotional_modulation(self, component: str) -> float:
        """
        Get emotional modulation factor for component (0.5 - 1.5x).

        Component-specific modulation:
        - Attention: Boosted by curiosity (+50%) and engagement (+30%)
        - Memory: Boosted by engagement (+40%), reduced by frustration (-30%)
        - Expert: Reduced by frustration (-50%), boosted by progress (+20%)
        - Consensus: Boosted by progress (+30%) (confidence in validation)
        - Consolidation: Boosted by engagement (+50%) (better encoding)
        """
        if component == "attention":
            return 1.0 + (self.emotional_state.curiosity * 0.5) + (self.emotional_state.engagement * 0.3)
        elif component == "memory":
            return 1.0 + (self.emotional_state.engagement * 0.4) - (self.emotional_state.frustration * 0.3)
        elif component == "expert":
            return 1.0 + (self.emotional_state.progress * 0.2) - (self.emotional_state.frustration * 0.5)
        elif component == "consensus":
            return 1.0 + (self.emotional_state.progress * 0.3)
        elif component == "consolidation":
            return 1.0 + (self.emotional_state.engagement * 0.5)
        else:
            return 1.0

    def can_afford_with_emotion(
        self,
        cost: Dict[str, float],
        component: str,
    ) -> tuple:
        """Check affordability with emotional modulation."""
        # Apply emotional modulation
        modulation = self.get_emotional_modulation(component)
        modulated_cost = {k: v / modulation for k, v in cost.items()}

        # Check with underlying resource budget
        return self.resource_budget.can_afford(modulated_cost), modulated_cost

    def consume_with_emotion(
        self,
        cost: Dict[str, float],
        component: str,
    ):
        """Consume resources with emotional modulation."""
        modulation = self.get_emotional_modulation(component)
        modulated_cost = {k: v / modulation for k, v in cost.items()}
        self.resource_budget.consume(modulated_cost)

    def recover(self):
        """Recover resources (metabolic state determines rate)."""
        self.resource_budget.recover()

    def get_operational_mode(self) -> OperationalMode:
        """Get operational mode from underlying budget."""
        return self.resource_budget.assess_operational_mode()

    def snapshot(self) -> Dict:
        """Snapshot current state."""
        return {
            'metabolic_state': self.metabolic_state.value,
            'emotional_state': self.emotional_state.to_dict(),
            'resources': {
                'compute': self.resource_budget.compute_atp,
                'memory': self.resource_budget.memory_atp,
                'tool': self.resource_budget.tool_atp,
            },
            'operational_mode': self.get_operational_mode().value,
        }


@dataclass
class EmotionalTurn:
    """A conversation turn with emotional dynamics."""
    turn_id: int
    user_message: str
    context: str
    emotional_trigger: Optional[str] = None  # "success", "failure", "interruption", "discovery"
    expected_emotion: Optional[Dict] = None


def run_session_120():
    """Run Session 120 emotional/metabolic state integration."""

    logger.info("="*80)
    logger.info("SESSION 120: EMOTIONAL/METABOLIC STATE INTEGRATION")
    logger.info("="*80)
    logger.info("Goal: Extend multi-resource framework with emotional/metabolic dimensions")
    logger.info("")
    logger.info("Emotional States: curiosity, frustration, engagement, progress")
    logger.info("Metabolic States: WAKE, FOCUS, REST, DREAM, CRISIS")
    logger.info("")
    logger.info("Test Scenario: 8-turn conversation with emotional triggers")
    logger.info("="*80)
    logger.info("\n")

    # Create emotional/metabolic budget
    budget = EmotionalMetabolicBudget(
        metabolic_state=MetabolicState.WAKE,
        emotional_state=EmotionalState(
            curiosity=0.6,
            frustration=0.0,
            engagement=0.5,
            progress=0.5,
        ),
    )

    # Test scenario with emotional triggers
    turns = [
        EmotionalTurn(1, "Tell me about neural networks", "intro",
                     emotional_trigger="discovery", expected_emotion={'curiosity': 0.2}),
        EmotionalTurn(2, "How does backpropagation work?", "learning",
                     emotional_trigger="success", expected_emotion={'progress': 0.15, 'engagement': 0.1}),
        EmotionalTurn(3, "Wait, I don't understand that explanation", "confusion",
                     emotional_trigger="failure", expected_emotion={'frustration': 0.3, 'engagement': -0.1}),
        EmotionalTurn(4, "Can you simplify it?", "retry",
                     emotional_trigger="recovery", expected_emotion={'frustration': -0.2}),
        EmotionalTurn(5, "Ah! That makes sense now!", "breakthrough",
                     emotional_trigger="success", expected_emotion={'progress': 0.2, 'curiosity': 0.15}),
        EmotionalTurn(6, "This is really interesting! Tell me more!", "excitement",
                     emotional_trigger="discovery", expected_emotion={'curiosity': 0.2, 'engagement': 0.2}),
        EmotionalTurn(7, "I'm getting tired, need a break", "fatigue",
                     emotional_trigger="fatigue", expected_emotion={'engagement': -0.3}),
        EmotionalTurn(8, "One more question before I go", "final",
                     emotional_trigger=None, expected_emotion={}),
    ]

    turn_results = []

    for turn in turns:
        logger.info(f"\n{'='*80}")
        logger.info(f"TURN {turn.turn_id}: {turn.user_message}")
        logger.info(f"{'='*80}")

        # Apply emotional trigger
        if turn.emotional_trigger and turn.expected_emotion:
            logger.info(f"  [EMOTION] Trigger: {turn.emotional_trigger}")
            budget.update_emotional_state(
                curiosity_delta=turn.expected_emotion.get('curiosity', 0.0),
                frustration_delta=turn.expected_emotion.get('frustration', 0.0),
                engagement_delta=turn.expected_emotion.get('engagement', 0.0),
                progress_delta=turn.expected_emotion.get('progress', 0.0),
            )

            logger.info(f"    Curiosity: {budget.emotional_state.curiosity:.2f}")
            logger.info(f"    Frustration: {budget.emotional_state.frustration:.2f}")
            logger.info(f"    Engagement: {budget.emotional_state.engagement:.2f}")
            logger.info(f"    Progress: {budget.emotional_state.progress:.2f}")

        # Check for metabolic state transition
        budget.transition_metabolic_state(
            event="crisis" if turn.emotional_trigger == "crisis" else "normal"
        )

        # Simulate component operations with emotional modulation
        components = ['attention', 'memory', 'expert']
        component_results = {}

        for component in components:
            base_cost = {'compute': 30.0, 'memory': 20.0}

            affordable, modulated_cost = budget.can_afford_with_emotion(base_cost, component)
            modulation = budget.get_emotional_modulation(component)

            logger.info(f"\n  [{component.upper()}]")
            logger.info(f"    Emotional modulation: {modulation:.2f}x")
            logger.info(f"    Base cost: {sum(base_cost.values()):.1f} ATP")
            logger.info(f"    Modulated cost: {sum(modulated_cost.values()):.1f} ATP")
            logger.info(f"    Affordable: {affordable}")

            if affordable:
                budget.consume_with_emotion(base_cost, component)
                component_results[component] = {
                    'executed': True,
                    'modulation': modulation,
                    'cost': sum(modulated_cost.values()),
                }
            else:
                component_results[component] = {
                    'executed': False,
                    'modulation': modulation,
                    'cost': sum(modulated_cost.values()),
                }

        # Recovery
        budget.recover()

        # Record turn
        turn_result = {
            'turn_id': turn.turn_id,
            'trigger': turn.emotional_trigger,
            'snapshot': budget.snapshot(),
            'components': component_results,
        }
        turn_results.append(turn_result)

        logger.info(f"\n  Final state: {budget.metabolic_state.value}")
        logger.info(f"  Resources: compute={budget.resource_budget.compute_atp:.1f}, "
                   f"memory={budget.resource_budget.memory_atp:.1f}")

    # Analysis
    logger.info(f"\n\n{'='*80}")
    logger.info("SESSION 120 COMPLETE - EMOTIONAL/METABOLIC INTEGRATION SUCCESS!")
    logger.info(f"{'='*80}\n")

    # Emotional trajectory analysis
    logger.info("Emotional Trajectory:")
    for i, history_point in enumerate(budget.emotion_history, 1):
        logger.info(f"  Turn {i}: C={history_point['curiosity']:.2f}, "
                   f"F={history_point['frustration']:.2f}, "
                   f"E={history_point['engagement']:.2f}, "
                   f"P={history_point['progress']:.2f}")

    # Metabolic state transitions
    logger.info(f"\nMetabolic State Transitions: {len(budget.state_history)}")
    for transition in budget.state_history:
        logger.info(f"  {transition['old_state']} → {transition['new_state']} "
                   f"(trigger: {transition['event']})")

    # Component modulation effects
    logger.info("\nComponent Modulation Effects:")
    for component in ['attention', 'memory', 'expert']:
        modulations = [r['components'][component]['modulation']
                      for r in turn_results if component in r['components']]
        avg_mod = np.mean(modulations)
        logger.info(f"  {component}: {avg_mod:.2f}x average modulation")

    # Save results
    output = {
        'session': 120,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'description': 'Emotional/metabolic state integration',
        'turn_results': turn_results,
        'emotion_history': budget.emotion_history,
        'state_history': budget.state_history,
    }

    output_file = 'sage/experiments/session120_emotional_metabolic_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n\nResults saved to: {output_file}")
    logger.info("\n" + "="*80 + "\n")

    return output


if __name__ == '__main__':
    results = run_session_120()
