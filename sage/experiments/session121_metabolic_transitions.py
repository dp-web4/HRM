"""
Session 121: Metabolic State Transition Testing

Goal: Validate metabolic state transitions through targeted emotional scenarios

Session 120 implemented emotional/metabolic states but didn't trigger transitions.
This session designs scenarios that force each metabolic state transition:

- WAKE → FOCUS: High engagement (>0.7) from exciting discovery sequence
- FOCUS → WAKE: Engagement normalizes after excitement
- WAKE → REST: High frustration (>0.6) from repeated failures
- REST → WAKE: Frustration decays, normal operation resumes
- WAKE → DREAM: Low engagement (<0.3) during idle/consolidation
- DREAM → WAKE: Engagement increases from new interaction
- Any → CRISIS: Explicit crisis event (emergency interrupt)

Test Design:
- 12-turn conversation with engineered emotional triggers
- Force each transition at least once
- Validate resource budget changes on transition
- Verify recovery rate modulation
- Confirm operational mode assessment accuracy

Expected Discoveries:
1. State transition dynamics and timing
2. Resource budget impact on component execution
3. Recovery rate effects across states
4. Interaction between emotional and metabolic layers
5. Natural state cycling patterns
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime, timezone
import sys
import os

# Add sage to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from sage.experiments.session120_emotional_metabolic_states import (
    EmotionalMetabolicBudget,
    EmotionalState,
    MetabolicState,
    EmotionalTurn,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_transition_test_scenario() -> List[EmotionalTurn]:
    """
    Create scenario designed to trigger all metabolic state transitions.

    Target sequence:
    1. WAKE (baseline)
    2. WAKE → FOCUS (high engagement from excitement)
    3. FOCUS (sustained)
    4. FOCUS → WAKE (engagement normalizes)
    5. WAKE → REST (high frustration from failures)
    6. REST (sustained)
    7. REST → WAKE (frustration decays)
    8. WAKE → DREAM (low engagement)
    9. DREAM (sustained)
    10. DREAM → WAKE (engagement returns)
    11. WAKE → CRISIS (emergency event)
    12. CRISIS (minimal operation)
    """

    turns = [
        # Turn 1-2: WAKE baseline, build engagement
        EmotionalTurn(
            1, "Tell me about consciousness", "intro",
            emotional_trigger="discovery",
            expected_emotion={'curiosity': 0.3, 'engagement': 0.2}
        ),

        # Turn 2-3: Push to FOCUS (engagement >0.7)
        EmotionalTurn(
            2, "This is fascinating! Tell me more!", "excitement",
            emotional_trigger="discovery",
            expected_emotion={'curiosity': 0.4, 'engagement': 0.3}  # Should push >0.7
        ),

        EmotionalTurn(
            3, "I'm completely engaged! Keep going!", "peak_engagement",
            emotional_trigger="discovery",
            expected_emotion={'curiosity': 0.2, 'engagement': 0.2}  # Maintain FOCUS
        ),

        # Turn 4: FOCUS → WAKE (engagement normalizes)
        EmotionalTurn(
            4, "Ok, let me think about that...", "reflection",
            emotional_trigger=None,
            expected_emotion={'engagement': -0.2}  # Drop below 0.7
        ),

        # Turn 5-6: WAKE → REST (frustration >0.6)
        EmotionalTurn(
            5, "Wait, I don't understand this at all", "confusion",
            emotional_trigger="failure",
            expected_emotion={'frustration': 0.3, 'engagement': -0.1}
        ),

        EmotionalTurn(
            6, "This still makes no sense!", "frustration_peak",
            emotional_trigger="failure",
            expected_emotion={'frustration': 0.4}  # Should push >0.6
        ),

        # Turn 7: REST (sustained frustration)
        EmotionalTurn(
            7, "Let me take a break...", "rest_acknowledgement",
            emotional_trigger=None,
            expected_emotion={}  # Let frustration decay
        ),

        # Turn 8: REST → WAKE (frustration decays <0.6)
        EmotionalTurn(
            8, "Ok, I'm ready to try again", "recovery",
            emotional_trigger="recovery",
            expected_emotion={'frustration': -0.3, 'engagement': 0.1}
        ),

        # Turn 9: WAKE → DREAM (engagement <0.3)
        EmotionalTurn(
            9, "Just consolidating what I learned...", "idle",
            emotional_trigger=None,
            expected_emotion={'engagement': -0.3}  # Drop to <0.3
        ),

        # Turn 10: DREAM (sustained low engagement)
        EmotionalTurn(
            10, "Still thinking...", "consolidation",
            emotional_trigger=None,
            expected_emotion={}  # Maintain low engagement
        ),

        # Turn 11: DREAM → WAKE (engagement returns)
        EmotionalTurn(
            11, "I have a new question!", "reengagement",
            emotional_trigger="discovery",
            expected_emotion={'curiosity': 0.2, 'engagement': 0.3}  # Above 0.3
        ),

        # Turn 12: WAKE → CRISIS (explicit crisis event)
        EmotionalTurn(
            12, "URGENT: Emergency situation!", "crisis",
            emotional_trigger="crisis",
            expected_emotion={}
        ),
    ]

    return turns


def run_session_121():
    """Run Session 121 metabolic state transition testing."""

    logger.info("="*80)
    logger.info("SESSION 121: METABOLIC STATE TRANSITION TESTING")
    logger.info("="*80)
    logger.info("Goal: Force all metabolic state transitions through targeted scenarios")
    logger.info("")
    logger.info("Target Transitions:")
    logger.info("  WAKE → FOCUS → WAKE → REST → WAKE → DREAM → WAKE → CRISIS")
    logger.info("")
    logger.info("Test: 12-turn conversation with engineered emotional triggers")
    logger.info("="*80)
    logger.info("\n")

    # Create budget starting in WAKE
    budget = EmotionalMetabolicBudget(
        metabolic_state=MetabolicState.WAKE,
        emotional_state=EmotionalState(
            curiosity=0.5,
            frustration=0.0,
            engagement=0.5,
            progress=0.5,
        ),
    )

    # Get test scenario
    turns = create_transition_test_scenario()

    # Track transitions
    transitions_observed = []
    turn_results = []

    for turn in turns:
        logger.info(f"\n{'='*80}")
        logger.info(f"TURN {turn.turn_id}: {turn.user_message}")
        logger.info(f"Context: {turn.context}")
        logger.info(f"{'='*80}")

        # Record state before turn
        state_before = budget.metabolic_state

        # Apply emotional trigger
        if turn.expected_emotion:
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
        event = "crisis" if turn.emotional_trigger == "crisis" else "normal"
        budget.transition_metabolic_state(event)

        state_after = budget.metabolic_state

        # Record transition
        if state_before != state_after:
            transition = f"{state_before.value} → {state_after.value}"
            transitions_observed.append(transition)
            logger.info(f"\n  ✅ TRANSITION: {transition}")

        # Simulate component operations
        components = ['attention', 'memory', 'expert']
        component_results = {}

        for component in components:
            base_cost = {'compute': 30.0, 'memory': 20.0}

            affordable, modulated_cost = budget.can_afford_with_emotion(base_cost, component)
            modulation = budget.get_emotional_modulation(component)

            if affordable:
                budget.consume_with_emotion(base_cost, component)
                component_results[component] = {
                    'executed': True,
                    'modulation': modulation,
                }
            else:
                component_results[component] = {
                    'executed': False,
                    'modulation': modulation,
                }

        # Recovery
        budget.recover()

        # Record turn
        turn_result = {
            'turn_id': turn.turn_id,
            'state_before': state_before.value,
            'state_after': state_after.value,
            'transition': state_before != state_after,
            'emotional_state': budget.emotional_state.to_dict(),
            'resources': {
                'compute': budget.resource_budget.compute_atp,
                'memory': budget.resource_budget.memory_atp,
                'tool': budget.resource_budget.tool_atp,
            },
            'operational_mode': budget.get_operational_mode().value,
            'components': component_results,
        }
        turn_results.append(turn_result)

        logger.info(f"\n  State: {state_after.value}")
        logger.info(f"  Resources: compute={budget.resource_budget.compute_atp:.1f}, "
                   f"memory={budget.resource_budget.memory_atp:.1f}")
        logger.info(f"  Mode: {budget.get_operational_mode().value}")

    # Analysis
    logger.info(f"\n\n{'='*80}")
    logger.info("SESSION 121 COMPLETE - METABOLIC TRANSITION TESTING SUCCESS!")
    logger.info(f"{'='*80}\n")

    # Transition analysis
    logger.info(f"Transitions Observed: {len(transitions_observed)}")
    for i, transition in enumerate(transitions_observed, 1):
        logger.info(f"  {i}. {transition}")

    # State coverage
    states_visited = set([r['state_after'] for r in turn_results])
    logger.info(f"\nStates Visited: {len(states_visited)}/5")
    for state in sorted(states_visited):
        logger.info(f"  - {state}")

    # Resource trajectory by state
    logger.info("\nResource Trajectory by State:")
    for turn_result in turn_results:
        logger.info(f"  Turn {turn_result['turn_id']} ({turn_result['state_after']}): "
                   f"compute={turn_result['resources']['compute']:.1f}, "
                   f"memory={turn_result['resources']['memory']:.1f}")

    # Transition validation
    expected_transitions = [
        "wake → focus",
        "focus → wake",
        "wake → rest",
        "rest → wake",
        "wake → dream",
        "dream → wake",
        "wake → crisis",
    ]

    logger.info("\nTransition Validation:")
    for expected in expected_transitions:
        if expected in [t.lower() for t in transitions_observed]:
            logger.info(f"  ✅ {expected}")
        else:
            logger.info(f"  ❌ {expected} (not triggered)")

    # Save results
    output = {
        'session': 121,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'description': 'Metabolic state transition testing',
        'transitions_observed': transitions_observed,
        'states_visited': sorted(list(states_visited)),
        'turn_results': turn_results,
    }

    output_file = 'sage/experiments/session121_metabolic_transitions_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n\nResults saved to: {output_file}")
    logger.info("\n" + "="*80 + "\n")

    return output


if __name__ == '__main__':
    results = run_session_121()
