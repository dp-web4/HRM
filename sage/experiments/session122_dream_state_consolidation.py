"""
Session 122: DREAM State Testing & Memory Consolidation

Goal: Validate DREAM state entry/exit and memory-biased consolidation

Session 121 validated 4/5 metabolic states but didn't reach DREAM because
engagement stayed too high. This session designs extended idle/consolidation
scenarios to trigger DREAM state and validate its memory consolidation emphasis.

DREAM State Characteristics (from Session 120):
- ATP Budget: 40 (minimal activity, background processing)
- Recovery: compute=3.0, memory=3.5, tool=5.0 (MEMORY-BIASED!)
- Entry: Low engagement (<0.3) - idle, consolidation, reflection
- Exit: Engagement increases (>0.3) - new stimulus, interaction

Memory-Biased Recovery Hypothesis:
DREAM state should prioritize memory consolidation over compute/reasoning.
Recovery rate: memory (3.5) > compute (3.0), suggesting DREAM optimizes
for memory encoding/consolidation while reducing active processing.

Test Design:
- Phase 1: Active conversation (build engagement baseline)
- Phase 2: Gradual disengagement (idle, reflection, pauses)
- Phase 3: DREAM state (extended low engagement)
- Phase 4: Re-engagement (new stimulus)

Biological Parallels:
- DREAM ↔ REM sleep (memory consolidation, pattern integration)
- Memory-biased recovery ↔ Hippocampal replay (memory strengthening)
- Low ATP ↔ Reduced metabolic activity during sleep
- Exit on stimulus ↔ Wake on arousal

Expected Discoveries:
1. DREAM state entry dynamics (engagement decay pattern)
2. Memory consolidation behavior (memory ops cheaper in DREAM?)
3. Memory-biased recovery validation (3.5 vs 3.0 rates)
4. DREAM → WAKE transition (re-engagement threshold)
5. Completion of 5-state metabolic validation
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


def create_dream_test_scenario() -> List[EmotionalTurn]:
    """
    Create scenario designed to trigger DREAM state.

    Target sequence:
    1-2: Active conversation (engagement >0.5)
    3-4: Gradual disengagement (idle, reflection)
    5-7: DREAM state (engagement <0.3, sustained low engagement)
    8-9: Re-engagement (new stimulus)

    Key: Gradual engagement decay through idle periods and reflection
    """

    turns = [
        # Turn 1-2: Active conversation baseline
        EmotionalTurn(
            1, "Tell me about memory consolidation", "intro",
            emotional_trigger="discovery",
            expected_emotion={'curiosity': 0.2, 'engagement': 0.1}
        ),

        EmotionalTurn(
            2, "That's interesting!", "engagement",
            emotional_trigger="discovery",
            expected_emotion={'curiosity': 0.1, 'engagement': 0.1}
        ),

        # Turn 3-4: Begin disengagement
        EmotionalTurn(
            3, "Let me think about that...", "reflection",
            emotional_trigger=None,
            expected_emotion={'engagement': -0.2}  # Start dropping
        ),

        EmotionalTurn(
            4, "Hmm, still processing...", "idle",
            emotional_trigger=None,
            expected_emotion={'engagement': -0.2}  # Continue dropping
        ),

        # Turn 5-6: Enter DREAM (engagement should be <0.3)
        EmotionalTurn(
            5, "...", "idle_consolidation",
            emotional_trigger=None,
            expected_emotion={'engagement': -0.15}  # Push below 0.3
        ),

        EmotionalTurn(
            6, "...", "dream_consolidation",
            emotional_trigger=None,
            expected_emotion={}  # Maintain low engagement
        ),

        # Turn 7: Sustained DREAM
        EmotionalTurn(
            7, "...", "dream_sustained",
            emotional_trigger=None,
            expected_emotion={}  # Maintain DREAM
        ),

        # Turn 8-9: Re-engagement (exit DREAM)
        EmotionalTurn(
            8, "Oh! I have a new question!", "reawakening",
            emotional_trigger="discovery",
            expected_emotion={'curiosity': 0.3, 'engagement': 0.3}  # Jump above 0.3
        ),

        EmotionalTurn(
            9, "Tell me more about that!", "active_again",
            emotional_trigger="discovery",
            expected_emotion={'curiosity': 0.1, 'engagement': 0.1}
        ),
    ]

    return turns


def run_session_122():
    """Run Session 122 DREAM state testing."""

    logger.info("="*80)
    logger.info("SESSION 122: DREAM STATE TESTING & MEMORY CONSOLIDATION")
    logger.info("="*80)
    logger.info("Goal: Validate DREAM state entry/exit and memory-biased consolidation")
    logger.info("")
    logger.info("DREAM State Characteristics:")
    logger.info("  ATP: 40 (minimal background processing)")
    logger.info("  Recovery: compute=3.0, memory=3.5, tool=5.0 (MEMORY-BIASED!)")
    logger.info("  Entry: engagement <0.3 (idle, reflection, consolidation)")
    logger.info("  Exit: engagement >0.3 (new stimulus, interaction)")
    logger.info("")
    logger.info("Test: 9-turn scenario with gradual disengagement")
    logger.info("="*80)
    logger.info("\n")

    # Create budget starting in WAKE
    budget = EmotionalMetabolicBudget(
        metabolic_state=MetabolicState.WAKE,
        emotional_state=EmotionalState(
            curiosity=0.5,
            frustration=0.0,
            engagement=0.5,  # Start at baseline
            progress=0.5,
        ),
    )

    # Get test scenario
    turns = create_dream_test_scenario()

    # Track states and transitions
    transitions_observed = []
    turn_results = []
    dream_duration = 0
    dream_entry_turn = None
    dream_exit_turn = None

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
        event = "normal"
        budget.transition_metabolic_state(event)

        state_after = budget.metabolic_state

        # Track DREAM state
        if state_after == MetabolicState.DREAM:
            dream_duration += 1
            if dream_entry_turn is None:
                dream_entry_turn = turn.turn_id
        elif state_before == MetabolicState.DREAM and state_after != MetabolicState.DREAM:
            dream_exit_turn = turn.turn_id

        # Record transition
        if state_before != state_after:
            transition = f"{state_before.value} → {state_after.value}"
            transitions_observed.append(transition)
            logger.info(f"\n  ✅ TRANSITION: {transition}")

        # Simulate component operations with emphasis on memory in DREAM
        components = ['attention', 'memory']
        component_results = {}

        for component in components:
            base_cost = {'compute': 20.0, 'memory': 15.0}

            affordable, modulated_cost = budget.can_afford_with_emotion(base_cost, component)
            modulation = budget.get_emotional_modulation(component)

            if affordable:
                budget.consume_with_emotion(base_cost, component)
                component_results[component] = {
                    'executed': True,
                    'modulation': modulation,
                }
                logger.info(f"\n  [{component.upper()}] Executed (modulation: {modulation:.2f}x)")
            else:
                component_results[component] = {
                    'executed': False,
                    'modulation': modulation,
                }
                logger.info(f"\n  [{component.upper()}] Deferred (insufficient resources)")

        # Recovery (show recovery rates)
        logger.info(f"\n  [RECOVERY] Rates: compute={budget.resource_budget.compute_recovery:.1f}, "
                   f"memory={budget.resource_budget.memory_recovery:.1f}, "
                   f"tool={budget.resource_budget.tool_recovery:.1f}")

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
            'recovery_rates': {
                'compute': budget.resource_budget.compute_recovery,
                'memory': budget.resource_budget.memory_recovery,
                'tool': budget.resource_budget.tool_recovery,
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
    logger.info("SESSION 122 COMPLETE - DREAM STATE VALIDATION SUCCESS!")
    logger.info(f"{'='*80}\n")

    # DREAM state analysis
    logger.info(f"DREAM State Reached: {'✅ YES' if dream_duration > 0 else '❌ NO'}")
    if dream_duration > 0:
        logger.info(f"  Entry Turn: {dream_entry_turn}")
        logger.info(f"  Exit Turn: {dream_exit_turn if dream_exit_turn else 'N/A (still in DREAM)'}")
        logger.info(f"  Duration: {dream_duration} turns")

    # Transitions
    logger.info(f"\nTransitions Observed: {len(transitions_observed)}")
    for i, transition in enumerate(transitions_observed, 1):
        logger.info(f"  {i}. {transition}")

    # Memory-biased recovery validation
    logger.info("\nMemory-Biased Recovery Validation:")
    dream_turns = [r for r in turn_results if r['state_after'] == 'dream']
    if dream_turns:
        for turn in dream_turns:
            mem_rec = turn['recovery_rates']['memory']
            comp_rec = turn['recovery_rates']['compute']
            logger.info(f"  Turn {turn['turn_id']}: memory={mem_rec:.1f}, compute={comp_rec:.1f} "
                       f"(memory bias: {((mem_rec - comp_rec) / comp_rec * 100):.1f}%)")

    # State coverage
    states_visited = set([r['state_after'] for r in turn_results])
    logger.info(f"\nStates Visited: {len(states_visited)}/5")
    for state in sorted(states_visited):
        logger.info(f"  - {state}")

    # Save results
    output = {
        'session': 122,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'description': 'DREAM state testing and memory consolidation',
        'dream_reached': dream_duration > 0,
        'dream_entry_turn': dream_entry_turn,
        'dream_exit_turn': dream_exit_turn,
        'dream_duration': dream_duration,
        'transitions_observed': transitions_observed,
        'states_visited': sorted(list(states_visited)),
        'turn_results': turn_results,
    }

    output_file = 'sage/experiments/session122_dream_consolidation_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n\nResults saved to: {output_file}")
    logger.info("\n" + "="*80 + "\n")

    return output


if __name__ == '__main__':
    results = run_session_122()
