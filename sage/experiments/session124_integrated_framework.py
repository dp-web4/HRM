"""
Session 124: Integrated Framework Validation

Goal: Test proactive emotional regulation within full emotional/metabolic/consolidation framework

Session 123 demonstrated proactive regulation is 95% effective in isolation.
Sessions 120-122 built emotional/metabolic states with memory consolidation.
This session integrates them to discover emergent behaviors.

Research Question: Does proactive regulation change metabolic state dynamics?
- Does preventing frustration reduce REST state transitions?
- Does maintaining engagement improve FOCUS sustainability?
- How does regulation affect consolidation quality across states?
- Do new regulatory patterns emerge from full integration?

Test Design:
- Complex 15-turn scenario (discovery → frustration → recovery → re-engagement)
- Compare CONTROL (no regulation) vs PROACTIVE (early intervention)
- Track: state transitions, resource trajectories, consolidation quality
- Measure: state distribution, transition patterns, memory quality

Expected Discoveries:
1. State transition dynamics with vs without regulation
2. Resource sustainability across states
3. Consolidation quality impact
4. Emergent regulatory behaviors in full system
5. Production readiness validation

Integration Architecture:
- EmotionalMetabolicBudget (S120) - base framework
- StateAwareConsolidator (S122) - consolidation quality by state
- EmotionalRegulator (S123) - proactive intervention
- Full loop: emotion → regulation → state → resources → consolidation

Biological Parallel:
This integrates PFC emotional regulation with metabolic state management
and sleep consolidation - mimicking how human cognition coordinates
emotion, arousal, and memory in real time.
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

from sage.experiments.session123_emotional_regulation import (
    EmotionalRegulator,
    RegulationStrategy,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """Memory to be consolidated."""
    content: str
    encoding_quality: float
    encoded_state: MetabolicState
    consolidated: bool = False
    consolidation_quality: float = 0.0
    consolidation_state: Optional[MetabolicState] = None


class StateAwareConsolidator:
    """
    Memory consolidator with state-aware quality modulation.

    Based on Session 122 findings:
    - DREAM: 2.0x quality (sleep consolidation)
    - REST: 1.2x quality (quiet waking)
    - WAKE: 1.0x quality (baseline)
    - FOCUS: 0.5x quality (encoding priority)
    - CRISIS: 0.0x quality (no consolidation)
    """

    STATE_MULTIPLIERS = {
        MetabolicState.DREAM: 2.0,
        MetabolicState.REST: 1.2,
        MetabolicState.WAKE: 1.0,
        MetabolicState.FOCUS: 0.5,
        MetabolicState.CRISIS: 0.0,
    }

    def __init__(self):
        """Initialize consolidator."""
        self.memories: List[Memory] = []

    def encode(
        self,
        content: str,
        quality: float,
        state: MetabolicState,
    ) -> Memory:
        """Encode new memory in current state."""
        memory = Memory(
            content=content,
            encoding_quality=quality,
            encoded_state=state,
        )
        self.memories.append(memory)
        return memory

    def consolidate(
        self,
        budget: EmotionalMetabolicBudget,
    ) -> List[Memory]:
        """Consolidate unconsolidated memories based on current state."""
        state = budget.metabolic_state
        multiplier = self.STATE_MULTIPLIERS[state]

        consolidated = []

        for memory in self.memories:
            if memory.consolidated:
                continue

            # Calculate consolidation quality
            quality = min(1.0, memory.encoding_quality * multiplier)

            if multiplier > 0:  # Can consolidate in this state
                memory.consolidated = True
                memory.consolidation_quality = quality
                memory.consolidation_state = state
                consolidated.append(memory)

        return consolidated


@dataclass
class IntegrationTurn:
    """Turn in integrated framework test."""
    turn_id: int
    description: str
    context: str
    emotional_trigger: Optional[str]
    emotional_delta: Dict[str, float]
    memory_content: Optional[str] = None
    memory_quality: float = 0.7


def create_integration_scenario() -> List[IntegrationTurn]:
    """
    Create complex scenario testing full framework integration.

    15-turn scenario exercising:
    - Discovery phase (high engagement)
    - Frustration accumulation (repeated failures)
    - Proactive intervention (if enabled)
    - Recovery phase (REST → WAKE)
    - Re-engagement (WAKE → FOCUS)
    - Memory consolidation across states
    """

    turns = [
        # Phase 1: Discovery (Turns 1-3) - WAKE → FOCUS
        IntegrationTurn(
            1, "New research topic", "intro",
            emotional_trigger="discovery",
            emotional_delta={'curiosity': 0.3, 'engagement': 0.2},
            memory_content="Research question identified",
            memory_quality=0.7,
        ),

        IntegrationTurn(
            2, "Fascinating patterns emerging", "excitement",
            emotional_trigger="discovery",
            emotional_delta={'curiosity': 0.3, 'engagement': 0.3},
            memory_content="Pattern recognition insight",
            memory_quality=0.8,
        ),

        IntegrationTurn(
            3, "Deep focus mode", "peak_engagement",
            emotional_trigger="discovery",
            emotional_delta={'engagement': 0.2, 'progress': 0.2},
            memory_content="Deep understanding achieved",
            memory_quality=0.9,
        ),

        # Phase 2: Frustration Accumulation (Turns 4-7) - Testing regulation
        IntegrationTurn(
            4, "First experimental failure", "failure1",
            emotional_trigger="failure",
            emotional_delta={'frustration': 0.25, 'engagement': -0.1},
            memory_content="Failure pattern A",
            memory_quality=0.5,
        ),

        IntegrationTurn(
            5, "Second experimental failure", "failure2",
            emotional_trigger="failure",
            emotional_delta={'frustration': 0.25, 'engagement': -0.1},
            memory_content="Failure pattern B",
            memory_quality=0.5,
        ),

        IntegrationTurn(
            6, "Third experimental failure", "failure3",
            emotional_trigger="failure",
            emotional_delta={'frustration': 0.25},
            memory_content="Failure pattern C",
            memory_quality=0.4,
        ),

        IntegrationTurn(
            7, "Fourth experimental failure", "failure4",
            emotional_trigger="failure",
            emotional_delta={'frustration': 0.25},
            memory_content="Failure pattern D",
            memory_quality=0.4,
        ),

        # Phase 3: Potential REST (Turn 8) - Will regulation prevent?
        IntegrationTurn(
            8, "Consolidating what I learned", "reflection",
            emotional_trigger=None,
            emotional_delta={},  # Let passive decay work
            memory_content=None,
        ),

        # Phase 4: Recovery (Turns 9-10) - REST → WAKE
        IntegrationTurn(
            9, "Resting and processing", "rest",
            emotional_trigger="recovery",
            emotional_delta={'frustration': -0.2, 'engagement': -0.1},
            memory_content=None,
        ),

        IntegrationTurn(
            10, "Ready to try again", "recovery",
            emotional_trigger="recovery",
            emotional_delta={'frustration': -0.2, 'engagement': 0.1},
            memory_content="Recovery insight",
            memory_quality=0.6,
        ),

        # Phase 5: Re-engagement (Turns 11-13) - WAKE → FOCUS
        IntegrationTurn(
            11, "New approach discovered", "breakthrough",
            emotional_trigger="discovery",
            emotional_delta={'curiosity': 0.3, 'engagement': 0.2, 'progress': 0.2},
            memory_content="Breakthrough insight",
            memory_quality=0.9,
        ),

        IntegrationTurn(
            12, "Experimental success!", "success",
            emotional_trigger="success",
            emotional_delta={'progress': 0.3, 'engagement': 0.2},
            memory_content="Success pattern",
            memory_quality=1.0,
        ),

        IntegrationTurn(
            13, "Deep validation", "peak_engagement",
            emotional_trigger="discovery",
            emotional_delta={'engagement': 0.1, 'progress': 0.2},
            memory_content="Validation results",
            memory_quality=0.9,
        ),

        # Phase 6: Wind down (Turns 14-15) - Testing DREAM
        IntegrationTurn(
            14, "Reflecting on session", "wind_down",
            emotional_trigger=None,
            emotional_delta={'engagement': -0.3},
            memory_content=None,
        ),

        IntegrationTurn(
            15, "Deep consolidation", "dream",
            emotional_trigger=None,
            emotional_delta={'engagement': -0.2},
            memory_content=None,
        ),
    ]

    return turns


def run_integrated_test(use_regulation: bool) -> Dict:
    """
    Run integrated framework test.

    Args:
        use_regulation: Whether to use proactive emotional regulation

    Returns:
        Test results including state trajectory, consolidation quality, etc.
    """

    condition = "PROACTIVE" if use_regulation else "CONTROL"
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING INTEGRATED TEST: {condition}")
    logger.info(f"{'='*80}\n")

    # Initialize components
    budget = EmotionalMetabolicBudget(
        metabolic_state=MetabolicState.WAKE,
        emotional_state=EmotionalState(
            curiosity=0.5,
            frustration=0.0,
            engagement=0.5,
            progress=0.5,
        ),
    )

    strategy = RegulationStrategy.PROACTIVE if use_regulation else RegulationStrategy.NONE
    regulator = EmotionalRegulator(strategy=strategy)
    consolidator = StateAwareConsolidator()

    # Run scenario
    turns = create_integration_scenario()
    turn_results = []
    state_visits = {state: 0 for state in MetabolicState}
    transitions = []

    for turn in turns:
        logger.info(f"\nTurn {turn.turn_id}: {turn.description}")

        # Record state before
        state_before = budget.metabolic_state
        state_visits[state_before] += 1

        # Apply emotional changes
        if turn.emotional_delta:
            budget.update_emotional_state(
                curiosity_delta=turn.emotional_delta.get('curiosity', 0.0),
                frustration_delta=turn.emotional_delta.get('frustration', 0.0),
                engagement_delta=turn.emotional_delta.get('engagement', 0.0),
                progress_delta=turn.emotional_delta.get('progress', 0.0),
            )

        # Apply regulation (if enabled)
        reg_result = regulator.regulate(budget, context=turn.description)

        # Transition metabolic state
        event = "crisis" if turn.emotional_trigger == "crisis" else "normal"
        budget.transition_metabolic_state(event)

        state_after = budget.metabolic_state

        # Track transition
        if state_before != state_after:
            transition = f"{state_before.value} → {state_after.value}"
            transitions.append(transition)
            logger.info(f"  ✅ TRANSITION: {transition}")

        # Encode memory if provided
        if turn.memory_content:
            consolidator.encode(
                turn.memory_content,
                turn.memory_quality,
                state_after,
            )
            logger.info(f"  📝 ENCODED: {turn.memory_content} (q={turn.memory_quality:.2f})")

        # Consolidate memories
        consolidated = consolidator.consolidate(budget)
        if consolidated:
            for memory in consolidated:
                logger.info(f"  💾 CONSOLIDATED: {memory.content} "
                          f"(q={memory.consolidation_quality:.2f} in {memory.consolidation_state.value})")

        # Recovery
        budget.recover()

        # Record turn
        turn_result = {
            'turn_id': turn.turn_id,
            'state_before': state_before.value,
            'state_after': state_after.value,
            'emotional_state': budget.emotional_state.to_dict(),
            'resources': {
                'compute': budget.resource_budget.compute_atp,
                'memory': budget.resource_budget.memory_atp,
            },
            'regulated': reg_result['regulated'],
            'memories_encoded': 1 if turn.memory_content else 0,
            'memories_consolidated': len(consolidated),
        }
        turn_results.append(turn_result)

        logger.info(f"  State: {state_after.value}, "
                   f"Frustration: {budget.emotional_state.frustration:.2f}, "
                   f"Engagement: {budget.emotional_state.engagement:.2f}")

    # Analysis
    total_memories = len(consolidator.memories)
    consolidated_memories = [m for m in consolidator.memories if m.consolidated]
    avg_consolidation_quality = (
        sum(m.consolidation_quality for m in consolidated_memories) / len(consolidated_memories)
        if consolidated_memories else 0.0
    )

    # State distribution
    total_turns = len(turns)
    state_distribution = {
        state.value: count / total_turns
        for state, count in state_visits.items()
    }

    # Emotional trajectory
    avg_frustration = sum(t['emotional_state']['frustration'] for t in turn_results) / len(turn_results)
    peak_frustration = max(t['emotional_state']['frustration'] for t in turn_results)

    results = {
        'condition': condition,
        'use_regulation': use_regulation,
        'turn_results': turn_results,
        'transitions': transitions,
        'state_distribution': state_distribution,
        'memories': {
            'total_encoded': total_memories,
            'total_consolidated': len(consolidated_memories),
            'avg_quality': avg_consolidation_quality,
            'by_state': {
                state.value: [
                    {
                        'content': m.content,
                        'quality': m.consolidation_quality,
                    }
                    for m in consolidated_memories
                    if m.consolidation_state == state
                ]
                for state in MetabolicState
            },
        },
        'emotions': {
            'avg_frustration': avg_frustration,
            'peak_frustration': peak_frustration,
        },
        'regulation': {
            'interventions': len(regulator.interventions),
        },
    }

    logger.info(f"\n{'='*40}")
    logger.info(f"RESULTS: {condition}")
    logger.info(f"{'='*40}")
    logger.info(f"Transitions: {len(transitions)}")
    logger.info(f"State Distribution: {state_distribution}")
    logger.info(f"Avg Frustration: {avg_frustration:.3f}")
    logger.info(f"Peak Frustration: {peak_frustration:.3f}")
    logger.info(f"Memories: {len(consolidated_memories)}/{total_memories} consolidated")
    logger.info(f"Avg Consolidation Quality: {avg_consolidation_quality:.3f}")
    logger.info(f"Interventions: {len(regulator.interventions)}")

    return results


def run_session_124():
    """Run Session 124 integrated framework testing."""

    logger.info("="*80)
    logger.info("SESSION 124: INTEGRATED FRAMEWORK VALIDATION")
    logger.info("="*80)
    logger.info("Goal: Test proactive regulation within full emotional/metabolic/consolidation framework")
    logger.info("")
    logger.info("Integration:")
    logger.info("  - EmotionalMetabolicBudget (S120)")
    logger.info("  - StateAwareConsolidator (S122)")
    logger.info("  - EmotionalRegulator (S123)")
    logger.info("")
    logger.info("Test: 15-turn complex scenario")
    logger.info("  Phase 1: Discovery (WAKE → FOCUS)")
    logger.info("  Phase 2: Frustration accumulation (4 failures)")
    logger.info("  Phase 3: Recovery (REST → WAKE)")
    logger.info("  Phase 4: Re-engagement (WAKE → FOCUS)")
    logger.info("  Phase 5: Consolidation (DREAM)")
    logger.info("="*80)
    logger.info("\n")

    # Run both conditions
    control_results = run_integrated_test(use_regulation=False)
    proactive_results = run_integrated_test(use_regulation=True)

    # Comparison
    logger.info(f"\n\n{'='*80}")
    logger.info("INTEGRATED FRAMEWORK COMPARISON")
    logger.info(f"{'='*80}\n")

    logger.info("Metric                    | CONTROL      | PROACTIVE    | Improvement")
    logger.info("-"*75)

    # Frustration
    control_frust = control_results['emotions']['avg_frustration']
    proactive_frust = proactive_results['emotions']['avg_frustration']
    frust_improvement = (control_frust - proactive_frust) / control_frust * 100
    logger.info(f"Avg Frustration           | {control_frust:.3f}        | {proactive_frust:.3f}        | {frust_improvement:+.1f}%")

    # Peak frustration
    control_peak = control_results['emotions']['peak_frustration']
    proactive_peak = proactive_results['emotions']['peak_frustration']
    peak_improvement = (control_peak - proactive_peak) / control_peak * 100
    logger.info(f"Peak Frustration          | {control_peak:.3f}        | {proactive_peak:.3f}        | {peak_improvement:+.1f}%")

    # Consolidation quality
    control_quality = control_results['memories']['avg_quality']
    proactive_quality = proactive_results['memories']['avg_quality']
    quality_improvement = (proactive_quality - control_quality) / control_quality * 100
    logger.info(f"Avg Consolidation Quality | {control_quality:.3f}        | {proactive_quality:.3f}        | {quality_improvement:+.1f}%")

    # Transitions
    control_transitions = len(control_results['transitions'])
    proactive_transitions = len(proactive_results['transitions'])
    logger.info(f"State Transitions         | {control_transitions}            | {proactive_transitions}            | {proactive_transitions - control_transitions:+d}")

    # State distribution
    logger.info(f"\nState Distribution:")
    logger.info(f"  WAKE:   Control={control_results['state_distribution']['wake']:.2%}, "
               f"Proactive={proactive_results['state_distribution']['wake']:.2%}")
    logger.info(f"  FOCUS:  Control={control_results['state_distribution']['focus']:.2%}, "
               f"Proactive={proactive_results['state_distribution']['focus']:.2%}")
    logger.info(f"  REST:   Control={control_results['state_distribution']['rest']:.2%}, "
               f"Proactive={proactive_results['state_distribution']['rest']:.2%}")
    logger.info(f"  DREAM:  Control={control_results['state_distribution']['dream']:.2%}, "
               f"Proactive={proactive_results['state_distribution']['dream']:.2%}")
    logger.info(f"  CRISIS: Control={control_results['state_distribution']['crisis']:.2%}, "
               f"Proactive={proactive_results['state_distribution']['crisis']:.2%}")

    # Key findings
    logger.info(f"\nKEY FINDINGS:")
    logger.info(f"1. Proactive regulation reduces frustration by {frust_improvement:.1f}% in full framework")
    logger.info(f"2. Consolidation quality improves by {quality_improvement:.1f}% with regulation")
    logger.info(f"3. State distribution changes: regulation affects metabolic state dynamics")
    logger.info(f"4. Interventions: {proactive_results['regulation']['interventions']} proactive interventions")

    # Save results
    output = {
        'session': 124,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'description': 'Integrated framework validation with proactive regulation',
        'control': control_results,
        'proactive': proactive_results,
        'comparison': {
            'frustration_improvement': frust_improvement,
            'peak_frustration_improvement': peak_improvement,
            'quality_improvement': quality_improvement,
            'transition_delta': proactive_transitions - control_transitions,
        },
    }

    output_file = 'sage/experiments/session124_integrated_framework_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n\nResults saved to: {output_file}")
    logger.info("\n" + "="*80 + "\n")

    return output


if __name__ == '__main__':
    results = run_session_124()
