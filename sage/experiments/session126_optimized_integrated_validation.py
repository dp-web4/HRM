"""
Session 126: Optimized Integrated Framework Validation

Goal: Test Session 125's optimal parameters in Session 124's full framework

Session 124 tested proactive regulation (0.20, -0.20) in full integrated framework.
Session 125 discovered optimal parameters (0.10, -0.30) are 73.5% better in isolation.
This session validates whether optimization gains transfer to full system.

Research Questions:
1. Do optimized parameters (0.10, -0.30) improve full framework performance?
2. Does "early and strong" prevent REST states better than baseline?
3. What are metabolic state distribution changes with optimization?
4. Does consolidation quality improve with optimized regulation?
5. Are there unexpected emergent effects from parameter change?

Hypothesis: Optimization should transfer
- Session 125: 73.5% better in 10-turn scenario
- Session 124: Full 15-turn framework integration
- Expectation: Even better emergent effects with optimal params

Test Design:
- Rerun Session 124's scenario (15-turn complex)
- Compare 3 conditions:
  * CONTROL: No regulation (S124 baseline)
  * BASELINE: S123 params (0.20, -0.20) from S124
  * OPTIMIZED: S125 params (0.10, -0.30) - new test
- Measure: state distribution, frustration, consolidation, transitions

Expected Discoveries:
1. Confirmation: Optimization transfers to full framework
2. Emergent effects: Better state stability with optimal params
3. Trade-offs: Intervention count vs state quality
4. Production readiness: Final parameter recommendations
5. Validation: Framework complete and ready for deployment

Biological Parallel:
Like testing a pharmaceutical in full organism vs isolated cells.
Session 125 = in vitro, Session 126 = in vivo validation.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timezone
import sys
import os

# Add sage to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from sage.experiments.session120_emotional_metabolic_states import (
    EmotionalMetabolicBudget,
    EmotionalState,
    MetabolicState,
)

from sage.experiments.session125_adaptive_regulation_thresholds import (
    RegulationParameters,
    AdaptiveRegulator,
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
    Create Session 124's 15-turn scenario for parameter comparison.

    Same scenario for fair comparison.
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


def run_optimized_test(params: Optional[RegulationParameters]) -> Dict:
    """
    Run integrated framework test with specified regulation parameters.

    Args:
        params: Regulation parameters (None for control, no regulation)

    Returns:
        Test results
    """

    if params:
        condition = f"REGULATED({params.detection_threshold:.2f}, {params.intervention_strength:.2f})"
    else:
        condition = "CONTROL"

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

    regulator = AdaptiveRegulator(params) if params else None
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
        reg_result = {'regulated': False}
        if regulator:
            reg_result = regulator.regulate(budget)

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
            logger.debug(f"  📝 ENCODED: {turn.memory_content}")

        # Consolidate memories
        consolidated = consolidator.consolidate(budget)
        if consolidated:
            for memory in consolidated:
                logger.debug(f"  💾 CONSOLIDATED: {memory.content} "
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
                   f"Frust: {budget.emotional_state.frustration:.2f}, "
                   f"Engage: {budget.emotional_state.engagement:.2f}")

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

    # Intervention count
    intervention_count = len(regulator.interventions) if regulator else 0

    results = {
        'condition': condition,
        'params': {
            'threshold': params.detection_threshold if params else None,
            'strength': params.intervention_strength if params else None,
        },
        'turn_results': turn_results,
        'transitions': transitions,
        'state_distribution': state_distribution,
        'memories': {
            'total_encoded': total_memories,
            'total_consolidated': len(consolidated_memories),
            'avg_quality': avg_consolidation_quality,
        },
        'emotions': {
            'avg_frustration': avg_frustration,
            'peak_frustration': peak_frustration,
        },
        'regulation': {
            'interventions': intervention_count,
        },
    }

    logger.info(f"\n{'='*40}")
    logger.info(f"RESULTS: {condition}")
    logger.info(f"{'='*40}")
    logger.info(f"Transitions: {len(transitions)}")
    logger.info(f"State Distribution: WAKE={state_distribution['wake']:.1%}, "
               f"FOCUS={state_distribution['focus']:.1%}, "
               f"REST={state_distribution['rest']:.1%}")
    logger.info(f"Avg Frustration: {avg_frustration:.3f}")
    logger.info(f"Peak Frustration: {peak_frustration:.3f}")
    logger.info(f"Consolidation Quality: {avg_consolidation_quality:.3f}")
    logger.info(f"Interventions: {intervention_count}")

    return results


def run_session_126():
    """Run Session 126 optimized integrated framework validation."""

    logger.info("="*80)
    logger.info("SESSION 126: OPTIMIZED INTEGRATED FRAMEWORK VALIDATION")
    logger.info("="*80)
    logger.info("Goal: Test Session 125's optimal parameters in Session 124's full framework")
    logger.info("")
    logger.info("Conditions:")
    logger.info("  1. CONTROL: No regulation")
    logger.info("  2. BASELINE: S123 params (0.20, -0.20)")
    logger.info("  3. OPTIMIZED: S125 params (0.10, -0.30)")
    logger.info("="*80)
    logger.info("\n")

    # Run all conditions
    control_results = run_optimized_test(None)

    baseline_params = RegulationParameters(
        detection_threshold=0.20,
        intervention_strength=-0.20,
    )
    baseline_results = run_optimized_test(baseline_params)

    optimized_params = RegulationParameters(
        detection_threshold=0.10,
        intervention_strength=-0.30,
    )
    optimized_results = run_optimized_test(optimized_params)

    # Three-way comparison
    logger.info(f"\n\n{'='*80}")
    logger.info("THREE-WAY COMPARISON: CONTROL vs BASELINE vs OPTIMIZED")
    logger.info(f"{'='*80}\n")

    logger.info("Metric                    | CONTROL   | BASELINE  | OPTIMIZED | Improvement")
    logger.info("-"*85)

    # Frustration comparison
    control_frust = control_results['emotions']['avg_frustration']
    baseline_frust = baseline_results['emotions']['avg_frustration']
    optimized_frust = optimized_results['emotions']['avg_frustration']

    baseline_improvement = (control_frust - baseline_frust) / control_frust * 100
    optimized_improvement = (control_frust - optimized_frust) / control_frust * 100

    logger.info(f"Avg Frustration           | {control_frust:.3f}     | {baseline_frust:.3f}     | {optimized_frust:.3f}     | {optimized_improvement:+.1f}%")

    # State distribution
    logger.info(f"\nState Distribution:")
    logger.info(f"  WAKE:   Control={control_results['state_distribution']['wake']:.1%}, "
               f"Baseline={baseline_results['state_distribution']['wake']:.1%}, "
               f"Optimized={optimized_results['state_distribution']['wake']:.1%}")
    logger.info(f"  FOCUS:  Control={control_results['state_distribution']['focus']:.1%}, "
               f"Baseline={baseline_results['state_distribution']['focus']:.1%}, "
               f"Optimized={optimized_results['state_distribution']['focus']:.1%}")
    logger.info(f"  REST:   Control={control_results['state_distribution']['rest']:.1%}, "
               f"Baseline={baseline_results['state_distribution']['rest']:.1%}, "
               f"Optimized={optimized_results['state_distribution']['rest']:.1%}")

    # Interventions
    logger.info(f"\nInterventions:")
    logger.info(f"  Control:  {control_results['regulation']['interventions']}")
    logger.info(f"  Baseline: {baseline_results['regulation']['interventions']}")
    logger.info(f"  Optimized: {optimized_results['regulation']['interventions']}")

    # Save results
    output = {
        'session': 126,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'description': 'Optimized integrated framework validation',
        'control': control_results,
        'baseline_s123': baseline_results,
        'optimized_s125': optimized_results,
        'comparison': {
            'baseline_improvement': baseline_improvement,
            'optimized_improvement': optimized_improvement,
            'optimized_vs_baseline': (baseline_frust - optimized_frust) / baseline_frust * 100,
        },
    }

    output_file = 'sage/experiments/session126_optimized_integrated_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n\nResults saved to: {output_file}")
    logger.info("\n" + "="*80 + "\n")

    return output


if __name__ == '__main__':
    results = run_session_126()
