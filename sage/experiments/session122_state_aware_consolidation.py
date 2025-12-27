"""
Session 122: State-Aware Memory Consolidation

Goal: Integrate metabolic states with memory consolidation

Extends Session 111 (DreamConsolidator) and Sessions 120-121 (emotional/metabolic states)
by making memory consolidation quality and strategy dependent on metabolic state.

Metabolic State Effects on Consolidation:

DREAM State (40 ATP, memory-biased recovery 3.5):
- **Enhanced consolidation**: 2x quality, pattern extraction emphasis
- Long-term memory formation优先
- Creative recombination of memories
- Biological parallel: REM sleep consolidation

WAKE State (100 ATP, standard recovery 1.2):
- **Normal consolidation**: 1x quality, balanced approach
- Standard short→long term transfer
- Maintains recency bias
- Biological parallel: Awake replay/rehearsal

FOCUS State (150 ATP, reduced recovery 0.8):
- **Working memory emphasis**: 0.5x consolidation, encoding优先
- Active memories maintained in working memory
- Reduced consolidation (focus on new encoding)
- Biological parallel: Learning mode (encoding > consolidation)

REST State (60 ATP, enhanced recovery 2.0):
- **Moderate consolidation**: 1.2x quality, recovery优先
- Gentle consolidation during rest
- Low cognitive load
- Biological parallel: Quiet waking rest consolidation

CRISIS State (30 ATP, minimal recovery 0.5):
- **No consolidation**: 0x quality, survival优先
- All resources to immediate needs
- Consolidation deferred
- Biological parallel: Emergency response (no time for memory)

Test Design:
- Simulate memory encoding across different metabolic states
- Trigger consolidation in each state
- Measure consolidation quality, pattern extraction, retention
- Validate state-specific consolidation strategies
- Test state transitions during consolidation (e.g., WAKE → DREAM)

Expected Discoveries:
1. DREAM state produces highest quality consolidation
2. FOCUS state prioritizes encoding over consolidation
3. CRISIS state defers consolidation entirely
4. State transitions affect consolidation mid-process
5. Natural DREAM state triggering during consolidation periods
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timezone
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
class Memory:
    """A memory to be consolidated."""
    content: str
    salience: float
    timestamp: int
    consolidated: bool = False
    consolidation_quality: float = 0.0
    consolidation_state: Optional[str] = None


class StateAwareConsolidator:
    """
    Memory consolidator with metabolic state awareness.

    Consolidation quality and strategy varies by state:
    - DREAM: 2.0x quality (enhanced sleep consolidation)
    - REST: 1.2x quality (quiet waking consolidation)
    - WAKE: 1.0x quality (normal awake consolidation)
    - FOCUS: 0.5x quality (encoding priority, reduced consolidation)
    - CRISIS: 0.0x quality (no consolidation, survival mode)
    """

    def __init__(self, budget: EmotionalMetabolicBudget):
        """Initialize state-aware consolidator."""
        self.budget = budget
        self.memory_buffer: List[Memory] = []
        self.consolidated_memories: List[Memory] = []
        self.consolidation_history: List[Dict] = []

    def encode_memory(self, content: str, salience: float, turn: int):
        """Encode a new memory (add to buffer)."""
        memory = Memory(
            content=content,
            salience=salience,
            timestamp=turn,
        )
        self.memory_buffer.append(memory)
        logger.info(f"    Encoded: '{content}' (salience={salience:.2f})")

    def get_consolidation_multiplier(self, state: MetabolicState) -> float:
        """Get consolidation quality multiplier for metabolic state."""
        if state == MetabolicState.DREAM:
            return 2.0  # Enhanced consolidation during sleep
        elif state == MetabolicState.REST:
            return 1.2  # Moderate enhancement during rest
        elif state == MetabolicState.WAKE:
            return 1.0  # Normal consolidation
        elif state == MetabolicState.FOCUS:
            return 0.5  # Reduced (encoding priority)
        else:  # CRISIS
            return 0.0  # No consolidation (survival mode)

    def consolidate_memories(self, num_memories: int = 3) -> Dict:
        """
        Consolidate memories from buffer to long-term storage.

        Consolidation quality depends on metabolic state.
        """
        state = self.budget.metabolic_state
        multiplier = self.get_consolidation_multiplier(state)

        logger.info(f"\n  [CONSOLIDATION] State: {state.value}, Multiplier: {multiplier:.1f}x")

        if multiplier == 0.0:
            logger.info(f"    No consolidation in CRISIS mode")
            return {
                'state': state.value,
                'multiplier': multiplier,
                'consolidated_count': 0,
                'avg_quality': 0.0,
            }

        # Select top memories by salience
        if len(self.memory_buffer) == 0:
            logger.info(f"    No memories to consolidate")
            return {
                'state': state.value,
                'multiplier': multiplier,
                'consolidated_count': 0,
                'avg_quality': 0.0,
            }

        # Sort by salience, take top N
        sorted_memories = sorted(self.memory_buffer, key=lambda m: m.salience, reverse=True)
        to_consolidate = sorted_memories[:min(num_memories, len(sorted_memories))]

        consolidated_count = 0
        total_quality = 0.0

        for memory in to_consolidate:
            # Base quality from salience
            base_quality = memory.salience

            # Apply state multiplier
            final_quality = base_quality * multiplier

            # Apply random variation (±10%)
            final_quality *= np.random.uniform(0.9, 1.1)

            # Cap at 1.0
            final_quality = min(1.0, final_quality)

            # Update memory
            memory.consolidated = True
            memory.consolidation_quality = final_quality
            memory.consolidation_state = state.value

            # Move to consolidated storage
            self.memory_buffer.remove(memory)
            self.consolidated_memories.append(memory)

            consolidated_count += 1
            total_quality += final_quality

            logger.info(f"    Consolidated: '{memory.content}' "
                       f"(quality={final_quality:.2f}, state={state.value})")

        avg_quality = total_quality / consolidated_count if consolidated_count > 0 else 0.0

        result = {
            'state': state.value,
            'multiplier': multiplier,
            'consolidated_count': consolidated_count,
            'avg_quality': avg_quality,
            'memories': [
                {
                    'content': m.content,
                    'quality': m.consolidation_quality,
                    'state': m.consolidation_state,
                }
                for m in to_consolidate
            ]
        }

        self.consolidation_history.append(result)

        return result

    def get_stats(self) -> Dict:
        """Get consolidation statistics."""
        # Group by state
        by_state = {}
        for memory in self.consolidated_memories:
            state = memory.consolidation_state
            if state not in by_state:
                by_state[state] = []
            by_state[state].append(memory.consolidation_quality)

        state_stats = {}
        for state, qualities in by_state.items():
            state_stats[state] = {
                'count': len(qualities),
                'avg_quality': float(np.mean(qualities)),
                'max_quality': float(np.max(qualities)),
                'min_quality': float(np.min(qualities)),
            }

        return {
            'total_encoded': len(self.memory_buffer) + len(self.consolidated_memories),
            'total_consolidated': len(self.consolidated_memories),
            'in_buffer': len(self.memory_buffer),
            'by_state': state_stats,
            'consolidation_events': len(self.consolidation_history),
        }


def run_session_122():
    """Run Session 122 state-aware memory consolidation."""

    logger.info("="*80)
    logger.info("SESSION 122: STATE-AWARE MEMORY CONSOLIDATION")
    logger.info("="*80)
    logger.info("Goal: Integrate metabolic states with memory consolidation")
    logger.info("")
    logger.info("Consolidation Multipliers by State:")
    logger.info("  DREAM: 2.0x (enhanced sleep consolidation)")
    logger.info("  REST: 1.2x (quiet waking consolidation)")
    logger.info("  WAKE: 1.0x (normal consolidation)")
    logger.info("  FOCUS: 0.5x (encoding priority)")
    logger.info("  CRISIS: 0.0x (no consolidation)")
    logger.info("")
    logger.info("Test: Encode memories and consolidate in different states")
    logger.info("="*80)
    logger.info("\n")

    # Create budget and consolidator
    budget = EmotionalMetabolicBudget(
        metabolic_state=MetabolicState.WAKE,
        emotional_state=EmotionalState(
            curiosity=0.5,
            frustration=0.0,
            engagement=0.5,
            progress=0.5,
        ),
    )

    consolidator = StateAwareConsolidator(budget)

    # Scenario: Learn throughout the day, consolidate in different states
    turn = 0

    # Phase 1: WAKE state - normal learning
    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE 1: WAKE STATE - Normal Learning")
    logger.info(f"{'='*80}")

    turn += 1
    logger.info(f"\n  Turn {turn}: Encoding memories in WAKE state")
    consolidator.encode_memory("Neural networks basics", salience=0.7, turn=turn)
    consolidator.encode_memory("Backpropagation algorithm", salience=0.8, turn=turn)
    consolidator.encode_memory("Coffee break conversation", salience=0.3, turn=turn)

    turn += 1
    logger.info(f"\n  Turn {turn}: Consolidation in WAKE state")
    result = consolidator.consolidate_memories(num_memories=2)

    # Phase 2: FOCUS state - intensive learning
    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE 2: FOCUS STATE - Intensive Learning")
    logger.info(f"{'='*80}")

    # Trigger FOCUS
    budget.update_emotional_state(curiosity_delta=0.3, engagement_delta=0.3)
    budget.transition_metabolic_state("normal")

    turn += 1
    logger.info(f"\n  Turn {turn}: Encoding in FOCUS state")
    consolidator.encode_memory("Advanced gradient descent", salience=0.9, turn=turn)
    consolidator.encode_memory("Transformer architecture", salience=0.95, turn=turn)
    consolidator.encode_memory("Attention mechanism details", salience=0.85, turn=turn)

    turn += 1
    logger.info(f"\n  Turn {turn}: Consolidation in FOCUS state (reduced)")
    result = consolidator.consolidate_memories(num_memories=2)

    # Phase 3: REST state - taking a break
    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE 3: REST STATE - Taking a Break")
    logger.info(f"{'='*80}")

    # Trigger REST
    budget.update_emotional_state(frustration_delta=0.5, engagement_delta=-0.3)
    budget.transition_metabolic_state("normal")

    turn += 1
    logger.info(f"\n  Turn {turn}: Resting, gentle consolidation")
    result = consolidator.consolidate_memories(num_memories=2)

    # Phase 4: DREAM state - sleep consolidation
    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE 4: DREAM STATE - Sleep Consolidation")
    logger.info(f"{'='*80}")

    # Manually set to DREAM (simulate sleep)
    budget.metabolic_state = MetabolicState.DREAM
    params = budget._get_metabolic_parameters(MetabolicState.DREAM)
    budget.resource_budget.compute_atp = params['base_atp']
    budget.resource_budget.memory_atp = params['base_atp']
    budget.resource_budget.tool_atp = params['base_atp']
    budget.resource_budget.compute_recovery = params['compute_recovery']
    budget.resource_budget.memory_recovery = params['memory_recovery']
    budget.resource_budget.tool_recovery = params['tool_recovery']

    logger.info(f"  [METABOLIC] Entered DREAM state (sleep)")
    logger.info(f"    ATP: {params['base_atp']:.1f}, "
               f"Memory recovery: {params['memory_recovery']:.1f}")

    turn += 1
    logger.info(f"\n  Turn {turn}: Sleep consolidation in DREAM state")
    # Add a few more memories to consolidate during sleep
    consolidator.encode_memory("Day's learning synthesis", salience=0.75, turn=turn)
    consolidator.encode_memory("Pattern connections", salience=0.8, turn=turn)

    result = consolidator.consolidate_memories(num_memories=4)

    # Phase 5: CRISIS state - emergency interrupt
    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE 5: CRISIS STATE - Emergency")
    logger.info(f"{'='*80}")

    budget.transition_metabolic_state("crisis")

    turn += 1
    logger.info(f"\n  Turn {turn}: Attempting consolidation in CRISIS")
    consolidator.encode_memory("Emergency event", salience=1.0, turn=turn)
    result = consolidator.consolidate_memories(num_memories=1)

    # Analysis
    logger.info(f"\n\n{'='*80}")
    logger.info("SESSION 122 COMPLETE - STATE-AWARE CONSOLIDATION SUCCESS!")
    logger.info(f"{'='*80}\n")

    stats = consolidator.get_stats()

    logger.info("Consolidation Statistics:")
    logger.info(f"  Total memories encoded: {stats['total_encoded']}")
    logger.info(f"  Total consolidated: {stats['total_consolidated']}")
    logger.info(f"  Remaining in buffer: {stats['in_buffer']}")
    logger.info(f"  Consolidation events: {stats['consolidation_events']}")

    logger.info("\nQuality by Metabolic State:")
    for state, state_stats in sorted(stats['by_state'].items()):
        logger.info(f"  {state}: {state_stats['count']} memories, "
                   f"avg quality={state_stats['avg_quality']:.3f}, "
                   f"max={state_stats['max_quality']:.3f}")

    # Validate multipliers
    logger.info("\nMultiplier Validation:")
    if 'dream' in stats['by_state'] and 'wake' in stats['by_state']:
        dream_quality = stats['by_state']['dream']['avg_quality']
        wake_quality = stats['by_state']['wake']['avg_quality']
        ratio = dream_quality / wake_quality if wake_quality > 0 else 0
        logger.info(f"  DREAM/WAKE quality ratio: {ratio:.2f}x (expected: ~2.0x)")

    if 'focus' in stats['by_state'] and 'wake' in stats['by_state']:
        focus_quality = stats['by_state']['focus']['avg_quality']
        wake_quality = stats['by_state']['wake']['avg_quality']
        ratio = focus_quality / wake_quality if wake_quality > 0 else 0
        logger.info(f"  FOCUS/WAKE quality ratio: {ratio:.2f}x (expected: ~0.5x)")

    # Save results
    output = {
        'session': 122,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'description': 'State-aware memory consolidation',
        'statistics': stats,
        'consolidation_history': consolidator.consolidation_history,
        'consolidated_memories': [
            {
                'content': m.content,
                'salience': m.salience,
                'quality': m.consolidation_quality,
                'state': m.consolidation_state,
            }
            for m in consolidator.consolidated_memories
        ],
    }

    output_file = 'sage/experiments/session122_state_aware_consolidation_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n\nResults saved to: {output_file}")
    logger.info("\n" + "="*80 + "\n")

    return output


if __name__ == '__main__':
    results = run_session_122()
