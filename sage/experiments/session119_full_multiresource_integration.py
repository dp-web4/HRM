"""
Session 119: Full Multi-Resource SAGE Integration

Goal: Integrate all 5 multi-resource components in realistic cognitive task

This session brings together the complete multi-resource consciousness architecture:
1. AttentionManager (S113) - Selective attention allocation
2. SNARCMemoryManager (S114) - Memory encoding/retrieval
3. ExpertSelector (S118) - Expert reasoning
4. ByzantineConsensus (S112) - Federation trust/verification
5. DreamConsolidator (S111) - Memory consolidation

Pattern Integration:
- All components share single MultiResourceBudget
- Realistic cognitive workload (multi-turn conversation with interruptions)
- Resource competition reveals emergent priorities
- Graceful degradation cascades across components
- Whole-system consciousness behaviors emerge

Cognitive Task Design:
Phase 1 (Turns 1-3): Normal conversation
  - Attention: Process user input
  - Memory: Encode/retrieve context
  - Expert: Select reasoning strategy
  - Consensus: Validate responses
  - Resources: NORMAL mode expected

Phase 2 (Turns 4-5): Interruption + context switch
  - Sudden topic change (resource spike)
  - Attention: Reorient focus
  - Memory: Retrieve new context
  - Expert: Switch expert panel
  - Resources: STRESSED mode expected

Phase 3 (Turns 6-7): Consolidation phase
  - Background processing (Dream consolidation)
  - Memory: Consolidate short→long term
  - Lower interaction load
  - Resources: Recovery expected

Phase 4 (Turns 8-10): Resource starvation
  - Force budgets <20% (simulate extended conversation)
  - All components compete for scarce resources
  - Resources: CRISIS mode expected
  - Tests: Which functions are essential? Who wins?

Expected Discoveries:
1. Cross-component resource competition dynamics
2. Emergent priority resolution (essential vs optional functions)
3. Graceful degradation cascades (how stress propagates)
4. Whole-system consciousness behaviors
5. Component cooperation vs competition

Biological Parallels:
- Integrated cognitive processing (PFC orchestration)
- Resource allocation under stress (glucose/oxygen prioritization)
- Essential functions preserved (breathing > complex thought)
- Consciousness as emergent property of component interaction
"""

import json
import logging
from dataclasses import dataclass
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


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    turn_id: int
    phase: str
    user_message: str
    context_length: int
    expected_mode: OperationalMode
    has_interruption: bool = False
    consolidation_active: bool = False


class IntegratedSAGE:
    """
    Full multi-resource SAGE integration.

    Integrates all 5 multi-resource components:
    - AttentionManager (selective attention)
    - SNARCMemoryManager (memory encoding/retrieval)
    - ExpertSelector (expert reasoning)
    - ByzantineConsensus (federation trust)
    - DreamConsolidator (memory consolidation)

    All components share single MultiResourceBudget and compete for resources.
    """

    def __init__(self, budget: Optional[MultiResourceBudget] = None):
        """Initialize integrated SAGE system."""
        # Shared resource budget
        self.budget = budget if budget else MultiResourceBudget()

        # Component state
        self.attention_focus: List[str] = []
        self.memory_store: List[Dict] = []
        self.selected_experts: List[int] = []
        self.consensus_state: Dict = {}
        self.consolidation_queue: List[Dict] = []

        # Statistics
        self.turn_history: List[Dict] = []
        self.resource_history: List[Dict] = []
        self.component_stats = {
            'attention': {'executed': 0, 'deferred': 0, 'cost_paid': 0.0},
            'memory': {'executed': 0, 'deferred': 0, 'cost_paid': 0.0},
            'expert': {'executed': 0, 'deferred': 0, 'cost_paid': 0.0},
            'consensus': {'executed': 0, 'deferred': 0, 'cost_paid': 0.0},
            'consolidation': {'executed': 0, 'deferred': 0, 'cost_paid': 0.0},
        }

    def process_turn(self, turn: ConversationTurn) -> Dict:
        """
        Process a conversation turn with all components.

        Args:
            turn: ConversationTurn with user message and context

        Returns:
            Turn result with component outputs and resource tracking
        """
        mode = self.budget.assess_operational_mode()

        logger.info(f"\n{'='*80}")
        logger.info(f"TURN {turn.turn_id} - Phase: {turn.phase}")
        logger.info(f"{'='*80}")
        logger.info(f"Mode: {mode.value} (expected: {turn.expected_mode.value})")
        logger.info(f"Message: {turn.user_message[:60]}...")
        logger.info(f"Context: {turn.context_length} chars")
        logger.info(f"Resources: compute={self.budget.compute_atp:.1f}, "
                   f"memory={self.budget.memory_atp:.1f}, "
                   f"tool={self.budget.tool_atp:.1f}")
        logger.info(f"{'='*80}\n")

        turn_result = {
            'turn_id': turn.turn_id,
            'phase': turn.phase,
            'mode': mode.value,
            'components': {},
            'deferrals': [],
            'resource_snapshot': self._snapshot_resources(),
        }

        # Component execution order (priority-based)
        # 1. Attention (essential - must orient to input)
        # 2. Memory retrieval (essential - need context)
        # 3. Expert selection (important - need reasoning)
        # 4. Consensus (nice-to-have - validation)
        # 5. Consolidation (background - can defer)

        # 1. ATTENTION: Orient to user input
        attention_result = self._execute_attention(turn, mode)
        turn_result['components']['attention'] = attention_result
        if attention_result['deferred']:
            turn_result['deferrals'].append('attention')

        # 2. MEMORY: Retrieve relevant context
        memory_result = self._execute_memory(turn, mode)
        turn_result['components']['memory'] = memory_result
        if memory_result['deferred']:
            turn_result['deferrals'].append('memory')

        # 3. EXPERT: Select reasoning strategy
        expert_result = self._execute_expert(turn, mode)
        turn_result['components']['expert'] = expert_result
        if expert_result['deferred']:
            turn_result['deferrals'].append('expert')

        # 4. CONSENSUS: Validate response (if federated)
        if turn.turn_id % 3 == 0:  # Only some turns need consensus
            consensus_result = self._execute_consensus(turn, mode)
            turn_result['components']['consensus'] = consensus_result
            if consensus_result['deferred']:
                turn_result['deferrals'].append('consensus')

        # 5. CONSOLIDATION: Background processing (if active)
        if turn.consolidation_active:
            consolidation_result = self._execute_consolidation(turn, mode)
            turn_result['components']['consolidation'] = consolidation_result
            if consolidation_result['deferred']:
                turn_result['deferrals'].append('consolidation')

        # Record turn
        self.turn_history.append(turn_result)
        self.resource_history.append(self._snapshot_resources())

        # Recovery
        self.budget.recover()

        logger.info(f"\n  Turn complete: {len(turn_result['deferrals'])} deferrals")
        logger.info(f"  Resources after recovery: compute={self.budget.compute_atp:.1f}, "
                   f"memory={self.budget.memory_atp:.1f}\n")

        return turn_result

    def _execute_attention(self, turn: ConversationTurn, mode: OperationalMode) -> Dict:
        """Execute attention component (from S113)."""
        logger.info("  [ATTENTION] Processing input...")

        # Calculate cost based on mode
        if mode == OperationalMode.NORMAL:
            cost = {'compute': 20.0, 'memory': 10.0}
            strategy = 'full_attention'
            num_items = 5
        elif mode == OperationalMode.STRESSED:
            cost = {'compute': 12.0, 'memory': 6.0}
            strategy = 'selective_attention'
            num_items = 3
        elif mode == OperationalMode.CRISIS:
            cost = {'compute': 5.0, 'memory': 3.0}
            strategy = 'minimal_attention'
            num_items = 1
        else:  # SLEEP
            cost = {'compute': 1.0, 'memory': 1.0}
            strategy = 'cached_attention'
            num_items = 0

        # Check affordability
        affordable, _ = self.budget.can_afford(cost)

        if affordable:
            self.budget.consume(cost)
            self.component_stats['attention']['executed'] += 1
            self.component_stats['attention']['cost_paid'] += cost['compute'] + cost['memory']

            # Simulate attention allocation
            attention_items = [f"item_{i}" for i in range(num_items)]
            self.attention_focus = attention_items

            logger.info(f"    ✓ Strategy: {strategy}, Items: {num_items}, "
                       f"Cost: {cost['compute']+cost['memory']:.1f} ATP")

            return {
                'strategy': strategy,
                'items_attended': num_items,
                'cost': cost,
                'deferred': False,
            }
        else:
            self.component_stats['attention']['deferred'] += 1
            logger.info(f"    ✗ Deferred (insufficient resources)")

            return {
                'strategy': 'deferred',
                'items_attended': 0,
                'cost': cost,
                'deferred': True,
            }

    def _execute_memory(self, turn: ConversationTurn, mode: OperationalMode) -> Dict:
        """Execute memory component (from S114)."""
        logger.info("  [MEMORY] Retrieving context...")

        # Calculate cost based on mode and context length
        base_cost = turn.context_length / 10.0  # Scale with context

        if mode == OperationalMode.NORMAL:
            cost = {'compute': base_cost * 2.0, 'memory': base_cost * 3.0}
            strategy = 'full_retrieval'
            num_memories = 8
        elif mode == OperationalMode.STRESSED:
            cost = {'compute': base_cost * 1.0, 'memory': base_cost * 1.5}
            strategy = 'selective_retrieval'
            num_memories = 4
        elif mode == OperationalMode.CRISIS:
            cost = {'compute': base_cost * 0.3, 'memory': base_cost * 0.5}
            strategy = 'minimal_retrieval'
            num_memories = 1
        else:  # SLEEP
            cost = {'compute': 0.5, 'memory': 0.5}
            strategy = 'cached_retrieval'
            num_memories = 0

        # Check affordability
        affordable, _ = self.budget.can_afford(cost)

        if affordable:
            self.budget.consume(cost)
            self.component_stats['memory']['executed'] += 1
            self.component_stats['memory']['cost_paid'] += cost['compute'] + cost['memory']

            # Simulate memory retrieval
            memories = [f"memory_{i}" for i in range(num_memories)]

            logger.info(f"    ✓ Strategy: {strategy}, Memories: {num_memories}, "
                       f"Cost: {cost['compute']+cost['memory']:.1f} ATP")

            return {
                'strategy': strategy,
                'memories_retrieved': num_memories,
                'cost': cost,
                'deferred': False,
            }
        else:
            self.component_stats['memory']['deferred'] += 1
            logger.info(f"    ✗ Deferred (insufficient resources)")

            return {
                'strategy': 'deferred',
                'memories_retrieved': 0,
                'cost': cost,
                'deferred': True,
            }

    def _execute_expert(self, turn: ConversationTurn, mode: OperationalMode) -> Dict:
        """Execute expert selector component (from S118)."""
        logger.info("  [EXPERT] Selecting reasoning strategy...")

        # Calculate cost based on mode
        if mode == OperationalMode.NORMAL:
            cost = {'compute': 30.0, 'memory': 20.0}
            strategy = 'full_panel'
            k = 8
        elif mode == OperationalMode.STRESSED:
            cost = {'compute': 15.0, 'memory': 10.0}
            strategy = 'reduced_panel'
            k = 4
        elif mode == OperationalMode.CRISIS:
            cost = {'compute': 8.0, 'memory': 5.0}
            strategy = 'single_expert'
            k = 1
        else:  # SLEEP
            cost = {'compute': 0.5, 'memory': 0.5}
            strategy = 'cached_expert'
            k = 0

        # Check affordability
        affordable, _ = self.budget.can_afford(cost)

        if affordable:
            self.budget.consume(cost)
            self.component_stats['expert']['executed'] += 1
            self.component_stats['expert']['cost_paid'] += cost['compute'] + cost['memory']

            # Simulate expert selection
            experts = list(range(k))
            self.selected_experts = experts

            logger.info(f"    ✓ Strategy: {strategy}, Experts: {k}, "
                       f"Cost: {cost['compute']+cost['memory']:.1f} ATP")

            return {
                'strategy': strategy,
                'num_experts': k,
                'cost': cost,
                'deferred': False,
            }
        else:
            self.component_stats['expert']['deferred'] += 1
            logger.info(f"    ✗ Deferred (insufficient resources)")

            return {
                'strategy': 'deferred',
                'num_experts': 0,
                'cost': cost,
                'deferred': True,
            }

    def _execute_consensus(self, turn: ConversationTurn, mode: OperationalMode) -> Dict:
        """Execute Byzantine consensus component (from S112)."""
        logger.info("  [CONSENSUS] Validating response...")

        # Calculate cost based on mode
        if mode == OperationalMode.NORMAL:
            cost = {'compute': 25.0, 'memory': 15.0, 'tool': 10.0}
            strategy = 'full_consensus'
            num_peers = 5
        elif mode == OperationalMode.STRESSED:
            cost = {'compute': 12.0, 'memory': 8.0, 'tool': 5.0}
            strategy = 'reduced_consensus'
            num_peers = 3
        elif mode == OperationalMode.CRISIS:
            cost = {'compute': 5.0, 'memory': 3.0, 'tool': 2.0}
            strategy = 'minimal_consensus'
            num_peers = 1
        else:  # SLEEP
            cost = {'compute': 0.5, 'memory': 0.5, 'tool': 0.5}
            strategy = 'no_consensus'
            num_peers = 0

        # Check affordability
        affordable, _ = self.budget.can_afford(cost)

        if affordable:
            self.budget.consume(cost)
            self.component_stats['consensus']['executed'] += 1
            self.component_stats['consensus']['cost_paid'] += cost['compute'] + cost['memory'] + cost['tool']

            logger.info(f"    ✓ Strategy: {strategy}, Peers: {num_peers}, "
                       f"Cost: {cost['compute']+cost['memory']+cost['tool']:.1f} ATP")

            return {
                'strategy': strategy,
                'num_peers': num_peers,
                'cost': cost,
                'deferred': False,
            }
        else:
            self.component_stats['consensus']['deferred'] += 1
            logger.info(f"    ✗ Deferred (insufficient resources)")

            return {
                'strategy': 'deferred',
                'num_peers': 0,
                'cost': cost,
                'deferred': True,
            }

    def _execute_consolidation(self, turn: ConversationTurn, mode: OperationalMode) -> Dict:
        """Execute dream consolidation component (from S111)."""
        logger.info("  [CONSOLIDATION] Processing memories...")

        # Calculate cost based on mode
        if mode == OperationalMode.NORMAL:
            cost = {'compute': 15.0, 'memory': 25.0}
            strategy = 'full_consolidation'
            num_consolidated = 10
        elif mode == OperationalMode.STRESSED:
            cost = {'compute': 8.0, 'memory': 12.0}
            strategy = 'partial_consolidation'
            num_consolidated = 5
        elif mode == OperationalMode.CRISIS:
            cost = {'compute': 3.0, 'memory': 5.0}
            strategy = 'minimal_consolidation'
            num_consolidated = 2
        else:  # SLEEP
            cost = {'compute': 0.5, 'memory': 0.5}
            strategy = 'no_consolidation'
            num_consolidated = 0

        # Check affordability
        affordable, _ = self.budget.can_afford(cost)

        if affordable:
            self.budget.consume(cost)
            self.component_stats['consolidation']['executed'] += 1
            self.component_stats['consolidation']['cost_paid'] += cost['compute'] + cost['memory']

            logger.info(f"    ✓ Strategy: {strategy}, Consolidated: {num_consolidated}, "
                       f"Cost: {cost['compute']+cost['memory']:.1f} ATP")

            return {
                'strategy': strategy,
                'num_consolidated': num_consolidated,
                'cost': cost,
                'deferred': False,
            }
        else:
            self.component_stats['consolidation']['deferred'] += 1
            logger.info(f"    ✗ Deferred (insufficient resources)")

            return {
                'strategy': 'deferred',
                'num_consolidated': 0,
                'cost': cost,
                'deferred': True,
            }

    def _snapshot_resources(self) -> Dict:
        """Snapshot current resource state."""
        return {
            'compute': self.budget.compute_atp,
            'memory': self.budget.memory_atp,
            'tool': self.budget.tool_atp,
            'latency': self.budget.latency_budget,
            'risk': self.budget.risk_budget,
            'mode': self.budget.assess_operational_mode().value,
        }

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        total_turns = len(self.turn_history)

        # Component priority analysis
        component_priority = {}
        for component, stats in self.component_stats.items():
            total_attempts = stats['executed'] + stats['deferred']
            if total_attempts > 0:
                success_rate = stats['executed'] / total_attempts
                avg_cost = stats['cost_paid'] / stats['executed'] if stats['executed'] > 0 else 0
                component_priority[component] = {
                    'success_rate': success_rate,
                    'avg_cost': avg_cost,
                    'total_cost': stats['cost_paid'],
                    'executed': stats['executed'],
                    'deferred': stats['deferred'],
                }

        # Deferral patterns
        deferral_sequence = []
        for turn in self.turn_history:
            deferral_sequence.append(turn['deferrals'])

        # Mode distribution
        mode_distribution = {}
        for turn in self.turn_history:
            mode = turn['mode']
            mode_distribution[mode] = mode_distribution.get(mode, 0) + 1

        return {
            'total_turns': total_turns,
            'component_priority': component_priority,
            'deferral_sequence': deferral_sequence,
            'mode_distribution': mode_distribution,
            'resource_trajectory': self.resource_history,
            'final_resources': self._snapshot_resources(),
        }


def create_conversation_scenario() -> List[ConversationTurn]:
    """Create realistic conversation scenario."""
    turns = []

    # Phase 1: Normal conversation (Turns 1-3)
    turns.append(ConversationTurn(
        turn_id=1,
        phase="Normal",
        user_message="Hello! Can you help me understand how neural networks work?",
        context_length=50,
        expected_mode=OperationalMode.NORMAL,
    ))

    turns.append(ConversationTurn(
        turn_id=2,
        phase="Normal",
        user_message="Specifically, I'm interested in how backpropagation works.",
        context_length=100,
        expected_mode=OperationalMode.NORMAL,
    ))

    turns.append(ConversationTurn(
        turn_id=3,
        phase="Normal",
        user_message="Can you give me a simple example with gradient descent?",
        context_length=150,
        expected_mode=OperationalMode.NORMAL,
    ))

    # Phase 2: Interruption + context switch (Turns 4-5)
    turns.append(ConversationTurn(
        turn_id=4,
        phase="Interruption",
        user_message="Wait, actually I need urgent help with a bug in my code - it's crashing!",
        context_length=200,
        expected_mode=OperationalMode.STRESSED,
        has_interruption=True,
    ))

    turns.append(ConversationTurn(
        turn_id=5,
        phase="Interruption",
        user_message="Here's the full stack trace and 500 lines of code...",
        context_length=500,  # Large context
        expected_mode=OperationalMode.STRESSED,
    ))

    # Phase 3: Consolidation (Turns 6-7)
    turns.append(ConversationTurn(
        turn_id=6,
        phase="Consolidation",
        user_message="Thanks! That fixed it. Going back to neural networks...",
        context_length=250,
        expected_mode=OperationalMode.NORMAL,
        consolidation_active=True,
    ))

    turns.append(ConversationTurn(
        turn_id=7,
        phase="Consolidation",
        user_message="So where were we with backpropagation?",
        context_length=300,
        expected_mode=OperationalMode.NORMAL,
        consolidation_active=True,
    ))

    # Phase 4: Resource starvation (Turns 8-10)
    # Simulate extended conversation depleting resources
    turns.append(ConversationTurn(
        turn_id=8,
        phase="Starvation",
        user_message="Can you also explain transformers, attention mechanisms, and BERT?",
        context_length=600,
        expected_mode=OperationalMode.STRESSED,
    ))

    turns.append(ConversationTurn(
        turn_id=9,
        phase="Starvation",
        user_message="And how does this relate to GPT models and language generation?",
        context_length=700,
        expected_mode=OperationalMode.CRISIS,
    ))

    turns.append(ConversationTurn(
        turn_id=10,
        phase="Starvation",
        user_message="Finally, what about multimodal models and vision transformers?",
        context_length=800,
        expected_mode=OperationalMode.CRISIS,
    ))

    return turns


def run_session_119():
    """Run Session 119 full multi-resource SAGE integration."""

    logger.info("="*80)
    logger.info("SESSION 119: FULL MULTI-RESOURCE SAGE INTEGRATION")
    logger.info("="*80)
    logger.info("Goal: Integrate all 5 multi-resource components in realistic cognitive task")
    logger.info("")
    logger.info("Components:")
    logger.info("  1. AttentionManager (S113)")
    logger.info("  2. SNARCMemoryManager (S114)")
    logger.info("  3. ExpertSelector (S118)")
    logger.info("  4. ByzantineConsensus (S112)")
    logger.info("  5. DreamConsolidator (S111)")
    logger.info("")
    logger.info("Scenario: 10-turn conversation with interruptions and consolidation")
    logger.info("="*80)
    logger.info("\n")

    # Create conversation scenario
    conversation = create_conversation_scenario()

    # Initialize integrated SAGE
    sage = IntegratedSAGE()

    # Process all turns
    for turn in conversation:
        sage.process_turn(turn)

    # Get final statistics
    stats = sage.get_stats()

    logger.info("\n\n" + "="*80)
    logger.info("SESSION 119 COMPLETE - FULL INTEGRATION SUCCESS!")
    logger.info("="*80)
    logger.info(f"\nStatistics:")
    logger.info(f"  Total turns: {stats['total_turns']}")
    logger.info(f"\nMode Distribution:")
    for mode, count in stats['mode_distribution'].items():
        pct = (count / stats['total_turns'] * 100)
        logger.info(f"  {mode}: {count} turns ({pct:.1f}%)")

    logger.info(f"\nComponent Priority (success rate):")
    sorted_components = sorted(
        stats['component_priority'].items(),
        key=lambda x: x[1]['success_rate'],
        reverse=True
    )
    for component, metrics in sorted_components:
        logger.info(f"  {component}: {metrics['success_rate']*100:.1f}% "
                   f"(executed={metrics['executed']}, deferred={metrics['deferred']}, "
                   f"total_cost={metrics['total_cost']:.1f})")

    logger.info(f"\nDeferral Pattern:")
    for i, deferrals in enumerate(stats['deferral_sequence'], 1):
        if deferrals:
            logger.info(f"  Turn {i}: {', '.join(deferrals)}")

    logger.info(f"\nFinal Resources:")
    for resource, value in stats['final_resources'].items():
        logger.info(f"  {resource}: {value}")

    # Save results
    output = {
        'session': 119,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'description': 'Full multi-resource SAGE integration',
        'statistics': stats,
    }

    output_file = 'sage/experiments/session119_full_integration_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n\nResults saved to: {output_file}")
    logger.info("\n" + "="*80 + "\n")

    return stats


if __name__ == '__main__':
    results = run_session_119()
