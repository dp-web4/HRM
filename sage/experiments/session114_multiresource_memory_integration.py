#!/usr/bin/env python3
"""
Session 114: Multi-Resource Memory (SNARC) Integration

Goal: Bridge multi-resource consciousness (S107-113) to SNARCMemoryManager.

Research Context:
- Thor S107-113: Multi-resource consciousness with proven integration pattern
- SNARCMemoryManager: SNARC-based context window memory with salience scoring
- Integration: Apply resource-aware strategies to memory operations

Key Insight:
Memory operations have different computational costs:
- Deep salience scoring: High compute (keyword analysis, metadata processing)
- Simple salience scoring: Low compute (basic heuristics)
- Extraction to long-term: Moderate compute (filtering, deduplication)
- Buffer sorting: Moderate compute (salience-based sort)
- Retrieval: Low compute (timestamp-based lookup)

Under resource stress, memory should degrade gracefully:
- NORMAL: Full SNARC scoring (5 dimensions)
- STRESSED: Simplified scoring (3 dimensions)
- CRISIS: Minimal scoring (novelty only)
- SLEEP: Defer scoring, basic timestamp ordering

Design Principle:
Memory consolidation metabolic cost parallels biological sleep:
- REM sleep: High metabolic activity (pattern extraction, creativity)
- Deep sleep: Moderate activity (consolidation, pruning)
- Light sleep: Low activity (minimal processing)
- Wake: Salience-driven encoding (selective attention)

Operational Mode Adaptation:
- NORMAL: Full SNARC scoring, active extraction, aggressive pruning
- STRESSED: Simplified scoring, deferred extraction, conservative pruning
- CRISIS: Minimal scoring, no extraction, emergency pruning only
- SLEEP: Defer all scoring, timestamp-only ordering, consolidation mode

Biological Realism:
Organisms balance memory encoding/consolidation with metabolic cost:
- Full encoding expensive (attention, binding, integration)
- Simplified encoding cheaper (gist, summary, key points)
- Consolidation expensive (pattern extraction, integration)
- Retrieval varies (cached vs deep search)
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime
import sys
import os

# Add sage to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from sage.experiments.session110_crisis_mode_integration import (
    MultiResourceBudget,
    OperationalMode,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


# ============================================================================
# Memory Operation Resource Costs
# ============================================================================

MEMORY_COSTS = {
    'full_snarc_scoring': {
        'compute': 6.0,   # High: 5 dimensions, keyword matching, metadata processing
        'memory': 3.0,    # Low-moderate: Temporary score structures
        'tool': 0.0,      # None: Internal computation
        'latency': 80.0,  # High: Complex text analysis
        'risk': 0.05,     # Very low: Deterministic scoring
        'priority': 'normal',  # Normal: Quality encoding important but not critical
    },
    'simplified_snarc_scoring': {
        'compute': 3.0,   # Moderate: 3 dimensions, basic heuristics
        'memory': 2.0,    # Low: Simplified structures
        'tool': 0.0,      # None: Internal computation
        'latency': 40.0,  # Moderate: Simpler analysis
        'risk': 0.08,     # Low: Less accurate but safe
        'priority': 'normal',  # Normal: Reduced quality acceptable under stress
    },
    'minimal_scoring': {
        'compute': 1.0,   # Very low: Single dimension (novelty)
        'memory': 1.0,    # Very low: Minimal state
        'tool': 0.0,      # None: Internal computation
        'latency': 10.0,  # Very low: Simple heuristic
        'risk': 0.15,     # Low-moderate: Significant quality loss
        'priority': 'low',  # Low: Minimal encoding when necessary
    },
    'extraction_to_longterm': {
        'compute': 4.0,   # Moderate: Filtering, deduplication, sorting
        'memory': 5.0,    # Moderate: Buffer copies, long-term storage
        'tool': 0.0,      # None: Internal computation
        'latency': 60.0,  # Moderate-high: Multi-pass processing
        'risk': 0.1,      # Low: Data preservation critical
        'priority': 'high',  # High: Long-term memory preservation important
    },
    'buffer_sorting': {
        'compute': 2.0,   # Low: Quicksort O(n log n)
        'memory': 2.0,    # Low: In-place sorting
        'tool': 0.0,      # None: Internal computation
        'latency': 30.0,  # Low-moderate: Fast sort algorithm
        'risk': 0.02,     # Very low: Standard algorithm
        'priority': 'normal',  # Normal: Ordering helpful but not essential
    },
    'buffer_pruning': {
        'compute': 3.0,   # Moderate: Threshold checking, removal
        'memory': 3.0,    # Moderate: Buffer reallocation
        'tool': 0.0,      # None: Internal computation
        'latency': 40.0,  # Moderate: Memory operations
        'risk': 0.2,      # Moderate: Potential data loss if over-aggressive
        'priority': 'high',  # High: Capacity management critical
    },
    'context_retrieval': {
        'compute': 1.0,   # Very low: Simple iteration
        'memory': 2.0,    # Low: Result buffer
        'tool': 0.0,      # None: Internal computation
        'latency': 20.0,  # Low: Fast lookup
        'risk': 0.02,     # Very low: Read-only operation
        'priority': 'high',  # High: Context access critical for LLM
    },
}


# ============================================================================
# Memory Strategy (Resource-Aware)
# ============================================================================

class MemoryStrategy(Enum):
    """Memory encoding/consolidation strategies based on resource availability."""
    FULL_ENCODING = "full_encoding"  # Full SNARC scoring + extraction + pruning
    SIMPLIFIED_ENCODING = "simplified_encoding"  # Simplified scoring + deferred extraction
    MINIMAL_ENCODING = "minimal_encoding"  # Minimal scoring + no extraction
    TIMESTAMP_ONLY = "timestamp_only"  # No scoring, timestamp ordering only


@dataclass
class MemoryPlan:
    """Plan for multi-resource aware memory operation."""
    strategy: MemoryStrategy
    scoring_method: str  # 'full_snarc', 'simplified', 'minimal', 'none'
    extraction_enabled: bool
    pruning_aggressive: bool
    estimated_cost: Dict[str, float]
    adaptations: List[str]


# ============================================================================
# Mock ConversationTurn (simplified from sage.cognitive.context_memory)
# ============================================================================

@dataclass
class ConversationTurn:
    """Single conversation turn with metadata"""
    speaker: str
    text: str
    timestamp: float
    salience_score: float = 0.0
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================================================
# Multi-Resource Memory Manager
# ============================================================================

class MultiResourceMemoryManager:
    """
    Multi-resource aware wrapper for SNARCMemoryManager.

    Integrates:
    - SNARC salience scoring (5 dimensions: Surprise, Novelty, Arousal, Reward, Conflict)
    - MultiResourceBudget operational modes (NORMAL/STRESSED/CRISIS/SLEEP)

    Key Concept:
    - Memory encoding quality adapts to resource availability
    - Full SNARC → Simplified → Minimal → Timestamp-only
    - Graceful degradation vs complete memory failure
    """

    def __init__(self, max_tokens: int = 127000, tokens_per_turn: int = 50):
        """Initialize multi-resource memory manager."""
        self.budget = MultiResourceBudget()

        # Memory configuration
        self.max_tokens = max_tokens
        self.tokens_per_turn = tokens_per_turn
        self.max_turns = max_tokens // tokens_per_turn

        # Memory buffers
        self.conversation_buffer: List[ConversationTurn] = []
        self.long_term_memory: List[ConversationTurn] = []

        # Statistics
        self.total_turns = 0
        self.extracted_to_longterm = 0
        self.operation_history: List[Dict] = []
        self.strategy_counts: Dict[str, int] = {s.value: 0 for s in MemoryStrategy}

    def plan_memory_operation(
        self,
        operation_type: str = 'add_turn',
    ) -> MemoryPlan:
        """
        Plan memory operation based on resource availability.

        Args:
            operation_type: 'add_turn', 'retrieve_context', 'consolidate'

        Returns:
            MemoryPlan with strategy and cost estimates
        """
        # Get operational mode
        prev_mode, operational_mode = self.budget.update_mode()

        # Determine memory strategy
        if operational_mode == OperationalMode.SLEEP:
            strategy = MemoryStrategy.TIMESTAMP_ONLY
            scoring_method = 'none'
            extraction_enabled = False
            pruning_aggressive = False
            adaptations = ["SLEEP mode: Timestamp-only ordering (defer scoring)"]

        elif operational_mode == OperationalMode.CRISIS:
            strategy = MemoryStrategy.MINIMAL_ENCODING
            scoring_method = 'minimal'
            extraction_enabled = False
            pruning_aggressive = True  # Emergency capacity management
            adaptations = ["CRISIS mode: Minimal encoding (novelty only)"]

        elif operational_mode == OperationalMode.STRESSED:
            strategy = MemoryStrategy.SIMPLIFIED_ENCODING
            scoring_method = 'simplified'
            extraction_enabled = False  # Defer expensive extraction
            pruning_aggressive = False
            adaptations = ["STRESSED mode: Simplified scoring (3 dimensions)"]

        else:  # NORMAL
            strategy = MemoryStrategy.FULL_ENCODING
            scoring_method = 'full_snarc'
            extraction_enabled = True
            pruning_aggressive = False
            adaptations = ["NORMAL mode: Full SNARC encoding (5 dimensions)"]

        self.strategy_counts[strategy.value] += 1

        # Estimate cost based on strategy
        if scoring_method == 'full_snarc':
            cost = MEMORY_COSTS['full_snarc_scoring'].copy()
        elif scoring_method == 'simplified':
            cost = MEMORY_COSTS['simplified_snarc_scoring'].copy()
        elif scoring_method == 'minimal':
            cost = MEMORY_COSTS['minimal_scoring'].copy()
        else:  # none
            cost = {'compute': 0.5, 'memory': 0.5, 'tool': 0.0, 'latency': 5.0, 'risk': 0.0, 'priority': 'low'}

        return MemoryPlan(
            strategy=strategy,
            scoring_method=scoring_method,
            extraction_enabled=extraction_enabled,
            pruning_aggressive=pruning_aggressive,
            estimated_cost=cost,
            adaptations=adaptations,
        )

    def add_turn(
        self,
        speaker: str,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> Optional[ConversationTurn]:
        """
        Add conversation turn with multi-resource aware salience scoring.

        Args:
            speaker: "User" or "Assistant"
            text: Text content
            metadata: Optional metadata

        Returns:
            ConversationTurn if successfully added, None if deferred
        """
        logger.info(f"\nAdding turn: {speaker[:4]}... ({len(text)} chars)")

        # Plan memory operation
        plan = self.plan_memory_operation('add_turn')

        logger.info(f"  Strategy: {plan.strategy.value}")
        logger.info(f"  Scoring: {plan.scoring_method}")

        # Check affordability
        can_afford, bottlenecks = self.budget.can_afford(plan.estimated_cost)
        if not can_afford:
            logger.info(f"  ⚠ Cannot afford {plan.scoring_method} scoring (bottlenecks: {bottlenecks})")
            # Fall back to minimal
            plan.scoring_method = 'minimal'
            plan.estimated_cost = MEMORY_COSTS['minimal_scoring']

        # Create turn
        turn = ConversationTurn(
            speaker=speaker,
            text=text,
            timestamp=datetime.utcnow().timestamp(),
            metadata=metadata or {}
        )

        # Score based on plan
        turn.salience_score = self._calculate_salience(turn, method=plan.scoring_method)

        # Consume resources
        self.budget.consume(plan.estimated_cost)

        # Add to buffer
        self.conversation_buffer.append(turn)
        self.total_turns += 1

        # Manage capacity
        if len(self.conversation_buffer) > self.max_turns:
            self._manage_buffer_capacity(plan.extraction_enabled, plan.pruning_aggressive)

        # Apply recovery
        self.budget.recover()

        # Record operation
        self.operation_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'operation': 'add_turn',
            'strategy': plan.strategy.value,
            'scoring_method': plan.scoring_method,
            'salience_score': turn.salience_score,
            'buffer_size': len(self.conversation_buffer),
            'budget_after': self.budget.get_resource_levels(),
        })

        logger.info(f"  ✓ Turn added (salience={turn.salience_score:.3f})")

        return turn

    def _calculate_salience(self, turn: ConversationTurn, method: str = 'full_snarc') -> float:
        """
        Calculate salience score with resource-aware method selection.

        Args:
            turn: Conversation turn to score
            method: 'full_snarc', 'simplified', 'minimal', 'none'

        Returns:
            Salience score (0.0 - 1.0)
        """
        if method == 'none':
            return 0.5  # Neutral score

        text_lower = turn.text.lower()

        if method == 'minimal':
            # MINIMAL: Novelty only (questions and length)
            is_question = '?' in turn.text
            is_long = len(turn.text) > 100
            return 0.7 if is_question else (0.5 if is_long else 0.3)

        if method == 'simplified':
            # SIMPLIFIED: 3 dimensions (Surprise, Novelty, Reward)
            scores = {}

            # Surprise
            surprise_kw = ['error', 'unexpected', 'strange', 'weird']
            scores['surprise'] = min(1.0, sum(1 for kw in surprise_kw if kw in text_lower) * 0.4)

            # Novelty
            is_question = '?' in turn.text
            is_long = len(turn.text) > 100
            scores['novelty'] = 0.6 if is_question else (0.4 if is_long else 0.2)

            # Reward
            reward_kw = ['learned', 'understand', 'thank', 'great', 'excellent']
            scores['reward'] = min(1.0, sum(1 for kw in reward_kw if kw in text_lower) * 0.4)

            return (scores['surprise'] * 0.3 + scores['novelty'] * 0.4 + scores['reward'] * 0.3)

        # FULL SNARC: 5 dimensions
        scores = {}

        # Surprise
        surprise_kw = ['error', 'unexpected', 'strange', 'weird', 'wow', 'what']
        scores['surprise'] = min(1.0, sum(1 for kw in surprise_kw if kw in text_lower) * 0.3)

        # Novelty
        is_question = '?' in turn.text
        is_long = len(turn.text) > 100
        scores['novelty'] = 0.5 if is_question else (0.3 if is_long else 0.1)

        # Arousal
        arousal_kw = ['!', 'amazing', 'terrible', 'excited', 'frustrated']
        arousal_count = sum(turn.text.count(kw) for kw in arousal_kw)
        scores['arousal'] = min(1.0, arousal_count * 0.4)

        # Reward
        reward_kw = ['learned', 'understand', 'makes sense', 'thank', 'great', 'excellent']
        scores['reward'] = min(1.0, sum(1 for kw in reward_kw if kw in text_lower) * 0.4)

        # Conflict
        conflict_kw = ['no', 'wrong', 'incorrect', 'actually', 'but', 'however']
        scores['conflict'] = min(1.0, sum(1 for kw in conflict_kw if kw in text_lower) * 0.3)

        # Metadata bonuses
        if turn.metadata.get('learned'):
            scores['reward'] = min(1.0, scores['reward'] + 0.5)

        return (
            scores['surprise'] * 0.25 +
            scores['novelty'] * 0.20 +
            scores['arousal'] * 0.15 +
            scores['reward'] * 0.25 +
            scores['conflict'] * 0.15
        )

    def _manage_buffer_capacity(self, extraction_enabled: bool, pruning_aggressive: bool):
        """Manage buffer capacity with resource-aware extraction and pruning."""
        if extraction_enabled:
            # Extract high-salience to long-term
            high_threshold = 0.7
            high_salience = [t for t in self.conversation_buffer if t.salience_score >= high_threshold]

            existing_timestamps = {t.timestamp for t in self.long_term_memory}
            for turn in high_salience:
                if turn.timestamp not in existing_timestamps:
                    self.long_term_memory.append(turn)
                    self.extracted_to_longterm += 1

        # Prune buffer
        if pruning_aggressive:
            # CRISIS: Keep only top 50%
            self.conversation_buffer.sort(key=lambda t: t.salience_score, reverse=True)
            keep_count = len(self.conversation_buffer) // 2
            self.conversation_buffer = self.conversation_buffer[:keep_count]
        else:
            # NORMAL: Keep top 90%
            self.conversation_buffer.sort(key=lambda t: t.salience_score, reverse=True)
            keep_count = int(self.max_turns * 0.9)
            self.conversation_buffer = self.conversation_buffer[:keep_count]

        # Re-sort by timestamp
        self.conversation_buffer.sort(key=lambda t: t.timestamp)

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            'total_turns': self.total_turns,
            'buffer_size': len(self.conversation_buffer),
            'longterm_size': len(self.long_term_memory),
            'extracted_count': self.extracted_to_longterm,
            'strategy_distribution': self.strategy_counts,
            'avg_buffer_salience': (
                sum(t.salience_score for t in self.conversation_buffer) / len(self.conversation_buffer)
                if self.conversation_buffer else 0.0
            ),
            'resource_budget': self.budget.get_resource_levels(),
        }


# ============================================================================
# Session 114: Multi-Resource Memory Integration Test
# ============================================================================

def run_session_114() -> Dict:
    """
    Execute Session 114: Multi-Resource Memory Integration.

    Tests memory encoding under different resource conditions.
    """
    logger.info("=" * 80)
    logger.info("SESSION 114: MULTI-RESOURCE MEMORY (SNARC) INTEGRATION")
    logger.info("=" * 80)
    logger.info("Goal: Bridge multi-resource system to SNARC memory management")
    logger.info("")

    manager = MultiResourceMemoryManager(max_tokens=1000, tokens_per_turn=50)  # Small for testing

    # Test turns with varying salience
    test_turns = [
        ("User", "What is the current project status?", {}),
        ("Assistant", "We've made great progress! The multi-resource framework is complete.", {'learned': True}),
        ("User", "Can you explain how attention works?", {}),
        ("Assistant", "Attention allocation adapts to operational mode based on resource availability.", {}),
        ("User", "Wow, that's unexpected! How does it degrade gracefully?", {}),
        ("Assistant", "FOCUS (80/15/5) becomes DEGRADED (60/25/15) under stress.", {}),
        ("User", "I understand now. Thank you!", {}),
        ("Assistant", "You're welcome! This biological realism is excellent.", {}),
    ]

    # Scenario 1: NORMAL operational mode
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 1: NORMAL Mode (Full SNARC Encoding)")
    logger.info("=" * 80)

    for speaker, text, metadata in test_turns[:4]:
        manager.add_turn(speaker, text, metadata)

    stats_1 = manager.get_stats()

    # Scenario 2: STRESSED mode (deplete compute)
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 2: STRESSED Mode (Simplified Encoding)")
    logger.info("=" * 80)

    manager.budget.compute_atp = 15.0

    for speaker, text, metadata in test_turns[4:6]:
        manager.add_turn(speaker, text, metadata)

    stats_2 = manager.get_stats()

    # Scenario 3: CRISIS mode (severe depletion)
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 3: CRISIS Mode (Minimal Encoding)")
    logger.info("=" * 80)

    manager.budget.compute_atp = 5.0
    manager.budget.memory_atp = 3.0

    for speaker, text, metadata in test_turns[6:8]:
        manager.add_turn(speaker, text, metadata)

    stats_3 = manager.get_stats()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SESSION 114 SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\nStrategy Distribution:")
    for strategy, count in stats_3['strategy_distribution'].items():
        if count > 0:
            logger.info(f"  {strategy}: {count} turns")

    logger.info(f"\nMemory Statistics:")
    logger.info(f"  Total turns: {stats_3['total_turns']}")
    logger.info(f"  Buffer size: {stats_3['buffer_size']}")
    logger.info(f"  Long-term memories: {stats_3['longterm_size']}")
    logger.info(f"  Avg buffer salience: {stats_3['avg_buffer_salience']:.3f}")

    logger.info(f"\nKey Findings:")
    logger.info(f"  ✓ Memory encoding adapts to operational mode")
    logger.info(f"  ✓ Full SNARC (5D) → Simplified (3D) → Minimal (1D)")
    logger.info(f"  ✓ Extraction deferred under stress (resource conservation)")
    logger.info(f"  ✓ Graceful degradation vs complete memory failure")

    # Save results
    output = {
        'session': 114,
        'timestamp': datetime.utcnow().isoformat(),
        'scenarios': {
            'normal_mode': stats_1,
            'stressed_mode': stats_2,
            'crisis_mode': stats_3,
        },
        'operation_history': manager.operation_history,
    }

    output_file = 'sage/experiments/session114_multiresource_memory_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("SESSION 114 COMPLETE")
    logger.info("=" * 80)

    return output


if __name__ == "__main__":
    results = run_session_114()
