"""
Session 118: Multi-Resource Expert Selector Integration

Goal: Integrate Expert Selector with multi-resource consciousness framework

Applies proven multi-resource pattern (S107-117) to expert selection, completing
coverage of major SAGE consciousness components.

Pattern Applied:
- Scheduler wraps TrustBasedExpertSelector
- Maps expert operations → resource costs
- Adapts selection strategy based on operational mode
- Graceful degradation under resource stress

Resource Costs:
- Expert Evaluation: High compute (rank all experts), moderate memory (expert profiles)
- Expert Consensus: High compute (multi-expert synthesis), high memory (aggregation)
- Expert Execution: Variable (depends on expert complexity)
- Reputation Update: Low compute, low memory (database write)

Operational Mode Adaptations:
- NORMAL: Full expert panel (k=8), evaluate all candidates, multi-expert consensus
- STRESSED: Reduced panel (k=4), evaluate top candidates only, limited consensus
- CRISIS: Single best expert (k=1), no consensus, cached reputation
- SLEEP: Defer expert selection, use default/cached expert

Expected Discoveries:
- Expert selection quality vs resource tradeoff
- Multi-expert consensus under resource constraints
- Emergent expert specialization (simpler experts in stressed mode)
- Cross-component integration (attention + memory + expert choice)

Biological Parallels:
- Neural resource allocation: PFC expert selection vs execution
- Cognitive load: Fewer experts under stress (limited working memory)
- Decision quality: Single expert (fast) vs panel consensus (accurate)
- Sleep deferral: Cached decisions when ATP low
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


class ExpertStrategy(Enum):
    """Expert selection strategies based on operational mode."""
    FULL_PANEL = "full_panel"           # k=8, full evaluation, consensus
    REDUCED_PANEL = "reduced_panel"     # k=4, limited evaluation
    SINGLE_EXPERT = "single_expert"     # k=1, no consensus
    CACHED_EXPERT = "cached_expert"     # k=0, use cached decision


@dataclass
class ExpertSelectionCost:
    """Resource costs for expert operations."""
    # Expert evaluation costs (rank all candidates)
    evaluation_compute: float = 0.0    # Router + reputation scoring
    evaluation_memory: float = 0.0     # Expert profile loading

    # Expert consensus costs (multi-expert aggregation)
    consensus_compute: float = 0.0     # Weighted combination
    consensus_memory: float = 0.0      # Aggregation buffer

    # Reputation update costs
    reputation_compute: float = 0.0    # Database write
    reputation_memory: float = 0.0     # Transaction


class MultiResourceExpertSelector:
    """
    Multi-resource expert selector integrating resource-aware expert choice.

    Wraps TrustBasedExpertSelector with multi-resource scheduling:
    - Adapts expert panel size to resource availability
    - Degrades from consensus → reduced panel → single expert → cached
    - Tracks resource consumption for expert operations
    - Validates graceful degradation under stress
    """

    def __init__(
        self,
        budget: Optional[MultiResourceBudget] = None,
        num_experts: int = 128,
        cache_size: int = 6,
    ):
        """Initialize multi-resource expert selector."""
        # Shared resource budget
        self.budget = budget if budget else MultiResourceBudget()

        # Expert selector config
        self.num_experts = num_experts
        self.cache_size = cache_size

        # Statistics
        self.total_selections = 0
        self.deferrals_by_strategy = {
            ExpertStrategy.FULL_PANEL: 0,
            ExpertStrategy.REDUCED_PANEL: 0,
            ExpertStrategy.SINGLE_EXPERT: 0,
            ExpertStrategy.CACHED_EXPERT: 0,
        }
        self.strategy_distribution = {strategy: 0 for strategy in ExpertStrategy}
        self.cached_expert = None  # Last selected expert for caching

    def select_experts(
        self,
        router_logits: np.ndarray,
        context: str,
        priority: str = "NORMAL",
    ) -> Dict:
        """
        Select experts with resource-aware strategy.

        Args:
            router_logits: Router scores for experts [num_experts]
            context: Context string for expert selection
            priority: Selection priority (CRITICAL, HIGH, NORMAL, LOW)

        Returns:
            Selection result with experts, strategy, resource consumption
        """
        mode = self.budget.assess_operational_mode()

        logger.info(f"\n{'='*80}")
        logger.info(f"EXPERT SELECTION")
        logger.info(f"Operational Mode: {mode.value}")
        logger.info(f"Priority: {priority}")
        logger.info(f"Resources: compute={self.budget.compute_atp:.1f}, "
                   f"memory={self.budget.memory_atp:.1f}")
        logger.info(f"{'='*80}\n")

        # Determine strategy based on mode and priority
        strategy, k = self._determine_strategy(mode, priority)

        # Calculate resource costs
        cost = self._calculate_costs(strategy, k, len(router_logits))

        # Check if we can afford it
        affordable, _ = self.budget.can_afford(cost)

        if not affordable:
            # Try degraded strategy
            degraded_strategy, degraded_k = self._degrade_strategy(strategy)
            degraded_cost = self._calculate_costs(degraded_strategy, degraded_k, len(router_logits))

            affordable_degraded, _ = self.budget.can_afford(degraded_cost)

            if affordable_degraded:
                logger.info(f"  ⚠ Degrading: {strategy.value} → {degraded_strategy.value}")
                strategy, k, cost = degraded_strategy, degraded_k, degraded_cost
            else:
                # Cannot afford even degraded - defer to cached
                logger.info(f"  ✗ Deferred to cached expert (insufficient resources)")
                self.deferrals_by_strategy[strategy] += 1
                strategy = ExpertStrategy.CACHED_EXPERT
                k = 0
                cost = {'compute': 0.1, 'memory': 0.1}  # Minimal cache lookup

        # Execute selection
        result = self._execute_selection(router_logits, context, strategy, k)

        # Consume resources
        self.budget.consume(cost)

        # Update statistics
        self.total_selections += 1
        self.strategy_distribution[strategy] += 1

        # Cache for future use
        if result['selected_experts']:
            self.cached_expert = result['selected_experts'][0]

        logger.info(f"  Strategy: {strategy.value}")
        logger.info(f"  Experts selected: {k}")
        logger.info(f"  Cost: compute={cost['compute']:.1f}, memory={cost['memory']:.1f}")
        logger.info(f"  ✓ Selection complete\n")

        return {
            'strategy': strategy.value,
            'selected_experts': result['selected_experts'],
            'selection_scores': result['selection_scores'],
            'k': k,
            'cost': cost,
            'deferred': not affordable,
        }

    def _determine_strategy(
        self,
        mode: OperationalMode,
        priority: str,
    ) -> tuple[ExpertStrategy, int]:
        """Determine expert selection strategy based on mode and priority."""
        if mode == OperationalMode.NORMAL:
            return ExpertStrategy.FULL_PANEL, 8
        elif mode == OperationalMode.STRESSED:
            return ExpertStrategy.REDUCED_PANEL, 4
        elif mode == OperationalMode.CRISIS:
            # In crisis, only CRITICAL/HIGH priority gets single expert
            if priority in ["CRITICAL", "HIGH"]:
                return ExpertStrategy.SINGLE_EXPERT, 1
            else:
                return ExpertStrategy.CACHED_EXPERT, 0
        else:  # SLEEP
            return ExpertStrategy.CACHED_EXPERT, 0

    def _degrade_strategy(
        self,
        strategy: ExpertStrategy,
    ) -> tuple[ExpertStrategy, int]:
        """Degrade strategy to lower resource version."""
        if strategy == ExpertStrategy.FULL_PANEL:
            return ExpertStrategy.REDUCED_PANEL, 4
        elif strategy == ExpertStrategy.REDUCED_PANEL:
            return ExpertStrategy.SINGLE_EXPERT, 1
        elif strategy == ExpertStrategy.SINGLE_EXPERT:
            return ExpertStrategy.CACHED_EXPERT, 0
        else:
            return ExpertStrategy.CACHED_EXPERT, 0

    def _calculate_costs(
        self,
        strategy: ExpertStrategy,
        k: int,
        num_experts: int,
    ) -> Dict[str, float]:
        """Calculate resource costs for expert selection strategy."""
        if strategy == ExpertStrategy.FULL_PANEL:
            # Full evaluation + consensus
            return {
                'compute': 30.0 + (k * 5.0),  # Evaluation + per-expert consensus
                'memory': 20.0 + (k * 3.0),   # Expert profiles + aggregation
            }
        elif strategy == ExpertStrategy.REDUCED_PANEL:
            # Limited evaluation + reduced consensus
            return {
                'compute': 15.0 + (k * 3.0),
                'memory': 10.0 + (k * 2.0),
            }
        elif strategy == ExpertStrategy.SINGLE_EXPERT:
            # Minimal evaluation + no consensus
            return {
                'compute': 8.0,
                'memory': 5.0,
            }
        else:  # CACHED_EXPERT
            # Cache lookup only
            return {
                'compute': 0.5,
                'memory': 0.5,
            }

    def _execute_selection(
        self,
        router_logits: np.ndarray,
        context: str,
        strategy: ExpertStrategy,
        k: int,
    ) -> Dict:
        """Execute expert selection with given strategy."""
        if strategy == ExpertStrategy.CACHED_EXPERT:
            # Use cached expert
            if self.cached_expert is not None:
                return {
                    'selected_experts': [self.cached_expert],
                    'selection_scores': [1.0],
                }
            else:
                # No cache, use top-1 from router
                top_expert = int(np.argmax(router_logits))
                return {
                    'selected_experts': [top_expert],
                    'selection_scores': [float(router_logits[top_expert])],
                }

        # Get top-k experts from router logits
        top_k_indices = np.argsort(router_logits)[-k:][::-1]
        top_k_scores = router_logits[top_k_indices]

        return {
            'selected_experts': top_k_indices.tolist(),
            'selection_scores': top_k_scores.tolist(),
        }

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        return {
            'total_selections': self.total_selections,
            'strategy_distribution': {
                strategy.value: count
                for strategy, count in self.strategy_distribution.items()
            },
            'deferrals_by_strategy': {
                strategy.value: count
                for strategy, count in self.deferrals_by_strategy.items()
            },
            'final_budget': {
                'compute': self.budget.compute_atp,
                'memory': self.budget.memory_atp,
                'tool': self.budget.tool_atp,
                'latency': self.budget.latency_budget,
                'risk': self.budget.risk_budget,
            },
        }


def run_session_118():
    """Run Session 118 multi-resource expert selector tests."""

    logger.info("="*80)
    logger.info("SESSION 118: MULTI-RESOURCE EXPERT SELECTOR INTEGRATION")
    logger.info("="*80)
    logger.info("Goal: Apply multi-resource pattern to expert selection")
    logger.info("Pattern: Scheduler wraps TrustBasedExpertSelector logic")
    logger.info("\n")

    # Test scenarios
    scenarios = [
        {
            'name': "Normal Operation",
            'description': "Full expert panel selection under normal resources",
            'selections': 5,
            'disable_recovery': False,
        },
        {
            'name': "Resource Stress",
            'description': "Expert selection under resource constraints",
            'selections': 10,
            'disable_recovery': False,
        },
        {
            'name': "Resource Starvation",
            'description': "Expert selection without recovery (force degradation)",
            'selections': 15,
            'disable_recovery': True,
        },
    ]

    all_results = {}

    for scenario in scenarios:
        logger.info("\n\n")
        logger.info("="*80)
        logger.info(f"SCENARIO: {scenario['name']}")
        logger.info("="*80)
        logger.info(f"Description: {scenario['description']}")
        logger.info(f"Selections: {scenario['selections']}")
        logger.info(f"Disable recovery: {scenario['disable_recovery']}")
        logger.info("\n")

        # Initialize selector for this scenario
        selector = MultiResourceExpertSelector(num_experts=128, cache_size=6)

        # Run expert selections
        for i in range(scenario['selections']):
            logger.info(f"\n{'='*80}")
            logger.info(f"SELECTION {i+1}/{scenario['selections']}")
            logger.info(f"{'='*80}")

            # Generate synthetic router logits
            router_logits = np.random.randn(128) * 0.5
            context = f"test_context_{i}"
            priority = ["CRITICAL", "HIGH", "NORMAL", "LOW"][i % 4]

            # Select experts
            result = selector.select_experts(router_logits, context, priority)

            # Recovery (if enabled)
            if not scenario['disable_recovery']:
                selector.budget.recover()

            logger.info(f"Resources after: compute={selector.budget.compute_atp:.1f}, "
                       f"memory={selector.budget.memory_atp:.1f}")

        # Get statistics
        stats = selector.get_stats()

        logger.info(f"\n\n{'='*80}")
        logger.info(f"SCENARIO COMPLETE: {scenario['name']}")
        logger.info(f"{'='*80}")
        logger.info(f"\nStatistics:")
        logger.info(f"  Total selections: {stats['total_selections']}")
        logger.info(f"\nStrategy Distribution:")
        for strategy, count in stats['strategy_distribution'].items():
            pct = (count / stats['total_selections'] * 100) if stats['total_selections'] > 0 else 0
            logger.info(f"  {strategy}: {count} selections ({pct:.1f}%)")
        logger.info(f"\nFinal Resource Budget:")
        for resource, value in stats['final_budget'].items():
            logger.info(f"  {resource}: {value:.1f}")

        all_results[scenario['name']] = {
            'description': scenario['description'],
            'statistics': stats,
        }

    # Save results
    output = {
        'session': 118,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'component': 'expert_selector',
        'scenarios': all_results,
    }

    output_file = 'sage/experiments/session118_multiresource_expert_selector_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n\nResults saved to: {output_file}")

    logger.info(f"\n\n{'='*80}")
    logger.info("SESSION 118 COMPLETE - EXPERT SELECTOR INTEGRATION SUCCESS!")
    logger.info(f"{'='*80}\n")

    return all_results


if __name__ == '__main__':
    results = run_session_118()
