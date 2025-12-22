#!/usr/bin/env python3
"""Session 91 Lambda Parameter Sweep.

Test different lambda_variance values to find optimal trust vs skill split.
"""

import logging
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.session91_regret_tracking import (
    RegretTrackingSelector,
    load_jsonl_conversations,
    simulate_expert_selection_with_signals,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test different lambda values."""

    logger.info("="*80)
    logger.info("SESSION 91: LAMBDA VARIANCE PARAMETER SWEEP")
    logger.info("="*80)

    num_experts = 128
    num_layers = 48
    num_generations = 810

    conversation_path = Path("phase1-hierarchical-cognitive/epistemic_bias_mapping/conversational_learning/conversation_sessions")

    if not conversation_path.exists():
        logger.error(f"Conversation path not found: {conversation_path}")
        conversations = []
    else:
        conversations = load_jsonl_conversations(conversation_path)
        logger.info(f"Loaded {len(conversations)} real conversations")

    # Test different lambda values
    lambda_values = [0.05, 0.1, 0.15, 0.2, 0.3]
    results = []

    for lambda_var in lambda_values:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing lambda_variance = {lambda_var}")
        logger.info(f"{'='*80}")

        selector = RegretTrackingSelector(
            num_experts=num_experts,
            num_layers=num_layers,
            epsilon=0.2,
            min_evidence_for_trust=1,
            reputation_weight=0.4,
            max_hot_experts=64,
            base_hysteresis_boost=0.2,
            switching_cost_weight=0.3,
            memory_cost_weight=0.2,
            max_swaps_per_gen=8,
            lambda_variance=lambda_var,
            regret_protection_threshold=0.5,
            reputation_db_path=Path(f"session91_lambda{lambda_var}_reputation.db"),
        )

        stats = simulate_expert_selection_with_signals(
            conversations=conversations,
            selector=selector,
            num_generations=num_generations,
            num_selections_per_gen=num_layers,
        )

        results.append({
            'lambda': lambda_var,
            'trust_pct': stats['trust_driven_pct'],
            'first_activation': stats['first_trust_activation'],
            'cache_hit': stats['cache_hit_rate'],
            'churn': stats['expert_churn_rate'],
            'total_regret': stats.get('total_cumulative_regret', 0),
        })

        logger.info(f"\nResults for λ={lambda_var}:")
        logger.info(f"  Trust-driven: {stats['trust_driven_pct']:.1f}%")
        logger.info(f"  First activation: Gen {stats['first_trust_activation']}")
        logger.info(f"  Cache hit: {stats['cache_hit_rate']*100:.1f}%")
        logger.info(f"  Churn: {stats['expert_churn_rate']:.3f}")

    # Analysis
    logger.info(f"\n{'='*80}")
    logger.info("PARAMETER SWEEP RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"{'Lambda':<10} {'Trust%':<10} {'Activation':<12} {'Cache%':<10} {'Churn':<10}")
    logger.info(f"{'-'*60}")

    best_activation = float('inf')
    best_lambda = None

    for r in results:
        logger.info(
            f"{r['lambda']:<10.2f} {r['trust_pct']:<10.1f} "
            f"{str(r['first_activation']):<12} {r['cache_hit']*100:<10.1f} "
            f"{r['churn']:<10.3f}"
        )

        if r['first_activation'] and r['first_activation'] < best_activation:
            best_activation = r['first_activation']
            best_lambda = r['lambda']

    logger.info(f"\n{'='*80}")
    logger.info(f"OPTIMAL: λ={best_lambda} → Activation at Gen {best_activation}")
    logger.info(f"{'='*80}")

    logger.info("\nNova's guidance: 'This single subtraction does wonders'")
    logger.info(f"Validated: λ={best_lambda} achieves fastest trust activation")


if __name__ == '__main__':
    main()
