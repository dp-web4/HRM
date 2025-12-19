#!/usr/bin/env python3
"""
Session 68: Trust-First vs Weighted Blend Comparison

Validates Thor's Session 72 paradigm shift by comparing:
1. Weighted blend (α=0.3): selection = 0.3×router + 0.7×trust
2. Trust-first: if has_evidence → pure_trust else free_router

Expected (from Thor S72):
- Weighted blend: ~17 experts
- Trust-first: ~58 experts (3.4x improvement)

Created: Session 68 (2025-12-18)
"""

import sys
from pathlib import Path
import numpy as np
import json
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.core.mrh_expert_selector import create_mrh_selector
from sage.core.trust_first_mrh_selector import create_trust_first_selector


def run_comparison():
    """Compare weighted blend vs trust-first over 50 generations."""
    print("=" * 70)
    print("Session 68: Paradigm Shift Validation")
    print("Weighted Blend vs Trust-First Architecture")
    print("=" * 70)

    np.random.seed(42)
    monopoly_experts = [73, 114, 95, 106]

    # Test 1: Weighted blend (α=0.3)
    print("\n1. Testing WEIGHTED BLEND (α=0.3)...")
    print("-" * 70)

    selector_weighted = create_mrh_selector(num_experts=128)
    selector_weighted.exploration_weight = 0.3

    # Register experts
    specialists = {0: [42, 17, 88], 1: [99, 55, 120], 2: [1, 63, 110]}
    for context_id, expert_ids in specialists.items():
        for expert_id in expert_ids:
            embeddings = np.random.randn(10, 8)
            embeddings[:, context_id] += 3.0
            selector_weighted.register_expert_contexts(expert_id, embeddings)
            for _ in range(5):
                selector_weighted.update_trust_for_expert(expert_id, context_id, 0.80 + np.random.randn() * 0.05)

    for expert_id in monopoly_experts:
        embeddings = np.random.randn(18, 8)
        embeddings[:6, 0] += 1.5
        embeddings[6:12, 1] += 1.5
        embeddings[12:, 2] += 1.5
        selector_weighted.register_expert_contexts(expert_id, embeddings)
        for context_id in range(3):
            for i in range(6):
                selector_weighted.update_trust_for_expert(expert_id, context_id, 0.32 - (i * 0.03))

    # Run 50 generations
    contexts = [0, 1, 2] * 17
    contexts = contexts[:50]
    all_experts = list(range(128))

    expert_usage_weighted = []
    for context_id in contexts:
        router_logits = np.random.randn(128).astype(np.float32) * 0.1
        router_logits[monopoly_experts] += 2.0

        result = selector_weighted.select_experts(router_logits, context=context_id, k=4, all_expert_ids=all_experts)
        expert_usage_weighted.extend(result.selected_expert_ids)

        for expert_id in result.selected_expert_ids:
            is_specialist = expert_id in specialists.get(context_id, [])
            quality = (0.78 + np.random.randn() * 0.03) if is_specialist else (0.22 + np.random.randn() * 0.03)
            selector_weighted.update_trust_for_expert(expert_id, context_id, quality)

    unique_weighted = len(set(expert_usage_weighted))
    stats_weighted = selector_weighted.get_statistics()

    print(f"  Unique experts: {unique_weighted}")
    print(f"  Utilization: {unique_weighted/128:.1%}")
    print(f"  MRH substitutions: {stats_weighted['total_mrh_substitutions']}")

    # Test 2: Trust-first
    print("\n2. Testing TRUST-FIRST (conditional)...")
    print("-" * 70)

    np.random.seed(42)  # Same initialization

    selector_trust = create_trust_first_selector(num_experts=128)

    # Register experts (same as weighted)
    for context_id, expert_ids in specialists.items():
        for expert_id in expert_ids:
            embeddings = np.random.randn(10, 8)
            embeddings[:, context_id] += 3.0
            selector_trust.register_expert_contexts(expert_id, embeddings)
            for _ in range(5):
                selector_trust.update_trust_for_expert(expert_id, context_id, 0.80 + np.random.randn() * 0.05)

    for expert_id in monopoly_experts:
        embeddings = np.random.randn(18, 8)
        embeddings[:6, 0] += 1.5
        embeddings[6:12, 1] += 1.5
        embeddings[12:, 2] += 1.5
        selector_trust.register_expert_contexts(expert_id, embeddings)
        for context_id in range(3):
            for i in range(6):
                selector_trust.update_trust_for_expert(expert_id, context_id, 0.32 - (i * 0.03))

    expert_usage_trust = []
    for context_id in contexts:
        router_logits = np.random.randn(128).astype(np.float32) * 0.1
        router_logits[monopoly_experts] += 2.0

        result = selector_trust.select_experts(router_logits, context=context_id, k=4, all_expert_ids=all_experts)
        expert_usage_trust.extend(result.selected_expert_ids)

        for expert_id in result.selected_expert_ids:
            is_specialist = expert_id in specialists.get(context_id, [])
            quality = (0.78 + np.random.randn() * 0.03) if is_specialist else (0.22 + np.random.randn() * 0.03)
            selector_trust.update_trust_for_expert(expert_id, context_id, quality)

    unique_trust = len(set(expert_usage_trust))
    stats_trust = selector_trust.get_statistics()

    print(f"  Unique experts: {unique_trust}")
    print(f"  Utilization: {unique_trust/128:.1%}")
    print(f"  Trust-driven selections: {stats_trust['trust_driven']}/{stats_trust['total_selections']} ({stats_trust['trust_driven_rate']:.1%})")

    # Results
    print("\n" + "=" * 70)
    print("3. Comparison Results")
    print("=" * 70)

    print(f"\nRouter Baseline (Session 69): 4 experts (3.1%)")
    print(f"\nWeighted Blend (α=0.3):      {unique_weighted} experts ({unique_weighted/128:.1%})")
    print(f"Trust-First (conditional):    {unique_trust} experts ({unique_trust/128:.1%})")

    improvement = (unique_trust / unique_weighted - 1) * 100 if unique_weighted > 0 else 0
    print(f"\nTrust-First improvement: +{improvement:.0f}%")

    # Thor's Session 72 target: 3.4x (17 → 58)
    thor_target_multiplier = 3.4
    expected_from_thor = unique_weighted * thor_target_multiplier

    print(f"\nThor S72 target: {thor_target_multiplier}x improvement")
    print(f"Expected (from Thor): ~{expected_from_thor:.0f} experts")
    print(f"Actual: {unique_trust} experts")

    if improvement >= 200:  # 3x = 200% improvement
        print(f"\n✓ PARADIGM SHIFT VALIDATED")
        print(f"  Trust-first architecture achieves >3x improvement")
    elif improvement >= 50:
        print(f"\n⚠ PARTIAL VALIDATION")
        print(f"  Trust-first shows improvement but below Thor's 3.4x")
    else:
        print(f"\n✗ NO SIGNIFICANT DIFFERENCE")
        print(f"  Trust-first did not outperform weighted blend")

    # Save results
    results = {
        'session': 68,
        'comparison': 'weighted_blend_vs_trust_first',
        'weighted_blend': {
            'alpha': 0.3,
            'unique_experts': unique_weighted,
            'utilization': unique_weighted / 128,
            'mrh_substitutions': stats_weighted['total_mrh_substitutions']
        },
        'trust_first': {
            'unique_experts': unique_trust,
            'utilization': unique_trust / 128,
            'trust_driven_rate': stats_trust['trust_driven_rate']
        },
        'improvement_pct': improvement,
        'thor_target_multiplier': thor_target_multiplier,
        'expert_usage': {
            'weighted': Counter([int(x) for x in expert_usage_weighted]).most_common(10),
            'trust_first': Counter([int(x) for x in expert_usage_trust]).most_common(10)
        }
    }

    output_file = Path(__file__).parent.parent / "experiments" / "session68_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_comparison()
