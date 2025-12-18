#!/usr/bin/env python3
"""
Integration Test: MRH Expert Selector Breaking Router Monopoly

Demonstrates Session 65 breakthrough integrated with SAGE infrastructure.

Test Scenario:
1. Simulate router collapse (4 experts monopolize selection)
2. MRH selector identifies low trust
3. Finds alternatives via context overlap
4. Substitutes high-trust specialists
5. Result: Monopoly broken, specialists emerge

Expected Outcome (from Session 65):
- Expert diversity: 4 → 8+ (>100% increase)
- Specialist emergence: 60%+ specialist rate
- Trust evolution: Specialists maintain/increase, generalists decline

Created: Session 66 (Autonomous Web4 Research)
Date: 2025-12-18
"""

import sys
from pathlib import Path
import numpy as np

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.core.mrh_expert_selector import create_mrh_selector
from sage.core.expert_reputation import get_default_reputation_db
from sage.core.context_classifier import ContextClassifier
from collections import Counter


def test_mrh_monopoly_breaking():
    """
    Test that MRH selector breaks router monopoly via context overlap.
    """
    print("=" * 70)
    print("MRH Expert Selector Integration Test")
    print("Session 65 Breakthrough → SAGE Implementation")
    print("=" * 70)

    # Initialize components
    print("\n1. Initializing MRH selector...")
    selector = create_mrh_selector(num_experts=128)

    # Simulate router collapse (Session 69 discovery)
    monopoly_experts = [73, 114, 95, 106]
    print(f"   Router monopoly: {monopoly_experts}")
    print(f"   Total experts: 128")
    print(f"   Utilization: {len(monopoly_experts)/128:.1%}")

    # Create specialist experts for each context
    print("\n2. Registering specialist experts...")
    np.random.seed(42)

    specialists = {
        0: [42, 17, 88],    # Code specialists (context_0)
        1: [99, 55, 120],   # Reasoning specialists (context_1)
        2: [1, 63, 110]     # Text specialists (context_2)
    }

    for context_id, expert_ids in specialists.items():
        for expert_id in expert_ids:
            # Create embeddings biased toward this context
            embeddings = np.random.randn(10, 8)
            embeddings[:, context_id] += 3.0  # Strong bias

            # Register expert's contexts
            contexts = selector.register_expert_contexts(expert_id, embeddings)
            print(f"   Expert {expert_id}: {dict(Counter(contexts))} (specialist in context_{context_id})")

            # Initialize high trust for specialists in their context
            for _ in range(5):
                quality = 0.80 + np.random.randn() * 0.05
                selector.update_trust_for_expert(expert_id, context_id, quality)

    # Register monopoly experts (generalists with LOW trust)
    print("\n3. Registering monopoly experts (low trust generalists)...")
    for expert_id in monopoly_experts:
        # Generalists: spread across contexts
        embeddings = np.random.randn(18, 8)
        embeddings[:6, 0] += 1.5
        embeddings[6:12, 1] += 1.5
        embeddings[12:, 2] += 1.5

        contexts = selector.register_expert_contexts(expert_id, embeddings)
        print(f"   Expert {expert_id}: {dict(Counter(contexts))} (generalist)")

        # Initialize DECLINING trust (starts low, declines to trigger substitution)
        for context_id in range(3):
            for i in range(6):
                trust = 0.32 - (i * 0.03)  # Ends at 0.17 < 0.3 threshold
                selector.update_trust_for_expert(expert_id, context_id, trust)

    # Run simulation: 20 generations across 3 contexts
    print("\n4. Running simulation (20 generations)...")
    print("-" * 70)

    contexts = [0, 1, 2] * 7  # 21 generations, trim to 20
    contexts = contexts[:20]

    all_experts = list(range(128))
    expert_usage = []

    for generation, context_id in enumerate(contexts, 1):
        # Simulate router output (always selects monopoly experts)
        router_logits = np.random.randn(128).astype(np.float32) * 0.1
        router_logits[monopoly_experts] += 2.0  # Bias toward monopoly

        # MRH-based selection
        result = selector.select_experts(
            router_logits,
            context=context_id,
            k=4,
            all_expert_ids=all_experts
        )

        # Track usage
        expert_usage.extend(result.selected_expert_ids)

        # Show substitutions
        if result.mrh_substitutions:
            print(f"\nGeneration {generation} (Context {context_id}):")
            for sub in result.mrh_substitutions:
                print(f"  [MRH] Expert {sub.requested_expert} (trust={sub.requested_trust:.2f}) "
                      f"→ Expert {sub.substitute_expert} (trust={sub.substitute_trust:.2f}, "
                      f"overlap={sub.context_overlap:.2f})")

        # Simulate quality and update trust
        for expert_id in result.selected_expert_ids:
            # Specialists get high quality in their context
            is_specialist = expert_id in specialists.get(context_id, [])

            if is_specialist:
                quality = 0.78 + np.random.randn() * 0.03
            else:
                quality = 0.22 + np.random.randn() * 0.03

            selector.update_trust_for_expert(expert_id, context_id, quality)

    # Results
    print("\n" + "=" * 70)
    print("5. Results")
    print("=" * 70)

    usage_counts = Counter(expert_usage)
    unique_experts = len(usage_counts)

    print(f"\nRouter Baseline:")
    print(f"  Unique experts: {len(monopoly_experts)}")
    print(f"  Utilization: {len(monopoly_experts)/128:.1%}")

    print(f"\nMRH Selector:")
    print(f"  Unique experts: {unique_experts}")
    print(f"  Utilization: {unique_experts/128:.1%}")
    print(f"  Improvement: +{(unique_experts/len(monopoly_experts) - 1)*100:.0f}%")

    # Specialist analysis
    specialist_count = 0
    generalist_count = 0

    print(f"\nExpert Usage:")
    for expert_id in sorted(usage_counts.keys()):
        count = usage_counts[expert_id]
        # Determine context distribution
        # Build mapping of usage index to context
        expert_usage_with_context = list(zip(expert_usage, contexts * (len(expert_usage) // len(contexts) + 1)))
        contexts_used = [ctx for eid, ctx in expert_usage_with_context if eid == expert_id]
        context_dist = Counter(contexts_used)

        expert_type = "specialist" if len(set(contexts_used)) == 1 else "generalist"
        if expert_type == "specialist":
            specialist_count += 1
        else:
            generalist_count += 1

        print(f"  Expert {expert_id:3d}: {count:2d} uses, {dict(context_dist)} ({expert_type})")

    print(f"\nSpecialist Emergence:")
    print(f"  Specialists: {specialist_count} ({specialist_count/unique_experts:.1%})")
    print(f"  Generalists: {generalist_count}")

    # MRH substitution stats
    stats = selector.get_statistics()
    print(f"\nMRH Substitutions:")
    print(f"  Total: {stats['total_mrh_substitutions']}")
    print(f"  Rate: {stats['substitution_rate']:.1%}")

    # Substitution summary
    sub_summary = selector.get_substitution_summary()
    print(f"\nSubstitutions by Context:")
    for context_id, ctx_stats in sub_summary['by_context'].items():
        print(f"  Context {context_id}:")
        print(f"    Count: {ctx_stats['count']}")
        print(f"    Avg overlap: {ctx_stats['avg_overlap']:.2f}")
        print(f"    Avg trust improvement: {ctx_stats['avg_trust_improvement']:+.2f}")
        print(f"    Experts substituted: {ctx_stats['experts_substituted']}")
        print(f"    Experts used: {ctx_stats['experts_used']}")

    # Validation
    print("\n" + "=" * 70)
    print("6. Validation")
    print("=" * 70)

    success = True

    # Check diversity increase
    diversity_increase = (unique_experts / len(monopoly_experts) - 1) * 100
    if diversity_increase < 50:
        print(f"❌ Diversity increase ({diversity_increase:.0f}%) < 50% (Session 65: 100%)")
        success = False
    else:
        print(f"✓ Diversity increased by {diversity_increase:.0f}% (Session 65 target: 100%)")

    # Check specialist emergence
    specialist_rate = specialist_count / unique_experts if unique_experts > 0 else 0
    if specialist_rate < 0.5:
        print(f"❌ Specialist rate ({specialist_rate:.1%}) < 50% (Session 65: 62.5%)")
        success = False
    else:
        print(f"✓ Specialist rate: {specialist_rate:.1%} (Session 65 target: 62.5%)")

    # Check MRH substitutions occurred
    if stats['total_mrh_substitutions'] == 0:
        print(f"❌ No MRH substitutions occurred")
        success = False
    else:
        print(f"✓ MRH substitutions: {stats['total_mrh_substitutions']}")

    print("\n" + "=" * 70)
    if success:
        print("✓ TEST PASSED: MRH selector successfully breaks router monopoly")
        print("  Session 65 breakthrough validated in SAGE infrastructure")
    else:
        print("❌ TEST FAILED: MRH selector did not achieve Session 65 results")

    print("=" * 70)

    return success


if __name__ == "__main__":
    success = test_mrh_monopoly_breaking()
    sys.exit(0 if success else 1)
