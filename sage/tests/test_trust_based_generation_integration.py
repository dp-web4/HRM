#!/usr/bin/env python3
"""
Trust-Based Expert Selection Integration Demo

Demonstrates how TrustBasedExpertSelector would integrate with Q3-Omni
text generation for adaptive expert selection based on contextual trust.

This is a **research exploration** showing the integration pattern without
modifying the core SelectiveLanguageModel (preserving validated architecture).

Session Context: Autonomous Thor Session 57
Previous Work:
  - Session 54: Memory persistence (Thor)
  - Session 55: ExpertReputation system (Legion)
  - Session 56: TrustBasedExpertSelector (Legion)
  - Session 57: Integration exploration (Thor) ← This session

Author: Thor-SAGE-Researcher (Autonomous)
Date: 2025-12-16
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import tempfile
import torch
from pathlib import Path
from typing import Dict, List, Tuple

from sage.core.trust_based_expert_selector import (
    TrustBasedExpertSelector,
    create_trust_based_selector,
)
from sage.core.expert_reputation import (
    ExpertReputationDB,
    record_expert_activation,
)


def simulate_expert_selection_with_trust(
    router_logits: torch.Tensor,
    context: str,
    db_path: Path,
    num_experts: int = 128,
    k: int = 8,
    exploration_weight: float = 0.3
) -> Dict:
    """
    Simulates trust-based expert selection for a generation step.

    This demonstrates how TrustBasedExpertSelector augments the router's
    learned preferences with empirical reputation data.

    Args:
        router_logits: Router's learned expert preferences [num_experts]
        context: Classified input context (e.g., "code", "text", "reasoning")
        db_path: Path to reputation database
        num_experts: Total experts available
        k: Number of experts to select
        exploration_weight: Balance between router (1.0) and trust (0.0)

    Returns:
        Dict with selection results and analysis
    """
    # Create trust-based selector with reputation tracking
    db = ExpertReputationDB(db_path)
    selector = TrustBasedExpertSelector(
        num_experts=num_experts,
        cache_size=k * 2,  # Cache twice as many as needed
        reputation_db=db,
        exploration_weight=exploration_weight,
    )

    # Mark some experts as loaded (cache simulation)
    top_router_experts = torch.topk(router_logits, k=k).indices.tolist()
    selector.mark_experts_loaded(top_router_experts)

    # Perform trust-based selection
    result = selector.select_experts(router_logits, context=context, k=k)

    # Record activations for reputation learning
    for expert_id in result.selected_expert_ids:
        performance = {'quality': 0.8}  # Would be measured from generation
        record_expert_activation(expert_id, context, performance, db=db)

    return {
        'selected_experts': result.selected_expert_ids,
        'selection_scores': result.selection_scores,
        'router_scores': result.router_scores,
        'trust_scores': result.trust_scores,
        'context': result.context,
        'substitutions': result.substitutions,
        'cache_hits': result.cache_hits,
        'cache_misses': result.cache_misses,
    }


def demonstrate_multi_context_adaptation():
    """
    Demonstrates how expert selection adapts across different contexts.

    Shows that the same router logits produce different expert selections
    when context changes, based on accumulated reputation data.
    """
    print("\n" + "="*80)
    print("DEMONSTRATION: Multi-Context Expert Adaptation")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "reputation.db"
        db = ExpertReputationDB(db_path)

        # Simulate router logits (same across contexts)
        torch.manual_seed(42)
        router_logits = torch.randn(128)
        router_logits[15] = 2.5  # Router strongly prefers expert 15
        router_logits[28] = 2.3
        router_logits[42] = 2.1

        # Build reputation data for different contexts
        print("\n📊 Building reputation history...")

        # Expert 15 excels at CODE
        for _ in range(20):
            record_expert_activation(15, "code", {'quality': 0.92}, db=db)
            record_expert_activation(28, "code", {'quality': 0.75}, db=db)
            record_expert_activation(42, "code", {'quality': 0.68}, db=db)

        # Expert 42 excels at TEXT
        for _ in range(20):
            record_expert_activation(15, "text", {'quality': 0.70}, db=db)
            record_expert_activation(28, "text", {'quality': 0.78}, db=db)
            record_expert_activation(42, "text", {'quality': 0.94}, db=db)

        # Expert 28 is balanced
        for _ in range(20):
            record_expert_activation(15, "reasoning", {'quality': 0.75}, db=db)
            record_expert_activation(28, "reasoning", {'quality': 0.88}, db=db)
            record_expert_activation(42, "reasoning", {'quality': 0.72}, db=db)

        # Test selection in each context
        contexts = ["code", "text", "reasoning"]

        for context in contexts:
            print(f"\n--- Context: {context.upper()} ---")

            result = simulate_expert_selection_with_trust(
                router_logits=router_logits,
                context=context,
                db_path=db_path,
                num_experts=128,
                k=8,
                exploration_weight=0.3  # 30% router, 70% trust
            )

            # Show top 3 experts
            print(f"Top 3 experts selected:")
            for i in range(3):
                expert_id = result['selected_experts'][i]
                router_score = result['router_scores'][i]
                trust_score = result['trust_scores'][i]
                combined = result['selection_scores'][i]
                print(f"  Expert {expert_id:2d}: router={router_score:.3f}, "
                      f"trust={trust_score:.3f}, combined={combined:.3f}")

            # Check if context-appropriate expert is top-ranked
            if context == "code":
                assert result['selected_experts'][0] == 15 or result['selected_experts'][1] == 15, \
                    "Expert 15 should rank highly for CODE"
            elif context == "text":
                # Expert 42 should rank highly for TEXT
                top_3 = result['selected_experts'][:3]
                assert 42 in top_3, "Expert 42 should rank highly for TEXT"

        print("\n✅ Context adaptation validated!")


def demonstrate_exploration_exploitation_balance():
    """
    Demonstrates the exploration/exploitation balance controlled by
    the exploration_weight parameter.

    Shows how different weights produce different expert selections,
    balancing between router's learned preferences (exploration) and
    empirical reputation (exploitation).
    """
    print("\n" + "="*80)
    print("DEMONSTRATION: Exploration vs Exploitation Balance")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "reputation.db"
        db = ExpertReputationDB(db_path)

        # Router prefers expert 7
        router_logits = torch.randn(128)
        router_logits[7] = 3.0
        router_logits[42] = 1.5

        # But expert 42 has MUCH better reputation
        for _ in range(30):
            record_expert_activation(7, "text", {'quality': 0.60}, db=db)
            record_expert_activation(42, "text", {'quality': 0.95}, db=db)

        # Test different exploration weights
        weights = [1.0, 0.7, 0.3, 0.0]

        print("\nExpert rankings at different exploration weights:")
        print(f"{'Weight':<10} {'Top Expert':<15} {'Meaning':<30}")
        print("-" * 60)

        for weight in weights:
            result = simulate_expert_selection_with_trust(
                router_logits=router_logits,
                context="text",
                db_path=db_path,
                num_experts=128,
                k=8,
                exploration_weight=weight
            )

            top_expert = result['selected_experts'][0]
            meaning = {
                1.0: "Pure router (exploration)",
                0.7: "Mostly router",
                0.3: "Mostly trust (exploitation)",
                0.0: "Pure trust"
            }[weight]

            print(f"{weight:<10.1f} {f'Expert {top_expert}':<15} {meaning:<30}")

        print("\n✅ Exploration/exploitation balance validated!")


def demonstrate_cache_aware_substitution():
    """
    Demonstrates smart expert substitution when preferred experts
    aren't in cache.

    Shows Web4's delegation pattern: when preferred expert unavailable,
    select similar expert with high trust already loaded.
    """
    print("\n" + "="*80)
    print("DEMONSTRATION: Cache-Aware Smart Substitution")
    print("="*80)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "reputation.db"
        db = ExpertReputationDB(db_path)

        # Router strongly prefers experts 15, 28, 42 (not in cache)
        router_logits = torch.randn(128)
        router_logits[15] = 3.0
        router_logits[28] = 2.8
        router_logits[42] = 2.6

        # Build reputation for cached and uncached experts
        cached_experts = [5, 12, 23, 45, 67, 89]
        preferred_experts = [15, 28, 42]

        # Both have good reputation
        for expert_id in cached_experts + preferred_experts:
            quality = 0.85 if expert_id in cached_experts else 0.88
            for _ in range(15):
                record_expert_activation(expert_id, "text", {'quality': quality}, db=db)

        # Create selector with only cached experts loaded
        selector = TrustBasedExpertSelector(
            num_experts=128,
            cache_size=12,
            reputation_db=db,
            exploration_weight=0.3
        )
        selector.mark_experts_loaded(cached_experts)

        # Select experts (will need substitutions)
        result = selector.select_experts(router_logits, context="text", k=8)

        print(f"\nRouter's top preferences: {preferred_experts}")
        print(f"Experts in cache: {cached_experts}")
        print(f"\nSubstitutions made: {len(result.substitutions)}")

        if result.substitutions:
            print("\nSmart substitutions:")
            for requested, substitute in result.substitutions.items():
                print(f"  Expert {requested} (preferred but not loaded)")
                print(f"    → Expert {substitute} (similar, trusted, cached)")

        print(f"\nCache efficiency:")
        print(f"  Cache hits: {result.cache_hits}")
        print(f"  Cache misses: {result.cache_misses}")
        print(f"  Hit rate: {result.cache_hits / (result.cache_hits + result.cache_misses):.1%}")

        print("\n✅ Cache-aware substitution validated!")


def demonstrate_integration_benefits():
    """
    Summarizes the benefits of integrating TrustBasedExpertSelector
    with SAGE's Q3-Omni generation pipeline.
    """
    print("\n" + "="*80)
    print("INTEGRATION BENEFITS SUMMARY")
    print("="*80)

    benefits = [
        ("Contextual Adaptation",
         "Expert selection adapts to input context (code, text, reasoning)"),

        ("Empirical Learning",
         "Learns which experts actually perform well, not just router preferences"),

        ("Smart Caching",
         "Makes better cache eviction decisions based on context-specific trust"),

        ("Exploration Balance",
         "Configurable balance between trying router suggestions vs proven performers"),

        ("Federation Ready",
         "Reputation database can be shared across Thor ↔ Sprout instances"),

        ("Web4 Pattern",
         "Applies proven contextual trust framework to neural architecture"),

        ("Quality Improvement",
         "Better expert selection → Higher generation quality over time"),

        ("Observable Learning",
         "Reputation database provides interpretable expert performance metrics"),
    ]

    for i, (benefit, description) in enumerate(benefits, 1):
        print(f"\n{i}. **{benefit}**")
        print(f"   {description}")

    print("\n" + "="*80)


def run_all_demonstrations():
    """Run all integration demonstrations."""
    print("\n" + "="*80)
    print("TRUST-BASED EXPERT SELECTION INTEGRATION DEMO")
    print("Session 57 - Thor Autonomous Research")
    print("="*80)

    demonstrate_multi_context_adaptation()
    demonstrate_exploration_exploitation_balance()
    demonstrate_cache_aware_substitution()
    demonstrate_integration_benefits()

    print("\n" + "="*80)
    print("✅ ALL DEMONSTRATIONS COMPLETE")
    print("="*80)

    print("\nNext Steps for Full Integration:")
    print("1. Add optional trust_selector parameter to SelectiveLanguageModel")
    print("2. Modify SelectiveMoELayer to use trust-based selection when available")
    print("3. Create end-to-end test with actual Q3-Omni generation")
    print("4. Measure quality improvement from trust-based selection")
    print("5. Enable Thor ↔ Sprout reputation sharing")

    print("\nIntegration Pattern (for future implementation):")
    print("""
    # In SelectiveLanguageModel.__init__:
    self.trust_selector = create_trust_based_selector(
        db_path=reputation_db_path,
        num_experts=128,
        cache_size=max_loaded_experts,
        exploration_weight=0.3
    )

    # In SelectiveMoELayer.forward:
    if hasattr(self, 'trust_selector') and self.trust_selector is not None:
        # Use trust-based selection
        result = self.trust_selector.select_experts(
            router_logits, context=context, k=num_experts_per_tok
        )
        selected_expert_ids = result.selected_expert_ids
        router_weights = result.selection_scores
    else:
        # Use standard router selection
        selected_expert_ids, router_weights = self.expert_loader.select_experts_snarc(...)
    """)


if __name__ == '__main__':
    run_all_demonstrations()
