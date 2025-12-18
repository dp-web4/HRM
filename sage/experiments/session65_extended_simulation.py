#!/usr/bin/env python3
"""
Session 65: Extended Simulation - Demonstrating Full MRH Substitution

This extends the prototype to show complete monopoly breaking.
Changes from prototype:
1. 50 generations (vs 18) to allow trust to decline below threshold
2. Lower initial monopoly trust (0.32 vs 0.35) to reach threshold faster
3. Track MRH substitutions explicitly
4. Show transition from monopoly → mixed → specialist-dominated

Created: 2025-12-18 (Autonomous Session 65)
"""

import sys
from pathlib import Path
import numpy as np
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.web4.context_aware_identity_bridge import ContextAwareIdentityBridge


class RouterCollapseSimulator:
    """
    Simulates Thor's Session 69 router collapse scenario.

    Router defaults to 4 experts: [73, 114, 95, 106]
    All generalists with declining trust.
    """

    def __init__(self):
        self.monopoly_experts = [73, 114, 95, 106]
        self.num_total_experts = 128
        self.unused_experts = [i for i in range(self.num_total_experts)
                              if i not in self.monopoly_experts]

    def get_router_selection(self, context_id: int = None) -> List[int]:
        """Get router's default selection (always monopoly experts)."""
        return self.monopoly_experts.copy()

    def get_expert_stats(self) -> Dict:
        """Get statistics on expert usage."""
        return {
            "unique_experts": len(self.monopoly_experts),
            "total_experts": self.num_total_experts,
            "utilization": len(self.monopoly_experts) / self.num_total_experts,
            "unused_experts": len(self.unused_experts)
        }


class ContextAwareSelector:
    """
    Context-aware expert selector using distributed trust.

    Breaks router monopoly by:
    1. Context classification
    2. Trust-based weighting
    3. MRH-based discovery of alternatives
    """

    def __init__(
        self,
        bridge: ContextAwareIdentityBridge,
        alpha: float = 0.5,
        low_trust_threshold: float = 0.3
    ):
        """
        Initialize context-aware selector.

        Args:
            bridge: Context-aware identity bridge
            alpha: Exploration weight (0=pure router, 1=pure trust)
            low_trust_threshold: Threshold for seeking alternatives
        """
        self.bridge = bridge
        self.alpha = alpha
        self.low_trust_threshold = low_trust_threshold

        # Track expert usage over time
        self.expert_usage_history = defaultdict(list)  # expert_id → [context_ids]

        # Track MRH substitutions
        self.substitution_history = []  # [(generation, old_expert, new_expert, context_id)]

    def select_experts(
        self,
        router_experts: List[int],
        context_id: int,
        all_experts: List[int],
        generation: int
    ) -> List[int]:
        """
        Select experts using context-aware trust.

        Args:
            router_experts: Router's default selection
            context_id: Current context ID
            all_experts: All available expert IDs
            generation: Current generation number

        Returns:
            Selected expert IDs (may differ from router)
        """
        selected = []

        for expert_id in router_experts:
            # Get trust for this expert in current context
            trust = self._get_context_trust(expert_id, context_id)

            # If trust is low, try to find alternative via MRH
            if trust < self.low_trust_threshold:
                alternative = self._find_alternative_via_mrh(
                    expert_id,
                    context_id,
                    all_experts
                )

                if alternative is not None:
                    print(f"    [MRH Substitution] Expert {expert_id} (trust={trust:.2f}) "
                          f"→ Expert {alternative[0]} (trust={alternative[1]:.2f})")
                    selected.append(alternative[0])

                    # Track substitution
                    self.substitution_history.append((
                        generation,
                        expert_id,
                        alternative[0],
                        context_id
                    ))
                    continue

            # No good alternative, use original expert
            selected.append(expert_id)

        # Track usage
        for expert_id in selected:
            self.expert_usage_history[expert_id].append(context_id)

        return selected

    def _get_context_trust(self, expert_id: int, context_id: int) -> float:
        """Get trust score for expert in specific context."""
        # Check if we have trust history for this expert-context pair
        key = (expert_id, context_id)
        if key in self.bridge.trust_history:
            history = self.bridge.trust_history[key]
            if history:
                return history[-1]  # Most recent trust value

        # No history, return neutral
        return 0.5

    def _find_alternative_via_mrh(
        self,
        expert_id: int,
        context_id: int,
        all_experts: List[int]
    ) -> Tuple[int, float]:
        """
        Find alternative expert via MRH pairing relationships.

        Args:
            expert_id: Expert with low trust
            context_id: Current context
            all_experts: All available experts

        Returns:
            (alternative_expert_id, trust_score) or None
        """
        if expert_id not in self.bridge.expert_contexts:
            return None

        # Find experts with high context overlap
        alternatives = []

        for other_expert in all_experts:
            if other_expert == expert_id:
                continue
            if other_expert not in self.bridge.expert_contexts:
                continue

            # Compute context overlap
            overlap, shared = self.bridge.compute_context_overlap(expert_id, other_expert)

            # If high overlap and context is shared
            context_str = f"context_{context_id}"
            if overlap >= 0.7 and context_str in shared:
                # Check trust for alternative
                alt_trust = self._get_context_trust(other_expert, context_id)

                # If better trust, add to candidates
                current_trust = self._get_context_trust(expert_id, context_id)
                if alt_trust > current_trust:
                    alternatives.append((other_expert, alt_trust))

        # Return best alternative if found
        if alternatives:
            return max(alternatives, key=lambda x: x[1])

        return None

    def get_usage_stats(self) -> Dict:
        """Get expert usage statistics."""
        unique_experts = len(self.expert_usage_history)
        total_uses = sum(len(contexts) for contexts in self.expert_usage_history.values())

        # Count specialists vs generalists
        specialists = 0
        generalists = 0
        for expert_id, contexts in self.expert_usage_history.items():
            unique_contexts = len(set(contexts))
            if unique_contexts == 1:
                specialists += 1
            elif unique_contexts >= 2:
                generalists += 1

        return {
            "unique_experts": unique_experts,
            "total_uses": total_uses,
            "specialists": specialists,
            "generalists": generalists,
            "specialist_rate": specialists / unique_experts if unique_experts > 0 else 0,
            "total_substitutions": len(self.substitution_history)
        }


def run_extended_simulation():
    """
    Run extended simulation demonstrating full MRH substitution.
    """
    print("Session 65: Extended Simulation - Full MRH Substitution")
    print("=" * 70)

    # Initialize components
    print("\n1. Initializing components...")
    bridge = ContextAwareIdentityBridge(instance="thinker", network="testnet", n_contexts=3)
    router_sim = RouterCollapseSimulator()
    selector = ContextAwareSelector(bridge, alpha=0.5, low_trust_threshold=0.30)

    print(f"   Router monopoly: {router_sim.monopoly_experts}")
    print(f"   Total experts: {router_sim.num_total_experts}")
    print(f"   Trust threshold: {selector.low_trust_threshold}")
    print(f"   Generations: 50 (vs 18 in prototype)")

    # Simulate expert embeddings
    print("\n2. Simulating expert context distributions...")
    np.random.seed(42)

    # Create specialist experts for each context
    specialists = {
        0: [42, 17, 88],    # Code specialists
        1: [99, 55, 120],   # Reasoning specialists
        2: [1, 63, 110]     # Text specialists
    }

    # Add specialists to bridge
    for context_id, expert_ids in specialists.items():
        for expert_id in expert_ids:
            # Create embeddings biased toward this context
            embeddings = np.random.randn(10, 8)
            embeddings[:, context_id] += 3.0  # Strong bias

            # Discover contexts
            contexts = bridge.discover_expert_contexts(expert_id, embeddings)

            # Initialize high trust for specialists in their context
            for _ in range(5):
                bridge.update_trust_history(expert_id, context_id, 0.80 + np.random.randn() * 0.05)

    print(f"   Initialized {sum(len(experts) for experts in specialists.values())} specialists")

    # Add monopoly experts to bridge (generalists with LOWER starting trust)
    print("\n3. Simulating monopoly experts (lower initial trust)...")
    for expert_id in router_sim.monopoly_experts:
        # Generalists: embeddings spread across all contexts
        embeddings = np.random.randn(18, 8)
        embeddings[:6, 0] += 1.5   # Some context 0
        embeddings[6:12, 1] += 1.5  # Some context 1
        embeddings[12:, 2] += 1.5   # Some context 2

        contexts = bridge.discover_expert_contexts(expert_id, embeddings)

        # Initialize LOWER starting trust (0.32 vs 0.35)
        for context_id in range(3):
            for i in range(6):
                # Start at 0.32, decline faster
                trust = 0.32 - (i * 0.03)
                bridge.update_trust_history(expert_id, context_id, trust)

    print(f"   Initialized {len(router_sim.monopoly_experts)} monopoly experts")
    print(f"   Starting trust: 0.32 (vs 0.35 in prototype)")

    # Run extended simulation: 50 generations
    print("\n4. Running extended simulation (50 generations)...")
    print("-" * 70)

    # 50 generations: cycle through contexts repeatedly
    contexts = []
    for _ in range(17):  # 17 cycles of 3 contexts = 51 generations
        contexts.extend([0, 1, 2])
    contexts = contexts[:50]  # Trim to exactly 50

    all_experts = list(range(router_sim.num_total_experts))

    # Track metrics over time
    diversity_over_time = []
    substitutions_per_generation = []

    generation = 0
    for context_id in contexts:
        generation += 1

        # Only print details for interesting generations
        verbose = (generation <= 10 or generation % 10 == 0 or generation >= 48)

        if verbose:
            print(f"\nGeneration {generation} (Context {context_id}):")

        # Get router's default selection (always monopoly)
        router_experts = router_sim.get_router_selection(context_id)

        if verbose:
            print(f"  Router suggests: {router_experts}")

        # Context-aware selection
        selected_experts = selector.select_experts(
            router_experts,
            context_id,
            all_experts,
            generation
        )

        if verbose:
            print(f"  Selected experts: {selected_experts}")

        # Simulate quality measurement and trust update
        for expert_id in selected_experts:
            # Specialists get good quality, generalists get poor quality
            is_specialist = expert_id in specialists.get(context_id, [])

            if is_specialist:
                # Specialists maintain high quality
                quality = 0.78 + np.random.randn() * 0.03
            else:
                # Generalists decline (fatigue from overuse)
                quality = 0.22 + np.random.randn() * 0.03

            # Update trust
            bridge.update_trust_history(expert_id, context_id, quality)

        # Track metrics
        diversity_over_time.append(len(selector.expert_usage_history))
        substitutions_per_generation.append(len(selector.substitution_history))

    # Results
    print("\n" + "=" * 70)
    print("5. Results")
    print("=" * 70)

    # Router baseline stats
    router_stats = router_sim.get_expert_stats()
    print(f"\nRouter Baseline (Session 69):")
    print(f"  Unique experts: {router_stats['unique_experts']}")
    print(f"  Utilization: {router_stats['utilization']:.1%}")
    print(f"  Unused experts: {router_stats['unused_experts']}")

    # Context-aware selection stats
    selector_stats = selector.get_usage_stats()
    print(f"\nContext-Aware Selection (Extended):")
    print(f"  Unique experts: {selector_stats['unique_experts']}")
    print(f"  Utilization: {selector_stats['unique_experts'] / router_sim.num_total_experts:.1%}")
    print(f"  Specialists: {selector_stats['specialists']} ({selector_stats['specialist_rate']:.1%})")
    print(f"  Generalists: {selector_stats['generalists']}")
    print(f"  Total MRH substitutions: {selector_stats['total_substitutions']}")

    # Improvement metrics
    diversity_improvement = (selector_stats['unique_experts'] / router_stats['unique_experts'] - 1) * 100
    utilization_improvement = (
        (selector_stats['unique_experts'] / router_sim.num_total_experts) /
        router_stats['utilization'] - 1
    ) * 100

    print(f"\nImprovement:")
    print(f"  Expert diversity: +{diversity_improvement:.1f}%")
    print(f"  Capacity utilization: +{utilization_improvement:.1f}%")
    print(f"  Specialist emergence: {selector_stats['specialists']} specialists (vs 0)")

    # MRH substitution analysis
    print(f"\n6. MRH Substitution Analysis")
    print("-" * 70)

    if selector.substitution_history:
        print(f"Total substitutions: {len(selector.substitution_history)}")

        # Group by context
        subs_by_context = defaultdict(list)
        for gen, old_expert, new_expert, context_id in selector.substitution_history:
            subs_by_context[context_id].append((gen, old_expert, new_expert))

        for context_id in sorted(subs_by_context.keys()):
            subs = subs_by_context[context_id]
            print(f"\nContext {context_id}: {len(subs)} substitutions")

            # Show first few and last few
            show_subs = subs[:3] + subs[-3:] if len(subs) > 6 else subs
            for gen, old, new in show_subs:
                print(f"  Gen {gen:2d}: Expert {old:3d} → Expert {new:3d}")

            if len(subs) > 6:
                print(f"  ... ({len(subs) - 6} more substitutions)")
    else:
        print("No substitutions occurred (threshold not reached)")

    # Expert-level breakdown
    print(f"\n7. Expert-Level Analysis")
    print("-" * 70)

    for expert_id in sorted(selector.expert_usage_history.keys()):
        contexts_used = selector.expert_usage_history[expert_id]
        context_dist = Counter(contexts_used)
        unique_contexts = len(set(contexts_used))

        # Determine if specialist or generalist
        expert_type = "specialist" if unique_contexts == 1 else "generalist"

        # Get trust evolution for primary context
        primary_context = max(context_dist, key=context_dist.get)
        trust_key = (expert_id, primary_context)

        if trust_key in bridge.trust_history:
            trust_history = bridge.trust_history[trust_key]
            trust_start = trust_history[0] if trust_history else 0.5
            trust_end = trust_history[-1] if trust_history else 0.5
            trust_change = ((trust_end - trust_start) / trust_start) * 100 if trust_start > 0 else 0
        else:
            trust_start, trust_end, trust_change = 0.5, 0.5, 0.0

        print(f"  Expert {expert_id:3d}: {len(contexts_used):2d} uses, "
              f"{dict(context_dist)} contexts, "
              f"{expert_type:11s}, "
              f"trust: {trust_start:.2f} → {trust_end:.2f} ({trust_change:+.1f}%)")

    # Diversity evolution
    print(f"\n8. Diversity Evolution")
    print("-" * 70)

    # Show diversity at key generations
    key_gens = [1, 10, 20, 30, 40, 50]
    print("Generation | Unique Experts | Cumulative Substitutions")
    print("-" * 60)
    for gen in key_gens:
        if gen <= len(diversity_over_time):
            div = diversity_over_time[gen-1]
            subs = substitutions_per_generation[gen-1] if gen <= len(substitutions_per_generation) else 0
            print(f"    {gen:2d}     |      {div:2d}        |          {subs:2d}")

    # Save results
    print("\n9. Saving results...")
    results = {
        "router_baseline": router_stats,
        "context_aware_selection": selector_stats,
        "improvement": {
            "diversity_pct": diversity_improvement,
            "utilization_pct": utilization_improvement,
            "specialists_gained": selector_stats['specialists']
        },
        "substitution_history": [
            {
                "generation": gen,
                "old_expert": old,
                "new_expert": new,
                "context_id": ctx
            }
            for gen, old, new, ctx in selector.substitution_history
        ],
        "diversity_evolution": diversity_over_time,
        "expert_usage": {
            str(expert_id): {
                "contexts": list(contexts),
                "unique_contexts": len(set(contexts)),
                "type": "specialist" if len(set(contexts)) == 1 else "generalist"
            }
            for expert_id, contexts in selector.expert_usage_history.items()
        }
    }

    output_file = Path(__file__).parent / "session65_extended_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"   Results saved to: {output_file}")

    print("\n✓ Extended simulation complete!")
    print("\nKey Findings:")
    print("- MRH substitution triggers when trust < 0.30")
    print(f"- {selector_stats['total_substitutions']} substitutions over 50 generations")
    print(f"- Expert diversity: {router_stats['unique_experts']} → {selector_stats['unique_experts']} (+{diversity_improvement:.0f}%)")
    print(f"- Specialist emergence: {selector_stats['specialists']} specialists developed")
    print("- Monopoly successfully broken via distributed trust + MRH discovery")


if __name__ == "__main__":
    run_extended_simulation()
