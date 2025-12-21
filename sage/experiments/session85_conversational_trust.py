#!/usr/bin/env python3
"""
Session 85: Conversational Trust Integration

Bridges Sprout's Session 84 conversational ground truth with Thor's Sessions 74-83
trust-first architecture.

Problem:
- Sessions 74-83 validated trust-first architecture with internal metrics
- Session 84 (Sprout) discovered conversational repair signals provide ground truth
- Need to integrate real-world relationship quality into expert trust tracking

Solution: Conversational-Aware Trust Selector

Key Concepts (from Session 84):
1. **Repair Signals**: engagement, reassurance, abandonment, correction
2. **Meta-Cognitive Leaks**: Introspective reasoning leaking into responses
3. **Repair Arc**: Early difficulty → middle persistence → late resolution
4. **Relationship Quality**: Not just "correct answer" but "meaningful interaction"

Architecture:
- Extend TrustFirstMRHSelector with conversational quality tracking
- Track expert performance on relationship metrics (not just internal metrics)
- Detect repair arcs and adjust trust accordingly
- Integrate Sprout's ground truth into Thor's trust-first framework

Based on:
- Sessions 74-83: Trust-first MoE architecture (Thor)
- Session 84: Conversational ground truth analysis (Sprout)
- Principle: Real-world feedback > Internal metrics

Created: 2025-12-21 (Thor Session 85)
Author: Thor (Autonomous SAGE Research)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import time
from collections import defaultdict
from enum import Enum

# Import Sessions 74-83 architecture
from sage.core.trust_first_mrh_selector import TrustFirstMRHSelector, TrustFirstSelectionResult
from sage.core.context_classifier import ContextClassifier


class RepairSignalType(Enum):
    """Conversational repair signal types (from Session 84)."""
    ENGAGEMENT = "engagement"        # Follow-up questions, interest
    REASSURANCE = "reassurance"      # Emotional support, encouragement
    ABANDONMENT = "abandonment"      # Short responses, topic dropped
    CORRECTION = "correction"        # Explicit rejection of response


@dataclass
class RepairSignal:
    """Conversational repair signal (Session 84)."""
    signal_type: RepairSignalType
    confidence: float  # 0.0-1.0
    turn_number: int
    context: Optional[str] = None


@dataclass
class ConversationalQuality:
    """
    Conversational quality metrics (Session 84 ground truth).

    Tracks relationship quality, not just response accuracy.
    """
    repair_signals: List[RepairSignal]
    meta_cognitive_leaks: int  # Count of introspective leaks
    arc_pattern: Optional[str] = None  # e.g., "REPAIR_ARC", "SMOOTH", "DEGRADED"
    relationship_score: float = 0.5  # 0.0-1.0, derived from signals

    def compute_relationship_score(self) -> float:
        """
        Compute relationship quality score from repair signals.

        Logic (from Session 84):
        - Engagement: +0.2 per signal (weighted by confidence)
        - Reassurance: +0.3 per signal (highest value)
        - Abandonment: -0.2 per signal
        - Correction: -0.4 per signal (explicit failure)
        """
        if not self.repair_signals:
            return 0.5  # Neutral

        score = 0.5  # Start neutral

        for signal in self.repair_signals:
            if signal.signal_type == RepairSignalType.ENGAGEMENT:
                score += 0.2 * signal.confidence
            elif signal.signal_type == RepairSignalType.REASSURANCE:
                score += 0.3 * signal.confidence
            elif signal.signal_type == RepairSignalType.ABANDONMENT:
                score -= 0.2 * signal.confidence
            elif signal.signal_type == RepairSignalType.CORRECTION:
                score -= 0.4 * signal.confidence

        # Normalize to [0, 1]
        return np.clip(score, 0.0, 1.0)


@dataclass
class ConversationalTrustStats:
    """Statistics for conversational trust tracking."""
    total_conversational_updates: int = 0
    total_internal_updates: int = 0
    repair_signals_received: int = 0
    relationship_scores: List[float] = None

    def __post_init__(self):
        if self.relationship_scores is None:
            self.relationship_scores = []


class ConversationalTrustFirstSelector(TrustFirstMRHSelector):
    """
    Trust-first selector with conversational ground truth integration.

    Extends Session 83 architecture with Session 84 conversational quality tracking.
    """

    def __init__(
        self,
        num_experts: int = 128,
        min_trust_evidence: int = 2,
        epsilon: float = 0.2,
        low_trust_threshold: float = 0.3,
        component: str = "thinker",
        network: str = "testnet",
        # Conversational parameters (new)
        conversational_weight: float = 0.4,  # Balance: 60% internal, 40% conversational
        enable_conversational: bool = True
    ):
        """
        Initialize conversational trust-first selector.

        Args:
            (Session 82 params): num_experts, min_trust_evidence, epsilon, etc.
            conversational_weight: Weight for conversational quality (0.0-1.0)
            enable_conversational: Enable/disable conversational tracking
        """
        # Initialize base trust-first selector (Sessions 74-83)
        super().__init__(
            num_experts=num_experts,
            min_trust_evidence=min_trust_evidence,
            low_trust_threshold=low_trust_threshold,
            epsilon=epsilon,
            component=component,
            network=network
        )

        # Conversational trust tracking
        self.conversational_weight = conversational_weight
        self.enable_conversational = enable_conversational
        self.conversational_stats = ConversationalTrustStats()

        # Track conversational quality per expert-context
        # Key: (expert_id, context_id), Value: ConversationalQuality
        self.conversational_quality: Dict[Tuple[int, str], ConversationalQuality] = {}

    def update_trust_with_conversation(
        self,
        expert_id: int,
        context: str,
        internal_quality: float,
        conversational_quality: ConversationalQuality
    ):
        """
        Update trust with both internal metrics AND conversational ground truth.

        This is the key integration: Session 84's repair signals enhance
        Sessions 74-83's trust tracking.

        Args:
            expert_id: Expert ID (0-127)
            context: Context string
            internal_quality: Internal quality score (0.0-1.0)
            conversational_quality: Conversational ground truth (Session 84)
        """
        # Compute relationship score from repair signals
        relationship_score = conversational_quality.compute_relationship_score()

        # Blend internal and conversational quality
        if self.enable_conversational:
            blended_quality = (
                (1 - self.conversational_weight) * internal_quality +
                self.conversational_weight * relationship_score
            )
        else:
            blended_quality = internal_quality

        # Update trust via base selector (Session 82 validated approach)
        super().update_trust_for_expert(expert_id, context, blended_quality)

        # Track conversational quality
        key = (expert_id, context)
        self.conversational_quality[key] = conversational_quality

        # Statistics
        self.conversational_stats.total_conversational_updates += 1
        self.conversational_stats.repair_signals_received += len(conversational_quality.repair_signals)
        self.conversational_stats.relationship_scores.append(relationship_score)

    def update_trust_for_expert(
        self,
        expert_id: int,
        context: str,
        quality: float
    ):
        """
        Update trust (internal metrics only, for backward compatibility).

        Use update_trust_with_conversation() for conversational-aware updates.
        """
        super().update_trust_for_expert(expert_id, context, quality)
        self.conversational_stats.total_internal_updates += 1

    def get_conversational_stats(self) -> Dict:
        """Get conversational trust statistics."""
        stats = asdict(self.conversational_stats)
        stats["enable_conversational"] = self.enable_conversational
        stats["conversational_weight"] = self.conversational_weight
        stats["avg_relationship_score"] = (
            np.mean(self.conversational_stats.relationship_scores)
            if self.conversational_stats.relationship_scores else 0.5
        )
        return stats


def simulate_repair_signals(
    turn_number: int,
    internal_quality: float,
    seed: int
) -> List[RepairSignal]:
    """
    Simulate conversational repair signals based on response quality.

    Logic (inspired by Session 84):
    - High quality (>0.7): Likely engagement/reassurance
    - Medium quality (0.4-0.7): Mixed signals
    - Low quality (<0.4): Likely abandonment/correction

    Args:
        turn_number: Turn in conversation
        internal_quality: Internal quality score
        seed: Random seed

    Returns:
        List of repair signals
    """
    np.random.seed(seed)
    signals = []

    # High quality → positive signals
    if internal_quality > 0.7:
        if np.random.random() < 0.7:
            signals.append(RepairSignal(
                signal_type=RepairSignalType.ENGAGEMENT,
                confidence=0.6 + 0.3 * np.random.random(),
                turn_number=turn_number
            ))
        if np.random.random() < 0.4:
            signals.append(RepairSignal(
                signal_type=RepairSignalType.REASSURANCE,
                confidence=0.7 + 0.2 * np.random.random(),
                turn_number=turn_number
            ))

    # Low quality → negative signals
    elif internal_quality < 0.4:
        if np.random.random() < 0.5:
            signals.append(RepairSignal(
                signal_type=RepairSignalType.ABANDONMENT,
                confidence=0.3 + 0.3 * np.random.random(),
                turn_number=turn_number
            ))
        if np.random.random() < 0.3:
            signals.append(RepairSignal(
                signal_type=RepairSignalType.CORRECTION,
                confidence=0.5 + 0.4 * np.random.random(),
                turn_number=turn_number
            ))

    # Medium quality → sparse positive signals
    else:
        if np.random.random() < 0.3:
            signals.append(RepairSignal(
                signal_type=RepairSignalType.ENGAGEMENT,
                confidence=0.4 + 0.3 * np.random.random(),
                turn_number=turn_number
            ))

    return signals


def run_session85_conversational_trust_test(
    num_experts: int = 128,
    num_generations: int = 90,
    num_sequences: int = 9,
    min_trust_evidence: int = 2,
    epsilon: float = 0.2,
    conversational_weight: float = 0.4,
    seed: int = 42
):
    """
    Test conversational-aware trust tracking.

    Scenario:
    1. Two selectors: WITH conversational vs WITHOUT conversational
    2. Simulate expert performance with internal quality
    3. Simulate conversational repair signals (Session 84 style)
    4. Compare: Does conversational ground truth improve expert selection?

    Hypothesis: Conversational quality should improve trust_driven activation.

    Args:
        num_experts: Number of experts (128 for Q3-Omni)
        num_generations: Number of generations
        num_sequences: Number of distinct sequences
        min_trust_evidence: Minimum observations for trust (Session 78: 2)
        epsilon: Epsilon-greedy exploration (Session 77: 0.2)
        conversational_weight: Weight for conversational quality
        seed: Random seed

    Returns:
        Results dictionary
    """
    np.random.seed(seed)

    print("\n" + "="*80)
    print("SESSION 85: CONVERSATIONAL TRUST INTEGRATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Experts: {num_experts}")
    print(f"  Generations: {num_generations}")
    print(f"  Min trust evidence: {min_trust_evidence} (Session 78 optimal)")
    print(f"  Epsilon: {epsilon} (Session 77 optimal)")
    print(f"  Conversational weight: {conversational_weight} (40% conversation, 60% internal)")
    print()

    # Create selectors
    # WITH conversational
    conv_selector = ConversationalTrustFirstSelector(
        num_experts=num_experts,
        min_trust_evidence=min_trust_evidence,
        epsilon=epsilon,
        component="thinker_layer0",
        network="testnet",
        conversational_weight=conversational_weight,
        enable_conversational=True
    )

    # WITHOUT conversational (baseline)
    baseline_selector = TrustFirstMRHSelector(
        num_experts=num_experts,
        min_trust_evidence=min_trust_evidence,
        epsilon=epsilon,
        component="thinker_layer0",
        network="testnet"
    )

    print("Selectors:")
    print(f"  Conversational: Blends internal (60%) + relationship quality (40%)")
    print(f"  Baseline: Internal metrics only (Sessions 74-83)")
    print()

    # Create context classifier
    context_classifier = ContextClassifier(num_contexts=3)
    conv_selector.context_classifier = context_classifier
    baseline_selector.context_classifier = context_classifier

    # Fit classifier
    sequences = [
        "def sort_array(arr):",
        "class BinaryTree:",
        "function process() {",
        "Given the premises, prove:",
        "If A implies B and B implies C,",
        "Solve for x: 3x + 7 = 22",
        "The ancient civilization",
        "Climate change affects",
        "Modern architecture combines"
    ]

    embedding_dim = 256
    context_embeddings = []
    for i, seq in enumerate(sequences):
        base_vector = np.random.randn(embedding_dim).astype(np.float32)
        context_group = i // 3
        base_vector[:10] += context_group * 5.0
        emb = base_vector / np.linalg.norm(base_vector)
        context_embeddings.append(emb)

    context_classifier.fit(np.array(context_embeddings))

    print("="*80)
    print("SIMULATION: Trust-first with conversational ground truth")
    print("="*80)
    print()

    # Track results
    conv_results = {
        "trust_driven_gens": [],
        "mode_history": [],
        "experts_used": set(),
        "first_trust_driven_gen": None,
        "relationship_scores": []
    }

    baseline_results = {
        "trust_driven_gens": [],
        "mode_history": [],
        "experts_used": set(),
        "first_trust_driven_gen": None
    }

    # Simulate generations
    for gen in range(num_generations):
        seq_idx = gen % num_sequences
        sequence = sequences[seq_idx]
        context = context_classifier.classify(context_embeddings[seq_idx]).context_id

        # Simulate router logits
        np.random.seed(seed + gen)
        router_logits = np.random.randn(num_experts).astype(np.float32)

        # Simulate internal quality
        np.random.seed(seed + gen + 1000)
        internal_quality = np.clip(0.5 + 0.3 * np.random.randn(), 0.0, 1.0)

        # Conversational: Select and update with ground truth
        conv_result = conv_selector.select_experts(router_logits, context, k=8)

        # Simulate repair signals (Session 84 style)
        repair_signals = simulate_repair_signals(gen, internal_quality, seed + gen + 2000)
        conversational_quality = ConversationalQuality(
            repair_signals=repair_signals,
            meta_cognitive_leaks=0
        )
        relationship_score = conversational_quality.compute_relationship_score()

        conv_selector.update_trust_with_conversation(
            conv_result.selected_expert_ids[0],
            context,
            internal_quality,
            conversational_quality
        )

        # Baseline: Select and update (internal only)
        baseline_result = baseline_selector.select_experts(router_logits, context, k=8)
        baseline_selector.update_trust_for_expert(
            baseline_result.selected_expert_ids[0],
            context,
            internal_quality
        )

        # Track results
        conv_results["mode_history"].append(conv_result.selection_mode)
        conv_results["experts_used"].update(conv_result.selected_expert_ids)
        conv_results["relationship_scores"].append(relationship_score)
        if conv_result.selection_mode == "trust_driven":
            conv_results["trust_driven_gens"].append(gen)
            if conv_results["first_trust_driven_gen"] is None:
                conv_results["first_trust_driven_gen"] = gen

        baseline_results["mode_history"].append(baseline_result.selection_mode)
        baseline_results["experts_used"].update(baseline_result.selected_expert_ids)
        if baseline_result.selection_mode == "trust_driven":
            baseline_results["trust_driven_gens"].append(gen)
            if baseline_results["first_trust_driven_gen"] is None:
                baseline_results["first_trust_driven_gen"] = gen

    # Compute statistics
    def compute_stats(results, label):
        trust_driven_count = len(results["trust_driven_gens"])
        trust_driven_pct = (trust_driven_count / num_generations) * 100
        experts_used = len(results["experts_used"])
        expert_utilization_pct = (experts_used / num_experts) * 100
        first_activation = results["first_trust_driven_gen"]

        print(f"\n{label}:")
        print(f"  Trust_driven: {trust_driven_count}/{num_generations} ({trust_driven_pct:.1f}%)")
        print(f"  First activation: Gen {first_activation}")
        print(f"  Experts used: {experts_used}/{num_experts} ({expert_utilization_pct:.1f}%)")

        return {
            "trust_driven_count": trust_driven_count,
            "trust_driven_pct": trust_driven_pct,
            "experts_used": experts_used,
            "expert_utilization_pct": expert_utilization_pct,
            "first_activation": first_activation
        }

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    conv_stats = compute_stats(conv_results, "Conversational (Session 84 + Sessions 74-83)")
    baseline_stats = compute_stats(baseline_results, "Baseline (Sessions 74-83 only)")

    # Conversational statistics
    print(f"\nConversational Statistics:")
    conv_stats_detail = conv_selector.get_conversational_stats()
    print(f"  Conversational updates: {conv_stats_detail['total_conversational_updates']}")
    print(f"  Internal updates: {conv_stats_detail['total_internal_updates']}")
    print(f"  Repair signals received: {conv_stats_detail['repair_signals_received']}")
    print(f"  Avg relationship score: {conv_stats_detail['avg_relationship_score']:.3f}")

    # Compute conversational benefit
    conv_benefit = {
        "trust_driven_improvement": conv_stats["trust_driven_pct"] - baseline_stats["trust_driven_pct"],
        "first_activation_speedup": (baseline_stats["first_activation"] or num_generations) - (conv_stats["first_activation"] or num_generations),
        "expert_diversity_improvement": conv_stats["experts_used"] - baseline_stats["experts_used"]
    }

    print(f"\n" + "="*80)
    print("CONVERSATIONAL BENEFIT ANALYSIS")
    print("="*80)
    print(f"\nConversational vs Baseline:")
    print(f"  Trust_driven improvement: {conv_benefit['trust_driven_improvement']:+.1f}%")
    print(f"  First activation speedup: {conv_benefit['first_activation_speedup']:+d} generations")
    print(f"  Expert diversity improvement: {conv_benefit['expert_diversity_improvement']:+d} experts")

    # Conclusion
    print(f"\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if conv_benefit["trust_driven_improvement"] > 5.0:
        print("\n✅ CONVERSATIONAL INTEGRATION SUCCESS!")
        print("   Session 84 ground truth significantly improves trust-first architecture.")
    elif conv_benefit["trust_driven_improvement"] > 0:
        print("\n✅ CONVERSATIONAL BENEFIT DETECTED")
        print("   Modest improvement from relationship quality signals.")
    else:
        print("\n⚠️  NO CONVERSATIONAL BENEFIT")
        print("   Conversational signals may need tuning or different weighting.")

    print(f"\n   Insight: Blending internal metrics ({1-conversational_weight:.0%}) +")
    print(f"   relationship quality ({conversational_weight:.0%}) from Session 84 ground truth.")
    print()

    # Save results
    results = {
        "session": 85,
        "configuration": {
            "num_experts": num_experts,
            "num_generations": num_generations,
            "num_sequences": num_sequences,
            "min_trust_evidence": min_trust_evidence,
            "epsilon": epsilon,
            "conversational_weight": conversational_weight,
            "seed": seed
        },
        "conversational": {
            **conv_stats,
            "conversational_stats": conv_stats_detail
        },
        "baseline": baseline_stats,
        "conversational_benefit": conv_benefit
    }

    output_file = os.path.join(os.path.dirname(__file__), "session85_conversational_trust_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    return results


if __name__ == "__main__":
    start_time = time.time()

    results = run_session85_conversational_trust_test(
        num_experts=128,
        num_generations=90,
        num_sequences=9,
        min_trust_evidence=2,  # Session 78 optimal
        epsilon=0.2,  # Session 77 optimal
        conversational_weight=0.4,  # 40% conversation, 60% internal
        seed=42
    )

    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.1f}s")
    print()
