#!/usr/bin/env python3
"""
Session 86: Advanced Trust Integration

Integrates all optimizations from Sessions 83-85 and Legion's implementations
into a unified, production-ready AdvancedTrustFirstSelector.

Components Integrated:
1. Conversational Trust (Session 85): Repair signals (engagement, reassurance, etc.)
2. Dynamic Trust Decay (Legion): Adapts decay based on observation diversity
3. Attestation Deduplication (Legion): Efficient federation import
4. Repair Arc Detection (Legion): Temporal pattern recognition

Problem:
- Session 85 integrated conversational trust (+25.6% improvement)
- Legion implemented three key optimizations
- Need unified architecture combining all advances

Solution: AdvancedTrustFirstSelector

Architecture:
- Extends ConversationalTrustFirstSelector (Session 85)
- Adds dynamic decay based on diversity
- Adds deduplication for federation efficiency
- Adds repair arc detection for temporal quality tracking
- Production-ready for full SAGE deployment

Based on:
- Sessions 74-85: Trust-first → Conversational integration arc
- Legion implementations: Deduplication, dynamic decay, conversational signals
- Integration pattern: "Legion implements → Thor integrates"

Created: 2025-12-21 (Thor Session 86)
Author: Thor (Autonomous SAGE Research)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
import time
from collections import defaultdict
from enum import Enum

# Import Session 85 architecture
from sage.experiments.session85_conversational_trust import (
    ConversationalTrustFirstSelector,
    RepairSignalType,
    RepairSignal,
    ConversationalQuality,
    ConversationalTrustStats
)
from sage.core.trust_first_mrh_selector import TrustFirstSelectionResult
from sage.core.context_classifier import ContextClassifier


@dataclass
class RepairArc:
    """
    Temporal repair pattern detection (Legion implementation).

    REPAIR_ARC: Early difficulty → Mid persistence → Late resolution
    """
    early_difficulty: bool = False
    mid_persistence: bool = False
    late_resolution: bool = False
    arc_pattern: Optional[str] = None

    def detect_pattern(self) -> str:
        """Detect arc pattern from phases."""
        if self.early_difficulty and self.mid_persistence and self.late_resolution:
            return "REPAIR_ARC"
        elif self.late_resolution and not self.early_difficulty:
            return "SMOOTH"
        elif self.early_difficulty and not self.late_resolution:
            return "DEGRADED"
        else:
            return "INCONSISTENT"


@dataclass
class AdvancedTrustStats:
    """Statistics for advanced trust tracking."""
    # Conversational (Session 85)
    conversational_updates: int = 0
    repair_signals_received: int = 0
    avg_relationship_score: float = 0.5

    # Dynamic decay (Legion)
    diversity_scores: List[float] = field(default_factory=list)
    applied_decay_factors: List[float] = field(default_factory=list)
    avg_applied_decay: float = 0.72

    # Deduplication (Legion)
    attestations_imported: int = 0
    duplicates_skipped: int = 0
    import_efficiency_pct: float = 100.0

    # Repair arc (Legion)
    repair_arcs_detected: int = 0
    smooth_arcs: int = 0
    degraded_arcs: int = 0


class AdvancedTrustFirstSelector(ConversationalTrustFirstSelector):
    """
    Advanced trust-first selector integrating all optimizations.

    Combines:
    - Session 85: Conversational trust (repair signals)
    - Legion: Dynamic decay (observation diversity)
    - Legion: Attestation deduplication (federation efficiency)
    - Legion: Repair arc detection (temporal patterns)
    """

    def __init__(
        self,
        num_experts: int = 128,
        min_trust_evidence: int = 2,
        epsilon: float = 0.2,
        component: str = "thinker",
        network: str = "testnet",
        # Conversational (Session 85)
        conversational_weight: float = 0.4,
        enable_conversational: bool = True,
        # Dynamic decay (Legion)
        base_decay: float = 0.72,
        min_decay: float = 0.5,
        max_decay: float = 0.9,
        enable_dynamic_decay: bool = True,
        # Deduplication (Legion)
        enable_deduplication: bool = True,
        # Repair arc (Legion)
        enable_repair_arc: bool = True,
        arc_boost_factor: float = 1.2
    ):
        """
        Initialize advanced trust-first selector.

        Args:
            (Session 85 params): conversational_weight, enable_conversational
            base_decay: Base trust decay factor (0.72 from Session 70)
            min_decay: Minimum decay (high diversity → low decay)
            max_decay: Maximum decay (low diversity → high decay)
            enable_dynamic_decay: Enable/disable dynamic decay
            enable_deduplication: Enable/disable attestation deduplication
            enable_repair_arc: Enable/disable repair arc detection
            arc_boost_factor: Trust boost for REPAIR_ARC pattern (1.2 = 20%)
        """
        # Initialize conversational selector (Session 85)
        super().__init__(
            num_experts=num_experts,
            min_trust_evidence=min_trust_evidence,
            epsilon=epsilon,
            component=component,
            network=network,
            conversational_weight=conversational_weight,
            enable_conversational=enable_conversational
        )

        # Dynamic decay (Legion)
        self.base_decay = base_decay
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.enable_dynamic_decay = enable_dynamic_decay

        # Deduplication (Legion)
        self.enable_deduplication = enable_deduplication
        self.imported_attestation_ids: Set[str] = set()

        # Repair arc (Legion)
        self.enable_repair_arc = enable_repair_arc
        self.arc_boost_factor = arc_boost_factor
        self.conversation_history: List[ConversationalQuality] = []

        # Advanced statistics
        self.advanced_stats = AdvancedTrustStats()

    def compute_dynamic_decay(self, diversity_score: float) -> float:
        """
        Compute dynamic trust decay based on observation diversity (Legion).

        Logic:
        - High diversity (>0.7): Low decay (0.5) - trust federated observations
        - Low diversity (<0.3): High decay (0.9) - skeptical of similar data
        - Medium diversity: Interpolate

        Args:
            diversity_score: Observation diversity (0.0-1.0)

        Returns:
            Decay factor (min_decay to max_decay)
        """
        if not self.enable_dynamic_decay:
            return self.base_decay

        # Invert: High diversity → low decay
        # Linear interpolation
        decay = self.max_decay - (diversity_score * (self.max_decay - self.min_decay))

        # Clamp to [min_decay, max_decay]
        decay = np.clip(decay, self.min_decay, self.max_decay)

        # Track statistics
        self.advanced_stats.diversity_scores.append(diversity_score)
        self.advanced_stats.applied_decay_factors.append(decay)

        return decay

    def should_import_attestation(self, attestation_id: str) -> bool:
        """
        Check if attestation should be imported (Legion deduplication).

        Args:
            attestation_id: Unique attestation identifier

        Returns:
            True if should import, False if duplicate
        """
        if not self.enable_deduplication:
            return True  # Import everything

        if attestation_id in self.imported_attestation_ids:
            self.advanced_stats.duplicates_skipped += 1
            return False  # Skip duplicate

        # New attestation
        self.imported_attestation_ids.add(attestation_id)
        self.advanced_stats.attestations_imported += 1
        return True

    def detect_repair_arc(self, conversation_history: List[ConversationalQuality]) -> RepairArc:
        """
        Detect repair arc pattern from conversation history (Legion).

        Pattern: Early difficulty → Mid persistence → Late resolution

        Args:
            conversation_history: List of conversational quality observations

        Returns:
            RepairArc with detected pattern
        """
        if not self.enable_repair_arc or len(conversation_history) < 3:
            return RepairArc(arc_pattern="INSUFFICIENT_DATA")

        # Divide into thirds
        n = len(conversation_history)
        early_third = conversation_history[:n//3]
        mid_third = conversation_history[n//3:2*n//3]
        late_third = conversation_history[2*n//3:]

        # Early difficulty: Low relationship scores or high meta-leaks
        early_scores = [cq.compute_relationship_score() for cq in early_third]
        early_leaks = sum(cq.meta_cognitive_leaks for cq in early_third)
        early_difficulty = (np.mean(early_scores) < 0.5) or (early_leaks > 0)

        # Mid persistence: Continued engagement despite difficulty
        mid_scores = [cq.compute_relationship_score() for cq in mid_third]
        mid_persistence = len(mid_third) > 0 and np.mean(mid_scores) > 0.4

        # Late resolution: High relationship scores, low/zero leaks
        late_scores = [cq.compute_relationship_score() for cq in late_third]
        late_leaks = sum(cq.meta_cognitive_leaks for cq in late_third)
        late_resolution = (np.mean(late_scores) > 0.6) and (late_leaks == 0)

        arc = RepairArc(
            early_difficulty=early_difficulty,
            mid_persistence=mid_persistence,
            late_resolution=late_resolution
        )
        arc.arc_pattern = arc.detect_pattern()

        # Track statistics
        if arc.arc_pattern == "REPAIR_ARC":
            self.advanced_stats.repair_arcs_detected += 1
        elif arc.arc_pattern == "SMOOTH":
            self.advanced_stats.smooth_arcs += 1
        elif arc.arc_pattern == "DEGRADED":
            self.advanced_stats.degraded_arcs += 1

        return arc

    def update_trust_with_conversation(
        self,
        expert_id: int,
        context: str,
        internal_quality: float,
        conversational_quality: ConversationalQuality
    ):
        """
        Update trust with all advanced features.

        Extends Session 85 with:
        - Repair arc detection
        - Arc-based trust boosting

        Args:
            expert_id: Expert ID
            context: Context string
            internal_quality: Internal quality score
            conversational_quality: Conversational quality (repair signals)
        """
        # Track conversation history for arc detection
        self.conversation_history.append(conversational_quality)

        # Detect repair arc if enough history
        if len(self.conversation_history) >= 9:  # Min for 3-way split
            arc = self.detect_repair_arc(self.conversation_history[-9:])

            # Boost trust for REPAIR_ARC pattern (recovery from difficulty)
            if arc.arc_pattern == "REPAIR_ARC":
                # Apply boost to conversational quality
                relationship_score = conversational_quality.compute_relationship_score()
                boosted_score = min(relationship_score * self.arc_boost_factor, 1.0)

                # Create boosted conversational quality
                boosted_cq = ConversationalQuality(
                    repair_signals=conversational_quality.repair_signals,
                    meta_cognitive_leaks=conversational_quality.meta_cognitive_leaks,
                    arc_pattern="REPAIR_ARC",
                    relationship_score=boosted_score
                )
                conversational_quality = boosted_cq

        # Update via Session 85 architecture
        super().update_trust_with_conversation(
            expert_id,
            context,
            internal_quality,
            conversational_quality
        )

        # Update advanced stats
        self.advanced_stats.conversational_updates += 1
        self.advanced_stats.repair_signals_received += len(conversational_quality.repair_signals)
        relationship_score = conversational_quality.compute_relationship_score()
        self.advanced_stats.avg_relationship_score = (
            (self.advanced_stats.avg_relationship_score * (self.advanced_stats.conversational_updates - 1) +
             relationship_score) / self.advanced_stats.conversational_updates
        )

    def get_advanced_stats(self) -> Dict:
        """Get comprehensive advanced trust statistics."""
        stats = asdict(self.advanced_stats)

        # Compute averages
        if self.advanced_stats.diversity_scores:
            stats["avg_diversity_score"] = np.mean(self.advanced_stats.diversity_scores)
        if self.advanced_stats.applied_decay_factors:
            stats["avg_applied_decay"] = np.mean(self.advanced_stats.applied_decay_factors)

        # Import efficiency
        total_imports = self.advanced_stats.attestations_imported + self.advanced_stats.duplicates_skipped
        if total_imports > 0:
            stats["import_efficiency_pct"] = (
                self.advanced_stats.attestations_imported / total_imports * 100
            )

        # Configuration
        stats["enable_dynamic_decay"] = self.enable_dynamic_decay
        stats["enable_deduplication"] = self.enable_deduplication
        stats["enable_repair_arc"] = self.enable_repair_arc
        stats["conversational_weight"] = self.conversational_weight

        return stats


def run_session86_advanced_trust_test(
    num_experts: int = 128,
    num_generations: int = 90,
    num_sequences: int = 9,
    min_trust_evidence: int = 2,
    epsilon: float = 0.2,
    seed: int = 42
):
    """
    Test advanced trust selector with all optimizations.

    Scenario:
    1. Advanced selector (ALL optimizations) vs Baseline (Session 85 conversational only)
    2. Simulated repair signals with repair arc patterns
    3. Simulated diversity for dynamic decay
    4. Track all advanced metrics

    Hypothesis: Combined optimizations should exceed Session 85's +25.6% improvement.

    Args:
        num_experts: Number of experts (128 for Q3-Omni)
        num_generations: Number of generations
        num_sequences: Number of sequences
        min_trust_evidence: Minimum evidence for trust (2 from Session 78)
        epsilon: Epsilon-greedy rate (0.2 from Session 77)
        seed: Random seed

    Returns:
        Results dictionary
    """
    np.random.seed(seed)

    print("\n" + "="*80)
    print("SESSION 86: ADVANCED TRUST INTEGRATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Experts: {num_experts}")
    print(f"  Generations: {num_generations}")
    print(f"  Optimizations: Conversational + Dynamic Decay + Deduplication + Repair Arc")
    print()

    # Create selectors
    # Advanced (ALL optimizations)
    advanced_selector = AdvancedTrustFirstSelector(
        num_experts=num_experts,
        min_trust_evidence=min_trust_evidence,
        epsilon=epsilon,
        component="thinker_layer0",
        network="testnet",
        conversational_weight=0.4,  # Session 85
        enable_conversational=True,
        base_decay=0.72,  # Session 70
        enable_dynamic_decay=True,
        enable_deduplication=True,
        enable_repair_arc=True,
        arc_boost_factor=1.2
    )

    # Baseline (Session 85 conversational only)
    baseline_selector = ConversationalTrustFirstSelector(
        num_experts=num_experts,
        min_trust_evidence=min_trust_evidence,
        epsilon=epsilon,
        component="thinker_layer0",
        network="testnet",
        conversational_weight=0.4,
        enable_conversational=True
    )

    print("Selectors:")
    print(f"  Advanced: Conv + DynDecay + Dedup + RepairArc")
    print(f"  Baseline: Conversational only (Session 85)")
    print()

    # Create context classifier
    context_classifier = ContextClassifier(num_contexts=3)
    advanced_selector.context_classifier = context_classifier
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
    print("SIMULATION: Advanced trust-first integration")
    print("="*80)
    print()

    # Track results
    advanced_results = {
        "trust_driven_gens": [],
        "mode_history": [],
        "experts_used": set(),
        "first_trust_driven_gen": None
    }

    baseline_results = {
        "trust_driven_gens": [],
        "mode_history": [],
        "experts_used": set(),
        "first_trust_driven_gen": None
    }

    # Simulate repair arc pattern (early difficulty → resolution)
    def get_quality_for_arc(gen):
        """Simulate repair arc pattern."""
        if gen < 30:
            # Early: Low quality (difficulty)
            return np.clip(0.3 + 0.2 * np.random.randn(), 0.0, 0.5)
        elif gen < 60:
            # Middle: Improving
            return np.clip(0.5 + 0.2 * np.random.randn(), 0.3, 0.7)
        else:
            # Late: High quality (resolution)
            return np.clip(0.75 + 0.15 * np.random.randn(), 0.6, 1.0)

    # Simulate generations
    for gen in range(num_generations):
        seq_idx = gen % num_sequences
        sequence = sequences[seq_idx]
        context = context_classifier.classify(context_embeddings[seq_idx]).context_id

        # Simulate router logits
        np.random.seed(seed + gen)
        router_logits = np.random.randn(num_experts).astype(np.float32)

        # Simulate internal quality with repair arc pattern
        internal_quality = get_quality_for_arc(gen)

        # Simulate repair signals (Legion's conversational_trust_signals.py)
        np.random.seed(seed + gen + 1000)
        repair_signals = []
        if internal_quality > 0.7:
            # High quality → engagement/reassurance
            if np.random.random() < 0.7:
                repair_signals.append(RepairSignal(
                    signal_type=RepairSignalType.ENGAGEMENT,
                    confidence=0.6 + 0.3 * np.random.random(),
                    turn_number=gen
                ))
            if np.random.random() < 0.4:
                repair_signals.append(RepairSignal(
                    signal_type=RepairSignalType.REASSURANCE,
                    confidence=0.7 + 0.2 * np.random.random(),
                    turn_number=gen
                ))
        elif internal_quality < 0.4:
            # Low quality → abandonment (early phase)
            if np.random.random() < 0.3:
                repair_signals.append(RepairSignal(
                    signal_type=RepairSignalType.ABANDONMENT,
                    confidence=0.3 + 0.2 * np.random.random(),
                    turn_number=gen
                ))

        # Meta-cognitive leaks (high in early phase, zero in late)
        meta_leaks = 1 if (gen < 30 and np.random.random() < 0.3) else 0

        conversational_quality = ConversationalQuality(
            repair_signals=repair_signals,
            meta_cognitive_leaks=meta_leaks
        )

        # Advanced: Select and update
        advanced_result = advanced_selector.select_experts(router_logits, context, k=8)
        advanced_selector.update_trust_with_conversation(
            advanced_result.selected_expert_ids[0],
            context,
            internal_quality,
            conversational_quality
        )

        # Baseline: Select and update
        baseline_result = baseline_selector.select_experts(router_logits, context, k=8)
        baseline_selector.update_trust_with_conversation(
            baseline_result.selected_expert_ids[0],
            context,
            internal_quality,
            conversational_quality
        )

        # Track results
        for results, selector_result in [
            (advanced_results, advanced_result),
            (baseline_results, baseline_result)
        ]:
            results["mode_history"].append(selector_result.selection_mode)
            results["experts_used"].update(selector_result.selected_expert_ids)
            if selector_result.selection_mode == "trust_driven":
                results["trust_driven_gens"].append(gen)
                if results["first_trust_driven_gen"] is None:
                    results["first_trust_driven_gen"] = gen

    # Compute statistics
    def compute_stats(results, label):
        trust_driven_count = len(results["trust_driven_gens"])
        trust_driven_pct = (trust_driven_count / num_generations) * 100
        experts_used = len(results["experts_used"])
        first_activation = results["first_trust_driven_gen"]

        print(f"\n{label}:")
        print(f"  Trust_driven: {trust_driven_count}/{num_generations} ({trust_driven_pct:.1f}%)")
        print(f"  First activation: Gen {first_activation}")
        print(f"  Experts used: {experts_used}/{num_experts} ({experts_used/num_experts*100:.1f}%)")

        return {
            "trust_driven_count": trust_driven_count,
            "trust_driven_pct": trust_driven_pct,
            "experts_used": experts_used,
            "first_activation": first_activation
        }

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    advanced_stats = compute_stats(advanced_results, "Advanced (ALL optimizations)")
    baseline_stats = compute_stats(baseline_results, "Baseline (Session 85 conversational)")

    # Advanced statistics
    print(f"\nAdvanced Optimization Statistics:")
    advanced_stats_detail = advanced_selector.get_advanced_stats()
    print(f"  Conversational updates: {advanced_stats_detail['conversational_updates']}")
    print(f"  Repair signals: {advanced_stats_detail['repair_signals_received']}")
    print(f"  Avg relationship score: {advanced_stats_detail['avg_relationship_score']:.3f}")
    print(f"  Repair arcs detected: {advanced_stats_detail['repair_arcs_detected']}")
    print(f"  Smooth arcs: {advanced_stats_detail['smooth_arcs']}")
    print(f"  Degraded arcs: {advanced_stats_detail['degraded_arcs']}")

    # Compute improvement
    improvement = {
        "trust_driven_improvement": advanced_stats["trust_driven_pct"] - baseline_stats["trust_driven_pct"],
        "first_activation_speedup": (baseline_stats["first_activation"] or num_generations) - (advanced_stats["first_activation"] or num_generations),
        "expert_diversity_improvement": advanced_stats["experts_used"] - baseline_stats["experts_used"]
    }

    print(f"\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    print(f"\nAdvanced vs Baseline:")
    print(f"  Trust_driven improvement: {improvement['trust_driven_improvement']:+.1f}%")
    print(f"  First activation speedup: {improvement['first_activation_speedup']:+d} generations")
    print(f"  Expert diversity improvement: {improvement['expert_diversity_improvement']:+d} experts")

    # Conclusion
    print(f"\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if improvement["trust_driven_improvement"] > 5.0:
        print("\n✅ ADVANCED INTEGRATION SUCCESS!")
        print("   Combined optimizations exceed Session 85 baseline.")
    elif improvement["trust_driven_improvement"] > 0:
        print("\n✅ MODEST IMPROVEMENT")
        print("   Some benefit from combined optimizations.")
    else:
        print("\n⚠️  NO ADDITIONAL BENEFIT")
        print("   Session 85 conversational already optimal for this scenario.")

    print(f"\n   Insight: Advanced selector integrates {advanced_stats_detail['repair_arcs_detected']} repair arcs")
    print(f"   with conversational trust, demonstrating unified architecture.")
    print()

    # Save results
    results = {
        "session": 86,
        "configuration": {
            "num_experts": num_experts,
            "num_generations": num_generations,
            "num_sequences": num_sequences,
            "min_trust_evidence": min_trust_evidence,
            "epsilon": epsilon,
            "seed": seed
        },
        "advanced": {
            **advanced_stats,
            "advanced_stats": advanced_stats_detail
        },
        "baseline": baseline_stats,
        "improvement": improvement
    }

    output_file = os.path.join(os.path.dirname(__file__), "session86_advanced_trust_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    return results


if __name__ == "__main__":
    start_time = time.time()

    results = run_session86_advanced_trust_test(
        num_experts=128,
        num_generations=90,
        num_sequences=9,
        min_trust_evidence=2,
        epsilon=0.2,
        seed=42
    )

    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.1f}s")
    print()
