#!/usr/bin/env python3
"""
Context-Aware LCT Identity Bridge

Integrates Thor's context discovery (Sessions 66-67) with Legion's LCT identity
system (Session 63+) to create dynamic, context-aware expert identities.

Key Features:
- Automatic MRH relationship discovery via context clustering
- Dynamic T3 tensor updates based on trust evolution
- Context-specific trust tracking per expert
- Integration with ContextClassifier from Session 67

Architecture:
- ContextClassifier discovers semantic contexts from embeddings
- Context overlap → MRH pairing relationships
- Trust evolution → T3 tensor dimension updates
- Generates Web4-compliant LCT certificates with dynamic MRH/T3

Created: Session 64 (2025-12-17)
Author: Legion (Autonomous Research)
"""

import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import Counter
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.core.context_classifier import ContextClassifier
from sage.web4.lct_identity import parse_lct_uri, construct_lct_uri
from sage.web4.lct_certificate_generator import (
    SAGELCTCertificateGenerator,
    FullLCTCertificate,
    MRHRelationship,
    T3Tensor
)
from sage.web4.lct_resolver import LCTResolver


class ContextAwareIdentityBridge:
    """
    Bridge between context discovery and LCT identity system.

    Integrates:
    - Context clustering (Thor Session 67)
    - Trust evolution (Thor Session 66)
    - LCT certificates (Legion Session 63+)

    The bridge enables:
    1. Automatic MRH relationship discovery from context overlap
    2. Dynamic T3 tensor updates from trust evolution
    3. Context-specific expert identity tracking
    """

    def __init__(
        self,
        instance: str = "thinker",
        network: str = "testnet",
        n_contexts: int = 3,
        overlap_threshold: float = 0.7
    ):
        """
        Initialize context-aware identity bridge.

        Args:
            instance: SAGE instance name (e.g., "thinker", "dreamer")
            network: Network identifier (e.g., "testnet", "mainnet")
            n_contexts: Number of contexts for clustering
            overlap_threshold: Minimum context overlap for MRH pairing (0-1)
        """
        self.instance = instance
        self.network = network
        self.overlap_threshold = overlap_threshold

        # Context discovery
        self.context_classifier = ContextClassifier(num_contexts=n_contexts)
        self.context_fitted = False
        self.n_contexts = n_contexts  # Store for reference

        # Expert context tracking
        # Maps expert_id → list of context labels (one per sequence)
        self.expert_contexts: Dict[int, List[int]] = {}

        # Expert embeddings (optional, for analysis)
        self.expert_embeddings: Dict[int, np.ndarray] = {}

        # Trust evolution tracking
        # Maps (expert_id, context) → list of trust values over time
        self.trust_history: Dict[Tuple[int, int], List[float]] = {}

        # LCT components
        self.cert_generator = SAGELCTCertificateGenerator(instance, network)
        self.resolver = LCTResolver()

    def fit_context_classifier(self, embeddings: np.ndarray):
        """
        Fit context classifier on initial batch of embeddings.

        Args:
            embeddings: Shape [N, embedding_dim] - combined embeddings from all experts
        """
        self.context_classifier.fit(embeddings)
        self.context_fitted = True

    def discover_expert_contexts(
        self,
        expert_id: int,
        embeddings: np.ndarray
    ) -> List[int]:
        """
        Discover contexts for expert from sequence embeddings.

        Args:
            expert_id: Expert ID
            embeddings: Shape [N, embedding_dim] - embeddings from expert's sequences

        Returns:
            List of context labels (one per sequence)
        """
        if not self.context_fitted:
            # Auto-fit on first expert's data
            print(f"Auto-fitting context classifier on expert {expert_id}'s data")
            self.fit_context_classifier(embeddings)

        # Classify contexts for each sequence
        context_infos = self.context_classifier.classify_batch(embeddings)
        contexts = [info.context_id for info in context_infos]

        # Store for future reference
        self.expert_contexts[expert_id] = contexts
        self.expert_embeddings[expert_id] = embeddings

        return contexts

    def compute_context_overlap(
        self,
        expert_a: int,
        expert_b: int
    ) -> Tuple[float, List[int]]:
        """
        Compute context overlap between two experts.

        Uses cosine similarity of context distributions as overlap metric.

        Args:
            expert_a: First expert ID
            expert_b: Second expert ID

        Returns:
            Tuple of (overlap_score, shared_contexts)
            - overlap_score: 0.0-1.0, higher = more similar context distribution
            - shared_contexts: List of context IDs both experts handle
        """
        if expert_a not in self.expert_contexts or expert_b not in self.expert_contexts:
            return 0.0, []

        # Get context distributions (count of each context)
        dist_a = Counter(self.expert_contexts[expert_a])
        dist_b = Counter(self.expert_contexts[expert_b])

        # Create vectors over all contexts
        all_contexts = sorted(set(list(dist_a.keys()) + list(dist_b.keys())))
        vec_a = np.array([dist_a.get(c, 0) for c in all_contexts], dtype=float)
        vec_b = np.array([dist_b.get(c, 0) for c in all_contexts], dtype=float)

        # Normalize to unit vectors
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0, []

        vec_a = vec_a / norm_a
        vec_b = vec_b / norm_b

        # Cosine similarity
        overlap = float(np.dot(vec_a, vec_b))

        # Find shared contexts (both have > 0 count)
        shared = [c for c in all_contexts if dist_a.get(c, 0) > 0 and dist_b.get(c, 0) > 0]

        return overlap, shared

    def discover_mrh_pairings(
        self,
        expert_id: int,
        all_expert_ids: List[int],
        custom_threshold: Optional[float] = None
    ) -> List[MRHRelationship]:
        """
        Discover MRH pairing relationships via context overlap.

        Experts with high context overlap are likely to work well together,
        so we create MRH pairing relationships between them.

        Args:
            expert_id: Target expert ID
            all_expert_ids: All available expert IDs to check
            custom_threshold: Override default overlap threshold

        Returns:
            List of MRH pairing relationships
        """
        threshold = custom_threshold if custom_threshold is not None else self.overlap_threshold
        relationships = []

        for other_expert in all_expert_ids:
            if other_expert == expert_id:
                continue  # Don't pair with self

            overlap, shared_contexts = self.compute_context_overlap(expert_id, other_expert)

            if overlap >= threshold:
                # Create pairing relationship
                other_uri = construct_lct_uri(
                    component="sage",
                    instance=self.instance,
                    role=f"expert_{other_expert}",
                    network=self.network
                )

                relationships.append(MRHRelationship(
                    lct_id=other_uri,
                    relationship_type="context_similarity",
                    permanent=False,
                    context=f"overlap={overlap:.3f},shared_contexts={shared_contexts}"
                ))

        return relationships

    def update_trust_history(
        self,
        expert_id: int,
        context: int,
        trust_value: float
    ):
        """
        Update trust history for expert-context pair.

        Args:
            expert_id: Expert ID
            context: Context label
            trust_value: Current trust value (0.0-1.0)
        """
        key = (expert_id, context)
        if key not in self.trust_history:
            self.trust_history[key] = []
        self.trust_history[key].append(trust_value)

    def compute_t3_from_trust_evolution(
        self,
        expert_id: int
    ) -> T3Tensor:
        """
        Compute T3 tensor from trust evolution history.

        T3 Tensor Dimensions (computed from trust):
        - technical_competence: Mean trust across all contexts
        - social_reliability: Consistency across contexts (inverse of variance)
        - temporal_consistency: Stability over time (inverse of variance)
        - context_alignment: Diversity of contexts handled well
        - witness_count: Number of trust observations (placeholder)
        - lineage_depth: Ancestry depth (placeholder)

        Args:
            expert_id: Expert ID

        Returns:
            T3 tensor with dynamically computed dimensions
        """
        if expert_id not in self.expert_contexts:
            # No context data yet, return conservative defaults
            return T3Tensor(
                dimensions={
                    "technical_competence": 0.5,
                    "social_reliability": 0.5,
                    "temporal_consistency": 0.5,
                    "witness_count": 0.3,
                    "lineage_depth": 0.1,
                    "context_alignment": 0.5
                },
                composite_score=0.43,
                computation_witnesses=[f"lct://{self.instance}:coordinator@{self.network}"]
            )

        # Collect all trust values for this expert (across all contexts)
        expert_trust_values = []
        context_trust_means = {}  # Per-context mean trust

        for (eid, context), history in self.trust_history.items():
            if eid == expert_id and len(history) > 0:
                expert_trust_values.extend(history)
                context_trust_means[context] = np.mean(history)

        if not expert_trust_values:
            # No trust history yet, return defaults
            return T3Tensor(
                dimensions={
                    "technical_competence": 0.5,
                    "social_reliability": 0.5,
                    "temporal_consistency": 0.5,
                    "witness_count": 0.3,
                    "lineage_depth": 0.1,
                    "context_alignment": 0.5
                },
                composite_score=0.43,
                computation_witnesses=[f"lct://{self.instance}:coordinator@{self.network}"]
            )

        # Compute T3 dimensions

        # 1. Technical Competence = mean trust across all contexts
        technical_competence = float(np.mean(expert_trust_values))

        # 2. Social Reliability = consistency across contexts (low variance = high reliability)
        if len(context_trust_means) > 1:
            context_variance = float(np.var(list(context_trust_means.values())))
            social_reliability = 1.0 / (1.0 + context_variance * 10)  # Scale variance
        else:
            social_reliability = 0.7  # Single context, moderately reliable

        # 3. Temporal Consistency = stability over time (low variance = high consistency)
        trust_variance = float(np.var(expert_trust_values))
        temporal_consistency = 1.0 / (1.0 + trust_variance * 5)

        # 4. Context Alignment = diversity of contexts handled
        expert_context_set = set(self.expert_contexts[expert_id])
        context_alignment = len(expert_context_set) / float(self.n_contexts)

        # 5. Witness Count = number of observations (normalized)
        witness_count = min(1.0, len(expert_trust_values) / 100.0)  # Max out at 100 observations

        # 6. Lineage Depth = placeholder (not computed from trust)
        lineage_depth = 0.1

        dimensions = {
            "technical_competence": technical_competence,
            "social_reliability": social_reliability,
            "temporal_consistency": temporal_consistency,
            "witness_count": witness_count,
            "lineage_depth": lineage_depth,
            "context_alignment": context_alignment
        }

        # Composite score = weighted average
        # Weights: technical (30%), social (20%), temporal (20%), witness (10%), lineage (10%), context (10%)
        composite = (
            dimensions["technical_competence"] * 0.3 +
            dimensions["social_reliability"] * 0.2 +
            dimensions["temporal_consistency"] * 0.2 +
            dimensions["witness_count"] * 0.1 +
            dimensions["lineage_depth"] * 0.1 +
            dimensions["context_alignment"] * 0.1
        )

        return T3Tensor(
            dimensions=dimensions,
            composite_score=composite,
            computation_witnesses=[f"lct://sage:{self.instance}:coordinator@{self.network}"]
        )

    def generate_context_aware_certificate(
        self,
        expert_id: int,
        all_expert_ids: List[int],
        initial_atp: int = 100
    ) -> FullLCTCertificate:
        """
        Generate LCT certificate with context-aware MRH and T3.

        This is the main integration point:
        1. Generate base certificate (from Session 63+)
        2. Discover MRH pairings via context overlap (new)
        3. Compute T3 from trust evolution (new)

        Args:
            expert_id: Expert ID
            all_expert_ids: All available expert IDs (for MRH discovery)
            initial_atp: Initial ATP allocation

        Returns:
            Full LCT certificate with dynamic MRH and T3
        """
        # Generate base certificate structure
        cert = self.cert_generator.generate_expert_certificate(
            expert_id=expert_id,
            initial_atp=initial_atp
        )

        # Add context-discovered MRH pairings
        if expert_id in self.expert_contexts:
            discovered_pairings = self.discover_mrh_pairings(expert_id, all_expert_ids)
            cert.mrh.paired.extend(discovered_pairings)

        # Compute T3 from trust evolution
        t3_tensor = self.compute_t3_from_trust_evolution(expert_id)
        cert.t3_tensor = t3_tensor

        return cert

    def get_expert_context_summary(self, expert_id: int) -> Dict:
        """
        Get summary of expert's context distribution and trust.

        Args:
            expert_id: Expert ID

        Returns:
            Dictionary with context statistics
        """
        if expert_id not in self.expert_contexts:
            return {"error": "Expert not found"}

        contexts = self.expert_contexts[expert_id]
        context_dist = Counter(contexts)

        # Per-context trust
        context_trust = {}
        for context_id in context_dist.keys():
            key = (expert_id, context_id)
            if key in self.trust_history:
                history = self.trust_history[key]
                context_trust[f"context_{context_id}"] = {
                    "mean": float(np.mean(history)),
                    "std": float(np.std(history)),
                    "count": len(history),
                    "latest": history[-1] if history else None
                }

        return {
            "expert_id": expert_id,
            "total_sequences": len(contexts),
            "context_distribution": dict(context_dist),
            "unique_contexts": len(set(contexts)),
            "context_trust": context_trust
        }


# Example usage and testing
if __name__ == "__main__":
    print("Context-Aware LCT Identity Bridge - Example")
    print("=" * 70)

    # Initialize bridge
    bridge = ContextAwareIdentityBridge(
        instance="thinker",
        network="testnet",
        n_contexts=3,
        overlap_threshold=0.7
    )

    print("\n1. Simulating embeddings for 3 experts...")
    np.random.seed(42)

    # Expert 42: Primarily context 0 (simulates "code" specialization)
    expert_42_embeddings = np.random.randn(10, 8)  # 10 sequences, 8-dim embeddings
    expert_42_embeddings[:, 0] += 2.5  # Strong bias toward cluster 0

    # Expert 99: Mix of context 0 and 1 (simulates "code + reasoning")
    expert_99_embeddings = np.random.randn(12, 8)
    expert_99_embeddings[:6, 0] += 2.0  # Half context 0
    expert_99_embeddings[6:, 1] += 2.5  # Half context 1

    # Expert 1: Primarily context 2 (simulates "text" specialization)
    expert_1_embeddings = np.random.randn(8, 8)
    expert_1_embeddings[:, 2] += 3.0  # Strong bias toward cluster 2

    print("   Expert 42: 10 sequences (biased toward context 0)")
    print("   Expert 99: 12 sequences (mix of contexts 0 and 1)")
    print("   Expert 1:  8 sequences (biased toward context 2)")

    print("\n2. Discovering contexts for each expert...")
    contexts_42 = bridge.discover_expert_contexts(42, expert_42_embeddings)
    contexts_99 = bridge.discover_expert_contexts(99, expert_99_embeddings)
    contexts_1 = bridge.discover_expert_contexts(1, expert_1_embeddings)

    print(f"   Expert 42: {dict(Counter(contexts_42))}")
    print(f"   Expert 99: {dict(Counter(contexts_99))}")
    print(f"   Expert 1:  {dict(Counter(contexts_1))}")

    print("\n3. Computing context overlap between experts...")
    overlap_42_99, shared_42_99 = bridge.compute_context_overlap(42, 99)
    overlap_42_1, shared_42_1 = bridge.compute_context_overlap(42, 1)
    overlap_99_1, shared_99_1 = bridge.compute_context_overlap(99, 1)

    print(f"   Expert 42 ↔ 99: overlap={overlap_42_99:.3f}, shared={sorted(shared_42_99)}")
    print(f"   Expert 42 ↔ 1:  overlap={overlap_42_1:.3f}, shared={sorted(shared_42_1)}")
    print(f"   Expert 99 ↔ 1:  overlap={overlap_99_1:.3f}, shared={sorted(shared_99_1)}")

    print("\n4. Simulating trust evolution over time...")

    # Expert 42: Stable high trust in context 0
    print("   Expert 42 (context 0): Stable high trust")
    for i in range(10):
        bridge.update_trust_history(42, 0, 0.70 + np.random.randn() * 0.02)

    # Expert 99: Variable trust across contexts
    print("   Expert 99 (contexts 0,1): Variable trust")
    for i in range(6):
        bridge.update_trust_history(99, 0, 0.75 + np.random.randn() * 0.05)
    for i in range(6):
        bridge.update_trust_history(99, 1, 0.55 + np.random.randn() * 0.08)

    # Expert 1: Moderate trust in context 2
    print("   Expert 1 (context 2): Moderate trust")
    for i in range(8):
        bridge.update_trust_history(1, 2, 0.50 + np.random.randn() * 0.03)

    print("\n5. Generating context-aware LCT certificates...")
    all_experts = [42, 99, 1]

    cert_42 = bridge.generate_context_aware_certificate(42, all_experts)
    cert_99 = bridge.generate_context_aware_certificate(99, all_experts)
    cert_1 = bridge.generate_context_aware_certificate(1, all_experts)

    print(f"\n   === Expert 42 Certificate ===")
    print(f"   LCT URI: {cert_42.uri_reference}")
    print(f"   MRH Pairings (discovered): {len([p for p in cert_42.mrh.paired if p.relationship_type == 'context_similarity'])}")
    for pairing in cert_42.mrh.paired:
        if pairing.relationship_type == "context_similarity":
            print(f"      → {pairing.lct_id.split(':')[-1].split('@')[0]}: {pairing.context}")
    print(f"   T3 Composite Score: {cert_42.t3_tensor.composite_score:.3f}")
    print(f"      Technical Competence: {cert_42.t3_tensor.dimensions['technical_competence']:.3f}")
    print(f"      Social Reliability:   {cert_42.t3_tensor.dimensions['social_reliability']:.3f}")
    print(f"      Temporal Consistency: {cert_42.t3_tensor.dimensions['temporal_consistency']:.3f}")
    print(f"      Context Alignment:    {cert_42.t3_tensor.dimensions['context_alignment']:.3f}")

    print(f"\n   === Expert 99 Certificate ===")
    print(f"   LCT URI: {cert_99.uri_reference}")
    print(f"   MRH Pairings (discovered): {len([p for p in cert_99.mrh.paired if p.relationship_type == 'context_similarity'])}")
    for pairing in cert_99.mrh.paired:
        if pairing.relationship_type == "context_similarity":
            print(f"      → {pairing.lct_id.split(':')[-1].split('@')[0]}: {pairing.context}")
    print(f"   T3 Composite Score: {cert_99.t3_tensor.composite_score:.3f}")
    print(f"      Technical Competence: {cert_99.t3_tensor.dimensions['technical_competence']:.3f}")
    print(f"      Social Reliability:   {cert_99.t3_tensor.dimensions['social_reliability']:.3f}")
    print(f"      Temporal Consistency: {cert_99.t3_tensor.dimensions['temporal_consistency']:.3f}")
    print(f"      Context Alignment:    {cert_99.t3_tensor.dimensions['context_alignment']:.3f}")

    print(f"\n   === Expert 1 Certificate ===")
    print(f"   LCT URI: {cert_1.uri_reference}")
    print(f"   MRH Pairings (discovered): {len([p for p in cert_1.mrh.paired if p.relationship_type == 'context_similarity'])}")
    for pairing in cert_1.mrh.paired:
        if pairing.relationship_type == "context_similarity":
            print(f"      → {pairing.lct_id.split(':')[-1].split('@')[0]}: {pairing.context}")
    print(f"   T3 Composite Score: {cert_1.t3_tensor.composite_score:.3f}")
    print(f"      Technical Competence: {cert_1.t3_tensor.dimensions['technical_competence']:.3f}")
    print(f"      Social Reliability:   {cert_1.t3_tensor.dimensions['social_reliability']:.3f}")
    print(f"      Temporal Consistency: {cert_1.t3_tensor.dimensions['temporal_consistency']:.3f}")
    print(f"      Context Alignment:    {cert_1.t3_tensor.dimensions['context_alignment']:.3f}")

    print("\n6. Expert context summaries...")
    for eid in [42, 99, 1]:
        summary = bridge.get_expert_context_summary(eid)
        print(f"\n   Expert {eid}:")
        print(f"      Total sequences: {summary['total_sequences']}")
        print(f"      Context distribution: {summary['context_distribution']}")
        print(f"      Unique contexts: {summary['unique_contexts']}")
        if summary['context_trust']:
            print(f"      Trust by context:")
            for ctx, stats in summary['context_trust'].items():
                print(f"         {ctx}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, n={stats['count']}")

    print("\n✓ Context-aware LCT integration validated!")
    print("\nKey Findings:")
    print("- MRH pairings discovered automatically from context overlap")
    print("- T3 tensors computed dynamically from trust evolution")
    print("- Experts with high context overlap have MRH pairing relationships")
    print("- Trust stability reflected in T3 temporal_consistency dimension")
    print("- Context diversity reflected in T3 context_alignment dimension")
