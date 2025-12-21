"""
Session 87: Multi-Dimensional Trust Integration

Integrates Legion's MultiDimensionalTrustScorer with Thor's AdvancedTrustFirstSelector.

Evolution Path:
- Session 85 (Thor): Conversational trust (+25.6%)
- Session 86 (Thor): Unified architecture (context-dependent optimizations)
- Legion S79 Track 1: Multi-dimensional framework (+10%)
- Session 87 (Thor): Multi-dimensional integration (ALL dimensions)

Architecture:
    TrustFirstMRHSelector (Session 77)
        ↓
    ConversationalTrustFirstSelector (Session 85)
        ↓
    AdvancedTrustFirstSelector (Session 86)
        ↓
    MultiDimensionalTrustFirstSelector (Session 87) ← NEW

Integration Strategy:
1. Internal Quality: From AdvancedTrustFirstSelector's trust observations
2. Conversational Trust: From ConversationalTrustFirstSelector's repair signals
3. Byzantine Consensus: Simulated from multi-expert agreement
4. Federation Trust: From federation attestations with dynamic decay

Expected Result: > 10% improvement (Legion's multi-dimensional showed +10%)
"""

import sys
import random
import statistics
import numpy as np
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import time


# ============================================================================
# TRUST DIMENSIONS (from Legion's Multi-Dimensional Trust)
# ============================================================================

@dataclass
class InternalQualityScore:
    """Internal quality from expert observations (Thor)."""
    expert_id: int
    context: str
    quality: float  # 0.0-1.0
    observation_count: int
    confidence: float


@dataclass
class ConversationalTrustScore:
    """Conversational trust from repair signals (Sprout/Thor)."""
    expert_id: int
    context: str
    relationship_score: float  # 0.0-1.0
    engagement_count: int
    reassurance_count: int
    abandonment_count: int
    correction_count: int
    arc_pattern: Optional[str] = None


@dataclass
class ByzantineConsensusScore:
    """Byzantine consensus from multi-expert agreement (Legion)."""
    expert_id: int
    context: str
    consensus_quality: float  # 0.0-1.0
    num_attestations: int
    outliers_detected: int
    consensus_confidence: float


@dataclass
class FederationTrustScore:
    """Federation trust from cross-society attestations (Legion)."""
    expert_id: int
    context: str
    federated_quality: float  # 0.0-1.0 (with decay)
    source_societies: List[str]
    diversity_score: float
    decay_factor: float


@dataclass
class MultiDimensionalTrustScore:
    """Composite trust score from all dimensions."""
    expert_id: int
    context: str

    # Individual dimensions
    internal_quality: Optional[InternalQualityScore]
    conversational_trust: Optional[ConversationalTrustScore]
    byzantine_consensus: Optional[ByzantineConsensusScore]
    federation_trust: Optional[FederationTrustScore]

    # Composite
    composite_score: float
    confidence: float
    dimensions_available: int
    trust_tier: str  # "HIGH", "MEDIUM", "LOW", "UNKNOWN"


# ============================================================================
# MULTI-DIMENSIONAL TRUST SCORER (adapted from Legion)
# ============================================================================

class MultiDimensionalTrustScorer:
    """
    Computes composite trust scores from multiple validation dimensions.

    Adapted from Legion Session 79 Track 1.
    """

    def __init__(
        self,
        internal_weight: float = 0.35,
        conversational_weight: float = 0.25,
        byzantine_weight: float = 0.25,
        federation_weight: float = 0.15
    ):
        """
        Args:
            internal_weight: Weight for internal quality (Thor)
            conversational_weight: Weight for conversational trust (Sprout)
            byzantine_weight: Weight for Byzantine consensus (Legion)
            federation_weight: Weight for federation trust (Legion)
        """
        # Validate weights sum to 1.0
        total = internal_weight + conversational_weight + byzantine_weight + federation_weight
        assert abs(total - 1.0) < 0.01, f"Weights must sum to 1.0, got {total}"

        self.internal_weight = internal_weight
        self.conversational_weight = conversational_weight
        self.byzantine_weight = byzantine_weight
        self.federation_weight = federation_weight

        self.stats = {
            'total_scores_computed': 0,
            'dimension_usage': {
                'internal': 0,
                'conversational': 0,
                'byzantine': 0,
                'federation': 0
            }
        }

    def compute_composite_score(
        self,
        expert_id: int,
        context: str,
        internal_quality: Optional[InternalQualityScore] = None,
        conversational_trust: Optional[ConversationalTrustScore] = None,
        byzantine_consensus: Optional[ByzantineConsensusScore] = None,
        federation_trust: Optional[FederationTrustScore] = None
    ) -> MultiDimensionalTrustScore:
        """
        Compute composite trust score from available dimensions.

        Gracefully handles missing dimensions by re-normalizing weights.
        """
        self.stats['total_scores_computed'] += 1

        dimensions_available = 0
        total_weighted_score = 0.0
        total_weight_used = 0.0

        # 1. Internal Quality
        if internal_quality is not None:
            dimensions_available += 1
            self.stats['dimension_usage']['internal'] += 1
            score_contribution = internal_quality.quality * internal_quality.confidence
            total_weighted_score += score_contribution * self.internal_weight
            total_weight_used += self.internal_weight

        # 2. Conversational Trust
        if conversational_trust is not None:
            dimensions_available += 1
            self.stats['dimension_usage']['conversational'] += 1
            total_weighted_score += conversational_trust.relationship_score * self.conversational_weight
            total_weight_used += self.conversational_weight

        # 3. Byzantine Consensus
        if byzantine_consensus is not None:
            dimensions_available += 1
            self.stats['dimension_usage']['byzantine'] += 1
            score_contribution = (byzantine_consensus.consensus_quality *
                                byzantine_consensus.consensus_confidence)
            total_weighted_score += score_contribution * self.byzantine_weight
            total_weight_used += self.byzantine_weight

        # 4. Federation Trust
        if federation_trust is not None:
            dimensions_available += 1
            self.stats['dimension_usage']['federation'] += 1
            total_weighted_score += federation_trust.federated_quality * self.federation_weight
            total_weight_used += self.federation_weight

        # Normalize by weights used
        if total_weight_used > 0:
            composite_score = total_weighted_score / total_weight_used
        else:
            composite_score = 0.5  # Neutral if no dimensions

        # Confidence based on dimension availability
        confidence = dimensions_available / 4.0

        # Trust tier classification
        if dimensions_available == 0:
            trust_tier = "UNKNOWN"
        elif composite_score >= 0.7:
            trust_tier = "HIGH"
        elif composite_score >= 0.4:
            trust_tier = "MEDIUM"
        else:
            trust_tier = "LOW"

        return MultiDimensionalTrustScore(
            expert_id=expert_id,
            context=context,
            internal_quality=internal_quality,
            conversational_trust=conversational_trust,
            byzantine_consensus=byzantine_consensus,
            federation_trust=federation_trust,
            composite_score=composite_score,
            confidence=confidence,
            dimensions_available=dimensions_available,
            trust_tier=trust_tier
        )


# ============================================================================
# MULTI-DIMENSIONAL TRUST-FIRST SELECTOR
# ============================================================================

class MultiDimensionalTrustFirstSelector:
    """
    Multi-dimensional trust-first expert selector.

    Integrates Session 86's AdvancedTrustFirstSelector with Legion's
    MultiDimensionalTrustScorer for comprehensive trust evaluation.

    Dimensions:
    1. Internal Quality: Expert observation history
    2. Conversational Trust: Repair signals and relationship quality
    3. Byzantine Consensus: Multi-expert agreement on quality
    4. Federation Trust: Cross-society attestations with dynamic decay
    """

    def __init__(
        self,
        num_experts: int,
        epsilon: float = 0.2,
        min_trust_evidence: int = 2,
        decay_factor: float = 0.72,
        # Multi-dimensional weights
        internal_weight: float = 0.35,
        conversational_weight: float = 0.25,
        byzantine_weight: float = 0.25,
        federation_weight: float = 0.15,
        # Feature toggles
        enable_conversational: bool = True,
        enable_byzantine: bool = True,
        enable_federation: bool = True
    ):
        """
        Args:
            num_experts: Number of experts in MoE
            epsilon: Exploration rate (ε-greedy)
            min_trust_evidence: Minimum observations before trusting expert
            decay_factor: Trust decay factor for old observations
            *_weight: Weights for multi-dimensional trust (must sum to 1.0)
            enable_*: Feature toggles for each dimension
        """
        self.num_experts = num_experts
        self.epsilon = epsilon
        self.min_trust_evidence = min_trust_evidence
        self.decay_factor = decay_factor

        self.enable_conversational = enable_conversational
        self.enable_byzantine = enable_byzantine
        self.enable_federation = enable_federation

        # Multi-dimensional scorer
        self.md_scorer = MultiDimensionalTrustScorer(
            internal_weight=internal_weight,
            conversational_weight=conversational_weight,
            byzantine_weight=byzantine_weight,
            federation_weight=federation_weight
        )

        # Expert observation history (internal quality)
        self.expert_observations = defaultdict(list)

        # Conversational trust data
        self.conversational_signals = defaultdict(list)

        # Byzantine consensus data
        self.byzantine_attestations = defaultdict(list)

        # Federation trust data
        self.federation_attestations = defaultdict(list)

        # Stats
        self.stats = {
            'trust_driven_count': 0,
            'exploration_count': 0,
            'total_selections': 0,
            'first_trust_activation': None,
            'multi_dimensional_scores': [],
            'experts_used': set()
        }

    def select_expert(
        self,
        router_logits: np.ndarray,
        context: str
    ) -> Tuple[int, str]:
        """
        Select expert using multi-dimensional trust-first strategy.

        Returns:
            (expert_id, selection_reason)
        """
        self.stats['total_selections'] += 1

        # Build multi-dimensional trust scores for all experts
        md_scores = {}
        for expert_id in range(self.num_experts):
            # 1. Internal Quality (always available from observations)
            internal_quality = self._compute_internal_quality(expert_id, context)

            # 2. Conversational Trust (if enabled and available)
            conversational_trust = None
            if self.enable_conversational:
                conversational_trust = self._compute_conversational_trust(expert_id, context)

            # 3. Byzantine Consensus (if enabled and available)
            byzantine_consensus = None
            if self.enable_byzantine:
                byzantine_consensus = self._compute_byzantine_consensus(expert_id, context)

            # 4. Federation Trust (if enabled and available)
            federation_trust = None
            if self.enable_federation:
                federation_trust = self._compute_federation_trust(expert_id, context)

            # Compute composite score
            md_score = self.md_scorer.compute_composite_score(
                expert_id=expert_id,
                context=context,
                internal_quality=internal_quality,
                conversational_trust=conversational_trust,
                byzantine_consensus=byzantine_consensus,
                federation_trust=federation_trust
            )
            md_scores[expert_id] = md_score

        # Find expert with highest composite trust score
        best_expert = max(md_scores.keys(), key=lambda e: md_scores[e].composite_score)
        best_score = md_scores[best_expert]

        # Trust-first selection with ε-greedy exploration
        if (best_score.internal_quality and
            best_score.internal_quality.observation_count >= self.min_trust_evidence and
            random.random() > self.epsilon):
            # Trust-driven selection
            selected_expert = best_expert
            reason = f"trust_driven (composite={best_score.composite_score:.3f}, dims={best_score.dimensions_available})"

            self.stats['trust_driven_count'] += 1
            if self.stats['first_trust_activation'] is None:
                self.stats['first_trust_activation'] = self.stats['total_selections']
        else:
            # Exploration (follow router)
            selected_expert = int(np.argmax(router_logits))
            reason = "exploration (following router)"
            self.stats['exploration_count'] += 1

        self.stats['experts_used'].add(selected_expert)
        self.stats['multi_dimensional_scores'].append(md_scores[selected_expert])

        return selected_expert, reason

    def update_observation(
        self,
        expert_id: int,
        context: str,
        quality: float
    ):
        """Update internal quality observations."""
        self.expert_observations[(expert_id, context)].append(quality)

    def update_conversational_signal(
        self,
        expert_id: int,
        context: str,
        signal_type: str,  # "ENGAGEMENT", "REASSURANCE", "ABANDONMENT", "CORRECTION"
        relationship_score: float
    ):
        """Update conversational trust signals."""
        if not self.enable_conversational:
            return

        self.conversational_signals[(expert_id, context)].append({
            'signal_type': signal_type,
            'relationship_score': relationship_score
        })

    def _compute_internal_quality(
        self,
        expert_id: int,
        context: str
    ) -> Optional[InternalQualityScore]:
        """Compute internal quality score from observations."""
        obs = self.expert_observations.get((expert_id, context), [])
        if not obs:
            return None

        # Apply decay to older observations
        decayed_obs = []
        for i, quality in enumerate(obs):
            age = len(obs) - i - 1
            decayed_quality = quality * (self.decay_factor ** age)
            decayed_obs.append(decayed_quality)

        avg_quality = np.mean(decayed_obs)
        observation_count = len(obs)

        # Confidence based on observation count
        confidence = min(1.0, observation_count / (self.min_trust_evidence * 2))

        return InternalQualityScore(
            expert_id=expert_id,
            context=context,
            quality=float(avg_quality),
            observation_count=observation_count,
            confidence=float(confidence)
        )

    def _compute_conversational_trust(
        self,
        expert_id: int,
        context: str
    ) -> Optional[ConversationalTrustScore]:
        """Compute conversational trust from repair signals."""
        signals = self.conversational_signals.get((expert_id, context), [])
        if not signals:
            return None

        # Count signal types
        engagement_count = sum(1 for s in signals if s['signal_type'] == 'ENGAGEMENT')
        reassurance_count = sum(1 for s in signals if s['signal_type'] == 'REASSURANCE')
        abandonment_count = sum(1 for s in signals if s['signal_type'] == 'ABANDONMENT')
        correction_count = sum(1 for s in signals if s['signal_type'] == 'CORRECTION')

        # Relationship score (from latest signals)
        recent_scores = [s['relationship_score'] for s in signals[-5:]]
        relationship_score = np.mean(recent_scores) if recent_scores else 0.5

        return ConversationalTrustScore(
            expert_id=expert_id,
            context=context,
            relationship_score=float(relationship_score),
            engagement_count=engagement_count,
            reassurance_count=reassurance_count,
            abandonment_count=abandonment_count,
            correction_count=correction_count,
            arc_pattern=None  # Could detect REPAIR_ARC pattern here
        )

    def _compute_byzantine_consensus(
        self,
        expert_id: int,
        context: str
    ) -> Optional[ByzantineConsensusScore]:
        """Compute Byzantine consensus from multi-expert attestations."""
        attestations = self.byzantine_attestations.get((expert_id, context), [])
        if not attestations:
            return None

        # Consensus quality (median of attestations)
        consensus_quality = np.median(attestations)

        # Outlier detection
        if len(attestations) >= 3:
            q1, q3 = np.percentile(attestations, [25, 75])
            iqr = q3 - q1
            outliers = sum(1 for a in attestations if a < q1 - 1.5*iqr or a > q3 + 1.5*iqr)
        else:
            outliers = 0

        # Confidence based on agreement
        std_dev = np.std(attestations) if len(attestations) > 1 else 0.3
        consensus_confidence = max(0.5, 1.0 - std_dev)

        return ByzantineConsensusScore(
            expert_id=expert_id,
            context=context,
            consensus_quality=float(consensus_quality),
            num_attestations=len(attestations),
            outliers_detected=int(outliers),
            consensus_confidence=float(consensus_confidence)
        )

    def _compute_federation_trust(
        self,
        expert_id: int,
        context: str
    ) -> Optional[FederationTrustScore]:
        """Compute federation trust from cross-society attestations."""
        attestations = self.federation_attestations.get((expert_id, context), [])
        if not attestations:
            return None

        # Extract quality scores and sources
        qualities = [a['quality'] for a in attestations]
        sources = list(set(a['source'] for a in attestations))

        # Diversity score (based on source variety)
        diversity_score = len(sources) / 3.0  # Normalize (max 3 societies)

        # Dynamic decay based on diversity
        decay = self.decay_factor + (1 - self.decay_factor) * diversity_score

        # Federated quality (with decay)
        avg_quality = np.mean(qualities)
        federated_quality = avg_quality * decay

        return FederationTrustScore(
            expert_id=expert_id,
            context=context,
            federated_quality=float(federated_quality),
            source_societies=sources,
            diversity_score=float(diversity_score),
            decay_factor=float(decay)
        )


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_session87_multidimensional_test():
    """
    Test multi-dimensional trust integration.

    Compares:
    - Multi-dimensional (ALL dimensions) vs
    - Internal-only (baseline)
    """
    print("=" * 80)
    print("SESSION 87: MULTI-DIMENSIONAL TRUST INTEGRATION")
    print("=" * 80)
    print()

    random.seed(42)
    np.random.seed(42)

    num_experts = 128
    num_generations = 90
    num_sequences = 9

    print("Configuration:")
    print(f"  Experts: {num_experts}")
    print(f"  Generations: {num_generations}")
    print(f"  Dimensions: Internal + Conversational + Byzantine + Federation")
    print()

    # Multi-dimensional selector (ALL dimensions)
    md_selector = MultiDimensionalTrustFirstSelector(
        num_experts=num_experts,
        epsilon=0.2,
        min_trust_evidence=2,
        enable_conversational=True,
        enable_byzantine=True,
        enable_federation=True
    )

    # Baseline selector (internal-only)
    baseline_selector = MultiDimensionalTrustFirstSelector(
        num_experts=num_experts,
        epsilon=0.2,
        min_trust_evidence=2,
        enable_conversational=False,
        enable_byzantine=False,
        enable_federation=False
    )

    print("Selectors:")
    print("  Multi-dimensional: Internal + Conversational + Byzantine + Federation")
    print("  Baseline: Internal only")
    print()

    print("=" * 80)
    print("SIMULATION: Multi-dimensional trust-first selection")
    print("=" * 80)
    print()

    start_time = time.time()

    # Simulate generations
    # Use persistent contexts (same contexts repeat across generations)
    contexts = [f"seq{seq_idx}" for seq_idx in range(num_sequences)]

    for gen in range(num_generations):
        for seq_idx in range(num_sequences):
            context = contexts[seq_idx]  # Persistent context

            # Simulate router logits (monopoly on few experts initially)
            router_logits = np.zeros(num_experts)
            if gen < 30:
                # Initial monopoly (4 experts)
                monopoly_experts = [0, 1, 2, 3]
                router_logits[monopoly_experts] = np.random.dirichlet([1]*4)
            else:
                # Gradual diversification
                router_logits = np.random.dirichlet([0.5]*num_experts)

            # Multi-dimensional selection
            md_expert, md_reason = md_selector.select_expert(router_logits, context)

            # Baseline selection
            bl_expert, bl_reason = baseline_selector.select_expert(router_logits, context)

            # Simulate response quality (better for diverse experts)
            md_quality = 0.6 + 0.3 * (md_expert / num_experts) + random.uniform(-0.1, 0.1)
            md_quality = max(0.0, min(1.0, md_quality))

            bl_quality = 0.6 + 0.3 * (bl_expert / num_experts) + random.uniform(-0.1, 0.1)
            bl_quality = max(0.0, min(1.0, bl_quality))

            # Update internal quality observations
            md_selector.update_observation(md_expert, context, md_quality)
            baseline_selector.update_observation(bl_expert, context, bl_quality)

            # Update multi-dimensional signals (for MD selector only)
            if md_quality > 0.7:
                md_selector.update_conversational_signal(
                    md_expert, context, "ENGAGEMENT", md_quality
                )
                md_selector.byzantine_attestations[(md_expert, context)].append(md_quality)
                md_selector.federation_attestations[(md_expert, context)].append({
                    'quality': md_quality,
                    'source': random.choice(['thor', 'sprout', 'legion'])
                })
            elif md_quality < 0.4:
                md_selector.update_conversational_signal(
                    md_expert, context, "ABANDONMENT", md_quality
                )

    elapsed = time.time() - start_time

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    # Multi-dimensional results
    md_trust_driven = md_selector.stats['trust_driven_count']
    md_total = md_selector.stats['total_selections']
    md_trust_pct = (md_trust_driven / md_total * 100) if md_total > 0 else 0
    md_first_activation = md_selector.stats['first_trust_activation']
    md_experts_used = len(md_selector.stats['experts_used'])

    # Baseline results
    bl_trust_driven = baseline_selector.stats['trust_driven_count']
    bl_total = baseline_selector.stats['total_selections']
    bl_trust_pct = (bl_trust_driven / bl_total * 100) if bl_total > 0 else 0
    bl_first_activation = baseline_selector.stats['first_trust_activation']
    bl_experts_used = len(baseline_selector.stats['experts_used'])

    print("Multi-dimensional (ALL dimensions):")
    print(f"  Trust_driven: {md_trust_driven}/{md_total} ({md_trust_pct:.1f}%)")
    print(f"  First activation: Gen {md_first_activation}")
    print(f"  Experts used: {md_experts_used}/{num_experts} ({md_experts_used/num_experts*100:.1f}%)")
    print()

    print("Baseline (Internal only):")
    print(f"  Trust_driven: {bl_trust_driven}/{bl_total} ({bl_trust_pct:.1f}%)")
    print(f"  First activation: Gen {bl_first_activation}")
    print(f"  Experts used: {bl_experts_used}/{num_experts} ({bl_experts_used/num_experts*100:.1f}%)")
    print()

    # Dimension usage stats
    print("Multi-Dimensional Stats:")
    print(f"  Total MD scores computed: {md_selector.md_scorer.stats['total_scores_computed']}")
    print("  Dimension usage:")
    for dim, count in md_selector.md_scorer.stats['dimension_usage'].items():
        pct = (count / md_selector.md_scorer.stats['total_scores_computed'] * 100) if md_selector.md_scorer.stats['total_scores_computed'] > 0 else 0
        print(f"    {dim:15s}: {count:4d} ({pct:5.1f}%)")
    print()

    # Average dimensions available
    if md_selector.stats['multi_dimensional_scores']:
        avg_dims = np.mean([s.dimensions_available for s in md_selector.stats['multi_dimensional_scores']])
        avg_confidence = np.mean([s.confidence for s in md_selector.stats['multi_dimensional_scores']])
        avg_composite = np.mean([s.composite_score for s in md_selector.stats['multi_dimensional_scores']])

        print("  Average dimensions available: {:.1f} / 4".format(avg_dims))
        print("  Average confidence: {:.3f}".format(avg_confidence))
        print("  Average composite score: {:.3f}".format(avg_composite))
        print()

    # Improvement analysis
    print("=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)
    print()

    trust_improvement = md_trust_pct - bl_trust_pct
    activation_speedup = bl_first_activation - md_first_activation if (bl_first_activation and md_first_activation) else 0
    diversity_improvement = md_experts_used - bl_experts_used

    print("Multi-dimensional vs Baseline:")
    print(f"  Trust_driven improvement: {trust_improvement:+.1f}%")
    print(f"  First activation speedup: {activation_speedup:+d} generations")
    print(f"  Expert diversity improvement: {diversity_improvement:+d} experts")
    print()

    # Conclusion
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    if trust_improvement > 5:
        print("✅ SIGNIFICANT IMPROVEMENT")
        print("   Multi-dimensional trust substantially outperforms internal-only.")
        print()
        print(f"   Insight: {avg_dims:.1f} dimensions on average, providing")
        print(f"   {avg_confidence:.1%} confidence in composite trust scores.")
    elif trust_improvement > 0:
        print("✅ MODEST IMPROVEMENT")
        print("   Multi-dimensional trust shows benefit over internal-only.")
    else:
        print("⚠️  NO IMPROVEMENT")
        print("   Multi-dimensional trust did not outperform internal-only.")
        print("   May need more diverse test scenarios or weight tuning.")
    print()

    # Save results
    results = {
        'session': 87,
        'configuration': {
            'num_experts': num_experts,
            'num_generations': num_generations,
            'num_sequences': num_sequences,
            'epsilon': 0.2,
            'min_trust_evidence': 2
        },
        'multi_dimensional': {
            'trust_driven_count': md_trust_driven,
            'trust_driven_pct': md_trust_pct,
            'first_activation': md_first_activation,
            'experts_used': md_experts_used,
            'avg_dimensions_available': float(avg_dims) if 'avg_dims' in locals() else 0,
            'avg_confidence': float(avg_confidence) if 'avg_confidence' in locals() else 0,
            'avg_composite_score': float(avg_composite) if 'avg_composite' in locals() else 0,
            'dimension_usage': md_selector.md_scorer.stats['dimension_usage']
        },
        'baseline': {
            'trust_driven_count': bl_trust_driven,
            'trust_driven_pct': bl_trust_pct,
            'first_activation': bl_first_activation,
            'experts_used': bl_experts_used
        },
        'improvement': {
            'trust_driven_improvement': trust_improvement,
            'activation_speedup': activation_speedup,
            'diversity_improvement': diversity_improvement
        }
    }

    results_file = "/home/dp/ai-workspace/HRM/sage/experiments/session87_multidimensional_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_file}")
    print()
    print(f"Total execution time: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    run_session87_multidimensional_test()
