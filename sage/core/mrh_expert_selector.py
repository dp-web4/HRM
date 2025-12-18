#!/usr/bin/env python3
"""
MRH-Based Expert Selector - Session 65 Integration
=================================================

Extends trust-based expert selection with MRH (Markov Relevancy Horizon)
discovery for intelligent expert substitution.

**Session 65 Breakthrough**: MRH substitution breaks router monopoly
- Router collapse: 4 experts out of 128 (96.875% waste)
- MRH solution: 100% diversity increase (4 → 8 experts)
- Specialist emergence: 5 specialists (62.5% rate)
- Key mechanism: Context overlap-based alternative discovery

**Core Innovation**: Expert substitution = relationship discovery problem
- Traditional: Substitute via semantic cluster membership (coarse)
- MRH: Substitute via context overlap relationships (fine-grained)
- Result: Better specialists emerge through precise pairing

**Web4 Pattern**: MRH relationship discovery
- High context overlap → potential substitution relationship
- Trust-based filtering → only high-trust alternatives
- Adaptive learning → MRH pairings evolve with usage

**Author**: Legion (Autonomous Web4 Research Session 66)
**Date**: 2025-12-18
**Provenance**: Session 65 (MRH substitution breakthrough) → SAGE implementation
"""

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

from sage.core.expert_reputation import (
    ExpertReputation,
    ExpertReputationDB,
    get_default_reputation_db
)
from sage.core.context_classifier import ContextClassifier
from sage.web4.context_aware_identity_bridge import ContextAwareIdentityBridge


@dataclass
class MRHSubstitution:
    """
    Record of an MRH-based expert substitution.

    Tracks:
    - Which expert was requested
    - Which expert was used instead
    - Why (context overlap, trust scores)
    - Outcome (quality delta)
    """
    generation: int
    requested_expert: int
    substitute_expert: int
    context: str
    context_overlap: float          # Cosine similarity of context distributions
    requested_trust: float
    substitute_trust: float
    quality_delta: Optional[float] = None  # Set after observing outcome


@dataclass
class MRHSelectionResult:
    """
    Result of MRH-based expert selection.

    Extends ExpertSelectionResult with MRH substitution tracking.
    """
    selected_expert_ids: List[int]
    selection_scores: List[float]
    router_scores: List[float]
    trust_scores: List[float]
    context: str
    mrh_substitutions: List[MRHSubstitution]  # MRH-based substitutions made
    cache_hits: int
    cache_misses: int


class MRHExpertSelector:
    """
    MRH-based expert selection combining router weights, trust scores, and
    context overlap-based substitution.

    **Session 65 Integration**:
    - Uses ContextAwareIdentityBridge for context overlap computation
    - MRH discovery: Find experts with high context overlap
    - Trust filtering: Only substitute with high-trust alternatives
    - Adaptive learning: Track substitution outcomes

    **Algorithm**:
    1. Classify input context
    2. Get contextual trust for all experts
    3. Combine: α×router_logits + (1-α)×trust_scores
    4. Select top-k by combined score
    5. For low-trust experts, find MRH alternatives
    6. Substitute if alternative has:
       - High context overlap (> 0.7)
       - Shared context with current input
       - Higher trust than requested expert

    **Key Difference from TrustBasedExpertSelector**:
    - TrustBasedExpertSelector: Substitution via semantic cluster (coarse)
    - MRHExpertSelector: Substitution via context overlap (fine-grained)
    - Result: Better specialist discovery and monopoly breaking
    """

    def __init__(
        self,
        num_experts: int = 128,
        exploration_weight: float = 0.5,
        low_trust_threshold: float = 0.3,
        overlap_threshold: float = 0.7,
        reputation_db: Optional[ExpertReputationDB] = None,
        component: str = "thinker",
        network: str = "testnet",
        context_classifier: Optional[ContextClassifier] = None
    ):
        """
        Initialize MRH-based expert selector.

        Args:
            num_experts: Total number of experts (128 for Q3-Omni thinker)
            exploration_weight: α in selection formula (0-1)
                               α=0 → pure trust (exploitation)
                               α=1 → pure router (exploration)
                               α=0.5 → balanced (Session 65 optimal)
            low_trust_threshold: Trigger MRH substitution when trust < threshold
            overlap_threshold: Minimum context overlap for MRH pairing (0.7 from Session 65)
            reputation_db: Expert reputation database
            component: "thinker" or "talker"
            network: Network identifier for LCT
            context_classifier: Optional ContextClassifier for automatic context detection
        """
        self.num_experts = num_experts
        self.exploration_weight = exploration_weight
        self.low_trust_threshold = low_trust_threshold
        self.overlap_threshold = overlap_threshold
        self.component = component
        self.network = network

        # Reputation database
        self.reputation_db = reputation_db if reputation_db else get_default_reputation_db()

        # Context classifier
        self.context_classifier = context_classifier

        # Context-aware identity bridge for MRH discovery
        self.bridge = ContextAwareIdentityBridge(
            instance=component,
            network=network,
            overlap_threshold=overlap_threshold
        )

        # MRH substitution tracking
        self.substitution_history: List[MRHSubstitution] = []
        self.generation = 0

        # Statistics
        self.total_selections = 0
        self.total_mrh_substitutions = 0

    def select_experts(
        self,
        router_logits: Union['torch.Tensor', np.ndarray],
        context: Optional[str] = None,
        k: int = 8,
        input_embedding: Optional[np.ndarray] = None,
        all_expert_ids: Optional[List[int]] = None
    ) -> MRHSelectionResult:
        """
        Select top-k experts using MRH-augmented trust-based routing.

        Args:
            router_logits: Router output scores [num_experts]
            context: Input context classification (string ID)
            k: Number of experts to select
            input_embedding: Input representation for automatic context classification
            all_expert_ids: All available expert IDs for MRH discovery
                          (defaults to range(num_experts))

        Returns:
            MRHSelectionResult with selected experts and substitution info
        """
        self.total_selections += 1
        self.generation += 1

        # Determine context
        if context is None:
            if self.context_classifier is not None and input_embedding is not None:
                context_info = self.context_classifier.classify(input_embedding)
                context = context_info.context_id
            else:
                context = "general"

        # Default all_expert_ids
        if all_expert_ids is None:
            all_expert_ids = list(range(self.num_experts))

        # Convert router logits to numpy
        if HAS_TORCH and torch is not None and isinstance(router_logits, torch.Tensor):
            router_scores = router_logits.detach().cpu().numpy().astype(np.float32)
        else:
            router_scores = np.array(router_logits, dtype=np.float32)

        # Get contextual trust for each expert
        trust_scores = self._get_contextual_trust_scores(context)

        # Combine router weights with trust
        # Formula: selection = α×router + (1-α)×trust
        α = self.exploration_weight
        combined_scores = α * router_scores + (1 - α) * trust_scores

        # Select top-k experts by combined score
        top_k_indices = np.argsort(combined_scores)[-k:][::-1]
        selected_scores = combined_scores[top_k_indices]
        selected_router = router_scores[top_k_indices]
        selected_trust = trust_scores[top_k_indices]

        # MRH substitution: Replace low-trust experts with high-trust alternatives
        final_experts = []
        mrh_substitutions = []

        for i, expert_id in enumerate(top_k_indices):
            expert_trust = selected_trust[i]

            # Check if trust is below threshold
            if expert_trust < self.low_trust_threshold:
                # Try to find MRH alternative
                alternative = self._find_mrh_alternative(
                    expert_id,
                    context,
                    all_expert_ids
                )

                if alternative is not None:
                    alt_id, alt_trust, overlap = alternative

                    # Record substitution
                    sub = MRHSubstitution(
                        generation=self.generation,
                        requested_expert=int(expert_id),
                        substitute_expert=alt_id,
                        context=context,
                        context_overlap=overlap,
                        requested_trust=float(expert_trust),
                        substitute_trust=alt_trust
                    )
                    mrh_substitutions.append(sub)
                    self.substitution_history.append(sub)
                    self.total_mrh_substitutions += 1

                    # Use alternative
                    final_experts.append(alt_id)
                    continue

            # No substitution needed or no alternative found
            final_experts.append(int(expert_id))

        return MRHSelectionResult(
            selected_expert_ids=final_experts,
            selection_scores=selected_scores.tolist(),
            router_scores=selected_router.tolist(),
            trust_scores=selected_trust.tolist(),
            context=context,
            mrh_substitutions=mrh_substitutions,
            cache_hits=0,  # Not tracking cache in this version
            cache_misses=0
        )

    def _get_contextual_trust_scores(self, context: str) -> np.ndarray:
        """
        Get trust scores for all experts in given context.

        Reads from bridge.trust_history (MRH source of truth).
        Falls back to reputation_db if not in bridge.

        Args:
            context: Context identifier

        Returns:
            Array of trust scores [num_experts], range 0-1
        """
        trust_scores = np.zeros(self.num_experts, dtype=np.float32)

        for expert_id in range(self.num_experts):
            # Use same logic as _get_context_trust
            trust_scores[expert_id] = self._get_context_trust(expert_id, context)

        return trust_scores

    def _find_mrh_alternative(
        self,
        expert_id: int,
        context: str,
        all_expert_ids: List[int]
    ) -> Optional[Tuple[int, float, float]]:
        """
        Find MRH alternative expert via context overlap.

        **Session 65 Core Algorithm**:
        1. Compute context overlap between requested expert and all others
        2. Filter for:
           - High overlap (> overlap_threshold)
           - Shared context with current input
           - Higher trust than requested expert
        3. Return best alternative by trust

        Args:
            expert_id: Expert with low trust
            context: Current context
            all_expert_ids: All available experts

        Returns:
            (alternative_id, trust_score, overlap) or None
        """
        # Check if expert is in bridge (has context distribution)
        if expert_id not in self.bridge.expert_contexts:
            return None

        # Find experts with high context overlap
        alternatives = []

        for other_expert in all_expert_ids:
            if other_expert == expert_id:
                continue
            if other_expert not in self.bridge.expert_contexts:
                continue

            # Compute context overlap
            overlap, shared = self.bridge.compute_context_overlap(expert_id, other_expert)

            # Check if high overlap and context is shared
            # CRITICAL FIX from Session 65: context must match format in shared list
            # shared contains strings like 'context_0', 'context_1', 'context_2'
            if isinstance(context, int):
                context_str = f"context_{context}"
            elif isinstance(context, str) and not context.startswith("context_"):
                # If it's a string but not in right format, try to convert
                try:
                    context_num = int(context)
                    context_str = f"context_{context_num}"
                except ValueError:
                    context_str = context if context.startswith("context_") else f"context_{context}"
            else:
                context_str = context

            if overlap >= self.overlap_threshold and context_str in shared:
                # Check trust for alternative
                alt_trust = self._get_context_trust(other_expert, context)

                # Only substitute if alternative has better trust
                current_trust = self._get_context_trust(expert_id, context)
                if alt_trust > current_trust:
                    alternatives.append((other_expert, alt_trust, overlap))

        if not alternatives:
            return None

        # Return best alternative by trust
        return max(alternatives, key=lambda x: x[1])

    def _get_context_trust(self, expert_id: int, context: str) -> float:
        """
        Get trust score for expert in specific context.

        Reads from bridge.trust_history (source of truth for MRH).
        Falls back to reputation_db if not in bridge.
        """
        # Try bridge first (MRH source of truth)
        key = (expert_id, context)
        if key in self.bridge.trust_history:
            history = self.bridge.trust_history[key]
            if history:
                return history[-1]  # Most recent trust value

        # Fallback to reputation DB
        rep = self.reputation_db.get_reputation(expert_id, self.component)
        if rep is None:
            return 0.5

        return rep.get_context_trust(context, default=0.5)

    def register_expert_contexts(
        self,
        expert_id: int,
        embeddings: np.ndarray
    ) -> List[str]:
        """
        Register expert's context distribution via embedding clustering.

        This populates the bridge's expert_contexts for MRH discovery.

        Args:
            expert_id: Expert ID
            embeddings: [N, embedding_dim] expert activations

        Returns:
            List of discovered contexts
        """
        contexts = self.bridge.discover_expert_contexts(expert_id, embeddings)
        return contexts

    def update_trust_for_expert(
        self,
        expert_id: int,
        context: str,
        quality: float
    ):
        """
        Update trust score for expert in context.

        Args:
            expert_id: Expert ID
            context: Context identifier
            quality: Observed quality (0-1)
        """
        self.bridge.update_trust_history(expert_id, context, quality)

    def get_statistics(self) -> Dict:
        """
        Get selector statistics.

        Returns:
            Dictionary with selection and substitution stats
        """
        return {
            'total_selections': self.total_selections,
            'total_mrh_substitutions': self.total_mrh_substitutions,
            'substitution_rate': self.total_mrh_substitutions / max(1, self.total_selections * 8),
            'exploration_weight': self.exploration_weight,
            'low_trust_threshold': self.low_trust_threshold,
            'overlap_threshold': self.overlap_threshold,
            'generation': self.generation,
            'unique_experts_used': len(set(
                s.substitute_expert for s in self.substitution_history
            ) | set(
                s.requested_expert for s in self.substitution_history
            ))
        }

    def get_substitution_summary(self) -> Dict:
        """
        Get summary of MRH substitutions by context.

        Returns:
            Dictionary with substitution analysis
        """
        if not self.substitution_history:
            return {"total": 0, "by_context": {}}

        by_context = {}
        for sub in self.substitution_history:
            if sub.context not in by_context:
                by_context[sub.context] = {
                    'count': 0,
                    'avg_overlap': 0.0,
                    'avg_trust_improvement': 0.0,
                    'experts_substituted': set(),
                    'experts_used': set()
                }

            ctx_stats = by_context[sub.context]
            ctx_stats['count'] += 1
            ctx_stats['avg_overlap'] += sub.context_overlap
            ctx_stats['avg_trust_improvement'] += (sub.substitute_trust - sub.requested_trust)
            ctx_stats['experts_substituted'].add(sub.requested_expert)
            ctx_stats['experts_used'].add(sub.substitute_expert)

        # Compute averages
        for ctx, stats in by_context.items():
            count = stats['count']
            stats['avg_overlap'] /= count
            stats['avg_trust_improvement'] /= count
            stats['experts_substituted'] = len(stats['experts_substituted'])
            stats['experts_used'] = len(stats['experts_used'])

        return {
            'total': len(self.substitution_history),
            'by_context': by_context
        }


# Convenience functions

def create_mrh_selector(
    num_experts: int = 128,
    component: str = "thinker",
    network: str = "testnet",
    context_classifier: Optional[ContextClassifier] = None
) -> MRHExpertSelector:
    """
    Create MRH-based expert selector with Session 65 optimal parameters.

    Args:
        num_experts: Total experts available
        component: "thinker" or "talker"
        network: Network identifier
        context_classifier: Optional ContextClassifier

    Returns:
        Configured MRHExpertSelector
    """
    return MRHExpertSelector(
        num_experts=num_experts,
        exploration_weight=0.5,      # Session 65 optimal
        low_trust_threshold=0.3,     # Session 65 threshold
        overlap_threshold=0.7,       # Session 65 threshold
        component=component,
        network=network,
        context_classifier=context_classifier
    )
