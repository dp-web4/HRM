#!/usr/bin/env python3
"""
Trust-Based Expert Selector - Web4 Pattern Applied to SAGE MoE
=============================================================

Implements trust-augmented expert selection combining router learned weights
with empirical expert reputation.

**Core Innovation**: Expert selection = trust problem
- Router weights = learned preferences (baseline)
- Observed reputation = empirical evidence (adaptive)
- Combined decision = exploration + exploitation balance

**Web4 Pattern**: Trust-based routing with contextual reliability

**Architecture**:
- Combines router logits with contextual trust scores
- Enables smart expert substitution when cache limited
- Tracks expert usage for reputation learning
- Supports federation-ready reputation sharing

**Author**: Claude (Legion Autonomous Web4 Research Session 56)
**Date**: 2025-12-16
**Provenance**: Web4 S55 (expert reputation design) → SAGE implementation
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

from sage.core.expert_reputation import (
    ExpertReputation,
    ExpertReputationDB,
    get_default_reputation_db
)
from sage.core.context_classifier import ContextClassifier


@dataclass
class ExpertSelectionResult:
    """
    Result of trust-based expert selection.

    Contains:
    - Selected expert IDs
    - Selection scores (combined router + trust)
    - Substitution info (if substitutions were made)
    - Context classification
    """
    selected_expert_ids: List[int]      # Expert IDs selected
    selection_scores: List[float]       # Combined scores for selected
    router_scores: List[float]          # Original router logits
    trust_scores: List[float]           # Trust scores applied
    context: str                         # Classified context
    substitutions: Dict[int, int]        # {requested_id: substitute_id}
    cache_hits: int                      # How many were already in cache
    cache_misses: int                    # How many needed loading


class TrustBasedExpertSelector:
    """
    Trust-based expert selection combining router weights with reputation.

    **Web4 Pattern Applied**:
    - Entity selection → Expert selection
    - Contextual trust → Expert context-specific reliability
    - Delegation → Expert substitution when preferred unavailable

    **Algorithm**:
    1. Classify input context
    2. Get contextual trust for all experts
    3. Combine: α×router_logits + (1-α)×trust_scores
    4. Select top-k by combined score
    5. Check cache, substitute if needed

    **Substitution Logic**:
    - Requested expert not in cache?
    - Find similar expert in cache (same cluster, high trust)
    - Use substitute with quality tracking
    - Learn: Does substitution work? Update reputation
    """

    def __init__(
        self,
        num_experts: int = 128,
        cache_size: int = 6,
        exploration_weight: float = 0.3,
        substitution_threshold: float = 0.6,
        reputation_db: Optional[ExpertReputationDB] = None,
        component: str = "thinker",
        context_classifier: Optional[ContextClassifier] = None
    ):
        """
        Initialize trust-based expert selector.

        Args:
            num_experts: Total number of experts (128 for Q3-Omni thinker)
            cache_size: How many experts can fit in memory
            exploration_weight: α in selection formula (0-1)
                               α=0 → pure reputation (exploitation)
                               α=1 → pure router (exploration)
                               α=0.3 → balanced (default)
            substitution_threshold: Minimum trust score for substitution
            reputation_db: Expert reputation database (creates default if None)
            component: "thinker" or "talker"
            context_classifier: Optional ContextClassifier for automatic context detection
                              If provided, uses input_embedding to classify context
                              If None, requires manual context string in select_experts()
        """
        self.num_experts = num_experts
        self.cache_size = cache_size
        self.exploration_weight = exploration_weight
        self.substitution_threshold = substitution_threshold
        self.component = component

        # Reputation database
        self.reputation_db = reputation_db if reputation_db else get_default_reputation_db()

        # Context classifier (optional)
        self.context_classifier = context_classifier

        # Expert cache (expert_id → loaded status)
        self.loaded_experts: Dict[int, bool] = {}

        # Statistics
        self.total_selections = 0
        self.total_substitutions = 0
        self.cache_hit_rate = 0.0

    def select_experts(
        self,
        router_logits: Union['torch.Tensor', np.ndarray],
        context: Optional[str] = None,
        k: int = 8,
        input_embedding: Optional[np.ndarray] = None
    ) -> ExpertSelectionResult:
        """
        Select top-k experts using trust-augmented routing.

        Args:
            router_logits: Router output scores [num_experts] (torch.Tensor or numpy array)
            context: Input context classification (string ID)
                    If None and context_classifier provided, will classify input_embedding
                    If None and no classifier, uses "general" as default context
            k: Number of experts to select
            input_embedding: Input representation for automatic context classification
                           Required if context=None and context_classifier is provided

        Returns:
            ExpertSelectionResult with selected experts and metadata
        """
        self.total_selections += 1

        # Determine context
        if context is None:
            if self.context_classifier is not None and input_embedding is not None:
                # Use context classifier to determine context from embedding
                context_info = self.context_classifier.classify(input_embedding)
                context = context_info.context_id
            else:
                # No context provided and no way to classify: use default
                context = "general"

        # Convert router logits to numpy for easier manipulation (ensure float32)
        if HAS_TORCH and torch is not None and isinstance(router_logits, torch.Tensor):
            router_scores = router_logits.detach().cpu().numpy().astype(np.float32)
        else:
            router_scores = np.array(router_logits, dtype=np.float32)

        # Get contextual trust for each expert
        trust_scores = self._get_contextual_trust_scores(context)

        # Combine router weights with observed trust
        # Formula: selection = α×router + (1-α)×trust
        α = self.exploration_weight
        combined_scores = α * router_scores + (1 - α) * trust_scores

        # Select top-k experts by combined score
        top_k_indices = np.argsort(combined_scores)[-k:][::-1]  # Descending order
        selected_scores = combined_scores[top_k_indices]
        selected_router = router_scores[top_k_indices]
        selected_trust = trust_scores[top_k_indices]

        # Check cache availability and handle substitutions
        final_experts = []
        substitutions = {}
        cache_hits = 0
        cache_misses = 0

        for expert_id in top_k_indices:
            if self._is_expert_loaded(expert_id):
                # Expert in cache: use directly
                final_experts.append(expert_id)
                cache_hits += 1
            else:
                # Expert not in cache: try substitution
                substitute = self._find_substitute(expert_id, context)

                if substitute is not None:
                    # Use substitute
                    final_experts.append(substitute)
                    substitutions[int(expert_id)] = substitute
                    self.total_substitutions += 1
                    cache_hits += 1  # Substitute was in cache
                else:
                    # No suitable substitute: use requested expert
                    # (Will need to be loaded, evicting if necessary)
                    final_experts.append(expert_id)
                    cache_misses += 1

        # Update cache hit rate
        self.cache_hit_rate = (
            (self.cache_hit_rate * (self.total_selections - 1) + (cache_hits / k))
            / self.total_selections
        )

        return ExpertSelectionResult(
            selected_expert_ids=[int(e) for e in final_experts],
            selection_scores=selected_scores.tolist(),
            router_scores=selected_router.tolist(),
            trust_scores=selected_trust.tolist(),
            context=context,
            substitutions=substitutions,
            cache_hits=cache_hits,
            cache_misses=cache_misses
        )

    def _get_contextual_trust_scores(self, context: str) -> np.ndarray:
        """
        Get trust scores for all experts in given context.

        Args:
            context: Context identifier

        Returns:
            Array of trust scores [num_experts], range 0-1
        """
        trust_scores = np.zeros(self.num_experts, dtype=np.float32)

        for expert_id in range(self.num_experts):
            rep = self.reputation_db.get_reputation(expert_id, self.component)

            if rep is None:
                # Unknown expert: neutral prior (exploration)
                trust_scores[expert_id] = 0.5
            else:
                # Known expert: use contextual trust
                trust_scores[expert_id] = rep.get_context_trust(context, default=0.5)

        return trust_scores

    def _is_expert_loaded(self, expert_id: int) -> bool:
        """
        Check if expert is currently loaded in cache.

        Args:
            expert_id: Expert to check

        Returns:
            True if loaded, False otherwise
        """
        return self.loaded_experts.get(expert_id, False)

    def _find_substitute(self, requested_expert: int, context: str) -> Optional[int]:
        """
        Find suitable substitute expert already in cache.

        **Web4 Pattern**: Delegation to similar trusted entity when preferred unavailable

        Substitution criteria:
        1. Expert is loaded in cache
        2. Same semantic cluster (similar specialization)
        3. High trust in current context (> substitution_threshold)
        4. Historical substitution success (optional boost)

        Args:
            requested_expert: Expert that was requested
            context: Current input context

        Returns:
            Expert ID of substitute, or None if no suitable substitute
        """
        requested_rep = self.reputation_db.get_reputation(requested_expert, self.component)

        if requested_rep is None:
            # Unknown expert, can't find similar
            return None

        # Find loaded experts with similar specialization and high context trust
        candidates = []

        for loaded_id in self.loaded_experts.keys():
            if not self.loaded_experts[loaded_id]:
                continue  # Not actually loaded

            loaded_rep = self.reputation_db.get_reputation(loaded_id, self.component)
            if loaded_rep is None:
                continue

            # Similarity criteria:
            # 1. Same semantic cluster (if known)
            if (requested_rep.semantic_cluster >= 0 and
                loaded_rep.semantic_cluster >= 0 and
                loaded_rep.semantic_cluster != requested_rep.semantic_cluster):
                continue  # Different cluster, not similar enough

            # 2. High trust in current context
            context_trust = loaded_rep.get_context_trust(context, default=0.5)

            if context_trust < self.substitution_threshold:
                continue  # Trust too low for substitution

            # 3. Check historical substitution success (if any)
            substitution_quality = 0.0
            for sub_id, quality_delta, sub_context in loaded_rep.substituted_for:
                if sub_id == requested_expert and sub_context == context:
                    substitution_quality = quality_delta
                    break

            # Substitution score: trust + historical quality
            score = 0.7 * context_trust + 0.3 * (0.5 + substitution_quality)

            candidates.append((loaded_id, score))

        if not candidates:
            return None

        # Return best substitute
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def mark_expert_loaded(self, expert_id: int, loaded: bool = True):
        """
        Mark expert as loaded/unloaded in cache.

        Args:
            expert_id: Expert ID
            loaded: True if loaded, False if evicted
        """
        self.loaded_experts[expert_id] = loaded

    def mark_experts_loaded(self, expert_ids: List[int]):
        """
        Mark multiple experts as loaded.

        Args:
            expert_ids: List of expert IDs now in cache
        """
        for expert_id in expert_ids:
            self.loaded_experts[expert_id] = True

    def get_statistics(self) -> Dict:
        """
        Get selector statistics.

        Returns:
            Dictionary with selection stats
        """
        return {
            'total_selections': self.total_selections,
            'total_substitutions': self.total_substitutions,
            'substitution_rate': self.total_substitutions / max(1, self.total_selections * 8),
            'cache_hit_rate': self.cache_hit_rate,
            'experts_loaded': sum(1 for loaded in self.loaded_experts.values() if loaded),
            'cache_size': self.cache_size,
            'exploration_weight': self.exploration_weight
        }

    def should_evict_expert(self, expert_id: int, context: str) -> bool:
        """
        Determine if expert should be evicted from cache.

        Eviction criteria (prefer keeping):
        - High trust in current context
        - High co-activation frequency
        - Recent usage

        Args:
            expert_id: Expert to potentially evict
            context: Current context

        Returns:
            True if should evict, False if should keep
        """
        rep = self.reputation_db.get_reputation(expert_id, self.component)

        if rep is None:
            return True  # Unknown expert, safe to evict

        # Get context trust
        context_trust = rep.get_context_trust(context, default=0.5)

        # Get recency (time since last use)
        import time
        if rep.last_used is None:
            recency = 0.0
        else:
            time_since_use = time.time() - rep.last_used
            recency = 1.0 / (1.0 + time_since_use / 3600)  # Decay over hours

        # Eviction score: Lower = more likely to evict
        # High trust + recent use = keep in cache
        eviction_score = 0.6 * context_trust + 0.4 * recency

        # Evict if score below threshold
        return eviction_score < 0.4


# Convenience functions

def create_trust_based_selector(
    num_experts: int = 128,
    cache_size: int = 6,
    component: str = "thinker",
    context_classifier: Optional[ContextClassifier] = None
) -> TrustBasedExpertSelector:
    """
    Create trust-based expert selector with defaults.

    Args:
        num_experts: Total experts available
        cache_size: Memory capacity (number of experts)
        component: "thinker" or "talker"
        context_classifier: Optional ContextClassifier for automatic context detection

    Returns:
        Configured TrustBasedExpertSelector
    """
    return TrustBasedExpertSelector(
        num_experts=num_experts,
        cache_size=cache_size,
        exploration_weight=0.3,  # Balanced exploration/exploitation
        substitution_threshold=0.6,
        component=component,
        context_classifier=context_classifier
    )


def select_experts_with_trust(
    router_logits: Union['torch.Tensor', np.ndarray],
    context: str,
    k: int = 8,
    selector: Optional[TrustBasedExpertSelector] = None
) -> ExpertSelectionResult:
    """
    Convenience function for trust-based expert selection.

    Args:
        router_logits: Router output [num_experts] (torch.Tensor or numpy array)
        context: Input context
        k: Number of experts to select
        selector: Selector instance (creates default if None)

    Returns:
        ExpertSelectionResult
    """
    if selector is None:
        selector = create_trust_based_selector()

    return selector.select_experts(router_logits, context, k)
