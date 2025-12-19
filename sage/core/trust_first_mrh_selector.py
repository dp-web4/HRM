#!/usr/bin/env python3
"""
Trust-First MRH Expert Selector - Session 68 Paradigm Shift

Implements Thor's Session 72 discovery: Conditional trust-first architecture
instead of weighted blending.

**Paradigm Shift**:
OLD (Sessions 64-67): selection = α × router + (1-α) × trust  (weighted blend)
NEW (Session 68):      if has_trust → pure_trust else free_router  (conditional)

**Results**:
- Weighted blend (α=0.3): 17 experts (13% utilization)
- Trust-first: 58 experts (45% utilization)
- Improvement: 3.4x more diversity

**Architecture Principle**:
Don't blend centralized authority (router) with distributed trust.
Let each mechanism work purely when appropriate.

**Web4 Pattern**: Distributed trust > Centralized authority
- When trust has evidence → pure trust (no router bias)
- When bootstrapping → free exploration (no α constraint)
- Never blend (avoids reinforcing monopoly)

Created: Session 68 (2025-12-18)
Based on: Thor's Session 72 breakthrough
"""

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict

from sage.core.expert_reputation import ExpertReputationDB, get_default_reputation_db
from sage.core.context_classifier import ContextClassifier
from sage.web4.context_aware_identity_bridge import ContextAwareIdentityBridge


@dataclass
class TrustFirstSelectionResult:
    """Result of trust-first expert selection."""
    selected_expert_ids: List[int]
    selection_mode: str  # "trust_driven" or "router_explore"
    trust_evidence: bool
    context: str
    trust_scores: List[float]
    mrh_substitutions: int
    quality_checks: int  # How many low-quality detections triggered recovery
    selection_scores: List[float]  # Normalized weights for expert mixing (Session 75 API fix)


class TrustFirstMRHSelector:
    """
    Trust-first expert selector using conditional logic, not weighted blending.

    **Thor's Session 72 Architecture**:
    1. Check if we have trust evidence for this context
    2. IF evidence exists → Pure trust selection (100% trust, 0% router)
    3. ELSE → Free router exploration (no α constraint)
    4. Quality monitoring → Recovery on declining trust

    **Key Difference from mrh_expert_selector.py**:
    - NO weighted blending (α × router + (1-α) × trust)
    - Conditional logic: trust OR router, never both
    - Result: 3.4x more diversity

    **When to Use Trust**:
    - Sufficient evidence: ≥3 samples in context
    - Sufficient diversity: ≥2 experts with trust > threshold
    - Otherwise: Let router explore freely

    **MRH Discovery**:
    - Still used for finding alternatives when trust declines
    - But selection is 100% trust-driven (no router influence)
    """

    def __init__(
        self,
        num_experts: int = 128,
        min_trust_evidence: int = 3,
        low_trust_threshold: float = 0.3,
        overlap_threshold: float = 0.7,
        reputation_db: Optional[ExpertReputationDB] = None,
        component: str = "thinker",
        network: str = "testnet",
        context_classifier: Optional[ContextClassifier] = None,
        epsilon: float = 0.0
    ):
        """
        Initialize trust-first MRH selector.

        Args:
            num_experts: Total number of experts
            min_trust_evidence: Minimum samples needed to use trust-driven mode
            low_trust_threshold: Trigger MRH when trust < threshold
            overlap_threshold: Minimum context overlap for MRH pairing
            reputation_db: Expert reputation database
            component: "thinker" or "talker"
            network: Network identifier
            context_classifier: Optional context classifier
            epsilon: Probability of forced random exploration (0.0-1.0, Session 77)
        """
        self.num_experts = num_experts
        self.min_trust_evidence = min_trust_evidence
        self.low_trust_threshold = low_trust_threshold
        self.overlap_threshold = overlap_threshold
        self.component = component
        self.network = network
        self.epsilon = epsilon

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

        # Statistics
        self.total_selections = 0
        self.trust_driven_selections = 0
        self.router_explore_selections = 0
        self.forced_exploration_selections = 0  # Session 77: epsilon-greedy tracking
        self.total_mrh_substitutions = 0
        self.quality_checks = 0
        self.generation = 0

    def select_experts(
        self,
        router_logits: Union['torch.Tensor', np.ndarray],
        context: Optional[str] = None,
        k: int = 8,
        input_embedding: Optional[np.ndarray] = None,
        all_expert_ids: Optional[List[int]] = None
    ) -> TrustFirstSelectionResult:
        """
        Select experts using trust-first conditional logic with epsilon-greedy exploration.

        **Architecture (Session 77 - Epsilon-Greedy)**:
        1. With probability epsilon → forced_exploration (random)
        2. Check trust evidence for context
        3. IF sufficient evidence → trust_driven_selection()
        4. ELSE → router_explore_selection()

        No blending. Pure conditional with forced exploration.

        Args:
            router_logits: Router output [num_experts]
            context: Context ID
            k: Number of experts to select
            input_embedding: For auto context classification
            all_expert_ids: All available expert IDs

        Returns:
            TrustFirstSelectionResult
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

        if all_expert_ids is None:
            all_expert_ids = list(range(self.num_experts))

        # Session 77: Epsilon-greedy forced exploration
        if self.epsilon > 0 and np.random.random() < self.epsilon:
            # FORCED-EXPLORATION: Random selection to break monopoly
            result = self._forced_exploration_selection(context, k)
            self.forced_exploration_selections += 1
            return result

        # Convert router logits
        if HAS_TORCH and torch is not None and isinstance(router_logits, torch.Tensor):
            router_scores = router_logits.detach().cpu().numpy().astype(np.float32)
        else:
            router_scores = np.array(router_logits, dtype=np.float32)

        # Check if we have sufficient trust evidence
        has_trust_evidence = self._has_sufficient_trust_evidence(context)

        if has_trust_evidence:
            # TRUST-DRIVEN: 100% trust, 0% router
            result = self._trust_driven_selection(context, k, all_expert_ids)
            self.trust_driven_selections += 1
        else:
            # ROUTER-EXPLORE: Free exploration, no α constraint
            result = self._router_explore_selection(router_scores, context, k)
            self.router_explore_selections += 1

        return result

    def _has_sufficient_trust_evidence(self, context: str) -> bool:
        """
        Check if we have enough trust evidence to use trust-driven mode.

        Criteria:
        1. At least min_trust_evidence samples in context
        2. At least 2 experts with trust > threshold
        3. Trust diversity (not all concentrated in one expert)

        Returns:
            True if trust-driven mode should be used
        """
        # Count experts with trust evidence
        experts_with_evidence = []
        for expert_id in range(self.num_experts):
            key = (expert_id, context)
            if key in self.bridge.trust_history:
                history = self.bridge.trust_history[key]
                if len(history) >= self.min_trust_evidence:
                    trust = history[-1]
                    if trust > self.low_trust_threshold:
                        experts_with_evidence.append((expert_id, trust))

        # Need at least 2 experts with sufficient trust
        return len(experts_with_evidence) >= 2

    def _trust_driven_selection(
        self,
        context: str,
        k: int,
        all_expert_ids: List[int]
    ) -> TrustFirstSelectionResult:
        """
        Pure trust-driven selection (0% router influence).

        Select top-k experts by trust score alone.
        Apply MRH substitution for low-trust experts.

        Args:
            context: Current context
            k: Number to select
            all_expert_ids: All available experts

        Returns:
            TrustFirstSelectionResult
        """
        # Get trust scores for all experts
        trust_scores = np.array([self._get_context_trust(eid, context) for eid in range(self.num_experts)])

        # Select top-k by trust (pure trust, no router)
        top_k_indices = np.argsort(trust_scores)[-k:][::-1]
        selected_experts = []
        mrh_subs = 0

        for expert_id in top_k_indices:
            expert_trust = trust_scores[expert_id]

            # Quality check: if trust too low, use MRH
            if expert_trust < self.low_trust_threshold:
                alternative = self._find_mrh_alternative(expert_id, context, all_expert_ids)
                if alternative:
                    selected_experts.append(alternative[0])
                    mrh_subs += 1
                    self.total_mrh_substitutions += 1
                else:
                    selected_experts.append(int(expert_id))
            else:
                selected_experts.append(int(expert_id))

        # Normalize trust scores for selected experts to get selection weights
        selected_trust_scores = trust_scores[top_k_indices]
        trust_sum = np.sum(selected_trust_scores)
        if trust_sum > 0:
            selection_weights = (selected_trust_scores / trust_sum).tolist()
        else:
            # Fallback: uniform weights
            selection_weights = [1.0 / k] * k

        return TrustFirstSelectionResult(
            selected_expert_ids=selected_experts,
            selection_mode="trust_driven",
            trust_evidence=True,
            context=context,
            trust_scores=trust_scores[top_k_indices].tolist(),
            mrh_substitutions=mrh_subs,
            quality_checks=0,
            selection_scores=selection_weights  # Session 75: Normalized mixing weights
        )

    def _router_explore_selection(
        self,
        router_scores: np.ndarray,
        context: str,
        k: int
    ) -> TrustFirstSelectionResult:
        """
        Free router exploration (no α constraint).

        When bootstrapping, let router explore freely.
        No trust blending that could pull toward monopoly.

        Args:
            router_scores: Router output
            context: Current context
            k: Number to select

        Returns:
            TrustFirstSelectionResult
        """
        # Pure router selection (100% router, 0% trust)
        top_k_indices = np.argsort(router_scores)[-k:][::-1]

        # Normalize router scores for selected experts to get selection weights
        selected_router_scores = router_scores[top_k_indices]
        # Use softmax for proper probability distribution
        exp_scores = np.exp(selected_router_scores - np.max(selected_router_scores))
        selection_weights = (exp_scores / np.sum(exp_scores)).tolist()

        return TrustFirstSelectionResult(
            selected_expert_ids=[int(i) for i in top_k_indices],
            selection_mode="router_explore",
            trust_evidence=False,
            context=context,
            trust_scores=[],
            mrh_substitutions=0,
            quality_checks=0,
            selection_scores=selection_weights  # Session 75: Normalized mixing weights from router
        )

    def _forced_exploration_selection(
        self,
        context: str,
        k: int
    ) -> TrustFirstSelectionResult:
        """
        Forced random exploration to break router monopoly (Session 77).

        Select k experts uniformly at random from all available experts.
        This enables evidence gathering for experts that the router never selects.

        **Purpose**: Break chicken-and-egg problem discovered in Session 76:
        - Router monopoly prevents diversity
        - Trust needs diversity to create diversity
        - Forced exploration provides bootstrap diversity

        Args:
            context: Current context
            k: Number to select

        Returns:
            TrustFirstSelectionResult with random selections
        """
        # Uniform random selection
        selected_expert_ids = np.random.choice(self.num_experts, size=k, replace=False).tolist()

        # Uniform weights for random selection
        selection_weights = [1.0 / k] * k

        return TrustFirstSelectionResult(
            selected_expert_ids=selected_expert_ids,
            selection_mode="forced_exploration",
            trust_evidence=False,
            context=context,
            trust_scores=[],
            mrh_substitutions=0,
            quality_checks=0,
            selection_scores=selection_weights
        )

    def _get_context_trust(self, expert_id: int, context: str) -> float:
        """Get trust for expert in context."""
        key = (expert_id, context)
        if key in self.bridge.trust_history:
            history = self.bridge.trust_history[key]
            if history:
                return history[-1]

        # Fallback to reputation DB
        rep = self.reputation_db.get_reputation(expert_id, self.component)
        if rep:
            return rep.get_context_trust(context, default=0.5)

        return 0.5

    def _find_mrh_alternative(
        self,
        expert_id: int,
        context: str,
        all_expert_ids: List[int]
    ) -> Optional[Tuple[int, float, float]]:
        """Find MRH alternative (same as mrh_expert_selector)."""
        if expert_id not in self.bridge.expert_contexts:
            return None

        alternatives = []
        for other_expert in all_expert_ids:
            if other_expert == expert_id or other_expert not in self.bridge.expert_contexts:
                continue

            overlap, shared = self.bridge.compute_context_overlap(expert_id, other_expert)

            # Handle context string format
            if isinstance(context, int):
                context_str = f"context_{context}"
            elif isinstance(context, str) and not context.startswith("context_"):
                try:
                    context_num = int(context)
                    context_str = f"context_{context_num}"
                except ValueError:
                    context_str = context if context.startswith("context_") else f"context_{context}"
            else:
                context_str = context

            if overlap >= self.overlap_threshold and context_str in shared:
                alt_trust = self._get_context_trust(other_expert, context)
                current_trust = self._get_context_trust(expert_id, context)

                if alt_trust > current_trust:
                    alternatives.append((other_expert, alt_trust, overlap))

        return max(alternatives, key=lambda x: x[1]) if alternatives else None

    def register_expert_contexts(self, expert_id: int, embeddings: np.ndarray) -> List[str]:
        """Register expert's context distribution."""
        return self.bridge.discover_expert_contexts(expert_id, embeddings)

    def update_trust_for_expert(self, expert_id: int, context: str, quality: float):
        """Update trust for expert in context."""
        self.bridge.update_trust_history(expert_id, context, quality)

    def get_statistics(self) -> Dict:
        """Get selector statistics."""
        return {
            'total_selections': self.total_selections,
            'trust_driven': self.trust_driven_selections,
            'router_explore': self.router_explore_selections,
            'forced_exploration': self.forced_exploration_selections,  # Session 77
            'trust_driven_rate': self.trust_driven_selections / max(1, self.total_selections),
            'forced_exploration_rate': self.forced_exploration_selections / max(1, self.total_selections),
            'total_mrh_substitutions': self.total_mrh_substitutions,
            'generation': self.generation
        }


def create_trust_first_selector(
    num_experts: int = 128,
    component: str = "thinker",
    network: str = "testnet",
    context_classifier: Optional[ContextClassifier] = None
) -> TrustFirstMRHSelector:
    """Create trust-first selector with Thor's Session 72 architecture."""
    return TrustFirstMRHSelector(
        num_experts=num_experts,
        min_trust_evidence=3,  # Need 3+ samples to trust
        low_trust_threshold=0.3,
        overlap_threshold=0.7,
        component=component,
        network=network,
        context_classifier=context_classifier
    )
