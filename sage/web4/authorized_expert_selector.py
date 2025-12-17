#!/usr/bin/env python3
"""
Authorized Expert Selector - Complete Web4 ↔ SAGE Integration

Extends TrustBasedExpertSelector with:
1. ATP-based cache allocation
2. Authorization checks (ACT integration stub)
3. Trust tensor synchronization
4. Complete Web4 economic model

This is the culmination of Web4 ↔ SAGE integration:
- Session 59: ExpertIdentityBridge (expert_id ↔ LCT ID)
- Session 60: ATPResourceAllocator (ATP economics)
- Session 61: TrustTensorSync (reputation sync)
- Session 61: AuthorizedExpertSelector ← THIS FILE

Design Philosophy:
- Extends, doesn't replace: TrustBasedExpertSelector still works
- Optional integration: Can disable auth/ATP/sync individually
- Production-ready: Handles failures gracefully
- Observable: Tracks all integration points

Created: Session 61 (2025-12-16)
Part of: Web4 ↔ SAGE integration (Session 57 design)
"""

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

try:
    from sage.core.trust_based_expert_selector import (
        TrustBasedExpertSelector,
        ExpertSelectionResult
    )
    from sage.web4.expert_identity import ExpertIdentityBridge
    from sage.web4.atp_allocator import ATPResourceAllocator
    from sage.web4.trust_tensor_sync import TrustTensorSync
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.trust_based_expert_selector import (
        TrustBasedExpertSelector,
        ExpertSelectionResult
    )
    from web4.expert_identity import ExpertIdentityBridge
    from web4.atp_allocator import ATPResourceAllocator
    from web4.trust_tensor_sync import TrustTensorSync


# =============================================================================
# AUTHORIZATION CLIENT (STUB)
# =============================================================================

@dataclass
class AuthorizationResult:
    """Result of authorization check."""
    allowed: bool
    reason: str = ""
    policy_id: Optional[str] = None


class Web4AuthClient:
    """
    Stub implementation of Web4 ACT authorization client.

    In production, this would connect to Web4 ACT (Agent Coordination
    with Trust) system for authorization decisions.

    For now, implements simple allow/deny lists for development.
    """

    def __init__(self, default_allow: bool = True):
        """
        Initialize authorization client.

        Args:
            default_allow: If True, allow by default. If False, deny by default.
        """
        self.default_allow = default_allow
        self.allow_list: Dict[str, List[str]] = {}  # agent_id → [resource_ids]
        self.deny_list: Dict[str, List[str]] = {}   # agent_id → [resource_ids]

    def check_authorization(
        self,
        agent: str,
        resource: str,
        action: str = "select",
        context: Optional[str] = None
    ) -> bool:
        """
        Check if agent is authorized to perform action on resource.

        Args:
            agent: LCT ID of requesting agent
            resource: LCT ID of resource (expert)
            action: Action to perform (e.g., "select")
            context: Optional context for context-specific authorization

        Returns:
            True if authorized, False otherwise
        """
        # Check deny list first
        if agent in self.deny_list and resource in self.deny_list[agent]:
            return False

        # Check allow list
        if agent in self.allow_list and resource in self.allow_list[agent]:
            return True

        # Default policy
        return self.default_allow

    def add_allow(self, agent: str, resource: str) -> None:
        """Add agent-resource pair to allow list."""
        if agent not in self.allow_list:
            self.allow_list[agent] = []
        if resource not in self.allow_list[agent]:
            self.allow_list[agent].append(resource)

    def add_deny(self, agent: str, resource: str) -> None:
        """Add agent-resource pair to deny list."""
        if agent not in self.deny_list:
            self.deny_list[agent] = []
        if resource not in self.deny_list[agent]:
            self.deny_list[agent].append(resource)


# =============================================================================
# EXTENDED SELECTION RESULT
# =============================================================================

@dataclass
class AuthorizedExpertSelectionResult(ExpertSelectionResult):
    """
    Extended selection result with Web4 integration info.

    Adds:
    - ATP cost/payment tracking
    - Authorization failures
    - Trust sync status
    """
    atp_cost: Optional[int] = None           # Total ATP cost for selection
    atp_payment: Optional[int] = None        # ATP offered by agent
    atp_sufficient: bool = True              # Did agent pay enough?
    unauthorized_experts: List[int] = field(default_factory=list)  # Experts denied
    trust_synced: bool = False               # Did trust sync succeed?
    sync_errors: List[str] = field(default_factory=list)  # Sync error messages


# =============================================================================
# AUTHORIZED EXPERT SELECTOR
# =============================================================================

class AuthorizedExpertSelector(TrustBasedExpertSelector):
    """
    Trust-based expert selector with Web4 authorization, ATP, and sync.

    Extends TrustBasedExpertSelector with:
    - Authorization checks (Web4 ACT integration)
    - ATP-based cache allocation (economic model)
    - Trust tensor synchronization (distributed learning)

    Complete Web4 ↔ SAGE integration in one component.

    Usage:
        selector = AuthorizedExpertSelector(
            num_experts=128,
            cache_size=6,
            identity_bridge=identity_bridge,
            atp_allocator=atp_allocator,
            trust_sync=trust_sync,
            auth_client=auth_client  # Optional
        )

        result = selector.select_experts(
            router_logits,
            context="code_generation",
            k=8,
            agent_lct="lct://web4/agent/alice",
            atp_payment=1000
        )
    """

    def __init__(
        self,
        *args,
        identity_bridge: ExpertIdentityBridge,
        atp_allocator: ATPResourceAllocator,
        trust_sync: TrustTensorSync,
        auth_client: Optional[Web4AuthClient] = None,
        enable_authorization: bool = False,
        enable_atp: bool = True,
        enable_trust_sync: bool = True,
        **kwargs
    ):
        """
        Initialize authorized expert selector.

        Args:
            *args: Passed to TrustBasedExpertSelector
            identity_bridge: Expert ID ↔ LCT ID mapping
            atp_allocator: ATP-based cache allocation
            trust_sync: SAGE ↔ Web4 trust synchronization
            auth_client: Web4 ACT authorization client (None = no auth)
            enable_authorization: Enable authorization checks
            enable_atp: Enable ATP cost computation
            enable_trust_sync: Enable trust synchronization
            **kwargs: Passed to TrustBasedExpertSelector
        """
        super().__init__(*args, **kwargs)

        # Web4 integration components
        self.identity_bridge = identity_bridge
        self.atp_allocator = atp_allocator
        self.trust_sync = trust_sync
        self.auth_client = auth_client

        # Feature flags
        self.enable_authorization = enable_authorization and auth_client is not None
        self.enable_atp = enable_atp
        self.enable_trust_sync = enable_trust_sync

        # Statistics
        self.authorization_denials = 0
        self.atp_insufficient_payments = 0
        self.trust_sync_successes = 0
        self.trust_sync_failures = 0

    def select_experts(
        self,
        router_logits: Union['torch.Tensor', np.ndarray],
        context: Optional[str] = None,
        k: int = 8,
        agent_lct: Optional[str] = None,
        atp_payment: Optional[int] = None,
        input_embedding: Optional[np.ndarray] = None
    ) -> AuthorizedExpertSelectionResult:
        """
        Select experts with authorization, ATP, and trust sync.

        Args:
            router_logits: Router output scores [num_experts]
            context: Context identifier (or None to auto-classify)
            k: Number of experts to select
            agent_lct: Requesting agent's LCT ID (for authorization)
            atp_payment: ATP offered for selection
            input_embedding: Input embedding (for context classification)

        Returns:
            AuthorizedExpertSelectionResult with Web4 integration info
        """
        # Get base selection from TrustBasedExpertSelector
        base_result = super().select_experts(
            router_logits=router_logits,
            context=context,
            k=k,
            input_embedding=input_embedding
        )

        # Create extended result
        result = AuthorizedExpertSelectionResult(
            selected_expert_ids=base_result.selected_expert_ids,
            selection_scores=base_result.selection_scores,
            router_scores=base_result.router_scores,
            trust_scores=base_result.trust_scores,
            context=base_result.context,
            substitutions=base_result.substitutions,
            cache_hits=base_result.cache_hits,
            cache_misses=base_result.cache_misses
        )

        # Apply authorization filtering
        if self.enable_authorization and agent_lct:
            result = self._apply_authorization(result, agent_lct)

        # Compute ATP costs
        if self.enable_atp:
            result = self._compute_atp_costs(result, atp_payment)

        # Sync trust to Web4
        if self.enable_trust_sync:
            result = self._sync_trust(result)

        return result

    def _apply_authorization(
        self,
        result: AuthorizedExpertSelectionResult,
        agent_lct: str
    ) -> AuthorizedExpertSelectionResult:
        """
        Apply authorization filtering to selected experts.

        Args:
            result: Base selection result
            agent_lct: Requesting agent's LCT ID

        Returns:
            Updated result with authorization applied
        """
        authorized_experts = []
        unauthorized_experts = []

        for expert_id in result.selected_expert_ids:
            # Get or register expert LCT ID
            expert_lct = self.identity_bridge.get_lct(expert_id)
            if expert_lct is None:
                expert_lct = self.identity_bridge.register_expert(expert_id)

            # Check authorization
            authorized = self.auth_client.check_authorization(
                agent=agent_lct,
                resource=expert_lct,
                action="select",
                context=result.context
            )

            if authorized:
                authorized_experts.append(expert_id)
            else:
                unauthorized_experts.append(expert_id)
                self.authorization_denials += 1

        # Try to find substitutes for unauthorized experts
        k = len(result.selected_expert_ids)
        while len(authorized_experts) < k and unauthorized_experts:
            unauthorized_id = unauthorized_experts.pop(0)

            # Find substitute (from parent class)
            substitute = self._find_substitute(unauthorized_id, result.context)

            if substitute and substitute not in authorized_experts:
                # Check if substitute is authorized
                substitute_lct = self.identity_bridge.get_lct(substitute)
                if substitute_lct and self.auth_client.check_authorization(
                    agent=agent_lct,
                    resource=substitute_lct,
                    action="select",
                    context=result.context
                ):
                    authorized_experts.append(substitute)
                    result.substitutions[unauthorized_id] = substitute

        # Update result
        result.selected_expert_ids = authorized_experts[:k]
        result.unauthorized_experts = unauthorized_experts

        return result

    def _compute_atp_costs(
        self,
        result: AuthorizedExpertSelectionResult,
        atp_payment: Optional[int]
    ) -> AuthorizedExpertSelectionResult:
        """
        Compute ATP costs for selected experts.

        Args:
            result: Selection result
            atp_payment: ATP offered by agent (None = no payment)

        Returns:
            Updated result with ATP info
        """
        if atp_payment is None:
            return result

        # Compute cache utilization
        cache_utilization = len(self.loaded_experts) / max(self.cache_size, 1)

        # Compute cost for each expert
        costs = []
        for expert_id in result.selected_expert_ids:
            # Get expert reputation for quality premium
            rep = self.reputation_db.get_reputation(expert_id, component=self.component)
            reputation = rep.get_context_trust(result.context, default=0.5) if rep else 0.5

            # Compute ATP cost
            cost = self.atp_allocator.compute_cost(
                expert_id=expert_id,
                reputation=reputation,
                cache_utilization=cache_utilization
            )
            costs.append(cost)

        # Update result
        total_cost = sum(costs)
        result.atp_cost = total_cost
        result.atp_payment = atp_payment
        result.atp_sufficient = atp_payment >= total_cost

        if not result.atp_sufficient:
            self.atp_insufficient_payments += 1

        return result

    def _sync_trust(
        self,
        result: AuthorizedExpertSelectionResult
    ) -> AuthorizedExpertSelectionResult:
        """
        Sync trust for selected experts to Web4.

        Args:
            result: Selection result

        Returns:
            Updated result with sync status
        """
        sync_errors = []

        for expert_id in result.selected_expert_ids:
            try:
                self.trust_sync.export_to_web4(expert_id, result.context)
                self.trust_sync_successes += 1
            except Exception as e:
                # Don't fail selection on sync errors
                error_msg = f"Expert {expert_id}: {str(e)}"
                sync_errors.append(error_msg)
                self.trust_sync_failures += 1

        result.trust_synced = len(sync_errors) == 0
        result.sync_errors = sync_errors

        return result

    def record_quality(
        self,
        expert_ids: List[int],
        quality_score: float,
        context: str,
        atp_cost_paid: Optional[int] = None
    ) -> Optional[int]:
        """
        Record quality for expert usage and compute ATP reward.

        Args:
            expert_ids: Experts that generated output
            quality_score: Output quality (0-1)
            context: Context identifier
            atp_cost_paid: ATP paid for selection (None = no reward)

        Returns:
            ATP reward amount (or None if no ATP)
        """
        # Record quality in reputation DB (parent class functionality)
        for expert_id in expert_ids:
            rep = self.reputation_db.get_reputation(expert_id, component=self.component)
            if rep:
                rep.update_context_trust(context, quality_score)
                self.reputation_db.save(rep)

        # Compute ATP reward if cost was paid
        if atp_cost_paid is not None and self.enable_atp:
            # Use first expert for reward computation (could be averaged)
            reward = self.atp_allocator.compute_reward(
                expert_id=expert_ids[0],
                quality_score=quality_score,
                cost_paid=atp_cost_paid
            )
            return reward

        return None

    def get_statistics(self) -> Dict:
        """Get integration statistics."""
        base_stats = {
            'total_selections': self.total_selections,
            'total_substitutions': self.total_substitutions,
            'cache_hit_rate': self.cache_hit_rate
        }

        web4_stats = {
            'authorization_denials': self.authorization_denials,
            'atp_insufficient_payments': self.atp_insufficient_payments,
            'trust_sync_successes': self.trust_sync_successes,
            'trust_sync_failures': self.trust_sync_failures
        }

        return {**base_stats, **web4_stats}


# Convenience function

def create_authorized_selector(
    num_experts: int,
    cache_size: int,
    identity_bridge: ExpertIdentityBridge,
    atp_allocator: ATPResourceAllocator,
    trust_sync: TrustTensorSync,
    auth_client: Optional[Web4AuthClient] = None,
    **kwargs
) -> AuthorizedExpertSelector:
    """
    Create authorized expert selector with default settings.

    Args:
        num_experts: Total number of experts
        cache_size: Cache capacity
        identity_bridge: Expert ID ↔ LCT ID mapping
        atp_allocator: ATP resource allocator
        trust_sync: Trust tensor synchronization
        auth_client: Authorization client (None = no authorization)
        **kwargs: Additional arguments for TrustBasedExpertSelector

    Returns:
        AuthorizedExpertSelector instance
    """
    return AuthorizedExpertSelector(
        num_experts=num_experts,
        cache_size=cache_size,
        identity_bridge=identity_bridge,
        atp_allocator=atp_allocator,
        trust_sync=trust_sync,
        auth_client=auth_client,
        **kwargs
    )
