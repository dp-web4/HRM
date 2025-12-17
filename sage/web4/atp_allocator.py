#!/usr/bin/env python3
"""
ATP Resource Allocator - Web4 Economic Model for SAGE

Implements ATP (Agent Trust Protocol energy) based resource allocation for
SAGE expert selection and cache management. Provides economic pricing signals
for neural compute resources.

Core Economics:
- Cost = base_cost × (1 + scarcity_premium) × (1 + quality_premium)
- Scarcity premium: Higher when cache is full (supply/demand)
- Quality premium: High-reputation experts cost more (quality signal)
- Rewards: Quality-based refunds + bonuses (incentive alignment)

Integration Points:
- ExpertIdentityBridge: Maps expert_id → LCT identity
- TrustBasedExpertSelector: Cache management with ATP priority
- ExpertReputationDB: Quality scores for pricing
- Web4 ATP System: Economic foundation (when deployed)

Design Philosophy:
- Price reflects value: Scarcity + quality determine cost
- Incentive alignment: Good performance → ATP rewards
- Resource efficiency: High-value requests get priority
- Market discovery: Prices adapt to demand

Created: Session 60 (2025-12-16)
Part of: Web4 ↔ SAGE integration (Session 57 design)
Previous: Session 59 (ExpertIdentityBridge)
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
import json
import heapq


@dataclass
class ATPCost:
    """ATP cost breakdown for expert selection."""
    expert_id: int
    base_cost: int
    scarcity_premium: float
    quality_premium: float
    total_cost: int
    cache_utilization: float
    reputation: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ATPReward:
    """ATP reward for expert performance."""
    expert_id: int
    quality_score: float
    cost_paid: int
    refund: int
    bonus: int
    total_reward: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class AllocationRequest:
    """Request for cache allocation with ATP payment."""
    expert_id: int
    atp_payment: int
    priority_score: float  # ATP × trust × recency
    request_time: float = field(default_factory=time.time)
    agent_id: Optional[str] = None  # Requesting agent's LCT ID


@dataclass
class AllocationStats:
    """Statistics about ATP allocation."""
    total_requests: int
    successful_allocations: int
    total_atp_spent: int
    total_atp_rewarded: int
    average_cost: float
    average_quality: float
    cache_hit_rate: float
    price_volatility: float  # Std dev of costs over time


class ATPResourceAllocator:
    """
    Allocates cache resources based on ATP payments.

    Implements economic pricing for expert selection:
    1. Compute cost based on scarcity, quality, and base overhead
    2. Allocate cache to highest ATP payers
    3. Reward good performance with refunds + bonuses
    4. Track statistics for economic analysis

    Usage:
        allocator = ATPResourceAllocator(base_cost_per_expert=100)

        # Compute cost
        cost = allocator.compute_cost(
            expert_id=42,
            reputation=0.85,
            cache_utilization=0.9
        )

        # Allocate cache
        allocated = allocator.allocate_cache(
            requests=[(42, 500), (7, 300), (15, 800)],
            cache_size=2
        )
        # Returns: [15, 42] (highest payers)

        # Reward performance
        reward = allocator.compute_reward(
            expert_id=42,
            quality_score=0.9,
            cost_paid=500
        )
        # Returns: 750 (150% of cost for high quality)
    """

    def __init__(
        self,
        base_cost_per_expert: int = 100,
        cache_contention_threshold: float = 0.8,
        max_contention_multiplier: float = 3.0,
        quality_premium_factor: float = 0.5,
        high_quality_threshold: float = 0.8,
        acceptable_quality_threshold: float = 0.5,
        high_quality_bonus: float = 0.5,
        stats_path: Optional[Path] = None
    ):
        """
        Initialize ATP resource allocator.

        Args:
            base_cost_per_expert: Base ATP cost to load an expert
            cache_contention_threshold: Utilization above which scarcity premium applies
            max_contention_multiplier: Maximum scarcity multiplier (at 100% cache)
            quality_premium_factor: How much reputation affects price (0-1)
            high_quality_threshold: Quality above this gets full refund + bonus
            acceptable_quality_threshold: Quality above this gets partial refund
            high_quality_bonus: Bonus multiplier for high quality (0.5 = 50% bonus)
            stats_path: Path to save allocation statistics
        """
        self.base_cost = base_cost_per_expert
        self.contention_threshold = cache_contention_threshold
        self.max_contention_multiplier = max_contention_multiplier
        self.quality_factor = quality_premium_factor
        self.high_quality_threshold = high_quality_threshold
        self.acceptable_quality_threshold = acceptable_quality_threshold
        self.high_quality_bonus = high_quality_bonus
        self.stats_path = Path(stats_path) if stats_path else None

        # Track allocation history
        self.cost_history: List[ATPCost] = []
        self.reward_history: List[ATPReward] = []
        self.allocation_history: List[AllocationRequest] = []

        # Statistics
        self.total_requests = 0
        self.successful_allocations = 0
        self.total_atp_spent = 0
        self.total_atp_rewarded = 0

    def compute_cost(
        self,
        expert_id: int,
        reputation: float,
        cache_utilization: float
    ) -> int:
        """
        Compute ATP cost to select expert.

        Cost formula:
            base_cost × (1 + scarcity_premium) × (1 + quality_premium)

        Scarcity premium:
            - 0% if cache_utilization < threshold (0.8)
            - Linear increase to max_multiplier at 100% cache
            - Reflects supply/demand dynamics

        Quality premium:
            - 0% for reputation=0
            - quality_factor×100% for reputation=1.0
            - High-quality experts cost more (premium service)

        Args:
            expert_id: Expert to price
            reputation: Expert reputation score (0-1)
            cache_utilization: Current cache fullness (0-1)

        Returns:
            ATP cost (integer)
        """
        # Base cost to load expert (VRAM allocation, disk I/O)
        cost = float(self.base_cost)

        # Scarcity premium: cache contention increases price
        scarcity_premium = 0.0
        if cache_utilization > self.contention_threshold:
            # Linear interpolation from threshold to 100%
            over_threshold = (cache_utilization - self.contention_threshold)
            remaining_capacity = (1.0 - self.contention_threshold)
            scarcity_factor = over_threshold / max(remaining_capacity, 0.01)
            scarcity_premium = scarcity_factor * (self.max_contention_multiplier - 1.0)

        cost *= (1.0 + scarcity_premium)

        # Quality premium: high-reputation experts cost more
        quality_premium = reputation * self.quality_factor
        cost *= (1.0 + quality_premium)

        total_cost = int(cost)

        # Record cost breakdown
        cost_record = ATPCost(
            expert_id=expert_id,
            base_cost=self.base_cost,
            scarcity_premium=scarcity_premium,
            quality_premium=quality_premium,
            total_cost=total_cost,
            cache_utilization=cache_utilization,
            reputation=reputation
        )
        self.cost_history.append(cost_record)

        return total_cost

    def allocate_cache(
        self,
        requests: List[Tuple[int, int]],  # [(expert_id, atp_payment)]
        cache_size: int,
        trust_scores: Optional[Dict[int, float]] = None,
        recency_scores: Optional[Dict[int, float]] = None
    ) -> List[int]:
        """
        Allocate cache slots to highest-priority requests.

        Priority = ATP payment × trust_score × recency_score

        This combines:
        - Economic signal (ATP payment)
        - Quality signal (trust score)
        - Temporal signal (recency)

        Args:
            requests: List of (expert_id, atp_payment) tuples
            cache_size: Number of experts that fit in cache
            trust_scores: Optional trust scores per expert (default 1.0)
            recency_scores: Optional recency scores per expert (default 1.0)

        Returns:
            List of expert IDs allocated cache slots (up to cache_size)
        """
        trust_scores = trust_scores or {}
        recency_scores = recency_scores or {}

        # Compute priority scores
        allocation_requests = []
        for expert_id, atp_payment in requests:
            trust = trust_scores.get(expert_id, 1.0)
            recency = recency_scores.get(expert_id, 1.0)
            priority = atp_payment * trust * recency

            alloc_req = AllocationRequest(
                expert_id=expert_id,
                atp_payment=atp_payment,
                priority_score=priority
            )
            allocation_requests.append(alloc_req)

        # Sort by priority (descending)
        allocation_requests.sort(key=lambda x: x.priority_score, reverse=True)

        # Allocate top N requests
        allocated = [req.expert_id for req in allocation_requests[:cache_size]]

        # Record allocation history
        self.allocation_history.extend(allocation_requests)
        self.total_requests += len(requests)
        self.successful_allocations += len(allocated)
        self.total_atp_spent += sum(req.atp_payment for req in allocation_requests[:cache_size])

        return allocated

    def compute_reward(
        self,
        expert_id: int,
        quality_score: float,
        cost_paid: int
    ) -> int:
        """
        Compute ATP reward for expert performance.

        Reward structure:
        - High quality (>0.8): 100% refund + 50% bonus
        - Acceptable quality (>0.5): Partial refund (proportional to quality)
        - Poor quality (<0.5): No refund (cost sunk)

        This incentivizes:
        - Quality over quantity (high quality gets rewarded)
        - Risk/reward balance (acceptable quality breaks even)
        - Cost of failure (poor quality loses ATP)

        Args:
            expert_id: Expert that generated output
            quality_score: Output quality (0-1)
            cost_paid: ATP paid upfront

        Returns:
            ATP reward (refund + bonus)
        """
        # High quality: full refund + bonus
        if quality_score >= self.high_quality_threshold:
            refund = cost_paid
            bonus = int(cost_paid * self.high_quality_bonus)
            total_reward = refund + bonus

        # Acceptable quality: partial refund (linear)
        elif quality_score >= self.acceptable_quality_threshold:
            # Linear interpolation from acceptable_threshold to high_threshold
            # At acceptable_threshold: 50% refund
            # At high_threshold: 100% refund
            quality_range = self.high_quality_threshold - self.acceptable_quality_threshold
            quality_excess = quality_score - self.acceptable_quality_threshold
            refund_factor = 0.5 + 0.5 * (quality_excess / max(quality_range, 0.01))
            refund = int(cost_paid * refund_factor)
            bonus = 0
            total_reward = refund

        # Poor quality: no refund
        else:
            refund = 0
            bonus = 0
            total_reward = 0

        # Record reward
        reward_record = ATPReward(
            expert_id=expert_id,
            quality_score=quality_score,
            cost_paid=cost_paid,
            refund=refund,
            bonus=bonus,
            total_reward=total_reward
        )
        self.reward_history.append(reward_record)
        self.total_atp_rewarded += total_reward

        return total_reward

    def get_statistics(self) -> AllocationStats:
        """
        Get allocation statistics.

        Returns:
            AllocationStats with economic metrics
        """
        # Average cost
        avg_cost = (
            sum(c.total_cost for c in self.cost_history) / len(self.cost_history)
            if self.cost_history else 0.0
        )

        # Average quality
        avg_quality = (
            sum(r.quality_score for r in self.reward_history) / len(self.reward_history)
            if self.reward_history else 0.0
        )

        # Cache hit rate (successful allocations / total requests)
        cache_hit_rate = (
            self.successful_allocations / max(self.total_requests, 1)
        )

        # Price volatility (std dev of costs)
        if len(self.cost_history) > 1:
            costs = [c.total_cost for c in self.cost_history]
            mean_cost = sum(costs) / len(costs)
            variance = sum((c - mean_cost) ** 2 for c in costs) / len(costs)
            price_volatility = variance ** 0.5
        else:
            price_volatility = 0.0

        return AllocationStats(
            total_requests=self.total_requests,
            successful_allocations=self.successful_allocations,
            total_atp_spent=self.total_atp_spent,
            total_atp_rewarded=self.total_atp_rewarded,
            average_cost=avg_cost,
            average_quality=avg_quality,
            cache_hit_rate=cache_hit_rate,
            price_volatility=price_volatility
        )

    def get_cost_breakdown(self, expert_id: int) -> Optional[ATPCost]:
        """
        Get most recent cost breakdown for expert.

        Args:
            expert_id: Expert to query

        Returns:
            Most recent ATPCost or None
        """
        for cost in reversed(self.cost_history):
            if cost.expert_id == expert_id:
                return cost
        return None

    def get_reward_history(self, expert_id: int) -> List[ATPReward]:
        """
        Get reward history for expert.

        Args:
            expert_id: Expert to query

        Returns:
            List of ATPReward records
        """
        return [r for r in self.reward_history if r.expert_id == expert_id]

    def get_allocation_history(
        self,
        expert_id: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[AllocationRequest]:
        """
        Get allocation request history.

        Args:
            expert_id: Filter by expert (None = all)
            limit: Max number of records (None = all)

        Returns:
            List of AllocationRequest records
        """
        history = self.allocation_history

        if expert_id is not None:
            history = [a for a in history if a.expert_id == expert_id]

        if limit is not None:
            history = history[-limit:]

        return history

    def reset_statistics(self) -> None:
        """Reset all statistics and history."""
        self.cost_history.clear()
        self.reward_history.clear()
        self.allocation_history.clear()
        self.total_requests = 0
        self.successful_allocations = 0
        self.total_atp_spent = 0
        self.total_atp_rewarded = 0

    def save_statistics(self, path: Optional[Path] = None) -> None:
        """
        Save allocation statistics to disk.

        Args:
            path: Path to save (defaults to self.stats_path)
        """
        if path is None:
            if self.stats_path is None:
                raise ValueError("No stats path specified")
            path = self.stats_path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize statistics
        data = {
            'statistics': asdict(self.get_statistics()),
            'cost_history': [asdict(c) for c in self.cost_history],
            'reward_history': [asdict(r) for r in self.reward_history],
            'allocation_history': [asdict(a) for a in self.allocation_history],
            'config': {
                'base_cost': self.base_cost,
                'contention_threshold': self.contention_threshold,
                'max_contention_multiplier': self.max_contention_multiplier,
                'quality_factor': self.quality_factor,
                'high_quality_threshold': self.high_quality_threshold,
                'acceptable_quality_threshold': self.acceptable_quality_threshold,
                'high_quality_bonus': self.high_quality_bonus
            }
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_statistics(cls, path: Path) -> 'ATPResourceAllocator':
        """
        Load allocator with statistics from disk.

        Args:
            path: Path to load

        Returns:
            ATPResourceAllocator with loaded statistics
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Statistics not found at {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        # Create allocator with saved config
        config = data['config']
        allocator = cls(
            base_cost_per_expert=config['base_cost'],
            cache_contention_threshold=config['contention_threshold'],
            max_contention_multiplier=config['max_contention_multiplier'],
            quality_premium_factor=config['quality_factor'],
            high_quality_threshold=config['high_quality_threshold'],
            acceptable_quality_threshold=config['acceptable_quality_threshold'],
            high_quality_bonus=config['high_quality_bonus'],
            stats_path=path
        )

        # Restore statistics
        allocator.cost_history = [ATPCost(**c) for c in data['cost_history']]
        allocator.reward_history = [ATPReward(**r) for r in data['reward_history']]
        allocator.allocation_history = [AllocationRequest(**a) for a in data['allocation_history']]

        stats = data['statistics']
        allocator.total_requests = stats['total_requests']
        allocator.successful_allocations = stats['successful_allocations']
        allocator.total_atp_spent = stats['total_atp_spent']
        allocator.total_atp_rewarded = stats['total_atp_rewarded']

        return allocator


# Convenience functions

def create_atp_allocator(
    base_cost: int = 100,
    stats_path: Optional[Path] = None
) -> ATPResourceAllocator:
    """
    Create ATP allocator with default settings.

    Args:
        base_cost: Base ATP cost per expert
        stats_path: Path for statistics persistence

    Returns:
        ATPResourceAllocator instance
    """
    return ATPResourceAllocator(
        base_cost_per_expert=base_cost,
        stats_path=stats_path
    )


def compute_expert_cost(
    allocator: ATPResourceAllocator,
    expert_id: int,
    reputation: float,
    cache_utilization: float
) -> int:
    """
    Compute ATP cost for expert selection.

    Convenience wrapper for allocator.compute_cost().

    Args:
        allocator: ATP allocator
        expert_id: Expert to price
        reputation: Expert reputation (0-1)
        cache_utilization: Cache fullness (0-1)

    Returns:
        ATP cost
    """
    return allocator.compute_cost(expert_id, reputation, cache_utilization)


def allocate_expert_cache(
    allocator: ATPResourceAllocator,
    requests: List[Tuple[int, int]],
    cache_size: int
) -> List[int]:
    """
    Allocate cache to highest ATP payers.

    Args:
        allocator: ATP allocator
        requests: List of (expert_id, atp_payment)
        cache_size: Cache capacity

    Returns:
        List of allocated expert IDs
    """
    return allocator.allocate_cache(requests, cache_size)
