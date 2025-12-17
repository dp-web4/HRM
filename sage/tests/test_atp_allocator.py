#!/usr/bin/env python3
"""
Tests for ATPResourceAllocator

Validates ATP-based resource allocation, economic pricing, and reward dynamics.

Test Coverage:
1. Cost computation (base, scarcity, quality premiums)
2. Cache allocation (priority queue, ATP-based)
3. Reward computation (quality-based refunds/bonuses)
4. Statistics tracking
5. Persistence (save/load)
6. Economic scenarios (high contention, quality variance)
7. Allocation history
8. Edge cases

Created: Session 60 (2025-12-16)
"""

import tempfile
from pathlib import Path
import time

try:
    from sage.web4.atp_allocator import (
        ATPResourceAllocator,
        ATPCost,
        ATPReward,
        AllocationRequest,
        AllocationStats,
        create_atp_allocator,
        compute_expert_cost,
        allocate_expert_cache
    )
    HAS_MODULE = True
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from web4.atp_allocator import (
        ATPResourceAllocator,
        ATPCost,
        ATPReward,
        AllocationRequest,
        AllocationStats,
        create_atp_allocator,
        compute_expert_cost,
        allocate_expert_cache
    )
    HAS_MODULE = True


def test_initialization():
    """Test allocator initialization."""
    # Default initialization
    allocator = ATPResourceAllocator()
    assert allocator.base_cost == 100
    assert allocator.contention_threshold == 0.8
    assert allocator.max_contention_multiplier == 3.0
    assert allocator.quality_factor == 0.5
    assert len(allocator.cost_history) == 0

    # Custom configuration
    allocator = ATPResourceAllocator(
        base_cost_per_expert=200,
        cache_contention_threshold=0.7,
        quality_premium_factor=0.3
    )
    assert allocator.base_cost == 200
    assert allocator.contention_threshold == 0.7
    assert allocator.quality_factor == 0.3

    print("✓ Initialization")


def test_base_cost():
    """Test base cost computation (no premiums)."""
    allocator = ATPResourceAllocator(base_cost_per_expert=100)

    # Low cache utilization, low reputation
    cost = allocator.compute_cost(
        expert_id=42,
        reputation=0.0,
        cache_utilization=0.5
    )
    assert cost == 100  # Just base cost

    # Check cost breakdown
    breakdown = allocator.get_cost_breakdown(42)
    assert breakdown is not None
    assert breakdown.expert_id == 42
    assert breakdown.base_cost == 100
    assert breakdown.total_cost == 100
    assert breakdown.scarcity_premium == 0.0
    assert breakdown.quality_premium == 0.0

    print("✓ Base cost")


def test_scarcity_premium():
    """Test scarcity premium when cache is full."""
    allocator = ATPResourceAllocator(
        base_cost_per_expert=100,
        cache_contention_threshold=0.8,
        max_contention_multiplier=3.0
    )

    # Below threshold: no premium
    cost_low = allocator.compute_cost(42, reputation=0.0, cache_utilization=0.5)
    assert cost_low == 100

    # At threshold: no premium
    cost_threshold = allocator.compute_cost(43, reputation=0.0, cache_utilization=0.8)
    assert cost_threshold == 100

    # Above threshold: linear premium
    cost_mid = allocator.compute_cost(44, reputation=0.0, cache_utilization=0.9)
    # At 90%: halfway between threshold (0.8) and 100%
    # Scarcity factor = (0.9 - 0.8) / (1.0 - 0.8) = 0.5
    # Premium = 0.5 * (3.0 - 1.0) = 1.0
    # Cost = 100 * (1 + 1.0) = 200
    assert cost_mid == 200

    # At 100% cache: maximum premium
    cost_max = allocator.compute_cost(45, reputation=0.0, cache_utilization=1.0)
    # Scarcity factor = 1.0
    # Premium = 1.0 * (3.0 - 1.0) = 2.0
    # Cost = 100 * (1 + 2.0) = 300
    assert cost_max == 300

    print("✓ Scarcity premium")


def test_quality_premium():
    """Test quality premium for high-reputation experts."""
    allocator = ATPResourceAllocator(
        base_cost_per_expert=100,
        quality_premium_factor=0.5
    )

    # Low reputation: minimal premium
    cost_low = allocator.compute_cost(42, reputation=0.0, cache_utilization=0.5)
    assert cost_low == 100

    # Medium reputation: moderate premium
    cost_mid = allocator.compute_cost(43, reputation=0.5, cache_utilization=0.5)
    # Quality premium = 0.5 * 0.5 = 0.25
    # Cost = 100 * (1 + 0.25) = 125
    assert cost_mid == 125

    # High reputation: maximum premium
    cost_high = allocator.compute_cost(44, reputation=1.0, cache_utilization=0.5)
    # Quality premium = 1.0 * 0.5 = 0.5
    # Cost = 100 * (1 + 0.5) = 150
    assert cost_high == 150

    print("✓ Quality premium")


def test_combined_premiums():
    """Test combined scarcity + quality premiums."""
    allocator = ATPResourceAllocator(
        base_cost_per_expert=100,
        cache_contention_threshold=0.8,
        max_contention_multiplier=3.0,
        quality_premium_factor=0.5
    )

    # High cache + high reputation: both premiums
    cost = allocator.compute_cost(
        expert_id=42,
        reputation=1.0,
        cache_utilization=1.0
    )
    # Scarcity premium: 2.0 → cost = 100 * 3.0 = 300
    # Quality premium: 0.5 → cost = 300 * 1.5 = 450
    assert cost == 450

    breakdown = allocator.get_cost_breakdown(42)
    assert breakdown.scarcity_premium == 2.0
    assert breakdown.quality_premium == 0.5
    assert breakdown.total_cost == 450

    print("✓ Combined premiums")


def test_cache_allocation():
    """Test cache allocation to highest ATP payers."""
    allocator = ATPResourceAllocator()

    # Requests with varying ATP payments
    requests = [
        (1, 100),  # Expert 1: 100 ATP
        (2, 500),  # Expert 2: 500 ATP
        (3, 200),  # Expert 3: 200 ATP
        (4, 800),  # Expert 4: 800 ATP
        (5, 50),   # Expert 5: 50 ATP
    ]

    # Allocate cache for 3 experts
    allocated = allocator.allocate_cache(requests, cache_size=3)

    # Should allocate to highest payers: 4 (800), 2 (500), 3 (200)
    assert len(allocated) == 3
    assert 4 in allocated  # Highest
    assert 2 in allocated
    assert 3 in allocated
    assert 1 not in allocated  # Too low
    assert 5 not in allocated  # Too low

    # Check statistics
    stats = allocator.get_statistics()
    assert stats.total_requests == 5
    assert stats.successful_allocations == 3
    assert stats.total_atp_spent == 800 + 500 + 200  # Top 3 payments

    print("✓ Cache allocation")


def test_cache_allocation_with_trust():
    """Test cache allocation with trust scores."""
    allocator = ATPResourceAllocator()

    requests = [
        (1, 100),  # 100 ATP, 0.5 trust → priority = 50
        (2, 100),  # 100 ATP, 1.0 trust → priority = 100
        (3, 100),  # 100 ATP, 0.8 trust → priority = 80
    ]

    trust_scores = {1: 0.5, 2: 1.0, 3: 0.8}

    # Allocate cache for 2 experts
    allocated = allocator.allocate_cache(
        requests,
        cache_size=2,
        trust_scores=trust_scores
    )

    # Should prioritize by ATP × trust: 2 (100), 3 (80)
    assert len(allocated) == 2
    assert 2 in allocated  # Highest priority
    assert 3 in allocated
    assert 1 not in allocated  # Lowest priority

    print("✓ Cache allocation with trust")


def test_reward_high_quality():
    """Test reward for high-quality output."""
    allocator = ATPResourceAllocator(
        base_cost_per_expert=100,
        high_quality_threshold=0.8,
        high_quality_bonus=0.5
    )

    # High quality: full refund + 50% bonus
    reward = allocator.compute_reward(
        expert_id=42,
        quality_score=0.9,
        cost_paid=100
    )
    assert reward == 150  # 100 refund + 50 bonus

    # Check reward breakdown
    reward_history = allocator.get_reward_history(42)
    assert len(reward_history) == 1
    assert reward_history[0].refund == 100
    assert reward_history[0].bonus == 50
    assert reward_history[0].total_reward == 150

    print("✓ High quality reward")


def test_reward_acceptable_quality():
    """Test reward for acceptable quality output."""
    allocator = ATPResourceAllocator(
        base_cost_per_expert=100,
        high_quality_threshold=0.8,
        acceptable_quality_threshold=0.5
    )

    # Acceptable quality: partial refund (linear)
    # At 0.5: 50% refund
    reward_low = allocator.compute_reward(42, quality_score=0.5, cost_paid=100)
    assert reward_low == 50

    # At 0.65: 75% refund (midpoint between 0.5 and 0.8)
    reward_mid = allocator.compute_reward(43, quality_score=0.65, cost_paid=100)
    assert reward_mid == 75

    # At 0.79: ~97% refund (just below high threshold)
    reward_high = allocator.compute_reward(44, quality_score=0.79, cost_paid=100)
    assert 95 <= reward_high <= 100

    print("✓ Acceptable quality reward")


def test_reward_poor_quality():
    """Test no reward for poor quality output."""
    allocator = ATPResourceAllocator(
        acceptable_quality_threshold=0.5
    )

    # Poor quality: no refund
    reward = allocator.compute_reward(
        expert_id=42,
        quality_score=0.3,
        cost_paid=100
    )
    assert reward == 0

    # Check reward breakdown
    reward_history = allocator.get_reward_history(42)
    assert len(reward_history) == 1
    assert reward_history[0].refund == 0
    assert reward_history[0].bonus == 0

    print("✓ Poor quality reward")


def test_statistics_tracking():
    """Test statistics computation."""
    allocator = ATPResourceAllocator(base_cost_per_expert=100)

    # Simulate allocation requests
    requests = [(i, 100 + i*10) for i in range(10)]
    allocator.allocate_cache(requests, cache_size=5)

    # Simulate cost computations
    for i in range(10):
        allocator.compute_cost(i, reputation=0.5 + i*0.05, cache_utilization=0.8 + i*0.01)

    # Simulate rewards
    for i in range(10):
        quality = 0.5 + i*0.05
        allocator.compute_reward(i, quality_score=quality, cost_paid=100)

    # Get statistics
    stats = allocator.get_statistics()

    assert stats.total_requests == 10
    assert stats.successful_allocations == 5
    assert stats.average_cost > 0
    assert 0.0 <= stats.average_quality <= 1.0
    assert 0.0 <= stats.cache_hit_rate <= 1.0
    assert stats.cache_hit_rate == 0.5  # 5/10

    print(f"  Total ATP spent: {stats.total_atp_spent}")
    print(f"  Total ATP rewarded: {stats.total_atp_rewarded}")
    print(f"  Average cost: {stats.average_cost:.2f}")
    print(f"  Average quality: {stats.average_quality:.2f}")
    print(f"  Price volatility: {stats.price_volatility:.2f}")
    print("✓ Statistics tracking")


def test_allocation_history():
    """Test allocation history tracking."""
    allocator = ATPResourceAllocator()

    # Make allocations
    requests1 = [(1, 100), (2, 200)]
    allocator.allocate_cache(requests1, cache_size=2)

    requests2 = [(3, 300), (4, 400)]
    allocator.allocate_cache(requests2, cache_size=1)

    # Get all history
    all_history = allocator.get_allocation_history()
    assert len(all_history) == 4

    # Get history for specific expert
    expert1_history = allocator.get_allocation_history(expert_id=1)
    assert len(expert1_history) == 1
    assert expert1_history[0].expert_id == 1
    assert expert1_history[0].atp_payment == 100

    # Get limited history
    recent = allocator.get_allocation_history(limit=2)
    assert len(recent) == 2

    print("✓ Allocation history")


def test_persistence():
    """Test save/load functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_path = Path(tmpdir) / "atp_stats.json"

        # Create allocator and generate statistics
        allocator1 = ATPResourceAllocator(
            base_cost_per_expert=200,
            stats_path=stats_path
        )

        # Simulate activity
        for i in range(5):
            allocator1.compute_cost(i, reputation=0.8, cache_utilization=0.9)
            allocator1.compute_reward(i, quality_score=0.85, cost_paid=200)

        requests = [(i, 100 + i*50) for i in range(5)]
        allocator1.allocate_cache(requests, cache_size=3)

        # Save statistics
        allocator1.save_statistics()
        assert stats_path.exists()

        # Load into new allocator
        allocator2 = ATPResourceAllocator.load_statistics(stats_path)

        # Verify configuration preserved
        assert allocator2.base_cost == 200

        # Verify statistics preserved
        stats1 = allocator1.get_statistics()
        stats2 = allocator2.get_statistics()
        assert stats2.total_requests == stats1.total_requests
        assert stats2.successful_allocations == stats1.successful_allocations
        assert stats2.total_atp_spent == stats1.total_atp_spent
        assert stats2.total_atp_rewarded == stats1.total_atp_rewarded

        # Verify history preserved
        assert len(allocator2.cost_history) == len(allocator1.cost_history)
        assert len(allocator2.reward_history) == len(allocator1.reward_history)
        assert len(allocator2.allocation_history) == len(allocator1.allocation_history)

        print("✓ Persistence")


def test_economic_scenario_high_contention():
    """Test economic behavior under high cache contention."""
    allocator = ATPResourceAllocator(base_cost_per_expert=100)

    # Simulate increasing cache pressure
    costs = []
    for utilization in [0.0, 0.5, 0.8, 0.9, 0.95, 1.0]:
        cost = allocator.compute_cost(
            expert_id=42,
            reputation=0.5,
            cache_utilization=utilization
        )
        costs.append(cost)

    # Costs should increase with utilization
    assert costs[0] < costs[3]  # 0% < 90%
    assert costs[3] < costs[5]  # 90% < 100%

    # At 100% cache, cost should be significantly higher
    assert costs[5] > costs[0] * 2  # At least 2x base cost

    print(f"  Cost at 0% cache: {costs[0]}")
    print(f"  Cost at 50% cache: {costs[1]}")
    print(f"  Cost at 80% cache: {costs[2]}")
    print(f"  Cost at 90% cache: {costs[3]}")
    print(f"  Cost at 100% cache: {costs[5]}")
    print("✓ High contention scenario")


def test_economic_scenario_quality_variance():
    """Test economic behavior across quality variance."""
    allocator = ATPResourceAllocator(base_cost_per_expert=100)

    # Simulate expert with varying quality
    cost_paid = 100
    qualities = [0.3, 0.5, 0.65, 0.8, 0.95]
    rewards = []

    for quality in qualities:
        reward = allocator.compute_reward(
            expert_id=42,
            quality_score=quality,
            cost_paid=cost_paid
        )
        rewards.append(reward)

    # Rewards should increase with quality
    assert rewards[0] < rewards[2] < rewards[4]

    # Poor quality gets no reward
    assert rewards[0] == 0

    # High quality gets bonus
    assert rewards[4] > cost_paid

    print(f"  Reward at quality 0.3: {rewards[0]}")
    print(f"  Reward at quality 0.5: {rewards[1]}")
    print(f"  Reward at quality 0.65: {rewards[2]}")
    print(f"  Reward at quality 0.8: {rewards[3]}")
    print(f"  Reward at quality 0.95: {rewards[4]}")
    print("✓ Quality variance scenario")


def test_reset_statistics():
    """Test statistics reset."""
    allocator = ATPResourceAllocator()

    # Generate statistics
    allocator.compute_cost(42, 0.5, 0.8)
    allocator.compute_reward(42, 0.7, 100)
    allocator.allocate_cache([(1, 100), (2, 200)], cache_size=1)

    # Verify statistics exist
    assert len(allocator.cost_history) > 0
    assert len(allocator.reward_history) > 0
    assert allocator.total_requests > 0

    # Reset
    allocator.reset_statistics()

    # Verify cleared
    assert len(allocator.cost_history) == 0
    assert len(allocator.reward_history) == 0
    assert allocator.total_requests == 0
    assert allocator.total_atp_spent == 0

    print("✓ Reset statistics")


def test_convenience_functions():
    """Test convenience wrapper functions."""
    # Create allocator
    allocator = create_atp_allocator(base_cost=150)
    assert isinstance(allocator, ATPResourceAllocator)
    assert allocator.base_cost == 150

    # Compute cost
    cost = compute_expert_cost(allocator, 42, reputation=0.7, cache_utilization=0.6)
    assert cost > 0

    # Allocate cache
    requests = [(1, 100), (2, 200), (3, 150)]
    allocated = allocate_expert_cache(allocator, requests, cache_size=2)
    assert len(allocated) == 2
    assert 2 in allocated  # Highest payer

    print("✓ Convenience functions")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    allocator = ATPResourceAllocator()

    # Empty requests
    allocated = allocator.allocate_cache([], cache_size=5)
    assert len(allocated) == 0

    # Cache size larger than requests
    requests = [(1, 100), (2, 200)]
    allocated = allocator.allocate_cache(requests, cache_size=10)
    assert len(allocated) == 2

    # Zero cache size
    allocated = allocator.allocate_cache(requests, cache_size=0)
    assert len(allocated) == 0

    # Zero reputation
    cost = allocator.compute_cost(42, reputation=0.0, cache_utilization=0.5)
    assert cost == 100  # Just base cost

    # Zero quality
    reward = allocator.compute_reward(42, quality_score=0.0, cost_paid=100)
    assert reward == 0

    # Cache utilization > 1.0 (shouldn't happen, but handle gracefully)
    cost = allocator.compute_cost(42, reputation=0.5, cache_utilization=1.5)
    assert cost > 0

    print("✓ Edge cases")


if __name__ == "__main__":
    print("Testing ATPResourceAllocator...\n")

    test_initialization()
    test_base_cost()
    test_scarcity_premium()
    test_quality_premium()
    test_combined_premiums()
    test_cache_allocation()
    test_cache_allocation_with_trust()
    test_reward_high_quality()
    test_reward_acceptable_quality()
    test_reward_poor_quality()
    test_statistics_tracking()
    test_allocation_history()
    test_persistence()
    test_economic_scenario_high_contention()
    test_economic_scenario_quality_variance()
    test_reset_statistics()
    test_convenience_functions()
    test_edge_cases()

    print("\n✅ All tests passed!")
    print("\nATPResourceAllocator validated:")
    print("- Cost computation (base, scarcity, quality premiums)")
    print("- Cache allocation (ATP-based priority)")
    print("- Reward computation (quality-based)")
    print("- Statistics tracking and persistence")
    print("- Economic scenarios (contention, quality variance)")
    print("- Ready for SAGE integration")
