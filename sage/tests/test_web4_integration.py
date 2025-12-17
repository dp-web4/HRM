#!/usr/bin/env python3
"""
Tests for Web4 Integration (ExpertIdentityBridge + ATPResourceAllocator)

Validates the integration between expert identity management and ATP-based
resource allocation.

Test Coverage:
1. Identity registration + cost computation
2. LCT-based allocation tracking
3. Expert metadata + ATP pricing
4. Economic flow (costs → rewards → surplus)
5. Multi-expert scenarios
6. Persistence across components

Created: Session 60 (2025-12-16)
"""

import tempfile
from pathlib import Path

try:
    from sage.web4.expert_identity import ExpertIdentityBridge
    from sage.web4.atp_allocator import ATPResourceAllocator
    HAS_MODULE = True
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from web4.expert_identity import ExpertIdentityBridge
    from web4.atp_allocator import ATPResourceAllocator
    HAS_MODULE = True


def test_basic_integration():
    """Test basic integration: identity + cost."""
    # Initialize components
    identity_bridge = ExpertIdentityBridge(namespace="sage_test")
    atp_allocator = ATPResourceAllocator(base_cost_per_expert=100)

    # Register expert
    lct_id = identity_bridge.register_expert(42, description="Test expert")
    assert lct_id == "lct://sage_test/expert/42"

    # Compute cost for expert
    cost = atp_allocator.compute_cost(
        expert_id=42,
        reputation=0.8,
        cache_utilization=0.5
    )
    assert cost == 140  # 100 * 1.4 (quality premium)

    # Verify identity
    assert identity_bridge.get_expert_id(lct_id) == 42

    print("✓ Basic integration")


def test_lct_based_allocation():
    """Test allocation with LCT identity tracking."""
    identity_bridge = ExpertIdentityBridge(namespace="sage_test")
    atp_allocator = ATPResourceAllocator()

    # Register experts
    experts = [1, 2, 3, 4, 5]
    lct_ids = {}
    for expert_id in experts:
        lct_id = identity_bridge.register_expert(expert_id)
        lct_ids[expert_id] = lct_id

    # Create allocation requests
    requests = [(expert_id, 100 + expert_id * 50) for expert_id in experts]

    # Allocate cache
    allocated = atp_allocator.allocate_cache(requests, cache_size=3)

    # Verify allocated experts have LCT IDs
    for expert_id in allocated:
        lct_id = identity_bridge.get_lct(expert_id)
        assert lct_id is not None
        assert lct_id.startswith("lct://sage_test/expert/")
        assert identity_bridge.get_expert_id(lct_id) == expert_id

    print("✓ LCT-based allocation")


def test_metadata_with_pricing():
    """Test expert metadata integration with ATP pricing."""
    identity_bridge = ExpertIdentityBridge(namespace="sage_test")
    atp_allocator = ATPResourceAllocator()

    # Register expert with metadata
    expert_id = 42
    metadata = {
        "specialization": "code_generation",
        "quality_tier": "premium",
        "reputation": 0.9
    }
    identity_bridge.register_expert(
        expert_id,
        description="Premium code expert",
        metadata=metadata
    )

    # Get identity
    identity = identity_bridge.get_identity(expert_id)
    assert identity.metadata["quality_tier"] == "premium"

    # Use reputation from metadata for pricing
    reputation = identity.metadata["reputation"]
    cost = atp_allocator.compute_cost(
        expert_id=expert_id,
        reputation=reputation,
        cache_utilization=0.9
    )

    # High reputation + high cache → high cost
    assert cost > 200  # Significant premium

    print("✓ Metadata with pricing")


def test_economic_flow():
    """Test complete economic flow: cost → allocation → reward."""
    identity_bridge = ExpertIdentityBridge(namespace="sage_test")
    atp_allocator = ATPResourceAllocator(base_cost_per_expert=100)

    # Register experts
    experts = list(range(10))
    for expert_id in experts:
        identity_bridge.register_expert(expert_id)

    # Compute costs (varying reputation)
    reputations = {i: 0.5 + i * 0.05 for i in experts}
    costs = {}
    for expert_id in experts:
        cost = atp_allocator.compute_cost(
            expert_id=expert_id,
            reputation=reputations[expert_id],
            cache_utilization=0.8
        )
        costs[expert_id] = cost

    # Create requests (agents pay cost + premium)
    requests = [(expert_id, costs[expert_id] + 50) for expert_id in experts]

    # Allocate cache
    allocated = atp_allocator.allocate_cache(requests, cache_size=5)
    assert len(allocated) == 5

    # Simulate generation with quality scores
    qualities = {i: 0.6 + i * 0.04 for i in experts}
    total_spent = 0
    total_rewarded = 0

    for expert_id in allocated:
        cost_paid = costs[expert_id] + 50
        quality = qualities[expert_id]

        reward = atp_allocator.compute_reward(
            expert_id=expert_id,
            quality_score=quality,
            cost_paid=cost_paid
        )

        total_spent += cost_paid
        total_rewarded += reward

    # Check economic flow
    stats = atp_allocator.get_statistics()
    assert stats.total_atp_spent == total_spent
    assert stats.total_atp_rewarded == total_rewarded

    # High quality should generate surplus
    if stats.average_quality > 0.8:
        assert total_rewarded > total_spent

    print(f"  Total spent: {total_spent}")
    print(f"  Total rewarded: {total_rewarded}")
    print(f"  Net flow: {total_rewarded - total_spent:+d}")
    print("✓ Economic flow")


def test_multi_expert_scenario():
    """Test realistic multi-expert allocation scenario."""
    identity_bridge = ExpertIdentityBridge(namespace="sage_legion")
    atp_allocator = ATPResourceAllocator(base_cost_per_expert=100)

    # Register 8 experts with descriptions
    expert_descriptions = {
        0: "Code generation",
        1: "Reasoning",
        2: "Math",
        3: "Creative writing",
        4: "Code review",
        5: "Debug",
        6: "Optimization",
        7: "Documentation"
    }

    identity_bridge.register_batch(
        expert_ids=list(expert_descriptions.keys()),
        descriptions=expert_descriptions
    )

    # Simulate reputation distribution
    reputations = {
        0: 0.92, 1: 0.88, 2: 0.85, 3: 0.75,
        4: 0.82, 5: 0.70, 6: 0.78, 7: 0.65
    }

    # Compute costs at high cache utilization
    cache_utilization = 0.9
    costs = {}
    for expert_id in expert_descriptions.keys():
        cost = atp_allocator.compute_cost(
            expert_id=expert_id,
            reputation=reputations[expert_id],
            cache_utilization=cache_utilization
        )
        costs[expert_id] = cost

    # Simulate agent requests with varying urgency
    urgencies = [50, 20, 100, 10, 30, 0, 40, 0]
    requests = [
        (expert_id, costs[expert_id] + urgency)
        for expert_id, urgency in zip(expert_descriptions.keys(), urgencies)
    ]

    # Allocate cache (4 slots)
    allocated = atp_allocator.allocate_cache(
        requests,
        cache_size=4,
        trust_scores=reputations
    )

    # Should allocate to high payers with high trust
    assert len(allocated) <= 4

    # Simulate quality outcomes
    qualities = {
        0: 0.92, 1: 0.88, 2: 0.95, 3: 0.75,
        4: 0.82, 5: 0.70, 6: 0.78, 7: 0.65
    }

    rewards = []
    for expert_id in allocated:
        cost_paid = next(payment for eid, payment in requests if eid == expert_id)
        reward = atp_allocator.compute_reward(
            expert_id=expert_id,
            quality_score=qualities[expert_id],
            cost_paid=cost_paid
        )
        rewards.append(reward)

    # Check statistics
    stats = atp_allocator.get_statistics()
    assert stats.successful_allocations == len(allocated)
    assert stats.average_quality > 0.0

    print(f"  Allocated: {allocated}")
    print(f"  Average quality: {stats.average_quality:.2f}")
    print(f"  Cache hit rate: {stats.cache_hit_rate*100:.0f}%")
    print("✓ Multi-expert scenario")


def test_persistence_integration():
    """Test persistence across both components."""
    with tempfile.TemporaryDirectory() as tmpdir:
        identity_path = Path(tmpdir) / "identity.json"
        stats_path = Path(tmpdir) / "stats.json"

        # Create and populate components
        identity_bridge1 = ExpertIdentityBridge(
            namespace="sage_test",
            registry_path=identity_path
        )
        atp_allocator1 = ATPResourceAllocator(
            base_cost_per_expert=100,
            stats_path=stats_path
        )

        # Register experts
        for i in range(5):
            identity_bridge1.register_expert(i, description=f"Expert {i}")

        # Generate statistics
        for i in range(5):
            cost = atp_allocator1.compute_cost(i, reputation=0.8, cache_utilization=0.7)
            atp_allocator1.compute_reward(i, quality_score=0.85, cost_paid=cost)

        # Save both
        identity_bridge1.save()
        atp_allocator1.save_statistics()

        # Load both
        identity_bridge2 = ExpertIdentityBridge.load(identity_path)
        atp_allocator2 = ATPResourceAllocator.load_statistics(stats_path)

        # Verify identity persisted
        assert len(identity_bridge2.expert_to_lct) == 5
        for i in range(5):
            assert identity_bridge2.is_registered(i)

        # Verify statistics persisted
        stats1 = atp_allocator1.get_statistics()
        stats2 = atp_allocator2.get_statistics()
        assert stats2.total_atp_spent == stats1.total_atp_spent
        assert stats2.total_atp_rewarded == stats1.total_atp_rewarded

        print("✓ Persistence integration")


def test_namespace_isolation():
    """Test namespace isolation with multiple bridges and allocators."""
    # Create two separate SAGE instances
    bridge_legion = ExpertIdentityBridge(namespace="sage_legion")
    bridge_thor = ExpertIdentityBridge(namespace="sage_thor")

    # Separate allocators (different economic policies)
    allocator_legion = ATPResourceAllocator(base_cost_per_expert=100)
    allocator_thor = ATPResourceAllocator(base_cost_per_expert=200)

    # Register same expert IDs in both namespaces
    for i in range(5):
        bridge_legion.register_expert(i)
        bridge_thor.register_expert(i)

    # LCT IDs should be different
    for i in range(5):
        lct_legion = bridge_legion.get_lct(i)
        lct_thor = bridge_thor.get_lct(i)
        assert lct_legion != lct_thor
        assert "sage_legion" in lct_legion
        assert "sage_thor" in lct_thor

    # Costs should be different (different base costs)
    cost_legion = allocator_legion.compute_cost(0, 0.5, 0.5)
    cost_thor = allocator_thor.compute_cost(0, 0.5, 0.5)
    assert cost_legion != cost_thor

    print("✓ Namespace isolation")


def test_edge_cases():
    """Test edge cases in integration."""
    identity_bridge = ExpertIdentityBridge(namespace="sage_test")
    atp_allocator = ATPResourceAllocator()

    # Expert not registered yet
    lct_id = identity_bridge.get_lct(999)
    assert lct_id is None

    # Cost computation works regardless of registration
    cost = atp_allocator.compute_cost(999, reputation=0.5, cache_utilization=0.5)
    assert cost > 0

    # Register after cost computation
    lct_id = identity_bridge.register_expert(999)
    assert lct_id is not None

    # Empty allocation
    allocated = atp_allocator.allocate_cache([], cache_size=5)
    assert len(allocated) == 0

    print("✓ Edge cases")


if __name__ == "__main__":
    print("Testing Web4 Integration...\n")

    test_basic_integration()
    test_lct_based_allocation()
    test_metadata_with_pricing()
    test_economic_flow()
    test_multi_expert_scenario()
    test_persistence_integration()
    test_namespace_isolation()
    test_edge_cases()

    print("\n✅ All integration tests passed!")
    print("\nWeb4 Integration validated:")
    print("- ExpertIdentityBridge + ATPResourceAllocator")
    print("- LCT-based expert identification")
    print("- Economic pricing with metadata")
    print("- Complete cost → allocation → reward flow")
    print("- Persistence across components")
    print("- Namespace isolation")
    print("- Ready for TrustBasedExpertSelector integration")
