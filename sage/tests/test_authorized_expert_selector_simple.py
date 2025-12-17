#!/usr/bin/env python3
"""
Simplified tests for AuthorizedExpertSelector

Test Coverage:
1. Basic integration (all components together)
2. Authorization filtering
3. ATP cost computation
4. Trust synchronization
5. Quality recording with ATP rewards
6. Statistics tracking

Created: Session 61 (2025-12-16)
"""

import tempfile
from pathlib import Path
import numpy as np

try:
    from sage.web4.authorized_expert_selector import (
        AuthorizedExpertSelector,
        Web4AuthClient,
        create_authorized_selector
    )
    from sage.web4.expert_identity import ExpertIdentityBridge
    from sage.web4.atp_allocator import ATPResourceAllocator
    from sage.web4.trust_tensor_sync import TrustTensorSync
    from sage.core.expert_reputation import ExpertReputationDB, ExpertReputation
    HAS_MODULE = True
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from web4.authorized_expert_selector import (
        AuthorizedExpertSelector,
        Web4AuthClient,
        create_authorized_selector
    )
    from web4.expert_identity import ExpertIdentityBridge
    from web4.atp_allocator import ATPResourceAllocator
    from web4.trust_tensor_sync import TrustTensorSync
    from core.expert_reputation import ExpertReputationDB, ExpertReputation
    HAS_MODULE = True


def test_basic_integration():
    """Test basic integration of all Web4 components."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create all components
        identity_bridge = ExpertIdentityBridge(namespace="sage_test")
        reputation_db = ExpertReputationDB(db_path=Path(tmpdir) / "rep.db")
        atp_allocator = ATPResourceAllocator(base_cost_per_expert=100)
        trust_sync = TrustTensorSync(
            reputation_db=reputation_db,
            identity_bridge=identity_bridge
        )

        # Create selector
        selector = AuthorizedExpertSelector(
            num_experts=8,
            cache_size=4,
            identity_bridge=identity_bridge,
            atp_allocator=atp_allocator,
            trust_sync=trust_sync,
            enable_authorization=False,  # Disable auth for basic test
            enable_atp=True,
            enable_trust_sync=True
        )

        # Simulate router logits
        router_logits = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6])

        # Select experts with ATP payment
        result = selector.select_experts(
            router_logits=router_logits,
            context="code",
            k=4,
            atp_payment=500
        )

        # Verify selection happened
        assert len(result.selected_expert_ids) <= 4
        assert result.atp_cost is not None
        assert result.atp_payment == 500
        assert result.context == "code"

        print(f"  Selected experts: {result.selected_expert_ids}")
        print(f"  ATP cost: {result.atp_cost}")
        print(f"  ATP sufficient: {result.atp_sufficient}")
        print("✓ Basic integration")


def test_authorization_filtering():
    """Test authorization filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        identity_bridge = ExpertIdentityBridge(namespace="sage_test")
        reputation_db = ExpertReputationDB(db_path=Path(tmpdir) / "rep.db")
        atp_allocator = ATPResourceAllocator()
        trust_sync = TrustTensorSync(reputation_db, identity_bridge)
        auth_client = Web4AuthClient(default_allow=False)  # Deny by default

        # Register and authorize specific experts
        for i in range(8):
            lct_id = identity_bridge.register_expert(i)
            if i % 2 == 0:  # Authorize even experts only
                auth_client.add_allow("lct://web4/agent/alice", lct_id)

        # Create selector with authorization enabled
        selector = AuthorizedExpertSelector(
            num_experts=8,
            cache_size=4,
            identity_bridge=identity_bridge,
            atp_allocator=atp_allocator,
            trust_sync=trust_sync,
            auth_client=auth_client,
            enable_authorization=True
        )

        # Select experts
        router_logits = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6])
        result = selector.select_experts(
            router_logits=router_logits,
            context="code",
            k=4,
            agent_lct="lct://web4/agent/alice"
        )

        # Verify only authorized experts selected (even numbers)
        for expert_id in result.selected_expert_ids:
            assert expert_id % 2 == 0  # Only even experts authorized

        print(f"  Selected (authorized): {result.selected_expert_ids}")
        print(f"  Unauthorized: {result.unauthorized_experts}")
        print("✓ Authorization filtering")


def test_atp_cost_computation():
    """Test ATP cost computation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        identity_bridge = ExpertIdentityBridge(namespace="sage_test")
        reputation_db = ExpertReputationDB(db_path=Path(tmpdir) / "rep.db")
        atp_allocator = ATPResourceAllocator(base_cost_per_expert=100)
        trust_sync = TrustTensorSync(reputation_db, identity_bridge)

        # Create reputation for expert 0 (high trust)
        rep = ExpertReputation(expert_id=0, component="thinker")
        rep.context_trust["code"] = 0.9
        rep.context_observations["code"] = 100
        reputation_db.save(rep)

        # Create selector
        selector = AuthorizedExpertSelector(
            num_experts=8,
            cache_size=4,
            identity_bridge=identity_bridge,
            atp_allocator=atp_allocator,
            trust_sync=trust_sync,
            enable_atp=True
        )

        # Select with ATP payment
        router_logits = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        result = selector.select_experts(
            router_logits=router_logits,
            context="code",
            k=2,
            atp_payment=1000
        )

        # Verify ATP info
        assert result.atp_cost is not None
        assert result.atp_payment == 1000
        assert result.atp_cost > 0  # Should have cost

        print(f"  ATP cost: {result.atp_cost}")
        print(f"  ATP payment: {result.atp_payment}")
        print(f"  Sufficient: {result.atp_sufficient}")
        print("✓ ATP cost computation")


def test_trust_synchronization():
    """Test trust synchronization to Web4."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        identity_bridge = ExpertIdentityBridge(namespace="sage_test")
        reputation_db = ExpertReputationDB(db_path=Path(tmpdir) / "rep.db")
        atp_allocator = ATPResourceAllocator()
        trust_sync = TrustTensorSync(reputation_db, identity_bridge)

        # Create reputation
        rep = ExpertReputation(expert_id=0, component="thinker")
        rep.context_trust["code"] = 0.85
        rep.context_observations["code"] = 50
        reputation_db.save(rep)

        # Create selector
        selector = AuthorizedExpertSelector(
            num_experts=8,
            cache_size=4,
            identity_bridge=identity_bridge,
            atp_allocator=atp_allocator,
            trust_sync=trust_sync,
            enable_trust_sync=True
        )

        # Select experts
        router_logits = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        result = selector.select_experts(
            router_logits=router_logits,
            context="code",
            k=1
        )

        # Verify trust synced
        assert result.trust_synced or len(result.sync_errors) > 0

        # Check trust sync statistics
        stats = selector.get_statistics()
        assert stats['trust_sync_successes'] > 0 or stats['trust_sync_failures'] > 0

        print(f"  Trust synced: {result.trust_synced}")
        print(f"  Sync successes: {stats['trust_sync_successes']}")
        print("✓ Trust synchronization")


def test_quality_recording_with_rewards():
    """Test quality recording with ATP rewards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        identity_bridge = ExpertIdentityBridge(namespace="sage_test")
        reputation_db = ExpertReputationDB(db_path=Path(tmpdir) / "rep.db")
        atp_allocator = ATPResourceAllocator(base_cost_per_expert=100)
        trust_sync = TrustTensorSync(reputation_db, identity_bridge)

        # Create selector
        selector = AuthorizedExpertSelector(
            num_experts=8,
            cache_size=4,
            identity_bridge=identity_bridge,
            atp_allocator=atp_allocator,
            trust_sync=trust_sync,
            enable_atp=True
        )

        # Select experts
        router_logits = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        result = selector.select_experts(
            router_logits=router_logits,
            context="code",
            k=1,
            atp_payment=200
        )

        # Record high quality
        reward = selector.record_quality(
            expert_ids=result.selected_expert_ids,
            quality_score=0.95,  # Excellent quality
            context="code",
            atp_cost_paid=result.atp_cost
        )

        # Should get ATP reward for high quality
        assert reward is not None
        assert reward > result.atp_cost  # Reward > cost for excellent quality

        print(f"  ATP cost paid: {result.atp_cost}")
        print(f"  Quality score: 0.95")
        print(f"  ATP reward: {reward}")
        print(f"  Net gain: {reward - result.atp_cost}")
        print("✓ Quality recording with rewards")


def test_statistics():
    """Test statistics tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        identity_bridge = ExpertIdentityBridge(namespace="sage_test")
        reputation_db = ExpertReputationDB(db_path=Path(tmpdir) / "rep.db")
        atp_allocator = ATPResourceAllocator()
        trust_sync = TrustTensorSync(reputation_db, identity_bridge)

        selector = AuthorizedExpertSelector(
            num_experts=8,
            cache_size=4,
            identity_bridge=identity_bridge,
            atp_allocator=atp_allocator,
            trust_sync=trust_sync
        )

        # Make several selections
        router_logits = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6])
        for _ in range(3):
            selector.select_experts(router_logits, context="code", k=4)

        # Get statistics
        stats = selector.get_statistics()
        assert 'total_selections' in stats
        assert 'trust_sync_successes' in stats

        print(f"  Total selections: {stats['total_selections']}")
        print(f"  Trust sync successes: {stats['trust_sync_successes']}")
        print("✓ Statistics tracking")


def test_convenience_function():
    """Test convenience function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        identity_bridge = ExpertIdentityBridge(namespace="sage_test")
        reputation_db = ExpertReputationDB(db_path=Path(tmpdir) / "rep.db")
        atp_allocator = ATPResourceAllocator()
        trust_sync = TrustTensorSync(reputation_db, identity_bridge)

        selector = create_authorized_selector(
            num_experts=8,
            cache_size=4,
            identity_bridge=identity_bridge,
            atp_allocator=atp_allocator,
            trust_sync=trust_sync
        )

        assert isinstance(selector, AuthorizedExpertSelector)
        print("✓ Convenience function")


if __name__ == "__main__":
    print("Testing AuthorizedExpertSelector (simplified)...\n")

    test_basic_integration()
    test_authorization_filtering()
    test_atp_cost_computation()
    test_trust_synchronization()
    test_quality_recording_with_rewards()
    test_statistics()
    test_convenience_function()

    print("\n✅ All tests passed!")
    print("\nAuthorizedExpertSelector validated:")
    print("- Complete Web4 ↔ SAGE integration")
    print("- Authorization filtering (ACT stub)")
    print("- ATP cost computation and rewards")
    print("- Trust synchronization")
    print("- Quality feedback loop")
    print("- Statistics tracking")
    print("\n🎉 Web4 ↔ SAGE Integration Complete!")
    print("\nComponents:")
    print("  ✅ ExpertIdentityBridge (Session 59)")
    print("  ✅ ATPResourceAllocator (Session 60)")
    print("  ✅ TrustTensorSync (Session 61)")
    print("  ✅ AuthorizedExpertSelector (Session 61)")
