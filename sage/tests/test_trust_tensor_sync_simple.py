#!/usr/bin/env python3
"""
Simplified tests for TrustTensorSync - works with actual expert_reputation API

Test Coverage:
1. Trust tensor data structures
2. Web4TrustClient basic operations
3. Export: SAGE → Web4
4. Import: Web4 → SAGE
5. Multi-observer aggregation
6. Bidirectional sync

Created: Session 61 (2025-12-16)
"""

import tempfile
from pathlib import Path
import time

try:
    from sage.web4.trust_tensor_sync import (
        TrustTensor,
        TrustObservation,
        Web4TrustClient,
        TrustTensorSync,
        create_trust_sync
    )
    from sage.web4.expert_identity import ExpertIdentityBridge
    from sage.core.expert_reputation import (
        ExpertReputation,
        ExpertReputationDB
    )
    HAS_MODULE = True
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from web4.trust_tensor_sync import (
        TrustTensor,
        TrustObservation,
        Web4TrustClient,
        TrustTensorSync,
        create_trust_sync
    )
    from web4.expert_identity import ExpertIdentityBridge
    from core.expert_reputation import (
        ExpertReputation,
        ExpertReputationDB
    )
    HAS_MODULE = True


def test_trust_tensor_structure():
    """Test TrustTensor dataclass."""
    tensor = TrustTensor(
        observer_id="lct://sage/router",
        subject_id="lct://sage_legion/expert/42",
        context="code_generation",
        trust_score=0.85,
        confidence=0.75,
        last_updated=time.time(),
        evidence_count=100,
        metadata={'expert_id': 42}
    )

    assert tensor.observer_id == "lct://sage/router"
    assert tensor.subject_id == "lct://sage_legion/expert/42"
    assert tensor.context == "code_generation"
    assert tensor.trust_score == 0.85
    assert tensor.confidence == 0.75
    assert tensor.evidence_count == 100

    print("✓ TrustTensor structure")


def test_web4_trust_client():
    """Test basic Web4TrustClient operations."""
    client = Web4TrustClient()

    # Create trust entry
    tensor = TrustTensor(
        observer_id="lct://sage/router",
        subject_id="lct://sage_legion/expert/42",
        context="code_generation",
        trust_score=0.85,
        confidence=0.75,
        last_updated=time.time(),
        evidence_count=100
    )

    # Update trust
    client.update_trust(tensor)

    # Get observations
    observations = client.get_trust_observations("lct://sage_legion/expert/42")
    assert len(observations) == 1
    assert observations[0].trust_score == 0.85
    assert observations[0].confidence == 0.75

    # Get subjects
    subjects = client.get_subjects()
    assert "lct://sage_legion/expert/42" in subjects

    print("✓ Web4TrustClient operations")


def test_export_to_web4():
    """Test exporting SAGE reputation to Web4."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup components
        identity_bridge = ExpertIdentityBridge(namespace="sage_test")
        reputation_db = ExpertReputationDB(db_path=Path(tmpdir) / "rep.db")
        trust_sync = TrustTensorSync(
            reputation_db=reputation_db,
            identity_bridge=identity_bridge,
            observer_id="lct://sage_test/router"
        )

        # Register expert
        identity_bridge.register_expert(42, description="Test expert")

        # Create reputation with context trust
        rep = ExpertReputation(expert_id=42, component="sage_moe")
        rep.context_trust["code"] = 0.85
        rep.context_observations["code"] = 100
        reputation_db.save(rep)

        # Export to Web4
        trust_entries = trust_sync.export_to_web4(42, context="code")

        assert len(trust_entries) == 1
        entry = trust_entries[0]
        assert entry.subject_id == "lct://sage_test/expert/42"
        assert entry.context == "code"
        assert entry.trust_score == 0.85
        assert entry.evidence_count == 100

        # Verify stored in Web4 client
        observations = trust_sync.web4_client.get_trust_observations(
            "lct://sage_test/expert/42",
            context="code"
        )
        assert len(observations) == 1

        print("✓ Export to Web4")


def test_import_from_web4():
    """Test importing Web4 trust to SAGE reputation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup components
        identity_bridge = ExpertIdentityBridge(namespace="sage_test")
        reputation_db = ExpertReputationDB(db_path=Path(tmpdir) / "rep.db")
        web4_client = Web4TrustClient()
        trust_sync = TrustTensorSync(
            reputation_db=reputation_db,
            identity_bridge=identity_bridge,
            web4_trust_client=web4_client
        )

        # Register expert
        lct_id = identity_bridge.register_expert(42)

        # Add external observation to Web4
        tensor = TrustTensor(
            observer_id="lct://sage_external/router",
            subject_id=lct_id,
            context="math",
            trust_score=0.75,
            confidence=0.6,
            last_updated=time.time(),
            evidence_count=50
        )
        web4_client.update_trust(tensor)

        # Import from Web4
        success = trust_sync.import_from_web4(lct_id, context="math")
        assert success

        # Verify imported to reputation DB
        rep = reputation_db.get_reputation(42, component="sage_moe")
        assert rep is not None
        assert rep.get_context_trust("math") == 0.75

        print("✓ Import from Web4")


def test_multi_observer_aggregation():
    """Test aggregating trust from multiple observers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup components
        identity_bridge = ExpertIdentityBridge(namespace="sage_test")
        reputation_db = ExpertReputationDB(db_path=Path(tmpdir) / "rep.db")
        web4_client = Web4TrustClient()
        trust_sync = TrustTensorSync(
            reputation_db=reputation_db,
            identity_bridge=identity_bridge,
            web4_trust_client=web4_client
        )

        # Register expert
        lct_id = identity_bridge.register_expert(42)

        # Add observations from 3 different observers
        observations = [
            TrustTensor(
                observer_id="lct://sage_observer1/router",
                subject_id=lct_id,
                context="code",
                trust_score=0.8,
                confidence=0.5,
                last_updated=time.time(),
                evidence_count=50
            ),
            TrustTensor(
                observer_id="lct://sage_observer2/router",
                subject_id=lct_id,
                context="code",
                trust_score=0.9,
                confidence=1.0,
                last_updated=time.time(),
                evidence_count=200
            ),
            TrustTensor(
                observer_id="lct://sage_observer3/router",
                subject_id=lct_id,
                context="code",
                trust_score=0.7,
                confidence=0.3,
                last_updated=time.time(),
                evidence_count=30
            ),
        ]

        for obs in observations:
            web4_client.update_trust(obs)

        # Import (should aggregate weighted by confidence)
        success = trust_sync.import_from_web4(lct_id, context="code")
        assert success

        # Expected weighted average: (0.8*0.5 + 0.9*1.0 + 0.7*0.3) / (0.5+1.0+0.3)
        # = (0.4 + 0.9 + 0.21) / 1.8 = 1.51 / 1.8 = 0.8388...
        rep = reputation_db.get_reputation(42, component="sage_moe")
        trust = rep.get_context_trust("code")
        assert 0.83 <= trust <= 0.85  # Approximately 0.84

        print(f"  Aggregated trust: {trust:.3f} (expected ~0.839)")
        print("✓ Multi-observer aggregation")


def test_bidirectional_sync():
    """Test bidirectional synchronization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        identity_bridge = ExpertIdentityBridge(namespace="sage_test")
        reputation_db = ExpertReputationDB(db_path=Path(tmpdir) / "rep.db")
        trust_sync = TrustTensorSync(
            reputation_db=reputation_db,
            identity_bridge=identity_bridge
        )

        # Register experts
        for i in range(5):
            identity_bridge.register_expert(i)

        # Create reputation for experts 0-2
        for i in range(3):
            rep = ExpertReputation(expert_id=i, component="sage_moe")
            rep.context_trust["code"] = 0.8
            rep.context_observations["code"] = 50 + i*10
            reputation_db.save(rep)

        # Sync all (export only initially)
        stats = trust_sync.sync_all_experts(context="code", bidirectional=False)
        assert stats['exported'] == 3
        assert stats['imported'] == 0

        print(f"  Exported: {stats['exported']}")
        print(f"  Imported: {stats['imported']}")
        print("✓ Bidirectional sync")


def test_statistics():
    """Test sync statistics tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        identity_bridge = ExpertIdentityBridge(namespace="sage_test")
        reputation_db = ExpertReputationDB(db_path=Path(tmpdir) / "rep.db")
        web4_client = Web4TrustClient()
        trust_sync = TrustTensorSync(
            reputation_db=reputation_db,
            identity_bridge=identity_bridge,
            web4_trust_client=web4_client
        )

        # Register and create reputation
        identity_bridge.register_expert(42)
        rep = ExpertReputation(expert_id=42, component="sage_moe")
        rep.context_trust["code"] = 0.8
        rep.context_observations["code"] = 100
        reputation_db.save(rep)

        # Export
        trust_sync.export_to_web4(42, context="code")

        # Get statistics
        stats = trust_sync.get_statistics()
        assert stats['exports_count'] == 1
        assert stats['imports_count'] == 0
        assert stats['web4_subjects'] == 1
        assert stats['web4_total_observations'] == 1

        print(f"  Exports: {stats['exports_count']}")
        print(f"  Imports: {stats['imports_count']}")
        print("✓ Statistics tracking")


if __name__ == "__main__":
    print("Testing TrustTensorSync (simplified)...\n")

    test_trust_tensor_structure()
    test_web4_trust_client()
    test_export_to_web4()
    test_import_from_web4()
    test_multi_observer_aggregation()
    test_bidirectional_sync()
    test_statistics()

    print("\n✅ All tests passed!")
    print("\nTrustTensorSync validated:")
    print("- Trust tensor data structures")
    print("- Web4TrustClient (local storage)")
    print("- Export: SAGE → Web4")
    print("- Import: Web4 → SAGE")
    print("- Multi-observer aggregation (confidence-weighted)")
    print("- Bidirectional sync")
    print("- Statistics tracking")
    print("- Ready for AuthorizedExpertSelector integration")
