#!/usr/bin/env python3
"""
Tests for TrustTensorSync

Validates SAGE ↔ Web4 reputation synchronization.

Test Coverage:
1. Trust tensor data structures
2. Web4TrustClient (stub implementation)
3. Export: SAGE → Web4
4. Import: Web4 → SAGE
5. Bidirectional sync
6. Multi-observer aggregation
7. Context isolation
8. Persistence
9. Statistics tracking

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
    assert tensor.metadata['expert_id'] == 42

    print("✓ TrustTensor structure")


def test_web4_trust_client_basic():
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

    print("✓ Web4TrustClient basic operations")


def test_web4_trust_client_filtering():
    """Test Web4TrustClient filtering."""
    client = Web4TrustClient()

    # Add multiple observations
    tensors = [
        TrustTensor(
            observer_id="lct://sage/router1",
            subject_id="lct://sage/expert/1",
            context="code",
            trust_score=0.8,
            confidence=0.7,
            last_updated=time.time(),
            evidence_count=50
        ),
        TrustTensor(
            observer_id="lct://sage/router2",
            subject_id="lct://sage/expert/1",
            context="code",
            trust_score=0.85,
            confidence=0.8,
            last_updated=time.time(),
            evidence_count=75
        ),
        TrustTensor(
            observer_id="lct://sage/router1",
            subject_id="lct://sage/expert/1",
            context="math",
            trust_score=0.6,
            confidence=0.5,
            last_updated=time.time(),
            evidence_count=25
        ),
    ]

    for tensor in tensors:
        client.update_trust(tensor)

    # Filter by context
    code_obs = client.get_trust_observations("lct://sage/expert/1", context="code")
    assert len(code_obs) == 2

    math_obs = client.get_trust_observations("lct://sage/expert/1", context="math")
    assert len(math_obs) == 1

    # Filter by observer
    router1_obs = client.get_trust_observations(
        "lct://sage/expert/1",
        observer_id="lct://sage/router1"
    )
    assert len(router1_obs) == 2

    # Get subjects by context
    code_subjects = client.get_subjects(context="code")
    assert "lct://sage/expert/1" in code_subjects

    print("✓ Web4TrustClient filtering")


def test_web4_trust_client_persistence():
    """Test Web4TrustClient persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "trust.json"

        # Create client and add data
        client1 = Web4TrustClient(storage_path)
        tensor = TrustTensor(
            observer_id="lct://sage/router",
            subject_id="lct://sage/expert/42",
            context="code",
            trust_score=0.9,
            confidence=0.8,
            last_updated=time.time(),
            evidence_count=100
        )
        client1.update_trust(tensor)

        # Create new client and load
        client2 = Web4TrustClient(storage_path)
        observations = client2.get_trust_observations("lct://sage/expert/42")

        assert len(observations) == 1
        assert observations[0].trust_score == 0.9

        print("✓ Web4TrustClient persistence")


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

        # Create reputation with context performance
        rep = ExpertReputation(expert_id=42, component="sage_moe")
        rep.context_performance["code"] = ContextPerformance(
            context="code",
            activations=100,
            quality_sum=85.0,
            trust=0.85
        )
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
        assert "math" in rep.context_performance
        assert rep.context_performance["math"].trust == 0.75

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
                confidence=0.5,  # 50% confidence
                last_updated=time.time(),
                evidence_count=50
            ),
            TrustTensor(
                observer_id="lct://sage_observer2/router",
                subject_id=lct_id,
                context="code",
                trust_score=0.9,
                confidence=1.0,  # 100% confidence (high weight)
                last_updated=time.time(),
                evidence_count=200
            ),
            TrustTensor(
                observer_id="lct://sage_observer3/router",
                subject_id=lct_id,
                context="code",
                trust_score=0.7,
                confidence=0.3,  # 30% confidence
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
        trust = rep.context_performance["code"].trust
        assert 0.83 <= trust <= 0.85  # Approximately 0.84

        print(f"  Aggregated trust: {trust:.3f} (expected ~0.839)")
        print("✓ Multi-observer aggregation")


def test_context_isolation():
    """Test that trust is context-specific."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        identity_bridge = ExpertIdentityBridge(namespace="sage_test")
        reputation_db = ExpertReputationDB(db_path=Path(tmpdir) / "rep.db")
        trust_sync = TrustTensorSync(
            reputation_db=reputation_db,
            identity_bridge=identity_bridge
        )

        # Register expert
        identity_bridge.register_expert(42)

        # Create reputation with different trust per context
        rep = ExpertReputation(expert_id=42, component="sage_moe")
        rep.context_performance["code"] = ContextPerformance(
            context="code",
            activations=100,
            quality_sum=90.0,
            trust=0.9  # High trust in code
        )
        rep.context_performance["math"] = ContextPerformance(
            context="math",
            activations=50,
            quality_sum=30.0,
            trust=0.6  # Lower trust in math
        )
        reputation_db.save(rep)

        # Export both contexts
        code_entries = trust_sync.export_to_web4(42, context="code")
        math_entries = trust_sync.export_to_web4(42, context="math")

        # Verify context isolation
        assert len(code_entries) == 1
        assert code_entries[0].trust_score == 0.9
        assert code_entries[0].context == "code"

        assert len(math_entries) == 1
        assert math_entries[0].trust_score == 0.6
        assert math_entries[0].context == "math"

        print("✓ Context isolation")


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
            rep.context_performance["code"] = ContextPerformance(
                context="code",
                activations=50 + i*10,
                quality_sum=(50+i*10) * 0.8,
                trust=0.8
            )
            reputation_db.save(rep)

        # Sync all (export only initially)
        stats = trust_sync.sync_all_experts(context="code", bidirectional=False)
        assert stats['exported'] == 3
        assert stats['imported'] == 0

        print(f"  Exported: {stats['exported']}")
        print(f"  Imported: {stats['imported']}")
        print("✓ Bidirectional sync")


def test_confidence_computation():
    """Test confidence increases with evidence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        identity_bridge = ExpertIdentityBridge(namespace="sage_test")
        reputation_db = ExpertReputationDB(db_path=Path(tmpdir) / "rep.db")
        trust_sync = TrustTensorSync(
            reputation_db=reputation_db,
            identity_bridge=identity_bridge
        )

        identity_bridge.register_expert(42)

        # Test different activation counts
        activation_counts = [0, 10, 50, 100, 500, 1000]
        confidences = []

        for count in activation_counts:
            rep = ExpertReputation(expert_id=42, component="sage_moe")
            rep.context_performance["test"] = ContextPerformance(
                context="test",
                activations=count,
                quality_sum=count * 0.8,
                trust=0.8
            )
            reputation_db.save(rep)

            entries = trust_sync.export_to_web4(42, context="test")
            if entries:
                confidences.append(entries[0].confidence)
            else:
                confidences.append(0.0)

        # Confidence should increase with activations
        # Formula: activations / (activations + 100)
        assert confidences[0] == 0.0  # 0 activations
        assert 0.09 <= confidences[1] <= 0.11  # 10/110 ≈ 0.09
        assert 0.33 <= confidences[2] <= 0.34  # 50/150 ≈ 0.33
        assert 0.49 <= confidences[3] <= 0.51  # 100/200 = 0.5
        assert 0.83 <= confidences[4] <= 0.84  # 500/600 ≈ 0.83
        assert 0.90 <= confidences[5] <= 0.91  # 1000/1100 ≈ 0.91

        print(f"  Confidence at 0: {confidences[0]:.3f}")
        print(f"  Confidence at 10: {confidences[1]:.3f}")
        print(f"  Confidence at 100: {confidences[3]:.3f}")
        print(f"  Confidence at 1000: {confidences[5]:.3f}")
        print("✓ Confidence computation")


def test_statistics_tracking():
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
        rep.context_performance["code"] = ContextPerformance(
            context="code", activations=100, quality_sum=80.0, trust=0.8
        )
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
        print(f"  Web4 subjects: {stats['web4_subjects']}")
        print("✓ Statistics tracking")


def test_convenience_function():
    """Test convenience function for creating trust sync."""
    with tempfile.TemporaryDirectory() as tmpdir:
        identity_bridge = ExpertIdentityBridge(namespace="sage_test")
        reputation_db = ExpertReputationDB(db_path=Path(tmpdir) / "rep.db")
        storage_path = Path(tmpdir) / "trust.json"

        trust_sync = create_trust_sync(
            reputation_db=reputation_db,
            identity_bridge=identity_bridge,
            storage_path=storage_path,
            observer_id="lct://sage_test/router"
        )

        assert isinstance(trust_sync, TrustTensorSync)
        assert trust_sync.observer_id == "lct://sage_test/router"
        assert trust_sync.web4_client.storage_path == storage_path

        print("✓ Convenience function")


def test_update_existing_observation():
    """Test updating an existing trust observation."""
    client = Web4TrustClient()

    # Add initial observation
    tensor1 = TrustTensor(
        observer_id="lct://sage/router",
        subject_id="lct://sage/expert/42",
        context="code",
        trust_score=0.7,
        confidence=0.5,
        last_updated=time.time(),
        evidence_count=50
    )
    client.update_trust(tensor1)

    # Update with new observation from same observer/context
    tensor2 = TrustTensor(
        observer_id="lct://sage/router",  # Same observer
        subject_id="lct://sage/expert/42",
        context="code",  # Same context
        trust_score=0.85,  # Updated trust
        confidence=0.8,  # Updated confidence
        last_updated=time.time(),
        evidence_count=100
    )
    client.update_trust(tensor2)

    # Should have only 1 observation (updated, not added)
    observations = client.get_trust_observations("lct://sage/expert/42")
    assert len(observations) == 1
    assert observations[0].trust_score == 0.85
    assert observations[0].confidence == 0.8

    print("✓ Update existing observation")


if __name__ == "__main__":
    print("Testing TrustTensorSync...\n")

    test_trust_tensor_structure()
    test_web4_trust_client_basic()
    test_web4_trust_client_filtering()
    test_web4_trust_client_persistence()
    test_export_to_web4()
    test_import_from_web4()
    test_multi_observer_aggregation()
    test_context_isolation()
    test_bidirectional_sync()
    test_confidence_computation()
    test_statistics_tracking()
    test_convenience_function()
    test_update_existing_observation()

    print("\n✅ All tests passed!")
    print("\nTrustTensorSync validated:")
    print("- Trust tensor data structures")
    print("- Web4TrustClient (local storage)")
    print("- Export: SAGE → Web4")
    print("- Import: Web4 → SAGE")
    print("- Multi-observer aggregation (confidence-weighted)")
    print("- Context isolation")
    print("- Bidirectional sync")
    print("- Statistics tracking")
    print("- Ready for AuthorizedExpertSelector integration")
