#!/usr/bin/env python3
"""
Tests for ExpertIdentityBridge

Validates expert identity registration and Web4 LCT integration.

Test Coverage:
1. Expert registration (single and batch)
2. Bidirectional lookup (expert_id ↔ lct_id)
3. Identity metadata management
4. Registry persistence (save/load)
5. Statistics tracking
6. Registry validation
7. Error handling
8. Convenience functions

Created: Session 59 (2025-12-16)
"""

import tempfile
from pathlib import Path
import time

try:
    from sage.web4.expert_identity import (
        ExpertIdentityBridge,
        ExpertIdentity,
        IdentityStats,
        create_identity_bridge,
        register_expert_with_lct,
        lookup_expert_lct
    )
    HAS_MODULE = True
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from web4.expert_identity import (
        ExpertIdentityBridge,
        ExpertIdentity,
        IdentityStats,
        create_identity_bridge,
        register_expert_with_lct,
        lookup_expert_lct
    )
    HAS_MODULE = True


def test_initialization():
    """Test bridge initialization with various configurations."""
    # Default initialization
    bridge = ExpertIdentityBridge()
    assert bridge.namespace == "sage"
    assert len(bridge.expert_to_lct) == 0
    assert bridge.auto_save == True

    # Custom configuration
    bridge = ExpertIdentityBridge(
        namespace="sage_custom",
        auto_save=False
    )
    assert bridge.namespace == "sage_custom"
    assert bridge.auto_save == False

    print("✓ Initialization")


def test_expert_registration():
    """Test registering individual experts."""
    bridge = ExpertIdentityBridge(namespace="sage_test")

    # Register expert
    lct_id = bridge.register_expert(42, description="Test expert")
    assert lct_id == "lct://sage_test/expert/42"
    assert bridge.is_registered(42)
    assert bridge.get_lct(42) == lct_id
    assert bridge.get_expert_id(lct_id) == 42

    # Register another
    lct_id_2 = bridge.register_expert(7, description="Another expert")
    assert lct_id_2 == "lct://sage_test/expert/7"
    assert bridge.is_registered(7)

    # Try to register duplicate (should fail)
    try:
        bridge.register_expert(42)
        assert False, "Should have raised ValueError for duplicate"
    except ValueError as e:
        assert "already registered" in str(e)

    print("✓ Expert registration")


def test_batch_registration():
    """Test registering multiple experts at once."""
    bridge = ExpertIdentityBridge(namespace="sage_batch")

    # Register batch
    expert_ids = [1, 2, 3, 4, 5]
    descriptions = {
        1: "Code expert",
        2: "Math expert",
        3: "Reasoning expert"
    }

    results = bridge.register_batch(expert_ids, descriptions)

    assert len(results) == 5
    for expert_id in expert_ids:
        assert expert_id in results
        assert results[expert_id].startswith("lct://sage_batch/expert/")
        assert bridge.is_registered(expert_id)

    # Re-registering same batch should be idempotent
    results_2 = bridge.register_batch(expert_ids)
    assert results == results_2  # Same LCT IDs

    print("✓ Batch registration")


def test_bidirectional_lookup():
    """Test expert_id ↔ lct_id bidirectional lookup."""
    bridge = ExpertIdentityBridge(namespace="sage_lookup")

    # Register experts
    for i in range(10):
        bridge.register_expert(i)

    # Test forward lookup (expert_id → lct_id)
    for i in range(10):
        lct_id = bridge.get_lct(i)
        assert lct_id == f"lct://sage_lookup/expert/{i}"

    # Test reverse lookup (lct_id → expert_id)
    for i in range(10):
        lct_id = f"lct://sage_lookup/expert/{i}"
        expert_id = bridge.get_expert_id(lct_id)
        assert expert_id == i

    # Test non-existent lookups
    assert bridge.get_lct(999) is None
    assert bridge.get_expert_id("lct://sage_lookup/expert/999") is None

    print("✓ Bidirectional lookup")


def test_identity_metadata():
    """Test identity metadata management."""
    bridge = ExpertIdentityBridge(namespace="sage_meta")

    # Register with metadata
    metadata = {
        "specialization": "code_generation",
        "training_date": "2025-12-01",
        "version": "1.0"
    }
    bridge.register_expert(42, description="Code expert", metadata=metadata)

    # Retrieve identity
    identity = bridge.get_identity(42)
    assert identity is not None
    assert identity.expert_id == 42
    assert identity.description == "Code expert"
    assert identity.metadata["specialization"] == "code_generation"

    # Update metadata (merge)
    bridge.update_metadata(42, {"version": "1.1", "quality": "high"}, merge=True)
    identity = bridge.get_identity(42)
    assert identity.metadata["version"] == "1.1"  # Updated
    assert identity.metadata["specialization"] == "code_generation"  # Preserved
    assert identity.metadata["quality"] == "high"  # Added

    # Update metadata (replace)
    bridge.update_metadata(42, {"new_field": "value"}, merge=False)
    identity = bridge.get_identity(42)
    assert "new_field" in identity.metadata
    assert "specialization" not in identity.metadata  # Replaced

    print("✓ Identity metadata")


def test_registry_persistence():
    """Test save/load functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "registry.json"

        # Create bridge and register experts
        bridge1 = ExpertIdentityBridge(
            namespace="sage_persist",
            registry_path=registry_path
        )

        for i in range(5):
            bridge1.register_expert(i, description=f"Expert {i}")

        # Save should be automatic, but call explicitly
        bridge1.save()

        # Load into new bridge
        bridge2 = ExpertIdentityBridge.load(registry_path)

        # Verify same namespace
        assert bridge2.namespace == "sage_persist"

        # Verify all experts loaded
        assert len(bridge2.expert_to_lct) == 5
        for i in range(5):
            assert bridge2.is_registered(i)
            assert bridge2.get_lct(i) == bridge1.get_lct(i)

            identity1 = bridge1.get_identity(i)
            identity2 = bridge2.get_identity(i)
            assert identity1.expert_id == identity2.expert_id
            assert identity1.lct_id == identity2.lct_id
            assert identity1.description == identity2.description

        print(f"  Saved and loaded {len(bridge2.expert_to_lct)} experts")
        print("✓ Registry persistence")


def test_statistics():
    """Test statistics tracking."""
    bridge = ExpertIdentityBridge(namespace="sage_stats")

    # Register experts over time (simulate)
    for i in range(10):
        bridge.register_expert(i)
        time.sleep(0.001)  # Small delay to show registration rate

    # Get statistics
    stats = bridge.get_statistics()

    assert isinstance(stats, IdentityStats)
    assert stats.total_experts == 10
    assert stats.active_experts == 10
    assert "sage_stats" in stats.namespaces
    assert stats.registration_rate > 0
    assert stats.last_registration > 0

    print(f"  Total experts: {stats.total_experts}")
    print(f"  Registration rate: {stats.registration_rate:.2f} per day")
    print("✓ Statistics")


def test_registry_validation():
    """Test registry consistency validation."""
    bridge = ExpertIdentityBridge(namespace="sage_validate")

    # Register some experts
    for i in range(5):
        bridge.register_expert(i)

    # Should validate successfully
    assert bridge.validate_registry() == True

    # Corrupt registry (break bidirectional mapping)
    bridge.expert_to_lct[99] = "lct://sage_validate/expert/99"
    # But don't add to lct_to_expert (inconsistent!)

    # Should fail validation
    assert bridge.validate_registry() == False

    # Fix corruption
    del bridge.expert_to_lct[99]
    assert bridge.validate_registry() == True

    print("✓ Registry validation")


def test_get_all_methods():
    """Test methods that return all experts/LCTs."""
    bridge = ExpertIdentityBridge(namespace="sage_all")

    expert_ids = [1, 5, 10, 42, 100]
    for expert_id in expert_ids:
        bridge.register_expert(expert_id)

    # Get all experts
    all_experts = bridge.get_all_experts()
    assert set(all_experts) == set(expert_ids)

    # Get all LCT IDs
    all_lcts = bridge.get_all_lct_ids()
    assert len(all_lcts) == len(expert_ids)
    for lct_id in all_lcts:
        assert lct_id.startswith("lct://sage_all/expert/")

    print(f"  {len(all_experts)} experts registered")
    print(f"  {len(all_lcts)} LCT IDs generated")
    print("✓ Get all methods")


def test_error_handling():
    """Test error handling for invalid operations."""
    bridge = ExpertIdentityBridge(namespace="sage_errors")

    # Update metadata for non-existent expert
    try:
        bridge.update_metadata(999, {"key": "value"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not registered" in str(e)

    # Load from non-existent path
    try:
        ExpertIdentityBridge.load(Path("/tmp/nonexistent_registry.json"))
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

    # Save without registry path
    bridge_no_path = ExpertIdentityBridge(namespace="test", registry_path=None)
    try:
        bridge_no_path.save()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No registry path" in str(e)

    print("✓ Error handling")


def test_convenience_functions():
    """Test convenience wrapper functions."""
    # Create bridge
    bridge = create_identity_bridge(namespace="sage_convenience")
    assert isinstance(bridge, ExpertIdentityBridge)
    assert bridge.namespace == "sage_convenience"

    # Register with convenience function
    lct_id = register_expert_with_lct(bridge, 42, description="Test")
    assert lct_id == "lct://sage_convenience/expert/42"

    # Lookup with convenience function
    result = lookup_expert_lct(bridge, 42)
    assert result == lct_id

    # Lookup non-existent
    result = lookup_expert_lct(bridge, 999)
    assert result is None

    print("✓ Convenience functions")


def test_namespace_isolation():
    """Test that different namespaces are isolated."""
    bridge1 = ExpertIdentityBridge(namespace="sage_instance1")
    bridge2 = ExpertIdentityBridge(namespace="sage_instance2")

    # Register same expert ID in both
    lct1 = bridge1.register_expert(42)
    lct2 = bridge2.register_expert(42)

    # Should have different LCT IDs (different namespaces)
    assert lct1 != lct2
    assert lct1 == "lct://sage_instance1/expert/42"
    assert lct2 == "lct://sage_instance2/expert/42"

    # Lookups should be isolated
    assert bridge1.get_expert_id(lct1) == 42
    assert bridge1.get_expert_id(lct2) is None  # Not in bridge1
    assert bridge2.get_expert_id(lct2) == 42
    assert bridge2.get_expert_id(lct1) is None  # Not in bridge2

    print("✓ Namespace isolation")


def test_auto_save():
    """Test automatic saving after registration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "auto_save_registry.json"

        # Create bridge with auto-save enabled
        bridge = ExpertIdentityBridge(
            namespace="sage_auto",
            registry_path=registry_path,
            auto_save=True
        )

        # Register expert (should auto-save)
        bridge.register_expert(42)

        # Registry file should exist
        assert registry_path.exists()

        # Load should work
        bridge2 = ExpertIdentityBridge.load(registry_path)
        assert bridge2.is_registered(42)

        print("✓ Auto-save")


if __name__ == "__main__":
    print("Testing ExpertIdentityBridge...\n")

    test_initialization()
    test_expert_registration()
    test_batch_registration()
    test_bidirectional_lookup()
    test_identity_metadata()
    test_registry_persistence()
    test_statistics()
    test_registry_validation()
    test_get_all_methods()
    test_error_handling()
    test_convenience_functions()
    test_namespace_isolation()
    test_auto_save()

    print("\n✅ All tests passed!")
    print("\nExpertIdentityBridge validated:")
    print("- Expert registration (single and batch)")
    print("- Bidirectional lookup (expert_id ↔ lct_id)")
    print("- Identity metadata management")
    print("- Registry persistence (save/load)")
    print("- Statistics tracking")
    print("- Registry validation")
    print("- Namespace isolation")
    print("- Ready for Web4 integration")
