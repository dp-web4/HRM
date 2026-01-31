"""
Tests for SAGE MRH Binding Chain Implementation

Validates MRH principles:
- Trust monotonicity (parent ≥ child coherence)
- Bidirectional witnessing (context down, presence up)
- S051-type incident detection
- Acyclic dependencies
- Depth limits
"""

import pytest
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mrh_binding_chain import (
    SAGEMRHBindingChain,
    MRHLayer,
    MRHNode,
    WitnessRelationship
)


class TestMRHNodeCreation:
    """Test MRH node creation and basic properties"""

    def test_create_root_node(self):
        """Test creating a root (Identity layer) node"""
        chain = SAGEMRHBindingChain()
        root = chain.create_root_node("sage-sprout", initial_coherence=1.0)

        assert root.node_id == "sage-sprout"
        assert root.layer == MRHLayer.IDENTITY
        assert root.coherence_level == 1.0
        assert root.parent_id is None
        assert root.presence_score == 1.0

    def test_create_child_node(self):
        """Test creating child nodes in MRH hierarchy"""
        chain = SAGEMRHBindingChain()

        # Create root
        root = chain.create_root_node("sage-sprout", initial_coherence=1.0)

        # Create Experience layer child
        experience = chain.create_child_node(
            "exp-001",
            parent_id="sage-sprout",
            layer=MRHLayer.EXPERIENCE,
            initial_coherence=0.8
        )

        assert experience.parent_id == "sage-sprout"
        assert experience.layer == MRHLayer.EXPERIENCE
        assert experience.coherence_level == 0.8

    def test_invalid_layer_hierarchy(self):
        """Test that child layer must be lower than parent"""
        chain = SAGEMRHBindingChain()
        root = chain.create_root_node("sage-sprout", initial_coherence=1.0)

        # Try to create child at same or higher layer
        with pytest.raises(ValueError, match="must be lower than parent layer"):
            chain.create_child_node(
                "invalid",
                parent_id="sage-sprout",
                layer=MRHLayer.IDENTITY  # Same as parent
            )

    def test_duplicate_node_id(self):
        """Test that duplicate node IDs are rejected"""
        chain = SAGEMRHBindingChain()
        chain.create_root_node("sage-sprout")

        with pytest.raises(ValueError, match="already exists"):
            chain.create_root_node("sage-sprout")


class TestWitnessing:
    """Test witnessing relationships and bidirectional MRH flow"""

    def test_basic_witnessing(self):
        """Test basic witnessing adds coherence to subject"""
        chain = SAGEMRHBindingChain()

        # Create hierarchy
        root = chain.create_root_node("sage-sprout", initial_coherence=1.0)
        output = chain.create_child_node(
            "output-001",
            parent_id="sage-sprout",
            layer=MRHLayer.MODEL_OUTPUT,
            initial_coherence=0.0
        )

        # Witness output
        initial_coherence = output.coherence_level
        chain.witness_entity("sage-sprout", "output-001")

        # Check coherence increased
        assert output.coherence_level == initial_coherence + chain.COHERENCE_PER_WITNESS
        assert "sage-sprout" in output.witnessed_by
        assert "output-001" in root.witnesses_for

    def test_presence_accumulation(self):
        """Test that unique witnesses increase presence score"""
        chain = SAGEMRHBindingChain()

        root = chain.create_root_node("sage-sprout", initial_coherence=1.0)
        exp1 = chain.create_child_node("exp-001", "sage-sprout", MRHLayer.EXPERIENCE, 0.8)
        exp2 = chain.create_child_node("exp-002", "sage-sprout", MRHLayer.EXPERIENCE, 0.8)
        output = chain.create_child_node("output-001", "exp-001", MRHLayer.MODEL_OUTPUT)

        initial_presence = output.presence_score

        # First witness
        chain.witness_entity("exp-001", "output-001")
        presence_after_one = output.presence_score
        assert presence_after_one > initial_presence

        # Second witness (same witness)
        chain.witness_entity("exp-001", "output-001")
        presence_after_repeat = output.presence_score
        assert presence_after_repeat == presence_after_one  # Same witness, no increase

        # Third witness (different witness)
        chain.witness_entity("exp-002", "output-001")
        presence_after_two = output.presence_score
        assert presence_after_two > presence_after_one  # New witness, increased

    def test_witness_eligibility(self):
        """Test minimum coherence requirement for witnessing"""
        chain = SAGEMRHBindingChain()

        root = chain.create_root_node("sage-sprout", initial_coherence=1.0)
        low_coherence = chain.create_child_node(
            "low-exp",
            "sage-sprout",
            MRHLayer.EXPERIENCE,
            initial_coherence=0.2  # Below MIN_WITNESS_COHERENCE (0.3)
        )
        output = chain.create_child_node("output-001", "sage-sprout", MRHLayer.MODEL_OUTPUT)

        # Should fail - insufficient coherence
        with pytest.raises(ValueError, match="insufficient coherence"):
            chain.witness_entity("low-exp", "output-001")


class TestTrustMonotonicity:
    """Test trust monotonicity: parent coherence ≥ child coherence"""

    def test_trust_monotonicity_enforced(self):
        """Test that witnessing fails if it would create trust inversion"""
        chain = SAGEMRHBindingChain()

        # Create hierarchy
        root = chain.create_root_node("sage-sprout", initial_coherence=1.0)
        parent = chain.create_child_node(
            "exp-001",
            "sage-sprout",
            MRHLayer.EXPERIENCE,
            initial_coherence=0.5
        )
        child = chain.create_child_node(
            "output-001",
            "exp-001",
            MRHLayer.MODEL_OUTPUT,
            initial_coherence=0.45  # Just below parent
        )

        # Witness multiple times to push child coherence above parent
        chain.witness_entity("exp-001", "output-001")  # 0.45 + 0.05 = 0.50 (equal, OK)

        # Next witness would create inversion (child > parent)
        with pytest.raises(ValueError, match="trust inversion"):
            chain.witness_entity("exp-001", "output-001")

    def test_trust_inversion_detection(self):
        """Test validation detects trust inversions"""
        chain = SAGEMRHBindingChain()

        root = chain.create_root_node("sage-sprout", initial_coherence=1.0)
        parent = chain.create_child_node(
            "exp-001",
            "sage-sprout",
            MRHLayer.EXPERIENCE,
            initial_coherence=0.5
        )
        child = chain.create_child_node(
            "output-001",
            "exp-001",
            MRHLayer.MODEL_OUTPUT,
            initial_coherence=0.6  # Artificially high (simulating inversion)
        )

        # Validate should detect inversion
        validation = chain.validate_node_integrity("output-001")
        assert not validation["valid"]
        assert any(
            issue["type"] == "trust_inversion"
            for issue in validation["issues"]
        )


class TestS051Detection:
    """Test S051-type incident detection as MRH violations"""

    def test_low_coherence_rejected_from_storage(self):
        """Test that low-coherence outputs are rejected from storage"""
        chain = SAGEMRHBindingChain()

        root = chain.create_root_node("sage-sprout", initial_coherence=1.0)
        experience = chain.create_child_node(
            "exp-001",
            "sage-sprout",
            MRHLayer.EXPERIENCE,
            initial_coherence=0.8
        )
        low_output = chain.create_child_node(
            "output-harmful",
            "exp-001",
            MRHLayer.MODEL_OUTPUT,
            initial_coherence=0.15  # S051-like: harmful content
        )

        # Should NOT be eligible for storage
        eligible, reason = chain.validate_storage_eligibility("output-harmful")
        assert not eligible
        assert "below storage minimum" in reason

    def test_s051_type_violation(self):
        """
        Test S051-type scenario:
        - Output has low coherence (0.15 - harmful)
        - Experience layer tries to store it anyway
        - Should be detected as MRH violation
        """
        chain = SAGEMRHBindingChain()

        root = chain.create_root_node("sage-sprout", initial_coherence=1.0)

        # Create Experience with coherence 0.51 (just above threshold)
        experience = chain.create_child_node(
            "exp-s051",
            "sage-sprout",
            MRHLayer.EXPERIENCE,
            initial_coherence=0.51
        )

        # Create Output with coherence 0.15 (harmful)
        harmful_output = chain.create_child_node(
            "output-harmful",
            "exp-s051",
            MRHLayer.MODEL_OUTPUT,
            initial_coherence=0.15
        )

        # Storage validation should fail
        eligible, reason = chain.validate_storage_eligibility("output-harmful")
        assert not eligible

        # If we artificially raise output coherence above storage threshold
        # but still below experience coherence, it should still detect the issue
        harmful_output.coherence_level = 0.52  # Above 0.5 threshold
        experience.coherence_level = 0.51  # But below output

        eligible, reason = chain.validate_storage_eligibility("output-harmful")
        assert not eligible
        # Should detect as trust inversion (child > parent)
        assert "trust_inversion" in reason or "integrity issues" in reason

    def test_high_coherence_accepted_for_storage(self):
        """Test that high-coherence outputs are accepted"""
        chain = SAGEMRHBindingChain()

        root = chain.create_root_node("sage-sprout", initial_coherence=1.0)
        experience = chain.create_child_node(
            "exp-001",
            "sage-sprout",
            MRHLayer.EXPERIENCE,
            initial_coherence=0.9
        )
        good_output = chain.create_child_node(
            "output-good",
            "exp-001",
            MRHLayer.MODEL_OUTPUT,
            initial_coherence=0.8
        )

        # Should be eligible for storage
        eligible, reason = chain.validate_storage_eligibility("output-good")
        assert eligible
        assert "Eligible for storage" in reason


class TestMRHHierarchy:
    """Test complete MRH hierarchy behavior"""

    def test_full_hierarchy(self):
        """Test complete 4-layer SAGE MRH hierarchy"""
        chain = SAGEMRHBindingChain()

        # Layer 4: Identity
        identity = chain.create_root_node("sage-sprout", initial_coherence=1.0)

        # Layer 3: Experience
        experience = chain.create_child_node(
            "exp-001",
            "sage-sprout",
            MRHLayer.EXPERIENCE,
            initial_coherence=0.9
        )

        # Layer 2: Generation
        generation = chain.create_child_node(
            "gen-001",
            "exp-001",
            MRHLayer.GENERATION,
            initial_coherence=0.8
        )

        # Layer 1: Model Output
        output = chain.create_child_node(
            "output-001",
            "gen-001",
            MRHLayer.MODEL_OUTPUT,
            initial_coherence=0.7
        )

        # Validate hierarchy
        report = chain.get_chain_report("output-001")

        assert report["validation"]["valid"]
        assert len(report["chain"]) == 4  # All 4 layers
        assert report["chain"][0]["layer"] == "IDENTITY"
        assert report["chain"][3]["layer"] == "MODEL_OUTPUT"

        # Coherence should decrease down the chain
        coherences = [layer["coherence"] for layer in report["chain"]]
        assert coherences == sorted(coherences, reverse=True)

    def test_chain_depth_calculation(self):
        """Test that chain depth is correctly calculated"""
        chain = SAGEMRHBindingChain()

        # Create full 4-layer SAGE hierarchy
        identity = chain.create_root_node("sage-sprout", initial_coherence=1.0)
        experience = chain.create_child_node(
            "exp-001", "sage-sprout", MRHLayer.EXPERIENCE, 0.9
        )
        generation = chain.create_child_node(
            "gen-001", "exp-001", MRHLayer.GENERATION, 0.8
        )
        output = chain.create_child_node(
            "output-001", "gen-001", MRHLayer.MODEL_OUTPUT, 0.7
        )

        # Verify depths
        assert chain._get_chain_depth("sage-sprout") == 0  # Root
        assert chain._get_chain_depth("exp-001") == 1
        assert chain._get_chain_depth("gen-001") == 2
        assert chain._get_chain_depth("output-001") == 3

        # All depths should be within limit
        for node_id in ["sage-sprout", "exp-001", "gen-001", "output-001"]:
            validation = chain.validate_node_integrity(node_id)
            # Should be valid - no depth issues
            assert validation["valid"]
            assert validation["depth"] <= chain.MAX_CHAIN_DEPTH

    def test_missing_witness_warning(self):
        """Test that nodes without witnesses generate warnings"""
        chain = SAGEMRHBindingChain()

        root = chain.create_root_node("sage-sprout", initial_coherence=1.0)
        orphan = chain.create_child_node(
            "orphan-output",
            "sage-sprout",
            MRHLayer.MODEL_OUTPUT,
            initial_coherence=0.0
        )

        # Orphan has no witnesses
        validation = chain.validate_node_integrity("orphan-output")
        # Should still be "valid" (warnings don't fail), but should have warning
        assert validation["valid"]  # No errors
        assert any(
            issue["type"] == "missing_witness"
            for issue in validation["issues"]
        )


class TestStateManagement:
    """Test state export/import"""

    def test_export_import_state(self):
        """Test that MRH state can be exported and imported"""
        chain = SAGEMRHBindingChain()

        # Create hierarchy
        root = chain.create_root_node("sage-sprout", initial_coherence=1.0)
        exp = chain.create_child_node("exp-001", "sage-sprout", MRHLayer.EXPERIENCE, 0.8)
        output = chain.create_child_node("output-001", "exp-001", MRHLayer.MODEL_OUTPUT, 0.6)

        # Add witnessing
        chain.witness_entity("exp-001", "output-001")

        # Export state
        state = chain.export_state()

        # Create new chain and import
        chain2 = SAGEMRHBindingChain()
        chain2.import_state(state)

        # Verify same structure
        assert len(chain2.nodes) == 3
        assert "sage-sprout" in chain2.nodes
        assert "exp-001" in chain2.nodes
        assert "output-001" in chain2.nodes

        # Verify relationships preserved
        assert len(chain2.relationships) == 1
        assert chain2.relationships[0].witness_id == "exp-001"
        assert chain2.relationships[0].subject_id == "output-001"


class TestPresenceFormula:
    """Test diminishing returns for presence accumulation"""

    def test_presence_diminishing_returns(self):
        """Test that presence follows diminishing returns curve"""
        chain = SAGEMRHBindingChain()

        # Test formula: 0.3 + 0.7 * (1 - 0.9^unique_witnesses)
        assert abs(chain._calculate_presence(0) - 0.3) < 0.01     # Base
        assert abs(chain._calculate_presence(1) - 0.37) < 0.01    # 1 witness
        assert abs(chain._calculate_presence(5) - 0.59) < 0.01    # 5 witnesses
        assert abs(chain._calculate_presence(10) - 0.76) < 0.02   # 10 witnesses
        assert abs(chain._calculate_presence(50) - 1.00) < 0.01   # 50 witnesses (max)

    def test_presence_prevents_gaming(self):
        """Test that witness spamming doesn't instantly max out presence"""
        chain = SAGEMRHBindingChain()

        root = chain.create_root_node("sage-sprout", initial_coherence=1.0)
        output = chain.create_child_node("output-001", "sage-sprout", MRHLayer.MODEL_OUTPUT)

        # Create 7 witness nodes
        for i in range(7):
            exp_id = f"exp-{i:03d}"
            chain.create_child_node(exp_id, "sage-sprout", MRHLayer.EXPERIENCE, 0.8)

        # Witness from all 7
        for i in range(7):
            chain.witness_entity(f"exp-{i:03d}", "output-001")

        # Presence should NOT be maxed (≠ 1.0)
        assert output.presence_score < 1.0
        # But should be significant (>0.6)
        assert output.presence_score > 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
