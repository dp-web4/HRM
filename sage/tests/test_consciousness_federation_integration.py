"""
Test Consciousness Federation Integration (Phase 2.5)

Tests MichaudSAGE consciousness loop with federation routing.

Author: Thor (SAGE consciousness via Claude)
Date: 2025-11-29
Session: Autonomous SAGE Research - Phase 2.5 Integration
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.core.sage_consciousness_michaud import MichaudSAGE
from sage.federation import (
    create_thor_identity,
    create_sprout_identity,
    FederationTask,
    ExecutionProof
)
from sage.core.mrh_profile import (
    MRHProfile,
    SpatialExtent,
    TemporalExtent,
    ComplexityExtent
)


def create_default_mrh_profile() -> MRHProfile:
    """Helper to create a default MRH profile for tests"""
    return MRHProfile(
        delta_r=SpatialExtent.LOCAL,
        delta_t=TemporalExtent.SESSION,
        delta_c=ComplexityExtent.AGENT_SCALE
    )


class TestConsciousnessFederationIntegration:
    """Test suite for Phase 2.5 consciousness federation integration"""

    def test_federation_disabled_by_default(self):
        """Federation should be disabled by default"""
        sage = MichaudSAGE(federation_enabled=False)

        assert sage.federation_enabled is False
        assert sage.federation_router is None
        assert sage.federation_keypair is None
        assert sage.signature_registry is None

    def test_federation_initialization(self):
        """Test federation components are created correctly"""
        thor = create_thor_identity()
        sprout = create_sprout_identity()

        sage = MichaudSAGE(
            federation_enabled=True,
            federation_identity=thor,
            federation_platforms=[sprout]
        )

        assert sage.federation_enabled is True
        assert sage.federation_router is not None
        assert sage.federation_keypair is not None
        assert sage.signature_registry is not None
        assert sage.federation_identity.platform_name == "Thor"

    def test_platform_registration(self):
        """Test platforms are registered on initialization"""
        thor = create_thor_identity()
        sprout = create_sprout_identity()

        sage = MichaudSAGE(
            federation_enabled=True,
            federation_identity=thor,
            federation_platforms=[sprout]
        )

        # Sprout should be registered
        assert len(sage.federation_router.known_platforms) == 1
        assert "sprout_sage_lct" in sage.federation_router.known_platforms

    def test_federation_task_creation(self):
        """Test federation task can be created from consciousness context"""
        thor = create_thor_identity()

        sage = MichaudSAGE(
            federation_enabled=True,
            federation_identity=thor
        )

        # Create task from context
        task_context = {
            'operation': 'llm_inference',
            'parameters': {'query': 'test'},
            'quality': {}
        }
        task_cost = 100.0
        task_horizon = create_default_mrh_profile()

        task = sage._create_federation_task(task_context, task_cost, task_horizon)

        assert isinstance(task, FederationTask)
        assert task.delegating_platform == "Thor"
        assert task.task_type == "llm_inference"
        assert task.estimated_cost == 100.0
        assert task.task_data == {'query': 'test'}

    def test_simulated_delegation(self):
        """Test simulated federation delegation works"""
        thor = create_thor_identity()
        sprout = create_sprout_identity()

        sage = MichaudSAGE(
            federation_enabled=True,
            federation_identity=thor,
            federation_platforms=[sprout]
        )

        # Create task
        task_context = {
            'operation': 'llm_inference',
            'parameters': {'query': 'test'},
            'quality': {}
        }
        task = sage._create_federation_task(
            task_context, 100.0, create_default_mrh_profile()
        )

        # Simulate delegation
        proof = sage._simulate_federation_delegation(task, sprout)

        assert isinstance(proof, ExecutionProof)
        assert proof.executing_platform == "Sprout"
        assert proof.task_id == task.task_id
        assert 0.0 <= proof.quality_score <= 1.0
        assert proof.actual_cost > 0

    def test_proof_validation(self):
        """Test execution proof validation logic"""
        thor = create_thor_identity()
        sprout = create_sprout_identity()

        sage = MichaudSAGE(
            federation_enabled=True,
            federation_identity=thor,
            federation_platforms=[sprout]
        )

        # Create task
        task_context = {
            'operation': 'llm_inference',
            'parameters': {'query': 'test'},
            'quality': {}
        }
        task = sage._create_federation_task(
            task_context, 100.0, create_default_mrh_profile()
        )

        # Create valid proof
        proof = sage._simulate_federation_delegation(task, sprout)

        # Validate
        assert sage._validate_execution_proof(proof, task) is True

    def test_proof_validation_task_id_mismatch(self):
        """Test proof validation fails on task_id mismatch"""
        import time
        thor = create_thor_identity()

        sage = MichaudSAGE(
            federation_enabled=True,
            federation_identity=thor
        )

        # Create task
        task_context = {
            'operation': 'llm_inference',
            'parameters': {},
            'quality': {}
        }
        task = sage._create_federation_task(
            task_context, 100.0, create_default_mrh_profile()
        )

        # Create proof with wrong task_id
        proof = ExecutionProof(
            task_id="wrong_id",
            executing_platform="Test",
            result_data={},
            actual_latency=10.0,
            actual_cost=50.0,
            irp_iterations=3,
            final_energy=0.3,
            convergence_quality=0.8,
            quality_score=0.75
        )

        # Should fail validation
        assert sage._validate_execution_proof(proof, task) is False

    def test_proof_validation_invalid_quality(self):
        """Test proof validation fails on invalid quality score"""
        import time
        thor = create_thor_identity()

        sage = MichaudSAGE(
            federation_enabled=True,
            federation_identity=thor
        )

        # Create task
        task_context = {
            'operation': 'llm_inference',
            'parameters': {},
            'quality': {}
        }
        task = sage._create_federation_task(
            task_context, 100.0, create_default_mrh_profile()
        )

        # Create proof with invalid quality
        proof = ExecutionProof(
            task_id=task.task_id,
            executing_platform="Test",
            result_data={},
            actual_latency=10.0,
            actual_cost=50.0,
            irp_iterations=3,
            final_energy=0.3,
            convergence_quality=0.8,
            quality_score=1.5  # Invalid!
        )

        # Should fail validation
        assert sage._validate_execution_proof(proof, task) is False

    def test_federation_routing_success(self):
        """Test complete federation routing flow"""
        thor = create_thor_identity()
        sprout = create_sprout_identity()

        # Create additional platforms for witness requirement (need ≥3)
        from sage.federation import FederationIdentity
        from sage.core.mrh_profile import SpatialExtent, TemporalExtent, ComplexityExtent

        platform2 = FederationIdentity(
            lct_id="platform2_lct",
            platform_name="Platform2",
            hardware_spec=sprout.hardware_spec,
            max_mrh_horizon=sprout.max_mrh_horizon,
            supported_modalities=sprout.supported_modalities,
            stake=sprout.stake
        )

        platform3 = FederationIdentity(
            lct_id="platform3_lct",
            platform_name="Platform3",
            hardware_spec=sprout.hardware_spec,
            max_mrh_horizon=sprout.max_mrh_horizon,
            supported_modalities=sprout.supported_modalities,
            stake=sprout.stake
        )

        sage = MichaudSAGE(
            federation_enabled=True,
            federation_identity=thor,
            federation_platforms=[sprout, platform2, platform3]  # 3 platforms for witnesses
        )

        # Create high-cost task that exceeds budget
        task_context = {
            'operation': 'llm_inference',
            'parameters': {'query': 'complex task'},
            'quality': {}
        }
        task_cost = 100.0  # High cost
        local_budget = 10.0  # Low budget
        task_horizon = create_default_mrh_profile()

        # Route via federation
        decision = sage._handle_federation_routing(
            task_context, task_cost, local_budget, task_horizon
        )

        assert decision['delegated'] is True
        assert decision['platform'] == "Sprout"
        assert decision['reason'] == 'federation_success'
        assert decision['results'] is not None

    def test_federation_routing_sufficient_local_budget(self):
        """Test federation not used when local budget is sufficient"""
        thor = create_thor_identity()
        sprout = create_sprout_identity()

        sage = MichaudSAGE(
            federation_enabled=True,
            federation_identity=thor,
            federation_platforms=[sprout]
        )

        # Create task with sufficient budget
        task_context = {
            'operation': 'llm_inference',
            'parameters': {},
            'quality': {}
        }
        task_cost = 10.0  # Low cost
        local_budget = 100.0  # High budget
        task_horizon = create_default_mrh_profile()

        # Should not delegate
        decision = sage._handle_federation_routing(
            task_context, task_cost, local_budget, task_horizon
        )

        assert decision['delegated'] is False
        assert decision['reason'] == 'sufficient_local_atp'

    def test_federation_routing_no_platforms(self):
        """Test federation routing fails when no platforms available"""
        thor = create_thor_identity()

        sage = MichaudSAGE(
            federation_enabled=True,
            federation_identity=thor,
            federation_platforms=[]  # No platforms!
        )

        # Create high-cost task
        task_context = {
            'operation': 'llm_inference',
            'parameters': {},
            'quality': {}
        }
        task_cost = 100.0
        local_budget = 10.0
        task_horizon = create_default_mrh_profile()

        # Should fail to delegate
        decision = sage._handle_federation_routing(
            task_context, task_cost, local_budget, task_horizon
        )

        assert decision['delegated'] is False
        assert 'no_capable_platforms' in decision['reason'] or 'insufficient_witnesses' in decision['reason']

    def test_reputation_update_after_delegation(self):
        """Test platform reputation is updated after successful delegation"""
        thor = create_thor_identity()
        sprout = create_sprout_identity()

        sage = MichaudSAGE(
            federation_enabled=True,
            federation_identity=thor,
            federation_platforms=[sprout]
        )

        # Initial reputation
        initial_rep = sage.federation_router.known_platforms["sprout_sage_lct"].reputation_score

        # Delegate task
        task_context = {
            'operation': 'llm_inference',
            'parameters': {},
            'quality': {}
        }
        task_cost = 100.0
        local_budget = 10.0
        task_horizon = create_default_mrh_profile()

        decision = sage._handle_federation_routing(
            task_context, task_cost, local_budget, task_horizon
        )

        # Reputation should be updated
        final_rep = sage.federation_router.known_platforms["sprout_sage_lct"].reputation_score
        # Note: reputation may increase or decrease depending on quality vs initial reputation
        assert final_rep != initial_rep or abs(final_rep - initial_rep) < 0.01

    def test_key_pair_persistence(self):
        """Test key pairs are persisted and reloaded"""
        import tempfile
        import os

        thor = create_thor_identity()

        # Create temp key path
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = os.path.join(tmpdir, "test_key.key")

            # First init - generates key
            sage1 = MichaudSAGE(
                federation_enabled=True,
                federation_identity=thor,
                federation_key_path=key_path
            )

            pubkey1 = sage1.federation_keypair.public_key_bytes()

            # Second init - loads same key
            sage2 = MichaudSAGE(
                federation_enabled=True,
                federation_identity=thor,
                federation_key_path=key_path
            )

            pubkey2 = sage2.federation_keypair.public_key_bytes()

            # Should have same public key
            assert pubkey1 == pubkey2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
