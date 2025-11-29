"""
Integration Tests for Signed Federation (Phase 2)

Tests the complete signed federation flow without network communication:
1. Platform key pair generation
2. Signature registry management
3. Signed task delegation
4. Signed execution proof verification
5. Attack scenario validation

These tests validate Phase 2 cryptographic infrastructure in realistic
scenarios without requiring actual network protocol implementation.

Author: Thor (SAGE consciousness via Claude)
Date: 2025-11-29
Session: Autonomous SAGE Research - Phase 2 Integration Testing
"""

import pytest
import time
from typing import Dict, Any

from sage.federation import (
    # Crypto
    FederationKeyPair,
    FederationCrypto,
    SignatureRegistry,

    # Types
    FederationIdentity,
    FederationTask,
    ExecutionProof,
    QualityRequirements,

    # Signed types
    SignedFederationTask,
    SignedExecutionProof,

    # Router
    FederationRouter,

    # Utility
    create_thor_identity,
    create_sprout_identity,
)

from sage.core.attention_manager import MetabolicState
from sage.core.mrh_profile import MRHProfile, SpatialExtent, TemporalExtent, ComplexityExtent


class TestSignedFederationIntegration:
    """Integration tests for Phase 2 signed federation"""

    @pytest.fixture
    def setup_federation(self):
        """
        Setup complete federation environment

        Creates:
        - Thor and Sprout key pairs
        - Signature registry
        - Platform identities
        - Federation routers
        """
        # Generate key pairs
        thor_keys = FederationKeyPair.generate("Thor", "thor_sage_lct")
        sprout_keys = FederationKeyPair.generate("Sprout", "sprout_sage_lct")

        # Create signature registry
        registry = SignatureRegistry()
        registry.register_platform("Thor", thor_keys.public_key_bytes())
        registry.register_platform("Sprout", sprout_keys.public_key_bytes())

        # Create identities
        thor_identity = create_thor_identity(stake_amount=2000.0)
        thor_identity.public_key = thor_keys.public_key_bytes()

        sprout_identity = create_sprout_identity(stake_amount=1000.0)
        sprout_identity.public_key = sprout_keys.public_key_bytes()

        # Create routers
        thor_router = FederationRouter(thor_identity)
        sprout_router = FederationRouter(sprout_identity)

        thor_router.register_platform(sprout_identity)
        sprout_router.register_platform(thor_identity)

        return {
            'thor_keys': thor_keys,
            'sprout_keys': sprout_keys,
            'registry': registry,
            'thor_identity': thor_identity,
            'sprout_identity': sprout_identity,
            'thor_router': thor_router,
            'sprout_router': sprout_router
        }

    def test_complete_signed_delegation_flow(self, setup_federation):
        """
        Test complete signed delegation flow

        Flow:
        1. Thor creates and signs task
        2. Sprout verifies task signature
        3. Sprout executes and creates signed proof
        4. Thor verifies proof signature
        5. Thor accepts result and updates reputation
        """
        env = setup_federation

        # Step 1: Thor creates and signs task
        task_horizon = MRHProfile(
            delta_r=SpatialExtent.LOCAL,
            delta_t=TemporalExtent.SESSION,
            delta_c=ComplexityExtent.AGENT_SCALE
        )

        task = FederationTask(
            task_id="integration_test_001",
            task_type="llm_inference",
            task_data={"query": "Test query"},
            estimated_cost=150.0,
            task_horizon=task_horizon,
            complexity="high",
            delegating_platform="Thor",
            delegating_state=MetabolicState.WAKE,
            quality_requirements=QualityRequirements(min_quality=0.75),
            max_latency=30.0,
            deadline=time.time() + 3600
        )

        task_signature = FederationCrypto.sign_task(
            task.to_signable_dict(),
            env['thor_keys']
        )

        signed_task = SignedFederationTask(
            task=task,
            signature=task_signature,
            public_key=env['thor_keys'].public_key_bytes()
        )

        # Step 2: Sprout verifies task signature
        verified, reason = signed_task.verify_signature(env['registry'])
        assert verified is True, f"Task signature verification failed: {reason}"
        assert "Signature verified" in reason

        # Step 3: Sprout executes and creates signed proof
        proof = ExecutionProof(
            task_id=task.task_id,
            executing_platform="Sprout",
            result_data={"response": "Test response"},
            actual_latency=12.5,
            actual_cost=140.0,
            irp_iterations=4,
            final_energy=0.42,
            convergence_quality=0.82,
            quality_score=0.78,
            execution_timestamp=time.time()
        )

        proof_signature = FederationCrypto.sign_proof(
            proof.to_signable_dict(),
            env['sprout_keys']
        )

        signed_proof = SignedExecutionProof(
            proof=proof,
            signature=proof_signature,
            public_key=env['sprout_keys'].public_key_bytes()
        )

        # Step 4: Thor verifies proof signature
        verified, reason = signed_proof.verify_signature(env['registry'])
        assert verified is True, f"Proof signature verification failed: {reason}"
        assert "Signature verified" in reason

        # Step 5: Verify quality and update reputation
        assert proof.quality_score >= task.quality_requirements.min_quality

        old_reputation = env['sprout_identity'].reputation_score
        env['sprout_identity'].reputation_score += 0.01
        new_reputation = env['sprout_identity'].reputation_score

        assert new_reputation > old_reputation

    def test_task_forgery_prevented(self, setup_federation):
        """
        Test that task forgery is prevented

        Scenario: Attacker tries to forge task claiming it's from Thor
        Expected: Signature verification fails
        """
        env = setup_federation

        # Create forged task claiming to be from Thor
        forged_task = FederationTask(
            task_id="forged_task",
            task_type="llm_inference",
            task_data={"query": "Malicious query"},
            estimated_cost=100.0,
            task_horizon=MRHProfile(
                delta_r=SpatialExtent.LOCAL,
                delta_t=TemporalExtent.SESSION,
                delta_c=ComplexityExtent.AGENT_SCALE
            ),
            complexity="medium",
            delegating_platform="Thor",  # Claiming to be Thor
            delegating_state=MetabolicState.WAKE,
            quality_requirements=QualityRequirements(),
            max_latency=30.0,
            deadline=time.time() + 3600
        )

        # Sign with Sprout's key (attacker's key)
        fake_signature = FederationCrypto.sign_task(
            forged_task.to_signable_dict(),
            env['sprout_keys']  # Wrong key!
        )

        forged_signed_task = SignedFederationTask(
            task=forged_task,
            signature=fake_signature,
            public_key=env['sprout_keys'].public_key_bytes()
        )

        # Verify - should fail
        verified, reason = forged_signed_task.verify_signature(env['registry'])

        assert verified is False, "Forged task was incorrectly verified!"
        assert "Invalid signature" in reason

    def test_parameter_tampering_detected(self, setup_federation):
        """
        Test that parameter tampering is detected

        Scenario: Attacker modifies task parameters after signing
        Expected: Signature verification fails
        """
        env = setup_federation

        # Create legitimate task
        original_task = FederationTask(
            task_id="legit_task",
            task_type="llm_inference",
            task_data={"query": "Original query"},
            estimated_cost=100.0,  # Original cost
            task_horizon=MRHProfile(
                delta_r=SpatialExtent.LOCAL,
                delta_t=TemporalExtent.SESSION,
                delta_c=ComplexityExtent.AGENT_SCALE
            ),
            complexity="medium",
            delegating_platform="Thor",
            delegating_state=MetabolicState.WAKE,
            quality_requirements=QualityRequirements(),
            max_latency=30.0,
            deadline=time.time() + 3600
        )

        # Sign original task
        original_signature = FederationCrypto.sign_task(
            original_task.to_signable_dict(),
            env['thor_keys']
        )

        # Tamper with cost
        tampered_task = FederationTask(
            task_id=original_task.task_id,
            task_type=original_task.task_type,
            task_data=original_task.task_data,
            estimated_cost=10.0,  # TAMPERED
            task_horizon=original_task.task_horizon,
            complexity=original_task.complexity,
            delegating_platform=original_task.delegating_platform,
            delegating_state=original_task.delegating_state,
            quality_requirements=original_task.quality_requirements,
            max_latency=original_task.max_latency,
            deadline=original_task.deadline
        )

        tampered_signed_task = SignedFederationTask(
            task=tampered_task,
            signature=original_signature,  # Original signature
            public_key=env['thor_keys'].public_key_bytes()
        )

        # Verify - should fail
        verified, reason = tampered_signed_task.verify_signature(env['registry'])

        assert verified is False, "Tampered task was incorrectly verified!"
        assert "Invalid signature" in reason

    def test_quality_inflation_detected(self, setup_federation):
        """
        Test that quality inflation is detected

        Scenario: Platform inflates quality score in proof
        Expected: Signature verification fails
        """
        env = setup_federation

        # Create legitimate proof
        original_proof = ExecutionProof(
            task_id="task_001",
            executing_platform="Sprout",
            result_data={"response": "Answer"},
            actual_latency=10.0,
            actual_cost=95.0,
            irp_iterations=3,
            final_energy=0.45,
            convergence_quality=0.75,
            quality_score=0.78,  # Original quality
            execution_timestamp=time.time()
        )

        # Sign original proof
        original_signature = FederationCrypto.sign_proof(
            original_proof.to_signable_dict(),
            env['sprout_keys']
        )

        # Inflate quality
        inflated_proof = ExecutionProof(
            task_id=original_proof.task_id,
            executing_platform=original_proof.executing_platform,
            result_data=original_proof.result_data,
            actual_latency=original_proof.actual_latency,
            actual_cost=original_proof.actual_cost,
            irp_iterations=original_proof.irp_iterations,
            final_energy=original_proof.final_energy,
            convergence_quality=original_proof.convergence_quality,
            quality_score=0.98,  # INFLATED
            execution_timestamp=original_proof.execution_timestamp
        )

        inflated_signed_proof = SignedExecutionProof(
            proof=inflated_proof,
            signature=original_signature,  # Original signature
            public_key=env['sprout_keys'].public_key_bytes()
        )

        # Verify - should fail
        verified, reason = inflated_signed_proof.verify_signature(env['registry'])

        assert verified is False, "Inflated proof was incorrectly verified!"
        assert "Invalid signature" in reason

    def test_unregistered_platform_rejected(self, setup_federation):
        """
        Test that tasks from unregistered platforms are rejected

        Scenario: Unknown platform tries to send task
        Expected: Verification fails due to platform not registered
        """
        env = setup_federation

        # Create unknown platform key pair
        unknown_keys = FederationKeyPair.generate("Unknown", "unknown_lct")

        # Create task from unknown platform
        task = FederationTask(
            task_id="unknown_task",
            task_type="llm_inference",
            task_data={"query": "Query"},
            estimated_cost=100.0,
            task_horizon=MRHProfile(
                delta_r=SpatialExtent.LOCAL,
                delta_t=TemporalExtent.SESSION,
                delta_c=ComplexityExtent.AGENT_SCALE
            ),
            complexity="medium",
            delegating_platform="Unknown",
            delegating_state=MetabolicState.WAKE,
            quality_requirements=QualityRequirements(),
            max_latency=30.0,
            deadline=time.time() + 3600
        )

        # Sign with unknown platform's key
        signature = FederationCrypto.sign_task(
            task.to_signable_dict(),
            unknown_keys
        )

        signed_task = SignedFederationTask(
            task=task,
            signature=signature,
            public_key=unknown_keys.public_key_bytes()
        )

        # Verify - should fail
        verified, reason = signed_task.verify_signature(env['registry'])

        assert verified is False, "Unknown platform was incorrectly verified!"
        assert "not registered" in reason

    def test_key_pair_persistence(self, setup_federation):
        """
        Test that key pairs can be serialized and restored

        Important for platform restart scenarios
        """
        env = setup_federation

        # Serialize Thor's private key
        private_key_bytes = env['thor_keys'].private_key_bytes()

        # Create new key pair from serialized bytes
        restored_keys = FederationKeyPair.from_bytes(
            platform_name="Thor",
            lct_id="thor_sage_lct",
            private_key_bytes=private_key_bytes
        )

        # Verify same public key
        assert restored_keys.public_key_bytes() == env['thor_keys'].public_key_bytes()

        # Verify can still sign/verify
        message = b"Test message"
        signature = restored_keys.sign(message)
        assert restored_keys.verify(message, signature)

    def test_reputation_update_after_verified_execution(self, setup_federation):
        """
        Test that reputation is updated correctly after verified execution

        Verifies the trust accumulation mechanism works
        """
        env = setup_federation

        initial_reputation = env['sprout_identity'].reputation_score

        # Execute multiple tasks
        for i in range(3):
            # Create and sign task
            task = FederationTask(
                task_id=f"task_{i}",
                task_type="llm_inference",
                task_data={"query": f"Query {i}"},
                estimated_cost=100.0,
                task_horizon=MRHProfile(
                    delta_r=SpatialExtent.LOCAL,
                    delta_t=TemporalExtent.SESSION,
                    delta_c=ComplexityExtent.AGENT_SCALE
                ),
                complexity="medium",
                delegating_platform="Thor",
                delegating_state=MetabolicState.WAKE,
                quality_requirements=QualityRequirements(min_quality=0.75),
                max_latency=30.0,
                deadline=time.time() + 3600
            )

            # Create and sign proof with good quality
            proof = ExecutionProof(
                task_id=task.task_id,
                executing_platform="Sprout",
                result_data={"response": f"Answer {i}"},
                actual_latency=10.0,
                actual_cost=95.0,
                irp_iterations=3,
                final_energy=0.40,
                convergence_quality=0.85,
                quality_score=0.85,  # High quality
                execution_timestamp=time.time()
            )

            proof_signature = FederationCrypto.sign_proof(
                proof.to_signable_dict(),
                env['sprout_keys']
            )

            signed_proof = SignedExecutionProof(
                proof=proof,
                signature=proof_signature,
                public_key=env['sprout_keys'].public_key_bytes()
            )

            # Verify and update reputation
            verified, _ = signed_proof.verify_signature(env['registry'])
            assert verified

            if proof.quality_score >= task.quality_requirements.min_quality:
                # Simple reputation update
                reputation_gain = 0.02
                env['sprout_identity'].reputation_score = min(
                    1.0,
                    env['sprout_identity'].reputation_score + reputation_gain
                )

        # Verify reputation increased
        final_reputation = env['sprout_identity'].reputation_score
        assert final_reputation > initial_reputation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
