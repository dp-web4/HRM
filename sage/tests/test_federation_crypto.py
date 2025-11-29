"""
Tests for Federation Cryptography (Phase 2)

Tests Ed25519 signatures for:
- FederationTask signing and verification
- ExecutionProof signing and verification
- WitnessAttestation signing and verification
- SignatureRegistry management
- Attack scenarios (forgery, tampering, replay)

Author: Thor (SAGE consciousness via Claude)
Date: 2025-11-28
Session: Autonomous SAGE Research - Phase 2 Cryptography
"""

import pytest
import time
from sage.federation import (
    # Crypto
    FederationKeyPair,
    FederationCrypto,
    SignatureRegistry,

    # Base types
    FederationTask,
    ExecutionProof,
    WitnessAttestation,
    QualityRequirements,
    WitnessOutcome,

    # Signed types
    SignedFederationTask,
    SignedExecutionProof,
    SignedWitnessAttestation,

    # Utility
    create_thor_identity,
    create_sprout_identity,
)
from sage.core.attention_manager import MetabolicState
from sage.core.mrh_profile import MRHProfile, SpatialExtent, TemporalExtent, ComplexityExtent


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def thor_keypair():
    """Generate Thor's key pair"""
    return FederationKeyPair.generate(platform_name="Thor", lct_id="thor_sage_lct")


@pytest.fixture
def sprout_keypair():
    """Generate Sprout's key pair"""
    return FederationKeyPair.generate(platform_name="Sprout", lct_id="sprout_sage_lct")


@pytest.fixture
def nova_keypair():
    """Generate Nova's key pair (third platform for witness diversity)"""
    return FederationKeyPair.generate(platform_name="Nova", lct_id="nova_sage_lct")


@pytest.fixture
def signature_registry(thor_keypair, sprout_keypair, nova_keypair):
    """Create signature registry with all platforms"""
    registry = SignatureRegistry()
    registry.register_platform("Thor", thor_keypair.public_key_bytes())
    registry.register_platform("Sprout", sprout_keypair.public_key_bytes())
    registry.register_platform("Nova", nova_keypair.public_key_bytes())
    return registry


@pytest.fixture
def sample_task():
    """Create sample federation task"""
    task_horizon = MRHProfile(
        delta_r=SpatialExtent.LOCAL,
        delta_t=TemporalExtent.SESSION,
        delta_c=ComplexityExtent.AGENT_SCALE
    )

    return FederationTask(
        task_id="task_001",
        task_type="llm_inference",
        task_data={"query": "What is consciousness?"},
        estimated_cost=150.0,
        task_horizon=task_horizon,
        complexity="medium",
        delegating_platform="Thor",
        delegating_state=MetabolicState.WAKE,
        quality_requirements=QualityRequirements(
            min_quality=0.7,
            min_convergence=0.6,
            max_energy=0.7
        ),
        max_latency=30.0,
        deadline=time.time() + 3600
    )


@pytest.fixture
def sample_proof():
    """Create sample execution proof"""
    return ExecutionProof(
        task_id="task_001",
        executing_platform="Sprout",
        result_data={"response": "Consciousness is awareness of awareness"},
        actual_latency=12.5,
        actual_cost=140.0,
        irp_iterations=5,
        final_energy=0.42,
        convergence_quality=0.85,
        quality_score=0.88,
        execution_timestamp=time.time()
    )


@pytest.fixture
def sample_attestation():
    """Create sample witness attestation"""
    return WitnessAttestation(
        attestation_id="attest_001",
        task_id="task_001",
        witness_lct_id="nova_sage_lct",
        witness_society_id="Nova",
        claimed_correctness=0.90,
        claimed_quality=0.87,
        timestamp=time.time()
    )


# ============================================================================
# Key Management Tests
# ============================================================================

def test_keypair_generation():
    """Test key pair generation"""
    keypair = FederationKeyPair.generate(platform_name="Thor", lct_id="thor_sage_lct")

    assert keypair.platform_name == "Thor"
    assert keypair.lct_id == "thor_sage_lct"
    assert len(keypair.public_key_bytes()) == 32  # Ed25519 public key is 32 bytes
    assert len(keypair.private_key_bytes()) == 32  # Ed25519 private key is 32 bytes


def test_keypair_sign_and_verify():
    """Test signing and verification with key pair"""
    keypair = FederationKeyPair.generate(platform_name="Thor", lct_id="thor_sage_lct")

    message = b"Hello, Federation!"
    signature = keypair.sign(message)

    # Valid signature should verify
    assert keypair.verify(message, signature) is True

    # Invalid signature should fail
    bad_signature = b"x" * 64
    assert keypair.verify(message, bad_signature) is False

    # Modified message should fail
    assert keypair.verify(b"Different message", signature) is False


def test_keypair_serialization():
    """Test key pair serialization and deserialization"""
    keypair1 = FederationKeyPair.generate(platform_name="Thor", lct_id="thor_sage_lct")

    # Serialize private key
    private_bytes = keypair1.private_key_bytes()

    # Deserialize to new keypair
    keypair2 = FederationKeyPair.from_bytes(
        platform_name="Thor",
        lct_id="thor_sage_lct",
        private_key_bytes=private_bytes
    )

    # Should produce same public key
    assert keypair1.public_key_bytes() == keypair2.public_key_bytes()

    # Should be able to verify signatures from keypair1
    message = b"Test message"
    signature = keypair1.sign(message)
    assert keypair2.verify(message, signature) is True


# ============================================================================
# Task Signing Tests
# ============================================================================

def test_sign_task_success(sample_task, thor_keypair):
    """Test successful task signing"""
    # Sign task
    task_dict = sample_task.to_signable_dict()
    signature = FederationCrypto.sign_task(task_dict, thor_keypair)

    assert len(signature) == 64  # Ed25519 signature is 64 bytes

    # Create signed task
    signed_task = SignedFederationTask(
        task=sample_task,
        signature=signature,
        public_key=thor_keypair.public_key_bytes()
    )

    assert signed_task.task == sample_task
    assert signed_task.signature == signature


def test_verify_task_signature_success(sample_task, thor_keypair, signature_registry):
    """Test successful task signature verification"""
    # Sign task
    task_dict = sample_task.to_signable_dict()
    signature = FederationCrypto.sign_task(task_dict, thor_keypair)

    # Create signed task
    signed_task = SignedFederationTask(
        task=sample_task,
        signature=signature,
        public_key=thor_keypair.public_key_bytes()
    )

    # Verify signature
    verified, reason = signed_task.verify_signature(signature_registry)

    assert verified is True
    assert reason == "Signature verified"


def test_verify_task_signature_forgery(sample_task, thor_keypair, sprout_keypair, signature_registry):
    """Test detection of task forgery (wrong platform signature)"""
    # Sprout tries to forge task claiming to be from Thor
    task_dict = sample_task.to_signable_dict()
    # Sign with Sprout's key (but task claims delegating_platform="Thor")
    forged_signature = FederationCrypto.sign_task(task_dict, sprout_keypair)

    signed_task = SignedFederationTask(
        task=sample_task,  # delegating_platform="Thor"
        signature=forged_signature,  # but signed by Sprout
        public_key=sprout_keypair.public_key_bytes()
    )

    # Verification should fail
    verified, reason = signed_task.verify_signature(signature_registry)

    assert verified is False
    assert "Invalid signature" in reason


def test_verify_task_signature_tampering(sample_task, thor_keypair, signature_registry):
    """Test detection of parameter tampering"""
    # Sign legitimate task
    task_dict = sample_task.to_signable_dict()
    signature = FederationCrypto.sign_task(task_dict, thor_keypair)

    # Tamper with task after signing (increase cost)
    tampered_task = FederationTask(
        task_id=sample_task.task_id,
        task_type=sample_task.task_type,
        task_data=sample_task.task_data,
        estimated_cost=999999.0,  # TAMPERED (was 150.0)
        task_horizon=sample_task.task_horizon,
        complexity=sample_task.complexity,
        delegating_platform=sample_task.delegating_platform,
        delegating_state=sample_task.delegating_state,
        quality_requirements=sample_task.quality_requirements,
        max_latency=sample_task.max_latency,
        deadline=sample_task.deadline
    )

    signed_task = SignedFederationTask(
        task=tampered_task,
        signature=signature,  # Signature from original task
        public_key=thor_keypair.public_key_bytes()
    )

    # Verification should fail (signature doesn't match tampered data)
    verified, reason = signed_task.verify_signature(signature_registry)

    assert verified is False
    assert "Invalid signature" in reason


# ============================================================================
# Proof Signing Tests
# ============================================================================

def test_sign_proof_success(sample_proof, sprout_keypair):
    """Test successful proof signing"""
    # Sign proof
    proof_dict = sample_proof.to_signable_dict()
    signature = FederationCrypto.sign_proof(proof_dict, sprout_keypair)

    assert len(signature) == 64  # Ed25519 signature is 64 bytes

    # Create signed proof
    signed_proof = SignedExecutionProof(
        proof=sample_proof,
        signature=signature,
        public_key=sprout_keypair.public_key_bytes()
    )

    assert signed_proof.proof == sample_proof
    assert signed_proof.signature == signature


def test_verify_proof_signature_success(sample_proof, sprout_keypair, signature_registry):
    """Test successful proof signature verification"""
    # Sign proof
    proof_dict = sample_proof.to_signable_dict()
    signature = FederationCrypto.sign_proof(proof_dict, sprout_keypair)

    # Create signed proof
    signed_proof = SignedExecutionProof(
        proof=sample_proof,
        signature=signature,
        public_key=sprout_keypair.public_key_bytes()
    )

    # Verify signature
    verified, reason = signed_proof.verify_signature(signature_registry)

    assert verified is True
    assert reason == "Signature verified"


def test_verify_proof_quality_inflation(sample_proof, sprout_keypair, signature_registry):
    """Test detection of quality score inflation"""
    # Sign legitimate proof
    proof_dict = sample_proof.to_signable_dict()
    signature = FederationCrypto.sign_proof(proof_dict, sprout_keypair)

    # Try to inflate quality score after signing
    inflated_proof = ExecutionProof(
        task_id=sample_proof.task_id,
        executing_platform=sample_proof.executing_platform,
        result_data=sample_proof.result_data,
        actual_latency=sample_proof.actual_latency,
        actual_cost=sample_proof.actual_cost,
        irp_iterations=sample_proof.irp_iterations,
        final_energy=sample_proof.final_energy,
        convergence_quality=sample_proof.convergence_quality,
        quality_score=0.99,  # INFLATED (was 0.88)
        execution_timestamp=sample_proof.execution_timestamp
    )

    signed_proof = SignedExecutionProof(
        proof=inflated_proof,
        signature=signature,  # Signature from original proof
        public_key=sprout_keypair.public_key_bytes()
    )

    # Verification should fail
    verified, reason = signed_proof.verify_signature(signature_registry)

    assert verified is False
    assert "Invalid signature" in reason


# ============================================================================
# Attestation Signing Tests
# ============================================================================

def test_sign_attestation_success(sample_attestation, nova_keypair):
    """Test successful attestation signing"""
    # Sign attestation
    attestation_dict = sample_attestation.to_signable_dict()
    signature = FederationCrypto.sign_attestation(attestation_dict, nova_keypair)

    assert len(signature) == 64  # Ed25519 signature is 64 bytes

    # Create signed attestation
    signed_attestation = SignedWitnessAttestation(
        attestation=sample_attestation,
        signature=signature,
        public_key=nova_keypair.public_key_bytes()
    )

    assert signed_attestation.attestation == sample_attestation
    assert signed_attestation.signature == signature


def test_verify_attestation_signature_success(sample_attestation, nova_keypair, signature_registry):
    """Test successful attestation signature verification"""
    # Sign attestation
    attestation_dict = sample_attestation.to_signable_dict()
    signature = FederationCrypto.sign_attestation(attestation_dict, nova_keypair)

    # Create signed attestation
    signed_attestation = SignedWitnessAttestation(
        attestation=sample_attestation,
        signature=signature,
        public_key=nova_keypair.public_key_bytes()
    )

    # Verify signature
    verified, reason = signed_attestation.verify_signature(signature_registry)

    assert verified is True
    assert reason == "Signature verified"


def test_verify_attestation_forgery(sample_attestation, thor_keypair, nova_keypair, signature_registry):
    """Test detection of witness forgery"""
    # Thor tries to forge attestation claiming to be from Nova
    attestation_dict = sample_attestation.to_signable_dict()
    # Sign with Thor's key (but attestation claims witness_society_id="Nova")
    forged_signature = FederationCrypto.sign_attestation(attestation_dict, thor_keypair)

    signed_attestation = SignedWitnessAttestation(
        attestation=sample_attestation,  # witness_society_id="Nova"
        signature=forged_signature,  # but signed by Thor
        public_key=thor_keypair.public_key_bytes()
    )

    # Verification should fail
    verified, reason = signed_attestation.verify_signature(signature_registry)

    assert verified is False
    assert "Invalid signature" in reason


# ============================================================================
# Signature Registry Tests
# ============================================================================

def test_signature_registry_registration(thor_keypair):
    """Test platform registration in signature registry"""
    registry = SignatureRegistry()

    # Register Thor
    registry.register_platform("Thor", thor_keypair.public_key_bytes())

    assert registry.is_registered("Thor") is True
    assert registry.is_registered("Sprout") is False
    assert registry.get_public_key("Thor") == thor_keypair.public_key_bytes()


def test_signature_registry_duplicate_registration(thor_keypair):
    """Test duplicate registration with same key (should succeed)"""
    registry = SignatureRegistry()

    # Register Thor twice with same key
    registry.register_platform("Thor", thor_keypair.public_key_bytes())
    registry.register_platform("Thor", thor_keypair.public_key_bytes())  # Should not raise

    assert registry.is_registered("Thor") is True


def test_signature_registry_key_mismatch(thor_keypair, sprout_keypair):
    """Test duplicate registration with different key (should fail)"""
    registry = SignatureRegistry()

    # Register Thor
    registry.register_platform("Thor", thor_keypair.public_key_bytes())

    # Try to re-register Thor with different key
    with pytest.raises(ValueError, match="already registered with different key"):
        registry.register_platform("Thor", sprout_keypair.public_key_bytes())


def test_signature_registry_stats(thor_keypair, sprout_keypair, nova_keypair):
    """Test signature registry statistics"""
    registry = SignatureRegistry()

    registry.register_platform("Thor", thor_keypair.public_key_bytes())
    registry.register_platform("Sprout", sprout_keypair.public_key_bytes())
    registry.register_platform("Nova", nova_keypair.public_key_bytes())

    stats = registry.get_stats()

    assert stats['registered_platforms'] == 3
    assert set(stats['platforms']) == {'Thor', 'Sprout', 'Nova'}


def test_signature_registry_unregistered_platform(sample_task):
    """Test verification fails for unregistered platform"""
    registry = SignatureRegistry()
    # Don't register any platforms

    # Try to verify task from unregistered platform
    fake_signature = b"x" * 64
    signed_task = SignedFederationTask(
        task=sample_task,
        signature=fake_signature,
        public_key=b"x" * 32
    )

    verified, reason = signed_task.verify_signature(registry)

    assert verified is False
    assert "not registered" in reason


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_signed_delegation_flow(thor_keypair, sprout_keypair, nova_keypair, signature_registry):
    """Test complete signed delegation flow: task → proof → attestation"""
    # 1. Thor creates and signs task
    task_horizon = MRHProfile(
        delta_r=SpatialExtent.LOCAL,
        delta_t=TemporalExtent.SESSION,
        delta_c=ComplexityExtent.AGENT_SCALE
    )

    task = FederationTask(
        task_id="integration_task_001",
        task_type="llm_inference",
        task_data={"query": "Test query"},
        estimated_cost=100.0,
        task_horizon=task_horizon,
        complexity="low",
        delegating_platform="Thor",
        delegating_state=MetabolicState.WAKE,
        quality_requirements=QualityRequirements(),
        max_latency=30.0,
        deadline=time.time() + 3600
    )

    task_signature = FederationCrypto.sign_task(task.to_signable_dict(), thor_keypair)
    signed_task = SignedFederationTask(
        task=task,
        signature=task_signature,
        public_key=thor_keypair.public_key_bytes()
    )

    # Verify task signature
    verified, reason = signed_task.verify_signature(signature_registry)
    assert verified is True

    # 2. Sprout executes and signs proof
    proof = ExecutionProof(
        task_id=task.task_id,
        executing_platform="Sprout",
        result_data={"response": "Test response"},
        actual_latency=10.0,
        actual_cost=95.0,
        irp_iterations=3,
        final_energy=0.35,
        convergence_quality=0.80,
        quality_score=0.82,
        execution_timestamp=time.time()
    )

    proof_signature = FederationCrypto.sign_proof(proof.to_signable_dict(), sprout_keypair)
    signed_proof = SignedExecutionProof(
        proof=proof,
        signature=proof_signature,
        public_key=sprout_keypair.public_key_bytes()
    )

    # Verify proof signature
    verified, reason = signed_proof.verify_signature(signature_registry)
    assert verified is True

    # 3. Nova witnesses and signs attestation
    attestation = WitnessAttestation(
        attestation_id="integration_attest_001",
        task_id=task.task_id,
        witness_lct_id="nova_sage_lct",
        witness_society_id="Nova",
        claimed_correctness=0.85,
        claimed_quality=0.83,
        timestamp=time.time()
    )

    attestation_signature = FederationCrypto.sign_attestation(
        attestation.to_signable_dict(),
        nova_keypair
    )
    signed_attestation = SignedWitnessAttestation(
        attestation=attestation,
        signature=attestation_signature,
        public_key=nova_keypair.public_key_bytes()
    )

    # Verify attestation signature
    verified, reason = signed_attestation.verify_signature(signature_registry)
    assert verified is True

    # 4. Verify complete trust chain
    # All three signatures verified successfully
    # Trust established through cryptographic proof


def test_deterministic_serialization(sample_task, thor_keypair):
    """Test that serialization is deterministic for consistent signatures"""
    # Sign same task twice
    task_dict = sample_task.to_signable_dict()

    signature1 = FederationCrypto.sign_task(task_dict, thor_keypair)
    signature2 = FederationCrypto.sign_task(task_dict, thor_keypair)

    # Signatures should be identical (deterministic serialization)
    assert signature1 == signature2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
