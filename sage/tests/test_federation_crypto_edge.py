"""
Tests for Edge-Optimized Federation Cryptography Module

Tests that the edge-optimized crypto module (using PyNaCl when available)
provides identical functionality to the standard module while offering
better performance on ARM64 edge devices.

Author: Sprout (SAGE consciousness via Claude)
Date: 2025-12-01
Session: 40 - Edge Optimization Research
"""

import pytest
import json
import time

from sage.federation.federation_crypto_edge import (
    FederationKeyPair,
    FederationCrypto,
    SignatureRegistry,
    sign_task,
    sign_proof,
    verify_task_signature,
    verify_proof_signature,
    get_crypto_backend,
    get_backend_performance,
    benchmark_cross_compatibility,
    BACKEND
)


# ============================================================================
# Backend Detection Tests
# ============================================================================

def test_backend_detection():
    """Test that backend is properly detected"""
    backend = get_crypto_backend()
    assert backend in ["pynacl", "cryptography"]
    assert backend == BACKEND


def test_backend_performance():
    """Test that performance benchmarks run successfully"""
    perf = get_backend_performance()
    assert 'backend' in perf
    assert 'signing_ops_per_sec' in perf
    assert 'verification_ops_per_sec' in perf
    assert perf['signing_ops_per_sec'] > 0
    assert perf['verification_ops_per_sec'] > 0


# ============================================================================
# Key Management Tests
# ============================================================================

def test_keypair_generation():
    """Test key pair generation"""
    keypair = FederationKeyPair.generate("TestPlatform", "test_lct_id")
    assert keypair.platform_name == "TestPlatform"
    assert keypair.lct_id == "test_lct_id"
    assert keypair.backend in ["pynacl", "cryptography"]


def test_keypair_sign_and_verify():
    """Test signing and verification with generated keypair"""
    keypair = FederationKeyPair.generate("TestPlatform", "test_lct_id")
    message = b"Test message for signing"

    signature = keypair.sign(message)
    assert len(signature) == 64  # Ed25519 signatures are 64 bytes

    verified = keypair.verify(message, signature)
    assert verified is True


def test_keypair_verify_invalid_signature():
    """Test that invalid signatures are rejected"""
    keypair = FederationKeyPair.generate("TestPlatform", "test_lct_id")
    message = b"Test message"

    # Create invalid signature (wrong bytes)
    invalid_signature = b'\x00' * 64

    verified = keypair.verify(message, invalid_signature)
    assert verified is False


def test_keypair_serialization():
    """Test key serialization and loading"""
    keypair1 = FederationKeyPair.generate("TestPlatform", "test_lct_id")

    # Serialize private key
    private_bytes = keypair1.private_key_bytes()
    assert len(private_bytes) == 32  # Ed25519 private keys are 32 bytes

    # Load from bytes
    keypair2 = FederationKeyPair.from_bytes(
        "TestPlatform", "test_lct_id", private_bytes
    )

    # Verify same public key
    assert keypair1.public_key_bytes() == keypair2.public_key_bytes()

    # Verify signatures are interchangeable
    message = b"Test message"
    sig1 = keypair1.sign(message)
    assert keypair2.verify(message, sig1) is True


# ============================================================================
# Task Signing Tests
# ============================================================================

def test_sign_task_success():
    """Test signing a federation task"""
    keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")

    task_dict = {
        'task_id': 'test_task_001',
        'task_type': 'llm_inference',
        'delegating_platform': 'thor_sage_lct',
        'estimated_cost': 50.0,
        'quality_requirements': {'min_quality': 0.7}
    }

    signature = FederationCrypto.sign_task(task_dict, keypair)
    assert len(signature) == 64


def test_verify_task_signature_success():
    """Test verifying a valid task signature"""
    keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")

    task_dict = {
        'task_id': 'test_task_001',
        'task_type': 'llm_inference',
        'delegating_platform': 'thor_sage_lct',
        'estimated_cost': 50.0
    }

    signature = FederationCrypto.sign_task(task_dict, keypair)

    # Recreate the message that was signed
    task_json = json.dumps(task_dict, sort_keys=True)
    task_bytes = task_json.encode('utf-8')

    verified = FederationCrypto.verify_signature(
        keypair.public_key_bytes(),
        task_bytes,
        signature
    )
    assert verified is True


def test_verify_task_signature_tampering():
    """Test that tampered tasks fail verification"""
    keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")

    task_dict = {
        'task_id': 'test_task_001',
        'estimated_cost': 50.0
    }

    signature = FederationCrypto.sign_task(task_dict, keypair)

    # Tamper with the task
    tampered_dict = task_dict.copy()
    tampered_dict['estimated_cost'] = 100.0  # Doubled!

    tampered_json = json.dumps(tampered_dict, sort_keys=True)
    tampered_bytes = tampered_json.encode('utf-8')

    verified = FederationCrypto.verify_signature(
        keypair.public_key_bytes(),
        tampered_bytes,
        signature
    )
    assert verified is False


# ============================================================================
# Proof Signing Tests
# ============================================================================

def test_sign_proof_success():
    """Test signing an execution proof"""
    keypair = FederationKeyPair.generate("Sprout", "sprout_sage_lct")

    proof_dict = {
        'task_id': 'test_task_001',
        'executing_platform': 'sprout_sage_lct',
        'quality_score': 0.85,
        'actual_cost': 48.5
    }

    signature = FederationCrypto.sign_proof(proof_dict, keypair)
    assert len(signature) == 64


def test_verify_proof_quality_inflation():
    """Test that quality inflation is detected"""
    keypair = FederationKeyPair.generate("Sprout", "sprout_sage_lct")

    proof_dict = {
        'task_id': 'test_task_001',
        'quality_score': 0.65  # Below threshold
    }

    signature = FederationCrypto.sign_proof(proof_dict, keypair)

    # Attacker inflates quality
    inflated_dict = proof_dict.copy()
    inflated_dict['quality_score'] = 0.95  # Inflated!

    inflated_json = json.dumps(inflated_dict, sort_keys=True)
    inflated_bytes = inflated_json.encode('utf-8')

    verified = FederationCrypto.verify_signature(
        keypair.public_key_bytes(),
        inflated_bytes,
        signature
    )
    assert verified is False


# ============================================================================
# Attestation Signing Tests
# ============================================================================

def test_sign_attestation_success():
    """Test signing a witness attestation"""
    keypair = FederationKeyPair.generate("Witness1", "witness1_lct")

    attestation_dict = {
        'task_id': 'test_task_001',
        'witness_platform': 'witness1_lct',
        'observed_quality': 0.82,
        'attestation_time': 1234567890.0
    }

    signature = FederationCrypto.sign_attestation(attestation_dict, keypair)
    assert len(signature) == 64


# ============================================================================
# Signature Registry Tests
# ============================================================================

def test_signature_registry_registration():
    """Test platform registration in signature registry"""
    registry = SignatureRegistry()
    keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")

    registry.register_platform("Thor", keypair.public_key_bytes())

    assert registry.is_registered("Thor") is True
    assert registry.is_registered("Unknown") is False
    assert registry.get_public_key("Thor") == keypair.public_key_bytes()


def test_signature_registry_verify_task():
    """Test task signature verification via registry"""
    registry = SignatureRegistry()
    keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")
    registry.register_platform("Thor", keypair.public_key_bytes())

    task_dict = {
        'task_id': 'test_task_001',
        'delegating_platform': 'thor_sage_lct',
        'estimated_cost': 50.0
    }

    signature = FederationCrypto.sign_task(task_dict, keypair)

    verified, reason = registry.verify_task_signature(
        task_dict, signature, "Thor"
    )

    assert verified is True
    assert "verified" in reason.lower()


def test_signature_registry_unregistered_platform():
    """Test that unregistered platforms are rejected"""
    registry = SignatureRegistry()

    task_dict = {'task_id': 'test'}
    signature = b'\x00' * 64

    verified, reason = registry.verify_task_signature(
        task_dict, signature, "UnknownPlatform"
    )

    assert verified is False
    assert "not registered" in reason.lower()


def test_signature_registry_stats():
    """Test registry statistics"""
    registry = SignatureRegistry()
    keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")
    registry.register_platform("Thor", keypair.public_key_bytes())

    stats = registry.get_stats()

    assert stats['registered_platforms'] == 1
    assert 'Thor' in stats['platforms']
    assert stats['crypto_backend'] in ['pynacl', 'cryptography']


# ============================================================================
# Cross-Library Compatibility Tests
# ============================================================================

def test_cross_library_compatibility():
    """Test that signatures are cross-library compatible"""
    results = benchmark_cross_compatibility()

    assert results['backend'] in ['pynacl', 'cryptography']

    for test in results['tests']:
        if test['passed'] is not None:  # Skip N/A tests
            assert test['passed'] is True, f"Test {test['test']} failed"


def test_cross_module_interoperability():
    """Test interoperability with standard crypto module"""
    try:
        from sage.federation.federation_crypto import (
            FederationKeyPair as StandardKeyPair,
            FederationCrypto as StandardCrypto
        )
    except ImportError:
        pytest.skip("Standard crypto module not available")

    # Generate key with edge module
    edge_keypair = FederationKeyPair.generate("Test", "test_lct")

    # Sign with edge module
    task_dict = {'task_id': 'cross_test', 'cost': 50.0}
    edge_sig = FederationCrypto.sign_task(task_dict, edge_keypair)

    # Verify with standard module
    task_bytes = json.dumps(task_dict, sort_keys=True).encode()
    verified = StandardCrypto.verify_signature(
        edge_keypair.public_key_bytes(),
        task_bytes,
        edge_sig
    )

    assert verified is True


# ============================================================================
# Convenience Function Tests
# ============================================================================

class MockTask:
    """Mock FederationTask for testing convenience functions"""
    def __init__(self):
        self.task_id = "mock_task_001"
        self.task_type = "llm_inference"
        self.estimated_cost = 50.0

    def to_signable_dict(self):
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'estimated_cost': self.estimated_cost
        }

    def to_dict(self):
        return self.to_signable_dict()


class MockProof:
    """Mock ExecutionProof for testing convenience functions"""
    def __init__(self):
        self.task_id = "mock_task_001"
        self.quality_score = 0.85
        self.actual_cost = 48.0

    def to_signable_dict(self):
        return {
            'task_id': self.task_id,
            'quality_score': self.quality_score,
            'actual_cost': self.actual_cost
        }

    def to_dict(self):
        return self.to_signable_dict()


def test_convenience_sign_task():
    """Test convenience sign_task function"""
    keypair = FederationKeyPair.generate("Thor", "thor_lct")
    task = MockTask()

    signature_hex = sign_task(task, keypair.private_key_bytes())

    assert len(signature_hex) == 128  # 64 bytes = 128 hex chars


def test_convenience_verify_task():
    """Test convenience verify_task_signature function"""
    keypair = FederationKeyPair.generate("Thor", "thor_lct")
    task = MockTask()

    signature_hex = sign_task(task, keypair.private_key_bytes())

    verified = verify_task_signature(
        task.to_signable_dict(),
        signature_hex,
        keypair.public_key_bytes()
    )

    assert verified is True


def test_convenience_sign_proof():
    """Test convenience sign_proof function"""
    keypair = FederationKeyPair.generate("Sprout", "sprout_lct")
    proof = MockProof()

    signature_hex = sign_proof(proof, keypair.private_key_bytes())

    assert len(signature_hex) == 128


def test_convenience_verify_proof():
    """Test convenience verify_proof_signature function"""
    keypair = FederationKeyPair.generate("Sprout", "sprout_lct")
    proof = MockProof()

    signature_hex = sign_proof(proof, keypair.private_key_bytes())

    verified = verify_proof_signature(
        proof.to_signable_dict(),
        signature_hex,
        keypair.public_key_bytes()
    )

    assert verified is True


# ============================================================================
# Hash Calculation Tests
# ============================================================================

def test_calculate_task_hash():
    """Test deterministic task hash calculation"""
    task_dict = {'task_id': 'test_001', 'cost': 50.0}

    hash1 = FederationCrypto.calculate_task_hash(task_dict)
    hash2 = FederationCrypto.calculate_task_hash(task_dict)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex


def test_calculate_proof_hash():
    """Test deterministic proof hash calculation"""
    proof_dict = {'task_id': 'test_001', 'quality': 0.85}

    hash1 = FederationCrypto.calculate_proof_hash(proof_dict)
    hash2 = FederationCrypto.calculate_proof_hash(proof_dict)

    assert hash1 == hash2


def test_calculate_attestation_hash():
    """Test deterministic attestation hash calculation"""
    att_dict = {'task_id': 'test_001', 'witness': 'platform_a'}

    hash1 = FederationCrypto.calculate_attestation_hash(att_dict)
    hash2 = FederationCrypto.calculate_attestation_hash(att_dict)

    assert hash1 == hash2


# ============================================================================
# Performance Verification Tests
# ============================================================================

@pytest.mark.parametrize("iterations", [100, 500])
def test_signing_performance(iterations):
    """Test that signing performance is reasonable"""
    keypair = FederationKeyPair.generate("PerfTest", "perf_lct")
    message = b"Performance test message"

    start = time.perf_counter()
    for _ in range(iterations):
        keypair.sign(message)
    elapsed = time.perf_counter() - start

    ops_per_sec = iterations / elapsed

    # Should achieve at least 1000 ops/sec on any reasonable hardware
    assert ops_per_sec > 1000, f"Signing too slow: {ops_per_sec:.0f} ops/sec"


@pytest.mark.parametrize("iterations", [100, 500])
def test_verification_performance(iterations):
    """Test that verification performance is reasonable"""
    keypair = FederationKeyPair.generate("PerfTest", "perf_lct")
    message = b"Performance test message"
    signature = keypair.sign(message)

    start = time.perf_counter()
    for _ in range(iterations):
        keypair.verify(message, signature)
    elapsed = time.perf_counter() - start

    ops_per_sec = iterations / elapsed

    # Should achieve at least 500 ops/sec on any reasonable hardware
    assert ops_per_sec > 500, f"Verification too slow: {ops_per_sec:.0f} ops/sec"
