"""
Federation Cryptography - Edge Optimized
SAGE Phase 2 - Ed25519 Signatures with ARM64/Edge Optimization

This module provides the same API as federation_crypto.py but uses PyNaCl
(libsodium) when available for 1.8x faster signing on ARM64 edge devices.

Performance on Jetson Orin Nano 8GB (ARM64):
- PyNaCl:       18,655 signs/sec, 7,194 verifies/sec
- Cryptography: 10,014 signs/sec, 4,468 verifies/sec
- Speedup:      1.86x signing, 1.61x verification

Cross-library compatibility: VERIFIED
- PyNaCl signatures can be verified by cryptography
- Cryptography signatures can be verified by PyNaCl
- Same Ed25519 curve and standard

Author: Sprout (SAGE consciousness via Claude)
Date: 2025-12-01
Session: 40 - Edge Optimization Research
"""

import hashlib
import json
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

# Try PyNaCl first (faster on ARM64), fall back to cryptography
try:
    from nacl.signing import SigningKey, VerifyKey
    from nacl.exceptions import BadSignatureError
    BACKEND = "pynacl"
except ImportError:
    BACKEND = "cryptography"

if BACKEND == "cryptography":
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey
    )
    from cryptography.hazmat.primitives import serialization
    from cryptography.exceptions import InvalidSignature


# ============================================================================
# Backend-Agnostic Key Management
# ============================================================================

@dataclass
class FederationKeyPair:
    """
    Ed25519 key pair for SAGE platform (backend-agnostic)

    Automatically uses PyNaCl on edge devices for better performance.
    API-compatible with cryptography-based version.
    """
    _private_key: Any  # SigningKey or Ed25519PrivateKey
    _public_key: Any   # VerifyKey or Ed25519PublicKey
    platform_name: str
    lct_id: str
    backend: str = field(default=BACKEND)

    def sign(self, message: bytes) -> bytes:
        """Sign message with private key"""
        if self.backend == "pynacl":
            signed = self._private_key.sign(message)
            return signed.signature
        else:
            return self._private_key.sign(message)

    def verify(self, message: bytes, signature: bytes) -> bool:
        """Verify signature with public key"""
        try:
            if self.backend == "pynacl":
                self._public_key.verify(message, signature)
            else:
                self._public_key.verify(signature, message)
            return True
        except (BadSignatureError if BACKEND == "pynacl" else InvalidSignature):
            return False
        except Exception:
            return False

    def public_key_bytes(self) -> bytes:
        """Serialize public key for transmission (interoperable)"""
        if self.backend == "pynacl":
            return bytes(self._public_key)
        else:
            return self._public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )

    def private_key_bytes(self) -> bytes:
        """Serialize private key for storage (KEEP SECRET)"""
        if self.backend == "pynacl":
            return bytes(self._private_key)
        else:
            return self._private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption()
            )

    @staticmethod
    def generate(platform_name: str, lct_id: str) -> 'FederationKeyPair':
        """Generate new random key pair using best available backend"""
        if BACKEND == "pynacl":
            signing_key = SigningKey.generate()
            verify_key = signing_key.verify_key
            return FederationKeyPair(
                _private_key=signing_key,
                _public_key=verify_key,
                platform_name=platform_name,
                lct_id=lct_id,
                backend="pynacl"
            )
        else:
            private_key = Ed25519PrivateKey.generate()
            public_key = private_key.public_key()
            return FederationKeyPair(
                _private_key=private_key,
                _public_key=public_key,
                platform_name=platform_name,
                lct_id=lct_id,
                backend="cryptography"
            )

    @staticmethod
    def from_bytes(
        platform_name: str,
        lct_id: str,
        private_key_bytes: bytes
    ) -> 'FederationKeyPair':
        """Load key pair from stored private key bytes"""
        if BACKEND == "pynacl":
            signing_key = SigningKey(private_key_bytes)
            verify_key = signing_key.verify_key
            return FederationKeyPair(
                _private_key=signing_key,
                _public_key=verify_key,
                platform_name=platform_name,
                lct_id=lct_id,
                backend="pynacl"
            )
        else:
            private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
            public_key = private_key.public_key()
            return FederationKeyPair(
                _private_key=private_key,
                _public_key=public_key,
                platform_name=platform_name,
                lct_id=lct_id,
                backend="cryptography"
            )

    @staticmethod
    def load_public_key(public_key_bytes: bytes):
        """Load public key from bytes (for signature verification)"""
        if BACKEND == "pynacl":
            return VerifyKey(public_key_bytes)
        else:
            return Ed25519PublicKey.from_public_bytes(public_key_bytes)


class FederationCrypto:
    """
    Cryptographic operations for SAGE federation (edge-optimized)

    Same API as original, but uses PyNaCl when available.
    """

    @staticmethod
    def sign_task(task_dict: Dict[str, Any], keypair: FederationKeyPair) -> bytes:
        """Sign FederationTask with platform's private key"""
        task_json = json.dumps(task_dict, sort_keys=True)
        task_bytes = task_json.encode('utf-8')
        return keypair.sign(task_bytes)

    @staticmethod
    def sign_proof(proof_dict: Dict[str, Any], keypair: FederationKeyPair) -> bytes:
        """Sign ExecutionProof with platform's private key"""
        proof_json = json.dumps(proof_dict, sort_keys=True)
        proof_bytes = proof_json.encode('utf-8')
        return keypair.sign(proof_bytes)

    @staticmethod
    def sign_attestation(
        attestation_dict: Dict[str, Any],
        keypair: FederationKeyPair
    ) -> bytes:
        """Sign WitnessAttestation with platform's private key"""
        attestation_json = json.dumps(attestation_dict, sort_keys=True)
        attestation_bytes = attestation_json.encode('utf-8')
        return keypair.sign(attestation_bytes)

    @staticmethod
    def verify_signature(
        public_key_bytes: bytes,
        message: bytes,
        signature: bytes
    ) -> bool:
        """Verify Ed25519 signature (backend-agnostic)"""
        try:
            if BACKEND == "pynacl":
                verify_key = VerifyKey(public_key_bytes)
                verify_key.verify(message, signature)
            else:
                public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
                public_key.verify(signature, message)
            return True
        except Exception:
            return False

    @staticmethod
    def calculate_task_hash(task_dict: Dict[str, Any]) -> str:
        """Calculate deterministic hash for FederationTask"""
        task_json = json.dumps(task_dict, sort_keys=True)
        return hashlib.sha256(task_json.encode()).hexdigest()

    @staticmethod
    def calculate_proof_hash(proof_dict: Dict[str, Any]) -> str:
        """Calculate deterministic hash for ExecutionProof"""
        proof_json = json.dumps(proof_dict, sort_keys=True)
        return hashlib.sha256(proof_json.encode()).hexdigest()

    @staticmethod
    def calculate_attestation_hash(attestation_dict: Dict[str, Any]) -> str:
        """Calculate deterministic hash for WitnessAttestation"""
        attestation_json = json.dumps(attestation_dict, sort_keys=True)
        return hashlib.sha256(attestation_json.encode()).hexdigest()


# ============================================================================
# Signature Registry (unchanged API, uses edge-optimized verification)
# ============================================================================

@dataclass
class SignatureRegistry:
    """
    Registry of platform public keys and signature verification

    Same API as original, uses edge-optimized backend.
    """
    registry: Dict[str, bytes] = field(default_factory=dict)
    signature_cache: Dict[str, bytes] = field(default_factory=dict)

    def register_platform(self, platform_name: str, public_key_bytes: bytes):
        """Register platform's public key"""
        if platform_name in self.registry:
            existing_key = self.registry[platform_name]
            if existing_key != public_key_bytes:
                raise ValueError(
                    f"Platform {platform_name} already registered with different key"
                )
        else:
            self.registry[platform_name] = public_key_bytes

    def get_public_key(self, platform_name: str) -> Optional[bytes]:
        """Get registered public key for platform"""
        return self.registry.get(platform_name)

    def is_registered(self, platform_name: str) -> bool:
        """Check if platform has registered public key"""
        return platform_name in self.registry

    def verify_task_signature(
        self,
        task_dict: Dict[str, Any],
        signature: bytes,
        claimed_platform: str
    ) -> Tuple[bool, str]:
        """Verify task signature against registered public key"""
        if not self.is_registered(claimed_platform):
            return (False, f"Platform {claimed_platform} not registered")

        public_key_bytes = self.get_public_key(claimed_platform)
        task_json = json.dumps(task_dict, sort_keys=True)
        task_bytes = task_json.encode('utf-8')

        verified = FederationCrypto.verify_signature(
            public_key_bytes, task_bytes, signature
        )

        if verified:
            content_hash = FederationCrypto.calculate_task_hash(task_dict)
            self.signature_cache[content_hash] = signature
            return (True, "Signature verified")
        else:
            return (False, "Invalid signature")

    def verify_proof_signature(
        self,
        proof_dict: Dict[str, Any],
        signature: bytes,
        claimed_platform: str
    ) -> Tuple[bool, str]:
        """Verify execution proof signature"""
        if not self.is_registered(claimed_platform):
            return (False, f"Platform {claimed_platform} not registered")

        public_key_bytes = self.get_public_key(claimed_platform)
        proof_json = json.dumps(proof_dict, sort_keys=True)
        proof_bytes = proof_json.encode('utf-8')

        verified = FederationCrypto.verify_signature(
            public_key_bytes, proof_bytes, signature
        )

        if verified:
            content_hash = FederationCrypto.calculate_proof_hash(proof_dict)
            self.signature_cache[content_hash] = signature
            return (True, "Signature verified")
        else:
            return (False, "Invalid signature")

    def verify_attestation_signature(
        self,
        attestation_dict: Dict[str, Any],
        signature: bytes,
        claimed_witness: str
    ) -> Tuple[bool, str]:
        """Verify witness attestation signature"""
        if not self.is_registered(claimed_witness):
            return (False, f"Platform {claimed_witness} not registered")

        public_key_bytes = self.get_public_key(claimed_witness)
        attestation_json = json.dumps(attestation_dict, sort_keys=True)
        attestation_bytes = attestation_json.encode('utf-8')

        verified = FederationCrypto.verify_signature(
            public_key_bytes, attestation_bytes, signature
        )

        if verified:
            content_hash = FederationCrypto.calculate_attestation_hash(attestation_dict)
            self.signature_cache[content_hash] = signature
            return (True, "Signature verified")
        else:
            return (False, "Invalid signature")

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            'registered_platforms': len(self.registry),
            'cached_signatures': len(self.signature_cache),
            'platforms': list(self.registry.keys()),
            'crypto_backend': BACKEND
        }


# ============================================================================
# Convenience Functions (same API as original)
# ============================================================================

def sign_task(task: 'FederationTask', signing_key: bytes) -> str:
    """Sign FederationTask with platform's Ed25519 private key"""
    keypair = FederationKeyPair.from_bytes("unknown", "unknown", signing_key)
    task_dict = task.to_signable_dict() if hasattr(task, 'to_signable_dict') else task.to_dict()
    signature_bytes = FederationCrypto.sign_task(task_dict, keypair)
    return signature_bytes.hex()


def sign_proof(proof: 'ExecutionProof', signing_key: bytes) -> str:
    """Sign ExecutionProof with platform's Ed25519 private key"""
    keypair = FederationKeyPair.from_bytes("unknown", "unknown", signing_key)
    proof_dict = proof.to_signable_dict() if hasattr(proof, 'to_signable_dict') else proof.to_dict()
    signature_bytes = FederationCrypto.sign_proof(proof_dict, keypair)
    return signature_bytes.hex()


def verify_task_signature(task_dict: Dict[str, Any], signature_hex: str, verify_key: bytes) -> bool:
    """Verify FederationTask signature"""
    try:
        signature_bytes = bytes.fromhex(signature_hex)
        task_json = json.dumps(task_dict, sort_keys=True)
        task_bytes = task_json.encode('utf-8')
        return FederationCrypto.verify_signature(verify_key, task_bytes, signature_bytes)
    except:
        return False


def verify_proof_signature(proof_dict: Dict[str, Any], signature_hex: str, verify_key: bytes) -> bool:
    """Verify ExecutionProof signature"""
    try:
        signature_bytes = bytes.fromhex(signature_hex)
        proof_json = json.dumps(proof_dict, sort_keys=True)
        proof_bytes = proof_json.encode('utf-8')
        return FederationCrypto.verify_signature(verify_key, proof_bytes, signature_bytes)
    except:
        return False


# ============================================================================
# Edge-Specific Utilities
# ============================================================================

def get_crypto_backend() -> str:
    """Return which crypto backend is in use"""
    return BACKEND


def get_backend_performance() -> Dict[str, Any]:
    """Get performance characteristics of current backend"""
    import time

    keypair = FederationKeyPair.generate("test", "test_lct")
    message = b"Test message for performance measurement"
    iterations = 2000

    # Signing benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        keypair.sign(message)
    sign_time = time.perf_counter() - start

    # Verification benchmark
    signature = keypair.sign(message)
    pub_bytes = keypair.public_key_bytes()

    start = time.perf_counter()
    for _ in range(iterations):
        FederationCrypto.verify_signature(pub_bytes, message, signature)
    verify_time = time.perf_counter() - start

    return {
        'backend': BACKEND,
        'signing_ops_per_sec': int(iterations / sign_time),
        'verification_ops_per_sec': int(iterations / verify_time),
        'test_iterations': iterations
    }


def benchmark_cross_compatibility() -> Dict[str, Any]:
    """Test cross-library signature compatibility"""
    results = {'backend': BACKEND, 'tests': []}

    # Generate key with current backend
    keypair = FederationKeyPair.generate("test", "test_lct")
    message = b"Cross-compatibility test message"
    signature = keypair.sign(message)
    pub_bytes = keypair.public_key_bytes()

    # Self-verify
    self_verify = FederationCrypto.verify_signature(pub_bytes, message, signature)
    results['tests'].append({
        'test': 'self_verification',
        'passed': self_verify
    })

    # Try cross-library verification if both are available
    try:
        if BACKEND == "pynacl":
            # Verify with cryptography
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            crypto_pub = Ed25519PublicKey.from_public_bytes(pub_bytes)
            crypto_pub.verify(signature, message)
            results['tests'].append({
                'test': 'pynacl_to_cryptography',
                'passed': True
            })
        else:
            # Verify with pynacl
            from nacl.signing import VerifyKey
            nacl_pub = VerifyKey(pub_bytes)
            nacl_pub.verify(message, signature)
            results['tests'].append({
                'test': 'cryptography_to_pynacl',
                'passed': True
            })
    except ImportError:
        results['tests'].append({
            'test': 'cross_library',
            'passed': None,
            'note': 'Other library not available'
        })
    except Exception as e:
        results['tests'].append({
            'test': 'cross_library',
            'passed': False,
            'error': str(e)
        })

    return results
