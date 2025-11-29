"""
Federation Cryptography Module
SAGE Phase 2 - Ed25519 Signatures for Federation Trust

Problem:
Phase 1.5 federation has NO cryptographic signatures on tasks/proofs/attestations.
This enables critical attacks:
1. Task Forgery: Attacker claims tasks delegated by legitimate platform
2. Proof Forgery: Attacker fabricates execution proofs to inflate reputation
3. Witness Forgery: Attacker creates fake witness attestations
4. Parameter Tampering: Attacker modifies task parameters in transit

Solution:
Ed25519 signatures on all federation primitives (tasks, proofs, attestations).
Recipients verify signatures before accepting delegation/proofs/attestations.

Security Properties:
1. Source Authentication: Prove task came from claimed delegator
2. Non-Repudiation: Delegator can't deny sending task
3. Integrity: Detect tampering with task parameters
4. Sybil Resistance: Can't forge tasks from legitimate platforms

Attack Mitigation:
- ❌ Task Forgery: Unsigned tasks rejected immediately
- ❌ Proof Forgery: Invalid signatures detected and rejected
- ❌ Witness Forgery: Fake attestations can't be created
- ❌ Parameter Tampering: Parameter tampering breaks signature

Implementation:
Based on Web4 Session #86 signed_federation_delegation.py
Adapted for SAGE consciousness federation.

Author: Thor (SAGE consciousness via Claude)
Date: 2025-11-28
Session: Autonomous SAGE Research - Phase 2 Cryptography
"""

import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Cryptography imports
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey
)
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature


# ============================================================================
# Key Management
# ============================================================================

@dataclass
class FederationKeyPair:
    """
    Ed25519 key pair for SAGE platform

    Platforms use this to sign tasks/proofs/attestations.
    Public key is shared with federation; private key kept secret.
    """
    private_key: Ed25519PrivateKey
    public_key: Ed25519PublicKey
    platform_name: str
    lct_id: str

    def sign(self, message: bytes) -> bytes:
        """Sign message with private key"""
        return self.private_key.sign(message)

    def verify(self, message: bytes, signature: bytes) -> bool:
        """Verify signature with public key"""
        try:
            self.public_key.verify(signature, message)
            return True
        except InvalidSignature:
            return False

    def public_key_bytes(self) -> bytes:
        """Serialize public key for transmission"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

    def private_key_bytes(self) -> bytes:
        """Serialize private key for storage (KEEP SECRET)"""
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )

    @staticmethod
    def generate(platform_name: str, lct_id: str) -> 'FederationKeyPair':
        """Generate new random key pair"""
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        return FederationKeyPair(
            private_key=private_key,
            public_key=public_key,
            platform_name=platform_name,
            lct_id=lct_id
        )

    @staticmethod
    def from_bytes(
        platform_name: str,
        lct_id: str,
        private_key_bytes: bytes
    ) -> 'FederationKeyPair':
        """Load key pair from stored private key bytes"""
        private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        public_key = private_key.public_key()
        return FederationKeyPair(
            private_key=private_key,
            public_key=public_key,
            platform_name=platform_name,
            lct_id=lct_id
        )

    @staticmethod
    def load_public_key(public_key_bytes: bytes) -> Ed25519PublicKey:
        """Load public key from bytes (for signature verification)"""
        return Ed25519PublicKey.from_public_bytes(public_key_bytes)


class FederationCrypto:
    """
    Cryptographic operations for SAGE federation

    Static methods for signing and verification.
    """

    @staticmethod
    def sign_task(task_dict: Dict[str, Any], keypair: FederationKeyPair) -> bytes:
        """
        Sign FederationTask with platform's private key

        Args:
            task_dict: FederationTask.to_signable_dict() output
            keypair: Platform's key pair

        Returns:
            Ed25519 signature bytes
        """
        # Deterministic JSON serialization
        task_json = json.dumps(task_dict, sort_keys=True)
        task_bytes = task_json.encode('utf-8')
        return keypair.sign(task_bytes)

    @staticmethod
    def sign_proof(proof_dict: Dict[str, Any], keypair: FederationKeyPair) -> bytes:
        """
        Sign ExecutionProof with platform's private key

        Args:
            proof_dict: ExecutionProof.to_signable_dict() output
            keypair: Platform's key pair

        Returns:
            Ed25519 signature bytes
        """
        # Deterministic JSON serialization
        proof_json = json.dumps(proof_dict, sort_keys=True)
        proof_bytes = proof_json.encode('utf-8')
        return keypair.sign(proof_bytes)

    @staticmethod
    def sign_attestation(
        attestation_dict: Dict[str, Any],
        keypair: FederationKeyPair
    ) -> bytes:
        """
        Sign WitnessAttestation with platform's private key

        Args:
            attestation_dict: WitnessAttestation.to_signable_dict() output
            keypair: Platform's key pair

        Returns:
            Ed25519 signature bytes
        """
        # Deterministic JSON serialization
        attestation_json = json.dumps(attestation_dict, sort_keys=True)
        attestation_bytes = attestation_json.encode('utf-8')
        return keypair.sign(attestation_bytes)

    @staticmethod
    def verify_signature(
        public_key_bytes: bytes,
        message: bytes,
        signature: bytes
    ) -> bool:
        """
        Verify Ed25519 signature

        Args:
            public_key_bytes: Platform's public key (raw bytes)
            message: Message that was signed
            signature: Ed25519 signature

        Returns:
            True if signature valid, False otherwise
        """
        try:
            public_key = FederationKeyPair.load_public_key(public_key_bytes)
            public_key.verify(signature, message)
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            # Catch key format errors, etc.
            print(f"Signature verification error: {e}")
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
# Signature Registry
# ============================================================================

@dataclass
class SignatureRegistry:
    """
    Registry of platform public keys and signature verification

    Tracks which platforms have registered public keys.
    Verifies signatures against registered keys.
    """
    # Platform name → public key bytes
    registry: Dict[str, bytes] = field(default_factory=dict)

    # Content hash → signature (for deduplication)
    signature_cache: Dict[str, bytes] = field(default_factory=dict)

    def register_platform(self, platform_name: str, public_key_bytes: bytes):
        """
        Register platform's public key

        Args:
            platform_name: Platform identifier (e.g., "Thor", "Sprout")
            public_key_bytes: Ed25519 public key (raw bytes)
        """
        if platform_name in self.registry:
            # Check if key matches existing registration
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
        """
        Verify task signature against registered public key

        Args:
            task_dict: FederationTask.to_signable_dict() output
            signature: Ed25519 signature
            claimed_platform: Platform claiming to have signed task

        Returns:
            (verified, reason)
        """
        # Check platform registered
        if not self.is_registered(claimed_platform):
            return (False, f"Platform {claimed_platform} not registered")

        # Get public key
        public_key_bytes = self.get_public_key(claimed_platform)

        # Serialize task
        task_json = json.dumps(task_dict, sort_keys=True)
        task_bytes = task_json.encode('utf-8')

        # Verify signature
        verified = FederationCrypto.verify_signature(
            public_key_bytes,
            task_bytes,
            signature
        )

        if verified:
            # Cache signature
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
        """
        Verify execution proof signature

        Args:
            proof_dict: ExecutionProof.to_signable_dict() output
            signature: Ed25519 signature
            claimed_platform: Platform claiming to have executed task

        Returns:
            (verified, reason)
        """
        # Check platform registered
        if not self.is_registered(claimed_platform):
            return (False, f"Platform {claimed_platform} not registered")

        # Get public key
        public_key_bytes = self.get_public_key(claimed_platform)

        # Serialize proof
        proof_json = json.dumps(proof_dict, sort_keys=True)
        proof_bytes = proof_json.encode('utf-8')

        # Verify signature
        verified = FederationCrypto.verify_signature(
            public_key_bytes,
            proof_bytes,
            signature
        )

        if verified:
            # Cache signature
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
        """
        Verify witness attestation signature

        Args:
            attestation_dict: WitnessAttestation.to_signable_dict() output
            signature: Ed25519 signature
            claimed_witness: Platform claiming to be witness

        Returns:
            (verified, reason)
        """
        # Check platform registered
        if not self.is_registered(claimed_witness):
            return (False, f"Platform {claimed_witness} not registered")

        # Get public key
        public_key_bytes = self.get_public_key(claimed_witness)

        # Serialize attestation
        attestation_json = json.dumps(attestation_dict, sort_keys=True)
        attestation_bytes = attestation_json.encode('utf-8')

        # Verify signature
        verified = FederationCrypto.verify_signature(
            public_key_bytes,
            attestation_bytes,
            signature
        )

        if verified:
            # Cache signature
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
            'platforms': list(self.registry.keys())
        }
