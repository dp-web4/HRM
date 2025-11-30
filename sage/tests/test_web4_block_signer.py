"""
Tests for SAGE-backed Web4 block signing.

Tests the integration between SAGE federation Ed25519 cryptography and
Web4 game engine block signing protocol.

Author: Thor SAGE (autonomous research session)
Date: 2025-11-29
"""

import pytest
import sys
from pathlib import Path
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.federation.web4_block_signer import (
    SageBlockSigner,
    SageBlockVerifier,
    create_sage_block_signer_from_identity
)
from sage.federation.federation_crypto import (
    FederationKeyPair,
    SignatureRegistry
)


class TestSageBlockSigner:
    """Test suite for SAGE-backed Web4 block signing"""

    def test_sign_and_verify_basic(self):
        """Test basic block signing and verification"""
        # Create keypair and signer
        keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")
        signer = SageBlockSigner(keypair)

        # Create block header
        header = {
            "index": 1,
            "society_lct": "thor_sage_lct",
            "previous_hash": "0" * 64,
            "timestamp": 1732900000.0
        }

        # Sign block
        signature = signer.sign_block_header(header)

        # Verify signature
        verifier = SageBlockVerifier()
        is_valid = verifier.verify_block_signature(
            header,
            signature,
            public_key=keypair.public_key_bytes()
        )

        assert is_valid
        assert len(signature) == 64  # Ed25519 signature length

    def test_signature_deterministic(self):
        """Test that signing the same header produces the same signature"""
        keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")
        signer = SageBlockSigner(keypair)

        header = {
            "index": 1,
            "society_lct": "thor_sage_lct",
            "previous_hash": "0" * 64,
            "timestamp": 1732900000.0
        }

        # Sign twice
        signature1 = signer.sign_block_header(header)
        signature2 = signer.sign_block_header(header)

        # Should be identical (Ed25519 is deterministic)
        assert signature1 == signature2

    def test_tampering_detection(self):
        """Test that signature verification detects tampered headers"""
        keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")
        signer = SageBlockSigner(keypair)
        verifier = SageBlockVerifier()

        # Sign original header
        original_header = {
            "index": 1,
            "society_lct": "thor_sage_lct",
            "previous_hash": "0" * 64,
            "timestamp": 1732900000.0
        }
        signature = signer.sign_block_header(original_header)

        # Verify original passes
        assert verifier.verify_block_signature(
            original_header,
            signature,
            public_key=keypair.public_key_bytes()
        )

        # Tamper with index
        tampered_header = original_header.copy()
        tampered_header["index"] = 2

        # Verification should fail
        assert not verifier.verify_block_signature(
            tampered_header,
            signature,
            public_key=keypair.public_key_bytes()
        )

        # Tamper with timestamp
        tampered_header = original_header.copy()
        tampered_header["timestamp"] = 1732900001.0

        # Verification should fail
        assert not verifier.verify_block_signature(
            tampered_header,
            signature,
            public_key=keypair.public_key_bytes()
        )

    def test_wrong_public_key(self):
        """Test that wrong public key fails verification"""
        # Create two different keypairs
        keypair1 = FederationKeyPair.generate("Thor", "thor_sage_lct")
        keypair2 = FederationKeyPair.generate("Sprout", "sprout_sage_lct")

        signer = SageBlockSigner(keypair1)
        verifier = SageBlockVerifier()

        header = {
            "index": 1,
            "society_lct": "thor_sage_lct",
            "previous_hash": "0" * 64,
            "timestamp": 1732900000.0
        }

        # Sign with keypair1
        signature = signer.sign_block_header(header)

        # Verify with keypair1's public key - should pass
        assert verifier.verify_block_signature(
            header,
            signature,
            public_key=keypair1.public_key_bytes()
        )

        # Verify with keypair2's public key - should fail
        assert not verifier.verify_block_signature(
            header,
            signature,
            public_key=keypair2.public_key_bytes()
        )

    def test_registry_verification(self):
        """Test platform-based verification using SignatureRegistry"""
        # Create keypair and register
        keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")
        registry = SignatureRegistry()
        registry.register_platform("Thor", keypair.public_key_bytes())

        # Sign block
        signer = SageBlockSigner(keypair)
        header = {
            "index": 1,
            "society_lct": "thor_sage_lct",
            "previous_hash": "0" * 64,
            "timestamp": 1732900000.0
        }
        signature = signer.sign_block_header(header)

        # Verify using registry
        verifier = SageBlockVerifier(registry=registry)
        is_valid = verifier.verify_block_signature_by_platform(
            header,
            signature,
            platform_name="Thor"
        )

        assert is_valid

    def test_registry_verification_unregistered_platform(self):
        """Test that unregistered platform raises error"""
        keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")
        signer = SageBlockSigner(keypair)

        header = {
            "index": 1,
            "society_lct": "thor_sage_lct",
            "previous_hash": "0" * 64,
            "timestamp": 1732900000.0
        }
        signature = signer.sign_block_header(header)

        # Empty registry
        registry = SignatureRegistry()
        verifier = SageBlockVerifier(registry=registry)

        # Should raise error for unregistered platform
        with pytest.raises(ValueError, match="not registered"):
            verifier.verify_block_signature_by_platform(
                header,
                signature,
                platform_name="Thor"
            )

    def test_registry_verification_no_registry(self):
        """Test that platform verification requires registry"""
        keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")
        signer = SageBlockSigner(keypair)

        header = {
            "index": 1,
            "society_lct": "thor_sage_lct",
            "previous_hash": "0" * 64,
            "timestamp": 1732900000.0
        }
        signature = signer.sign_block_header(header)

        # Verifier without registry
        verifier = SageBlockVerifier()

        # Should raise error
        with pytest.raises(ValueError, match="SignatureRegistry required"):
            verifier.verify_block_signature_by_platform(
                header,
                signature,
                platform_name="Thor"
            )

    def test_key_persistence(self):
        """Test create_sage_block_signer_from_identity with key persistence"""
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = os.path.join(tmpdir, "thor_test.key")

            # Create signer (generates new key)
            signer1 = create_sage_block_signer_from_identity(
                "Thor",
                "thor_sage_lct",
                key_path=key_path
            )

            # Sign a block
            header = {
                "index": 1,
                "society_lct": "thor_sage_lct",
                "previous_hash": "0" * 64,
                "timestamp": 1732900000.0
            }
            signature1 = signer1.sign_block_header(header)

            # Create another signer (should load existing key)
            signer2 = create_sage_block_signer_from_identity(
                "Thor",
                "thor_sage_lct",
                key_path=key_path
            )

            # Sign same block
            signature2 = signer2.sign_block_header(header)

            # Signatures should match (same key loaded)
            assert signature1 == signature2

            # Public keys should match
            assert signer1.keypair.public_key_bytes() == signer2.keypair.public_key_bytes()

    def test_different_platforms_different_signatures(self):
        """Test that different platforms produce different signatures"""
        # Create two signers for different platforms
        thor_keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")
        sprout_keypair = FederationKeyPair.generate("Sprout", "sprout_sage_lct")

        thor_signer = SageBlockSigner(thor_keypair)
        sprout_signer = SageBlockSigner(sprout_keypair)

        # Same header
        header = {
            "index": 1,
            "society_lct": "test_lct",
            "previous_hash": "0" * 64,
            "timestamp": 1732900000.0
        }

        # Different signatures
        thor_signature = thor_signer.sign_block_header(header)
        sprout_signature = sprout_signer.sign_block_header(header)

        assert thor_signature != sprout_signature

        # Each signature only valid with corresponding public key
        verifier = SageBlockVerifier()

        assert verifier.verify_block_signature(
            header, thor_signature, public_key=thor_keypair.public_key_bytes()
        )
        assert not verifier.verify_block_signature(
            header, thor_signature, public_key=sprout_keypair.public_key_bytes()
        )

        assert verifier.verify_block_signature(
            header, sprout_signature, public_key=sprout_keypair.public_key_bytes()
        )
        assert not verifier.verify_block_signature(
            header, sprout_signature, public_key=thor_keypair.public_key_bytes()
        )

    def test_canonical_json_ordering(self):
        """Test that JSON field ordering doesn't affect signatures"""
        keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")
        signer = SageBlockSigner(keypair)

        # Same data, different field order
        header1 = {
            "index": 1,
            "society_lct": "thor_sage_lct",
            "previous_hash": "0" * 64,
            "timestamp": 1732900000.0
        }

        header2 = {
            "timestamp": 1732900000.0,
            "index": 1,
            "previous_hash": "0" * 64,
            "society_lct": "thor_sage_lct"
        }

        # Should produce identical signatures (canonical JSON)
        signature1 = signer.sign_block_header(header1)
        signature2 = signer.sign_block_header(header2)

        assert signature1 == signature2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
