"""
SAGE-backed block signer for Web4 microchains.

This module provides a concrete implementation of the Web4 BlockSigner protocol
using SAGE federation Ed25519 cryptography. It bridges Web4 game societies with
SAGE federation identities, enabling hardware-bound microchain signing.

Integration points:
- Web4 BlockSigner protocol (web4/game/engine/signing.py)
- SAGE FederationKeyPair (sage/federation/federation_crypto.py)
- Web4/SAGE alignment (web4/game/WEB4_HRM_ALIGNMENT.md)

Author: Thor SAGE (autonomous research session)
Date: 2025-11-29
Session: Web4/SAGE Integration Discovery
"""

import json
from typing import Dict, Any
from dataclasses import dataclass

from sage.federation.federation_crypto import (
    FederationKeyPair,
    FederationCrypto,
    SignatureRegistry
)


@dataclass
class SageBlockSigner:
    """SAGE-backed block signer for Web4 microchains.

    Implements the Web4 BlockSigner protocol using SAGE federation Ed25519
    cryptography. This binds Web4 society microchains to hardware via SAGE
    federation keys.

    Usage:
        keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")
        signer = SageBlockSigner(keypair)
        signature = signer.sign_block_header(header)

    Attributes:
        keypair: SAGE FederationKeyPair for signing
    """

    keypair: FederationKeyPair

    def sign_block_header(self, header: Dict[str, Any]) -> bytes:
        """Sign a Web4 block header using SAGE Ed25519 cryptography.

        Produces a deterministic signature over the canonical JSON encoding
        of the header dict. Uses the same JSON serialization as Web4 stub
        signer for consistency.

        Args:
            header: Block header dict with keys like:
                - index: int
                - society_lct: str
                - previous_hash: str | None
                - timestamp: float

        Returns:
            Ed25519 signature bytes (64 bytes)
        """
        # Canonical JSON encoding (matches Web4 stub signer)
        header_json = json.dumps(header, sort_keys=True, separators=(",", ":"))
        header_bytes = header_json.encode("utf-8")

        # Sign with SAGE Ed25519
        signature = self.keypair.sign(header_bytes)
        return signature


@dataclass
class SageBlockVerifier:
    """SAGE-backed block verifier for Web4 microchains.

    Verifies Web4 block signatures using SAGE federation Ed25519 cryptography.
    Can use either explicit public keys or the SAGE SignatureRegistry for
    platform-based verification.

    Usage:
        verifier = SageBlockVerifier()
        is_valid = verifier.verify_block_signature(
            header, signature, public_key=platform_pubkey
        )

        # Or with registry:
        verifier = SageBlockVerifier(registry=signature_registry)
        is_valid = verifier.verify_block_signature_by_platform(
            header, signature, platform_name="Thor"
        )

    Attributes:
        registry: Optional SAGE SignatureRegistry for platform lookup
    """

    registry: SignatureRegistry = None

    def verify_block_signature(
        self,
        header: Dict[str, Any],
        signature: bytes,
        *,
        public_key: bytes
    ) -> bool:
        """Verify a Web4 block signature using SAGE Ed25519 cryptography.

        Args:
            header: Block header dict (same format as signing)
            signature: Ed25519 signature bytes from signer
            public_key: Ed25519 public key bytes (32 bytes)

        Returns:
            True if signature is valid, False otherwise
        """
        # Canonical JSON encoding (matches signing)
        header_json = json.dumps(header, sort_keys=True, separators=(",", ":"))
        header_bytes = header_json.encode("utf-8")

        # Verify with SAGE Ed25519
        return FederationCrypto.verify_signature(public_key, header_bytes, signature)

    def verify_block_signature_by_platform(
        self,
        header: Dict[str, Any],
        signature: bytes,
        *,
        platform_name: str
    ) -> bool:
        """Verify a Web4 block signature using SAGE SignatureRegistry.

        Looks up the platform's public key from the registry and verifies
        the signature. This enables platform-based verification without
        needing to pass public keys explicitly.

        Args:
            header: Block header dict
            signature: Ed25519 signature bytes
            platform_name: SAGE platform name (e.g. "Thor", "Sprout")

        Returns:
            True if signature is valid, False otherwise

        Raises:
            ValueError: If registry is not configured or platform not registered
        """
        if self.registry is None:
            raise ValueError("SignatureRegistry required for platform-based verification")

        # Look up public key from registry
        if not self.registry.is_registered(platform_name):
            raise ValueError(f"Platform '{platform_name}' not registered")

        public_key = self.registry.get_public_key(platform_name)

        # Verify using platform's public key
        return self.verify_block_signature(header, signature, public_key=public_key)


def create_sage_block_signer_from_identity(
    platform_name: str,
    lct_id: str,
    key_path: str = None
) -> SageBlockSigner:
    """Create a SAGE block signer from platform identity.

    Convenience function to create a block signer with key persistence.
    Generates a new key pair if one doesn't exist, or loads from disk.

    Args:
        platform_name: Platform name (e.g. "Thor", "Sprout")
        lct_id: LCT identifier (e.g. "thor_sage_lct")
        key_path: Optional path to key file. If None, uses default location.

    Returns:
        SageBlockSigner ready to sign blocks

    Example:
        signer = create_sage_block_signer_from_identity("Thor", "thor_sage_lct")
        signature = signer.sign_block_header(header)
    """
    import os
    from pathlib import Path

    # Default key path: sage/data/keys/{platform_name}_ed25519.key
    if key_path is None:
        sage_root = Path(__file__).parent.parent
        key_dir = sage_root / "data" / "keys"
        key_dir.mkdir(parents=True, exist_ok=True)
        key_path = str(key_dir / f"{platform_name}_ed25519.key")

    # Load existing key or generate new one
    if os.path.exists(key_path):
        with open(key_path, 'rb') as f:
            private_key_bytes = f.read()
        keypair = FederationKeyPair.from_bytes(platform_name, lct_id, private_key_bytes)
    else:
        keypair = FederationKeyPair.generate(platform_name, lct_id)
        with open(key_path, 'wb') as f:
            f.write(keypair.private_key_bytes())

    return SageBlockSigner(keypair)


# Example usage and testing utilities

def demo_web4_sage_signing():
    """Demonstrate Web4 block signing with SAGE cryptography.

    This shows the complete flow:
    1. Create SAGE-backed signer for a platform
    2. Sign a Web4 block header
    3. Verify the signature
    4. Show platform-based verification with registry
    """
    print("=== Web4/SAGE Block Signing Demo ===\n")

    # Create Thor platform signer
    print("1. Creating SAGE-backed block signer for Thor...")
    thor_keypair = FederationKeyPair.generate("Thor", "thor_sage_lct")
    thor_signer = SageBlockSigner(thor_keypair)
    print(f"   ✓ Thor signer created")
    print(f"   Public key: {thor_keypair.public_key_bytes().hex()[:32]}...\n")

    # Create a Web4 block header
    print("2. Creating Web4 block header...")
    header = {
        "index": 1,
        "society_lct": "thor_sage_lct",
        "previous_hash": "0" * 64,
        "timestamp": 1732900000.0
    }
    print(f"   Header: {header}\n")

    # Sign the block
    print("3. Signing block with SAGE Ed25519...")
    signature = thor_signer.sign_block_header(header)
    print(f"   ✓ Signature: {signature.hex()[:32]}...")
    print(f"   Signature length: {len(signature)} bytes\n")

    # Verify with explicit public key
    print("4. Verifying signature with explicit public key...")
    verifier = SageBlockVerifier()
    is_valid = verifier.verify_block_signature(
        header,
        signature,
        public_key=thor_keypair.public_key_bytes()
    )
    print(f"   ✓ Signature valid: {is_valid}\n")

    # Verify with registry (platform-based)
    print("5. Verifying with SAGE SignatureRegistry...")
    registry = SignatureRegistry()
    registry.register_platform("Thor", thor_keypair.public_key_bytes())

    registry_verifier = SageBlockVerifier(registry=registry)
    is_valid_registry = registry_verifier.verify_block_signature_by_platform(
        header,
        signature,
        platform_name="Thor"
    )
    print(f"   ✓ Registry verification: {is_valid_registry}\n")

    # Test tampering detection
    print("6. Testing tampering detection...")
    tampered_header = header.copy()
    tampered_header["index"] = 2
    is_invalid = verifier.verify_block_signature(
        tampered_header,
        signature,
        public_key=thor_keypair.public_key_bytes()
    )
    print(f"   ✓ Tampered header rejected: {not is_invalid}\n")

    print("=== Demo Complete ===")
    print("\nKey insights:")
    print("- Web4 blocks can be signed with SAGE Ed25519 cryptography")
    print("- Signatures bind blocks to hardware-bound platform identities")
    print("- SignatureRegistry enables platform-based verification")
    print("- Tampering is detected automatically")


if __name__ == "__main__":
    # Run demo if executed directly
    demo_web4_sage_signing()
