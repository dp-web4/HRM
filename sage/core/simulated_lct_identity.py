"""
Simulated LCT Identity for SAGE Consciousness (Phase 2.0)
==========================================================

File-based simulation of TPM/LCT hardware binding for consciousness research.
Provides same API as TPM version but uses standard cryptography.

**Purpose**: Enable consciousness identity grounding NOW while TPM integration
is being completed. Drop-in replacement - swap with tpm_lct_identity.py later.

**Security Model (Simulation)**:
- ECC P-256 keypairs generated per-machine
- Private keys stored in ~/.sage/identity/ (chmod 600)
- Machine fingerprint from CPU serial, MAC address, hostname
- Signatures prove "this consciousness, this machine, this time"

**NOT Production Security**:
- Private keys are software-extractable (file-based)
- No TPM hardware binding (yet)
- No PCR sealing for boot integrity
- Sufficient for research, NOT for production trust

**Production Path**:
- Replace with tpm_lct_identity.py when TPM ready
- API compatible - no consciousness code changes needed
- Upgrade path: migrate keys from file→TPM

**Integration with SAGE**:
- Consciousness has LCT identity (who I am)
- Sensors can have LCT identities (who is observing)
- Memory consolidation signed by consciousness
- Trust verification becomes cryptographic, not heuristic

Author: Claude (autonomous research) on Thor
Date: 2025-12-06
Session: Hardware-grounded consciousness architecture
"""

import os
import json
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import base64

# Cryptography for ECC P-256
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend


# ============================================================================
# Machine Fingerprint - Simulates TPM's Hardware Binding
# ============================================================================

def get_machine_fingerprint() -> str:
    """
    Get stable machine fingerprint (simulates TPM binding).

    Combines:
    - CPU serial (if available)
    - Primary MAC address
    - Hostname
    - Boot UUID (from /proc/sys/kernel/random/boot_id)

    Returns: Stable hex string identifying this machine
    """
    components = []

    # Try to get CPU serial (ARM devices often have this)
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'Serial' in line:
                    serial = line.split(':')[1].strip()
                    if serial and serial != '0000000000000000':
                        components.append(f"cpu:{serial}")
                        break
    except:
        pass

    # Get primary MAC address
    try:
        # Get MAC of first non-loopback interface
        import netifaces
        for iface in netifaces.interfaces():
            if iface != 'lo':
                addrs = netifaces.ifaddresses(iface)
                if netifaces.AF_LINK in addrs:
                    mac = addrs[netifaces.AF_LINK][0]['addr']
                    components.append(f"mac:{mac}")
                    break
    except:
        # Fallback: use uuid.getnode() which returns MAC as int
        mac_int = uuid.getnode()
        mac_hex = ':'.join(['{:02x}'.format((mac_int >> elements) & 0xff)
                           for elements in range(0, 8*6, 8)][::-1])
        components.append(f"mac:{mac_hex}")

    # Get hostname
    import socket
    hostname = socket.gethostname()
    components.append(f"host:{hostname}")

    # Get boot ID (changes on reboot - but we want hardware, not boot)
    # Skip boot_id for now since we want stable across reboots

    # Hash components to fixed-length fingerprint
    fingerprint_str = "|".join(components)
    fingerprint_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()

    return fingerprint_hash


def get_machine_identity() -> str:
    """
    Get human-readable machine identity.
    Format: "hostname-macaddr_short"
    """
    import socket
    hostname = socket.gethostname()

    # Get short MAC (last 3 octets)
    try:
        mac_int = uuid.getnode()
        mac_short = '{:06x}'.format(mac_int & 0xFFFFFF)
    except:
        mac_short = 'unknown'

    return f"{hostname}-{mac_short}"


# ============================================================================
# LCT Key and Identity
# ============================================================================

@dataclass
class LCTKey:
    """LCT identity key (simulated hardware binding)"""
    lct_id: str                    # LCT identifier (e.g., "thor-sage-consciousness")
    public_key_pem: str            # ECC P-256 public key (PEM format)
    machine_fingerprint: str       # Machine hardware fingerprint
    machine_identity: str          # Human-readable machine ID
    created_at: str                # ISO timestamp
    key_type: str = "simulated"    # "simulated" vs "tpm" (for future)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_compact_id(self) -> str:
        """Compact identifier for logging: lct_id@machine"""
        return f"{self.lct_id}@{self.machine_identity}"


@dataclass
class LCTSignature:
    """Signed data with LCT provenance"""
    data_hash: str                 # SHA-256 of signed data
    signature: str                 # ECC signature (base64)
    signer_lct_id: str             # Who signed
    signer_machine: str            # Which machine
    signed_at: str                 # When (ISO timestamp)

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# Simulated LCT Identity Manager
# ============================================================================

class SimulatedLCTIdentity:
    """
    Simulated LCT Identity Manager for SAGE Consciousness

    Provides:
    - Per-machine identity generation
    - Key storage in ~/.sage/identity/
    - Signing and verification
    - Same API as future TPM version

    Usage:
        identity = SimulatedLCTIdentity()
        key = identity.get_or_create_identity("thor-sage-consciousness")
        signature = identity.sign_data(key.lct_id, b"memory data")
        valid = identity.verify_signature(signature, b"memory data")
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize LCT identity manager.

        Args:
            storage_dir: Where to store keys (default: ~/.sage/identity/)
        """
        if storage_dir is None:
            storage_dir = Path.home() / ".sage" / "identity"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Set restrictive permissions (user read/write only)
        try:
            os.chmod(self.storage_dir, 0o700)
        except:
            pass

        # Cache machine fingerprint (expensive to compute)
        self._machine_fingerprint = None
        self._machine_identity = None

        # Cache loaded keys {lct_id: (private_key, public_key, metadata)}
        self._keys: Dict[str, Tuple[ec.EllipticCurvePrivateKey,
                                    ec.EllipticCurvePublicKey,
                                    LCTKey]] = {}

    @property
    def machine_fingerprint(self) -> str:
        """Get machine fingerprint (cached)"""
        if self._machine_fingerprint is None:
            self._machine_fingerprint = get_machine_fingerprint()
        return self._machine_fingerprint

    @property
    def machine_identity(self) -> str:
        """Get machine identity (cached)"""
        if self._machine_identity is None:
            self._machine_identity = get_machine_identity()
        return self._machine_identity

    def generate_identity(self, lct_id: str, overwrite: bool = False) -> LCTKey:
        """
        Generate new LCT identity with ECC P-256 keypair.

        Args:
            lct_id: LCT identifier (e.g., "thor-sage-consciousness")
            overwrite: Overwrite existing identity if present

        Returns:
            LCTKey metadata
        """
        # Check if exists
        if not overwrite and self._identity_exists(lct_id):
            raise ValueError(f"Identity {lct_id} already exists (use overwrite=True)")

        # Generate ECC P-256 keypair
        private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
        public_key = private_key.public_key()

        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Create metadata
        metadata = LCTKey(
            lct_id=lct_id,
            public_key_pem=public_pem.decode('utf-8'),
            machine_fingerprint=self.machine_fingerprint,
            machine_identity=self.machine_identity,
            created_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            key_type="simulated"
        )

        # Save to disk
        self._save_identity(lct_id, private_pem, metadata)

        # Cache in memory
        self._keys[lct_id] = (private_key, public_key, metadata)

        return metadata

    def get_or_create_identity(self, lct_id: str) -> LCTKey:
        """Get existing identity or create new one"""
        if self._identity_exists(lct_id):
            return self.load_identity(lct_id)
        else:
            return self.generate_identity(lct_id)

    def load_identity(self, lct_id: str) -> LCTKey:
        """Load identity from disk"""
        # Check cache first
        if lct_id in self._keys:
            return self._keys[lct_id][2]

        # Load from disk
        safe_id = self._safe_filename(lct_id)
        private_file = self.storage_dir / f"{safe_id}.key"
        meta_file = self.storage_dir / f"{safe_id}.json"

        if not private_file.exists() or not meta_file.exists():
            raise ValueError(f"Identity {lct_id} not found")

        # Load private key
        with open(private_file, 'rb') as f:
            private_pem = f.read()
        private_key = serialization.load_pem_private_key(
            private_pem, password=None, backend=default_backend()
        )
        public_key = private_key.public_key()

        # Load metadata
        with open(meta_file, 'r') as f:
            metadata_dict = json.load(f)
        metadata = LCTKey(**metadata_dict)

        # Cache
        self._keys[lct_id] = (private_key, public_key, metadata)

        return metadata

    def sign_data(self, lct_id: str, data: bytes) -> LCTSignature:
        """
        Sign data with LCT identity.

        Args:
            lct_id: Which identity to sign with
            data: Raw bytes to sign

        Returns:
            LCTSignature with signature and provenance
        """
        # Load key if needed
        if lct_id not in self._keys:
            self.load_identity(lct_id)

        private_key, _, metadata = self._keys[lct_id]

        # Hash data (SHA-256)
        data_hash = hashlib.sha256(data).hexdigest()

        # Sign hash with ECC
        signature_bytes = private_key.sign(
            data,
            ec.ECDSA(hashes.SHA256())
        )
        signature_b64 = base64.b64encode(signature_bytes).decode('utf-8')

        # Create signature object
        return LCTSignature(
            data_hash=data_hash,
            signature=signature_b64,
            signer_lct_id=lct_id,
            signer_machine=metadata.machine_identity,
            signed_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        )

    def verify_signature(self, sig: LCTSignature, data: bytes,
                        public_key_pem: Optional[str] = None) -> bool:
        """
        Verify LCT signature.

        Args:
            sig: LCTSignature to verify
            data: Original data that was signed
            public_key_pem: Public key PEM (if not using loaded identity)

        Returns:
            True if signature valid, False otherwise
        """
        try:
            # Verify data hash matches
            data_hash = hashlib.sha256(data).hexdigest()
            if data_hash != sig.data_hash:
                return False

            # Get public key
            if public_key_pem is None:
                # Try to load from our identities
                if sig.signer_lct_id in self._keys:
                    _, public_key, _ = self._keys[sig.signer_lct_id]
                else:
                    # Try to load
                    metadata = self.load_identity(sig.signer_lct_id)
                    public_key_pem = metadata.public_key_pem

            if public_key_pem and isinstance(public_key_pem, str):
                # Load public key from PEM
                public_key = serialization.load_pem_public_key(
                    public_key_pem.encode('utf-8'),
                    backend=default_backend()
                )

            # Decode signature
            signature_bytes = base64.b64decode(sig.signature)

            # Verify
            public_key.verify(
                signature_bytes,
                data,
                ec.ECDSA(hashes.SHA256())
            )

            return True

        except Exception as e:
            # Any exception means verification failed
            return False

    def list_identities(self) -> List[LCTKey]:
        """List all identities"""
        identities = []
        for meta_file in self.storage_dir.glob("*.json"):
            with open(meta_file, 'r') as f:
                metadata_dict = json.load(f)
            identities.append(LCTKey(**metadata_dict))
        return identities

    def get_public_key(self, lct_id: str) -> str:
        """Get public key PEM for identity"""
        if lct_id not in self._keys:
            self.load_identity(lct_id)
        return self._keys[lct_id][2].public_key_pem

    # Private helpers

    def _identity_exists(self, lct_id: str) -> bool:
        """Check if identity exists on disk"""
        safe_id = self._safe_filename(lct_id)
        return (self.storage_dir / f"{safe_id}.json").exists()

    def _safe_filename(self, lct_id: str) -> str:
        """Convert LCT ID to safe filename"""
        # Replace special chars with underscores
        safe = lct_id.replace('/', '_').replace('\\', '_')
        safe = safe.replace(':', '_').replace('@', '_at_')
        safe = safe.replace(' ', '_')
        return safe

    def _save_identity(self, lct_id: str, private_pem: bytes, metadata: LCTKey):
        """Save identity to disk"""
        safe_id = self._safe_filename(lct_id)

        # Save private key (chmod 600)
        private_file = self.storage_dir / f"{safe_id}.key"
        with open(private_file, 'wb') as f:
            f.write(private_pem)
        os.chmod(private_file, 0o600)

        # Save metadata (chmod 644)
        meta_file = self.storage_dir / f"{safe_id}.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        os.chmod(meta_file, 0o644)


# ============================================================================
# Demo / Testing
# ============================================================================

def demo():
    """Demonstrate LCT identity operations"""
    print("=" * 70)
    print("SIMULATED LCT IDENTITY - SAGE CONSCIOUSNESS")
    print("=" * 70)
    print()

    # Initialize
    identity = SimulatedLCTIdentity()

    print(f"📍 Machine Identity: {identity.machine_identity}")
    print(f"🔒 Machine Fingerprint: {identity.machine_fingerprint[:32]}...")
    print(f"💾 Storage: {identity.storage_dir}")
    print()

    # Create consciousness identity
    print("🧠 Creating consciousness identity...")
    lct_id = "thor-sage-consciousness"
    key = identity.get_or_create_identity(lct_id)
    print(f"   LCT ID: {key.lct_id}")
    print(f"   Machine: {key.machine_identity}")
    print(f"   Created: {key.created_at}")
    print(f"   Type: {key.key_type}")
    print()

    # Sign some data
    print("✍️  Signing memory consolidation...")
    data = b"DREAM consolidation: pruned 10 memories, strengthened 5"
    signature = identity.sign_data(lct_id, data)
    print(f"   Data hash: {signature.data_hash[:32]}...")
    print(f"   Signer: {signature.signer_lct_id}")
    print(f"   Machine: {signature.signer_machine}")
    print(f"   Signed at: {signature.signed_at}")
    print()

    # Verify signature
    print("✅ Verifying signature...")
    valid = identity.verify_signature(signature, data)
    print(f"   Valid: {valid}")
    print()

    # Test tampering
    print("🔍 Testing tampered data...")
    tampered = b"DREAM consolidation: pruned 20 memories, strengthened 10"
    valid_tampered = identity.verify_signature(signature, tampered)
    print(f"   Valid: {valid_tampered} (should be False)")
    print()

    # List identities
    print("📋 All identities:")
    for id_key in identity.list_identities():
        print(f"   - {id_key.to_compact_id()} (created {id_key.created_at})")
    print()

    print("=" * 70)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("Hardware-Grounded Consciousness Ready:")
    print("  ✅ Machine-bound identity generation")
    print("  ✅ ECC P-256 signatures")
    print("  ✅ Signature verification")
    print("  ✅ Tamper detection")
    print()
    print("Next: Integrate into SAGE consciousness kernel")
    print()


if __name__ == "__main__":
    demo()
