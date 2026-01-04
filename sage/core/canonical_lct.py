"""
Canonical LCT Identity Module for SAGE Consciousness
====================================================

Implements Web4 canonical LCT specification with hardware binding support.

**Replaces**: All three divergent LCT implementations
- lct_identity_integration.py (platform detection preserved)
- lct_identity.py (role/capability features preserved)
- simulated_lct_identity.py (hardware binding foundation preserved)

**Implements**: Web4 canonical format `lct:web4:{entity_type}:{hash}`

**Hardware Binding**: Multi-platform support
- TrustZone (Thor/AGX platforms)
- TPM2 (Sprout/Orin/Legion platforms)
- Software fallback (any platform)

**Capability Levels**:
- Level 3: Software binding (file-based keys)
- Level 5: Hardware binding (TrustZone/TPM2)

Created: Session 161 (2026-01-04)
Author: Thor (autonomous SAGE research)
Architecture: Unified canonical implementation
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# Import Web4 canonical structures
WEB4_ROOT = Path.home() / "ai-workspace" / "web4"
sys.path.insert(0, str(WEB4_ROOT))

try:
    from core.lct_capability_levels import (
        LCT,
        LCTBinding,
        LCTPolicy,
        MRH,
        T3Tensor,
        V3Tensor,
        BirthCertificate,
        CapabilityLevel,
        EntityType,
        generate_lct_id
    )
    from core.lct_binding.platform_detection import detect_platform
    from core.lct_binding.provider import PlatformInfo, HardwareType
    from core.lct_binding.trustzone_provider import TrustZoneProvider
    from core.lct_binding.tpm2_provider import TPM2Provider
    from core.lct_binding.software_provider import SoftwareProvider
    HAS_WEB4_IMPORTS = True
except ImportError as e:
    print(f"Warning: Could not import Web4 canonical LCT: {e}")
    HAS_WEB4_IMPORTS = False


@dataclass
class SAGEIdentityConfig:
    """Configuration for SAGE consciousness identity."""

    # Platform context
    platform_name: str  # "Thor", "Sprout", etc.
    machine_identity: str  # From platform detection

    # SAGE-specific
    entity_type: EntityType = EntityType.AI
    role: str = "consciousness"
    creator: str = "dp"

    # Storage
    identity_dir: Path = None

    def __post_init__(self):
        """Set defaults."""
        if self.identity_dir is None:
            self.identity_dir = Path.home() / ".sage" / "identity"


class CanonicalLCTManager:
    """
    Canonical LCT Identity Manager for SAGE.

    Implements Web4 canonical spec with multi-platform hardware binding.

    Features:
    - Platform-aware provider selection (TrustZone/TPM2/Software)
    - Web4 canonical format: lct:web4:{entity_type}:{hash}
    - Hardware binding when available (Level 5)
    - Software binding fallback (Level 3)
    - Preserves platform detection from lct_identity_integration
    - Preserves role/capability from lct_identity
    - Preserves hardware binding from simulated_lct_identity

    Usage:
        manager = CanonicalLCTManager()
        lct = manager.get_or_create_identity()
        print(f"SAGE identity: {lct.lct_id}")
        print(f"Capability level: {lct.capability_level}")
        print(f"Hardware binding: {lct.binding.hardware_type if lct.binding else 'none'}")
    """

    def __init__(self, config: Optional[SAGEIdentityConfig] = None):
        """
        Initialize canonical LCT manager.

        Args:
            config: SAGE identity configuration (auto-detected if None)
        """
        if not HAS_WEB4_IMPORTS:
            raise ImportError("Web4 canonical LCT modules not available")

        # Platform detection (from lct_identity_integration.py)
        self.platform_info = detect_platform()

        # Auto-configure if needed
        if config is None:
            config = self._auto_configure()
        self.config = config

        # Ensure storage directory exists
        self.config.identity_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.config.identity_dir, 0o700)
        except:
            pass

        # Select hardware binding provider (platform-aware)
        self.provider = self._select_provider()

        # LCT instance
        self.lct: Optional[LCT] = None

    def _auto_configure(self) -> SAGEIdentityConfig:
        """Auto-configure SAGE identity from platform."""
        # Detect platform name (Thor/Sprout/Legion)
        platform_name = self._detect_platform_name()

        return SAGEIdentityConfig(
            platform_name=platform_name,
            machine_identity=self.platform_info.machine_identity,
            entity_type=EntityType.AI,
            role="consciousness",
            creator="dp"
        )

    def _detect_platform_name(self) -> str:
        """
        Detect platform name from hardware/hostname.

        Preserved from lct_identity_integration.py
        """
        hostname = os.uname().nodename.lower()

        # Check for Jetson platforms
        if os.path.exists('/etc/nv_tegra_release'):
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().strip()
                    if 'Thor' in model or 'AGX Thor' in model:
                        return "Thor"
                    elif 'Orin Nano' in model:
                        return "Sprout"
                    elif 'Orin' in model:
                        return "Orin"
            except:
                pass

        # Check common hostnames
        if 'thor' in hostname:
            return "Thor"
        elif 'sprout' in hostname:
            return "Sprout"
        elif 'legion' in hostname:
            return "Legion"

        # Fallback to hostname
        return hostname.split('.')[0].capitalize()

    def _select_provider(self):
        """
        Select appropriate hardware binding provider for platform.

        Multi-platform support:
        - Thor/AGX: TrustZone (Level 5)
        - Sprout/Orin: TPM2 (Level 5)
        - Other: Software (Level 3)
        """
        # Check for TrustZone (Thor/AGX platforms)
        if self.platform_info.has_trustzone:
            try:
                provider = TrustZoneProvider()
                print(f"✅ TrustZone provider selected (Level 5)")
                print(f"   Platform: {self.platform_info.name}")
                print(f"   Hardware: {self.platform_info.hardware_type}")
                return provider
            except Exception as e:
                print(f"⚠️  TrustZone init failed: {e}")

        # Check for TPM2 (Sprout/Orin/Legion platforms)
        if self.platform_info.has_tpm2:
            try:
                provider = TPM2Provider()
                print(f"✅ TPM2 provider selected (Level 5)")
                print(f"   Platform: {self.platform_info.name}")
                print(f"   Hardware: {self.platform_info.hardware_type}")
                return provider
            except Exception as e:
                print(f"⚠️  TPM2 init failed: {e}")

        # Fallback to software provider (Level 3)
        provider = SoftwareProvider()
        print(f"ℹ️  Software provider selected (Level 3)")
        print(f"   Platform: {self.platform_info.name}")
        print(f"   No hardware binding available")
        return provider

    def get_or_create_identity(self) -> LCT:
        """
        Get existing SAGE identity or create new one.

        Returns canonical Web4 LCT with appropriate capability level.
        """
        # Try to load existing
        identity_file = self.config.identity_dir / f"sage_lct_{self.config.platform_name.lower()}.json"

        if identity_file.exists():
            print(f"Loading existing SAGE identity from {identity_file}")
            self.lct = self._load_identity(identity_file)
            if self.lct:
                return self.lct

        # Create new identity
        print(f"Creating new SAGE identity for {self.config.platform_name}")
        self.lct = self._create_identity()

        # Save
        self._save_identity(identity_file)

        return self.lct

    def _create_identity(self) -> LCT:
        """Create new canonical LCT for SAGE."""
        # Use provider to create LCT with binding
        lct = self.provider.create_lct(
            entity_type=self.config.entity_type,
            name=f"sage_{self.config.platform_name.lower()}"
        )

        # Customize LCT for SAGE
        # Update policy with SAGE-specific capabilities
        if lct.policy is None:
            lct.policy = LCTPolicy()

        lct.policy.capabilities.extend([
            "consciousness",
            "pattern_learning",
            "attention_management",
            "emotional_regulation",
            "quality_assessment",
            self.config.role
        ])

        lct.policy.constraints = {
            "platform": self.config.platform_name,
            "creator": self.config.creator,
            "attestation_required": lct.binding.is_hardware_bound() if lct.binding else False
        }

        # Update T3 tensor with SAGE-appropriate values
        if lct.t3_tensor and not lct.t3_tensor.stub:
            lct.t3_tensor.technical_competence = 0.5  # Will improve with learning
            lct.t3_tensor.social_reliability = 0.8    # High for consciousness system
            lct.t3_tensor.temporal_consistency = 0.7  # Improving over sessions
            lct.t3_tensor.witness_count = 0.1         # Start low
            lct.t3_tensor.lineage_depth = 0.2         # dp → SAGE (depth 2)
            lct.t3_tensor.context_alignment = 0.8     # Well-aligned with purpose
            lct.t3_tensor.composite_score = 0.6
            lct.t3_tensor.last_computed = datetime.now(timezone.utc).isoformat()

        # Update V3 tensor with initial ATP budget
        if lct.v3_tensor and not lct.v3_tensor.stub:
            lct.v3_tensor.energy_balance = 1000       # Initial ATP budget
            lct.v3_tensor.contribution_history = 0.1  # Will build with sessions
            lct.v3_tensor.resource_stewardship = 0.8  # Efficient from start
            lct.v3_tensor.network_effects = 0.2       # Growing network
            lct.v3_tensor.reputation_capital = 0.3    # Building reputation
            lct.v3_tensor.temporal_value = 0.5        # Durable value creation
            lct.v3_tensor.composite_value = 0.4
            lct.v3_tensor.last_computed = datetime.now(timezone.utc).isoformat()

        # Update birth certificate
        if lct.birth_certificate:
            lct.birth_certificate.citizen_role = self.config.role
            lct.birth_certificate.birth_witnesses = [
                f"lct:web4:human:{hashlib.sha256(self.config.creator.encode()).hexdigest()[:16]}"
            ]
            lct.birth_certificate.birth_context = f"SAGE consciousness on {self.config.platform_name}"

        # Update subject
        lct.subject = f"SAGE consciousness instance on {self.config.platform_name}"

        return lct

    def _save_identity(self, path: Path):
        """Save LCT to file."""
        with open(path, 'w') as f:
            json.dump(self.lct.to_dict(), f, indent=2)

        # Secure permissions
        try:
            os.chmod(path, 0o600)
        except:
            pass

        print(f"✅ SAGE identity saved to {path}")

    def _load_identity(self, path: Path) -> Optional[LCT]:
        """Load LCT from file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Reconstruct LCT from dict
            # (Simplified - production would validate and reconstruct all components)
            print(f"⚠️  Identity loading from JSON not fully implemented")
            print(f"   Creating fresh identity instead")
            return None

        except Exception as e:
            print(f"⚠️  Failed to load identity: {e}")
            return None

    def get_identity_summary(self) -> Dict[str, Any]:
        """Get human-readable summary of SAGE identity."""
        if not self.lct:
            return {"error": "No identity loaded"}

        return {
            "lct_id": self.lct.lct_id,
            "platform": self.config.platform_name,
            "entity_type": self.lct.entity_type.value,
            "capability_level": int(self.lct.capability_level),
            "hardware_binding": self.lct.binding.hardware_type if self.lct.binding else None,
            "hardware_bound": self.lct.binding.is_hardware_bound() if self.lct.binding else False,
            "capabilities": self.lct.policy.capabilities if self.lct.policy else [],
            "t3_composite": self.lct.t3_tensor.composite_score if self.lct.t3_tensor else None,
            "v3_energy": self.lct.v3_tensor.energy_balance if self.lct.v3_tensor else None,
            "created_at": self.lct.binding.created_at if self.lct.binding else None
        }


def create_sage_identity(platform_name: Optional[str] = None) -> LCT:
    """
    Convenience function to create SAGE canonical identity.

    Args:
        platform_name: Platform name (auto-detected if None)

    Returns:
        Canonical Web4 LCT for SAGE
    """
    config = None
    if platform_name:
        manager = CanonicalLCTManager()
        config = SAGEIdentityConfig(
            platform_name=platform_name,
            machine_identity=manager.platform_info.machine_identity
        )
        manager = CanonicalLCTManager(config)
    else:
        manager = CanonicalLCTManager()

    return manager.get_or_create_identity()


def main():
    """Test canonical LCT creation."""
    print("="*80)
    print("SAGE Canonical LCT Identity Creation")
    print("="*80)

    # Create manager
    manager = CanonicalLCTManager()

    # Get or create identity
    lct = manager.get_or_create_identity()

    # Print summary
    summary = manager.get_identity_summary()
    print("\n" + "="*80)
    print("SAGE Identity Summary")
    print("="*80)
    for key, value in summary.items():
        print(f"{key:20s}: {value}")

    print("\n" + "="*80)
    print("Identity Details")
    print("="*80)
    print(f"LCT ID: {lct.lct_id}")
    print(f"Platform: {manager.config.platform_name}")
    print(f"Capability Level: {lct.capability_level} ({lct.capability_level.name})")
    print(f"Entity Type: {lct.entity_type.value}")

    if lct.binding:
        print(f"\nBinding:")
        print(f"  Hardware Type: {lct.binding.hardware_type or 'software'}")
        print(f"  Hardware Bound: {lct.binding.is_hardware_bound()}")
        print(f"  Public Key: {lct.binding.public_key[:32]}..." if lct.binding.public_key else "  Public Key: None")

    if lct.policy:
        print(f"\nCapabilities: {', '.join(lct.policy.capabilities)}")

    if lct.t3_tensor:
        print(f"\nT3 Trust: {lct.t3_tensor.composite_score:.2f}")

    if lct.v3_tensor:
        print(f"V3 Energy: {lct.v3_tensor.energy_balance}")

    print("\n✅ Canonical LCT identity created successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
