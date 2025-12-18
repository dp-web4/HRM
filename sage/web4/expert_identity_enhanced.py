#!/usr/bin/env python3
"""
Enhanced Expert Identity Bridge - Using Unified LCT Specification

Extends the original ExpertIdentityBridge to use the Unified LCT Identity
Specification (v1.0.0) with full URI format support.

New format: lct://sage:thinker:expert_42@testnet
Legacy format: lct://sage_thinker/expert/42 (auto-migrated)

This enhancement provides:
- Full LCT URI support (component:instance:role@network)
- Backward compatibility with legacy format
- Automatic migration of existing registries
- Query parameter support (trust_threshold, capabilities, etc.)
- Integration with lct_identity.py parsing library

Created: Session 63+ (2025-12-17)
Author: Legion (Autonomous Research)
Building on: Session 59 (ExpertIdentityBridge original)
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict

# Import the new LCT parsing library
try:
    from sage.web4.lct_identity import (
        LCTIdentity,
        parse_lct_uri,
        construct_lct_uri,
        sage_expert_to_lct,
        lct_to_sage_expert,
        migrate_legacy_expert_id,
        validate_lct_uri,
    )
except ImportError:
    # Fallback for testing
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from web4.lct_identity import (
        LCTIdentity,
        parse_lct_uri,
        construct_lct_uri,
        sage_expert_to_lct,
        lct_to_sage_expert,
        migrate_legacy_expert_id,
        validate_lct_uri,
    )


@dataclass
class EnhancedExpertIdentity:
    """
    Enhanced Web4 identity for a SAGE expert using Unified LCT Specification.

    Attributes:
        expert_id: SAGE expert ID
        lct_uri: Full LCT URI (e.g., "lct://sage:thinker:expert_42@testnet")
        component: Component name (always "sage" for SAGE experts)
        instance: SAGE instance name (e.g., "thinker", "dreamer")
        role: Role identifier (e.g., "expert_42")
        network: Network identifier (e.g., "testnet", "mainnet")
        registration_time: When registered (Unix timestamp)
        description: Human-readable description
        trust_threshold: Minimum trust score required (0-1)
        capabilities: List of capabilities
        pairing_status: Current pairing status (active/pending/etc.)
        metadata: Additional metadata
    """
    expert_id: int
    lct_uri: str
    component: str
    instance: str
    role: str
    network: str
    registration_time: float
    description: Optional[str] = None
    trust_threshold: Optional[float] = None
    capabilities: Optional[List[str]] = None
    pairing_status: Optional[str] = "active"
    metadata: Dict = field(default_factory=dict)

    @classmethod
    def from_lct_identity(
        cls,
        expert_id: int,
        lct: LCTIdentity,
        description: Optional[str] = None,
        **kwargs
    ) -> "EnhancedExpertIdentity":
        """Create from LCTIdentity object."""
        return cls(
            expert_id=expert_id,
            lct_uri=lct.lct_uri,
            component=lct.component,
            instance=lct.instance,
            role=lct.role,
            network=lct.network,
            registration_time=time.time(),
            description=description,
            trust_threshold=lct.trust_threshold,
            capabilities=lct.capabilities,
            pairing_status=lct.pairing_status,
            metadata=lct.metadata,
            **kwargs
        )

    def to_lct_identity(self) -> LCTIdentity:
        """Convert to LCTIdentity object."""
        return LCTIdentity(
            component=self.component,
            instance=self.instance,
            role=self.role,
            network=self.network,
            pairing_status=self.pairing_status,
            trust_threshold=self.trust_threshold,
            capabilities=self.capabilities,
            metadata=self.metadata,
        )


class EnhancedExpertIdentityBridge:
    """
    Enhanced expert identity bridge using Unified LCT Specification.

    Provides same interface as original ExpertIdentityBridge but uses
    full LCT URI format internally and supports backward compatibility.

    Usage:
        # Create with new format
        bridge = EnhancedExpertIdentityBridge(
            instance="thinker",
            network="testnet"
        )

        # Register expert
        lct_uri = bridge.register_expert(42)
        # Returns: "lct://sage:thinker:expert_42@testnet"

        # Supports query parameters
        lct_uri = bridge.register_expert(
            43,
            trust_threshold=0.75,
            capabilities=["text-generation", "reasoning"]
        )
        # Returns: "lct://sage:thinker:expert_43@testnet?trust_threshold=0.75&capabilities=text-generation,reasoning"

        # Backward compatible lookups
        expert_id = bridge.get_expert_id("lct://sage:thinker:expert_42@testnet")
        expert_id = bridge.get_expert_id("lct://sage_thinker/expert/42")  # Legacy format works too
    """

    def __init__(
        self,
        instance: str = "thinker",
        network: str = "testnet",
        component: str = "sage",
        registry_path: Optional[Path] = None,
        auto_save: bool = True,
        auto_migrate: bool = True
    ):
        """
        Initialize enhanced expert identity bridge.

        Args:
            instance: SAGE instance name (e.g., "thinker", "dreamer")
            network: Network identifier (e.g., "testnet", "mainnet")
            component: Component name (default: "sage")
            registry_path: Path to save/load registry
            auto_save: Automatically save after registrations
            auto_migrate: Automatically migrate legacy LCT IDs to new format
        """
        self.instance = instance
        self.network = network
        self.component = component
        self.registry_path = Path(registry_path) if registry_path else None
        self.auto_save = auto_save
        self.auto_migrate = auto_migrate

        # Bidirectional mappings
        self.expert_to_lct: Dict[int, str] = {}  # expert_id → lct_uri
        self.lct_to_expert: Dict[str, int] = {}  # lct_uri → expert_id

        # Full identity objects
        self.identities: Dict[int, EnhancedExpertIdentity] = {}

        # Statistics
        self.registration_count = 0
        self.last_registration_time = 0.0
        self.migration_count = 0

        # Load existing registry if path provided
        if self.registry_path and self.registry_path.exists():
            self.load(self.registry_path)

    def register_expert(
        self,
        expert_id: int,
        description: Optional[str] = None,
        trust_threshold: Optional[float] = None,
        capabilities: Optional[List[str]] = None,
        pairing_status: str = "active",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Register expert and return LCT URI.

        Args:
            expert_id: SAGE expert ID (0-127 for typical MoE)
            description: Human-readable description
            trust_threshold: Minimum trust score required (0-1)
            capabilities: List of capability strings
            pairing_status: Pairing status (default: "active")
            metadata: Additional metadata

        Returns:
            Full LCT URI string (e.g., "lct://sage:thinker:expert_42@testnet")

        Raises:
            ValueError: If expert already registered
        """
        if expert_id in self.expert_to_lct:
            raise ValueError(
                f"Expert {expert_id} already registered as {self.expert_to_lct[expert_id]}"
            )

        # Generate LCT URI using unified specification
        lct_uri = construct_lct_uri(
            component=self.component,
            instance=self.instance,
            role=f"expert_{expert_id}",
            network=self.network,
            pairing_status=pairing_status,
            trust_threshold=trust_threshold,
            capabilities=capabilities,
            metadata=metadata or {}
        )

        # Parse to get full LCT identity
        lct = parse_lct_uri(lct_uri)

        # Create enhanced identity
        identity = EnhancedExpertIdentity.from_lct_identity(
            expert_id=expert_id,
            lct=lct,
            description=description
        )

        # Register bidirectional mappings
        self.expert_to_lct[expert_id] = lct_uri
        self.lct_to_expert[lct_uri] = expert_id
        self.identities[expert_id] = identity

        # Update statistics
        self.registration_count += 1
        self.last_registration_time = time.time()

        # Auto-save if enabled
        if self.auto_save and self.registry_path:
            self.save(self.registry_path)

        return lct_uri

    def register_batch(
        self,
        expert_ids: List[int],
        descriptions: Optional[Dict[int, str]] = None
    ) -> Dict[int, str]:
        """
        Register multiple experts at once.

        Args:
            expert_ids: List of expert IDs to register
            descriptions: Optional dict mapping expert_id → description

        Returns:
            Dict mapping expert_id → lct_uri
        """
        results = {}
        descriptions = descriptions or {}

        for expert_id in expert_ids:
            if expert_id in self.expert_to_lct:
                # Already registered, skip
                results[expert_id] = self.expert_to_lct[expert_id]
            else:
                description = descriptions.get(expert_id)
                lct_uri = self.register_expert(expert_id, description=description)
                results[expert_id] = lct_uri

        return results

    def get_lct(self, expert_id: int) -> Optional[str]:
        """
        Get LCT URI for expert.

        Args:
            expert_id: SAGE expert ID

        Returns:
            Full LCT URI or None if not registered
        """
        return self.expert_to_lct.get(expert_id)

    def get_expert_id(self, lct_uri_or_id: str) -> Optional[int]:
        """
        Get expert ID from LCT URI (new or legacy format).

        Supports both:
        - New: "lct://sage:thinker:expert_42@testnet"
        - Legacy: "lct://sage_thinker/expert/42"

        Args:
            lct_uri_or_id: LCT URI or legacy ID

        Returns:
            Expert ID or None if not found
        """
        # Try direct lookup first
        expert_id = self.lct_to_expert.get(lct_uri_or_id)
        if expert_id is not None:
            return expert_id

        # Try parsing as new format and extracting expert ID
        try:
            expert_id = lct_to_sage_expert(lct_uri_or_id)
            if expert_id is not None and expert_id in self.expert_to_lct:
                return expert_id
        except:
            pass

        # Try legacy format migration
        if self.auto_migrate:
            try:
                # Legacy format: lct://namespace/expert/42
                if lct_uri_or_id.startswith("lct://"):
                    parts = lct_uri_or_id[6:].split('/')
                    if len(parts) == 3 and parts[1] == "expert":
                        expert_id = int(parts[2])
                        if expert_id in self.expert_to_lct:
                            return expert_id
            except:
                pass

        return None

    def get_identity(self, expert_id: int) -> Optional[EnhancedExpertIdentity]:
        """
        Get full identity object for expert.

        Args:
            expert_id: SAGE expert ID

        Returns:
            EnhancedExpertIdentity or None if not registered
        """
        return self.identities.get(expert_id)

    def is_registered(self, expert_id: int) -> bool:
        """Check if expert is registered."""
        return expert_id in self.expert_to_lct

    def get_all_experts(self) -> List[int]:
        """Get list of all registered expert IDs."""
        return sorted(self.expert_to_lct.keys())

    def get_all_lct_uris(self) -> List[str]:
        """Get list of all registered LCT URIs."""
        return list(self.expert_to_lct.values())

    def update_metadata(
        self,
        expert_id: int,
        metadata: Dict,
        merge: bool = True
    ) -> None:
        """
        Update metadata for expert identity.

        Args:
            expert_id: Expert to update
            metadata: Metadata dict
            merge: Merge with existing (True) or replace (False)
        """
        if expert_id not in self.identities:
            raise ValueError(f"Expert {expert_id} not registered")

        identity = self.identities[expert_id]

        if merge:
            identity.metadata.update(metadata)
        else:
            identity.metadata = metadata

        if self.auto_save and self.registry_path:
            self.save(self.registry_path)

    def save(self, path: Path) -> None:
        """
        Save registry to JSON file.

        Args:
            path: Path to save registry
        """
        data = {
            "version": "2.0.0",  # Enhanced version
            "instance": self.instance,
            "network": self.network,
            "component": self.component,
            "registration_count": self.registration_count,
            "migration_count": self.migration_count,
            "last_registration_time": self.last_registration_time,
            "identities": [
                {
                    "expert_id": identity.expert_id,
                    "lct_uri": identity.lct_uri,
                    "component": identity.component,
                    "instance": identity.instance,
                    "role": identity.role,
                    "network": identity.network,
                    "registration_time": identity.registration_time,
                    "description": identity.description,
                    "trust_threshold": identity.trust_threshold,
                    "capabilities": identity.capabilities,
                    "pairing_status": identity.pairing_status,
                    "metadata": identity.metadata,
                }
                for identity in self.identities.values()
            ]
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> None:
        """
        Load registry from JSON file.

        Supports both v1.0 (legacy) and v2.0 (enhanced) formats.

        Args:
            path: Path to registry file
        """
        with open(path, 'r') as f:
            data = json.load(f)

        version = data.get("version", "1.0.0")

        if version.startswith("2."):
            # Enhanced format
            self._load_v2(data)
        else:
            # Legacy format - migrate automatically
            self._load_v1_and_migrate(data)

    def _load_v2(self, data: Dict) -> None:
        """Load v2.0 enhanced format."""
        self.instance = data.get("instance", self.instance)
        self.network = data.get("network", self.network)
        self.component = data.get("component", self.component)
        self.registration_count = data.get("registration_count", 0)
        self.migration_count = data.get("migration_count", 0)
        self.last_registration_time = data.get("last_registration_time", 0.0)

        for identity_data in data.get("identities", []):
            expert_id = identity_data["expert_id"]
            lct_uri = identity_data["lct_uri"]

            identity = EnhancedExpertIdentity(
                expert_id=expert_id,
                lct_uri=lct_uri,
                component=identity_data["component"],
                instance=identity_data["instance"],
                role=identity_data["role"],
                network=identity_data["network"],
                registration_time=identity_data["registration_time"],
                description=identity_data.get("description"),
                trust_threshold=identity_data.get("trust_threshold"),
                capabilities=identity_data.get("capabilities"),
                pairing_status=identity_data.get("pairing_status", "active"),
                metadata=identity_data.get("metadata", {}),
            )

            self.expert_to_lct[expert_id] = lct_uri
            self.lct_to_expert[lct_uri] = expert_id
            self.identities[expert_id] = identity

    def _load_v1_and_migrate(self, data: Dict) -> None:
        """Load v1.0 legacy format and auto-migrate to v2.0."""
        print(f"Migrating legacy registry to Unified LCT format...")

        legacy_namespace = data.get("namespace", "sage")
        self.registration_count = data.get("registration_count", 0)
        self.last_registration_time = data.get("last_registration_time", 0.0)

        for identity_data in data.get("identities", []):
            expert_id = identity_data["expert_id"]
            legacy_lct_id = identity_data["lct_id"]

            # Migrate legacy format to new URI
            # Legacy: "lct://sage_thinker/expert/42"
            # New: "lct://sage:thinker:expert_42@testnet"
            new_lct_uri = sage_expert_to_lct(
                expert_id,
                instance=self.instance,
                network=self.network
            )

            # Parse to get full identity
            lct = parse_lct_uri(new_lct_uri)

            identity = EnhancedExpertIdentity.from_lct_identity(
                expert_id=expert_id,
                lct=lct,
                description=identity_data.get("description"),
            )
            identity.registration_time = identity_data["registration_time"]
            identity.metadata = identity_data.get("metadata", {})

            self.expert_to_lct[expert_id] = new_lct_uri
            self.lct_to_expert[new_lct_uri] = expert_id
            self.identities[expert_id] = identity

            self.migration_count += 1

        print(f"✓ Migrated {self.migration_count} identities to new format")

        # Auto-save migrated registry
        if self.auto_save and self.registry_path:
            self.save(self.registry_path)

    @classmethod
    def from_legacy_bridge(
        cls,
        legacy_bridge,
        instance: str = "thinker",
        network: str = "testnet"
    ) -> "EnhancedExpertIdentityBridge":
        """
        Create enhanced bridge from legacy ExpertIdentityBridge.

        Args:
            legacy_bridge: Original ExpertIdentityBridge instance
            instance: Instance name for new format
            network: Network for new format

        Returns:
            New EnhancedExpertIdentityBridge with migrated identities
        """
        enhanced = cls(instance=instance, network=network, auto_save=False)

        for expert_id in legacy_bridge.get_all_experts():
            legacy_identity = legacy_bridge.get_identity(expert_id)

            enhanced.register_expert(
                expert_id=expert_id,
                description=legacy_identity.description,
                metadata=legacy_identity.metadata
            )

        enhanced.registration_count = legacy_bridge.registration_count
        enhanced.last_registration_time = legacy_bridge.last_registration_time
        enhanced.migration_count = len(legacy_bridge.get_all_experts())

        return enhanced


# Example usage and testing
if __name__ == "__main__":
    print("=== Enhanced Expert Identity Bridge ===\n")

    # Create bridge
    bridge = EnhancedExpertIdentityBridge(
        instance="thinker",
        network="testnet"
    )

    # Register experts
    lct1 = bridge.register_expert(
        42,
        description="Code generation expert",
        trust_threshold=0.75
    )
    print(f"Expert 42 registered: {lct1}")

    lct2 = bridge.register_expert(
        99,
        description="Reasoning expert",
        capabilities=["reasoning", "planning"],
        trust_threshold=0.85
    )
    print(f"Expert 99 registered: {lct2}")

    # Lookups
    print(f"\nLookups:")
    print(f"  Expert 42 LCT: {bridge.get_lct(42)}")
    print(f"  Expert from LCT: {bridge.get_expert_id(lct1)}")

    # Identity details
    identity = bridge.get_identity(99)
    print(f"\nExpert 99 identity:")
    print(f"  Component: {identity.component}")
    print(f"  Instance: {identity.instance}")
    print(f"  Role: {identity.role}")
    print(f"  Network: {identity.network}")
    print(f"  Trust threshold: {identity.trust_threshold}")
    print(f"  Capabilities: {identity.capabilities}")

    # All experts
    print(f"\nAll registered experts: {bridge.get_all_experts()}")

    print("\n=== Tests completed ===")
