#!/usr/bin/env python3
"""
Expert Identity Bridge - Web4 LCT Integration

Maps SAGE expert IDs to Web4 LCT (Lightweight Cryptographic Token) identities,
enabling experts to participate in the Web4 ecosystem with:
- Unique cryptographic identities
- ATP balance tracking
- Reputation/trust scores
- Authorization via ACT system

Design Philosophy:
- Bidirectional mapping (expert_id ↔ lct_id)
- Namespace isolation (different SAGE instances)
- Registry persistence (save/load mappings)
- Validation (prevent duplicate registrations)

Web4 Pattern: LCT Identity
- Every entity in Web4 has an LCT identity
- Format: lct://namespace/entity_type/entity_id
- Example: lct://sage/expert/42 (expert 42 in SAGE)

Integration Points:
- ExpertReputationDB → Web4 trust tensor (via LCT)
- TrustBasedExpertSelector → ATP resource allocator (via LCT)
- Cache allocation → Web4 authorization (via LCT)

Created: Session 59 (2025-12-16)
Part of: Web4 ↔ SAGE integration (Session 57 design)
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
import hashlib


@dataclass
class ExpertIdentity:
    """Web4 identity for a SAGE expert."""
    expert_id: int                      # SAGE expert ID
    lct_id: str                         # Web4 LCT identifier
    namespace: str                      # SAGE instance namespace
    registration_time: float            # When registered
    description: Optional[str] = None   # Human-readable description
    metadata: Dict = field(default_factory=dict)  # Additional metadata


@dataclass
class IdentityStats:
    """Statistics about identity registrations."""
    total_experts: int
    active_experts: int
    namespaces: List[str]
    registration_rate: float  # Registrations per day (recent)
    last_registration: float


class ExpertIdentityBridge:
    """
    Maps SAGE expert IDs to Web4 LCT identities.

    Provides bidirectional lookup and registry persistence for expert identities
    in the Web4 ecosystem.

    Usage:
        bridge = ExpertIdentityBridge(namespace="sage_legion")
        lct_id = bridge.register_expert(42, description="Code generation expert")
        # lct://sage_legion/expert/42

        # Lookup
        assert bridge.get_lct(42) == lct_id
        assert bridge.get_expert_id(lct_id) == 42

        # Save/load
        bridge.save("/path/to/registry.json")
        bridge2 = ExpertIdentityBridge.load("/path/to/registry.json")
    """

    def __init__(
        self,
        namespace: str = "sage",
        registry_path: Optional[Path] = None,
        auto_save: bool = True
    ):
        """
        Initialize expert identity bridge.

        Args:
            namespace: Instance namespace (e.g., "sage_legion", "sage_thor")
            registry_path: Path to save/load registry
            auto_save: Automatically save after registrations
        """
        self.namespace = namespace
        self.registry_path = Path(registry_path) if registry_path else None
        self.auto_save = auto_save

        # Bidirectional mappings
        self.expert_to_lct: Dict[int, str] = {}
        self.lct_to_expert: Dict[str, int] = {}

        # Full identity objects
        self.identities: Dict[int, ExpertIdentity] = {}

        # Statistics
        self.registration_count = 0
        self.last_registration_time = 0.0

        # Load existing registry if path provided
        if self.registry_path and self.registry_path.exists():
            self.load(self.registry_path)

    def register_expert(
        self,
        expert_id: int,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Register expert and return LCT ID.

        Args:
            expert_id: SAGE expert ID (0-127 for typical MoE)
            description: Human-readable description
            metadata: Additional metadata

        Returns:
            LCT identifier string (e.g., "lct://sage/expert/42")

        Raises:
            ValueError: If expert already registered
        """
        if expert_id in self.expert_to_lct:
            raise ValueError(f"Expert {expert_id} already registered as {self.expert_to_lct[expert_id]}")

        # Generate LCT ID
        lct_id = self._generate_lct_id(expert_id)

        # Create identity
        identity = ExpertIdentity(
            expert_id=expert_id,
            lct_id=lct_id,
            namespace=self.namespace,
            registration_time=time.time(),
            description=description,
            metadata=metadata or {}
        )

        # Register bidirectional mappings
        self.expert_to_lct[expert_id] = lct_id
        self.lct_to_expert[lct_id] = expert_id
        self.identities[expert_id] = identity

        # Update statistics
        self.registration_count += 1
        self.last_registration_time = time.time()

        # Auto-save if enabled
        if self.auto_save and self.registry_path:
            self.save(self.registry_path)

        return lct_id

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
            Dict mapping expert_id → lct_id
        """
        results = {}
        descriptions = descriptions or {}

        for expert_id in expert_ids:
            if expert_id in self.expert_to_lct:
                # Already registered, skip
                results[expert_id] = self.expert_to_lct[expert_id]
            else:
                description = descriptions.get(expert_id)
                lct_id = self.register_expert(expert_id, description=description)
                results[expert_id] = lct_id

        return results

    def get_lct(self, expert_id: int) -> Optional[str]:
        """
        Get LCT ID for expert.

        Args:
            expert_id: SAGE expert ID

        Returns:
            LCT identifier or None if not registered
        """
        return self.expert_to_lct.get(expert_id)

    def get_expert_id(self, lct_id: str) -> Optional[int]:
        """
        Get expert ID from LCT.

        Args:
            lct_id: LCT identifier

        Returns:
            Expert ID or None if not found
        """
        return self.lct_to_expert.get(lct_id)

    def get_identity(self, expert_id: int) -> Optional[ExpertIdentity]:
        """
        Get full identity object for expert.

        Args:
            expert_id: SAGE expert ID

        Returns:
            ExpertIdentity or None if not registered
        """
        return self.identities.get(expert_id)

    def is_registered(self, expert_id: int) -> bool:
        """Check if expert is registered."""
        return expert_id in self.expert_to_lct

    def get_all_experts(self) -> List[int]:
        """Get list of all registered expert IDs."""
        return list(self.expert_to_lct.keys())

    def get_all_lct_ids(self) -> List[str]:
        """Get list of all registered LCT IDs."""
        return list(self.lct_to_expert.keys())

    def update_metadata(
        self,
        expert_id: int,
        metadata: Dict,
        merge: bool = True
    ) -> None:
        """
        Update expert metadata.

        Args:
            expert_id: Expert to update
            metadata: New metadata
            merge: If True, merge with existing; if False, replace
        """
        if expert_id not in self.identities:
            raise ValueError(f"Expert {expert_id} not registered")

        identity = self.identities[expert_id]

        if merge:
            identity.metadata.update(metadata)
        else:
            identity.metadata = metadata.copy()

        # Auto-save if enabled
        if self.auto_save and self.registry_path:
            self.save(self.registry_path)

    def get_statistics(self) -> IdentityStats:
        """
        Get identity registry statistics.

        Returns:
            IdentityStats with registration metrics
        """
        # Count active (recently used) experts
        # For now, all registered are considered active
        # Could extend with usage tracking

        # Compute registration rate (recent)
        # Simplified: use total registrations and time since first
        if self.registration_count > 0 and self.identities:
            first_reg = min(id.registration_time for id in self.identities.values())
            time_span = time.time() - first_reg
            rate = self.registration_count / max(time_span / 86400, 1.0)  # Per day
        else:
            rate = 0.0

        return IdentityStats(
            total_experts=len(self.identities),
            active_experts=len(self.identities),  # TODO: Track actual usage
            namespaces=[self.namespace],
            registration_rate=rate,
            last_registration=self.last_registration_time
        )

    def validate_registry(self) -> bool:
        """
        Validate registry consistency.

        Checks:
        - Bidirectional mapping consistency
        - No duplicate LCT IDs
        - All identities have valid expert_ids

        Returns:
            True if valid, False otherwise
        """
        # Check bidirectional consistency
        for expert_id, lct_id in self.expert_to_lct.items():
            if self.lct_to_expert.get(lct_id) != expert_id:
                return False

        for lct_id, expert_id in self.lct_to_expert.items():
            if self.expert_to_lct.get(expert_id) != lct_id:
                return False

        # Check all identities exist
        for expert_id in self.expert_to_lct.keys():
            if expert_id not in self.identities:
                return False

        # Check for duplicate LCT IDs (shouldn't happen, but validate)
        if len(set(self.expert_to_lct.values())) != len(self.expert_to_lct):
            return False

        return True

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save identity registry to disk.

        Args:
            path: Path to save (defaults to self.registry_path)
        """
        if path is None:
            if self.registry_path is None:
                raise ValueError("No registry path specified")
            path = self.registry_path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize registry
        data = {
            'namespace': self.namespace,
            'registration_count': self.registration_count,
            'last_registration_time': self.last_registration_time,
            'identities': {
                str(expert_id): asdict(identity)
                for expert_id, identity in self.identities.items()
            }
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ExpertIdentityBridge':
        """
        Load identity registry from disk.

        Args:
            path: Path to load

        Returns:
            ExpertIdentityBridge with loaded registry
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Registry not found at {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        # Create bridge WITHOUT registry_path to avoid recursive load
        bridge = cls(
            namespace=data['namespace'],
            registry_path=None,  # Avoid triggering load in __init__
            auto_save=False
        )

        # Set registry_path after creation
        bridge.registry_path = path

        # Restore statistics
        bridge.registration_count = data['registration_count']
        bridge.last_registration_time = data['last_registration_time']

        # Restore identities
        for expert_id_str, identity_dict in data['identities'].items():
            expert_id = int(expert_id_str)
            identity = ExpertIdentity(**identity_dict)

            bridge.expert_to_lct[expert_id] = identity.lct_id
            bridge.lct_to_expert[identity.lct_id] = expert_id
            bridge.identities[expert_id] = identity

        # Re-enable auto-save
        bridge.auto_save = True

        return bridge

    def _generate_lct_id(self, expert_id: int) -> str:
        """
        Generate LCT identifier for expert.

        Format: lct://namespace/expert/expert_id

        Args:
            expert_id: SAGE expert ID

        Returns:
            LCT identifier string
        """
        return f"lct://{self.namespace}/expert/{expert_id}"

    def _verify_lct_format(self, lct_id: str) -> bool:
        """
        Verify LCT ID has correct format.

        Args:
            lct_id: LCT identifier to check

        Returns:
            True if valid format, False otherwise
        """
        # Expected format: lct://namespace/expert/expert_id
        if not lct_id.startswith("lct://"):
            return False

        parts = lct_id[6:].split('/')  # Remove "lct://"
        if len(parts) != 3:
            return False

        namespace, entity_type, entity_id = parts

        if entity_type != "expert":
            return False

        try:
            int(entity_id)
        except ValueError:
            return False

        return True


# Convenience functions

def create_identity_bridge(
    namespace: str = "sage",
    registry_path: Optional[Path] = None
) -> ExpertIdentityBridge:
    """
    Create expert identity bridge with default settings.

    Args:
        namespace: SAGE instance namespace
        registry_path: Path for registry persistence

    Returns:
        ExpertIdentityBridge instance
    """
    return ExpertIdentityBridge(
        namespace=namespace,
        registry_path=registry_path
    )


def register_expert_with_lct(
    bridge: ExpertIdentityBridge,
    expert_id: int,
    description: Optional[str] = None
) -> str:
    """
    Register expert and return LCT ID.

    Convenience wrapper for bridge.register_expert().

    Args:
        bridge: Identity bridge
        expert_id: Expert to register
        description: Optional description

    Returns:
        LCT identifier string
    """
    return bridge.register_expert(expert_id, description=description)


def lookup_expert_lct(
    bridge: ExpertIdentityBridge,
    expert_id: int
) -> Optional[str]:
    """
    Lookup LCT ID for expert.

    Args:
        bridge: Identity bridge
        expert_id: Expert to lookup

    Returns:
        LCT ID or None if not registered
    """
    return bridge.get_lct(expert_id)
