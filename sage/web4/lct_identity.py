#!/usr/bin/env python3
"""
LCT (Linked Context Token) Identity Parsing Library

Implements the Unified LCT Identity Specification (v1.0.0) for SAGE neural systems.
Provides parsing, validation, and construction of LCT URIs.

Specification: /home/dp/ai-workspace/web4/docs/LCT_UNIFIED_IDENTITY_SPECIFICATION.md

Created: Session 63+ (2025-12-17)
Author: Legion (Autonomous Research)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import re
from urllib.parse import urlparse, parse_qs, urlencode


@dataclass
class LCTIdentity:
    """
    Parsed LCT (Linked Context Token) identity.

    Represents a structured identity conforming to the Unified LCT Specification:
        lct://{component}:{instance}:{role}@{network}

    Examples:
        lct://sage:thinker:expert_42@testnet
        lct://web4-agent:guardian:coordinator@mainnet
        lct://act-validator:node1:consensus@testnet

    Attributes:
        component: System/component name (e.g., "sage", "web4-agent")
        instance: Instance identifier (e.g., "thinker", "guardian")
        role: Role within instance (e.g., "expert_42", "coordinator")
        network: Network identifier (e.g., "mainnet", "testnet", "local")
        version: LCT format version (default: "1.0.0")
        pairing_status: Relationship status (active/pending/expired/suspended/revoked)
        trust_threshold: Minimum trust score required (0-1)
        capabilities: List of capabilities
        public_key_hash: Public key hash or DID anchor
    """

    component: str
    instance: str
    role: str
    network: str
    version: str = "1.0.0"
    pairing_status: Optional[str] = None
    trust_threshold: Optional[float] = None
    capabilities: Optional[List[str]] = None
    public_key_hash: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate field values after initialization."""
        # Validate component name
        if not re.match(r"^[a-z][a-z0-9-]*$", self.component):
            raise ValueError(
                f"Invalid component name '{self.component}': "
                "must start with letter, contain only lowercase alphanumeric + hyphens"
            )
        if len(self.component) > 32:
            raise ValueError(f"Component name too long: {len(self.component)} > 32")

        # Validate instance name
        if not re.match(r"^[a-z0-9_]+$", self.instance):
            raise ValueError(
                f"Invalid instance name '{self.instance}': "
                "must contain only lowercase alphanumeric + underscores"
            )
        if len(self.instance) > 64:
            raise ValueError(f"Instance name too long: {len(self.instance)} > 64")

        # Validate role name
        if not re.match(r"^[a-z0-9_]+$", self.role):
            raise ValueError(
                f"Invalid role name '{self.role}': "
                "must contain only lowercase alphanumeric + underscores"
            )
        if len(self.role) > 128:
            raise ValueError(f"Role name too long: {len(self.role)} > 128")

        # Validate pairing status
        if self.pairing_status is not None:
            valid_statuses = {"pending", "active", "suspended", "expired", "revoked"}
            if self.pairing_status not in valid_statuses:
                raise ValueError(
                    f"Invalid pairing_status '{self.pairing_status}': "
                    f"must be one of {valid_statuses}"
                )

        # Validate trust threshold
        if self.trust_threshold is not None:
            if not (0.0 <= self.trust_threshold <= 1.0):
                raise ValueError(
                    f"Invalid trust_threshold {self.trust_threshold}: must be in [0, 1]"
                )

    @property
    def lct_uri(self) -> str:
        """
        Reconstruct full LCT URI from components.

        Returns:
            Full LCT URI string with query parameters and fragment if present
        """
        # Base URI: lct://component:instance:role@network
        base = f"lct://{self.component}:{self.instance}:{self.role}@{self.network}"

        # Build query parameters
        params = {}

        if self.version != "1.0.0":
            params["version"] = self.version

        if self.pairing_status:
            params["pairing_status"] = self.pairing_status

        if self.trust_threshold is not None:
            params["trust_threshold"] = str(self.trust_threshold)

        if self.capabilities:
            params["capabilities"] = ",".join(self.capabilities)

        # Add custom metadata
        for key, value in self.metadata.items():
            if key not in params:  # Don't override standard params
                params[key] = value

        # Construct full URI
        uri = base

        if params:
            query_string = urlencode(params)
            uri += f"?{query_string}"

        if self.public_key_hash:
            uri += f"#{self.public_key_hash}"

        return uri

    def __str__(self) -> str:
        """String representation returns full LCT URI."""
        return self.lct_uri

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"LCTIdentity(component={self.component!r}, "
            f"instance={self.instance!r}, role={self.role!r}, "
            f"network={self.network!r})"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        result = {
            "component": self.component,
            "instance": self.instance,
            "role": self.role,
            "network": self.network,
            "version": self.version,
            "lct_uri": self.lct_uri,
        }

        if self.pairing_status:
            result["pairing_status"] = self.pairing_status
        if self.trust_threshold is not None:
            result["trust_threshold"] = self.trust_threshold
        if self.capabilities:
            result["capabilities"] = self.capabilities
        if self.public_key_hash:
            result["public_key_hash"] = self.public_key_hash
        if self.metadata:
            result["metadata"] = self.metadata

        return result


def parse_lct_uri(lct_uri: str) -> LCTIdentity:
    """
    Parse LCT URI into structured identity.

    Args:
        lct_uri: LCT URI string (e.g., "lct://sage:thinker:expert_42@testnet")

    Returns:
        LCTIdentity object with parsed fields

    Raises:
        ValueError: If URI format is invalid

    Examples:
        >>> lct = parse_lct_uri("lct://sage:thinker:expert_42@testnet")
        >>> lct.component
        'sage'
        >>> lct.role
        'expert_42'

        >>> lct = parse_lct_uri("lct://sage:thinker:expert_42@testnet?trust_threshold=0.75")
        >>> lct.trust_threshold
        0.75
    """
    # Validate scheme
    if not lct_uri.startswith("lct://"):
        raise ValueError(f"Invalid LCT URI scheme: must start with 'lct://', got {lct_uri}")

    # Parse using urllib
    parsed = urlparse(lct_uri)

    # Extract authority (component:instance:role@network)
    authority = parsed.netloc
    path = parsed.path.lstrip("/")

    # Combine netloc and path for parsing
    # (some parsers put everything after // in netloc, some split at /)
    full_authority = authority + "/" + path if path else authority

    # Pattern: component:instance:role@network
    pattern = r"^([a-z][a-z0-9-]*):([a-z0-9_]+):([a-z0-9_]+)@([a-z0-9_-]+)"
    match = re.match(pattern, full_authority)

    if not match:
        raise ValueError(
            f"Invalid LCT authority format: {full_authority}\n"
            f"Expected: component:instance:role@network\n"
            f"Example: sage:thinker:expert_42@testnet"
        )

    component, instance, role, network = match.groups()

    # Parse query parameters
    query_params = parse_qs(parsed.query)

    # Extract standard parameters
    version = query_params.get("version", ["1.0.0"])[0]

    pairing_status_list = query_params.get("pairing_status", [None])
    pairing_status = pairing_status_list[0] if pairing_status_list[0] else None

    trust_threshold_str = query_params.get("trust_threshold", [None])[0]
    trust_threshold = float(trust_threshold_str) if trust_threshold_str else None

    capabilities_str = query_params.get("capabilities", [None])[0]
    capabilities = capabilities_str.split(",") if capabilities_str else None

    # Extract custom metadata (non-standard query params)
    metadata = {}
    standard_params = {"version", "pairing_status", "trust_threshold", "capabilities"}
    for key, values in query_params.items():
        if key not in standard_params and values:
            metadata[key] = values[0]

    # Parse fragment (public key hash)
    public_key_hash = parsed.fragment if parsed.fragment else None

    return LCTIdentity(
        component=component,
        instance=instance,
        role=role,
        network=network,
        version=version,
        pairing_status=pairing_status,
        trust_threshold=trust_threshold,
        capabilities=capabilities,
        public_key_hash=public_key_hash,
        metadata=metadata,
    )


def validate_lct_uri(lct_uri: str) -> bool:
    """
    Validate LCT URI format without raising exceptions.

    Args:
        lct_uri: LCT URI string to validate

    Returns:
        True if valid, False otherwise

    Examples:
        >>> validate_lct_uri("lct://sage:thinker:expert_42@testnet")
        True
        >>> validate_lct_uri("https://invalid.com")
        False
    """
    try:
        parse_lct_uri(lct_uri)
        return True
    except (ValueError, Exception):
        return False


def construct_lct_uri(
    component: str,
    instance: str,
    role: str,
    network: str = "testnet",
    **kwargs
) -> str:
    """
    Construct LCT URI from components.

    Args:
        component: System/component name
        instance: Instance identifier
        role: Role within instance
        network: Network identifier (default: "testnet")
        **kwargs: Optional parameters (pairing_status, trust_threshold, capabilities, etc.)

    Returns:
        Full LCT URI string

    Examples:
        >>> construct_lct_uri("sage", "thinker", "expert_42")
        'lct://sage:thinker:expert_42@testnet'

        >>> construct_lct_uri("sage", "thinker", "expert_42",
        ...                   network="mainnet",
        ...                   trust_threshold=0.75)
        'lct://sage:thinker:expert_42@mainnet?trust_threshold=0.75'
    """
    lct = LCTIdentity(
        component=component,
        instance=instance,
        role=role,
        network=network,
        **kwargs
    )
    return lct.lct_uri


def migrate_legacy_expert_id(legacy_id: str, network: str = "testnet") -> str:
    """
    Convert legacy SAGE expert ID to LCT URI.

    Legacy format: sage_thinker_expert_42
    New format: lct://sage:thinker:expert_42@testnet

    Args:
        legacy_id: Legacy expert ID string
        network: Network to use in URI (default: "testnet")

    Returns:
        LCT URI string

    Raises:
        ValueError: If legacy ID format is invalid

    Examples:
        >>> migrate_legacy_expert_id("sage_thinker_expert_42")
        'lct://sage:thinker:expert_42@testnet'

        >>> migrate_legacy_expert_id("sage_dreamer_expert_123", network="mainnet")
        'lct://sage:dreamer:expert_123@mainnet'
    """
    # Parse: sage_thinker_expert_42 or sage_thinker_coordinator
    parts = legacy_id.split("_")

    if len(parts) < 3:
        raise ValueError(
            f"Invalid legacy expert ID: {legacy_id}\n"
            f"Expected format: component_instance_role or component_instance_roletype_id"
        )

    component = parts[0]  # sage
    instance = parts[1]   # thinker

    # Handle both formats:
    # - sage_thinker_coordinator (3 parts)
    # - sage_thinker_expert_42 (4 parts)
    if len(parts) == 3:
        role = parts[2]  # coordinator
    else:
        # expert_42, guard_5, etc.
        role = "_".join(parts[2:])

    return construct_lct_uri(component, instance, role, network=network)


# Convenience functions for SAGE integration

def sage_expert_to_lct(
    expert_id: int,
    instance: str = "thinker",
    network: str = "testnet"
) -> str:
    """
    Convert SAGE expert ID to LCT URI.

    Args:
        expert_id: Expert number (e.g., 42)
        instance: SAGE instance name (default: "thinker")
        network: Network identifier (default: "testnet")

    Returns:
        LCT URI string

    Examples:
        >>> sage_expert_to_lct(42)
        'lct://sage:thinker:expert_42@testnet'

        >>> sage_expert_to_lct(123, instance="dreamer", network="mainnet")
        'lct://sage:dreamer:expert_123@mainnet'
    """
    return construct_lct_uri(
        component="sage",
        instance=instance,
        role=f"expert_{expert_id}",
        network=network
    )


def lct_to_sage_expert(lct_uri: str) -> Optional[int]:
    """
    Extract SAGE expert ID from LCT URI.

    Args:
        lct_uri: LCT URI string

    Returns:
        Expert ID (int) if valid SAGE expert LCT, None otherwise

    Examples:
        >>> lct_to_sage_expert("lct://sage:thinker:expert_42@testnet")
        42

        >>> lct_to_sage_expert("lct://web4-agent:guardian:coordinator@mainnet")
        None
    """
    try:
        lct = parse_lct_uri(lct_uri)

        # Validate it's a SAGE component
        if lct.component != "sage":
            return None

        # Extract expert ID from role (e.g., "expert_42" → 42)
        match = re.match(r"^expert_(\d+)$", lct.role)
        if not match:
            return None

        return int(match.group(1))

    except (ValueError, Exception):
        return None


# Test/validation utilities

def get_test_vectors() -> List[Dict]:
    """
    Return test vectors for LCT URI parsing validation.

    Returns:
        List of test cases with uri and expected parsed fields
    """
    return [
        {
            "uri": "lct://sage:thinker:expert_42@testnet",
            "parsed": {
                "component": "sage",
                "instance": "thinker",
                "role": "expert_42",
                "network": "testnet",
                "version": "1.0.0",
            }
        },
        {
            "uri": "lct://web4-agent:guardian:coordinator@mainnet?pairing_status=active&trust_threshold=0.75",
            "parsed": {
                "component": "web4-agent",
                "instance": "guardian",
                "role": "coordinator",
                "network": "mainnet",
                "pairing_status": "active",
                "trust_threshold": 0.75,
            }
        },
        {
            "uri": "lct://act-validator:node1:consensus@testnet?version=1.0.0#did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
            "parsed": {
                "component": "act-validator",
                "instance": "node1",
                "role": "consensus",
                "network": "testnet",
                "version": "1.0.0",
                "public_key_hash": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
            }
        },
        {
            "uri": "lct://sage:dreamer:expert_123@mainnet?capabilities=text-generation,code-completion",
            "parsed": {
                "component": "sage",
                "instance": "dreamer",
                "role": "expert_123",
                "network": "mainnet",
                "capabilities": ["text-generation", "code-completion"],
            }
        },
    ]


if __name__ == "__main__":
    # Example usage and validation
    print("=== LCT Identity Parsing Library ===\n")

    # Example 1: Parse simple LCT URI
    uri1 = "lct://sage:thinker:expert_42@testnet"
    lct1 = parse_lct_uri(uri1)
    print(f"Example 1: {uri1}")
    print(f"  Component: {lct1.component}")
    print(f"  Instance: {lct1.instance}")
    print(f"  Role: {lct1.role}")
    print(f"  Network: {lct1.network}")
    print()

    # Example 2: Parse with query parameters
    uri2 = "lct://web4-agent:guardian:coordinator@mainnet?pairing_status=active&trust_threshold=0.75"
    lct2 = parse_lct_uri(uri2)
    print(f"Example 2: {uri2}")
    print(f"  Pairing Status: {lct2.pairing_status}")
    print(f"  Trust Threshold: {lct2.trust_threshold}")
    print()

    # Example 3: Construct from components
    uri3 = construct_lct_uri("sage", "dreamer", "expert_123",
                             network="mainnet",
                             trust_threshold=0.85,
                             capabilities=["text-generation", "reasoning"])
    print(f"Example 3: Constructed URI")
    print(f"  {uri3}")
    print()

    # Example 4: SAGE expert conversion
    expert_uri = sage_expert_to_lct(42, instance="thinker")
    expert_id = lct_to_sage_expert(expert_uri)
    print(f"Example 4: SAGE Expert Conversion")
    print(f"  Expert 42 → {expert_uri}")
    print(f"  {expert_uri} → Expert {expert_id}")
    print()

    # Example 5: Legacy migration
    legacy_id = "sage_thinker_expert_99"
    migrated_uri = migrate_legacy_expert_id(legacy_id)
    print(f"Example 5: Legacy Migration")
    print(f"  {legacy_id} → {migrated_uri}")
    print()

    # Validation
    print("=== Test Vectors ===")
    for i, test_case in enumerate(get_test_vectors(), 1):
        uri = test_case["uri"]
        expected = test_case["parsed"]
        lct = parse_lct_uri(uri)

        print(f"\nTest {i}: {uri}")
        for key, expected_value in expected.items():
            actual_value = getattr(lct, key)
            match = "✓" if actual_value == expected_value else "✗"
            print(f"  {match} {key}: {actual_value}")

    print("\n=== All examples completed ===")
