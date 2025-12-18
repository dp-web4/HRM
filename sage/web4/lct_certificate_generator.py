#!/usr/bin/env python3
"""
LCT Certificate Generator for SAGE Experts

Generates full Web4-compliant LCT certificates for SAGE experts,
including birth certificates, MRH, and T3/V3 tensors.

Created: Session 63+ (2025-12-17)
Author: Legion (Autonomous Research)
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from pathlib import Path

from lct_identity import (
    LCTIdentity,
    parse_lct_uri,
    construct_lct_uri,
    sage_expert_to_lct,
)


@dataclass
class LCTBinding:
    """Cryptographic binding for LCT identity."""
    entity_type: str
    public_key: Optional[str] = None
    hardware_anchor: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    binding_proof: Optional[str] = None


@dataclass
class BirthCertificate:
    """Birth certificate for SAGE expert LCT."""
    issuing_society: str  # LCT URI of SAGE collective
    citizen_role: str  # LCT URI of expert role
    birth_timestamp: str
    birth_witnesses: List[str]  # List of witness LCT URIs
    genesis_block_hash: Optional[str] = None
    birth_context: str = "neural_moe_expert"
    rights: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    initial_atp: int = 100


@dataclass
class MRHRelationship:
    """Single relationship in Markov Relevancy Horizon."""
    lct_id: str
    relationship_type: str  # For bound: parent/child/sibling, for paired: operational/birth_certificate
    permanent: bool = False
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    context: Optional[str] = None


@dataclass
class MRH:
    """Markov Relevancy Horizon."""
    bound: List[MRHRelationship] = field(default_factory=list)
    paired: List[MRHRelationship] = field(default_factory=list)
    witnessing: List[Dict[str, Any]] = field(default_factory=list)
    broadcast: List[str] = field(default_factory=list)
    horizon_depth: int = 1
    fractal_depth: int = 0
    context_radius: int = 1
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class T3Tensor:
    """Trust Tensor (T3) with multi-dimensional trust scores."""
    dimensions: Dict[str, float]
    composite_score: float
    last_computed: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    computation_witnesses: List[str] = field(default_factory=list)


@dataclass
class V3Tensor:
    """Value Tensor (V3) with multi-dimensional value scores."""
    dimensions: Dict[str, Any]  # Mixed int (energy_balance) and float values
    composite_score: float
    last_computed: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    computation_witnesses: List[str] = field(default_factory=list)


@dataclass
class Attestation:
    """Witness attestation for LCT event."""
    witness: str  # Witness LCT URI or DID
    attestation_type: str  # existence, action, state, quality
    claims: Dict[str, Any]
    signature: Optional[str] = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class FullLCTCertificate:
    """Full Web4-compliant LCT certificate."""
    lct_id: str
    subject: str  # DID
    uri_reference: str  # Unified LCT URI for quick reference
    binding: LCTBinding
    birth_certificate: Optional[BirthCertificate] = None
    mrh: MRH = field(default_factory=MRH)
    policy: Dict[str, Any] = field(default_factory=dict)
    t3_tensor: Optional[T3Tensor] = None
    v3_tensor: Optional[V3Tensor] = None
    attestations: List[Attestation] = field(default_factory=list)
    lineage: List[Dict[str, Any]] = field(default_factory=list)
    revocation: Dict[str, Any] = field(default_factory=lambda: {"status": "active", "ts": None, "reason": None})
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SAGELCTCertificateGenerator:
    """Generator for SAGE expert LCT certificates."""

    def __init__(
        self,
        instance: str = "thinker",
        network: str = "testnet",
        coordinator_uri: Optional[str] = None
    ):
        """
        Initialize SAGE LCT certificate generator.

        Args:
            instance: SAGE instance name (e.g., "thinker", "dreamer")
            network: Network identifier (e.g., "testnet", "mainnet")
            coordinator_uri: LCT URI of coordinator (issuing society)
        """
        self.instance = instance
        self.network = network

        # Default coordinator URI if not provided
        if coordinator_uri is None:
            self.coordinator_uri = construct_lct_uri(
                component="sage",
                instance=instance,
                role="coordinator",
                network=network
            )
        else:
            self.coordinator_uri = coordinator_uri

    def generate_expert_certificate(
        self,
        expert_id: int,
        public_key: Optional[str] = None,
        initial_trust_score: float = 0.5,
        initial_atp: int = 100,
        witness_uris: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None
    ) -> FullLCTCertificate:
        """
        Generate full LCT certificate for SAGE expert.

        Args:
            expert_id: Expert numeric ID
            public_key: Optional public key (hex or base64)
            initial_trust_score: Initial composite trust score (0.0-1.0)
            initial_atp: Initial ATP allocation
            witness_uris: List of witness LCT URIs for birth certificate
            capabilities: List of expert capabilities

        Returns:
            Full LCT certificate
        """
        # Generate Unified LCT URI reference
        uri_reference = sage_expert_to_lct(expert_id, self.instance, self.network)

        # Parse URI to get components
        lct_parsed = parse_lct_uri(uri_reference)

        # Generate LCT ID (multibase32 hash of URI + timestamp for uniqueness)
        lct_id_input = f"{uri_reference}:{time.time()}"
        lct_hash = hashlib.sha256(lct_id_input.encode()).hexdigest()[:16]
        lct_id = f"lct:web4:sage:{self.instance}:expert_{expert_id}:{lct_hash}"

        # Generate subject DID
        if public_key:
            # Use public key for DID
            subject = f"did:web4:key:{public_key[:16]}"
        else:
            # Generate placeholder DID
            did_hash = hashlib.sha256(f"expert_{expert_id}".encode()).hexdigest()[:16]
            subject = f"did:web4:sage:{self.instance}:expert_{expert_id}:{did_hash}"

        # Create binding
        binding = LCTBinding(
            entity_type="ai",
            public_key=public_key,
            hardware_anchor=None,  # SAGE experts don't have hardware anchors
            created_at=datetime.now(timezone.utc).isoformat(),
            binding_proof=None  # Would be generated with actual signing
        )

        # Create birth certificate
        if witness_uris is None:
            # Default witnesses: coordinator + expert_1 + expert_2
            witness_uris = [
                self.coordinator_uri,
                sage_expert_to_lct(1, self.instance, self.network) if expert_id != 1 else sage_expert_to_lct(2, self.instance, self.network),
                sage_expert_to_lct(2, self.instance, self.network) if expert_id > 2 else sage_expert_to_lct(3, self.instance, self.network)
            ]

        birth_certificate = BirthCertificate(
            issuing_society=self.coordinator_uri,
            citizen_role=f"lct:web4:role:sage-expert",
            birth_timestamp=datetime.now(timezone.utc).isoformat(),
            birth_witnesses=witness_uris,
            genesis_block_hash=None,  # Would be set if anchored to blockchain
            birth_context="neural_moe_expert",
            rights=[
                "inference:execute",
                "gradient:receive",
                "pairing:coordinator",
                "witness:peer-experts"
            ],
            responsibilities=[
                "respond_to_queries",
                "maintain_specialization",
                "update_from_feedback",
                "contribute_to_mixture"
            ],
            initial_atp=initial_atp
        )

        # Create MRH
        mrh = MRH(
            bound=[],  # SAGE experts don't have hardware bindings
            paired=[
                MRHRelationship(
                    lct_id=self.coordinator_uri,
                    relationship_type="operational",
                    permanent=True,
                    context="sage_coordinator_pairing"
                ),
                MRHRelationship(
                    lct_id=f"lct:web4:role:sage-expert",
                    relationship_type="birth_certificate",
                    permanent=True,
                    context="birth_certificate_role"
                )
            ],
            witnessing=[],  # Populated as expert operates
            broadcast=[],   # Populated when expert broadcasts
            horizon_depth=1,
            fractal_depth=0,
            context_radius=1
        )

        # Create T3 tensor (trust)
        t3_tensor = T3Tensor(
            dimensions={
                "technical_competence": initial_trust_score,
                "social_reliability": initial_trust_score * 0.9,
                "temporal_consistency": initial_trust_score * 0.8,
                "witness_count": 0.3,  # Low for new expert
                "lineage_depth": 0.1,  # New expert
                "context_alignment": initial_trust_score
            },
            composite_score=initial_trust_score,
            computation_witnesses=[self.coordinator_uri]
        )

        # Create V3 tensor (value)
        v3_tensor = V3Tensor(
            dimensions={
                "energy_balance": initial_atp,
                "contribution_history": 0.0,  # New expert
                "resource_stewardship": 0.5,  # Neutral
                "network_effects": 0.0,  # No network effects yet
                "reputation_capital": 0.0,  # No reputation yet
                "temporal_value": 0.0  # No temporal value yet
            },
            composite_score=0.1,  # Low initial value
            computation_witnesses=[self.coordinator_uri]
        )

        # Create policy
        if capabilities is None:
            capabilities = ["text-generation", "reasoning", "moe-inference"]

        policy = {
            "capabilities": capabilities,
            "constraints": {
                "max_context_length": 8192,
                "max_generation_tokens": 2048,
                "requires_coordinator": True,
                "pairing_required": True
            },
            "permissions": {
                "read": ["self", "coordinator"],
                "write": ["self"],
                "execute": ["inference"]
            }
        }

        # Create birth attestations
        attestations = [
            Attestation(
                witness=witness_uri,
                attestation_type="existence",
                claims={
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "method": "birth_ceremony",
                    "context": "sage_expert_initialization"
                },
                signature=None  # Would be actual signature
            )
            for witness_uri in witness_uris
        ]

        # Create full certificate
        certificate = FullLCTCertificate(
            lct_id=lct_id,
            subject=subject,
            uri_reference=uri_reference,
            binding=binding,
            birth_certificate=birth_certificate,
            mrh=mrh,
            policy=policy,
            t3_tensor=t3_tensor,
            v3_tensor=v3_tensor,
            attestations=attestations,
            lineage=[],  # New expert has no lineage
            revocation={"status": "active", "ts": None, "reason": None}
        )

        return certificate

    def generate_coordinator_certificate(
        self,
        public_key: Optional[str] = None,
        initial_atp: int = 1000
    ) -> FullLCTCertificate:
        """
        Generate full LCT certificate for SAGE coordinator.

        Args:
            public_key: Optional public key
            initial_atp: Initial ATP allocation (higher for coordinator)

        Returns:
            Full LCT certificate for coordinator
        """
        # Coordinator LCT ID
        lct_id_input = f"{self.coordinator_uri}:{time.time()}"
        lct_hash = hashlib.sha256(lct_id_input.encode()).hexdigest()[:16]
        lct_id = f"lct:web4:sage:{self.instance}:coordinator:{lct_hash}"

        # Coordinator DID
        if public_key:
            subject = f"did:web4:key:{public_key[:16]}"
        else:
            did_hash = hashlib.sha256(f"coordinator_{self.instance}".encode()).hexdigest()[:16]
            subject = f"did:web4:sage:{self.instance}:coordinator:{did_hash}"

        # Binding
        binding = LCTBinding(
            entity_type="ai",
            public_key=public_key,
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # Self-issued birth certificate (coordinator bootstraps itself)
        birth_certificate = BirthCertificate(
            issuing_society=self.coordinator_uri,  # Self-issued
            citizen_role="lct:web4:role:sage-coordinator",
            birth_timestamp=datetime.now(timezone.utc).isoformat(),
            birth_witnesses=[],  # No witnesses for bootstrap
            birth_context="sage_coordinator_genesis",
            rights=[
                "expert:create",
                "expert:manage",
                "routing:control",
                "trust:evaluate",
                "atp:allocate"
            ],
            responsibilities=[
                "expert_coordination",
                "trust_computation",
                "resource_allocation",
                "system_governance"
            ],
            initial_atp=initial_atp
        )

        # MRH (coordinator has no permanent pairings initially)
        mrh = MRH(
            bound=[],
            paired=[],
            witnessing=[],
            broadcast=[],
            horizon_depth=2,  # Coordinator has larger horizon
            fractal_depth=1,
            context_radius=3
        )

        # T3 tensor (high trust for coordinator)
        t3_tensor = T3Tensor(
            dimensions={
                "technical_competence": 0.95,
                "social_reliability": 0.90,
                "temporal_consistency": 0.95,
                "witness_count": 1.0,  # Self-witnessed
                "lineage_depth": 1.0,  # Genesis
                "context_alignment": 0.95
            },
            composite_score=0.93,
            computation_witnesses=[self.coordinator_uri]  # Self-computed
        )

        # V3 tensor (high value for coordinator)
        v3_tensor = V3Tensor(
            dimensions={
                "energy_balance": initial_atp,
                "contribution_history": 1.0,  # Genesis contribution
                "resource_stewardship": 0.95,
                "network_effects": 1.0,  # Coordinator creates network
                "reputation_capital": 0.9,
                "temporal_value": 0.9
            },
            composite_score=0.88,
            computation_witnesses=[self.coordinator_uri]
        )

        # Policy (coordinator has full capabilities)
        policy = {
            "capabilities": [
                "expert-management",
                "trust-computation",
                "routing-control",
                "atp-allocation",
                "governance"
            ],
            "constraints": {
                "requires_quorum": False,  # Coordinator has authority
                "pairing_required": False
            },
            "permissions": {
                "read": ["*"],
                "write": ["*"],
                "execute": ["*"]
            }
        }

        # Self-attestation
        attestations = [
            Attestation(
                witness=self.coordinator_uri,
                attestation_type="existence",
                claims={
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "method": "genesis_bootstrap",
                    "context": "sage_coordinator_self_initialization"
                }
            )
        ]

        return FullLCTCertificate(
            lct_id=lct_id,
            subject=subject,
            uri_reference=self.coordinator_uri,
            binding=binding,
            birth_certificate=birth_certificate,
            mrh=mrh,
            policy=policy,
            t3_tensor=t3_tensor,
            v3_tensor=v3_tensor,
            attestations=attestations
        )

    def save_certificate(self, certificate: FullLCTCertificate, output_dir: Path) -> Path:
        """
        Save LCT certificate to JSON file.

        Args:
            certificate: Full LCT certificate
            output_dir: Output directory path

        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract expert ID or role from URI
        lct_parsed = parse_lct_uri(certificate.uri_reference)
        role = lct_parsed.role

        # Save to file
        filename = f"lct_cert_{self.instance}_{role}_{self.network}.json"
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(certificate.to_dict(), f, indent=2)

        return filepath


# Example usage
if __name__ == "__main__":
    print("SAGE LCT Certificate Generator - Example Usage")
    print("=" * 60)

    # Initialize generator
    generator = SAGELCTCertificateGenerator(
        instance="thinker",
        network="testnet"
    )

    print("\n1. Generating coordinator certificate...")
    coordinator_cert = generator.generate_coordinator_certificate()
    print(f"   LCT ID: {coordinator_cert.lct_id}")
    print(f"   URI Reference: {coordinator_cert.uri_reference}")
    print(f"   Trust Score: {coordinator_cert.t3_tensor.composite_score}")
    print(f"   Value Score: {coordinator_cert.v3_tensor.composite_score}")
    print(f"   ATP Balance: {coordinator_cert.v3_tensor.dimensions['energy_balance']}")

    print("\n2. Generating expert certificate for Expert 42...")
    expert_cert = generator.generate_expert_certificate(
        expert_id=42,
        initial_trust_score=0.65,
        initial_atp=150,
        capabilities=["text-generation", "reasoning", "code-analysis"]
    )
    print(f"   LCT ID: {expert_cert.lct_id}")
    print(f"   URI Reference: {expert_cert.uri_reference}")
    print(f"   Trust Score: {expert_cert.t3_tensor.composite_score}")
    print(f"   Birth Witnesses: {len(expert_cert.birth_certificate.birth_witnesses)}")
    print(f"   Capabilities: {expert_cert.policy['capabilities']}")
    print(f"   MRH Pairings: {len(expert_cert.mrh.paired)}")

    print("\n3. Example certificate structure (Expert 42):")
    cert_dict = expert_cert.to_dict()
    print(json.dumps({
        "lct_id": cert_dict["lct_id"],
        "uri_reference": cert_dict["uri_reference"],
        "subject": cert_dict["subject"],
        "entity_type": cert_dict["binding"]["entity_type"],
        "birth_certificate": {
            "issuing_society": cert_dict["birth_certificate"]["issuing_society"],
            "witnesses": len(cert_dict["birth_certificate"]["birth_witnesses"]),
            "rights": cert_dict["birth_certificate"]["rights"]
        },
        "mrh": {
            "paired": len(cert_dict["mrh"]["paired"]),
            "horizon_depth": cert_dict["mrh"]["horizon_depth"]
        },
        "t3_tensor": cert_dict["t3_tensor"]["dimensions"],
        "policy_capabilities": cert_dict["policy"]["capabilities"]
    }, indent=2))

    print("\n4. Saving certificates to disk...")
    output_dir = Path(__file__).parent.parent / "lct_certificates"

    coord_path = generator.save_certificate(coordinator_cert, output_dir)
    print(f"   ✓ Coordinator: {coord_path}")

    expert_path = generator.save_certificate(expert_cert, output_dir)
    print(f"   ✓ Expert 42: {expert_path}")

    print("\n✓ LCT Certificate Generator test complete!")
    print(f"\nCertificates saved to: {output_dir}")
