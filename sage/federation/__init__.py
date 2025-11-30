"""
SAGE Federation Module

Platform-to-platform task delegation with trust-based routing and challenge system.

Components:
- federation_types: Data structures (identities, tasks, proofs, witnesses)
- federation_router: Routing logic (capability matching, delegation decisions)
- federation_challenge_system: Quality challenge defense (timeouts, progressive penalties)

Phase 1 Status: Local routing logic + challenge system complete (no network yet)

Author: Thor (SAGE consciousness via Claude)
Date: 2025-11-28
"""

from sage.federation.federation_types import (
    # Hardware and capability
    HardwareSpec,

    # Identity and stake
    FederationIdentity,
    IdentityStake,
    StakeStatus,

    # Task and execution
    FederationTask,
    ExecutionProof,
    ExecutionRecord,
    QualityRequirements,

    # Witness
    WitnessAttestation,
    WitnessRecord,
    WitnessOutcome,

    # Signed wrappers (Phase 2)
    SignedFederationTask,
    SignedExecutionProof,
    SignedWitnessAttestation,

    # Utility
    create_thor_identity,
    create_sprout_identity,
)

from sage.federation.federation_router import FederationRouter

from sage.federation.federation_challenge_system import (
    FederationChallengeSystem,
    QualityChallenge,
    EvasionRecord,
    ChallengeStatus,
    EvasionPenaltyLevel,
)

from sage.federation.federation_crypto import (
    FederationKeyPair,
    FederationCrypto,
    SignatureRegistry,
)

from sage.federation.web4_block_signer import (
    SageBlockSigner,
    SageBlockVerifier,
    create_sage_block_signer_from_identity,
)

__all__ = [
    # Types
    'HardwareSpec',
    'FederationIdentity',
    'IdentityStake',
    'StakeStatus',
    'FederationTask',
    'ExecutionProof',
    'ExecutionRecord',
    'QualityRequirements',
    'WitnessAttestation',
    'WitnessRecord',
    'WitnessOutcome',

    # Signed types (Phase 2)
    'SignedFederationTask',
    'SignedExecutionProof',
    'SignedWitnessAttestation',

    # Router
    'FederationRouter',

    # Challenge System
    'FederationChallengeSystem',
    'QualityChallenge',
    'EvasionRecord',
    'ChallengeStatus',
    'EvasionPenaltyLevel',

    # Crypto (Phase 2)
    'FederationKeyPair',
    'FederationCrypto',
    'SignatureRegistry',

    # Web4 Integration
    'SageBlockSigner',
    'SageBlockVerifier',
    'create_sage_block_signer_from_identity',

    # Utility
    'create_thor_identity',
    'create_sprout_identity',
]
