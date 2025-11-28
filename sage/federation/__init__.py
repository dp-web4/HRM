"""
SAGE Federation Module

Platform-to-platform task delegation with trust-based routing.

Components:
- federation_types: Data structures (identities, tasks, proofs, witnesses)
- federation_router: Routing logic (capability matching, delegation decisions)

Phase 1 Status: Local routing logic complete (no network yet)

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

    # Utility
    create_thor_identity,
    create_sprout_identity,
)

from sage.federation.federation_router import FederationRouter

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

    # Router
    'FederationRouter',

    # Utility
    'create_thor_identity',
    'create_sprout_identity',
]
