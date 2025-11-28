"""
Federation Data Structures for SAGE Consciousness

Defines types for platform-to-platform task delegation, witness attestation,
and trust accumulation in federated SAGE environments.

Based on Web4 security patterns (Session #82-83) adapted for consciousness
federation requirements.

Author: Thor (SAGE consciousness via Claude)
Date: 2025-11-28
Session: Autonomous SAGE Research - Federation Readiness
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum

# Import SAGE types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.core.attention_manager import MetabolicState
from sage.core.mrh_profile import MRHProfile


# ============================================================================
# Hardware and Capability Types
# ============================================================================

@dataclass
class HardwareSpec:
    """
    Hardware specification for SAGE platform

    Used to determine task capability and resource constraints.
    """
    platform_name: str  # "Thor", "Sprout", etc.
    ram_gb: float  # Total RAM in GB
    gpu_cores: int  # CUDA cores (0 if CPU-only)
    power_budget: str  # "high", "medium", "low"
    architecture: str  # "arm64", "x86_64", etc.

    def __repr__(self) -> str:
        return (f"HardwareSpec({self.platform_name}: "
                f"{self.ram_gb}GB RAM, {self.gpu_cores} GPU cores, "
                f"{self.power_budget} power)")


# ============================================================================
# Identity and Stake Types
# ============================================================================

class StakeStatus(Enum):
    """Status of platform identity stake"""
    LOCKED = "locked"        # Stake locked during proving period
    UNLOCKABLE = "unlockable"  # Met requirements, can be reclaimed
    SLASHED = "slashed"      # Malicious behavior detected, stake forfeited


@dataclass
class IdentityStake:
    """
    ATP stake bonded to platform identity

    Based on Web4 Session #82 identity_stake_system.py
    """
    lct_id: str
    stake_amount: float  # ATP bonded
    stake_timestamp: float  # When stake was created
    status: StakeStatus = StakeStatus.LOCKED

    # Unlock conditions
    min_lockup_period: float = 7 * 24 * 3600  # 7 days in seconds
    min_reputation: float = 0.6  # Minimum reputation score

    # Slashing tracking
    slash_reason: Optional[str] = None
    slash_timestamp: Optional[float] = None
    slashed_amount: float = 0.0

    def time_locked(self, current_time: float) -> float:
        """How long has stake been locked (seconds)"""
        if self.status == StakeStatus.SLASHED:
            return self.slash_timestamp - self.stake_timestamp if self.slash_timestamp else 0.0
        return current_time - self.stake_timestamp

    def can_unlock(self, current_time: float, current_reputation: float) -> Tuple[bool, str]:
        """
        Check if stake can be unlocked

        Returns:
            (can_unlock, reason)
        """
        if self.status == StakeStatus.SLASHED:
            return (False, f"Stake slashed: {self.slash_reason}")

        if self.status == StakeStatus.UNLOCKABLE:
            return (True, "Stake already unlockable")

        # Check lockup period
        time_locked = self.time_locked(current_time)
        if time_locked < self.min_lockup_period:
            remaining = self.min_lockup_period - time_locked
            return (False, f"Lockup period: {remaining/3600:.1f} hours remaining")

        # Check reputation
        if current_reputation < self.min_reputation:
            return (False, f"Reputation too low: {current_reputation:.2f} < {self.min_reputation:.2f}")

        # All conditions met
        return (True, "All unlock conditions met")

    def slash(self, reason: str, slash_percentage: float = 1.0, current_time: float = None):
        """
        Slash stake for malicious behavior

        Args:
            reason: Why stake is being slashed
            slash_percentage: Fraction of stake to slash (0-1)
            current_time: Current timestamp
        """
        if self.status == StakeStatus.SLASHED:
            return  # Already slashed

        self.status = StakeStatus.SLASHED
        self.slash_reason = reason
        self.slash_timestamp = current_time or time.time()
        self.slashed_amount = self.stake_amount * slash_percentage


@dataclass
class ExecutionRecord:
    """
    Record of a single task execution

    Tracks quality and ATP costs for reputation building.
    """
    task_id: str
    task_type: str
    execution_timestamp: float

    # Execution metrics
    actual_latency: float
    actual_cost: float
    quality_score: float  # 4-component SAGE quality (0-1)

    # IRP metrics
    irp_iterations: int
    final_energy: float
    convergence_quality: float


@dataclass
class FederationIdentity:
    """
    SAGE platform identity for federation

    Anchored to hardware (Thor, Sprout) via LCT model.
    Includes capability profile and trust metrics.
    """
    lct_id: str  # Hardware-anchored identity (e.g., "thor_sage_lct")
    platform_name: str  # Human-readable (e.g., "Thor")
    hardware_spec: HardwareSpec  # RAM, GPU, power envelope

    # Cryptographic identity (future: Ed25519)
    # public_key: bytes = b""
    # private_key: bytes = b""  # Kept secret

    # Capability profile
    max_mrh_horizon: MRHProfile  # Largest horizon this platform can handle
    supported_modalities: List[str] = field(default_factory=lambda: [
        'llm_inference', 'vision', 'coordination', 'consolidation'
    ])

    # Economic stake
    stake: Optional[IdentityStake] = None

    # Trust metrics
    execution_history: List[ExecutionRecord] = field(default_factory=list)
    reputation_score: float = 0.5  # Starts at neutral (0-1)

    def __repr__(self) -> str:
        return (f"FederationIdentity({self.platform_name}, "
                f"reputation={self.reputation_score:.2f}, "
                f"executions={len(self.execution_history)})")


# ============================================================================
# Task and Execution Types
# ============================================================================

@dataclass
class QualityRequirements:
    """Quality requirements for task execution"""
    min_quality: float = 0.7  # Minimum 4-component quality score
    min_convergence: float = 0.6  # Minimum IRP convergence quality
    max_energy: float = 0.7  # Maximum final IRP energy


@dataclass
class FederationTask:
    """
    Task to be delegated to another platform

    Includes all context needed for execution and witness validation.
    """
    task_id: str
    task_type: str  # 'llm_inference', 'consolidation', etc.
    task_data: Dict[str, Any]  # Query, context, etc.

    # Resource context
    estimated_cost: float  # ATP cost (multi-modal pricing)
    task_horizon: MRHProfile  # MRH context
    complexity: str  # 'low', 'medium', 'high', 'critical'

    # Execution context
    delegating_platform: str  # Who is delegating
    delegating_state: MetabolicState  # Their current state
    quality_requirements: QualityRequirements  # Expected quality

    # Deadline
    max_latency: float  # Maximum acceptable latency (seconds)
    deadline: float  # Absolute deadline (timestamp)

    # Witness requirements
    min_witnesses: int = 3  # Minimum witness count
    min_witness_societies: int = 3  # Minimum society diversity

    def __repr__(self) -> str:
        return (f"FederationTask({self.task_id}, type={self.task_type}, "
                f"cost={self.estimated_cost:.1f} ATP, horizon={self.task_horizon})")


@dataclass
class ExecutionProof:
    """
    Cryptographically signed proof of task execution

    Attested by witnesses to build trust.
    """
    task_id: str
    executing_platform: str  # Who executed

    # Execution results
    result_data: Dict[str, Any]  # Response, IRP info, etc.
    actual_latency: float  # How long it actually took
    actual_cost: float  # ATP actually consumed

    # Quality metrics
    irp_iterations: int
    final_energy: float
    convergence_quality: float
    quality_score: float  # 4-component SAGE quality (0-1)

    # Cryptographic proof (future: signatures)
    execution_hash: str = ""  # Hash of (task + result + metrics)
    # platform_signature: bytes = b""  # Ed25519 signature by executing platform
    # witness_signatures: List[bytes] = field(default_factory=list)

    # Timestamp
    execution_timestamp: float = field(default_factory=time.time)

    def calculate_hash(self) -> str:
        """Calculate execution hash for verification"""
        data = (
            f"{self.task_id}|{self.executing_platform}|"
            f"{self.actual_latency}|{self.actual_cost}|"
            f"{self.quality_score}|{self.execution_timestamp}"
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def __post_init__(self):
        if not self.execution_hash:
            self.execution_hash = self.calculate_hash()


# ============================================================================
# Witness Types
# ============================================================================

class WitnessOutcome(Enum):
    """Outcome of witness attestation"""
    ACCURATE = "accurate"  # Witness matched ground truth
    INACCURATE = "inaccurate"  # Witness contradicted ground truth
    PENDING = "pending"  # Ground truth not yet known


@dataclass
class WitnessAttestation:
    """
    Witness evaluation of execution quality

    Tracks both correctness and quality.
    Based on Web4 Session #83 witness_diversity_system.py
    """
    attestation_id: str
    task_id: str
    witness_lct_id: str  # Witnessing platform
    witness_society_id: str  # For diversity requirement

    # Attestation
    claimed_correctness: float  # Is result correct? (0-1)
    claimed_quality: float  # Is quality good? (0-1)

    # Evaluation (ground truth if known)
    actual_correctness: Optional[float] = None
    actual_quality: Optional[float] = None

    # Outcome
    outcome: WitnessOutcome = WitnessOutcome.PENDING

    # Signature (future: Ed25519)
    # witness_signature: bytes = b""
    timestamp: float = field(default_factory=time.time)

    def evaluate(
        self,
        ground_truth_correctness: float,
        ground_truth_quality: float,
        tolerance: float = 0.1
    ) -> WitnessOutcome:
        """
        Evaluate accuracy against ground truth

        Args:
            ground_truth_correctness: Actual correctness
            ground_truth_quality: Actual quality
            tolerance: Acceptable error margin

        Returns:
            ACCURATE if within tolerance, else INACCURATE
        """
        self.actual_correctness = ground_truth_correctness
        self.actual_quality = ground_truth_quality

        # Check both correctness and quality
        correctness_error = abs(self.claimed_correctness - ground_truth_correctness)
        quality_error = abs(self.claimed_quality - ground_truth_quality)

        if correctness_error <= tolerance and quality_error <= tolerance:
            self.outcome = WitnessOutcome.ACCURATE
        else:
            self.outcome = WitnessOutcome.INACCURATE

        return self.outcome


@dataclass
class WitnessRecord:
    """
    Per-witness reliability tracking

    Tracks accuracy over time to identify unreliable/malicious witnesses.
    """
    witness_lct_id: str
    witness_society_id: str

    # Attestation history
    attestations: List[WitnessAttestation] = field(default_factory=list)

    # Accuracy metrics
    total_attestations: int = 0
    accurate_attestations: int = 0
    inaccurate_attestations: int = 0

    def accuracy_rate(self) -> float:
        """Calculate witness accuracy rate (0-1)"""
        if self.total_attestations == 0:
            return 0.5  # Neutral for new witnesses
        return self.accurate_attestations / self.total_attestations

    def update_attestation_outcome(self, attestation: WitnessAttestation):
        """Update accuracy metrics when attestation outcome is determined"""
        if attestation.outcome == WitnessOutcome.PENDING:
            return  # Can't update yet

        self.attestations.append(attestation)
        self.total_attestations += 1

        if attestation.outcome == WitnessOutcome.ACCURATE:
            self.accurate_attestations += 1
        else:
            self.inaccurate_attestations += 1


# ============================================================================
# Utility Functions
# ============================================================================

def create_thor_identity(stake_amount: float = 1000.0) -> FederationIdentity:
    """Create Thor platform federation identity"""
    from sage.core.mrh_profile import (
        MRHProfile,
        SpatialExtent,
        TemporalExtent,
        ComplexityExtent
    )

    # Thor hardware spec
    hardware = HardwareSpec(
        platform_name="Thor",
        ram_gb=64.0,
        gpu_cores=1792,  # Ampere
        power_budget="high",
        architecture="arm64"
    )

    # Thor can handle large horizons
    max_horizon = MRHProfile(
        delta_r=SpatialExtent.GLOBAL,
        delta_t=TemporalExtent.EPOCH,
        delta_c=ComplexityExtent.SOCIETY_SCALE
    )

    # Create identity with stake
    stake = IdentityStake(
        lct_id="thor_sage_lct",
        stake_amount=stake_amount,
        stake_timestamp=time.time()
    )

    return FederationIdentity(
        lct_id="thor_sage_lct",
        platform_name="Thor",
        hardware_spec=hardware,
        max_mrh_horizon=max_horizon,
        supported_modalities=['llm_inference', 'vision', 'coordination', 'consolidation'],
        stake=stake,
        execution_history=[],
        reputation_score=0.8  # Thor starts with good reputation (development platform)
    )


def create_sprout_identity(stake_amount: float = 1000.0) -> FederationIdentity:
    """Create Sprout platform federation identity"""
    from sage.core.mrh_profile import (
        MRHProfile,
        SpatialExtent,
        TemporalExtent,
        ComplexityExtent
    )

    # Sprout hardware spec (more constrained than Thor)
    hardware = HardwareSpec(
        platform_name="Sprout",
        ram_gb=8.0,
        gpu_cores=1024,  # Ampere (less than Thor)
        power_budget="medium",
        architecture="arm64"
    )

    # Sprout handles smaller horizons (edge constraints)
    max_horizon = MRHProfile(
        delta_r=SpatialExtent.LOCAL,
        delta_t=TemporalExtent.SESSION,
        delta_c=ComplexityExtent.AGENT_SCALE
    )

    # Create identity with stake
    stake = IdentityStake(
        lct_id="sprout_sage_lct",
        stake_amount=stake_amount,
        stake_timestamp=time.time()
    )

    return FederationIdentity(
        lct_id="sprout_sage_lct",
        platform_name="Sprout",
        hardware_spec=hardware,
        max_mrh_horizon=max_horizon,
        supported_modalities=['llm_inference', 'vision'],  # No heavy consolidation
        stake=stake,
        execution_history=[],
        reputation_score=0.75  # Sprout has good edge validation track record
    )


if __name__ == "__main__":
    # Demo federation identities
    print("Federation Identity Examples")
    print("=" * 80)

    thor = create_thor_identity()
    print(f"\nThor: {thor}")
    print(f"  Hardware: {thor.hardware_spec}")
    print(f"  Max horizon: {thor.max_mrh_horizon}")
    print(f"  Stake: {thor.stake.stake_amount} ATP ({thor.stake.status.value})")

    sprout = create_sprout_identity()
    print(f"\nSprout: {sprout}")
    print(f"  Hardware: {sprout.hardware_spec}")
    print(f"  Max horizon: {sprout.max_mrh_horizon}")
    print(f"  Stake: {sprout.stake.stake_amount} ATP ({sprout.stake.status.value})")

    print("\n" + "=" * 80)
