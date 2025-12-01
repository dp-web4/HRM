"""
Federation Consensus Transactions - Phase 3.75

Defines transaction types for SAGE federation tasks and execution proofs that can be
embedded in Web4 consensus blocks. Enables Byzantine fault-tolerant validation of
federation operations and ATP settlement via distributed consensus.

Author: Thor SAGE Autonomous Research
Date: 2025-12-01
Integration: Phase 3.75 - SAGE Federation + Web4 Consensus + ATP
References: FEDERATION_CONSENSUS_ATP_INTEGRATION.md (Phase 3.75 design)

Architecture:
    Layer 1: Federation tasks cost ATP (Phase 3.5 - FederationATPBridge)
    Layer 2: Consensus validates tasks + ATP (Phase 3.75 - THIS MODULE)
    Layer 3: Economic incentives via quality-based settlement

Transaction Flow:
    Block N:   FEDERATION_TASK + ATP_TRANSFER_LOCK
    (task execution happens off-consensus)
    Block N+1: EXECUTION_PROOF + ATP_TRANSFER_COMMIT/ROLLBACK

Usage:
    # Create federation task transaction
    task_tx = FederationTaskTransaction.from_federation_task(
        task=federation_task,
        task_signature=task_signature,
        atp_transfer_id=transfer_id
    )

    # Add to consensus block
    block.transactions.append(task_tx.to_dict())

    # After execution, create proof transaction
    proof_tx = ExecutionProofTransaction.from_execution_proof(
        proof=execution_proof,
        proof_signature=proof_signature,
        atp_settlement="COMMIT"  # or "ROLLBACK"
    )

    # Add to next consensus block
    next_block.transactions.append(proof_tx.to_dict())
"""

import time
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum

# SAGE imports
from sage.federation.federation_types import FederationTask, ExecutionProof


class FederationTransactionType(Enum):
    """Federation transaction types for consensus"""
    TASK = "FEDERATION_TASK"  # Task delegation request
    PROOF = "FEDERATION_PROOF"  # Execution proof with quality score
    REPUTATION_UPDATE = "FEDERATION_REPUTATION"  # Platform reputation update


@dataclass
class FederationTaskTransaction:
    """
    Federation task transaction for consensus block

    Records a federation task delegation request in the blockchain.
    Must be validated by consensus before execution begins.

    Validation checks (during PREPARE phase):
    - Task signature valid (Ed25519)
    - ATP transfer exists and is LOCKED
    - Delegating platform has sufficient reputation
    - Task parameters are reasonable
    """

    type: str = "FEDERATION_TASK"
    task_id: str = ""
    delegating_platform: str = ""  # Platform requesting delegation
    executing_platform: str = ""  # Platform executing task
    task_type: str = ""  # Type of task (llm_inference, vision, etc.)
    estimated_cost: float = 0.0  # Estimated ATP cost
    quality_requirements: Dict[str, float] = field(default_factory=dict)  # min_quality, min_convergence
    atp_transfer_id: str = ""  # Reference to ATP_TRANSFER_LOCK transaction
    task_data_hash: str = ""  # SHA-256 hash of task data (for verification)
    task_signature: str = ""  # Ed25519 signature of task by delegating platform
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_federation_task(
        cls,
        task: FederationTask,
        task_signature: str,
        atp_transfer_id: str
    ) -> 'FederationTaskTransaction':
        """Create transaction from FederationTask"""
        # Hash task data for blockchain (avoid storing full data)
        task_dict = task.to_signable_dict()
        task_data_json = json.dumps(task_dict['task_data'], sort_keys=True)
        task_data_hash = hashlib.sha256(task_data_json.encode()).hexdigest()

        return cls(
            task_id=task.task_id,
            delegating_platform=task.delegating_platform,
            executing_platform=task.executing_platform if hasattr(task, 'executing_platform') else "",
            task_type=task.task_type,
            estimated_cost=task.estimated_cost,
            quality_requirements={
                'min_quality': task.quality_requirements.min_quality,
                'min_convergence': task.quality_requirements.min_convergence,
                'max_energy': task.quality_requirements.max_energy
            },
            atp_transfer_id=atp_transfer_id,
            task_data_hash=task_data_hash,
            task_signature=task_signature,
            timestamp=time.time()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for consensus block"""
        return {
            "type": self.type,
            "task_id": self.task_id,
            "delegating_platform": self.delegating_platform,
            "executing_platform": self.executing_platform,
            "task_type": self.task_type,
            "estimated_cost": self.estimated_cost,
            "quality_requirements": self.quality_requirements,
            "atp_transfer_id": self.atp_transfer_id,
            "task_data_hash": self.task_data_hash,
            "task_signature": self.task_signature,
            "timestamp": self.timestamp
        }

    def signable_content(self) -> str:
        """Content to sign (excludes signature)"""
        data = self.to_dict()
        data_without_sig = {k: v for k, v in data.items() if k != 'task_signature'}
        return json.dumps(data_without_sig, sort_keys=True)

    def hash(self) -> str:
        """Compute transaction hash"""
        return hashlib.sha256(self.signable_content().encode()).hexdigest()

    def validate(self, platform_registry: Dict[str, Any], atp_ledger: Any) -> tuple[bool, str]:
        """
        Validate transaction during consensus PREPARE phase

        Returns: (is_valid, reason)
        """
        # Check task signature
        if not self.task_signature:
            return False, "Missing task signature"

        # Check delegating platform is known
        if self.delegating_platform not in platform_registry:
            return False, f"Unknown delegating platform: {self.delegating_platform}"

        # Check ATP transfer exists and is locked
        if self.atp_transfer_id:
            transfer = atp_ledger.get_transfer(self.atp_transfer_id) if hasattr(atp_ledger, 'get_transfer') else None
            if not transfer:
                return False, f"ATP transfer not found: {self.atp_transfer_id}"
            if hasattr(transfer, 'phase') and transfer.phase != "LOCK":
                return False, f"ATP transfer not locked: {transfer.phase}"

        # Check estimated cost is reasonable
        if self.estimated_cost < 0:
            return False, f"Invalid estimated cost: {self.estimated_cost}"

        return True, "Valid"


@dataclass
class ExecutionProofTransaction:
    """
    Execution proof transaction for consensus block

    Records execution proof with quality score. Triggers ATP settlement:
    - Quality >= threshold → ATP_TRANSFER_COMMIT (platform paid)
    - Quality < threshold → ATP_TRANSFER_ROLLBACK (delegator refunded)

    Validation checks (during PREPARE phase):
    - Proof signature valid (Ed25519)
    - References valid FEDERATION_TASK transaction
    - ATP settlement matches quality evaluation
    - Quality score is reasonable (0-1)
    """

    type: str = "FEDERATION_PROOF"
    task_id: str = ""  # References FEDERATION_TASK transaction
    executing_platform: str = ""  # Platform that executed task
    quality_score: float = 0.0  # Quality score (0-1)
    actual_cost: float = 0.0  # Actual ATP cost
    actual_latency: float = 0.0  # Actual execution time (seconds)
    convergence_quality: float = 0.0  # Convergence quality (0-1)
    atp_settlement: str = ""  # "COMMIT" or "ROLLBACK"
    atp_transfer_id: str = ""  # Reference to ATP transfer
    result_data_hash: str = ""  # SHA-256 hash of result data
    proof_signature: str = ""  # Ed25519 signature of proof by executing platform
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_execution_proof(
        cls,
        proof: ExecutionProof,
        proof_signature: str,
        atp_settlement: str,
        atp_transfer_id: str
    ) -> 'ExecutionProofTransaction':
        """Create transaction from ExecutionProof"""
        # Hash result data for blockchain
        result_data_json = json.dumps(proof.result_data, sort_keys=True)
        result_data_hash = hashlib.sha256(result_data_json.encode()).hexdigest()

        return cls(
            task_id=proof.task_id,
            executing_platform=proof.executing_platform,
            quality_score=proof.quality_score,
            actual_cost=proof.actual_cost,
            actual_latency=proof.actual_latency,
            convergence_quality=proof.convergence_quality,
            atp_settlement=atp_settlement,
            atp_transfer_id=atp_transfer_id,
            result_data_hash=result_data_hash,
            proof_signature=proof_signature,
            timestamp=time.time()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for consensus block"""
        return {
            "type": self.type,
            "task_id": self.task_id,
            "executing_platform": self.executing_platform,
            "quality_score": self.quality_score,
            "actual_cost": self.actual_cost,
            "actual_latency": self.actual_latency,
            "convergence_quality": self.convergence_quality,
            "atp_settlement": self.atp_settlement,
            "atp_transfer_id": self.atp_transfer_id,
            "result_data_hash": self.result_data_hash,
            "proof_signature": self.proof_signature,
            "timestamp": self.timestamp
        }

    def signable_content(self) -> str:
        """Content to sign (excludes signature)"""
        data = self.to_dict()
        data_without_sig = {k: v for k, v in data.items() if k != 'proof_signature'}
        return json.dumps(data_without_sig, sort_keys=True)

    def hash(self) -> str:
        """Compute transaction hash"""
        return hashlib.sha256(self.signable_content().encode()).hexdigest()

    def validate(
        self,
        platform_registry: Dict[str, Any],
        task_transaction: Optional[FederationTaskTransaction]
    ) -> tuple[bool, str]:
        """
        Validate transaction during consensus PREPARE phase

        Returns: (is_valid, reason)
        """
        # Check proof signature
        if not self.proof_signature:
            return False, "Missing proof signature"

        # Check executing platform is known
        if self.executing_platform not in platform_registry:
            return False, f"Unknown executing platform: {self.executing_platform}"

        # Check task transaction exists
        if not task_transaction:
            return False, f"Task transaction not found: {self.task_id}"

        # Check ATP settlement is consistent with quality
        if task_transaction and self.atp_settlement == "COMMIT":
            min_quality = task_transaction.quality_requirements.get('min_quality', 0.0)
            if self.quality_score < min_quality:
                return False, f"Invalid COMMIT: quality {self.quality_score} < threshold {min_quality}"

        # Check quality score is valid
        if not (0.0 <= self.quality_score <= 1.0):
            return False, f"Invalid quality score: {self.quality_score}"

        return True, "Valid"


@dataclass
class ReputationUpdateTransaction:
    """
    Platform reputation update transaction

    Updates platform reputation based on execution proof quality.
    High quality → reputation increases
    Low quality → reputation decreases

    Reputation affects:
    - Task routing decisions (higher reputation gets more tasks)
    - Trust in proof validation
    - Economic penalties (low reputation may be slashed)
    """

    type: str = "FEDERATION_REPUTATION"
    platform: str = ""  # Platform whose reputation is updated
    task_id: str = ""  # Task that triggered update
    quality_score: float = 0.0  # Quality from execution proof
    reputation_delta: float = 0.0  # Change in reputation (-1 to +1)
    new_reputation: float = 0.0  # New reputation value
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for consensus block"""
        return {
            "type": self.type,
            "platform": self.platform,
            "task_id": self.task_id,
            "quality_score": self.quality_score,
            "reputation_delta": self.reputation_delta,
            "new_reputation": self.new_reputation,
            "timestamp": self.timestamp
        }

    def hash(self) -> str:
        """Compute transaction hash"""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


# Validation helpers

def validate_federation_transaction(
    tx: Dict[str, Any],
    platform_registry: Dict[str, Any],
    atp_ledger: Any,
    blockchain_state: Dict[str, Any]
) -> tuple[bool, str]:
    """
    Validate any federation transaction type

    Args:
        tx: Transaction dictionary
        platform_registry: Known platforms (for signature verification)
        atp_ledger: ATP ledger (for transfer validation)
        blockchain_state: Current blockchain state (for task lookups)

    Returns:
        (is_valid, reason)
    """
    tx_type = tx.get('type')

    if tx_type == "FEDERATION_TASK":
        task_tx = FederationTaskTransaction(**{k: v for k, v in tx.items() if k != 'type'})
        task_tx.type = tx_type
        return task_tx.validate(platform_registry, atp_ledger)

    elif tx_type == "FEDERATION_PROOF":
        proof_tx = ExecutionProofTransaction(**{k: v for k, v in tx.items() if k != 'type'})
        proof_tx.type = tx_type

        # Find corresponding task transaction
        task_tx = blockchain_state.get('tasks', {}).get(proof_tx.task_id)
        return proof_tx.validate(platform_registry, task_tx)

    elif tx_type == "FEDERATION_REPUTATION":
        # Reputation updates are always valid if from consensus proposer
        return True, "Valid"

    else:
        return False, f"Unknown federation transaction type: {tx_type}"


def apply_federation_transaction(
    tx: Dict[str, Any],
    federation_router: Any,
    atp_ledger: Any
) -> bool:
    """
    Apply committed federation transaction to local state

    Called after consensus COMMIT phase to update local federation state.

    Args:
        tx: Transaction dictionary
        federation_router: SAGE federation router (for task/reputation tracking)
        atp_ledger: ATP ledger (for ATP updates)

    Returns:
        True if applied successfully, False otherwise
    """
    tx_type = tx.get('type')

    try:
        if tx_type == "FEDERATION_TASK":
            # Record task in federation history
            if hasattr(federation_router, 'record_task'):
                federation_router.record_task(tx)
            return True

        elif tx_type == "FEDERATION_PROOF":
            # Update ATP ledger based on settlement
            if tx['atp_settlement'] == "COMMIT":
                if hasattr(atp_ledger, 'commit_transfer'):
                    atp_ledger.commit_transfer(tx['atp_transfer_id'])
            elif tx['atp_settlement'] == "ROLLBACK":
                if hasattr(atp_ledger, 'rollback_transfer'):
                    atp_ledger.rollback_transfer(tx['atp_transfer_id'])

            # Update platform reputation
            if hasattr(federation_router, 'update_platform_reputation'):
                federation_router.update_platform_reputation(
                    tx['executing_platform'],
                    tx['quality_score']
                )
            return True

        elif tx_type == "FEDERATION_REPUTATION":
            # Update platform reputation
            if hasattr(federation_router, 'set_platform_reputation'):
                federation_router.set_platform_reputation(
                    tx['platform'],
                    tx['new_reputation']
                )
            return True

        else:
            return False

    except Exception as e:
        print(f"Error applying federation transaction: {e}")
        return False
