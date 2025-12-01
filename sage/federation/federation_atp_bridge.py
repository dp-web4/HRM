"""
Federation ATP Bridge - Integration of SAGE Federation with Web4 ATP Accounting

Connects SAGE Phase 3 Federation Network with Web4 ATP Ledger to enable
economic task delegation. Tasks cost ATP, quality determines ATP settlement.

Author: Thor SAGE Autonomous Research
Date: 2025-11-30
Integration: Phase 3.5 - SAGE Federation + Web4 ATP
References: FEDERATION_CONSENSUS_ATP_INTEGRATION.md

Usage:
    # Create bridge
    bridge = FederationATPBridge(
        federation_client=client,
        local_ledger=thor_atp_ledger,
        remote_ledgers={'sprout_sage_lct': sprout_atp_ledger}  # For testing
    )

    # Delegate task with ATP payment
    proof = bridge.delegate_with_payment(
        task=federation_task,
        target_platform='sprout_sage_lct',
        target_public_key=sprout_public_key,
        delegating_agent_lct='lct:sage:agent:thor_consciousness',
        executing_agent_lct='lct:sage:agent:sprout_consciousness'
    )

    # Check ATP settlement
    if proof and proof.atp_settlement == 'COMMIT':
        print(f"Task successful, ATP transferred: {task.estimated_cost}")
    elif proof and proof.atp_settlement == 'ROLLBACK':
        print(f"Task quality insufficient, ATP refunded")
"""

import time
from typing import Optional, Dict
from dataclasses import dataclass

# SAGE imports
from sage.federation.federation_service import FederationClient
from sage.federation.federation_types import FederationTask, ExecutionProof


# ATP Ledger imports (from web4)
# Note: In real deployment, this would import from web4 package
# For testing, we'll use duck typing and assume compatible interface
class ATPLedger:
    """
    ATP Ledger interface (compatible with web4/game/engine/atp_ledger.py)

    This is a minimal interface for integration. Real implementation
    in web4/game/engine/atp_ledger.py.
    """

    def initiate_transfer(
        self,
        from_lct: str,
        to_lct: str,
        amount: float,
        to_platform: str
    ) -> Optional['CrossPlatformTransfer']:
        """
        Initiate cross-platform ATP transfer

        Returns CrossPlatformTransfer if successful, None if insufficient balance
        """
        raise NotImplementedError("Use web4.game.engine.atp_ledger.ATPLedger")

    def commit_transfer(
        self,
        transfer_id: str,
        from_platform: str
    ) -> bool:
        """Commit ATP transfer (finalize payment)"""
        raise NotImplementedError("Use web4.game.engine.atp_ledger.ATPLedger")

    def rollback_transfer(
        self,
        transfer_id: str,
        reason: str = ""
    ) -> bool:
        """Rollback ATP transfer (refund)"""
        raise NotImplementedError("Use web4.game.engine.atp_ledger.ATPLedger")

    def get_balance(self, agent_lct: str) -> float:
        """Get ATP balance for agent"""
        raise NotImplementedError("Use web4.game.engine.atp_ledger.ATPLedger")


@dataclass
class CrossPlatformTransfer:
    """
    Cross-platform ATP transfer (from web4)

    This mirrors web4/game/engine/atp_ledger.py structure
    """
    transfer_id: str
    from_lct: str
    to_lct: str
    amount: float
    from_platform: str
    to_platform: str
    phase: str  # "LOCK", "COMMIT", "ROLLBACK", "COMPLETE"
    initiated_at: float
    timeout: float


class FederationATPBridge:
    """
    Bridge between SAGE Federation and Web4 ATP Accounting

    Enables economic task delegation:
    1. Delegate task → Lock ATP
    2. Execute task → Create proof
    3. Evaluate quality → Commit or Rollback ATP

    Quality-based settlement ensures executing platforms are paid
    only for high-quality work.
    """

    def __init__(
        self,
        federation_client: FederationClient,
        local_ledger: ATPLedger,
        remote_ledgers: Optional[Dict[str, ATPLedger]] = None
    ):
        """
        Initialize Federation ATP Bridge

        Args:
            federation_client: SAGE federation client for task delegation
            local_ledger: Local ATP ledger for delegating platform
            remote_ledgers: Remote ATP ledgers (for testing only)
                In production, ATP settlement happens via consensus
        """
        self.client = federation_client
        self.local_ledger = local_ledger
        self.remote_ledgers = remote_ledgers or {}

        # Statistics
        self.tasks_delegated = 0
        self.tasks_committed = 0
        self.tasks_rolled_back = 0
        self.total_atp_spent = 0.0
        self.total_atp_refunded = 0.0

    def delegate_with_payment(
        self,
        task: FederationTask,
        target_platform: str,
        target_public_key: bytes,
        delegating_agent_lct: str,
        executing_agent_lct: str,
        timeout: float = 60.0
    ) -> Optional[ExecutionProof]:
        """
        Delegate task with ATP payment

        Flow:
        1. Lock ATP for transfer (prevents double-spend)
        2. Delegate task via federation client
        3. Receive execution proof
        4. Evaluate quality against threshold
        5. Commit ATP if quality sufficient, rollback otherwise

        Args:
            task: Federation task to delegate
            target_platform: Target platform LCT ID
            target_public_key: Target platform's Ed25519 public key
            delegating_agent_lct: LCT of agent paying for task
            executing_agent_lct: LCT of agent receiving payment
            timeout: Task delegation timeout (seconds)

        Returns:
            ExecutionProof if successful, None on failure
            Proof includes atp_settlement field: "COMMIT" or "ROLLBACK"
        """

        print(f"\n[FederationATP] Delegating task with payment")
        print(f"  Task: {task.task_id}")
        print(f"  Cost: {task.estimated_cost:.1f} ATP")
        print(f"  Payer: {delegating_agent_lct}")
        print(f"  Payee: {executing_agent_lct}")

        # Step 1: Lock ATP for transfer
        print(f"\n[FederationATP] Step 1: Locking ATP...")

        transfer = self.local_ledger.initiate_transfer(
            from_lct=delegating_agent_lct,
            to_lct=executing_agent_lct,
            amount=task.estimated_cost,
            to_platform=target_platform
        )

        if not transfer:
            print(f"  ✗ Insufficient ATP balance")
            print(f"  Available: {self.local_ledger.get_balance(delegating_agent_lct):.1f} ATP")
            print(f"  Required: {task.estimated_cost:.1f} ATP")
            return None

        print(f"  ✓ ATP locked: {task.estimated_cost:.1f}")
        print(f"  Transfer ID: {transfer.transfer_id}")

        # Attach ATP transfer metadata to task
        task.atp_transfer_id = transfer.transfer_id
        task.delegating_agent_lct = delegating_agent_lct
        task.executing_agent_lct = executing_agent_lct

        # Step 2: Delegate task via federation
        print(f"\n[FederationATP] Step 2: Delegating task...")

        proof = self.client.delegate_task(
            task,
            target_platform_id=target_platform,
            target_public_key=target_public_key,
            timeout=timeout
        )

        if not proof:
            # Task delegation failed - rollback ATP
            print(f"  ✗ Task delegation failed")
            print(f"\n[FederationATP] Step 3: Rolling back ATP...")

            self.local_ledger.rollback_transfer(
                transfer.transfer_id,
                reason="Task delegation failed"
            )

            print(f"  ✓ ATP refunded: {task.estimated_cost:.1f}")
            self.tasks_rolled_back += 1
            self.total_atp_refunded += task.estimated_cost
            return None

        print(f"  ✓ Task execution complete")
        print(f"  Quality: {proof.quality_score:.2f}")
        print(f"  Actual cost: {proof.actual_cost:.1f} ATP")

        # Step 3: Evaluate quality and settle ATP
        print(f"\n[FederationATP] Step 3: Evaluating quality...")
        print(f"  Quality threshold: {task.quality_requirements.min_quality:.2f}")
        print(f"  Actual quality: {proof.quality_score:.2f}")

        if proof.quality_score >= task.quality_requirements.min_quality:
            # Quality acceptable - commit ATP transfer
            print(f"  ✓ Quality acceptable - committing ATP transfer")

            # Commit on local ledger (deduct from delegating agent)
            committed = self.local_ledger.commit_transfer(
                transfer.transfer_id,
                from_platform=self.client.local_identity.platform_name
            )

            if not committed:
                # Should not happen (ATP was locked), but handle gracefully
                print(f"  ✗ WARNING: Local commit failed (unexpected)")
                proof.atp_settlement = "ROLLBACK"
                proof.settlement_reason = "Local commit failed"
                self.tasks_rolled_back += 1
                return proof

            # In production, remote commit happens via consensus
            # For testing, commit on remote ledger if available
            if target_platform in self.remote_ledgers:
                remote_ledger = self.remote_ledgers[target_platform]
                # Credit executing agent on remote platform
                remote_ledger.accounts[executing_agent_lct].credit(task.estimated_cost)
                print(f"  ✓ Remote ATP credited (testing mode)")

            proof.atp_settlement = "COMMIT"
            proof.settlement_reason = f"Quality {proof.quality_score:.2f} >= threshold {task.quality_requirements.min_quality:.2f}"

            print(f"  ✓ ATP transferred: {task.estimated_cost:.1f}")

            self.tasks_committed += 1
            self.total_atp_spent += task.estimated_cost

        else:
            # Quality insufficient - rollback ATP transfer
            print(f"  ✗ Quality insufficient - rolling back ATP transfer")

            rolled_back = self.local_ledger.rollback_transfer(
                transfer.transfer_id,
                reason=f"Quality {proof.quality_score:.2f} < threshold {task.quality_requirements.min_quality:.2f}"
            )

            if not rolled_back:
                print(f"  ✗ WARNING: Rollback failed (unexpected)")

            proof.atp_settlement = "ROLLBACK"
            proof.settlement_reason = f"Quality {proof.quality_score:.2f} < threshold {task.quality_requirements.min_quality:.2f}"

            print(f"  ✓ ATP refunded: {task.estimated_cost:.1f}")

            self.tasks_rolled_back += 1
            self.total_atp_refunded += task.estimated_cost

        # Update statistics
        self.tasks_delegated += 1

        print(f"\n[FederationATP] Task delegation complete")
        print(f"  Settlement: {proof.atp_settlement}")
        print(f"  Reason: {proof.settlement_reason}")

        return proof

    def get_statistics(self) -> Dict[str, any]:
        """Get bridge statistics"""
        return {
            'tasks_delegated': self.tasks_delegated,
            'tasks_committed': self.tasks_committed,
            'tasks_rolled_back': self.tasks_rolled_back,
            'commit_rate': self.tasks_committed / max(self.tasks_delegated, 1),
            'total_atp_spent': self.total_atp_spent,
            'total_atp_refunded': self.total_atp_refunded,
            'net_atp_spent': self.total_atp_spent - self.total_atp_refunded
        }

    def print_statistics(self):
        """Print bridge statistics"""
        stats = self.get_statistics()

        print(f"\n" + "="*80)
        print(f"Federation ATP Bridge Statistics")
        print(f"="*80)
        print(f"\nTask Delegation:")
        print(f"  Total tasks: {stats['tasks_delegated']}")
        print(f"  Committed: {stats['tasks_committed']} ({stats['commit_rate']*100:.1f}%)")
        print(f"  Rolled back: {stats['tasks_rolled_back']} ({(1-stats['commit_rate'])*100:.1f}%)")

        print(f"\nATP Flow:")
        print(f"  Total spent: {stats['total_atp_spent']:.1f} ATP")
        print(f"  Total refunded: {stats['total_atp_refunded']:.1f} ATP")
        print(f"  Net spent: {stats['net_atp_spent']:.1f} ATP")

        print(f"\n" + "="*80)


# Integration helper for ExecutionProof
def add_atp_fields_to_proof(proof_class):
    """
    Add ATP settlement fields to ExecutionProof

    This is a monkey-patch for testing. In production, ExecutionProof
    should be extended with these fields.
    """
    if not hasattr(proof_class, 'atp_transfer_id'):
        proof_class.atp_transfer_id = None
    if not hasattr(proof_class, 'atp_settlement'):
        proof_class.atp_settlement = None
    if not hasattr(proof_class, 'settlement_reason'):
        proof_class.settlement_reason = None


def add_atp_fields_to_task(task_class):
    """
    Add ATP transfer fields to FederationTask

    This is a monkey-patch for testing. In production, FederationTask
    should be extended with these fields.
    """
    if not hasattr(task_class, 'atp_transfer_id'):
        task_class.atp_transfer_id = None
    if not hasattr(task_class, 'delegating_agent_lct'):
        task_class.delegating_agent_lct = None
    if not hasattr(task_class, 'executing_agent_lct'):
        task_class.executing_agent_lct = None
