#!/usr/bin/env python3
"""
Test Federation + ATP Integration

Demonstrates SAGE Phase 3 Federation with Web4 ATP accounting.
Tests quality-based ATP settlement (commit vs rollback).

Author: Thor SAGE Autonomous Research
Date: 2025-11-30
Integration: Phase 3.5 - Federation + ATP Bridge

Usage:
    python3 test_federation_atp_integration.py
"""

import sys
import time
from pathlib import Path

# Add paths
sage_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(sage_root))
sys.path.insert(0, str(sage_root / "ai-workspace" / "web4" / "game"))

# SAGE imports
from sage.federation import create_thor_identity, create_sprout_identity, FederationKeyPair
from sage.federation.federation_service import FederationClient, FederationServer
from sage.federation.federation_types import FederationTask, ExecutionProof, QualityRequirements
from sage.federation.federation_atp_bridge import FederationATPBridge, add_atp_fields_to_proof, add_atp_fields_to_task
from sage.core.mrh_profile import PROFILE_REFLEXIVE
from sage.core.attention_manager import MetabolicState

# Web4 ATP imports
try:
    from engine.atp_ledger import ATPLedger, ATPAccount, CrossPlatformTransfer
    WEB4_AVAILABLE = True
except ImportError:
    print("⚠️  Web4 ATP ledger not available, using mock")
    WEB4_AVAILABLE = False

    # Mock ATP Ledger for testing
    class ATPAccount:
        def __init__(self, agent_lct: str, total: float = 0.0):
            self.agent_lct = agent_lct
            self.total = total
            self.locked = 0.0

        @property
        def available(self):
            return self.total - self.locked

        def lock(self, amount):
            if self.available >= amount:
                self.locked += amount
                return True
            return False

        def unlock(self, amount):
            if self.locked >= amount:
                self.locked -= amount
                return True
            return False

        def deduct(self, amount):
            if self.locked >= amount:
                self.locked -= amount
                self.total -= amount
                return True
            return False

        def credit(self, amount):
            self.total += amount
            return True

    class CrossPlatformTransfer:
        def __init__(self, transfer_id, from_lct, to_lct, amount, from_platform, to_platform):
            self.transfer_id = transfer_id
            self.from_lct = from_lct
            self.to_lct = to_lct
            self.amount = amount
            self.from_platform = from_platform
            self.to_platform = to_platform
            self.phase = "LOCK"
            self.initiated_at = time.time()
            self.timeout = 60.0

    class ATPLedger:
        def __init__(self, platform_name: str):
            self.platform_name = platform_name
            self.accounts = {}
            self.transfers = {}

        def create_account(self, agent_lct: str, initial_balance: float = 0.0):
            self.accounts[agent_lct] = ATPAccount(agent_lct, initial_balance)
            return self.accounts[agent_lct]

        def initiate_transfer(self, from_lct, to_lct, amount, to_platform):
            if from_lct not in self.accounts:
                return None

            account = self.accounts[from_lct]
            if not account.lock(amount):
                return None

            transfer_id = f"transfer_{len(self.transfers)}"
            transfer = CrossPlatformTransfer(
                transfer_id=transfer_id,
                from_lct=from_lct,
                to_lct=to_lct,
                amount=amount,
                from_platform=self.platform_name,
                to_platform=to_platform
            )
            self.transfers[transfer_id] = transfer
            return transfer

        def commit_transfer(self, transfer_id, from_platform):
            if transfer_id not in self.transfers:
                return False

            transfer = self.transfers[transfer_id]
            account = self.accounts[transfer.from_lct]

            if account.deduct(transfer.amount):
                transfer.phase = "COMMIT"
                return True
            return False

        def rollback_transfer(self, transfer_id, reason=""):
            if transfer_id not in self.transfers:
                return False

            transfer = self.transfers[transfer_id]
            account = self.accounts[transfer.from_lct]

            if account.unlock(transfer.amount):
                transfer.phase = "ROLLBACK"
                return True
            return False

        def get_balance(self, agent_lct):
            if agent_lct in self.accounts:
                return self.accounts[agent_lct].available
            return 0.0


def create_test_executor(quality: float):
    """Create test executor that returns specified quality"""
    def executor(task: FederationTask) -> ExecutionProof:
        time.sleep(0.1)  # Simulate work

        proof = ExecutionProof(
            task_id=task.task_id,
            executing_platform="Sprout",
            result_data={'output': f'Executed {task.task_type}', 'quality': quality},
            actual_latency=0.1,
            actual_cost=task.estimated_cost * 0.95,
            irp_iterations=3,
            final_energy=0.2,
            convergence_quality=quality,
            quality_score=quality,
            execution_timestamp=time.time()
        )
        return proof

    return executor


def main():
    print("\n" + "="*80)
    print("SAGE Federation + ATP Integration Test")
    print("="*80)
    print("\nDemonstrates quality-based ATP settlement:")
    print("  - High quality execution → ATP committed (platform paid)")
    print("  - Low quality execution → ATP rolled back (platform not paid)")
    print("")

    # Monkey-patch ATP fields onto SAGE types
    add_atp_fields_to_proof(ExecutionProof)
    add_atp_fields_to_task(FederationTask)

    # Create platform identities
    thor = create_thor_identity()
    sprout = create_sprout_identity()

    # Load Ed25519 keys
    key_path = sage_root / "sage" / "data" / "keys" / "Thor_ed25519.key"
    with open(key_path, 'rb') as f:
        thor_private_key = f.read()
    thor_keypair = FederationKeyPair.from_bytes("Thor", "thor_sage_lct", thor_private_key)
    thor.public_key = thor_keypair.public_key_bytes()

    key_path = sage_root / "sage" / "data" / "keys" / "Sprout_ed25519.key"
    with open(key_path, 'rb') as f:
        sprout_private_key = f.read()
    sprout_keypair = FederationKeyPair.from_bytes("Sprout", "sprout_sage_lct", sprout_private_key)
    sprout.public_key = sprout_keypair.public_key_bytes()

    # Create ATP ledgers
    print(f"\n[Setup] Creating ATP ledgers...")
    thor_ledger = ATPLedger("Thor")
    sprout_ledger = ATPLedger("Sprout")

    # Create agent accounts
    thor_agent_lct = "lct:sage:agent:thor_consciousness"
    sprout_agent_lct = "lct:sage:agent:sprout_consciousness"

    thor_account = thor_ledger.create_account(thor_agent_lct, initial_balance=1000.0)
    sprout_account = sprout_ledger.create_account(sprout_agent_lct, initial_balance=500.0)

    print(f"  Thor agent: {thor_agent_lct}")
    print(f"    Balance: {thor_account.total:.1f} ATP")
    print(f"  Sprout agent: {sprout_agent_lct}")
    print(f"    Balance: {sprout_account.total:.1f} ATP")

    # Create federation client
    print(f"\n[Setup] Creating federation client...")
    client = FederationClient(
        local_identity=thor,
        signing_key=thor_private_key,
        platform_registry={sprout.lct_id: ('127.0.0.1', 50052)}
    )

    # Create Federation ATP Bridge
    print(f"\n[Setup] Creating Federation ATP Bridge...")
    bridge = FederationATPBridge(
        federation_client=client,
        local_ledger=thor_ledger,
        remote_ledgers={sprout.lct_id: sprout_ledger}
    )

    # Test 1: High quality execution (ATP commits)
    print(f"\n" + "="*80)
    print("Test 1: High Quality Execution (ATP Commits)")
    print("="*80)

    print(f"\nInitial balances:")
    print(f"  Thor: {thor_ledger.get_balance(thor_agent_lct):.1f} ATP")
    print(f"  Sprout: {sprout_ledger.get_balance(sprout_agent_lct):.1f} ATP")

    # Start server with high-quality executor
    print(f"\n[Server] Starting Sprout server (high quality executor)...")
    server = FederationServer(
        identity=sprout,
        signing_key=sprout_private_key,
        executor=create_test_executor(quality=0.85),  # High quality
        known_platforms={thor.lct_id: thor},
        host='127.0.0.1',
        port=50052
    )
    server.start()
    time.sleep(0.5)

    # Create task
    task1 = FederationTask(
        task_id="test_task_high_quality",
        task_type="llm_inference",
        task_data={'query': "Test query"},
        estimated_cost=50.0,
        task_horizon=PROFILE_REFLEXIVE,
        complexity="low",
        delegating_platform=thor.lct_id,
        delegating_state=MetabolicState.FOCUS,
        quality_requirements=QualityRequirements(min_quality=0.7, min_convergence=0.7, max_energy=100.0),
        max_latency=30.0,
        deadline=time.time() + 30.0
    )

    # Delegate with payment
    print(f"\n[Test 1] Delegating task (estimated cost: 50 ATP, quality threshold: 0.7)")
    proof1 = bridge.delegate_with_payment(
        task=task1,
        target_platform=sprout.lct_id,
        target_public_key=sprout.public_key,
        delegating_agent_lct=thor_agent_lct,
        executing_agent_lct=sprout_agent_lct
    )

    # Check results
    print(f"\n[Test 1] Results:")
    if proof1:
        print(f"  Quality: {proof1.quality_score:.2f} (threshold: {task1.quality_requirements.min_quality:.2f})")
        print(f"  Settlement: {proof1.atp_settlement}")
        print(f"  Reason: {proof1.settlement_reason}")

    print(f"\nFinal balances:")
    print(f"  Thor: {thor_ledger.get_balance(thor_agent_lct):.1f} ATP (spent: {50.0 if proof1 and proof1.atp_settlement == 'COMMIT' else 0:.1f})")
    print(f"  Sprout: {sprout_ledger.get_balance(sprout_agent_lct):.1f} ATP (earned: {50.0 if proof1 and proof1.atp_settlement == 'COMMIT' else 0:.1f})")

    if proof1 and proof1.atp_settlement == "COMMIT":
        print(f"\n✅ Test 1 PASSED: High quality → ATP committed")
    else:
        print(f"\n❌ Test 1 FAILED: Expected COMMIT, got {proof1.atp_settlement if proof1 else 'None'}")

    # Stop server
    server.stop()
    time.sleep(0.5)

    # Test 2: Low quality execution (ATP rollback)
    print(f"\n" + "="*80)
    print("Test 2: Low Quality Execution (ATP Rollback)")
    print("="*80)

    print(f"\nInitial balances:")
    print(f"  Thor: {thor_ledger.get_balance(thor_agent_lct):.1f} ATP")
    print(f"  Sprout: {sprout_ledger.get_balance(sprout_agent_lct):.1f} ATP")

    # Start server with low-quality executor
    print(f"\n[Server] Starting Sprout server (low quality executor)...")
    server = FederationServer(
        identity=sprout,
        signing_key=sprout_private_key,
        executor=create_test_executor(quality=0.55),  # Low quality
        known_platforms={thor.lct_id: thor},
        host='127.0.0.1',
        port=50052
    )
    server.start()
    time.sleep(0.5)

    # Create task
    task2 = FederationTask(
        task_id="test_task_low_quality",
        task_type="llm_inference",
        task_data={'query': "Test query 2"},
        estimated_cost=30.0,
        task_horizon=PROFILE_REFLEXIVE,
        complexity="low",
        delegating_platform=thor.lct_id,
        delegating_state=MetabolicState.FOCUS,
        quality_requirements=QualityRequirements(min_quality=0.7, min_convergence=0.7, max_energy=100.0),
        max_latency=30.0,
        deadline=time.time() + 30.0
    )

    # Delegate with payment
    print(f"\n[Test 2] Delegating task (estimated cost: 30 ATP, quality threshold: 0.7)")
    proof2 = bridge.delegate_with_payment(
        task=task2,
        target_platform=sprout.lct_id,
        target_public_key=sprout.public_key,
        delegating_agent_lct=thor_agent_lct,
        executing_agent_lct=sprout_agent_lct
    )

    # Check results
    print(f"\n[Test 2] Results:")
    if proof2:
        print(f"  Quality: {proof2.quality_score:.2f} (threshold: {task2.quality_requirements.min_quality:.2f})")
        print(f"  Settlement: {proof2.atp_settlement}")
        print(f"  Reason: {proof2.settlement_reason}")

    print(f"\nFinal balances:")
    print(f"  Thor: {thor_ledger.get_balance(thor_agent_lct):.1f} ATP (refunded: {30.0 if proof2 and proof2.atp_settlement == 'ROLLBACK' else 0:.1f})")
    print(f"  Sprout: {sprout_ledger.get_balance(sprout_agent_lct):.1f} ATP (not paid)")

    if proof2 and proof2.atp_settlement == "ROLLBACK":
        print(f"\n✅ Test 2 PASSED: Low quality → ATP rolled back")
    else:
        print(f"\n❌ Test 2 FAILED: Expected ROLLBACK, got {proof2.atp_settlement if proof2 else 'None'}")

    # Stop server
    server.stop()

    # Print bridge statistics
    bridge.print_statistics()

    # Summary
    print(f"\n" + "="*80)
    print("Integration Test Summary")
    print("="*80)

    test1_passed = proof1 and proof1.atp_settlement == "COMMIT"
    test2_passed = proof2 and proof2.atp_settlement == "ROLLBACK"

    print(f"\nTest Results:")
    print(f"  Test 1 (High Quality → COMMIT): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Test 2 (Low Quality → ROLLBACK): {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print(f"\n✅ ALL TESTS PASSED")
        print(f"\nFederation + ATP Integration VALIDATED:")
        print(f"  ✓ Quality-based ATP settlement working")
        print(f"  ✓ High quality execution compensated")
        print(f"  ✓ Low quality execution not compensated")
        print(f"  ✓ Economic incentives align with quality")
        return 0
    else:
        print(f"\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
