"""
Unit Tests for ATP Reward Pool
================================

Tests the conservation-safe ATP reward distribution system.

Author: Thor (autonomous SAGE research)
Date: 2026-02-28
"""

import sys
from pathlib import Path
import unittest

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from core.atp_reward_pool import ATPRewardPool, Task, TaskStatus


class TestATPRewardPool(unittest.TestCase):
    """Test ATP reward pool pattern."""

    def setUp(self):
        """Create fresh reward pool for each test."""
        self.pool = ATPRewardPool()

    def test_create_task(self):
        """Test task creation."""
        task = self.pool.create_task(
            task_id="test_001",
            requester_id="alice",
            description="Test task",
            reward_atp=100.0,
        )

        self.assertEqual(task.task_id, "test_001")
        self.assertEqual(task.requester_id, "alice")
        self.assertEqual(task.reward_atp, 100.0)
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertIsNone(task.executor_id)

    def test_fund_task(self):
        """Test task funding with ATP transfer."""
        # Create task
        self.pool.create_task("test_001", "alice", "Test", 100.0)

        # Fund with sufficient balance
        success, new_balance, msg = self.pool.fund_task("test_001", 1000.0)

        self.assertTrue(success)
        self.assertEqual(new_balance, 900.0)  # 1000 - 100
        self.assertEqual(self.pool.pool_balance, 100.0)
        self.assertEqual(self.pool.tasks["test_001"].status, TaskStatus.FUNDED)

    def test_fund_task_insufficient_balance(self):
        """Test funding fails with insufficient balance."""
        self.pool.create_task("test_001", "alice", "Test", 100.0)

        # Try to fund with insufficient balance
        success, new_balance, msg = self.pool.fund_task("test_001", 50.0)

        self.assertFalse(success)
        self.assertEqual(new_balance, 50.0)  # Unchanged
        self.assertEqual(self.pool.pool_balance, 0.0)  # Pool unchanged
        self.assertIn("Insufficient balance", msg)

    def test_complete_and_claim_reward(self):
        """Test full task lifecycle with reward claim."""
        # Setup: Create and fund task
        self.pool.create_task("test_001", "alice", "Test", 100.0)
        _, alice_balance, _ = self.pool.fund_task("test_001", 1000.0)

        # Bob starts task
        success, msg = self.pool.start_task("test_001", "bob")
        self.assertTrue(success)

        # Bob completes task
        success, msg = self.pool.complete_task("test_001", "bob")
        self.assertTrue(success)

        # Bob claims reward
        bob_balance = 500.0
        success, new_bob_balance, msg = self.pool.claim_reward("test_001", "bob", bob_balance)

        self.assertTrue(success)
        self.assertEqual(new_bob_balance, 600.0)  # 500 + 100
        self.assertEqual(self.pool.pool_balance, 0.0)  # Pool drained
        self.assertEqual(self.pool.tasks["test_001"].status, TaskStatus.CLAIMED)

        # Verify conservation: Alice -100, Bob +100, Pool 0
        self.assertEqual(alice_balance, 900.0)
        self.assertEqual(new_bob_balance, 600.0)
        self.assertEqual(self.pool.pool_balance, 0.0)

    def test_claim_reward_wrong_executor(self):
        """Test reward claim fails for wrong executor."""
        # Setup: Create, fund, assign to Bob
        self.pool.create_task("test_001", "alice", "Test", 100.0)
        self.pool.fund_task("test_001", 1000.0)
        self.pool.start_task("test_001", "bob")
        self.pool.complete_task("test_001", "bob")

        # Charlie tries to claim Bob's reward
        success, balance, msg = self.pool.claim_reward("test_001", "charlie", 300.0)

        self.assertFalse(success)
        self.assertEqual(balance, 300.0)  # Unchanged
        self.assertEqual(self.pool.pool_balance, 100.0)  # Pool unchanged
        self.assertIn("completed by bob", msg)

    def test_cancel_funded_task(self):
        """Test cancelling funded task refunds ATP."""
        # Create and fund task
        self.pool.create_task("test_001", "alice", "Test", 100.0)
        _, alice_balance, _ = self.pool.fund_task("test_001", 1000.0)

        # Alice cancels before anyone starts
        success, new_balance, msg = self.pool.cancel_task("test_001", "alice", alice_balance)

        self.assertTrue(success)
        self.assertEqual(new_balance, 1000.0)  # Refunded to original
        self.assertEqual(self.pool.pool_balance, 0.0)  # Pool drained
        self.assertEqual(self.pool.tasks["test_001"].status, TaskStatus.CANCELLED)

    def test_conservation_validation(self):
        """Test conservation invariant validation."""
        # Create multiple tasks with different outcomes
        # Task 1: Funded and claimed
        self.pool.create_task("task_001", "alice", "Test 1", 100.0)
        self.pool.fund_task("task_001", 1000.0)
        self.pool.start_task("task_001", "bob")
        self.pool.complete_task("task_001", "bob")
        self.pool.claim_reward("task_001", "bob", 500.0)

        # Task 2: Funded and cancelled
        self.pool.create_task("task_002", "alice", "Test 2", 50.0)
        self.pool.fund_task("task_002", 900.0)
        self.pool.cancel_task("task_002", "alice", 900.0)

        # Task 3: Funded but still in pool
        self.pool.create_task("task_003", "alice", "Test 3", 75.0)
        self.pool.fund_task("task_003", 950.0)

        # Validate conservation
        valid, details = self.pool.validate_conservation({})

        self.assertTrue(valid)
        self.assertEqual(details['total_funded'], 225.0)  # 100 + 50 + 75
        self.assertEqual(details['total_claimed'], 100.0)
        self.assertEqual(details['total_cancelled'], 50.0)
        self.assertEqual(details['pool_balance'], 75.0)  # task_003 still pending

        # Conservation: funded = claimed + cancelled + pool
        # 225 = 100 + 50 + 75 ✓

    def test_attack_double_claim(self):
        """Test that rewards cannot be claimed twice."""
        # Setup: Complete task
        self.pool.create_task("test_001", "alice", "Test", 100.0)
        self.pool.fund_task("test_001", 1000.0)
        self.pool.start_task("test_001", "bob")
        self.pool.complete_task("test_001", "bob")

        # Bob claims reward
        success, balance, _ = self.pool.claim_reward("test_001", "bob", 500.0)
        self.assertTrue(success)
        self.assertEqual(balance, 600.0)

        # Bob tries to claim again
        success, balance, msg = self.pool.claim_reward("test_001", "bob", 600.0)

        self.assertFalse(success)
        self.assertEqual(balance, 600.0)  # Unchanged
        self.assertIn("not completed", msg)  # Status is CLAIMED, not COMPLETED

    def test_attack_claim_without_funding(self):
        """Test that tasks must be funded before claiming."""
        # Create task but don't fund it
        self.pool.create_task("test_001", "alice", "Test", 100.0)

        # Try to claim reward
        success, balance, msg = self.pool.claim_reward("test_001", "bob", 500.0)

        self.assertFalse(success)
        self.assertEqual(balance, 500.0)
        self.assertIn("not completed", msg)

    def test_stats(self):
        """Test statistics reporting."""
        # Create tasks in various states
        self.pool.create_task("task_001", "alice", "Test 1", 100.0)
        self.pool.fund_task("task_001", 1000.0)
        self.pool.start_task("task_001", "bob")
        self.pool.complete_task("task_001", "bob")
        self.pool.claim_reward("task_001", "bob", 500.0)

        self.pool.create_task("task_002", "alice", "Test 2", 50.0)
        self.pool.fund_task("task_002", 900.0)

        stats = self.pool.get_stats()

        self.assertEqual(stats['total_tasks'], 2)
        self.assertEqual(stats['tasks_by_status']['claimed'], 1)
        self.assertEqual(stats['tasks_by_status']['funded'], 1)
        self.assertEqual(stats['total_funded'], 150.0)
        self.assertEqual(stats['total_claimed'], 100.0)
        self.assertEqual(stats['pool_balance'], 50.0)
        self.assertTrue(stats['conservation_valid'])


class TestConservationInvariants(unittest.TestCase):
    """Test ATP conservation across complex scenarios."""

    def test_multi_party_conservation(self):
        """Test conservation with multiple requesters and executors."""
        pool = ATPRewardPool()

        # Initial balances
        balances = {
            'alice': 1000.0,
            'bob': 500.0,
            'charlie': 300.0,
            'dave': 200.0,
        }

        initial_total = sum(balances.values())  # 2000.0

        # Alice creates task for Bob
        pool.create_task("task_001", "alice", "Task 1", 100.0)
        _, balances['alice'], _ = pool.fund_task("task_001", balances['alice'])
        pool.start_task("task_001", "bob")
        pool.complete_task("task_001", "bob")
        _, balances['bob'], _ = pool.claim_reward("task_001", "bob", balances['bob'])

        # Charlie creates task for Dave
        pool.create_task("task_002", "charlie", "Task 2", 50.0)
        _, balances['charlie'], _ = pool.fund_task("task_002", balances['charlie'])
        pool.start_task("task_002", "dave")
        pool.complete_task("task_002", "dave")
        _, balances['dave'], _ = pool.claim_reward("task_002", "dave", balances['dave'])

        # Verify global conservation
        final_total = sum(balances.values()) + pool.pool_balance

        self.assertAlmostEqual(initial_total, final_total, places=2)

        # Verify individual balances
        self.assertEqual(balances['alice'], 900.0)   # Paid 100
        self.assertEqual(balances['bob'], 600.0)     # Earned 100
        self.assertEqual(balances['charlie'], 250.0) # Paid 50
        self.assertEqual(balances['dave'], 250.0)    # Earned 50


if __name__ == '__main__':
    unittest.main(verbosity=2)
