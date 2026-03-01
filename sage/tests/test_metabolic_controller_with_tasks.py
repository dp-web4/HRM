#!/usr/bin/env python3
"""
Tests for MetabolicController + ATP Reward Pool Integration

Tests conservation-safe task management integrated with metabolic state control.

Test Coverage:
1. Task creation and funding from controller ATP
2. Task completion and reward claiming
3. Task cancellation and refund
4. Task expiry and cleanup
5. ATP conservation across controller + pool
6. State transitions with task cleanup
7. Multi-task scenarios
8. Error handling (insufficient ATP, invalid operations)

Author: Thor (autonomous SAGE research)
Date: 2026-02-28
"""

import unittest
import time
from pathlib import Path
import sys

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from core.metabolic_controller_with_tasks import (
    MetabolicControllerWithTasks,
    create_task_aware_metabolic_controller,
)
from core.atp_reward_pool import TaskStatus


class TestMetabolicControllerWithTasks(unittest.TestCase):
    """Test metabolic controller with task management integration."""

    def setUp(self):
        """Create controller for testing."""
        self.controller = MetabolicControllerWithTasks(
            initial_atp=100.0,
            max_atp=100.0,
            simulation_mode=True,  # Use cycle counts instead of wall time
        )

    def test_create_and_fund_task(self):
        """Test creating and funding a task from controller ATP."""
        initial_atp = self.controller.atp_current

        # Create task
        task_id = self.controller.create_consciousness_task(
            description="Test task",
            reward_atp=10.0,
            executor_id="executor_001",
        )

        self.assertIsNotNone(task_id)

        # Verify controller ATP decreased (reward + overhead)
        expected_atp = initial_atp - 10.0 - self.controller.task_overhead_atp
        self.assertAlmostEqual(self.controller.atp_current, expected_atp, places=2)

        # Verify pool balance increased
        self.assertAlmostEqual(self.controller.reward_pool.pool_balance, 10.0, places=2)

        # Verify task created and funded
        task = self.controller.reward_pool.tasks[task_id]
        self.assertEqual(task.status, TaskStatus.IN_PROGRESS)  # Auto-started
        self.assertEqual(task.executor_id, "executor_001")

    def test_complete_and_claim_reward(self):
        """Test completing task and claiming reward."""
        # Create task
        task_id = self.controller.create_consciousness_task(
            description="Test task",
            reward_atp=10.0,
            executor_id="executor_001",
        )

        atp_before_claim = self.controller.atp_current

        # Complete and claim
        success, reward = self.controller.complete_and_claim_task(
            task_id=task_id,
            executor_id="executor_001",
        )

        self.assertTrue(success)
        self.assertAlmostEqual(reward, 10.0, places=2)

        # Verify controller ATP increased (reward - overhead for claim)
        expected_atp = atp_before_claim + 10.0 - self.controller.task_overhead_atp
        self.assertAlmostEqual(self.controller.atp_current, expected_atp, places=2)

        # Verify pool balance decreased
        self.assertAlmostEqual(self.controller.reward_pool.pool_balance, 0.0, places=2)

        # Verify task claimed
        task = self.controller.reward_pool.tasks[task_id]
        self.assertEqual(task.status, TaskStatus.CLAIMED)

    def test_cancel_task_refund(self):
        """Test cancelling task and getting refund."""
        # Create task
        task_id = self.controller.create_consciousness_task(
            description="Test task",
            reward_atp=10.0,
            executor_id="executor_001",
        )

        # Get task reference before cancellation
        task = self.controller.reward_pool.tasks[task_id]

        # Manually reset status to FUNDED (cancel only works on PENDING/FUNDED)
        task.status = TaskStatus.FUNDED

        atp_before_cancel = self.controller.atp_current

        # Cancel task
        success, refund = self.controller.cancel_task(task_id)

        self.assertTrue(success)
        self.assertAlmostEqual(refund, 10.0, places=2)

        # Verify controller ATP increased (refund - overhead)
        expected_atp = atp_before_cancel + 10.0 - self.controller.task_overhead_atp
        self.assertAlmostEqual(self.controller.atp_current, expected_atp, places=2)

        # Verify pool balance decreased
        self.assertAlmostEqual(self.controller.reward_pool.pool_balance, 0.0, places=2)

    def test_task_expiry_cleanup(self):
        """Test automatic expiry and cleanup of old tasks."""
        # Create task with short expiry
        task_id = self.controller.create_consciousness_task(
            description="Test task",
            reward_atp=10.0,
            executor_id="executor_001",
            expires_in_seconds=0.1,  # 100ms expiry
        )

        # Wait for expiry
        time.sleep(0.2)

        # Manually set task to FUNDED so it can expire
        task = self.controller.reward_pool.tasks[task_id]
        task.status = TaskStatus.FUNDED

        atp_before_cleanup = self.controller.atp_current

        # Cleanup expired tasks
        expired = self.controller.cleanup_expired_tasks()

        self.assertIn(task_id, expired)
        self.assertAlmostEqual(expired[task_id], 10.0, places=2)

        # Verify controller ATP increased (refund)
        expected_atp = atp_before_cleanup + 10.0
        self.assertAlmostEqual(self.controller.atp_current, expected_atp, places=2)

    def test_insufficient_atp_for_task(self):
        """Test task creation fails with insufficient ATP."""
        # Drain ATP to near zero
        self.controller.atp_current = 1.0

        # Try to create expensive task
        task_id = self.controller.create_consciousness_task(
            description="Expensive task",
            reward_atp=10.0,
            executor_id="executor_001",
        )

        self.assertIsNone(task_id)  # Should fail
        self.assertEqual(self.controller.stats_tasks_failed, 1)

    def test_conservation_across_operations(self):
        """Test ATP conservation across task create/claim/cancel operations."""
        initial_atp = self.controller.atp_current

        # Create multiple tasks
        task1 = self.controller.create_consciousness_task(
            description="Task 1",
            reward_atp=5.0,
            executor_id="executor_001",
        )

        task2 = self.controller.create_consciousness_task(
            description="Task 2",
            reward_atp=8.0,
            executor_id="executor_002",
        )

        task3 = self.controller.create_consciousness_task(
            description="Task 3",
            reward_atp=3.0,
            executor_id="executor_003",
        )

        # Complete task 1
        self.controller.complete_and_claim_task(task1, "executor_001")

        # Cancel task 2 (manually set to FUNDED first)
        self.controller.reward_pool.tasks[task2].status = TaskStatus.FUNDED
        self.controller.cancel_task(task2)

        # Leave task 3 pending

        # Verify conservation using pool's internal accounting
        # Pool conservation: total_funded = total_claimed + pool_balance + total_expired + total_cancelled
        is_valid, msg = self.controller.verify_conservation()
        self.assertTrue(is_valid, msg)

    def test_multi_task_lifecycle(self):
        """Test full lifecycle of multiple concurrent tasks."""
        # Create 5 tasks
        tasks = []
        for i in range(5):
            task_id = self.controller.create_consciousness_task(
                description=f"Task {i}",
                reward_atp=2.0,
                executor_id=f"executor_{i:03d}",
            )
            tasks.append(task_id)

        # Verify all created
        self.assertEqual(len(tasks), 5)
        self.assertEqual(self.controller.stats_tasks_created, 5)

        # Complete 3 tasks
        for i in range(3):
            success, reward = self.controller.complete_and_claim_task(
                task_id=tasks[i],
                executor_id=f"executor_{i:03d}",
            )
            self.assertTrue(success)

        # Verify 3 completed
        self.assertEqual(self.controller.stats_tasks_completed, 3)

        # Check active tasks (2 remaining)
        active = self.controller.get_active_tasks()
        self.assertEqual(len(active), 2)

    def test_state_transition_cleanup(self):
        """Test task cleanup happens on state transitions."""
        # Create task with short expiry
        task_id = self.controller.create_consciousness_task(
            description="Test task",
            reward_atp=5.0,
            executor_id="executor_001",
            expires_in_seconds=0.1,
        )

        # Manually set to FUNDED so it can expire
        self.controller.reward_pool.tasks[task_id].status = TaskStatus.FUNDED

        # Wait for expiry
        time.sleep(0.2)

        # Force a state transition by draining ATP
        # First, wait long enough for hysteresis (5 cycles minimum)
        cycle_data = {
            'atp_consumed': 1.0,
            'attention_load': 1,
            'max_salience': 0.5,
            'crisis_detected': False,
        }

        # Run 5 cycles to satisfy hysteresis
        for _ in range(5):
            self.controller.update(cycle_data)

        # Now drain ATP to trigger WAKE → REST transition
        self.controller.atp_current = 25.0
        old_state = self.controller.current_state
        self.controller.update(cycle_data)
        new_state = self.controller.current_state

        # Verify state transition occurred
        self.assertNotEqual(old_state, new_state)

        # Check if cleanup happened (task should be expired now)
        task = self.controller.reward_pool.tasks[task_id]
        self.assertEqual(task.status, TaskStatus.EXPIRED)

    def test_task_statistics(self):
        """Test task statistics tracking."""
        # Create and complete tasks
        task1 = self.controller.create_consciousness_task(
            description="Task 1",
            reward_atp=5.0,
            executor_id="executor_001",
        )

        task2 = self.controller.create_consciousness_task(
            description="Task 2",
            reward_atp=3.0,
            executor_id="executor_002",
        )

        self.controller.complete_and_claim_task(task1, "executor_001")
        self.controller.complete_and_claim_task(task2, "executor_002")

        # Get stats
        stats = self.controller.get_task_stats()

        self.assertEqual(stats['tasks_created'], 2)
        self.assertEqual(stats['tasks_completed'], 2)
        self.assertAlmostEqual(stats['total_rewards_paid'], 8.0, places=2)
        self.assertEqual(stats['active_tasks'], 0)

    def test_conservation_verification(self):
        """Test explicit conservation verification."""
        # Perform various operations
        task1 = self.controller.create_consciousness_task(
            description="Task 1",
            reward_atp=10.0,
            executor_id="executor_001",
        )

        self.controller.complete_and_claim_task(task1, "executor_001")

        # Verify conservation
        is_valid, msg = self.controller.verify_conservation()

        self.assertTrue(is_valid)
        self.assertIn("verified", msg.lower())

    def test_get_extended_stats(self):
        """Test extended stats include task information."""
        # Create task
        task_id = self.controller.create_consciousness_task(
            description="Test task",
            reward_atp=5.0,
            executor_id="executor_001",
        )

        # Get stats
        stats = self.controller.get_stats()

        # Verify task stats included
        self.assertIn('tasks', stats)
        self.assertIn('conservation', stats)
        self.assertEqual(stats['tasks']['tasks_created'], 1)
        self.assertTrue(stats['conservation']['valid'])

    def test_convenience_constructor(self):
        """Test convenience constructor function."""
        controller = create_task_aware_metabolic_controller(
            initial_atp=50.0,
            max_atp=150.0,
        )

        self.assertEqual(controller.atp_current, 50.0)
        self.assertEqual(controller.atp_max, 150.0)
        self.assertIsNotNone(controller.reward_pool)


def run_tests():
    """Run all tests and print results."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
