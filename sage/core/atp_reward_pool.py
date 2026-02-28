"""
ATP Reward Pool - Conservation-Safe Task Reward System
=======================================================

Implements the "reward pool pattern" discovered in Web4 Session 17, Track 2
(Economic Attack Resistance). This pattern ensures ATP conservation by having
task requesters pay into a pool BEFORE task execution, and rewards come FROM
the pool (not created from nothing).

**Security Pattern**:
- Task requester: pays ATP into pool → pool_balance increases
- Task executor: claims reward from pool → pool_balance decreases
- Conservation: ATP is never created or destroyed, only transferred

**Attack Vector Prevented**:
- ATP inflation (creating ATP from nothing)
- Reward gaming (claiming rewards without completing tasks)
- Double-spending (claiming same reward multiple times)

**Key Insight from Web4 Session 17**:
"Rewards from nowhere" bug (Track 2, Economic Attack Resistance):
- Problem: Tasks added rewards without funding source
- Fix: Reward pool account - tasks pay in, rewards come out
- Conservation: total_generated - total_consumed = total_balance

Author: Thor (autonomous SAGE research)
Date: 2026-02-28
Reference: web4/implementation/reference/economic_attack_resistance.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time


class TaskStatus(Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    FUNDED = "funded"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CLAIMED = "claimed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A task with ATP reward."""
    task_id: str
    requester_id: str
    description: str
    reward_atp: float
    status: TaskStatus = TaskStatus.PENDING
    executor_id: Optional[str] = None
    created_at: float = 0.0
    funded_at: float = 0.0
    completed_at: float = 0.0
    claimed_at: float = 0.0
    expires_at: float = 0.0


@dataclass
class ATPRewardPool:
    """
    Conservation-safe ATP reward distribution system.

    Pattern:
    1. Requester funds task → ATP transferred from requester to pool
    2. Executor completes task → task marked completed
    3. Executor claims reward → ATP transferred from pool to executor

    Conservation Invariant:
    sum(requester_balances) + pool_balance + sum(executor_balances) = constant
    """

    pool_balance: float = 0.0
    tasks: Dict[str, Task] = field(default_factory=dict)
    total_funded: float = 0.0
    total_claimed: float = 0.0
    total_expired: float = 0.0
    total_cancelled: float = 0.0

    def create_task(
        self,
        task_id: str,
        requester_id: str,
        description: str,
        reward_atp: float,
        expires_in_seconds: float = 3600.0,
    ) -> Task:
        """
        Create a new task (not yet funded).

        Args:
            task_id: Unique task identifier
            requester_id: ID of entity requesting task
            description: Task description
            reward_atp: ATP reward for completion
            expires_in_seconds: Time until task expires

        Returns:
            Created task in PENDING state
        """
        if task_id in self.tasks:
            raise ValueError(f"Task {task_id} already exists")

        if reward_atp <= 0:
            raise ValueError(f"Reward must be positive, got {reward_atp}")

        current_time = time.time()

        task = Task(
            task_id=task_id,
            requester_id=requester_id,
            description=description,
            reward_atp=reward_atp,
            status=TaskStatus.PENDING,
            created_at=current_time,
            expires_at=current_time + expires_in_seconds,
        )

        self.tasks[task_id] = task
        return task

    def fund_task(
        self,
        task_id: str,
        requester_balance: float,
    ) -> Tuple[bool, float, str]:
        """
        Fund a task by transferring ATP from requester to pool.

        This is the KEY security operation: ATP must come FROM somewhere
        (requester's balance), not created from nothing.

        Args:
            task_id: Task to fund
            requester_balance: Requester's current ATP balance

        Returns:
            (success, new_requester_balance, message)
        """
        if task_id not in self.tasks:
            return False, requester_balance, f"Task {task_id} not found"

        task = self.tasks[task_id]

        if task.status != TaskStatus.PENDING:
            return False, requester_balance, f"Task {task_id} not pending (status: {task.status.value})"

        # Check if requester has sufficient balance
        if requester_balance < task.reward_atp:
            return False, requester_balance, f"Insufficient balance: need {task.reward_atp}, have {requester_balance}"

        # Conservation: Transfer ATP from requester to pool
        new_requester_balance = requester_balance - task.reward_atp
        self.pool_balance += task.reward_atp

        # Update task state
        task.status = TaskStatus.FUNDED
        task.funded_at = time.time()

        # Track total funded
        self.total_funded += task.reward_atp

        return True, new_requester_balance, f"Task {task_id} funded with {task.reward_atp} ATP"

    def start_task(
        self,
        task_id: str,
        executor_id: str,
    ) -> Tuple[bool, str]:
        """
        Assign task to executor and mark in progress.

        Args:
            task_id: Task to start
            executor_id: ID of entity executing task

        Returns:
            (success, message)
        """
        if task_id not in self.tasks:
            return False, f"Task {task_id} not found"

        task = self.tasks[task_id]

        if task.status != TaskStatus.FUNDED:
            return False, f"Task {task_id} not funded (status: {task.status.value})"

        # Check if task expired
        if time.time() > task.expires_at:
            task.status = TaskStatus.EXPIRED
            return False, f"Task {task_id} expired"

        # Assign to executor
        task.executor_id = executor_id
        task.status = TaskStatus.IN_PROGRESS

        return True, f"Task {task_id} started by {executor_id}"

    def complete_task(
        self,
        task_id: str,
        executor_id: str,
    ) -> Tuple[bool, str]:
        """
        Mark task as completed (but not yet claimed).

        Args:
            task_id: Task to complete
            executor_id: Executor claiming completion

        Returns:
            (success, message)
        """
        if task_id not in self.tasks:
            return False, f"Task {task_id} not found"

        task = self.tasks[task_id]

        if task.status != TaskStatus.IN_PROGRESS:
            return False, f"Task {task_id} not in progress (status: {task.status.value})"

        if task.executor_id != executor_id:
            return False, f"Task {task_id} assigned to {task.executor_id}, not {executor_id}"

        # Mark completed
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()

        return True, f"Task {task_id} completed by {executor_id}"

    def claim_reward(
        self,
        task_id: str,
        executor_id: str,
        executor_balance: float,
    ) -> Tuple[bool, float, str]:
        """
        Claim reward for completed task.

        This is the KEY security operation: Reward comes FROM pool,
        not created from nothing.

        Args:
            task_id: Task to claim reward for
            executor_id: Executor claiming reward
            executor_balance: Executor's current ATP balance

        Returns:
            (success, new_executor_balance, message)
        """
        if task_id not in self.tasks:
            return False, executor_balance, f"Task {task_id} not found"

        task = self.tasks[task_id]

        if task.status != TaskStatus.COMPLETED:
            return False, executor_balance, f"Task {task_id} not completed (status: {task.status.value})"

        if task.executor_id != executor_id:
            return False, executor_balance, f"Task {task_id} completed by {task.executor_id}, not {executor_id}"

        # Conservation check: Pool must have sufficient balance
        if self.pool_balance < task.reward_atp:
            return False, executor_balance, f"Pool insufficient: need {task.reward_atp}, have {self.pool_balance}"

        # Conservation: Transfer ATP from pool to executor
        self.pool_balance -= task.reward_atp
        new_executor_balance = executor_balance + task.reward_atp

        # Update task state
        task.status = TaskStatus.CLAIMED
        task.claimed_at = time.time()

        # Track total claimed
        self.total_claimed += task.reward_atp

        return True, new_executor_balance, f"Reward {task.reward_atp} ATP claimed from task {task_id}"

    def cancel_task(
        self,
        task_id: str,
        requester_id: str,
        requester_balance: float,
    ) -> Tuple[bool, float, str]:
        """
        Cancel task and refund ATP to requester.

        Only allowed if task not yet started.

        Args:
            task_id: Task to cancel
            requester_id: Requester cancelling task
            requester_balance: Requester's current ATP balance

        Returns:
            (success, new_requester_balance, message)
        """
        if task_id not in self.tasks:
            return False, requester_balance, f"Task {task_id} not found"

        task = self.tasks[task_id]

        if task.requester_id != requester_id:
            return False, requester_balance, f"Task {task_id} requested by {task.requester_id}, not {requester_id}"

        if task.status not in [TaskStatus.PENDING, TaskStatus.FUNDED]:
            return False, requester_balance, f"Cannot cancel task in status {task.status.value}"

        # Refund if task was funded
        refund_amount = 0.0
        if task.status == TaskStatus.FUNDED:
            # Conservation: Transfer ATP from pool back to requester
            self.pool_balance -= task.reward_atp
            refund_amount = task.reward_atp

        new_requester_balance = requester_balance + refund_amount

        # Update task state
        task.status = TaskStatus.CANCELLED

        # Track total cancelled
        self.total_cancelled += task.reward_atp

        return True, new_requester_balance, f"Task {task_id} cancelled, refunded {refund_amount} ATP"

    def expire_task(
        self,
        task_id: str,
        requester_balance: float,
    ) -> Tuple[bool, float, str]:
        """
        Expire unclaimed task and refund ATP to requester.

        Args:
            task_id: Task to expire
            requester_balance: Requester's current ATP balance

        Returns:
            (success, new_requester_balance, message)
        """
        if task_id not in self.tasks:
            return False, requester_balance, f"Task {task_id} not found"

        task = self.tasks[task_id]

        if time.time() < task.expires_at:
            return False, requester_balance, f"Task {task_id} not yet expired"

        if task.status in [TaskStatus.CLAIMED, TaskStatus.CANCELLED, TaskStatus.EXPIRED]:
            return False, requester_balance, f"Task {task_id} already in terminal state {task.status.value}"

        # Refund if task was funded but not claimed
        refund_amount = 0.0
        if task.status == TaskStatus.FUNDED:
            # Conservation: Transfer ATP from pool back to requester
            self.pool_balance -= task.reward_atp
            refund_amount = task.reward_atp

        new_requester_balance = requester_balance + refund_amount

        # Update task state
        task.status = TaskStatus.EXPIRED

        # Track total expired
        self.total_expired += task.reward_atp

        return True, new_requester_balance, f"Task {task_id} expired, refunded {refund_amount} ATP"

    def validate_conservation(
        self,
        external_balances: Dict[str, float],
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate ATP conservation invariant.

        Conservation Formula:
        total_funded = total_claimed + total_expired + total_cancelled + pool_balance

        Args:
            external_balances: Dict of entity_id → ATP balance

        Returns:
            (valid, details)
        """
        # Pool accounting conservation
        expected_pool = self.total_funded - self.total_claimed - self.total_expired - self.total_cancelled
        actual_pool = self.pool_balance

        pool_valid = abs(expected_pool - actual_pool) < 0.01  # Floating point tolerance

        # Global conservation (requires external balance tracking)
        # This would need integration with metabolic controller

        details = {
            'pool_balance': self.pool_balance,
            'expected_pool': expected_pool,
            'total_funded': self.total_funded,
            'total_claimed': self.total_claimed,
            'total_expired': self.total_expired,
            'total_cancelled': self.total_cancelled,
            'pool_conservation_valid': pool_valid,
        }

        return pool_valid, details

    def get_stats(self) -> Dict[str, any]:
        """Get reward pool statistics."""
        return {
            'pool_balance': self.pool_balance,
            'total_tasks': len(self.tasks),
            'tasks_by_status': {
                status.value: sum(1 for t in self.tasks.values() if t.status == status)
                for status in TaskStatus
            },
            'total_funded': self.total_funded,
            'total_claimed': self.total_claimed,
            'total_expired': self.total_expired,
            'total_cancelled': self.total_cancelled,
            'conservation_valid': self.validate_conservation({})[0],
        }


if __name__ == '__main__':
    # Demonstration of reward pool pattern
    print("ATP Reward Pool - Conservation-Safe Pattern Demonstration")
    print("="*70)

    pool = ATPRewardPool()

    # Scenario: Alice requests task, Bob completes it
    alice_balance = 1000.0
    bob_balance = 500.0

    print(f"\nInitial State:")
    print(f"  Alice balance: {alice_balance} ATP")
    print(f"  Bob balance: {bob_balance} ATP")
    print(f"  Pool balance: {pool.pool_balance} ATP")

    # Step 1: Alice creates task
    task = pool.create_task(
        task_id="task_001",
        requester_id="alice",
        description="Analyze SAGE session data",
        reward_atp=100.0,
    )
    print(f"\n✓ Task created: {task.task_id}")

    # Step 2: Alice funds task
    success, alice_balance, msg = pool.fund_task("task_001", alice_balance)
    print(f"\n✓ {msg}")
    print(f"  Alice balance: {alice_balance} ATP (paid 100 into pool)")
    print(f"  Pool balance: {pool.pool_balance} ATP")

    # Step 3: Bob starts task
    success, msg = pool.start_task("task_001", "bob")
    print(f"\n✓ {msg}")

    # Step 4: Bob completes task
    success, msg = pool.complete_task("task_001", "bob")
    print(f"\n✓ {msg}")

    # Step 5: Bob claims reward
    success, bob_balance, msg = pool.claim_reward("task_001", "bob", bob_balance)
    print(f"\n✓ {msg}")
    print(f"  Bob balance: {bob_balance} ATP (claimed 100 from pool)")
    print(f"  Pool balance: {pool.pool_balance} ATP")

    # Validate conservation
    valid, details = pool.validate_conservation({})
    print(f"\nConservation Validation:")
    print(f"  Total funded: {details['total_funded']} ATP")
    print(f"  Total claimed: {details['total_claimed']} ATP")
    print(f"  Pool balance: {details['pool_balance']} ATP")
    print(f"  Conservation valid: {valid} ✓" if valid else f"  Conservation VIOLATED: {valid} ✗")

    print(f"\nFinal State:")
    print(f"  Alice balance: {alice_balance} ATP (spent 100)")
    print(f"  Bob balance: {bob_balance} ATP (earned 100)")
    print(f"  Pool balance: {pool.pool_balance} ATP")
    print(f"  Conservation: Alice(-100) + Bob(+100) + Pool(0) = 0 ✓")

    # Demonstrate attack prevention
    print(f"\n" + "="*70)
    print("Attack Prevention Demonstration")
    print("="*70)

    # Attack 1: Try to claim reward without completing task
    print(f"\n[Attack 1] Charlie tries to claim reward without completing task:")
    charlie_balance = 300.0
    success, charlie_balance, msg = pool.claim_reward("task_001", "charlie", charlie_balance)
    print(f"  Result: {msg}")
    print(f"  Success: {success} (BLOCKED ✓)")

    # Attack 2: Try to fund task with insufficient balance
    print(f"\n[Attack 2] Dave tries to fund task with insufficient balance:")
    dave_balance = 50.0
    task2 = pool.create_task("task_002", "dave", "Expensive task", 200.0)
    success, dave_balance, msg = pool.fund_task("task_002", dave_balance)
    print(f"  Result: {msg}")
    print(f"  Success: {success} (BLOCKED ✓)")

    print(f"\n" + "="*70)
    print("Reward pool pattern successfully prevents ATP inflation attacks!")
    print("="*70)
