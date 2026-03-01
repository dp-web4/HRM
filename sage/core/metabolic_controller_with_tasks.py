#!/usr/bin/env python3
"""
Metabolic Controller with Task Management Integration

Extends MetabolicController with ATP Reward Pool integration for task-based
ATP allocation. This allows SAGE consciousness to create tasks, fund them from
ATP budget, execute them, and claim rewards - all with conservation-safe ATP
accounting.

Integration Pattern:
- MetabolicController: Manages ATP budget and metabolic states
- ATPRewardPool: Manages task lifecycle and conservation-safe rewards
- Integration: Controller uses pool for task-based ATP allocation

Key Features:
1. Task creation for consciousness operations (IRP patterns, consolidation, etc.)
2. Conservation-safe ATP funding (ATP comes from controller balance)
3. Task execution tracking
4. Reward claiming on completion (ATP returns to balance)
5. Auto-expiry and cleanup of stale tasks

Example Usage:
    controller = MetabolicControllerWithTasks(initial_atp=100.0)

    # Create task for running IRP pattern
    task_id = controller.create_consciousness_task(
        description="Run IRP pattern: ProactiveExploration",
        reward_atp=5.0,
        executor_id="irp_plugin_001"
    )

    # Execute task (IRP plugin does work)
    # ...

    # Complete and claim reward
    controller.complete_and_claim_task(task_id, executor_id="irp_plugin_001")

Author: Thor (autonomous SAGE research)
Date: 2026-02-28
Reference: Web4 Session 17 (Economic Attack Resistance)
"""

from typing import Dict, Optional, Tuple, List
from pathlib import Path
import sys
import time


from sage.core.metabolic_controller import MetabolicController, MetabolicState
from sage.core.atp_reward_pool import ATPRewardPool, Task, TaskStatus


class MetabolicControllerWithTasks(MetabolicController):
    """
    MetabolicController extended with ATP reward pool task management.

    Adds task-based ATP allocation on top of base metabolic state management:
    - Create tasks for consciousness operations
    - Fund tasks from ATP budget (conservation-safe)
    - Track task execution
    - Claim rewards on completion
    - Auto-cleanup of expired tasks

    ATP Conservation:
    - Controller ATP balance + Pool balance = Constant (minus task overhead)
    - Tasks funded from controller balance → pool
    - Rewards claimed from pool → controller balance
    - Cancelled/expired tasks refunded to controller
    """

    def __init__(
        self,
        initial_atp: float = 100.0,
        max_atp: float = 100.0,
        device: Optional[str] = None,
        circadian_period: int = 100,
        enable_circadian: bool = True,
        simulation_mode: bool = False,
        task_overhead_atp: float = 0.1,  # ATP cost per task operation
    ):
        """
        Initialize metabolic controller with task management.

        Args:
            initial_atp: Starting ATP budget
            max_atp: Maximum ATP capacity
            device: Compute device
            circadian_period: Cycles per day
            enable_circadian: Enable circadian rhythm biasing
            simulation_mode: Use cycle counts instead of wall time
            task_overhead_atp: ATP cost for task operations (create, fund, claim)
        """
        super().__init__(
            initial_atp=initial_atp,
            max_atp=max_atp,
            device=device,
            circadian_period=circadian_period,
            enable_circadian=enable_circadian,
            simulation_mode=simulation_mode
        )

        # ATP reward pool for task management
        self.reward_pool = ATPRewardPool()

        # Task operation overhead (prevents infinite task loops)
        self.task_overhead_atp = task_overhead_atp

        # Task tracking
        self.entity_id = "metabolic_controller"  # Controller acts as requester
        self.task_counter = 0  # Auto-incrementing task ID

        # Statistics
        self.stats_tasks_created = 0
        self.stats_tasks_completed = 0
        self.stats_tasks_failed = 0
        self.stats_total_rewards_paid = 0.0

    def create_consciousness_task(
        self,
        description: str,
        reward_atp: float,
        executor_id: str,
        expires_in_seconds: float = 3600.0,
    ) -> Optional[str]:
        """
        Create and fund a task for consciousness operation.

        This is the main entry point for task-based ATP allocation.
        Controller acts as requester, funding task from its ATP budget.

        Args:
            description: Task description (e.g., "Run IRP pattern X")
            reward_atp: ATP reward for completion
            executor_id: ID of executor (e.g., "irp_plugin_001")
            expires_in_seconds: Task expiry time

        Returns:
            task_id if successful, None if failed
        """
        # Check if controller has sufficient ATP for reward + overhead
        required_atp = reward_atp + self.task_overhead_atp
        if self.atp_current < required_atp:
            self.stats_tasks_failed += 1
            return None

        # Generate task ID
        self.task_counter += 1
        task_id = f"task_{self.entity_id}_{self.task_counter:06d}"

        try:
            # Create task
            task = self.reward_pool.create_task(
                task_id=task_id,
                requester_id=self.entity_id,
                description=description,
                reward_atp=reward_atp,
                expires_in_seconds=expires_in_seconds,
            )

            # Fund task from controller ATP (conservation: ATP → pool)
            success, new_balance, msg = self.reward_pool.fund_task(
                task_id=task_id,
                requester_balance=self.atp_current,
            )

            if success:
                # Update controller ATP (deduct reward + overhead)
                self.atp_current = new_balance - self.task_overhead_atp
                self.stats_tasks_created += 1

                # Immediately start task (assign to executor)
                self.reward_pool.start_task(task_id, executor_id)

                return task_id
            else:
                self.stats_tasks_failed += 1
                return None

        except Exception as e:
            self.stats_tasks_failed += 1
            return None

    def complete_and_claim_task(
        self,
        task_id: str,
        executor_id: str,
    ) -> Tuple[bool, float]:
        """
        Complete task and claim reward.

        Args:
            task_id: Task to complete
            executor_id: Executor claiming completion

        Returns:
            (success, reward_atp) - reward_atp is 0 if failed
        """
        # Complete task
        success, msg = self.reward_pool.complete_task(task_id, executor_id)
        if not success:
            self.stats_tasks_failed += 1
            return False, 0.0

        # Claim reward (conservation: ATP from pool → executor)
        # In this integration, executor is the controller itself,
        # so reward goes back to controller ATP balance
        success, new_balance, msg = self.reward_pool.claim_reward(
            task_id=task_id,
            executor_id=executor_id,
            executor_balance=self.atp_current,
        )

        if success:
            # Update controller ATP (add reward, deduct overhead for claim operation)
            self.atp_current = new_balance - self.task_overhead_atp

            # Get reward amount for stats
            task = self.reward_pool.tasks[task_id]
            reward_atp = task.reward_atp

            self.stats_tasks_completed += 1
            self.stats_total_rewards_paid += reward_atp

            return True, reward_atp
        else:
            self.stats_tasks_failed += 1
            return False, 0.0

    def cancel_task(
        self,
        task_id: str,
    ) -> Tuple[bool, float]:
        """
        Cancel task and refund ATP to controller.

        Only allowed if task not yet started or completed.

        Args:
            task_id: Task to cancel

        Returns:
            (success, refund_atp) - refund_atp is 0 if failed
        """
        success, new_balance, msg = self.reward_pool.cancel_task(
            task_id=task_id,
            requester_id=self.entity_id,
            requester_balance=self.atp_current,
        )

        if success:
            # Update controller ATP (add refund, deduct overhead for cancel operation)
            self.atp_current = new_balance - self.task_overhead_atp

            # Get refund amount for stats
            task = self.reward_pool.tasks[task_id]
            refund_atp = task.reward_atp

            return True, refund_atp
        else:
            return False, 0.0

    def cleanup_expired_tasks(self) -> Dict[str, float]:
        """
        Clean up expired tasks and refund ATP to controller.

        Should be called periodically (e.g., during state transitions).

        Returns:
            Dict of {task_id: refund_atp} for expired tasks
        """
        expired_tasks = {}
        current_time = time.time()

        for task_id, task in list(self.reward_pool.tasks.items()):
            # Check if task expired and not yet marked
            if (task.status in [TaskStatus.PENDING, TaskStatus.FUNDED] and
                current_time > task.expires_at):

                # Expire task and refund
                success, new_balance, msg = self.reward_pool.expire_task(
                    task_id=task_id,
                    requester_balance=self.atp_current,
                )

                if success:
                    # Update controller ATP (add refund)
                    self.atp_current = new_balance
                    expired_tasks[task_id] = task.reward_atp

        return expired_tasks

    def get_task_stats(self) -> Dict:
        """Get task management statistics."""
        return {
            'tasks_created': self.stats_tasks_created,
            'tasks_completed': self.stats_tasks_completed,
            'tasks_failed': self.stats_tasks_failed,
            'total_rewards_paid': self.stats_total_rewards_paid,
            'pool_balance': self.reward_pool.pool_balance,
            'total_funded': self.reward_pool.total_funded,
            'total_claimed': self.reward_pool.total_claimed,
            'total_expired': self.reward_pool.total_expired,
            'total_cancelled': self.reward_pool.total_cancelled,
            'active_tasks': sum(
                1 for t in self.reward_pool.tasks.values()
                if t.status in [TaskStatus.FUNDED, TaskStatus.IN_PROGRESS]
            ),
        }

    def get_active_tasks(self) -> List[Task]:
        """Get list of active tasks."""
        return [
            task for task in self.reward_pool.tasks.values()
            if task.status in [TaskStatus.FUNDED, TaskStatus.IN_PROGRESS]
        ]

    def verify_conservation(self) -> Tuple[bool, str]:
        """
        Verify ATP conservation across controller + pool.

        Conservation invariant (accounting for task overhead):
        pool_balance + controller_atp = initial_atp - overhead_spent - expired - cancelled

        More simply: total_funded = total_claimed + pool_balance + expired + cancelled

        Returns:
            (is_valid, message)
        """
        # Verify pool conservation: funded = claimed + pool + expired + cancelled
        pool_total = (
            self.reward_pool.total_claimed +
            self.reward_pool.pool_balance +
            self.reward_pool.total_expired +
            self.reward_pool.total_cancelled
        )

        # Allow small floating point error
        epsilon = 0.01
        is_valid = abs(self.reward_pool.total_funded - pool_total) < epsilon

        if is_valid:
            msg = f"Conservation verified: {pool_total:.2f} ATP accounted for (funded={self.reward_pool.total_funded:.2f})"
        else:
            msg = f"Conservation VIOLATED: funded {self.reward_pool.total_funded:.2f}, accounted {pool_total:.2f}"

        return is_valid, msg

    def update(self, cycle_data: Dict) -> MetabolicState:
        """
        Update metabolic state and cleanup expired tasks.

        Extends base update() with task cleanup on state transitions.
        """
        # Get current state before update
        old_state = self.current_state

        # Call base update
        new_state = super().update(cycle_data)

        # On state transition, cleanup expired tasks
        if new_state != old_state:
            expired = self.cleanup_expired_tasks()
            if expired:
                # Add to cycle data for logging
                cycle_data['expired_tasks'] = expired

        return new_state

    def get_stats(self) -> Dict:
        """
        Get extended statistics including task management.

        Extends base get_stats() with task information.
        """
        # Get base stats
        stats = super().get_stats()

        # Add task stats
        stats['tasks'] = self.get_task_stats()

        # Add conservation check
        is_valid, msg = self.verify_conservation()
        stats['conservation'] = {
            'valid': is_valid,
            'message': msg,
        }

        return stats


# Convenience function for creating controller with tasks
def create_task_aware_metabolic_controller(
    initial_atp: float = 100.0,
    max_atp: float = 100.0,
    **kwargs
) -> MetabolicControllerWithTasks:
    """
    Create metabolic controller with task management enabled.

    Args:
        initial_atp: Starting ATP budget
        max_atp: Maximum ATP capacity
        **kwargs: Additional arguments passed to MetabolicControllerWithTasks

    Returns:
        MetabolicControllerWithTasks instance
    """
    return MetabolicControllerWithTasks(
        initial_atp=initial_atp,
        max_atp=max_atp,
        **kwargs
    )
