"""
LCT-Aware ATP Permissions for SAGE Consciousness

Integrates LCT identity task scoping with ATP operations, enabling
task-based permission checks before ATP transfers and resource allocation.

Based on Legion Session #49 (Web4 LCT Permission System) but adapted
for SAGE consciousness architecture.

Author: Thor (SAGE autonomous research)
Date: 2025-12-02
Session: Autonomous SAGE Development - LCT-Aware ATP Operations
"""

from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time


class ATPPermission(Enum):
    """ATP operation permissions"""
    READ = "atp:read"          # Can query ATP balances
    WRITE = "atp:write"        # Can transfer ATP
    ALL = "atp:all"            # Full ATP access


@dataclass
class ResourceLimits:
    """
    Resource limits for task-scoped agents

    Defines maximum resources an agent with specific task scope can use.
    """
    atp_budget: float = 0.0             # Maximum ATP to spend
    memory_mb: int = 0                  # Maximum memory (MB)
    cpu_cores: int = 0                  # Maximum CPU cores
    max_concurrent_tasks: int = 0       # Maximum concurrent tasks

    created_at: float = field(default_factory=time.time)


# Task definitions with permissions and resource limits
TASK_PERMISSIONS = {
    # Perception: Read-only ATP, limited resources
    "perception": {
        "atp_permissions": {ATPPermission.READ},
        "can_delegate": False,
        "can_execute_code": False,
        "resource_limits": ResourceLimits(
            atp_budget=100.0,
            memory_mb=1024,
            cpu_cores=1,
            max_concurrent_tasks=5
        )
    },

    # Planning: Read-only ATP, limited resources
    "planning": {
        "atp_permissions": {ATPPermission.READ},
        "can_delegate": False,
        "can_execute_code": False,
        "resource_limits": ResourceLimits(
            atp_budget=100.0,
            memory_mb=2048,
            cpu_cores=2,
            max_concurrent_tasks=10
        )
    },

    # Planning.strategic: Enhanced planning with more resources
    "planning.strategic": {
        "atp_permissions": {ATPPermission.READ},
        "can_delegate": False,
        "can_execute_code": False,
        "resource_limits": ResourceLimits(
            atp_budget=200.0,
            memory_mb=4096,
            cpu_cores=4,
            max_concurrent_tasks=20
        )
    },

    # Execution.safe: Can execute code in sandbox, read/write ATP
    "execution.safe": {
        "atp_permissions": {ATPPermission.READ, ATPPermission.WRITE},
        "can_delegate": False,
        "can_execute_code": True,  # Sandboxed only
        "resource_limits": ResourceLimits(
            atp_budget=200.0,
            memory_mb=2048,
            cpu_cores=2,
            max_concurrent_tasks=10
        )
    },

    # Execution.code: Full code execution, read/write ATP
    "execution.code": {
        "atp_permissions": {ATPPermission.READ, ATPPermission.WRITE},
        "can_delegate": False,
        "can_execute_code": True,
        "resource_limits": ResourceLimits(
            atp_budget=500.0,
            memory_mb=8192,
            cpu_cores=4,
            max_concurrent_tasks=20
        )
    },

    # Delegation.federation: Can delegate to other platforms, read/write ATP
    "delegation.federation": {
        "atp_permissions": {ATPPermission.READ, ATPPermission.WRITE},
        "can_delegate": True,
        "can_execute_code": False,
        "resource_limits": ResourceLimits(
            atp_budget=1000.0,
            memory_mb=4096,
            cpu_cores=2,
            max_concurrent_tasks=50
        )
    },

    # Consciousness: Full consciousness loop, read/write ATP, can delegate
    "consciousness": {
        "atp_permissions": {ATPPermission.READ, ATPPermission.WRITE},
        "can_delegate": True,
        "can_execute_code": True,
        "resource_limits": ResourceLimits(
            atp_budget=1000.0,
            memory_mb=16384,
            cpu_cores=8,
            max_concurrent_tasks=100
        )
    },

    # Consciousness.sage: Enhanced SAGE consciousness with memory management
    # (Compatible with LUPS v1.0 consciousness.sage variant)
    "consciousness.sage": {
        "atp_permissions": {ATPPermission.READ, ATPPermission.WRITE},
        "can_delegate": True,
        "can_execute_code": True,
        "can_delete_memories": True,  # NEW: Can prune old memories for long-running loops
        "resource_limits": ResourceLimits(
            atp_budget=2000.0,        # Double standard consciousness
            memory_mb=32768,          # 32 GB (double standard consciousness)
            cpu_cores=16,             # 16 cores (double standard consciousness)
            max_concurrent_tasks=200  # Double standard consciousness
        )
    },

    # Admin.readonly: Read-only admin access
    "admin.readonly": {
        "atp_permissions": {ATPPermission.READ},
        "can_delegate": False,
        "can_execute_code": False,
        "resource_limits": ResourceLimits(
            atp_budget=100.0,
            memory_mb=1024,
            cpu_cores=1,
            max_concurrent_tasks=5
        )
    },

    # Admin.full: Full administrative access, unlimited resources
    "admin.full": {
        "atp_permissions": {ATPPermission.ALL},
        "can_delegate": True,
        "can_execute_code": True,
        "resource_limits": ResourceLimits(
            atp_budget=float('inf'),  # Unlimited
            memory_mb=1024 * 1024,    # 1TB (effectively unlimited)
            cpu_cores=128,
            max_concurrent_tasks=10000
        )
    }
}


class LCTATPPermissionChecker:
    """
    Permission checker for LCT-aware ATP operations

    Validates ATP operations against task-based permissions and resource limits.
    """

    def __init__(self, task: str):
        """
        Initialize permission checker for specific task

        Parameters:
        -----------
        task : str
            LCT task scope (e.g., "consciousness", "perception")
        """
        self.task = task

        # Get task permissions
        if task not in TASK_PERMISSIONS:
            raise ValueError(f"Unknown task: {task}. Valid tasks: {list(TASK_PERMISSIONS.keys())}")

        self.task_config = TASK_PERMISSIONS[task]
        self.atp_permissions = self.task_config["atp_permissions"]
        self.can_delegate = self.task_config["can_delegate"]
        self.can_execute_code = self.task_config["can_execute_code"]
        self.resource_limits = self.task_config["resource_limits"]

        # Track resource usage
        self.atp_spent = 0.0
        self.tasks_running = 0

    def check_atp_permission(self, operation: str) -> Tuple[bool, str]:
        """
        Check if task has permission for ATP operation

        Parameters:
        -----------
        operation : str
            ATP operation ("read", "write", "all")

        Returns:
        --------
        Tuple[bool, str]
            (allowed, reason)
        """
        # Map operation string to permission enum
        operation_map = {
            "read": ATPPermission.READ,
            "write": ATPPermission.WRITE,
            "all": ATPPermission.ALL
        }

        if operation not in operation_map:
            return False, f"Invalid ATP operation: {operation}"

        required_permission = operation_map[operation]

        # Check if task has required permission
        if required_permission in self.atp_permissions:
            return True, f"Task '{self.task}' has permission '{operation}'"

        # Check if task has ALL permission (covers everything)
        if ATPPermission.ALL in self.atp_permissions:
            return True, f"Task '{self.task}' has full ATP access"

        # Permission denied
        return False, f"Task '{self.task}' lacks permission '{operation}'"

    def check_atp_transfer(
        self,
        amount: float,
        from_lct: str,
        to_lct: str
    ) -> Tuple[bool, str]:
        """
        Check if ATP transfer is allowed

        Parameters:
        -----------
        amount : float
            ATP amount to transfer
        from_lct : str
            Source LCT identity
        to_lct : str
            Target LCT identity

        Returns:
        --------
        Tuple[bool, str]
            (allowed, reason)
        """
        # Check write permission
        can_write, reason = self.check_atp_permission("write")
        if not can_write:
            return False, reason

        # Check budget limit
        if self.atp_spent + amount > self.resource_limits.atp_budget:
            remaining = self.resource_limits.atp_budget - self.atp_spent
            return False, f"ATP budget exceeded (remaining: {remaining:.2f}, requested: {amount:.2f})"

        # Transfer allowed
        return True, f"ATP transfer allowed ({amount:.2f} within budget)"

    def record_atp_transfer(self, amount: float):
        """
        Record ATP transfer for budget tracking

        Parameters:
        -----------
        amount : float
            ATP amount transferred
        """
        self.atp_spent += amount

    def check_delegation(self, target_task: str) -> Tuple[bool, str]:
        """
        Check if task can delegate to another task

        Parameters:
        -----------
        target_task : str
            Target task to delegate to

        Returns:
        --------
        Tuple[bool, str]
            (allowed, reason)
        """
        if not self.can_delegate:
            return False, f"Task '{self.task}' cannot delegate"

        if target_task not in TASK_PERMISSIONS:
            return False, f"Unknown target task: {target_task}"

        return True, f"Delegation from '{self.task}' to '{target_task}' allowed"

    def check_code_execution(self) -> Tuple[bool, str]:
        """
        Check if task can execute code

        Returns:
        --------
        Tuple[bool, str]
            (allowed, reason)
        """
        if self.can_execute_code:
            return True, f"Task '{self.task}' can execute code"
        else:
            return False, f"Task '{self.task}' cannot execute code"

    def check_task_limit(self) -> Tuple[bool, str]:
        """
        Check if can start new concurrent task

        Returns:
        --------
        Tuple[bool, str]
            (allowed, reason)
        """
        if self.tasks_running >= self.resource_limits.max_concurrent_tasks:
            return False, f"Task limit reached ({self.tasks_running}/{self.resource_limits.max_concurrent_tasks})"

        return True, f"Can start task ({self.tasks_running + 1}/{self.resource_limits.max_concurrent_tasks})"

    def start_task(self):
        """Increment running task count"""
        self.tasks_running += 1

    def end_task(self):
        """Decrement running task count"""
        self.tasks_running = max(0, self.tasks_running - 1)

    def get_resource_summary(self) -> Dict[str, Any]:
        """
        Get resource usage summary

        Returns:
        --------
        Dict with resource usage and limits
        """
        return {
            "task": self.task,
            "atp": {
                "spent": self.atp_spent,
                "budget": self.resource_limits.atp_budget,
                "remaining": self.resource_limits.atp_budget - self.atp_spent,
                "percent_used": (self.atp_spent / self.resource_limits.atp_budget * 100)
                    if self.resource_limits.atp_budget > 0 else 0
            },
            "tasks": {
                "running": self.tasks_running,
                "max": self.resource_limits.max_concurrent_tasks,
                "available": self.resource_limits.max_concurrent_tasks - self.tasks_running
            },
            "memory_mb": self.resource_limits.memory_mb,
            "cpu_cores": self.resource_limits.cpu_cores,
            "permissions": {
                "atp": [p.value for p in self.atp_permissions],
                "can_delegate": self.can_delegate,
                "can_execute_code": self.can_execute_code
            }
        }


# Convenience functions for SAGE consciousness integration

def create_permission_checker(task: str) -> LCTATPPermissionChecker:
    """
    Create permission checker for task

    Parameters:
    -----------
    task : str
        LCT task scope

    Returns:
    --------
    LCTATPPermissionChecker
        Permission checker instance
    """
    return LCTATPPermissionChecker(task)


def get_task_permissions(task: str) -> Optional[Dict[str, Any]]:
    """
    Get permission configuration for task

    Parameters:
    -----------
    task : str
        LCT task scope

    Returns:
    --------
    Dict or None
        Task configuration or None if unknown task
    """
    return TASK_PERMISSIONS.get(task)


def list_available_tasks() -> list[str]:
    """Get list of available task scopes"""
    return list(TASK_PERMISSIONS.keys())
