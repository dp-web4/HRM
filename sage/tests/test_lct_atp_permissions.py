"""
Tests for LCT-Aware ATP Permissions

Tests task-based permission checking for ATP operations, including:
- Permission validation (read/write/all)
- ATP transfer budget limits
- Delegation permission checking
- Code execution permissions
- Concurrent task limits
- Resource usage tracking

Author: Thor (SAGE autonomous research)
Date: 2025-12-02
Session: Autonomous SAGE Development - LCT-Aware ATP Operations
"""

import pytest
from pathlib import Path
import sys

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from core.lct_atp_permissions import (
    ATPPermission,
    ResourceLimits,
    LCTATPPermissionChecker,
    TASK_PERMISSIONS,
    create_permission_checker,
    get_task_permissions,
    list_available_tasks
)


class TestResourceLimits:
    """Test ResourceLimits dataclass"""

    def test_resource_limits_creation(self):
        """Test creating resource limits"""
        limits = ResourceLimits(
            atp_budget=100.0,
            memory_mb=1024,
            cpu_cores=2,
            max_concurrent_tasks=10
        )

        assert limits.atp_budget == 100.0
        assert limits.memory_mb == 1024
        assert limits.cpu_cores == 2
        assert limits.max_concurrent_tasks == 10

    def test_resource_limits_defaults(self):
        """Test resource limits with defaults"""
        limits = ResourceLimits()

        assert limits.atp_budget == 0.0
        assert limits.memory_mb == 0
        assert limits.cpu_cores == 0
        assert limits.max_concurrent_tasks == 0


class TestTaskPermissions:
    """Test TASK_PERMISSIONS configuration"""

    def test_all_tasks_defined(self):
        """Test that expected tasks are defined"""
        expected_tasks = [
            "perception",
            "planning",
            "planning.strategic",
            "execution.safe",
            "execution.code",
            "delegation.federation",
            "consciousness",
            "admin.readonly",
            "admin.full"
        ]

        for task in expected_tasks:
            assert task in TASK_PERMISSIONS

    def test_task_permission_structure(self):
        """Test that each task has required permission fields"""
        for task, config in TASK_PERMISSIONS.items():
            assert "atp_permissions" in config
            assert "can_delegate" in config
            assert "can_execute_code" in config
            assert "resource_limits" in config

            # Check types
            assert isinstance(config["atp_permissions"], set)
            assert isinstance(config["can_delegate"], bool)
            assert isinstance(config["can_execute_code"], bool)
            assert isinstance(config["resource_limits"], ResourceLimits)

    def test_perception_permissions(self):
        """Test perception task has read-only permissions"""
        config = TASK_PERMISSIONS["perception"]

        assert ATPPermission.READ in config["atp_permissions"]
        assert ATPPermission.WRITE not in config["atp_permissions"]
        assert config["can_delegate"] is False
        assert config["can_execute_code"] is False

    def test_consciousness_permissions(self):
        """Test consciousness task has full permissions"""
        config = TASK_PERMISSIONS["consciousness"]

        assert ATPPermission.READ in config["atp_permissions"]
        assert ATPPermission.WRITE in config["atp_permissions"]
        assert config["can_delegate"] is True
        assert config["can_execute_code"] is True

    def test_admin_full_permissions(self):
        """Test admin.full has all permissions"""
        config = TASK_PERMISSIONS["admin.full"]

        assert ATPPermission.ALL in config["atp_permissions"]
        assert config["can_delegate"] is True
        assert config["can_execute_code"] is True
        assert config["resource_limits"].atp_budget == float('inf')


class TestLCTATPPermissionChecker:
    """Test LCTATPPermissionChecker class"""

    def test_checker_initialization(self):
        """Test creating permission checker"""
        checker = LCTATPPermissionChecker(task="consciousness")

        assert checker.task == "consciousness"
        assert checker.atp_spent == 0.0
        assert checker.tasks_running == 0

    def test_checker_invalid_task(self):
        """Test creating checker with invalid task fails"""
        with pytest.raises(ValueError, match="Unknown task"):
            LCTATPPermissionChecker(task="invalid_task")

    def test_check_atp_permission_read(self):
        """Test checking read permission"""
        # Perception has READ permission
        checker = LCTATPPermissionChecker(task="perception")
        can_read, reason = checker.check_atp_permission("read")

        assert can_read is True
        assert "permission 'read'" in reason

    def test_check_atp_permission_write_denied(self):
        """Test checking write permission when not allowed"""
        # Perception does NOT have WRITE permission
        checker = LCTATPPermissionChecker(task="perception")
        can_write, reason = checker.check_atp_permission("write")

        assert can_write is False
        assert "lacks permission 'write'" in reason

    def test_check_atp_permission_write_allowed(self):
        """Test checking write permission when allowed"""
        # Consciousness has WRITE permission
        checker = LCTATPPermissionChecker(task="consciousness")
        can_write, reason = checker.check_atp_permission("write")

        assert can_write is True

    def test_check_atp_permission_all(self):
        """Test checking ALL permission"""
        # Admin.full has ALL permission
        checker = LCTATPPermissionChecker(task="admin.full")
        can_all, reason = checker.check_atp_permission("all")

        assert can_all is True
        assert "permission 'all'" in reason

    def test_check_atp_permission_invalid_operation(self):
        """Test checking invalid operation"""
        checker = LCTATPPermissionChecker(task="consciousness")
        allowed, reason = checker.check_atp_permission("invalid_op")

        assert allowed is False
        assert "Invalid ATP operation" in reason

    def test_check_atp_transfer_allowed(self):
        """Test ATP transfer within budget"""
        checker = LCTATPPermissionChecker(task="consciousness")

        # Consciousness has 1000.0 ATP budget
        allowed, reason = checker.check_atp_transfer(
            amount=50.0,
            from_lct="lct:web4:agent:dp@Thor#consciousness",
            to_lct="lct:web4:agent:dp@Sprout#perception"
        )

        assert allowed is True
        assert "within budget" in reason

    def test_check_atp_transfer_denied_no_permission(self):
        """Test ATP transfer denied when no write permission"""
        checker = LCTATPPermissionChecker(task="perception")

        # Perception does NOT have WRITE permission
        allowed, reason = checker.check_atp_transfer(
            amount=10.0,
            from_lct="lct:web4:agent:dp@Thor#perception",
            to_lct="lct:web4:agent:dp@Sprout#perception"
        )

        assert allowed is False
        assert "lacks permission" in reason

    def test_check_atp_transfer_denied_budget_exceeded(self):
        """Test ATP transfer denied when budget exceeded"""
        checker = LCTATPPermissionChecker(task="execution.safe")

        # execution.safe has 200.0 ATP budget
        # Try to transfer more than budget
        allowed, reason = checker.check_atp_transfer(
            amount=300.0,
            from_lct="lct:web4:agent:dp@Thor#execution.safe",
            to_lct="lct:web4:agent:dp@Sprout#perception"
        )

        assert allowed is False
        assert "budget exceeded" in reason

    def test_record_atp_transfer(self):
        """Test recording ATP transfer updates budget"""
        checker = LCTATPPermissionChecker(task="consciousness")

        assert checker.atp_spent == 0.0

        checker.record_atp_transfer(50.0)
        assert checker.atp_spent == 50.0

        checker.record_atp_transfer(25.0)
        assert checker.atp_spent == 75.0

    def test_check_atp_transfer_after_spending(self):
        """Test ATP transfer validation after spending"""
        checker = LCTATPPermissionChecker(task="execution.safe")

        # execution.safe has 200.0 ATP budget
        # Spend 150.0
        checker.record_atp_transfer(150.0)

        # Try to spend another 100.0 (would exceed budget)
        allowed, reason = checker.check_atp_transfer(
            amount=100.0,
            from_lct="lct:web4:agent:dp@Thor#execution.safe",
            to_lct="lct:web4:agent:dp@Sprout#perception"
        )

        assert allowed is False
        assert "budget exceeded" in reason
        assert "remaining: 50.00" in reason

    def test_check_delegation_allowed(self):
        """Test delegation when allowed"""
        checker = LCTATPPermissionChecker(task="consciousness")

        # Consciousness CAN delegate
        allowed, reason = checker.check_delegation("perception")

        assert allowed is True
        assert "Delegation" in reason
        assert "allowed" in reason

    def test_check_delegation_denied_no_permission(self):
        """Test delegation when not allowed"""
        checker = LCTATPPermissionChecker(task="perception")

        # Perception CANNOT delegate
        allowed, reason = checker.check_delegation("planning")

        assert allowed is False
        assert "cannot delegate" in reason

    def test_check_delegation_invalid_target(self):
        """Test delegation to invalid target task"""
        checker = LCTATPPermissionChecker(task="consciousness")

        allowed, reason = checker.check_delegation("invalid_task")

        assert allowed is False
        assert "Unknown target task" in reason

    def test_check_code_execution_allowed(self):
        """Test code execution when allowed"""
        checker = LCTATPPermissionChecker(task="execution.code")

        allowed, reason = checker.check_code_execution()

        assert allowed is True
        assert "can execute code" in reason

    def test_check_code_execution_denied(self):
        """Test code execution when not allowed"""
        checker = LCTATPPermissionChecker(task="perception")

        allowed, reason = checker.check_code_execution()

        assert allowed is False
        assert "cannot execute code" in reason

    def test_check_task_limit_allowed(self):
        """Test starting task when under limit"""
        checker = LCTATPPermissionChecker(task="consciousness")

        # Consciousness has max 100 concurrent tasks
        allowed, reason = checker.check_task_limit()

        assert allowed is True
        assert "Can start task" in reason

    def test_check_task_limit_denied(self):
        """Test starting task when at limit"""
        checker = LCTATPPermissionChecker(task="perception")

        # Perception has max 5 concurrent tasks
        # Manually set to limit
        checker.tasks_running = 5

        allowed, reason = checker.check_task_limit()

        assert allowed is False
        assert "Task limit reached" in reason

    def test_start_and_end_task(self):
        """Test task counting"""
        checker = LCTATPPermissionChecker(task="consciousness")

        assert checker.tasks_running == 0

        checker.start_task()
        assert checker.tasks_running == 1

        checker.start_task()
        assert checker.tasks_running == 2

        checker.end_task()
        assert checker.tasks_running == 1

        checker.end_task()
        assert checker.tasks_running == 0

    def test_end_task_minimum_zero(self):
        """Test ending task doesn't go below zero"""
        checker = LCTATPPermissionChecker(task="consciousness")

        assert checker.tasks_running == 0

        checker.end_task()
        assert checker.tasks_running == 0  # Should stay at 0

    def test_get_resource_summary(self):
        """Test resource summary"""
        checker = LCTATPPermissionChecker(task="consciousness")

        # Spend some ATP and start tasks
        checker.record_atp_transfer(100.0)
        checker.start_task()
        checker.start_task()

        summary = checker.get_resource_summary()

        assert summary["task"] == "consciousness"
        assert summary["atp"]["spent"] == 100.0
        assert summary["atp"]["budget"] == 1000.0
        assert summary["atp"]["remaining"] == 900.0
        assert summary["atp"]["percent_used"] == 10.0
        assert summary["tasks"]["running"] == 2
        assert summary["tasks"]["max"] == 100
        assert summary["tasks"]["available"] == 98
        assert summary["memory_mb"] == 16384
        assert summary["cpu_cores"] == 8
        assert "atp:read" in summary["permissions"]["atp"]
        assert "atp:write" in summary["permissions"]["atp"]
        assert summary["permissions"]["can_delegate"] is True
        assert summary["permissions"]["can_execute_code"] is True


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_create_permission_checker(self):
        """Test create_permission_checker convenience function"""
        checker = create_permission_checker("consciousness")

        assert isinstance(checker, LCTATPPermissionChecker)
        assert checker.task == "consciousness"

    def test_get_task_permissions(self):
        """Test get_task_permissions convenience function"""
        config = get_task_permissions("perception")

        assert config is not None
        assert "atp_permissions" in config
        assert ATPPermission.READ in config["atp_permissions"]

    def test_get_task_permissions_invalid(self):
        """Test get_task_permissions with invalid task"""
        config = get_task_permissions("invalid_task")

        assert config is None

    def test_list_available_tasks(self):
        """Test list_available_tasks convenience function"""
        tasks = list_available_tasks()

        assert isinstance(tasks, list)
        assert "consciousness" in tasks
        assert "perception" in tasks
        assert "planning" in tasks
        assert len(tasks) == 9  # All defined tasks


class TestPermissionScenarios:
    """Test realistic permission scenarios"""

    def test_perception_agent_scenario(self):
        """Test perception agent with read-only permissions"""
        checker = create_permission_checker("perception")

        # Can read ATP balances
        can_read, _ = checker.check_atp_permission("read")
        assert can_read is True

        # Cannot transfer ATP
        can_transfer, _ = checker.check_atp_transfer(
            amount=10.0,
            from_lct="lct:web4:agent:dp@Thor#perception",
            to_lct="lct:web4:agent:dp@Sprout#perception"
        )
        assert can_transfer is False

        # Cannot delegate
        can_delegate, _ = checker.check_delegation("planning")
        assert can_delegate is False

        # Cannot execute code
        can_execute, _ = checker.check_code_execution()
        assert can_execute is False

    def test_consciousness_agent_scenario(self):
        """Test consciousness agent with full permissions"""
        checker = create_permission_checker("consciousness")

        # Can read ATP
        can_read, _ = checker.check_atp_permission("read")
        assert can_read is True

        # Can transfer ATP
        can_transfer, _ = checker.check_atp_transfer(
            amount=100.0,
            from_lct="lct:web4:agent:dp@Thor#consciousness",
            to_lct="lct:web4:agent:dp@Sprout#perception"
        )
        assert can_transfer is True

        # Can delegate
        can_delegate, _ = checker.check_delegation("perception")
        assert can_delegate is True

        # Can execute code
        can_execute, _ = checker.check_code_execution()
        assert can_execute is True

    def test_execution_agent_budget_management(self):
        """Test execution agent managing ATP budget"""
        checker = create_permission_checker("execution.code")

        # execution.code has 500.0 ATP budget
        config = get_task_permissions("execution.code")
        assert config["resource_limits"].atp_budget == 500.0

        # Transfer 200.0 (allowed)
        allowed, _ = checker.check_atp_transfer(
            amount=200.0,
            from_lct="lct:web4:agent:dp@Thor#execution.code",
            to_lct="lct:web4:agent:dp@Sprout#perception"
        )
        assert allowed is True
        checker.record_atp_transfer(200.0)

        # Transfer another 200.0 (allowed)
        allowed, _ = checker.check_atp_transfer(
            amount=200.0,
            from_lct="lct:web4:agent:dp@Thor#execution.code",
            to_lct="lct:web4:agent:dp@Sprout#perception"
        )
        assert allowed is True
        checker.record_atp_transfer(200.0)

        # Transfer another 200.0 (denied - would exceed 500.0 budget)
        allowed, reason = checker.check_atp_transfer(
            amount=200.0,
            from_lct="lct:web4:agent:dp@Thor#execution.code",
            to_lct="lct:web4:agent:dp@Sprout#perception"
        )
        assert allowed is False
        assert "budget exceeded" in reason

    def test_admin_full_unlimited(self):
        """Test admin.full has unlimited resources"""
        checker = create_permission_checker("admin.full")

        # Has ALL permission
        can_all, _ = checker.check_atp_permission("all")
        assert can_all is True

        # Can transfer huge amounts
        can_transfer, _ = checker.check_atp_transfer(
            amount=1000000.0,
            from_lct="lct:web4:agent:system:genesis@Thor#admin.full",
            to_lct="lct:web4:agent:dp@Thor#consciousness"
        )
        assert can_transfer is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
