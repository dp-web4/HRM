"""
Tests for ATP Permission Integration with RealSAGEConsciousness

Tests the integration of LCT-aware ATP permissions with the consciousness loop:
- Permission checker initialization
- ATP transfer with permission validation
- Budget tracking and enforcement
- Permission queries
- Resource summaries

Author: Thor (SAGE autonomous research)
Date: 2025-12-02
Session: Autonomous SAGE Development - Permission Integration with Consciousness
"""

import pytest
from pathlib import Path
import sys
import time

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from core.sage_consciousness_real import RealSAGEConsciousness


class TestConsciousnessPermissionInitialization:
    """Test permission checker initialization in consciousness"""

    def test_consciousness_initializes_permission_checker(self):
        """Test that consciousness initializes permission checker"""
        sage = RealSAGEConsciousness(
            task="consciousness",
            initial_atp=100.0
        )

        # Permission checker should be initialized
        assert hasattr(sage, 'permission_checker')
        assert sage.permission_checker is not None
        assert sage.permission_checker.task == "consciousness"

    def test_consciousness_has_correct_task_permissions(self):
        """Test consciousness task has correct permissions"""
        sage = RealSAGEConsciousness(
            task="consciousness",
            initial_atp=100.0
        )

        # Check permission configuration
        config = sage.permission_checker.task_config

        assert config['can_delegate'] is True
        assert config['can_execute_code'] is True
        assert config['resource_limits'].atp_budget == 1000.0

    def test_perception_task_has_limited_permissions(self):
        """Test perception task has read-only permissions"""
        sage = RealSAGEConsciousness(
            task="perception",
            initial_atp=100.0
        )

        config = sage.permission_checker.task_config

        assert config['can_delegate'] is False
        assert config['can_execute_code'] is False
        assert config['resource_limits'].atp_budget == 100.0


class TestATPTransferWithPermissions:
    """Test ATP transfer with permission checking"""

    def test_consciousness_can_transfer_atp(self):
        """Test consciousness can transfer ATP (has write permission)"""
        sage = RealSAGEConsciousness(
            task="consciousness",
            initial_atp=100.0
        )

        # Transfer ATP to another identity
        success, msg = sage.transfer_atp(
            amount=50.0,
            to_lct_uri="lct:web4:agent:dp@Sprout#perception",
            reason="Test transfer"
        )

        assert success is True
        assert "ATP Transfer" in msg
        assert sage.metabolic.atp_current == 50.0  # 100 - 50
        assert sage.permission_checker.atp_spent == 50.0

    def test_perception_cannot_transfer_atp(self):
        """Test perception task cannot transfer ATP (no write permission)"""
        sage = RealSAGEConsciousness(
            task="perception",
            initial_atp=100.0
        )

        # Try to transfer ATP (should fail)
        success, msg = sage.transfer_atp(
            amount=10.0,
            to_lct_uri="lct:web4:agent:dp@Sprout#perception",
            reason="Test transfer"
        )

        assert success is False
        assert "Permission denied" in msg
        assert "lacks permission 'write'" in msg
        assert sage.metabolic.atp_current == 100.0  # No change
        assert sage.permission_checker.atp_spent == 0.0  # Not recorded

    def test_transfer_fails_insufficient_atp(self):
        """Test transfer fails when insufficient ATP in metabolic system"""
        sage = RealSAGEConsciousness(
            task="consciousness",
            initial_atp=100.0
        )

        # Try to transfer more ATP than available
        success, msg = sage.transfer_atp(
            amount=150.0,
            to_lct_uri="lct:web4:agent:dp@Sprout#perception"
        )

        assert success is False
        assert "Insufficient ATP" in msg
        assert sage.metabolic.atp_current == 100.0  # No change
        assert sage.permission_checker.atp_spent == 0.0  # Not recorded

    def test_transfer_fails_budget_exceeded(self):
        """Test transfer fails when budget limit exceeded"""
        sage = RealSAGEConsciousness(
            task="execution.safe",  # 200 ATP budget
            initial_atp=500.0  # More ATP than budget allows spending
        )

        # Transfer within metabolic ATP but exceeds budget
        success1, msg1 = sage.transfer_atp(
            amount=150.0,
            to_lct_uri="lct:web4:agent:dp@Sprout#perception"
        )
        assert success1 is True  # First transfer succeeds

        # Try another transfer that would exceed 200 ATP budget
        success2, msg2 = sage.transfer_atp(
            amount=100.0,
            to_lct_uri="lct:web4:agent:dp@Sprout#perception"
        )
        assert success2 is False
        assert "Permission denied" in msg2
        assert "budget exceeded" in msg2

    def test_multiple_transfers_track_cumulative_spending(self):
        """Test multiple transfers track cumulative ATP spending"""
        sage = RealSAGEConsciousness(
            task="consciousness",  # 1000 ATP budget
            initial_atp=500.0
        )

        # Transfer 1
        success1, _ = sage.transfer_atp(
            amount=100.0,
            to_lct_uri="lct:web4:agent:dp@Sprout#perception"
        )
        assert success1 is True
        assert sage.permission_checker.atp_spent == 100.0
        assert sage.metabolic.atp_current == 400.0

        # Transfer 2
        success2, _ = sage.transfer_atp(
            amount=50.0,
            to_lct_uri="lct:web4:agent:dp@Sprout#planning"
        )
        assert success2 is True
        assert sage.permission_checker.atp_spent == 150.0
        assert sage.metabolic.atp_current == 350.0

        # Transfer 3
        success3, _ = sage.transfer_atp(
            amount=75.0,
            to_lct_uri="lct:web4:agent:dp@Sprout#execution.safe"
        )
        assert success3 is True
        assert sage.permission_checker.atp_spent == 225.0
        assert sage.metabolic.atp_current == 275.0


class TestPermissionChecking:
    """Test permission checking methods"""

    def test_check_atp_permission_consciousness(self):
        """Test consciousness can check its ATP permissions"""
        sage = RealSAGEConsciousness(
            task="consciousness",
            initial_atp=100.0
        )

        # Check read permission
        can_read, reason = sage.check_atp_permission("read")
        assert can_read is True

        # Check write permission
        can_write, reason = sage.check_atp_permission("write")
        assert can_write is True

    def test_check_atp_permission_perception(self):
        """Test perception can only read ATP"""
        sage = RealSAGEConsciousness(
            task="perception",
            initial_atp=100.0
        )

        # Check read permission
        can_read, reason = sage.check_atp_permission("read")
        assert can_read is True

        # Check write permission (should fail)
        can_write, reason = sage.check_atp_permission("write")
        assert can_write is False
        assert "lacks permission" in reason


class TestResourceSummary:
    """Test ATP resource summary methods"""

    def test_get_atp_resource_summary_structure(self):
        """Test resource summary has correct structure"""
        sage = RealSAGEConsciousness(
            task="consciousness",
            initial_atp=100.0
        )

        summary = sage.get_atp_resource_summary()

        # Check structure
        assert 'task' in summary
        assert 'atp' in summary
        assert 'tasks' in summary
        assert 'memory_mb' in summary
        assert 'cpu_cores' in summary
        assert 'permissions' in summary
        assert 'metabolic_atp' in summary

        # Check ATP details
        assert summary['task'] == "consciousness"
        assert summary['atp']['budget'] == 1000.0
        assert summary['atp']['spent'] == 0.0
        assert summary['atp']['remaining'] == 1000.0

        # Check metabolic ATP
        assert summary['metabolic_atp']['current'] == 100.0

    def test_resource_summary_after_transfers(self):
        """Test resource summary reflects ATP transfers"""
        sage = RealSAGEConsciousness(
            task="consciousness",
            initial_atp=200.0
        )

        # Make some transfers
        sage.transfer_atp(100.0, "lct:web4:agent:dp@Sprout#perception")
        sage.transfer_atp(50.0, "lct:web4:agent:dp@Sprout#planning")

        summary = sage.get_atp_resource_summary()

        # Check ATP tracking
        assert summary['atp']['spent'] == 150.0
        assert summary['atp']['remaining'] == 850.0  # 1000 budget - 150 spent
        assert summary['atp']['percent_used'] == 15.0

        # Check metabolic ATP
        assert summary['metabolic_atp']['current'] == 50.0  # 200 - 100 - 50

    def test_resource_summary_permissions(self):
        """Test resource summary includes permission details"""
        sage = RealSAGEConsciousness(
            task="consciousness",
            initial_atp=100.0
        )

        summary = sage.get_atp_resource_summary()

        # Check permissions
        assert 'atp' in summary['permissions']
        assert 'can_delegate' in summary['permissions']
        assert 'can_execute_code' in summary['permissions']

        assert summary['permissions']['can_delegate'] is True
        assert summary['permissions']['can_execute_code'] is True


class TestDifferentTaskTypes:
    """Test different task types with various permissions"""

    def test_planning_task_permissions(self):
        """Test planning task permissions"""
        sage = RealSAGEConsciousness(
            task="planning",
            initial_atp=100.0
        )

        # Planning can read but not write
        can_read, _ = sage.check_atp_permission("read")
        can_write, _ = sage.check_atp_permission("write")

        assert can_read is True
        assert can_write is False

        # Cannot delegate
        summary = sage.get_atp_resource_summary()
        assert summary['permissions']['can_delegate'] is False
        assert summary['atp']['budget'] == 100.0

    def test_execution_code_task_permissions(self):
        """Test execution.code task permissions"""
        sage = RealSAGEConsciousness(
            task="execution.code",
            initial_atp=500.0
        )

        # Can read and write
        can_read, _ = sage.check_atp_permission("read")
        can_write, _ = sage.check_atp_permission("write")

        assert can_read is True
        assert can_write is True

        # Can execute code but cannot delegate
        summary = sage.get_atp_resource_summary()
        assert summary['permissions']['can_execute_code'] is True
        assert summary['permissions']['can_delegate'] is False
        assert summary['atp']['budget'] == 500.0

    def test_admin_full_unlimited_permissions(self):
        """Test admin.full has unlimited permissions"""
        sage = RealSAGEConsciousness(
            task="admin.full",
            initial_atp=10000.0
        )

        # Has ALL permission
        can_all, _ = sage.check_atp_permission("all")
        assert can_all is True

        # Unlimited budget
        summary = sage.get_atp_resource_summary()
        assert summary['atp']['budget'] == float('inf')
        assert summary['permissions']['can_delegate'] is True
        assert summary['permissions']['can_execute_code'] is True


class TestIdentityIntegration:
    """Test LCT identity integration with permissions"""

    def test_identity_and_permissions_both_initialized(self):
        """Test both identity and permissions are initialized"""
        sage = RealSAGEConsciousness(
            lineage="dp",
            task="consciousness",
            initial_atp=100.0
        )

        # Check identity exists
        assert hasattr(sage, 'lct_identity')
        assert sage.lct_identity.task == "consciousness"
        # Note: lineage may be from persisted identity, so check it exists but not specific value
        assert sage.lct_identity.lineage is not None
        assert len(sage.lct_identity.lineage) > 0

        # Check permissions
        assert hasattr(sage, 'permission_checker')
        assert sage.permission_checker.task == "consciousness"

    def test_transfer_uses_identity_uri(self):
        """Test ATP transfer uses LCT identity URI"""
        sage = RealSAGEConsciousness(
            lineage="dp",
            task="consciousness",
            initial_atp=100.0
        )

        # Transfer should use identity URI as source
        success, msg = sage.transfer_atp(
            amount=25.0,
            to_lct_uri="lct:web4:agent:dp@Sprout#perception",
            reason="Test"
        )

        assert success is True
        # Message should contain source LCT URI
        lct_uri = str(sage.lct_identity)
        assert lct_uri in msg
        assert "dp@" in msg  # Lineage present
        assert "#consciousness" in msg  # Task present


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
