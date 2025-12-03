"""
Test consciousness.sage enhanced variant with memory management.

This test validates that the consciousness.sage task type:
1. Provides double ATP budget compared to standard consciousness
2. Provides enhanced resource limits (memory, CPU)
3. Includes can_delete_memories permission for long-running loops
4. Is compatible with LUPS v1.0 specification
"""

import pytest
from pathlib import Path
import sys

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from core.lct_atp_permissions import (
    create_permission_checker,
    get_task_permissions,
    ATPPermission
)


class TestConsciousnessSageVariant:
    """Test the enhanced consciousness.sage variant"""

    def test_consciousness_sage_exists(self):
        """Verify consciousness.sage task type exists"""
        checker = create_permission_checker("consciousness.sage")
        assert checker is not None
        assert checker.task == "consciousness.sage"

    def test_consciousness_sage_enhanced_atp_budget(self):
        """Verify consciousness.sage has double ATP budget"""
        standard = get_task_permissions("consciousness")
        enhanced = get_task_permissions("consciousness.sage")

        # Enhanced should have 2x ATP budget
        assert enhanced['resource_limits'].atp_budget == 2000.0
        assert standard['resource_limits'].atp_budget == 1000.0
        assert enhanced['resource_limits'].atp_budget == 2 * standard['resource_limits'].atp_budget

    def test_consciousness_sage_enhanced_memory(self):
        """Verify consciousness.sage has double memory"""
        standard = get_task_permissions("consciousness")
        enhanced = get_task_permissions("consciousness.sage")

        # Enhanced should have 2x memory
        assert enhanced['resource_limits'].memory_mb == 32768  # 32 GB
        assert standard['resource_limits'].memory_mb == 16384  # 16 GB
        assert enhanced['resource_limits'].memory_mb == 2 * standard['resource_limits'].memory_mb

    def test_consciousness_sage_enhanced_cpu(self):
        """Verify consciousness.sage has double CPU cores"""
        standard = get_task_permissions("consciousness")
        enhanced = get_task_permissions("consciousness.sage")

        # Enhanced should have 2x CPU cores
        assert enhanced['resource_limits'].cpu_cores == 16
        assert standard['resource_limits'].cpu_cores == 8
        assert enhanced['resource_limits'].cpu_cores == 2 * standard['resource_limits'].cpu_cores

    def test_consciousness_sage_enhanced_concurrent_tasks(self):
        """Verify consciousness.sage has double concurrent tasks"""
        standard = get_task_permissions("consciousness")
        enhanced = get_task_permissions("consciousness.sage")

        # Enhanced should have 2x concurrent tasks
        assert enhanced['resource_limits'].max_concurrent_tasks == 200
        assert standard['resource_limits'].max_concurrent_tasks == 100
        assert enhanced['resource_limits'].max_concurrent_tasks == 2 * standard['resource_limits'].max_concurrent_tasks

    def test_consciousness_sage_memory_delete_permission(self):
        """Verify consciousness.sage has memory delete permission"""
        standard = get_task_permissions("consciousness")
        enhanced = get_task_permissions("consciousness.sage")

        # Standard consciousness should NOT have memory delete
        assert not standard.get('can_delete_memories', False)

        # Enhanced consciousness.sage SHOULD have memory delete
        assert enhanced.get('can_delete_memories', False) is True

    def test_consciousness_sage_atp_permissions(self):
        """Verify consciousness.sage has READ+WRITE ATP permissions"""
        enhanced = get_task_permissions("consciousness.sage")

        # Should have READ and WRITE permissions
        assert ATPPermission.READ in enhanced['atp_permissions']
        assert ATPPermission.WRITE in enhanced['atp_permissions']
        assert len(enhanced['atp_permissions']) == 2

    def test_consciousness_sage_delegation(self):
        """Verify consciousness.sage can delegate tasks"""
        enhanced = get_task_permissions("consciousness.sage")
        assert enhanced['can_delegate'] is True

    def test_consciousness_sage_code_execution(self):
        """Verify consciousness.sage can execute code"""
        enhanced = get_task_permissions("consciousness.sage")
        assert enhanced['can_execute_code'] is True

    def test_consciousness_vs_consciousness_sage_compatibility(self):
        """Verify consciousness and consciousness.sage are compatible"""
        standard = get_task_permissions("consciousness")
        enhanced = get_task_permissions("consciousness.sage")

        # Both should have same permission types (enhanced has more resources)
        assert standard['atp_permissions'] == enhanced['atp_permissions']
        assert standard['can_delegate'] == enhanced['can_delegate']
        assert standard['can_execute_code'] == enhanced['can_execute_code']

        # Enhanced should have memory delete (only difference in permissions)
        assert not standard.get('can_delete_memories', False)
        assert enhanced.get('can_delete_memories', False)


class TestConsciousnessSagePermissionChecker:
    """Test permission checker with consciousness.sage variant"""

    def test_create_consciousness_sage_checker(self):
        """Test creating permission checker for consciousness.sage"""
        checker = create_permission_checker("consciousness.sage")
        assert checker.task == "consciousness.sage"

        # Verify enhanced ATP budget
        assert checker.task_config['resource_limits'].atp_budget == 2000.0

    def test_consciousness_sage_atp_operations(self):
        """Test ATP operations with consciousness.sage permissions"""
        checker = create_permission_checker("consciousness.sage")

        # Should allow READ operations
        assert checker.check_atp_permission(ATPPermission.READ)

        # Should allow WRITE operations
        assert checker.check_atp_permission(ATPPermission.WRITE)

    def test_consciousness_sage_resource_summary(self):
        """Test resource summary for consciousness.sage"""
        checker = create_permission_checker("consciousness.sage")
        summary = checker.get_resource_summary()

        # Verify enhanced resources in summary
        assert summary['atp']['budget'] == 2000.0
        assert summary['memory_mb'] == 32768
        assert summary['cpu_cores'] == 16
        assert summary['tasks']['max'] == 200

    def test_consciousness_sage_budget_tracking(self):
        """Test ATP budget tracking with consciousness.sage"""
        checker = create_permission_checker("consciousness.sage")

        # Record ATP spending
        checker.record_atp_transfer(50.0)

        # Verify spending tracked
        summary = checker.get_resource_summary()
        assert summary['atp']['spent'] == 50.0

        # Should still be under enhanced budget
        assert summary['atp']['spent'] < summary['atp']['budget']


class TestConsciousnessSageLUPSCompatibility:
    """Test LUPS v1.0 compatibility for consciousness.sage"""

    def test_lups_compatible_task_name(self):
        """Verify consciousness.sage follows LUPS naming convention"""
        enhanced = get_task_permissions("consciousness.sage")
        assert enhanced is not None
        # Task name should follow dot notation (base.variant)

    def test_lups_compatible_permissions(self):
        """Verify consciousness.sage follows LUPS permission structure"""
        enhanced = get_task_permissions("consciousness.sage")

        # LUPS requires these fields
        assert 'atp_permissions' in enhanced
        assert 'can_delegate' in enhanced
        assert 'can_execute_code' in enhanced
        assert 'resource_limits' in enhanced

    def test_lups_compatible_resource_limits(self):
        """Verify consciousness.sage resource limits follow LUPS structure"""
        enhanced = get_task_permissions("consciousness.sage")
        limits = enhanced['resource_limits']

        # LUPS requires these resource limit fields
        assert hasattr(limits, 'atp_budget')
        assert hasattr(limits, 'memory_mb')
        assert hasattr(limits, 'cpu_cores')
        assert hasattr(limits, 'max_concurrent_tasks')

    def test_consciousness_sage_memory_management_use_case(self):
        """Document consciousness.sage memory management use case"""
        enhanced = get_task_permissions("consciousness.sage")

        # Use case: Long-running SAGE consciousness loops need memory management
        # consciousness.sage provides:
        # 1. can_delete_memories permission (prune old memories)
        # 2. Double ATP budget for extended operation
        # 3. Double memory for handling large memory stores
        # 4. Double CPU for parallel memory operations

        assert enhanced.get('can_delete_memories', False) is True
        assert enhanced['resource_limits'].atp_budget == 2000.0
        assert enhanced['resource_limits'].memory_mb == 32768
        assert enhanced['resource_limits'].cpu_cores == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
