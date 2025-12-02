"""
Tests for LCT Identity Integration with SAGE Consciousness Loop

Tests that RealSAGEConsciousness properly initializes and uses LCT identity.

Author: Thor (SAGE autonomous research)
Date: 2025-12-02
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from core.lct_identity_integration import (
    LCTIdentity,
    LCTIdentityManager
)


class TestLCTConsciousnessIntegration:
    """Test LCT identity integration with consciousness loop"""

    def test_lct_identity_initialization(self):
        """Test that LCT identity can be initialized"""
        manager = LCTIdentityManager()
        identity = manager.create_identity(
            lineage="dp",
            task="consciousness",
            context="TestPlatform"
        )

        assert identity.lineage == "dp"
        assert identity.context == "TestPlatform"
        assert identity.task == "consciousness"

    def test_lct_uri_format(self):
        """Test LCT URI formatting"""
        manager = LCTIdentityManager()
        identity = manager.create_identity(
            lineage="dp",
            task="consciousness",
            context="Thor"
        )

        lct_uri = identity.to_lct_string()
        assert lct_uri == "lct:web4:agent:dp@Thor#consciousness"

    def test_identity_summary_structure(self):
        """Test identity summary has expected fields"""
        manager = LCTIdentityManager()
        identity = manager.create_identity(
            lineage="system:autonomous",
            task="consciousness",
            context="Thor"
        )

        summary = manager.get_identity_summary()

        # Check all expected fields present
        assert 'has_identity' in summary
        assert 'lct_uri' in summary
        assert 'lineage' in summary
        assert 'context' in summary
        assert 'task' in summary
        assert 'is_valid' in summary

        # Check values
        assert summary['has_identity'] is True
        assert summary['lineage'] == "system:autonomous"
        assert summary['context'] == "Thor"
        assert summary['task'] == "consciousness"
        assert summary['is_valid'] is True

    def test_multiple_task_scopes(self):
        """Test different task scopes"""
        manager = LCTIdentityManager()

        tasks = [
            "consciousness",
            "perception",
            "planning.strategic",
            "execution.code",
            "delegation.federation"
        ]

        for task in tasks:
            identity = manager.create_identity(
                lineage="dp",
                task=task,
                context="Thor"
            )

            assert identity.task == task
            lct_uri = identity.to_lct_string()
            assert lct_uri.endswith(f"#{task}")

    def test_hierarchical_lineage(self):
        """Test hierarchical lineage patterns"""
        manager = LCTIdentityManager()

        lineages = [
            "dp",
            "dp.assistant1",
            "dp.assistant1.task_manager",
            "org:anthropic",
            "system:genesis"
        ]

        for lineage in lineages:
            identity = manager.create_identity(
                lineage=lineage,
                task="consciousness",
                context="Thor"
            )

            assert identity.lineage == lineage
            lct_uri = identity.to_lct_string()
            assert f"{lineage}@Thor" in lct_uri

    def test_identity_persistence(self):
        """Test identity persists across manager instances"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Create identity in first manager
            manager1 = LCTIdentityManager(identity_dir=temp_dir)
            manager1._platform_context = "TestPlatform"

            identity1 = manager1.create_identity(
                lineage="dp",
                task="consciousness",
                context="TestPlatform"
            )
            manager1.save_identity(identity1)

            # Load identity in second manager
            manager2 = LCTIdentityManager(identity_dir=temp_dir)
            manager2._platform_context = "TestPlatform"

            identity2 = manager2.load_identity(context="TestPlatform")

            # Check identity persisted correctly
            assert identity2 is not None
            assert identity2.lineage == "dp"
            assert identity2.task == "consciousness"
            assert identity2.context == "TestPlatform"

        finally:
            shutil.rmtree(temp_dir)

    def test_identity_validation(self):
        """Test identity validation logic"""
        manager = LCTIdentityManager()

        # Valid identity
        valid_identity = manager.create_identity(
            lineage="dp",
            task="consciousness",
            context="Thor"
        )
        is_valid, reason = manager.validate_identity(valid_identity)
        assert is_valid is True
        assert reason == "Valid"

        # Invalid: empty lineage
        invalid_identity = LCTIdentity(
            lineage="",
            context="Thor",
            task="consciousness"
        )
        is_valid, reason = manager.validate_identity(invalid_identity)
        assert is_valid is False
        assert "lineage" in reason.lower()


# Note: RealSAGEConsciousness tests require LLM models, so we test the
# identity integration separately from the full consciousness loop


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
