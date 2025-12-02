"""
Tests for LCT Identity Integration with SAGE Consciousness

Tests:
1. LCTIdentity creation and formatting
2. Platform context detection
3. Identity persistence (save/load)
4. Identity validation
5. Integration with SAGE consciousness loop

Author: Thor (SAGE autonomous research)
Date: 2025-12-01
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path

from sage.core.lct_identity_integration import (
    LCTIdentity,
    LCTIdentityManager,
    initialize_lct_identity,
    get_lct_string
)


class TestLCTIdentity:
    """Test LCTIdentity dataclass"""

    def test_lct_identity_creation(self):
        """Test creating LCT identity"""
        identity = LCTIdentity(
            lineage="dp",
            context="Thor",
            task="consciousness"
        )

        assert identity.lineage == "dp"
        assert identity.context == "Thor"
        assert identity.task == "consciousness"
        assert identity.version == "1.0"

    def test_lct_string_formatting(self):
        """Test LCT URI string formatting"""
        identity = LCTIdentity(
            lineage="dp",
            context="Thor",
            task="consciousness"
        )

        lct_str = identity.to_lct_string()
        assert lct_str == "lct:web4:agent:dp@Thor#consciousness"

    def test_lct_identity_serialization(self):
        """Test identity serialization to dict"""
        identity = LCTIdentity(
            lineage="alice",
            context="Sprout",
            task="perception"
        )

        data = identity.to_dict()

        assert data["lineage"] == "alice"
        assert data["context"] == "Sprout"
        assert data["task"] == "perception"
        assert "lct_uri" in data
        assert data["lct_uri"] == "lct:web4:agent:alice@Sprout#perception"

    def test_lct_identity_deserialization(self):
        """Test identity deserialization from dict"""
        data = {
            "lineage": "bob",
            "context": "Legion",
            "task": "planning",
            "created_at": 1234567890.0,
            "version": "1.0"
        }

        identity = LCTIdentity.from_dict(data)

        assert identity.lineage == "bob"
        assert identity.context == "Legion"
        assert identity.task == "planning"
        assert identity.created_at == 1234567890.0

    def test_lct_string_helper(self):
        """Test get_lct_string convenience function"""
        lct_str = get_lct_string("dp", "Thor", "consciousness")
        assert lct_str == "lct:web4:agent:dp@Thor#consciousness"


class TestLCTIdentityManager:
    """Test LCTIdentityManager functionality"""

    @pytest.fixture
    def temp_identity_dir(self):
        """Create temporary directory for identity files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_identity_manager_creation(self, temp_identity_dir):
        """Test creating identity manager"""
        manager = LCTIdentityManager(identity_dir=temp_identity_dir)

        assert manager.identity_dir == Path(temp_identity_dir)
        assert manager.identity is None

    def test_platform_context_detection(self, temp_identity_dir):
        """Test platform context detection"""
        manager = LCTIdentityManager(identity_dir=temp_identity_dir)

        context = manager.detect_platform_context()

        # Should detect either Thor, Sprout, or hostname
        assert context is not None
        assert len(context) > 0
        print(f"Detected platform context: {context}")

    def test_create_identity(self, temp_identity_dir):
        """Test creating new identity"""
        manager = LCTIdentityManager(identity_dir=temp_identity_dir)

        identity = manager.create_identity(
            lineage="dp",
            task="consciousness",
            context="TestPlatform"
        )

        assert identity.lineage == "dp"
        assert identity.context == "TestPlatform"
        assert identity.task == "consciousness"
        assert manager.identity == identity

    def test_save_and_load_identity(self, temp_identity_dir):
        """Test saving and loading identity"""
        manager = LCTIdentityManager(identity_dir=temp_identity_dir)

        # Create and save identity
        identity = manager.create_identity(
            lineage="alice",
            task="perception",
            context="Sprout"
        )

        saved = manager.save_identity(identity)
        assert saved is True

        # Verify file exists
        identity_file = Path(temp_identity_dir) / "lct_identity_Sprout.json"
        assert identity_file.exists()

        # Load identity in new manager
        manager2 = LCTIdentityManager(identity_dir=temp_identity_dir)
        loaded_identity = manager2.load_identity(context="Sprout")

        assert loaded_identity is not None
        assert loaded_identity.lineage == "alice"
        assert loaded_identity.context == "Sprout"
        assert loaded_identity.task == "perception"

    def test_get_or_create_identity_creates_new(self, temp_identity_dir):
        """Test get_or_create when no identity exists"""
        manager = LCTIdentityManager(identity_dir=temp_identity_dir)

        # Manually set context to avoid platform-specific detection
        manager._platform_context = "TestPlatform"

        identity = manager.get_or_create_identity(
            lineage="bob",
            task="planning"
        )

        assert identity.lineage == "bob"
        assert identity.context == "TestPlatform"
        assert identity.task == "planning"

        # Verify it was saved
        identity_file = Path(temp_identity_dir) / "lct_identity_TestPlatform.json"
        assert identity_file.exists()

    def test_get_or_create_identity_loads_existing(self, temp_identity_dir):
        """Test get_or_create when identity exists"""
        # Create and save identity
        manager1 = LCTIdentityManager(identity_dir=temp_identity_dir)
        manager1._platform_context = "TestPlatform"

        identity1 = manager1.create_identity(
            lineage="charlie",
            task="execution",
            context="TestPlatform"
        )
        manager1.save_identity(identity1)

        # Get or create in new manager (should load existing)
        manager2 = LCTIdentityManager(identity_dir=temp_identity_dir)
        manager2._platform_context = "TestPlatform"

        identity2 = manager2.get_or_create_identity(
            lineage="should_not_use",
            task="should_not_use"
        )

        # Should have loaded existing identity, not created new
        assert identity2.lineage == "charlie"
        assert identity2.task == "execution"

    def test_validate_identity_valid(self, temp_identity_dir):
        """Test validating valid identity"""
        manager = LCTIdentityManager(identity_dir=temp_identity_dir)

        identity = manager.create_identity(
            lineage="dp",
            task="consciousness",
            context="Thor"
        )

        is_valid, reason = manager.validate_identity(identity)

        assert is_valid is True
        assert reason == "Valid"

    def test_validate_identity_empty_lineage(self, temp_identity_dir):
        """Test validating identity with empty lineage"""
        manager = LCTIdentityManager(identity_dir=temp_identity_dir)

        identity = LCTIdentity(lineage="", context="Thor", task="consciousness")

        is_valid, reason = manager.validate_identity(identity)

        assert is_valid is False
        assert "lineage" in reason.lower()

    def test_validate_identity_empty_context(self, temp_identity_dir):
        """Test validating identity with empty context"""
        manager = LCTIdentityManager(identity_dir=temp_identity_dir)

        identity = LCTIdentity(lineage="dp", context="", task="consciousness")

        is_valid, reason = manager.validate_identity(identity)

        assert is_valid is False
        assert "context" in reason.lower()

    def test_validate_identity_none(self, temp_identity_dir):
        """Test validating when no identity set"""
        manager = LCTIdentityManager(identity_dir=temp_identity_dir)

        is_valid, reason = manager.validate_identity()

        assert is_valid is False
        assert "no identity" in reason.lower()

    def test_get_identity_summary_no_identity(self, temp_identity_dir):
        """Test identity summary when no identity loaded"""
        manager = LCTIdentityManager(identity_dir=temp_identity_dir)

        summary = manager.get_identity_summary()

        assert summary["has_identity"] is False
        assert "platform_context" in summary
        assert "status" in summary

    def test_get_identity_summary_with_identity(self, temp_identity_dir):
        """Test identity summary with loaded identity"""
        manager = LCTIdentityManager(identity_dir=temp_identity_dir)

        identity = manager.create_identity(
            lineage="dp",
            task="consciousness",
            context="Thor"
        )

        summary = manager.get_identity_summary()

        assert summary["has_identity"] is True
        assert summary["lct_uri"] == "lct:web4:agent:dp@Thor#consciousness"
        assert summary["lineage"] == "dp"
        assert summary["context"] == "Thor"
        assert summary["task"] == "consciousness"
        assert summary["is_valid"] is True


class TestLCTIntegration:
    """Test LCT integration convenience functions"""

    @pytest.fixture
    def temp_identity_dir(self):
        """Create temporary directory for identity files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_initialize_lct_identity(self, temp_identity_dir, monkeypatch):
        """Test initialize_lct_identity convenience function"""
        # Monkeypatch the default identity directory
        import sage.core.lct_identity_integration as lct_module

        original_init = lct_module.LCTIdentityManager.__init__

        def patched_init(self, identity_dir=temp_identity_dir):
            original_init(self, identity_dir=identity_dir)

        monkeypatch.setattr(lct_module.LCTIdentityManager, "__init__", patched_init)

        manager, identity = initialize_lct_identity(
            lineage="dp",
            task="consciousness"
        )

        assert manager is not None
        assert identity is not None
        assert identity.lineage == "dp"
        assert identity.task == "consciousness"
        assert len(identity.context) > 0

    def test_hierarchical_lineage(self, temp_identity_dir):
        """Test hierarchical lineage formatting"""
        # Test different lineage patterns
        lineages = [
            "dp",
            "dp.assistant1",
            "dp.assistant1.task_manager",
            "org:anthropic",
            "system:genesis"
        ]

        for lineage in lineages:
            lct_str = get_lct_string(lineage, "Thor", "consciousness")
            assert lct_str.startswith("lct:web4:agent:")
            assert f"{lineage}@Thor#consciousness" in lct_str

    def test_task_scopes(self, temp_identity_dir):
        """Test different task scopes"""
        tasks = [
            "consciousness",
            "perception",
            "planning.strategic",
            "execution.code",
            "delegation.federation",
            "admin.full"
        ]

        for task in tasks:
            identity = LCTIdentity(
                lineage="dp",
                context="Thor",
                task=task
            )

            lct_str = identity.to_lct_string()
            assert lct_str.endswith(f"#{task}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
