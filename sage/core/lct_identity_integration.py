"""
LCT Identity Integration for SAGE Consciousness
Integrates Web4 LCT (Lineage-Context-Task) identity system with SAGE consciousness loop

This module provides SAGE with proper identity management:
1. Hardware-bound context (Thor, Sprout, etc.)
2. Lineage tracking (creator/authorization chain)
3. Task-scoped permissions (what the agent can do)

Format: lct:web4:agent:{lineage}@{context}#{task}
Example: lct:web4:agent:dp@Thor#consciousness

Author: Thor (SAGE autonomous research)
Date: 2025-12-01
Session: Autonomous SAGE Development - LCT Integration
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import time


@dataclass
class LCTIdentity:
    """
    LCT Identity for SAGE consciousness agent

    Components:
    - lineage: Who created/authorized this agent
    - context: What platform/environment agent runs in
    - task: What the agent is authorized to do
    """
    lineage: str  # Creator ID (e.g., "dp", "system:genesis")
    context: str  # Platform ID (e.g., "Thor", "Sprout")
    task: str     # Task scope (e.g., "consciousness", "perception")

    # Metadata
    created_at: float = field(default_factory=time.time)
    version: str = "1.0"

    def to_lct_string(self) -> str:
        """Format as LCT URI"""
        return f"lct:web4:agent:{self.lineage}@{self.context}#{self.task}"

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "lineage": self.lineage,
            "context": self.context,
            "task": self.task,
            "created_at": self.created_at,
            "version": self.version,
            "lct_uri": self.to_lct_string()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'LCTIdentity':
        """Deserialize from dictionary"""
        return cls(
            lineage=data["lineage"],
            context=data["context"],
            task=data["task"],
            created_at=data.get("created_at", time.time()),
            version=data.get("version", "1.0")
        )

    def __str__(self) -> str:
        return self.to_lct_string()


class LCTIdentityManager:
    """
    Manages LCT identity for SAGE consciousness

    Responsibilities:
    - Detect platform context (Thor, Sprout, etc.)
    - Create/load LCT identity
    - Persist identity across sessions
    - Validate identity components
    """

    def __init__(self, identity_dir: str = "sage/data/identity"):
        """
        Initialize LCT identity manager

        Parameters:
        -----------
        identity_dir : str
            Directory for storing identity files
        """
        self.identity_dir = Path(identity_dir)
        self.identity_dir.mkdir(parents=True, exist_ok=True)

        self.identity: Optional[LCTIdentity] = None
        self._platform_context: Optional[str] = None

    def detect_platform_context(self) -> str:
        """
        Detect platform context from hardware

        Returns:
        --------
        str
            Platform identifier (Thor, Sprout, etc.)
        """
        if self._platform_context:
            return self._platform_context

        # Try device-tree detection (Jetson platforms)
        device_tree_path = Path("/proc/device-tree/model")
        if device_tree_path.exists():
            try:
                with open(device_tree_path, 'r') as f:
                    model = f.read().strip().replace('\x00', '')

                # Jetson AGX Thor
                if 'AGX Thor' in model or 'Thor' in model:
                    self._platform_context = "Thor"
                    return "Thor"

                # Jetson Orin Nano (Sprout)
                if 'Orin Nano' in model:
                    self._platform_context = "Sprout"
                    return "Sprout"

                # Other Jetson platforms
                if 'Jetson' in model:
                    self._platform_context = f"Jetson-{model.split()[1]}"
                    return self._platform_context

            except Exception:
                pass

        # Fallback: hostname-based detection
        hostname = os.uname().nodename
        self._platform_context = hostname
        return hostname

    def create_identity(
        self,
        lineage: str,
        task: str = "consciousness",
        context: Optional[str] = None
    ) -> LCTIdentity:
        """
        Create new LCT identity for SAGE

        Parameters:
        -----------
        lineage : str
            Creator/authorization lineage (e.g., "dp", "system:genesis")
        task : str
            Task scope (default: "consciousness")
        context : str, optional
            Platform context (auto-detected if None)

        Returns:
        --------
        LCTIdentity
            New LCT identity
        """
        if context is None:
            context = self.detect_platform_context()

        identity = LCTIdentity(
            lineage=lineage,
            context=context,
            task=task
        )

        self.identity = identity
        return identity

    def load_identity(self, context: Optional[str] = None) -> Optional[LCTIdentity]:
        """
        Load existing LCT identity from disk

        Parameters:
        -----------
        context : str, optional
            Platform context (auto-detected if None)

        Returns:
        --------
        LCTIdentity or None
            Loaded identity, or None if not found
        """
        if context is None:
            context = self.detect_platform_context()

        identity_file = self.identity_dir / f"lct_identity_{context}.json"

        if not identity_file.exists():
            return None

        try:
            with open(identity_file, 'r') as f:
                data = json.load(f)

            identity = LCTIdentity.from_dict(data)
            self.identity = identity
            return identity

        except Exception as e:
            print(f"Warning: Failed to load identity from {identity_file}: {e}")
            return None

    def save_identity(self, identity: Optional[LCTIdentity] = None) -> bool:
        """
        Save LCT identity to disk

        Parameters:
        -----------
        identity : LCTIdentity, optional
            Identity to save (uses self.identity if None)

        Returns:
        --------
        bool
            True if saved successfully
        """
        if identity is None:
            identity = self.identity

        if identity is None:
            return False

        identity_file = self.identity_dir / f"lct_identity_{identity.context}.json"

        try:
            with open(identity_file, 'w') as f:
                json.dump(identity.to_dict(), f, indent=2)

            return True

        except Exception as e:
            print(f"Error: Failed to save identity to {identity_file}: {e}")
            return False

    def get_or_create_identity(
        self,
        lineage: str = "system:autonomous",
        task: str = "consciousness"
    ) -> LCTIdentity:
        """
        Get existing identity or create new one

        Parameters:
        -----------
        lineage : str
            Default lineage if creating new identity
        task : str
            Default task scope if creating new identity

        Returns:
        --------
        LCTIdentity
            Existing or newly created identity
        """
        # Try loading existing identity
        identity = self.load_identity()

        if identity is not None:
            return identity

        # Create new identity
        identity = self.create_identity(lineage=lineage, task=task)

        # Save for future sessions
        self.save_identity(identity)

        return identity

    def validate_identity(self, identity: Optional[LCTIdentity] = None) -> tuple[bool, str]:
        """
        Validate LCT identity components

        Parameters:
        -----------
        identity : LCTIdentity, optional
            Identity to validate (uses self.identity if None)

        Returns:
        --------
        tuple[bool, str]
            (is_valid, reason)
        """
        if identity is None:
            identity = self.identity

        if identity is None:
            return False, "No identity to validate"

        # Validate lineage
        if not identity.lineage or len(identity.lineage) == 0:
            return False, "Empty lineage"

        # Validate context
        if not identity.context or len(identity.context) == 0:
            return False, "Empty context"

        # Validate task
        if not identity.task or len(identity.task) == 0:
            return False, "Empty task"

        # Validate LCT URI format
        lct_uri = identity.to_lct_string()
        if not lct_uri.startswith("lct:web4:agent:"):
            return False, "Invalid LCT URI prefix"

        if "@" not in lct_uri or "#" not in lct_uri:
            return False, "Invalid LCT URI format (missing @ or #)"

        return True, "Valid"

    def get_identity_summary(self) -> Dict:
        """
        Get summary of current identity state

        Returns:
        --------
        Dict
            Identity summary with status
        """
        if self.identity is None:
            return {
                "has_identity": False,
                "platform_context": self.detect_platform_context(),
                "status": "No identity loaded"
            }

        is_valid, reason = self.validate_identity()

        return {
            "has_identity": True,
            "lct_uri": self.identity.to_lct_string(),
            "lineage": self.identity.lineage,
            "context": self.identity.context,
            "task": self.identity.task,
            "created_at": self.identity.created_at,
            "version": self.identity.version,
            "is_valid": is_valid,
            "validation_reason": reason
        }


# Convenience functions for SAGE consciousness integration

def initialize_lct_identity(
    lineage: str = "system:autonomous",
    task: str = "consciousness"
) -> tuple[LCTIdentityManager, LCTIdentity]:
    """
    Initialize LCT identity for SAGE consciousness

    This is the main entry point for SAGE consciousness loop integration.

    Parameters:
    -----------
    lineage : str
        Creator/authorization lineage
    task : str
        Task scope for this SAGE instance

    Returns:
    --------
    tuple[LCTIdentityManager, LCTIdentity]
        (identity_manager, identity)

    Example:
    --------
    >>> manager, identity = initialize_lct_identity(lineage="dp", task="consciousness")
    >>> print(identity)  # lct:web4:agent:dp@Thor#consciousness
    """
    manager = LCTIdentityManager()
    identity = manager.get_or_create_identity(lineage=lineage, task=task)

    return manager, identity


def get_lct_string(
    lineage: str,
    context: str,
    task: str
) -> str:
    """
    Format LCT identity as string

    Parameters:
    -----------
    lineage : str
        Creator ID
    context : str
        Platform ID
    task : str
        Task scope

    Returns:
    --------
    str
        LCT URI string

    Example:
    --------
    >>> get_lct_string("dp", "Thor", "consciousness")
    'lct:web4:agent:dp@Thor#consciousness'
    """
    return f"lct:web4:agent:{lineage}@{context}#{task}"
