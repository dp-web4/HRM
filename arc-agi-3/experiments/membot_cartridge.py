#!/usr/bin/env python3
"""
Membot Cartridge Interface — Cross-session memory for ARC-AGI-3.

Each game family gets a cartridge that accumulates:
- Winning action sequences
- Goal patterns and indicators
- Action effectiveness statistics
- Strategic insights from Navigator

Cartridges persist across sessions, enabling learning accumulation.
"""

import json
import requests
import time
import json
import os
import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

MEMBOT_URL = "http://localhost:8000"
CARTRIDGE_DIR = Path("arc-agi-3/experiments/cartridges")


class MembotCartridge:
    """Interface to membot for game-specific persistent memory."""

    def __init__(self, game_id: str):
        """
        Args:
            game_id: Full game ID (e.g., "sc25-f9b21a2f")
        """
        self.game_id = game_id
        self.game_family = game_id.split("-")[0]  # e.g., "sc25"
        self.cartridge_name = f"arc-agi-3/{self.game_family}"
        self.data = None

    def _get_filesystem_path(self) -> Path:
        """Get filesystem path for this cartridge."""
        CARTRIDGE_DIR.mkdir(parents=True, exist_ok=True)
        return CARTRIDGE_DIR / f"{self.game_family}.json"

    def _read_filesystem(self) -> Optional[dict]:
        """Read cartridge from filesystem."""
        path = self._get_filesystem_path()
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"    Membot: Filesystem read failed ({e})")
        return None

    def _write_filesystem(self, data: dict):
        """Write cartridge to filesystem."""
        path = self._get_filesystem_path()
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"    Membot: Filesystem write failed ({e})")

    def read(self) -> Optional[dict]:
        """Read cartridge from membot HTTP API, fallback to filesystem.

        Returns:
            Cartridge dict or empty if doesn't exist
        """
        # Try HTTP first
        try:
            resp = requests.get(
                f"{MEMBOT_URL}/cartridges/{self.cartridge_name}",
                timeout=1.0
            )
            if resp.status_code == 200:
                self.data = resp.json()
                return self.data
        except Exception:
            pass  # Fall through to filesystem

        # Try filesystem
        fs_data = self._read_filesystem()
        if fs_data:
            self.data = fs_data
            return self.data

        # Create new empty cartridge
        self.data = self._empty_cartridge()
        return self.data

    def write(self, data: dict = None):
        """Write cartridge to membot HTTP API, fallback to filesystem.

        Args:
            data: Cartridge dict (uses self.data if not provided)
        """
        if data is None:
            data = self.data
        if data is None:
            return

        data["last_updated"] = time.time()

        # Try HTTP first
        try:
            resp = requests.put(
                f"{MEMBOT_URL}/cartridges/{self.cartridge_name}",
                json=data,
                timeout=1.0
            )
            if resp.status_code in (200, 201):
                self.data = data
                return  # Success!
        except Exception:
            pass  # Fall through to filesystem

        # Fallback to filesystem
        self._write_filesystem(data)
        self.data = data

    def add_winning_sequence(self, level: int, actions: List[int],
                            state_hashes: List[str] = None):
        """Record a winning action sequence.

        Args:
            level: Level number that was completed
            actions: List of action integers
            state_hashes: Optional state hashes for each action
        """
        if self.data is None:
            self.data = self._empty_cartridge()

        from sage_driver import ACTION_LABELS  # Import here to avoid circular dep
        action_names = [ACTION_LABELS.get(a, f"A{a}") for a in actions]

        sequence = {
            "level": level,
            "actions": actions,
            "action_names": action_names,
            "state_hashes": state_hashes or [],
            "learned": time.time()
        }

        self.data["winning_sequences"].append(sequence)

        # Keep only last 20
        if len(self.data["winning_sequences"]) > 20:
            self.data["winning_sequences"] = self.data["winning_sequences"][-20:]

        self.write()

    def add_strategic_insight(self, hypothesis: str, confidence: float):
        """Add a Navigator hypothesis to strategic insights.

        Args:
            hypothesis: Pattern description from Navigator
            confidence: Confidence score (0.0-1.0)
        """
        if self.data is None:
            self.data = self._empty_cartridge()

        # Only add high-confidence insights
        if confidence < 0.7:
            return

        insight = f"{hypothesis} (confidence: {confidence:.1%})"

        # Avoid duplicates
        if insight not in self.data["strategic_insights"]:
            self.data["strategic_insights"].append(insight)

        # Keep only last 10
        if len(self.data["strategic_insights"]) > 10:
            self.data["strategic_insights"] = self.data["strategic_insights"][-10:]

        self.write()

    def update_action_effectiveness(self, driver):
        """Update action effectiveness from Driver stats.

        Args:
            driver: SageDriver instance with action_tries/action_changes
        """
        if self.data is None:
            self.data = self._empty_cartridge()

        from sage_driver import ACTION_LABELS

        # Global effectiveness
        global_stats = {}
        for action, tries in driver.action_tries.items():
            changes = driver.action_changes.get(action, 0)
            effectiveness = changes / max(tries, 1)
            action_name = ACTION_LABELS.get(action, f"A{action}")
            global_stats[action_name] = effectiveness

        self.data["action_effectiveness"]["global"] = global_stats

        # State-dependent effectiveness (top 5 states by tries)
        state_stats = {}
        for state_hash, actions in driver.state_action_tries.items():
            total_tries = sum(actions.values())
            if total_tries < 3:  # Skip rare states
                continue

            state_effectiveness = {}
            for action, tries in actions.items():
                changes = driver.state_action_changes[state_hash].get(action, 0)
                effectiveness = changes / max(tries, 1)
                action_name = ACTION_LABELS.get(action, f"A{action}")
                state_effectiveness[action_name] = effectiveness

            state_stats[state_hash] = state_effectiveness

        # Keep only top 5 states by total tries
        sorted_states = sorted(
            state_stats.items(),
            key=lambda x: sum(driver.state_action_tries[x[0]].values()),
            reverse=True
        )[:5]

        self.data["action_effectiveness"]["state_dependent"] = dict(sorted_states)

        self.write()

    def update_best_score(self, levels: int, steps: int):
        """Update best score if this run was better.

        Args:
            levels: Levels completed this run
            steps: Steps taken this run
        """
        if self.data is None:
            self.data = self._empty_cartridge()

        current_best = self.data.get("best_score", {"levels": 0, "steps": 999999})

        # Better if more levels, or same levels in fewer steps
        if (levels > current_best["levels"] or
            (levels == current_best["levels"] and steps < current_best["steps"])):
            self.data["best_score"] = {"levels": levels, "steps": steps}
            self.write()

    def increment_attempts(self):
        """Increment total attempts counter."""
        if self.data is None:
            self.data = self._empty_cartridge()

        self.data["total_attempts"] += 1
        self.write()

    # ==================== VISION MEMORY METHODS ====================

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert numpy frame to base64 PNG string.

        Args:
            frame: (H, W) numpy array with values 0-15 (ARC colors)

        Returns:
            Base64-encoded PNG string
        """
        # Ensure frame is 2D and contiguous
        if frame.ndim != 2:
            if frame.ndim == 3:
                frame = frame[:, :, -1]  # Take last channel
            else:
                raise ValueError(f"Invalid frame dimensions: {frame.shape}")

        # Ensure contiguous array for PIL
        frame = np.ascontiguousarray(frame)

        # Convert to PIL Image (need to convert to RGB for PNG)
        # ARC colors are 0-15, scale to 0-255 for visibility
        frame_scaled = (frame * 16).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(frame_scaled, mode='L')  # Grayscale

        # Convert to PNG bytes
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)

        # Encode as base64
        return base64.b64encode(buffer.read()).decode('utf-8')

    def _base64_to_frame(self, b64_str: str) -> np.ndarray:
        """Convert base64 PNG string back to numpy frame.

        Args:
            b64_str: Base64-encoded PNG string

        Returns:
            (H, W) numpy array with values 0-15
        """
        # Decode base64
        img_bytes = base64.b64decode(b64_str)
        buffer = io.BytesIO(img_bytes)

        # Load as PIL Image
        img = Image.open(buffer)
        frame_scaled = np.array(img)

        # Scale back from 0-255 to 0-15
        return (frame_scaled // 16).astype(np.uint8)

    def store_frame_snapshot(self, label: str, frame: np.ndarray, metadata: dict = None):
        """Store a frame snapshot in visual memory.

        Args:
            label: Unique identifier (e.g., "level_1_initial", "level_1_goal_state")
            frame: (H, W) numpy array with values 0-15
            metadata: Optional metadata dict (level, step, description, etc.)
        """
        if self.data is None:
            self.data = self._empty_cartridge()

        # Ensure visual_memory exists (for backward compatibility)
        if "visual_memory" not in self.data:
            self.data["visual_memory"] = {
                "snapshots": {},
                "action_outcomes": []
            }

        frame_b64 = self._frame_to_base64(frame)

        self.data["visual_memory"]["snapshots"][label] = {
            "frame_b64": frame_b64,
            "metadata": metadata or {},
            "timestamp": time.time()
        }

        self.write()

    def get_frame_snapshot(self, label: str) -> Optional[np.ndarray]:
        """Retrieve a frame snapshot from visual memory.

        Args:
            label: Snapshot identifier

        Returns:
            (H, W) numpy array or None if not found
        """
        if self.data is None:
            return None

        snapshot = self.data["visual_memory"]["snapshots"].get(label)
        if not snapshot:
            return None

        return self._base64_to_frame(snapshot["frame_b64"])

    def store_action_visual_outcome(self, action: int, before_frame: np.ndarray,
                                    after_frame: np.ndarray, level: int, step: int):
        """Store visual before/after of an action for learning.

        Args:
            action: Action number (1-6 for ARC-AGI)
            before_frame: Frame before action
            after_frame: Frame after action
            level: Current level
            step: Current step number
        """
        if self.data is None:
            self.data = self._empty_cartridge()

        # Ensure visual_memory exists (for backward compatibility)
        if "visual_memory" not in self.data:
            self.data["visual_memory"] = {
                "snapshots": {},
                "action_outcomes": []
            }

        # Validate frames
        if before_frame.size == 0 or after_frame.size == 0:
            return  # Skip invalid frames
        if before_frame.ndim not in (2, 3) or after_frame.ndim not in (2, 3):
            return  # Skip malformed frames

        outcome = {
            "action": action,
            "before_b64": self._frame_to_base64(before_frame),
            "after_b64": self._frame_to_base64(after_frame),
            "level": level,
            "step": step,
            "timestamp": time.time()
        }

        self.data["visual_memory"]["action_outcomes"].append(outcome)

        # Keep only last 50 action outcomes
        if len(self.data["visual_memory"]["action_outcomes"]) > 50:
            self.data["visual_memory"]["action_outcomes"] = \
                self.data["visual_memory"]["action_outcomes"][-50:]

        self.write()

    def compute_visual_similarity(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """Compute visual similarity between two frames.

        Args:
            frame_a: First frame
            frame_b: Second frame

        Returns:
            Similarity score [0, 1] where 1 = identical
        """
        if frame_a.shape != frame_b.shape:
            return 0.0

        # Simple pixel-wise similarity
        total_cells = frame_a.size
        matching_cells = np.sum(frame_a == frame_b)
        return matching_cells / total_cells

    def find_similar_snapshots(self, target_frame: np.ndarray,
                               threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Find snapshots visually similar to target frame.

        Args:
            target_frame: Frame to compare against
            threshold: Minimum similarity score (0-1)

        Returns:
            List of (label, similarity_score) tuples, sorted by similarity
        """
        if self.data is None:
            return []

        results = []
        for label, snapshot in self.data["visual_memory"]["snapshots"].items():
            stored_frame = self._base64_to_frame(snapshot["frame_b64"])
            similarity = self.compute_visual_similarity(target_frame, stored_frame)

            if similarity >= threshold:
                results.append((label, similarity))

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ==================== END VISION METHODS ====================

    def _empty_cartridge(self) -> dict:
        """Create empty cartridge structure."""
        return {
            "game_family": self.game_family,
            "winning_sequences": [],
            "goal_patterns": [],
            "action_effectiveness": {
                "global": {},
                "state_dependent": {}
            },
            "strategic_insights": [],
            "visual_memory": {
                "snapshots": {},  # label → {frame_b64, metadata, timestamp}
                "action_outcomes": []  # {action, before_b64, after_b64, level, step}
            },
            "total_attempts": 0,
            "best_score": {"levels": 0, "steps": 0},
            "created": time.time(),
            "last_updated": time.time()
        }

    def summary(self) -> str:
        """Generate summary string for logging."""
        if self.data is None:
            return "Cartridge: Not loaded"

        lines = [
            f"Cartridge: {self.game_family}",
            f"  Attempts: {self.data['total_attempts']}",
            f"  Best score: {self.data['best_score']['levels']} levels in {self.data['best_score']['steps']} steps",
            f"  Winning sequences: {len(self.data['winning_sequences'])}",
            f"  Strategic insights: {len(self.data['strategic_insights'])}",
        ]

        if self.data["winning_sequences"]:
            last_seq = self.data["winning_sequences"][-1]
            actions = " → ".join(last_seq["action_names"][:10])
            lines.append(f"  Last win: Level {last_seq['level']}: {actions}...")

        return "\n".join(lines)


def test_cartridge():
    """Test cartridge read/write."""
    print("Testing Membot Cartridge...")

    cart = MembotCartridge("test_game-abc123")

    # Read (should create empty if doesn't exist)
    data = cart.read()
    print(f"\nInitial cartridge:\n{cart.summary()}")

    # Add a winning sequence
    cart.add_winning_sequence(1, [1, 1, 1, 6, 6], ["a3f4e2", "b8c9d1", "c1d2e3", "d4e5f6", "e7f8g9"])
    print(f"\nAfter adding winning sequence:\n{cart.summary()}")

    # Add strategic insight
    cart.add_strategic_insight("Three UP actions enable ACTION6", 0.85)
    print(f"\nAfter adding insight:\n{cart.summary()}")

    # Read again to verify persistence
    cart2 = MembotCartridge("test_game-def456")
    data2 = cart2.read()
    print(f"\nNew cartridge (different game):\n{cart2.summary()}")


if __name__ == "__main__":
    test_cartridge()
