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
from pathlib import Path
from typing import Dict, List, Optional

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
