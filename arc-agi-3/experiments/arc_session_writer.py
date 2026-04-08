#!/usr/bin/env python3
"""
Session Writer — Live Visualization Support for Game Solvers.

Writes game state to /tmp/claude_solver/ for realtime visualization via game_viewer.py.

Based on claude_solver.py's viewer integration pattern.
"""

import os
import json
import numpy as np


STATE_DIR = "/tmp/claude_solver"
os.makedirs(STATE_DIR, exist_ok=True)


class SessionWriter:
    """Writes game session state for live visualization."""

    def __init__(self, game_id: str, win_levels: int, available_actions: list,
                 baseline: int = 0, player: str = "autonomous"):
        self.game_prefix = game_id.split("-")[0]
        self.session = {
            "game_id": game_id,
            "game_prefix": self.game_prefix,
            "player": player,
            "available_actions": available_actions,
            "win_levels": win_levels,
            "levels_completed": 0,
            "step": 0,
            "state": "PLAYING",
            "actions_log": [],
            "observations": [],
            "level_summaries": [],
            "level_solutions": {},
            "level_start_step": 0,
            "level_actions": [],
            "baseline": baseline,
            "attempt_num": 1,
        }
        self.save()

    def record_action(self, action: int, x: int = None, y: int = None,
                     observation: str = "", grid: np.ndarray = None):
        """Record one action and its result."""
        entry = {"action": action, "step": self.session["step"]}
        if x is not None and y is not None:
            entry["x"] = x
            entry["y"] = y

        self.session["actions_log"].append(entry)
        self.session.setdefault("level_actions", []).append(entry)
        self.session["step"] += 1

        if observation:
            self.session["observations"].append({
                "step": self.session["step"],
                "text": observation,
            })

        if grid is not None:
            self.save_grid(grid, "current")

        self.save()

    def record_level_up(self, new_level: int, winning_actions: list = None,
                        summary: str = ""):
        """Record level completion."""
        self.session["levels_completed"] = new_level

        if winning_actions:
            self.session["level_solutions"][str(new_level)] = {
                "actions": winning_actions,
                "steps": len(winning_actions),
            }

        if summary:
            self.session["level_summaries"].append({
                "level": new_level,
                "summary": summary,
            })

        # Reset level tracking
        self.session["level_start_step"] = self.session["step"]
        self.session["level_actions"] = []

        self.save()

    def record_game_end(self, state: str):
        """Record final game state (WON/LOST)."""
        self.session["state"] = state
        self.save()

    def new_attempt(self, attempt_num: int):
        """Start a new attempt (reset counters but keep learning)."""
        self.session["attempt_num"] = attempt_num
        self.session["levels_completed"] = 0
        self.session["step"] = 0
        self.session["level_start_step"] = 0
        self.session["actions_log"] = []
        self.session["level_actions"] = []
        self.session["state"] = "PLAYING"
        self.save()

    def save(self):
        """Save session.json for viewer."""
        path = os.path.join(STATE_DIR, "session.json")
        with open(path, "w") as f:
            json.dump(self.session, f, indent=2)

    def save_grid(self, grid: np.ndarray, name: str = "current"):
        """Save grid snapshot as .npy."""
        np.save(os.path.join(STATE_DIR, f"{name}_grid.npy"), grid)
