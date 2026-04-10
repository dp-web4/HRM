#!/usr/bin/env python3
"""
Session Writer — Live Visualization + Persistent Visual Memory.

Two outputs:
1. /tmp/sage_solver/ — live viewer data (overwritten each step)
2. shared-context/arc-agi-3/visual-memory/{game}/run_{timestamp}/ — persistent
   Every frame saved as PNG with step number. GIFs for animations.
   Start/final per level derived from sequence: start = first frame of level,
   final = last frame before next level advance.

The persistent directory structure:
  visual-memory/{game}/run_20260410_073000/
    step_001_L0_RIGHT.png
    step_002_L0_CLICK_32_45.png
    step_003_L0_anim.gif        (if multi-frame)
    step_004_L1_UP.png           (new level = new prefix)
    ...
    run.json                     (metadata: game_id, player, levels, actions)
"""

import os
import json
import time
import numpy as np
from PIL import Image


VIEWER_DIR = "/tmp/sage_solver"
VISUAL_MEMORY_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "shared-context", "arc-agi-3", "visual-memory")

COLOR_MAP = {
    0: (255,255,255), 1: (204,204,204), 2: (153,153,153), 3: (102,102,102),
    4: (51,51,51), 5: (0,0,0), 6: (229,58,163), 7: (255,123,204),
    8: (249,60,49), 9: (30,147,255), 10: (136,216,241), 11: (255,220,0),
    12: (255,133,27), 13: (146,18,49), 14: (79,204,48), 15: (163,86,214),
}

ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                5: "SELECT", 6: "CLICK", 7: "UNDO"}


def render_grid(grid, scale=4):
    if grid.ndim == 3:
        grid = grid[-1]
    if grid.ndim != 2:
        grid = np.full((64, 64), 4, dtype=np.int8)
    h, w = grid.shape
    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color = COLOR_MAP.get(int(grid[r, c]), (128, 128, 128))
            img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = color
    return Image.fromarray(img)


class SessionWriter:
    """Writes game state for live viewer AND persistent visual memory."""

    def __init__(self, game_id: str, win_levels: int = 0,
                 available_actions: list = None, baseline: int = 0,
                 player: str = "autonomous"):
        self.game_id = game_id
        self.game_prefix = game_id.split("-")[0]
        self.current_level = 0
        self.step = 0

        # Live viewer state
        os.makedirs(VIEWER_DIR, exist_ok=True)
        self.session = {
            "game_id": game_id,
            "game_prefix": self.game_prefix,
            "player": player,
            "available_actions": available_actions or [],
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

        # Persistent visual memory: per-run directory
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(
            VISUAL_MEMORY_ROOT, self.game_prefix, f"run_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Run metadata
        self.run_meta = {
            "game_id": game_id,
            "player": player,
            "started": ts,
            "win_levels": win_levels,
            "baseline": baseline,
            "steps": [],
        }

        self._level_start_grid = None
        self._prev_grid = None  # track previous frame for solved-state capture
        self._save_viewer()

    def record_step(self, action: int, grid: np.ndarray,
                    all_frames: list = None, levels_completed: int = 0,
                    x: int = None, y: int = None, state: str = "PLAYING"):
        """Record one step: save frame to viewer + persistent directory."""
        self.step += 1
        prev_level = self.current_level

        # Action description
        aname = ACTION_NAMES.get(action, f"A{action}")
        if action == 6 and x is not None:
            action_label = f"CLICK_{x}_{y}"
        else:
            action_label = aname

        # Update viewer session
        entry = {"action": action, "step": self.step}
        if x is not None and y is not None:
            entry["x"] = x
            entry["y"] = y
        self.session["actions_log"].append(entry)
        self.session.setdefault("level_actions", []).append(entry)
        self.session["step"] = self.step
        self.session["levels_completed"] = levels_completed
        self.session["state"] = state

        # Save to viewer (live, overwritten each step)
        np.save(os.path.join(VIEWER_DIR, "current_grid.npy"), grid)
        render_grid(grid).save(os.path.join(VIEWER_DIR, "frame.png"))

        # Save to persistent visual memory (append-only)
        # Use prev_level for filename — if level changed, this frame is the
        # SOLVED state of the previous level, not the start of the new one.
        frame_name = f"step_{self.step:04d}_L{prev_level}_{action_label}"
        png_path = os.path.join(self.run_dir, f"{frame_name}.png")
        render_grid(grid).save(png_path)

        # Save animation GIF if multi-frame
        gif_path = None
        if all_frames and len(all_frames) > 1:
            gif_name = f"step_{self.step:04d}_L{prev_level}_anim.gif"
            gif_path = os.path.join(self.run_dir, gif_name)
            gif_imgs = [render_grid(f, scale=2) for f in all_frames]
            gif_imgs[0].save(gif_path, save_all=True,
                             append_images=gif_imgs[1:],
                             duration=200, loop=1)
            # Also save to viewer animations dir
            anim_dir = os.path.join(VIEWER_DIR, "animations")
            os.makedirs(anim_dir, exist_ok=True)
            viewer_gif = os.path.join(
                anim_dir, f"anim_L{levels_completed}_S{self.step}.gif")
            gif_imgs[0].save(viewer_gif, save_all=True,
                             append_images=gif_imgs[1:],
                             duration=200, loop=1)

        # Track in run metadata
        step_meta = {
            "step": self.step,
            "level": prev_level,
            "action": aname,
            "png": os.path.basename(png_path),
        }
        if gif_path:
            step_meta["gif"] = os.path.basename(gif_path)
        if x is not None:
            step_meta["x"] = x
            step_meta["y"] = y
        if levels_completed > prev_level:
            step_meta["level_up"] = True
            step_meta["new_level"] = levels_completed
        self.run_meta["steps"].append(step_meta)

        # Level change detection — update live viewer grids
        if levels_completed > prev_level:
            # The "final" grid for the solved level is the PREVIOUS step's frame
            # (the state right before the winning action). The current grid is
            # already the next level's start (SDK transitions instantly).
            final_grid = self._prev_grid if self._prev_grid is not None else grid
            np.save(os.path.join(VIEWER_DIR,
                    f"level_{prev_level}_final_grid.npy"), final_grid)
            np.save(os.path.join(VIEWER_DIR,
                    f"level_{prev_level}_start_grid.npy"),
                    self._level_start_grid if self._level_start_grid is not None
                    else grid)
            np.save(os.path.join(VIEWER_DIR,
                    f"level_{levels_completed}_start_grid.npy"), grid)
            self._level_start_grid = grid

            self.current_level = levels_completed
            self.session["level_start_step"] = self.step
            self.session["level_actions"] = []
            self.session["level_summaries"].append(
                f"L{prev_level+1}: solved at step {self.step}")
        elif self._level_start_grid is None:
            self._level_start_grid = grid
            np.save(os.path.join(VIEWER_DIR,
                    f"level_0_start_grid.npy"), grid)

        # Track for next step's solved-state capture
        self._prev_grid = grid.copy()

        self._save_viewer()

    def record_level_up(self, new_level: int, winning_actions: list = None,
                        summary: str = ""):
        """Record level completion (called by solver loop)."""
        self.session["levels_completed"] = new_level
        if winning_actions:
            self.session["level_solutions"][str(new_level)] = {
                "actions": winning_actions,
                "steps": len(winning_actions),
            }
        self._save_viewer()

    def record_game_end(self, state: str, levels_completed: int = 0):
        """Record final game state and save run metadata."""
        self.session["state"] = state
        self.session["levels_completed"] = levels_completed
        self.run_meta["finished"] = time.strftime("%Y%m%d_%H%M%S")
        self.run_meta["result"] = state
        self.run_meta["levels_completed"] = levels_completed
        self.run_meta["total_steps"] = self.step

        # Save run metadata
        meta_path = os.path.join(self.run_dir, "run.json")
        with open(meta_path, "w") as f:
            json.dump(self.run_meta, f, indent=2)

        self._save_viewer()

    def new_attempt(self, attempt_num: int):
        """Start a new attempt."""
        self.session["attempt_num"] = attempt_num
        self.session["levels_completed"] = 0
        self.session["step"] = 0
        self.session["level_start_step"] = 0
        self.session["actions_log"] = []
        self.session["level_actions"] = []
        self.session["state"] = "PLAYING"
        self.current_level = 0
        # Don't reset self.step — keeps counting across attempts
        # (persistent dir gets all frames)
        self._save_viewer()

    def _save_viewer(self):
        """Save session.json for live viewer."""
        path = os.path.join(VIEWER_DIR, "session.json")
        with open(path, "w") as f:
            json.dump(self.session, f, indent=2)
