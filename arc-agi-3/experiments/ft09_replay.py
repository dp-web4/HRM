#!/usr/bin/env python3
"""
ft09 replay — runs verified algorithmic solution for viewer data.
Click sequences from ft09_solver.py (all verified).
"""

import sys, os, json, glob, shutil
import numpy as np
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_frame, get_all_frames

STATE_DIR = "/tmp/sage_solver"
INT_TO_GA = {a.value: a for a in GameAction}

COLOR_MAP = {
    0: (255,255,255), 1: (204,204,204), 2: (153,153,153), 3: (102,102,102),
    4: (51,51,51), 5: (0,0,0), 6: (229,58,163), 7: (255,123,204),
    8: (249,60,49), 9: (30,147,255), 10: (136,216,241), 11: (255,220,0),
    12: (255,133,27), 13: (146,18,49), 14: (79,204,48), 15: (163,86,214),
}

# Verified click sequences per level (from ft09_solver.py)
LEVEL_CLICKS = [
    # L1: 4 clicks
    [(38,54), (38,46), (38,38), (54,46)],
    # L2: 7 clicks
    [(22,32), (22,24), (30,48), (22,48), (22,16), (38,32), (38,24)],
    # L3: 14 clicks
    [(38,54), (46,30), (30,22), (22,54), (22,14), (14,22), (30,6),
     (46,38), (14,30), (30,54), (38,6), (30,38), (22,46), (22,6)],
    # L4: 16 clicks
    [(38,48), (38,48), (46,24), (30,16), (22,48), (22,48), (38,32),
     (30,24), (22,32), (22,32), (30,48), (30,48), (22,16), (22,16),
     (46,16), (30,32)],
    # L5: 21 clicks
    [(16,14), (32,38), (32,14), (24,6), (32,22), (40,38), (48,46),
     (32,46), (40,54), (32,54), (48,38), (16,22), (32,6), (16,38),
     (16,54), (16,30), (32,30), (48,22), (24,14), (40,46), (24,30)],
    # L6: 13 clicks
    [(22,16), (38,16), (14,24), (22,24), (38,32), (14,32), (46,32),
     (30,32), (46,40), (22,40), (54,40), (6,16), (6,8)],
]


def render_grid(grid, scale=4):
    h, w = grid.shape
    img = np.zeros((h*scale, w*scale, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color = COLOR_MAP.get(int(grid[r, c]), (128,128,128))
            img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = color
    return Image.fromarray(img)


def save_grid(grid, name):
    np.save(os.path.join(STATE_DIR, f"{name}_grid.npy"), grid)


def save_gif(all_frames, level, step):
    anim_dir = os.path.join(STATE_DIR, "animations")
    os.makedirs(anim_dir, exist_ok=True)
    gif_frames = [render_grid(f, scale=2) for f in all_frames]
    gif_path = os.path.join(anim_dir, f"anim_L{level}_S{step}.gif")
    gif_frames[0].save(gif_path, save_all=True,
                       append_images=gif_frames[1:], duration=200, loop=1)
    return gif_path


def main():
    # Clean up
    os.makedirs(STATE_DIR, exist_ok=True)
    for pattern in ["level_*_grid.npy", "current_grid.npy", "previous_grid.npy",
                    "frame.png", "*.gif", "*.png"]:
        for f in glob.glob(os.path.join(STATE_DIR, pattern)):
            os.remove(f)
    anim_dir = os.path.join(STATE_DIR, "animations")
    if os.path.isdir(anim_dir):
        shutil.rmtree(anim_dir)

    arcade = Arcade()
    env = arcade.make("ft09-0d8bbf25")
    fd = env.reset()
    grid = get_frame(fd)

    session = {
        "game_id": "ft09-0d8bbf25",
        "game_prefix": "ft09",
        "player": "mcnugget-algo",
        "available_actions": [6],
        "win_levels": fd.win_levels,
        "levels_completed": 0,
        "step": 0,
        "state": "NOT_FINISHED",
        "actions_log": [],
        "level_summaries": [],
        "level_solutions": {},
        "baseline": 163,
        "animations": [],
    }

    save_grid(grid, "current")
    save_grid(grid, "level_0_start")
    render_grid(grid).save(os.path.join(STATE_DIR, "frame.png"))

    step = 0
    total_clicks = 0
    ga = INT_TO_GA[6]  # CLICK

    print(f"Replaying ft09 verified solution ({fd.win_levels} levels)...")

    for level_idx, clicks in enumerate(LEVEL_CLICKS):
        level_start_grid = grid
        level_actions = []

        for x, y in clicks:
            fd = env.step(ga, data={"x": x, "y": y})
            step += 1
            total_clicks += 1
            level_actions.append({"action": 6, "x": x, "y": y})
            session["actions_log"].append({"action": 6, "x": x, "y": y})

            all_frames = get_all_frames(fd)
            prev_grid = grid
            grid = all_frames[-1]

            save_grid(grid, "current")
            render_grid(grid).save(os.path.join(STATE_DIR, "frame.png"))

            # Capture animations
            if len(all_frames) > 1:
                gif_path = save_gif(all_frames, fd.levels_completed, step)
                session["animations"].append({
                    "level": int(fd.levels_completed),
                    "step": step,
                    "frames": len(all_frames),
                })

            # Level change?
            if fd.levels_completed > level_idx:
                save_grid(prev_grid, f"level_{level_idx}_final")
                save_grid(grid, f"level_{fd.levels_completed}_start")
                session["level_solutions"][str(level_idx)] = {
                    "actions": level_actions,
                    "steps": len(level_actions),
                }
                session["level_summaries"].append(
                    f"L{level_idx+1}: {len(level_actions)} clicks"
                )
                print(f"  L{level_idx+1}: {len(level_actions)} clicks -> {fd.levels_completed}/{fd.win_levels}")
                break
        else:
            # Didn't level up during clicks — check state
            if fd.levels_completed > level_idx:
                save_grid(grid, f"level_{level_idx}_final")
                save_grid(grid, f"level_{fd.levels_completed}_start")
                session["level_solutions"][str(level_idx)] = {
                    "actions": level_actions,
                    "steps": len(level_actions),
                }
                session["level_summaries"].append(
                    f"L{level_idx+1}: {len(level_actions)} clicks"
                )
                print(f"  L{level_idx+1}: {len(level_actions)} clicks -> {fd.levels_completed}/{fd.win_levels}")
            else:
                print(f"  L{level_idx+1}: {len(level_actions)} clicks — NOT SOLVED (levels={fd.levels_completed})")

        session["levels_completed"] = int(fd.levels_completed)
        session["step"] = step
        session["state"] = fd.state.name

        if fd.state.name == "GAME_OVER":
            print(f"  GAME OVER at step {step}")
            break

    # Save final level data if won
    if fd.levels_completed >= fd.win_levels:
        last_level = fd.win_levels - 1
        if str(last_level) not in session["level_solutions"]:
            save_grid(grid, f"level_{last_level}_final")
            session["level_solutions"][str(last_level)] = {
                "actions": level_actions,
                "steps": len(level_actions),
            }
            session["level_summaries"].append(
                f"L{last_level+1}: {len(level_actions)} clicks"
            )

    with open(os.path.join(STATE_DIR, "session.json"), "w") as f:
        json.dump(session, f, indent=2)

    print(f"\nResult: {fd.levels_completed}/{fd.win_levels} in {total_clicks} total clicks")
    print(f"Baseline: 163, Efficiency: {163/max(total_clicks,1)*100:.0f}%")
    print(f"State: {fd.state.name}")

    grids = sorted(glob.glob(os.path.join(STATE_DIR, "level_*_grid.npy")))
    anims = sorted(glob.glob(os.path.join(STATE_DIR, "animations", "*.gif")))
    print(f"Saved: {len(grids)} grids, {len(anims)} animations")


if __name__ == "__main__":
    main()
