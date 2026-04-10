#!/usr/bin/env python3
"""
sc25 replay — runs verified solution through sage_solver for viewer data.

Produces: level start/final grids, animation GIFs, session.json, frame.png.
Uses the exact verified sequences from sc25_solutions.json.
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

DIAMOND = [(30,50), (25,55), (35,55), (30,60)]
LSHAPE = [(25,50), (30,50), (30,55)]
VLINE = [(30,50), (30,55), (30,60)]
UP, DOWN, LEFT, RIGHT, CLICK = 1, 2, 3, 4, 6


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
    env = arcade.make("sc25-f9b21a2f")
    fd = env.reset()
    grid = get_frame(fd)

    session = {
        "game_id": "sc25-f9b21a2f",
        "game_prefix": "sc25",
        "player": "claude",
        "available_actions": [1, 2, 3, 4, 6],
        "win_levels": 6,
        "levels_completed": 0,
        "step": 0,
        "state": "NOT_FINISHED",
        "actions_log": [],
        "observations": [],
        "level_summaries": [],
        "level_solutions": {},
        "level_start_step": 0,
        "level_actions": [],
        "baseline": 216,
        "animations": [],
    }

    save_grid(grid, "current")
    save_grid(grid, "level_0_start")
    render_grid(grid).save(os.path.join(STATE_DIR, "frame.png"))

    step = 0
    level_actions = []

    def do(action, x=None, y=None, desc=""):
        nonlocal fd, grid, step, level_actions
        ga = INT_TO_GA[action]
        if x is not None:
            fd = env.step(ga, data={"x": x, "y": y})
            entry = {"action": action, "x": x, "y": y}
        else:
            fd = env.step(ga)
            entry = {"action": action}
        step += 1
        session["actions_log"].append(entry)
        level_actions.append(entry)

        all_frames = get_all_frames(fd)
        prev_grid = grid
        grid = all_frames[-1]

        save_grid(grid, "current")
        render_grid(grid).save(os.path.join(STATE_DIR, "frame.png"))

        # Animation?
        if len(all_frames) > 1:
            gif_path = save_gif(all_frames, fd.levels_completed, step)
            session["animations"].append({
                "level": int(fd.levels_completed),
                "step": step,
                "frames": len(all_frames),
                "gif": gif_path,
            })

        # Level change?
        prev_levels = session["levels_completed"]
        if fd.levels_completed > prev_levels:
            # Save final grid of solved level
            save_grid(prev_grid, f"level_{prev_levels}_final")
            # Save start grid of new level
            save_grid(grid, f"level_{fd.levels_completed}_start")
            # Record solution
            session["level_solutions"][str(prev_levels)] = {
                "actions": level_actions[:],
                "steps": len(level_actions),
            }
            session["level_summaries"].append(
                f"L{prev_levels+1}: solved in {len(level_actions)} actions"
            )
            level_actions = []
            print(f"  Level {prev_levels+1} -> {fd.levels_completed+1} ({desc})")

        session["levels_completed"] = int(fd.levels_completed)
        session["step"] = step
        session["state"] = fd.state.name

    def spell(clicks, desc=""):
        for x, y in clicks:
            do(CLICK, x, y, desc)

    print("Replaying sc25 verified solution...")

    # === L1 ===
    print("L1: intro + DIAMOND + LEFT x12")
    do(RIGHT, desc="L1 intro")
    spell(DIAMOND, "L1 DIAMOND")
    for i in range(12):
        do(LEFT, desc="L1 LEFT")
        if fd.levels_completed >= 1: break

    # === L2 ===
    print("L2: intro + LSHAPE + UP")
    do(RIGHT, desc="L2 intro")
    spell(LSHAPE, "L2 LSHAPE")
    for i in range(8):
        do(UP, desc="L2 UP")
        if fd.levels_completed >= 2: break

    # === L3 ===
    print("L3: RIGHT + VLINE + LEFT 3 + DOWN 4 + LEFT 3")
    do(RIGHT, desc="L3 face RIGHT")
    spell(VLINE, "L3 VLINE")
    for i in range(3): do(LEFT, desc="L3 LEFT")
    for i in range(4): do(DOWN, desc="L3 DOWN")
    for i in range(3):
        do(LEFT, desc="L3 LEFT")
        if fd.levels_completed >= 3: break

    # === L4 ===
    print("L4: DIAMOND + DOWN 5 + LEFT 4 + VLINE + DIAMOND + DOWN 2 + RIGHT 6")
    spell(DIAMOND, "L4 DIAMOND shrink")
    for i in range(5): do(DOWN, desc="L4 DOWN")
    for i in range(4): do(LEFT, desc="L4 LEFT")
    spell(VLINE, "L4 VLINE")
    spell(DIAMOND, "L4 DIAMOND grow")
    for i in range(2): do(DOWN, desc="L4 DOWN")
    for i in range(8):
        do(RIGHT, desc="L4 RIGHT")
        if fd.levels_completed >= 4: break

    # === L5 ===
    print("L5: triple combo")
    spell(DIAMOND, "L5 DIAMOND shrink")
    spell(LSHAPE, "L5 LSHAPE tp")
    for i in range(7): do(LEFT, desc="L5 LEFT")
    for i in range(12): do(UP, desc="L5 UP")
    spell(VLINE, "L5 VLINE primary")
    for i in range(14): do(DOWN, desc="L5 DOWN")
    do(LEFT, desc="L5 face LEFT")
    spell(VLINE, "L5 VLINE secondary")
    spell(DIAMOND, "L5 DIAMOND grow")
    spell(LSHAPE, "L5 LSHAPE tp")
    for i in range(8):
        do(UP, desc="L5 UP to exit")
        if fd.levels_completed >= 5: break

    # === L6 ===
    print("L6: dual fireball + triple teleport")
    spell(DIAMOND, "L6 DIAMOND shrink")
    spell(LSHAPE, "L6 LSHAPE tp(13,41)")
    for i in range(2): do(RIGHT, desc="L6 RIGHT")
    for i in range(3): do(UP, desc="L6 UP")
    spell(VLINE, "L6 VLINE UP primary")
    spell(DIAMOND, "L6 DIAMOND grow")
    spell(LSHAPE, "L6 LSHAPE tp(53,37)")
    for i in range(3): do(LEFT, desc="L6 LEFT s2")
    spell(VLINE, "L6 VLINE LEFT secondary")
    spell(LSHAPE, "L6 LSHAPE tp(29,33)")
    spell(DIAMOND, "L6 DIAMOND shrink")
    for i in range(2): do(UP, desc="L6 UP")
    for i in range(2): do(RIGHT, desc="L6 RIGHT")
    for i in range(15):
        do(UP, desc="L6 UP to exit")
        if fd.levels_completed >= 6: break

    # Save final state
    if fd.levels_completed >= 6:
        save_grid(grid, f"level_5_final")
        session["level_solutions"]["5"] = {
            "actions": level_actions[:],
            "steps": len(level_actions),
        }
        session["level_summaries"].append(
            f"L6: solved in {len(level_actions)} actions"
        )

    session["levels_completed"] = int(fd.levels_completed)
    session["step"] = step
    session["state"] = fd.state.name

    with open(os.path.join(STATE_DIR, "session.json"), "w") as f:
        json.dump(session, f, indent=2)

    print(f"\nResult: {fd.levels_completed}/{fd.win_levels} in {step} total actions")
    print(f"State: {fd.state.name}")
    print(f"Session saved to {STATE_DIR}/session.json")

    # List saved files
    files = sorted(glob.glob(os.path.join(STATE_DIR, "*.npy")))
    print(f"\nSaved grids ({len(files)}):")
    for f in files:
        print(f"  {os.path.basename(f)}")

    anims = sorted(glob.glob(os.path.join(STATE_DIR, "animations", "*.gif")))
    print(f"\nAnimations ({len(anims)}):")
    for f in anims:
        print(f"  {os.path.basename(f)}")


if __name__ == "__main__":
    main()
