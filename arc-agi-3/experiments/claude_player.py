#!/usr/bin/env python3
"""
Claude Player — renders ARC-AGI-3 game frames as images for Claude to see.

This script handles the SDK side. Claude (the Claude Code session) is the model:
- Renders 64x64 grid as a scaled PNG image
- Generates describe_scene() text
- Accepts action commands
- Executes via SDK
- Returns next frame + results

Usage (from Claude Code):
    # Start a game
    python3 claude_player.py start <game_id>

    # Take an action (returns new frame)
    python3 claude_player.py act <game_id> <action> [x] [y]

    # Get current state
    python3 claude_player.py state <game_id>

    # Render current frame as PNG
    python3 claude_player.py render <game_id> [output_path]
"""

import sys, os, json, time
import numpy as np
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_frame, find_color_regions, background_color, color_name

# State persistence between calls
STATE_DIR = "/tmp/claude_player"
os.makedirs(STATE_DIR, exist_ok=True)

# Color map: ARC-AGI-3 color index → RGB
COLOR_MAP = {
    0: (40, 40, 40),      # black
    1: (0, 0, 255),       # blue
    2: (255, 0, 0),       # red
    3: (0, 200, 0),       # green
    4: (255, 255, 0),     # yellow
    5: (128, 128, 128),   # gray
    6: (255, 0, 255),     # magenta
    7: (255, 128, 0),     # orange
    8: (0, 255, 255),     # cyan
    9: (128, 80, 0),      # brown
    10: (255, 180, 180),  # pink
    11: (128, 0, 0),      # maroon
    12: (128, 128, 0),    # olive
    13: (0, 0, 128),      # navy
    14: (0, 128, 128),    # teal
    15: (255, 255, 255),  # white
}

INT_TO_GA = {a.value: a for a in GameAction}


def render_grid(grid, scale=4):
    """Render a 64x64 grid as a scaled RGB image."""
    h, w = grid.shape
    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color = COLOR_MAP.get(int(grid[r, c]), (128, 128, 128))
            img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = color
    return Image.fromarray(img)


def describe_scene(grid):
    """Generate scene description using GridVisionIRP."""
    try:
        from sage.irp.plugins.grid_vision_irp import GridVisionIRP, GridObservation
        regions = find_color_regions(grid, min_size=4)
        objects = [{"id": i, "color": int(r["color"]),
                    "bbox": [r.get("y_start",0), r.get("x_start",0), r.get("y_end",0), r.get("x_end",0)],
                    "centroid": [r.get("cy",0), r.get("cx",0)], "size": r.get("size",0)}
                   for i, r in enumerate(regions)]
        obs = GridObservation(frame_raw=grid, objects=objects, changes=[], moved=[],
                              step_number=0, action_taken=0, level_id="")
        gv = GridVisionIRP.__new__(GridVisionIRP)
        gv._buffer = []
        gv._frame_count = 0
        gv._prev_frame = None
        return gv.describe_scene(obs)
    except Exception as e:
        return f"(scene description error: {e})"


def save_state(game_id, env, fd, grid, step, actions_log):
    """Persist game state between CLI calls."""
    state = {
        "game_id": game_id,
        "step": step,
        "levels_completed": fd.levels_completed,
        "win_levels": fd.win_levels,
        "state": fd.state.name,
        "actions_log": actions_log,
        "available_actions": [a.value if hasattr(a, "value") else int(a)
                              for a in (fd.available_actions or [])],
    }
    with open(f"{STATE_DIR}/{game_id.split('-')[0]}_state.json", "w") as f:
        json.dump(state, f)
    np.save(f"{STATE_DIR}/{game_id.split('-')[0]}_grid.npy", grid)
    return state


def cmd_start(game_id):
    """Start a new game, render initial frame."""
    arcade = Arcade()
    env = arcade.make(game_id)
    fd = env.reset()
    grid = get_frame(fd)
    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (fd.available_actions or [])]

    # Render
    img_path = f"{STATE_DIR}/{game_id.split('-')[0]}_frame.png"
    render_grid(grid).save(img_path)

    # Scene description
    scene = describe_scene(grid)

    # Save state
    state = save_state(game_id, env, fd, grid, 0, [])

    action_names = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                    5: "SELECT/SUBMIT", 6: "CLICK(x,y)", 7: "UNDO"}
    avail_str = ", ".join(f"{a}={action_names.get(a, f'ACTION{a}')}" for a in available)

    print(f"GAME: {game_id}")
    print(f"LEVELS: 0/{fd.win_levels}")
    print(f"ACTIONS: {avail_str}")
    print(f"IMAGE: {img_path}")
    print(f"BASELINE: {sum(env.env_info.baseline_actions if hasattr(env, 'env_info') else [100])}")
    print(f"\nSCENE:\n{scene}")


def cmd_act(game_id, action, x=None, y=None):
    """Execute an action, render new frame."""
    arcade = Arcade()
    env = arcade.make(game_id)
    fd = env.reset()

    # Replay previous actions to get to current state
    state_path = f"{STATE_DIR}/{game_id.split('-')[0]}_state.json"
    actions_log = []
    if os.path.exists(state_path):
        with open(state_path) as f:
            prev_state = json.load(f)
        actions_log = prev_state.get("actions_log", [])
        # Replay
        for a in actions_log:
            ga = INT_TO_GA.get(a["action"])
            if ga:
                if a.get("x") is not None:
                    fd = env.step(ga, data={"x": a["x"], "y": a["y"]})
                else:
                    fd = env.step(ga)

    # Execute new action
    action = int(action)
    prev_grid = get_frame(fd).copy()
    ga = INT_TO_GA.get(action)
    if action == 6 and x is not None and y is not None:
        fd = env.step(ga, data={"x": int(x), "y": int(y)})
        actions_log.append({"action": action, "x": int(x), "y": int(y)})
    else:
        fd = env.step(ga)
        actions_log.append({"action": action})

    grid = get_frame(fd)
    step = len(actions_log)

    # Diff
    n_changed = int(np.sum(prev_grid != grid))
    level_up = fd.levels_completed > (step > 1 and prev_state.get("levels_completed", 0) or 0) if 'prev_state' in dir() else False

    # Render
    img_path = f"{STATE_DIR}/{game_id.split('-')[0]}_frame.png"
    render_grid(grid).save(img_path)

    # Scene description
    scene = describe_scene(grid)

    # Save state
    state = save_state(game_id, env, fd, grid, step, actions_log)

    action_names = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                    5: "SELECT", 6: f"CLICK({x},{y})", 7: "UNDO"}

    print(f"ACTION: {action_names.get(action, f'A{action}')}")
    print(f"RESULT: {n_changed}px changed")
    print(f"STEP: {step}")
    print(f"LEVELS: {fd.levels_completed}/{fd.win_levels}")
    print(f"STATE: {fd.state.name}")
    if fd.state.name == "WON":
        print("★★★ GAME WON! ★★★")
    elif fd.state.name in ("LOST", "GAME_OVER"):
        print("GAME OVER")
    print(f"IMAGE: {img_path}")
    print(f"\nSCENE:\n{scene}")


def cmd_render(game_id, output=None):
    """Just render current frame."""
    grid_path = f"{STATE_DIR}/{game_id.split('-')[0]}_grid.npy"
    if not os.path.exists(grid_path):
        print("No game state found. Run 'start' first.")
        return
    grid = np.load(grid_path)
    img_path = output or f"{STATE_DIR}/{game_id.split('-')[0]}_frame.png"
    render_grid(grid).save(img_path)
    print(f"IMAGE: {img_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: claude_player.py <start|act|render> <game_id> [action] [x] [y]")
        sys.exit(1)

    cmd = sys.argv[1]
    game_id = sys.argv[2]

    # Resolve game_id from prefix
    if "-" not in game_id:
        arcade = Arcade()
        matches = [e for e in arcade.get_environments() if game_id in e.game_id]
        if matches:
            game_id = matches[0].game_id
        else:
            print(f"No game matching '{game_id}'")
            sys.exit(1)

    if cmd == "start":
        cmd_start(game_id)
    elif cmd == "act":
        action = int(sys.argv[3])
        x = int(sys.argv[4]) if len(sys.argv) > 4 else None
        y = int(sys.argv[5]) if len(sys.argv) > 5 else None
        cmd_act(game_id, action, x, y)
    elif cmd == "render":
        output = sys.argv[3] if len(sys.argv) > 3 else None
        cmd_render(game_id, output)
    else:
        print(f"Unknown command: {cmd}")
