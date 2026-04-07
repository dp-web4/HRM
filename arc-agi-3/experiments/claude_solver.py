#!/usr/bin/env python3
"""
Claude Solver — Claude Code as the game-playing model.

One step, one image, semantic diff, remember everything.
Claude sees the game screen after every action and decides what to do.

This script is the GAME ENGINE SIDE. Claude Code (the interactive session)
is the REASONING SIDE. The script:
  1. Renders the current frame as PNG
  2. Generates scene description + semantic diff from previous frame
  3. Prints the context for Claude to read
  4. Accepts Claude's action command
  5. Executes it and loops

On level-up:
  - Summarizes lessons from completed level
  - Resets frame memory
  - Stores summary for cross-level recall

Usage (interactive, from Claude Code session):
    # Initialize a game session
    python3 claude_solver.py init <game_prefix>

    # Take one step (returns new image + semantic diff)
    python3 claude_solver.py step <action> [x] [y]

    # See current state without acting
    python3 claude_solver.py look

    # Get level summary (call after level-up)
    python3 claude_solver.py summarize

    # Get full session context (action history + observations)
    python3 claude_solver.py context
"""

import sys, os, json, time
import numpy as np
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_frame, find_color_regions, background_color, color_name

STATE_DIR = "/tmp/claude_solver"
os.makedirs(STATE_DIR, exist_ok=True)

COLOR_MAP = {
    0: (40, 40, 40), 1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 200, 0),
    4: (255, 255, 0), 5: (128, 128, 128), 6: (255, 0, 255), 7: (255, 128, 0),
    8: (0, 255, 255), 9: (128, 80, 0), 10: (255, 180, 180), 11: (128, 0, 0),
    12: (128, 128, 0), 13: (0, 0, 128), 14: (0, 128, 128), 15: (255, 255, 255),
}
COLOR_NAMES = ["black", "blue", "red", "green", "yellow", "gray",
               "magenta", "orange", "cyan", "brown", "pink", "maroon",
               "olive", "navy", "teal", "white"]
ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                5: "SELECT", 6: "CLICK", 7: "UNDO"}
INT_TO_GA = {a.value: a for a in GameAction}


def render_grid(grid, scale=4):
    h, w = grid.shape
    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color = COLOR_MAP.get(int(grid[r, c]), (128, 128, 128))
            img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = color
    return Image.fromarray(img)


def scene_description(grid):
    """Generate scene description using GridVisionIRP."""
    try:
        from sage.irp.plugins.grid_vision_irp import GridVisionIRP, GridObservation
        regions = find_color_regions(grid, min_size=4)
        objects = [{"id": i, "color": int(r["color"]),
                    "bbox": [r.get("y_start",0), r.get("x_start",0),
                             r.get("y_end",0), r.get("x_end",0)],
                    "centroid": [r.get("cy",0), r.get("cx",0)],
                    "size": r.get("size",0)}
                   for i, r in enumerate(regions)]
        obs = GridObservation(frame_raw=grid, objects=objects, changes=[], moved=[],
                              step_number=0, action_taken=0, level_id="")
        gv = GridVisionIRP.__new__(GridVisionIRP)
        gv._buffer = []
        gv._frame_count = 0
        gv._prev_frame = None
        return gv.describe_scene(obs)
    except Exception as e:
        return f"(scene error: {e})"


def semantic_diff(prev_grid, curr_grid):
    """Describe what changed between two frames in game terms, not pixel terms.

    Instead of '201px changed in region r5-46', says:
    'A large region in the center changed — something moved or was painted.
     New colors appeared: cyan. Colors disappeared: white.'
    """
    if prev_grid is None:
        return "(first frame — no previous to compare)"

    diff_mask = prev_grid != curr_grid
    n_changed = int(diff_mask.sum())

    if n_changed == 0:
        return "NOTHING CHANGED — your action had no visible effect."

    # Where did changes happen?
    coords = np.argwhere(diff_mask)
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    h, w = curr_grid.shape

    # What changed?
    old_colors = set(int(prev_grid[r, c]) for r, c in coords)
    new_colors = set(int(curr_grid[r, c]) for r, c in coords)
    appeared = new_colors - old_colors
    disappeared = old_colors - new_colors

    # Spatial description
    center_r, center_c = (r_min + r_max) / 2, (c_min + c_max) / 2
    if center_r < h * 0.33:
        v_pos = "top"
    elif center_r > h * 0.66:
        v_pos = "bottom"
    else:
        v_pos = "center"
    if center_c < w * 0.33:
        h_pos = "left"
    elif center_c > w * 0.66:
        h_pos = "right"
    else:
        h_pos = "center"
    location = f"{v_pos}-{h_pos}" if v_pos != h_pos else v_pos

    # Size classification
    change_area = (r_max - r_min + 1) * (c_max - c_min + 1)
    if n_changed <= 4:
        magnitude = "tiny change (cursor or selection indicator)"
    elif n_changed <= 30:
        magnitude = "small change (an object changed state)"
    elif n_changed <= 100:
        magnitude = "moderate change (something moved or transformed)"
    else:
        magnitude = "large change (major movement or state transition)"

    parts = [f"{magnitude} in the {location} area"]

    if appeared:
        parts.append(f"New colors appeared: {', '.join(COLOR_NAMES[c] for c in appeared)}")
    if disappeared:
        parts.append(f"Colors disappeared: {', '.join(COLOR_NAMES[c] for c in disappeared)}")

    # Check if it looks like movement (old colors appear where new colors were, and vice versa)
    overlap = old_colors & new_colors
    if overlap and n_changed > 20:
        parts.append("Pattern suggests MOVEMENT — colors shifted position")

    # Check for single-object changes (likely a toggle or selection)
    if n_changed <= 10 and len(new_colors) == 1:
        parts.append(f"A single spot changed to {COLOR_NAMES[list(new_colors)[0]]} — possible selection or toggle")

    return ". ".join(parts) + "."


def load_session():
    """Load persistent session state."""
    path = os.path.join(STATE_DIR, "session.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_session(session):
    """Save session state."""
    path = os.path.join(STATE_DIR, "session.json")
    with open(path, "w") as f:
        json.dump(session, f, indent=2)


def save_grid(grid, name="current"):
    np.save(os.path.join(STATE_DIR, f"{name}_grid.npy"), grid)


def load_grid(name="current"):
    path = os.path.join(STATE_DIR, f"{name}_grid.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def cmd_init(game_prefix):
    """Initialize a new game session."""
    arcade = Arcade()
    matches = [e for e in arcade.get_environments() if game_prefix in e.game_id]
    if not matches:
        print(f"No game matching '{game_prefix}'")
        return
    env_info = matches[0]
    game_id = env_info.game_id
    env = arcade.make(game_id)
    fd = env.reset()
    grid = get_frame(fd)
    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (fd.available_actions or [])]

    # Render image
    img_path = os.path.join(STATE_DIR, "frame.png")
    render_grid(grid).save(img_path)

    # Build scene
    scene = scene_description(grid)

    # Initialize session
    session = {
        "game_id": game_id,
        "game_prefix": game_prefix,
        "available_actions": available,
        "win_levels": fd.win_levels,
        "levels_completed": 0,
        "step": 0,
        "state": "NOT_FINISHED",
        "actions_log": [],
        "observations": [],
        "level_summaries": [],
        "level_start_step": 0,
        "baseline": sum(env_info.baseline_actions),
    }
    save_session(session)
    save_grid(grid, "current")
    save_grid(grid, "previous")  # no previous yet, but save for shape

    avail_str = ", ".join(f"{a}={ACTION_NAMES.get(a, f'A{a}')}" for a in available)

    print(f"GAME: {game_id}")
    print(f"LEVELS: 0/{fd.win_levels}")
    print(f"ACTIONS: {avail_str}")
    print(f"BASELINE: {session['baseline']} actions for 100% efficiency")
    print(f"IMAGE: {img_path}")
    print(f"\nSCENE:\n{scene}")
    print(f"\nThis is your first view of the game. Study the image carefully.")
    print(f"What type of game is this? What do you think the objective is?")
    print(f"When ready, use: python3 claude_solver.py step <action> [x] [y]")


def cmd_step(action, x=None, y=None):
    """Execute one action. Returns new image + semantic diff."""
    session = load_session()
    if not session:
        print("No active session. Run 'init <game>' first.")
        return

    # Reconstruct game state by replaying actions
    arcade = Arcade()
    env = arcade.make(session["game_id"])
    fd = env.reset()
    for a in session["actions_log"]:
        ga = INT_TO_GA.get(a["action"])
        if ga:
            if a.get("x") is not None:
                fd = env.step(ga, data={"x": a["x"], "y": a["y"]})
            else:
                fd = env.step(ga)

    # Load previous grid for diff
    prev_grid = load_grid("current")

    # Execute new action
    action = int(action)
    ga = INT_TO_GA.get(action)
    if action == 6 and x is not None and y is not None:
        fd = env.step(ga, data={"x": int(x), "y": int(y)})
        session["actions_log"].append({"action": action, "x": int(x), "y": int(y)})
        action_desc = f"CLICK({x},{y})"
    else:
        fd = env.step(ga)
        session["actions_log"].append({"action": action})
        action_desc = ACTION_NAMES.get(action, f"ACTION{action}")

    grid = get_frame(fd)
    session["step"] += 1
    prev_levels = session["levels_completed"]
    session["levels_completed"] = fd.levels_completed
    session["state"] = fd.state.name

    # Render new frame
    img_path = os.path.join(STATE_DIR, "frame.png")
    render_grid(grid).save(img_path)

    # Semantic diff
    diff = semantic_diff(prev_grid, grid)

    # Scene description
    scene = scene_description(grid)

    # Record observation
    observation = {
        "step": session["step"],
        "action": action_desc,
        "diff": diff,
        "levels": fd.levels_completed,
    }
    session["observations"].append(observation)

    # Save state
    save_grid(prev_grid, "previous")
    save_grid(grid, "current")

    # Check for level-up
    level_up = fd.levels_completed > prev_levels

    # Output
    print(f"STEP {session['step']}: {action_desc}")
    print(f"WHAT CHANGED: {diff}")
    print(f"LEVELS: {fd.levels_completed}/{session['win_levels']}")
    print(f"IMAGE: {img_path}")

    if level_up:
        print(f"\n★★★ LEVEL UP! Now at level {fd.levels_completed}/{session['win_levels']} ★★★")
        print(f"Run 'python3 claude_solver.py summarize' to capture what you learned,")
        print(f"then 'python3 claude_solver.py look' to see the new level.")

    if fd.state.name == "WON":
        print(f"\n★★★ GAME WON in {session['step']} steps! ★★★")
    elif fd.state.name in ("LOST", "GAME_OVER"):
        print(f"\nGAME OVER after {session['step']} steps.")

    print(f"\nSCENE:\n{scene}")

    save_session(session)


def cmd_look():
    """Show current state without taking an action."""
    session = load_session()
    if not session:
        print("No active session.")
        return

    grid = load_grid("current")
    img_path = os.path.join(STATE_DIR, "frame.png")
    render_grid(grid).save(img_path)
    scene = scene_description(grid)

    avail_str = ", ".join(f"{a}={ACTION_NAMES.get(a, f'A{a}')}"
                          for a in session["available_actions"])

    print(f"GAME: {session['game_id']}")
    print(f"STEP: {session['step']}")
    print(f"LEVELS: {session['levels_completed']}/{session['win_levels']}")
    print(f"ACTIONS: {avail_str}")
    print(f"STATE: {session['state']}")
    print(f"IMAGE: {img_path}")
    print(f"\nSCENE:\n{scene}")

    # Show recent observations
    recent = session["observations"][-5:]
    if recent:
        print(f"\nRECENT ACTIONS:")
        for obs in recent:
            print(f"  Step {obs['step']}: {obs['action']} → {obs['diff'][:80]}")


def cmd_summarize():
    """Summarize lessons from the current/completed level."""
    session = load_session()
    if not session:
        print("No active session.")
        return

    level_obs = [o for o in session["observations"]
                 if o["step"] > session.get("level_start_step", 0)]

    # Analyze what happened
    actions_taken = [o["action"] for o in level_obs]
    effects = [o["diff"] for o in level_obs]
    no_effect = sum(1 for e in effects if "NOTHING CHANGED" in e)
    movements = sum(1 for e in effects if "MOVEMENT" in e)
    toggles = sum(1 for e in effects if "toggle" in e.lower() or "selection" in e.lower())

    print(f"LEVEL SUMMARY (steps {session.get('level_start_step', 0)+1} to {session['step']}):")
    print(f"  Total actions: {len(level_obs)}")
    print(f"  No effect: {no_effect} ({no_effect/max(len(level_obs),1)*100:.0f}%)")
    print(f"  Movement: {movements}")
    print(f"  Toggles/selections: {toggles}")
    print(f"\n  Actions used: {', '.join(set(actions_taken))}")

    # Store summary
    summary = {
        "level": session["levels_completed"],
        "steps": len(level_obs),
        "wasted": no_effect,
        "actions_used": list(set(actions_taken)),
    }
    session["level_summaries"].append(summary)
    session["level_start_step"] = session["step"]
    save_session(session)

    print(f"\n  Summary saved. Previous level context cleared.")
    print(f"  Use 'look' to see the new level.")


def cmd_context():
    """Print full session context for Claude to reason about."""
    session = load_session()
    if not session:
        print("No active session.")
        return

    print(f"GAME: {session['game_id']}")
    print(f"STEP: {session['step']}")
    print(f"LEVELS: {session['levels_completed']}/{session['win_levels']}")
    print(f"BASELINE: {session['baseline']} actions")

    if session["level_summaries"]:
        print(f"\nPREVIOUS LEVEL LESSONS:")
        for s in session["level_summaries"]:
            print(f"  Level {s['level']}: {s['steps']} steps, "
                  f"{s['wasted']} wasted, used {', '.join(s['actions_used'])}")

    recent = session["observations"][-10:]
    if recent:
        print(f"\nRECENT OBSERVATIONS (last {len(recent)}):")
        for obs in recent:
            print(f"  Step {obs['step']}: {obs['action']} → {obs['diff'][:100]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: claude_solver.py <init|step|look|summarize|context> [args]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "init":
        cmd_init(sys.argv[2] if len(sys.argv) > 2 else "cd82")
    elif cmd == "step":
        action = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        x = int(sys.argv[3]) if len(sys.argv) > 3 else None
        y = int(sys.argv[4]) if len(sys.argv) > 4 else None
        cmd_step(action, x, y)
    elif cmd == "look":
        cmd_look()
    elif cmd == "summarize":
        cmd_summarize()
    elif cmd == "context":
        cmd_context()
    else:
        print(f"Unknown command: {cmd}")
