#!/usr/bin/env python3
"""
Lookahead — predict action consequences by cloning the env.

For each available action, deepcopy the env, execute the action,
observe the frame diff, and summarize as text. Gives the LLM
exact predictions of what each button does in the current state.

This is Approach A from the physics-for-deliberation exploration.
"""

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ACTION_NAMES = ["A0", "UP", "DOWN", "LEFT", "RIGHT", "SEL", "CLICK"]
COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "cyan", 9: "brown",
    10: "purple", 11: "teal", 12: "pink", 13: "lime",
    14: "dark-gray", 15: "white",
}


def lookahead(
    env: Any,
    fd: Any,
    actions: Optional[List[int]] = None,
    click_coords: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[str, float]:
    """Predict consequences of each action by cloning the env.

    Args:
        env: the live ARC-AGI environment
        fd: current FrameData
        actions: list of action indices to try (default: 1-6, skip A0)
        click_coords: for CLICK (action 6), list of (x,y) to try

    Returns:
        (text_summary, elapsed_seconds)
    """
    from arcengine import GameAction

    GA = {
        1: GameAction.ACTION1, 2: GameAction.ACTION2,
        3: GameAction.ACTION3, 4: GameAction.ACTION4,
        5: GameAction.ACTION5, 6: GameAction.ACTION6,
        7: GameAction.ACTION7,
    }

    if actions is None:
        actions = [1, 2, 3, 4, 5]  # UP DOWN LEFT RIGHT SEL (skip A0 and CLICK default)

    curr_frame = _get_frame(fd)
    curr_level = fd.levels_completed
    results = []
    t0 = time.time()

    for action_idx in actions:
        if action_idx not in GA:
            continue

        try:
            clone = copy.deepcopy(env)
            clone_fd = clone.step(GA[action_idx])
            clone_frame = _get_frame(clone_fd)
            desc = _describe_outcome(curr_frame, clone_frame, curr_level, clone_fd)
            results.append((action_idx, desc))
        except Exception as e:
            results.append((action_idx, f"error: {e}"))

    # CLICK with specific coordinates if provided
    if click_coords and 6 in (actions if actions else []) or click_coords:
        for x, y in (click_coords or [(32, 32)]):
            try:
                clone = copy.deepcopy(env)
                clone_fd = clone.step(GA[6], data={"x": x, "y": y})
                clone_frame = _get_frame(clone_fd)
                desc = _describe_outcome(curr_frame, clone_frame, curr_level, clone_fd)
                results.append((f"CLICK({x},{y})", desc))
            except Exception as e:
                results.append((f"CLICK({x},{y})", f"error: {e}"))

    elapsed = time.time() - t0

    # Format as text
    lines = ["=== Action predictions (lookahead) ==="]
    for action, desc in results:
        name = ACTION_NAMES[action] if isinstance(action, int) and action < len(ACTION_NAMES) else str(action)
        lines.append(f"  {name}: {desc}")
    lines.append(f"  (computed in {elapsed*1000:.0f}ms)")

    return "\n".join(lines), elapsed


def _get_frame(fd) -> np.ndarray:
    """Extract 2D frame from FrameData."""
    f = np.array(fd.frame)
    return f[-1] if f.ndim == 3 else f


def _describe_outcome(
    before: np.ndarray, after: np.ndarray,
    level_before: int, fd_after: Any,
) -> str:
    """Describe what changed between before and after frames."""
    if before.shape != after.shape:
        return "frame shape changed — major state transition"

    changed = before != after
    n_changed = int(np.sum(changed))
    total = before.size
    pct = n_changed / total * 100

    # Level change
    level_after = fd_after.levels_completed
    if level_after > level_before:
        return f"LEVEL ADVANCE! ({n_changed}px, {pct:.1f}% changed) → level {level_after}"

    # Game over
    if fd_after.state.name in ("GAME_OVER", "LOST"):
        return f"GAME OVER ({n_changed}px changed)"

    if n_changed == 0:
        return "no change — action blocked or no effect"

    if n_changed <= 2:
        return "trivial change (1-2px) — likely step counter only"

    # Where changed
    ys, xs = np.where(changed)
    cy, cx = int(np.mean(ys)), int(np.mean(xs))
    h, w = before.shape
    loc_y = "top" if cy < h // 3 else ("middle" if cy < 2 * h // 3 else "bottom")
    loc_x = "left" if cx < w // 3 else ("center" if cx < 2 * w // 3 else "right")

    # Color changes
    old_at = before[changed]
    new_at = after[changed]
    old_counts = np.bincount(old_at, minlength=16)
    new_counts = np.bincount(new_at, minlength=16)
    diff = new_counts.astype(int) - old_counts.astype(int)

    grew = [(COLOR_NAMES.get(i, f"c{i}"), int(diff[i]))
            for i in range(16) if diff[i] > 3]
    grew.sort(key=lambda x: -x[1])

    color_info = ""
    if grew:
        color_info = f" ({', '.join(f'+{n} {name}' for name, n in grew[:2])})"

    return f"{n_changed}px ({pct:.1f}%) changed in {loc_y}-{loc_x}{color_info}"


# ─── Quick test ───────────────────────────────────────────────────

if __name__ == "__main__":
    from arc_agi import Arcade
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('cd82-fb555c5d')
    fd = env.reset()

    text, elapsed = lookahead(env, fd)
    print(text)
    print(f"\nTotal time: {elapsed*1000:.0f}ms")

    # Try a few steps then lookahead again
    from arcengine import GameAction
    fd = env.step(GameAction.ACTION3)  # LEFT
    fd = env.step(GameAction.ACTION2)  # DOWN
    fd = env.step(GameAction.ACTION2)  # DOWN

    print("\n--- After LEFT, DOWN, DOWN ---")
    text, elapsed = lookahead(env, fd)
    print(text)
