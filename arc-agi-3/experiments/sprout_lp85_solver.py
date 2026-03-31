#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 LP85 Solver — Sprout

LP85 is a rotational puzzle:
- Buttons rotate groups of sprites along a cycle
- button_N_L = counterclockwise, button_N_R = clockwise
- Win: all bghvgbtwcb sprites aligned with goal sprites at (x+1, y+1)

Level 0: grid 32x19, 13-step budget
- button_A_L at grid (1, 8) = left rotate
- button_A_R at grid (28, 8) = right rotate
- bghvgbtwcb (piece) at (5, 2) needs goal at (6, 3)
- odkpvwbihk (goal marker) at (21, 3)

Strategy: Try all possible combinations of L/R button presses
within the step budget. Since only 1 button group with 2 options,
it's just a sequence of L/R presses (2^13 = 8192 possibilities, tractable).
But smarter: try clicking one button repeatedly (1-13 times).
"""
import sys, os, time, json
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from arcengine import GameAction
from arc_agi import Arcade


def get_frame(fd):
    grid = np.array(fd.frame)
    return grid[-1] if grid.ndim == 3 else grid


def grid_to_display(gx, gy, grid_w, grid_h, display_size=64):
    """Convert grid coords to display pixel coords (center of cell)."""
    scale_x = display_size / grid_w
    scale_y = display_size / grid_h
    # Center of the grid cell
    dx = int(gx * scale_x + scale_x / 2)
    dy = int(gy * scale_y + scale_y / 2)
    return max(0, min(63, dx)), max(0, min(63, dy))


def try_button_sequence(arcade, game_id, clicks, grid_w=32, grid_h=19, verbose=False):
    """Try a sequence of button clicks.
    clicks: list of 'L' or 'R'
    """
    # Button positions (grid coords for level 0)
    btn_l = grid_to_display(1, 8, grid_w, grid_h)  # button_A_L
    btn_r = grid_to_display(28, 8, grid_w, grid_h)  # button_A_R

    env = arcade.make(game_id)
    fd = env.reset()
    max_levels = fd.levels_completed
    steps = 0

    for click in clicks:
        if fd.state.name in ("WON", "GAME_OVER"):
            break
        bx, by = btn_l if click == 'L' else btn_r
        try:
            fd = env.step(GameAction.ACTION6, data={'x': bx, 'y': by})
            steps += 1
        except Exception as e:
            if verbose:
                print(f"  Error clicking {click} at ({bx},{by}): {e}")
            continue

        if fd.levels_completed > max_levels:
            max_levels = fd.levels_completed
            if verbose:
                print(f"  ★ Level {fd.levels_completed}/{fd.win_levels} after {steps} clicks!")

    return max_levels, fd.win_levels, steps, fd.state.name


def main():
    print("=" * 60)
    print("SAGE ARC-AGI-3 LP85 Solver — Sprout")
    print("=" * 60)

    arcade = Arcade()
    envs = arcade.get_environments()
    lp85 = [e for e in envs if "lp85" in (e.game_id if hasattr(e, "game_id") else str(e))]
    if not lp85:
        print("LP85 not found!")
        return

    game_id = lp85[0].game_id if hasattr(lp85[0], "game_id") else str(lp85[0])
    print(f"Game: {game_id}")

    # Button display positions
    btn_l = grid_to_display(1, 8, 32, 19)
    btn_r = grid_to_display(28, 8, 32, 19)
    print(f"Button L: grid(1,8) -> display{btn_l}")
    print(f"Button R: grid(28,8) -> display{btn_r}")

    # Strategy 1: Try pure L clicks (1 to 13)
    print("\n--- Pure Left rotations ---")
    best_l = 0
    for n in range(1, 14):
        lvl, win, steps, state = try_button_sequence(
            arcade, game_id, ['L'] * n, verbose=False)
        if lvl > 0:
            print(f"  L×{n}: {lvl}/{win} ★")
            best_l = max(best_l, lvl)
        elif n <= 5:
            print(f"  L×{n}: {lvl}/{win}")

    # Strategy 2: Try pure R clicks (1 to 13)
    print("\n--- Pure Right rotations ---")
    best_r = 0
    for n in range(1, 14):
        lvl, win, steps, state = try_button_sequence(
            arcade, game_id, ['R'] * n, verbose=False)
        if lvl > 0:
            print(f"  R×{n}: {lvl}/{win} ★")
            best_r = max(best_r, lvl)
        elif n <= 5:
            print(f"  R×{n}: {lvl}/{win}")

    # Strategy 3: Try mixed sequences (if pure didn't work well)
    print("\n--- Mixed L/R sequences ---")
    best_mix = 0
    # Try L then R combos
    for nl in range(1, 8):
        for nr in range(1, 14 - nl):
            lvl, win, steps, state = try_button_sequence(
                arcade, game_id, ['L'] * nl + ['R'] * nr, verbose=False)
            if lvl > best_mix:
                best_mix = lvl
                print(f"  L×{nl}+R×{nr}: {lvl}/{win} ★")

    # Strategy 4: If level 1 works, try continuing with level 2 buttons
    best_total = max(best_l, best_r, best_mix)
    if best_total > 0:
        print(f"\n--- Multi-level attempt (best single: {best_total}) ---")
        # The game resets with new level after completing one
        # Level 2 has different grid size and button positions
        # Try clicking same button positions repeatedly
        for n in range(1, 100):
            lvl, win, steps, state = try_button_sequence(
                arcade, game_id, ['R'] * n, verbose=(n <= 3 or n % 20 == 0))
            if lvl > best_total:
                best_total = lvl
                print(f"  R×{n}: {lvl}/{win} ★★ NEW BEST!")
            if state == "WON":
                print(f"  🏆 WON at R×{n}!")
                break

    print(f"\n{'='*60}")
    print(f"Best: {best_total} levels")

    # Store in membot
    try:
        import requests
        requests.post("http://localhost:8000/api/store",
            json={"text": f"ARC-AGI-3 lp85 solver: best={best_total} levels with button rotation strategy. "
                  f"Grid 32x19, buttons at L=(1,8)->display{btn_l}, R=(28,8)->display{btn_r}. "
                  f"Pure L best={best_l}, pure R best={best_r}, mixed best={best_mix}."},
            timeout=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
