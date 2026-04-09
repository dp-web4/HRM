#!/usr/bin/env python3
"""
sc25 solver — spell-casting maze puzzle, all 6 levels.

Key discovery: First action on each level produces an intro animation (22 frames).
That action's intent (movement/click) is consumed by the animation, not executed.
So every level effectively costs 1 "wasted" action for the intro.

Spells (3x3 toggle grid, auto-casts when pattern matches):
  DIAMOND (fpokrvgln): SIZE_CHANGE — toggles s1 (2x2, step=2) <-> s2 (4x4, step=4+2)
  LSHAPE  (jzukcpajs): TELEPORT   — cycles through target list
  VLINE   (aprnrzeyj): FIREBALL   — shoots in player's facing direction

Movement at s2: step=4 then bonus half-step of 2 (total 6px if both clear)
Movement at s1: step=2

The first action on each level is consumed by intro animation.
"""

import sys, os, json, time
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_frame, get_all_frames

INT_TO_GA = {a.value: a for a in GameAction}

# Spell click coordinates (from game source: spell_slot_click_actions)
CELL = {
    (0,0): (25,50), (0,1): (30,50), (0,2): (35,50),
    (1,0): (25,55), (1,1): (30,55), (1,2): (35,55),
    (2,0): (25,60), (2,1): (30,60), (2,2): (35,60),
}
DIAMOND_CLICKS = [CELL[(0,1)], CELL[(1,0)], CELL[(1,2)], CELL[(2,1)]]
LSHAPE_CLICKS  = [CELL[(0,0)], CELL[(1,0)], CELL[(2,0)], CELL[(2,1)]]
VLINE_CLICKS   = [CELL[(0,1)], CELL[(1,1)], CELL[(2,1)]]

UP, DOWN, LEFT, RIGHT, CLICK = 1, 2, 3, 4, 6


def act(env, action, x=None, y=None):
    """Execute one action, return frame_data."""
    ga = INT_TO_GA[action]
    if x is not None and y is not None:
        return env.step(ga, data={"x": x, "y": y})
    return env.step(ga)


def cast(env, spell_clicks, verbose=False):
    """Cast a spell by clicking the pattern cells."""
    for x, y in spell_clicks:
        fd = act(env, CLICK, x, y)
        if verbose:
            nf = np.array(fd.frame).shape[0]
            if nf > 1:
                print(f"    Cast animation: {nf} frames, levels={fd.levels_completed}")
    return fd


def move(env, direction, steps, target_level=None, verbose=False):
    """Move in a direction for N steps. Stop early on level change."""
    names = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}
    for i in range(steps):
        fd = act(env, direction)
        if target_level is not None and fd.levels_completed >= target_level:
            if verbose: print(f"    Level up at step {i+1}! levels={fd.levels_completed}")
            return fd
    return fd


def consume_intro(env, verbose=False):
    """Waste one action to consume the level intro animation."""
    fd = act(env, RIGHT)  # RIGHT is typically blocked, costs nothing
    nf = np.array(fd.frame).shape[0]
    if verbose and nf > 1:
        print(f"  Intro consumed ({nf} frames)")
    return fd


def solve_L1(env, verbose=False):
    """L1: Shrink with DIAMOND, walk LEFT 9 to exit."""
    # Player at (39,19) s2, Exit at (12,17). Spell: DIAMOND.
    # Shrink to s1, then walk LEFT.
    print("=== Level 1 (budget=50) ===")
    consume_intro(env, verbose)
    if verbose: print("  DIAMOND (shrink)")
    cast(env, DIAMOND_CLICKS, verbose)
    move(env, LEFT, 12, target_level=1, verbose=verbose)


def solve_L2(env, verbose=False):
    """L2: Teleport with LSHAPE, walk UP to exit."""
    # Player at (31,35) s2, Exit at (30,10). Spell: LSHAPE (teleport).
    # Teleport target at (31,19). Walk UP.
    print("=== Level 2 (budget=25) ===")
    consume_intro(env, verbose)
    if verbose: print("  LSHAPE (teleport)")
    cast(env, LSHAPE_CLICKS, verbose)
    move(env, UP, 5, target_level=2, verbose=verbose)


def solve_L3(env, verbose=False):
    """L3: Fireball RIGHT to hit switch, walk LEFT+DOWN to exit."""
    # Player at (35,22) s2 facing RIGHT. Switch at (55,22). Door at (27,34). Exit at (22,37).
    # Fire RIGHT to hit switch, door opens, walk to exit.
    print("=== Level 3 (budget=50) ===")
    consume_intro(env, verbose)
    if verbose: print("  VLINE (fireball RIGHT)")
    cast(env, VLINE_CLICKS, verbose)
    move(env, LEFT, 5, verbose=verbose)
    move(env, DOWN, 5, target_level=3, verbose=verbose)


def solve_L4(env, verbose=False):
    """L4: Shrink, navigate to switch, fireball, navigate to exit."""
    # Player at (35,19) s2. Switch at (7,28). Door at (47,35). Exit at (51,34).
    # Spells: VLINE, DIAMOND.
    print("=== Level 4 (budget=35) ===")
    consume_intro(env, verbose)

    if verbose: print("  DIAMOND (shrink)")
    cast(env, DIAMOND_CLICKS, verbose)

    # Navigate DOWN toward switch y=28, LEFT toward switch x=7
    move(env, DOWN, 5, verbose=verbose)
    move(env, LEFT, 10, verbose=verbose)

    if verbose: print("  VLINE (fireball LEFT)")
    cast(env, VLINE_CLICKS, verbose)

    # Navigate to exit at (51,34)
    move(env, RIGHT, 15, verbose=verbose)
    move(env, DOWN, 5, target_level=4, verbose=verbose)


def solve_L5(env, verbose=False):
    """L5: Complex spell combo with both doors."""
    # Player at (35,15) s2. Primary switch (15,11), secondary (3,43).
    # Primary door (51,16), secondary (51,20). Exit at (50,10).
    # TP-small: (29,39). TP-large: (51,35).
    print("=== Level 5 (budget=65) ===")
    consume_intro(env, verbose)

    if verbose: print("  DIAMOND (shrink)")
    cast(env, DIAMOND_CLICKS, verbose)

    if verbose: print("  LSHAPE (teleport to small target (29,39))")
    cast(env, LSHAPE_CLICKS, verbose)

    # Navigate to primary switch area
    move(env, LEFT, 7, verbose=verbose)
    move(env, UP, 14, verbose=verbose)

    if verbose: print("  VLINE (fireball UP -> primary)")
    cast(env, VLINE_CLICKS, verbose)

    # Navigate to secondary switch
    move(env, DOWN, 16, verbose=verbose)
    move(env, LEFT, 5, verbose=verbose)

    if verbose: print("  VLINE (fireball LEFT -> secondary)")
    cast(env, VLINE_CLICKS, verbose)

    # Grow, teleport, shrink, navigate to exit
    if verbose: print("  DIAMOND (grow)")
    cast(env, DIAMOND_CLICKS, verbose)
    if verbose: print("  LSHAPE (teleport to large target)")
    cast(env, LSHAPE_CLICKS, verbose)
    if verbose: print("  DIAMOND (shrink)")
    cast(env, DIAMOND_CLICKS, verbose)

    move(env, UP, 15, target_level=5, verbose=verbose)


def solve_L6(env, verbose=False):
    """L6: The final level. Budget=60.

    Route (52 actions + 1 intro = 53):
    0. Consume intro (1 action)
    1. DIAMOND (shrink to s1): 4
    2. LSHAPE (teleport to (13,41)): 4
    3. RIGHT 2, UP 3: 5 -> at (17,35) facing UP
    4. VLINE (fireball UP -> primary switch (17,33)): 3
    5. DIAMOND (grow to s2): 4
    6. LSHAPE (teleport to (53,37)): 4
    7. LEFT 3 at s2 step=4 -> (41,37): 3
    8. VLINE (fireball LEFT -> secondary switch (37,37)): 3
    9. LSHAPE (teleport to (29,33)): 4
    10. DIAMOND (shrink to s1): 4
    11. UP 2 -> (29,29): 2
    12. RIGHT 2 -> (33,29): 2
    13. UP 10 -> exit at (33,9): 10+
    """
    print("=== Level 6 FINAL (budget=60) ===")
    consume_intro(env, verbose)

    if verbose: print("  DIAMOND (shrink)")
    cast(env, DIAMOND_CLICKS, verbose)

    if verbose: print("  LSHAPE (teleport -> (13,41))")
    cast(env, LSHAPE_CLICKS, verbose)

    if verbose: print("  Navigate to (17,35)")
    move(env, RIGHT, 2, verbose=verbose)
    move(env, UP, 3, verbose=verbose)

    if verbose: print("  VLINE (fireball UP -> primary switch)")
    cast(env, VLINE_CLICKS, verbose)

    if verbose: print("  DIAMOND (grow)")
    cast(env, DIAMOND_CLICKS, verbose)

    if verbose: print("  LSHAPE (teleport -> (53,37))")
    cast(env, LSHAPE_CLICKS, verbose)

    if verbose: print("  LEFT 3 at s2")
    move(env, LEFT, 3, verbose=verbose)

    if verbose: print("  VLINE (fireball LEFT -> secondary switch)")
    cast(env, VLINE_CLICKS, verbose)

    if verbose: print("  LSHAPE (teleport -> (29,33))")
    cast(env, LSHAPE_CLICKS, verbose)

    if verbose: print("  DIAMOND (shrink)")
    cast(env, DIAMOND_CLICKS, verbose)

    if verbose: print("  Navigate UP 2, RIGHT 2, UP 10+")
    move(env, UP, 2, verbose=verbose)
    move(env, RIGHT, 2, verbose=verbose)
    move(env, UP, 12, target_level=6, verbose=verbose)


def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    arcade = Arcade()
    env = arcade.make("sc25-f9b21a2f")
    fd = env.reset()

    print(f"Game: sc25-f9b21a2f")
    print(f"Starting level: {fd.levels_completed}/{fd.win_levels}\n")

    solvers = [solve_L1, solve_L2, solve_L3, solve_L4, solve_L5, solve_L6]

    for i, solver in enumerate(solvers):
        try:
            solver(env, verbose=verbose)
        except Exception as e:
            print(f"ERROR on level {i+1}: {e}")
            import traceback
            traceback.print_exc()
            break

        # Check state with dummy step
        fd = act(env, UP)
        print(f"  -> {fd.levels_completed}/{fd.win_levels} state={fd.state}\n")

        if fd.levels_completed >= fd.win_levels:
            print(f"ALL {fd.win_levels} LEVELS SOLVED!")
            break

        if fd.levels_completed <= i:
            print(f"  WARNING: Level {i+1} NOT solved")
            break

    print(f"Final: {fd.levels_completed}/{fd.win_levels}")


if __name__ == "__main__":
    main()
