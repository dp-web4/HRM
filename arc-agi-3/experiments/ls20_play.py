#!/usr/bin/env python3
"""
ls20 Interactive Player — Fast multi-step execution with step counter tracking.

Play through ls20 levels by executing action sequences and observing results.
Saves frames for viewer. Designed for Claude Code to reason about.

Usage:
    python3 ls20_play.py              # Start game, show L1
    python3 ls20_play.py --actions "LEFT LEFT LEFT UP UP UP UP RIGHT RIGHT RIGHT UP UP UP"
    python3 ls20_play.py --explore     # Execute one of each direction to map movement
"""

import sys
import os
import json
import numpy as np
from collections import deque
from typing import List, Tuple, Set, Optional, Dict

sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_frame, get_all_frames

INT_TO_GA = {a.value: a for a in GameAction}
ACTION_NAMES = {1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT'}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4}

# SDK palette for rendering
SDK_PALETTE = {
    0: (255,255,255), 1: (204,204,204), 2: (153,153,153), 3: (102,102,102),
    4: (51,51,51), 5: (0,0,0), 6: (229,58,163), 7: (255,123,204),
    8: (249,60,49), 9: (30,147,255), 10: (136,216,241), 11: (255,220,0),
    12: (255,133,27), 13: (146,18,49), 14: (79,204,48), 15: (163,86,214),
}


def render_frame(grid, path, scale=4):
    """Save grid as PNG."""
    from PIL import Image
    if grid.ndim == 3:
        grid = grid[-1]
    h, w = grid.shape
    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color = SDK_PALETTE.get(int(grid[r, c]), (128, 128, 128))
            img[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = color
    Image.fromarray(img).save(path)


def find_cursor(grid) -> Tuple[int, int]:
    """Find the white cross cursor position in the grid.
    The cursor is a + pattern of colors 0 (white) and 1 (light gray)."""
    # Look for the characteristic cross pattern
    for r in range(1, grid.shape[0] - 1):
        for c in range(1, grid.shape[1] - 1):
            # Center of cross is white (0) with cross arms
            if grid[r, c] == 0:
                # Check if it's a cross shape (at least 3 neighbors are 0 or 1)
                neighbors = [grid[r-1,c], grid[r+1,c], grid[r,c-1], grid[r,c+1]]
                cross_count = sum(1 for n in neighbors if n in (0, 1))
                if cross_count >= 2:
                    return (c, r)  # (x, y)
    return (-1, -1)


def analyze_grid(grid):
    """Analyze the current grid for key elements."""
    # Find cursor
    cx, cy = find_cursor(grid)

    # Count colors
    unique, counts = np.unique(grid, return_counts=True)
    color_dist = dict(zip(unique.tolist(), counts.tolist()))

    # Find step counter (yellow bar at bottom, color 11)
    step_pixels = 0
    for r in range(grid.shape[0] - 5, grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 11:
                step_pixels += 1

    # Find notable colored objects (non-background)
    bg_colors = {3, 4, 5}  # gray, dark gray, black
    notable = {}
    for color_val in unique:
        if int(color_val) not in bg_colors and int(color_val) not in (0, 1) and counts[unique == color_val][0] > 2:
            notable[int(color_val)] = int(counts[unique == color_val][0])

    return {
        'cursor': (cx, cy),
        'step_bar_pixels': step_pixels,
        'notable_colors': notable,
        'color_dist': {int(k): int(v) for k, v in color_dist.items()},
    }


def play_actions(env, fd, actions: List[int], verbose=True):
    """Execute a sequence of actions, reporting each step."""
    grid = get_frame(fd)
    results = []

    for i, a in enumerate(actions):
        prev_grid = grid.copy()
        prev_levels = fd.levels_completed

        fd = env.step(INT_TO_GA[a])
        all_frames = get_all_frames(fd)
        grid = all_frames[-1]

        diff = (prev_grid != grid)
        n_changed = int(np.sum(diff))
        level_up = fd.levels_completed > prev_levels

        result = {
            'step': i + 1,
            'action': ACTION_NAMES[a],
            'px_changed': n_changed,
            'level_up': level_up,
            'levels': fd.levels_completed,
            'state': fd.state.name,
            'anim_frames': len(all_frames),
        }
        results.append(result)

        if verbose:
            sym = '★' if level_up else ('+' if n_changed > 10 else ('-' if n_changed > 0 else '·'))
            extra = f" [ANIMATION {len(all_frames)} frames]" if len(all_frames) > 1 else ""
            print(f"  {sym} {i+1:3d}. {ACTION_NAMES[a]:5s} | {n_changed:4d}px | L={fd.levels_completed}/{fd.win_levels}{extra}")

        if fd.state.name in ('GAME_OVER', 'WON'):
            break

    return fd, grid, results


# ─── Level-aware BFS solver ───

# Game coordinates from source analysis
COLOR_LIST = [12, 9, 14, 8]
ROTATION_LIST = [0, 90, 180, 270]

LEVEL_DATA = [
    # L1: start(34,45) goal(34,10) rot_pickup(19,30) steps=42 dec=1
    {'start': (34,45), 'goals': [(34,10)],
     'goal_cfg': [{'s':5, 'c':1, 'r':0}],  # color_idx, rot_idx
     'start_cfg': {'s':5, 'c':1, 'r':3},
     'rot_pickups': [(19,30)], 'color_pickups': [], 'shape_pickups': [],
     'refills': [], 'steps': 42, 'dec': 1,
     'walls': None},  # populated from solver
    # L2: start(29,40) goal(14,40) rot_pickup(49,45) refills[(15,16),(40,51)] steps=42 dec=2
    {'start': (29,40), 'goals': [(14,40)],
     'goal_cfg': [{'s':5, 'c':1, 'r':3}],
     'start_cfg': {'s':5, 'c':1, 'r':0},
     'rot_pickups': [(49,45)], 'color_pickups': [], 'shape_pickups': [],
     'refills': [(15,16),(40,51)], 'steps': 42, 'dec': 2,
     'walls': None},
    # L3: start(9,45) goal(54,50) col_pickup(29,45) rot_pickup(49,10) refills[(35,16),(20,31)] steps=42 dec=2
    {'start': (9,45), 'goals': [(54,50)],
     'goal_cfg': [{'s':5, 'c':1, 'r':2}],
     'start_cfg': {'s':5, 'c':0, 'r':0},
     'rot_pickups': [(49,10)], 'color_pickups': [(29,45)], 'shape_pickups': [],
     'refills': [(35,16),(20,31)], 'steps': 42, 'dec': 2,
     'walls': None},
    # L4: start(54,5) goal(9,5) col_pickup(34,30) refills[(35,51),(20,16)] steps=42 dec=1
    {'start': (54,5), 'goals': [(9,5)],
     'goal_cfg': [{'s':4, 'c':1, 'r':0}],
     'start_cfg': {'s':4, 'c':2, 'r':0},
     'rot_pickups': [], 'color_pickups': [(34,30)], 'shape_pickups': [],
     'refills': [(35,51),(20,16)], 'steps': 42, 'dec': 1,
     'walls': None},
    # L5: start(49,40) goal(54,5) col_pickup(29,25) rot_pickup(14,35) refills[(15,46),(45,6),(10,11)] steps=42 dec=2
    {'start': (49,40), 'goals': [(54,5)],
     'goal_cfg': [{'s':4, 'c':3, 'r':2}],
     'start_cfg': {'s':4, 'c':0, 'r':0},
     'rot_pickups': [(14,35)], 'color_pickups': [(29,25)], 'shape_pickups': [],
     'refills': [(15,46),(45,6),(10,11)], 'steps': 42, 'dec': 2,
     'walls': None},
    # L6: start(24,50) goals[(54,50),(54,35)] col_pickup(24,30) rot_pickup(34,40) refills[(40,6),(10,46),(10,6)] steps=42 dec=1
    {'start': (24,50), 'goals': [(54,50),(54,35)],
     'goal_cfg': [{'s':0, 'c':1, 'r':1}, {'s':0, 'c':3, 'r':2}],
     'start_cfg': {'s':0, 'c':2, 'r':0},
     'rot_pickups': [(34,40)], 'color_pickups': [(24,30)], 'shape_pickups': [],
     'refills': [(40,6),(10,46),(10,6)], 'steps': 42, 'dec': 1,
     'walls': None},
    # L7: start(19,15) goal(29,50) col_pickup(9,40) rot_pickup(54,10) refills[many] steps=42 dec=2
    {'start': (19,15), 'goals': [(29,50)],
     'goal_cfg': [{'s':1, 'c':3, 'r':2}],
     'start_cfg': {'s':1, 'c':0, 'r':0},
     'rot_pickups': [(54,10)], 'color_pickups': [(9,40)], 'shape_pickups': [],
     'refills': [(30,21),(50,6),(15,46),(40,6),(55,51),(10,6)], 'steps': 42, 'dec': 2,
     'walls': None},
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--actions', type=str, default='',
                        help='Space-separated actions (UP DOWN LEFT RIGHT)')
    parser.add_argument('--explore', action='store_true',
                        help='Execute each direction once to map movement')
    parser.add_argument('--all', action='store_true',
                        help='Try to solve all levels with known L1 solution + exploration')
    args = parser.parse_args()

    arcade = Arcade()
    env = arcade.make('ls20-9607627b')
    fd = env.reset()
    grid = get_frame(fd)

    print(f"GAME: ls20-9607627b")
    print(f"LEVELS: 0/{fd.win_levels}")
    print(f"ACTIONS: UP=1, DOWN=2, LEFT=3, RIGHT=4")

    # Save initial frame
    os.makedirs('/tmp/sage_solver', exist_ok=True)
    render_frame(grid, '/tmp/sage_solver/frame.png')

    info = analyze_grid(grid)
    print(f"CURSOR: {info['cursor']}")
    print(f"STEP BAR: {info['step_bar_pixels']}px")
    print(f"NOTABLE COLORS: {info['notable_colors']}")

    if args.explore:
        print("\n=== EXPLORATION: one of each direction ===")
        for a in [1, 2, 3, 4]:
            prev_grid = grid.copy()
            prev_cursor = find_cursor(grid)
            fd = env.step(INT_TO_GA[a])
            grid = get_frame(fd)
            new_cursor = find_cursor(grid)
            n_changed = int(np.sum(prev_grid != grid))
            dx = new_cursor[0] - prev_cursor[0]
            dy = new_cursor[1] - prev_cursor[1]
            print(f"  {ACTION_NAMES[a]:5s}: cursor {prev_cursor} -> {new_cursor} (dx={dx}, dy={dy}), {n_changed}px changed")

    elif args.actions:
        actions = [NAME_TO_INT[a.strip().upper()] for a in args.actions.split() if a.strip()]
        print(f"\n=== Executing {len(actions)} actions ===")
        fd, grid, results = play_actions(env, fd, actions)
        render_frame(grid, '/tmp/sage_solver/frame.png')
        info = analyze_grid(grid)
        print(f"\nFinal: L={fd.levels_completed}/{fd.win_levels}, state={fd.state.name}")
        print(f"Cursor: {info['cursor']}, Step bar: {info['step_bar_pixels']}px")

    else:
        print("\nUse --actions 'UP DOWN LEFT RIGHT ...' to play")
        print("Use --explore to test each direction")


if __name__ == '__main__':
    main()
