#!/usr/bin/env python3
"""Solve cd82: circular stamp painting puzzle.

World model:
- Canvas: 10x10 grid, starts all-white (color 0)
- Target: 10x10 pattern to match (ignoring X-diagonal cells)
- Orbit: 8 positions (0-7) around canvas on a 3x3 grid (center excluded)
  Position layout (grid row, col):
    7(0,0) 0(0,1) 1(0,2)
    6(1,0)  [X]   2(1,2)
    5(2,0) 4(2,1) 3(2,2)
- Navigate with UDLR (ACTION1-4) between adjacent positions
- Colors: palette of available colors, click swatch to select (ACTION6)
- Big stamp (ACTION5=SELECT): paints half canvas based on position
  0=top, 1=upper-right-tri, 2=right, 3=lower-right-tri,
  4=bottom, 5=lower-left-tri, 6=left, 7=upper-left-tri
- Small stamp (ACTION6 on ctwspzkygu): paints 3x4 or 4x3 near edge
  0=top-center, 2=right-center, 4=bottom-center, 6=left-center
- Win condition: canvas matches target on non-diagonal cells
  (cells where row!=col and row!=9-col)
- Stamps layer: last stamp wins on overlapping cells

Solutions found by brute-force search over stamp sequences:
  L0: 1 stamp  — B4(15)
  L1: 2 stamps — B0(15), B3(12)
  L2: 4 stamps — B2(14), B6(8), B7(15), S0(12)
  L3: 4 stamps — B0(12), B3(15), B6(9), S6(11)
  L4: 4 stamps — B0(9), B5(14), B3(12), S0(8)
  L5: 4 stamps — B2(14), B7(8), S0(15), S6(11)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
from collections import deque

arcade = Arcade()

# Orbit positions on 3x3 grid (excluding center)
ORBIT_TO_GRID = {
    0: (0, 1), 1: (0, 2), 2: (1, 2), 3: (2, 2),
    4: (2, 1), 5: (2, 0), 6: (1, 0), 7: (0, 0),
}

def nav_path(from_pos, to_pos):
    """Return list of action IDs (1=UP,2=DOWN,3=LEFT,4=RIGHT) to navigate orbit."""
    if from_pos == to_pos:
        return []
    start = ORBIT_TO_GRID[from_pos]
    target = ORBIT_TO_GRID[to_pos]
    grid_to_orbit = {v: k for k, v in ORBIT_TO_GRID.items()}
    visited = {start: None}
    queue = deque([start])
    while queue:
        gr, gc = queue.popleft()
        if (gr, gc) == target:
            path = []
            pos = (gr, gc)
            while visited[pos] is not None:
                prev, action = visited[pos]
                path.append(action)
                pos = prev
            path.reverse()
            return path
        for dgr, dgc, act in [(-1,0,1), (1,0,2), (0,-1,3), (0,1,4)]:
            nr, nc = gr + dgr, gc + dgc
            if 0 <= nr <= 2 and 0 <= nc <= 2 and (nr, nc) != (1, 1):
                if (nr, nc) not in visited:
                    visited[(nr, nc)] = ((gr, gc), act)
                    queue.append((nr, nc))
    return []

# Pre-computed solutions for each level
# Each entry: list of (stamp_type, orbit_position, color)
# stamp_type: 'big' (ACTION5) or 'small' (click ctwspzkygu)
SOLUTIONS = {
    0: [('big', 4, 15)],
    1: [('big', 0, 15), ('big', 3, 12)],
    2: [('big', 2, 14), ('big', 6, 8), ('big', 7, 15), ('small', 0, 12)],
    3: [('big', 0, 12), ('big', 3, 15), ('big', 6, 9), ('small', 6, 11)],
    4: [('big', 0, 9), ('big', 5, 14), ('big', 3, 12), ('small', 0, 8)],
    5: [('big', 2, 14), ('big', 7, 8), ('small', 0, 15), ('small', 6, 11)],
}


def click_color(env, game, color):
    """Click color swatch to select a color. Returns (fd, 1) or (None, 0)."""
    if game.knqmgavuh == color:
        return None, 0
    palette = [s for s in game.current_level.get_sprites() if s.name.startswith('pqkenviek')]
    for cs in palette:
        if int(cs.pixels[2, 2]) == color:
            scale, x_off, y_off = game.camera._calculate_scale_and_offset()
            cx = (cs.x + 2) * scale + x_off
            cy = (cs.y + 2) * scale + y_off
            fd = env.step(GameAction.ACTION6, data={'x': cx, 'y': cy})
            return fd, 1
    return None, 0


def do_big_stamp(env, game):
    """Execute big stamp (SELECT). Returns (fd, actions_used)."""
    fd = env.step(GameAction.ACTION5)
    actions = 1
    # Pump animation until complete
    while game.edjesyzxk:
        fd = env.step(GameAction.ACTION5)
        actions += 1
    return fd, actions


def do_small_stamp(env, game, orbit_pos):
    """Execute small stamp at given orbit position. Returns (fd, actions_used)."""
    small_sprite = game.oaouwdxbxc(orbit_pos)
    if not small_sprite:
        print(f"    ERROR: no small stamp sprite at pos {orbit_pos}")
        return None, 0
    scale, x_off, y_off = game.camera._calculate_scale_and_offset()
    cx = (small_sprite.x + small_sprite.width // 2) * scale + x_off
    cy = (small_sprite.y + small_sprite.height // 2) * scale + y_off
    fd = env.step(GameAction.ACTION6, data={'x': cx, 'y': cy})
    actions = 1
    # Pump animation
    while game.yfobpcuef:
        # During animation, any action pumps it. Use a neutral click.
        fd = env.step(GameAction.ACTION6, data={'x': cx, 'y': cy})
        actions += 1
    return fd, actions


# === SOLVE ===
env = arcade.make('cd82-fb555c5d')
fd = env.reset()
game = env._game

ACTION_MAP = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
              3: GameAction.ACTION3, 4: GameAction.ACTION4}

total_actions = 0

for lv in range(6):
    level = game.current_level
    print(f"\n=== Level {lv} ===")

    sequence = SOLUTIONS[lv]
    current_pos = game.xwmfgtlso
    lv_actions = 0

    for stype, target_pos, color in sequence:
        # 1. Select color
        fd_c, acts_c = click_color(env, game, color)
        lv_actions += acts_c
        if fd_c:
            fd = fd_c

        # 2. Navigate
        path = nav_path(current_pos, target_pos)
        for act in path:
            fd = env.step(ACTION_MAP[act])
            lv_actions += 1
        current_pos = target_pos

        # 3. Stamp
        if stype == 'big':
            fd_s, acts_s = do_big_stamp(env, game)
        else:
            fd_s, acts_s = do_small_stamp(env, game, target_pos)
        lv_actions += acts_s
        if fd_s:
            fd = fd_s

    total_actions += lv_actions
    print(f"  {lv_actions} actions, completed={fd.levels_completed}, state={fd.state.name}")

    if fd.state.name == 'WIN':
        print(f"\n=== ALL SOLVED! Total: {total_actions} ===")
        break

    if fd.levels_completed <= lv:
        # Debug
        canvas = game.current_level.get_sprites_by_name('xytrjjbyib')
        if canvas:
            c = canvas[0].pixels
            print(f"  Canvas state:")
            for r in range(10):
                print(f"    {list(c[r])}")
        print(f"  LEVEL NOT COMPLETED")
        break

print(f"\nFinal: completed={fd.levels_completed}, state={fd.state.name}, total={total_actions}")
