#!/usr/bin/env python3
"""
tu93 solver — maze navigation with arrows, bouncers, and delayed entities.

Core mechanics (from fleet learning):
- Arrows LETHAL: activate from specific direction (6px in arrow's facing direction).
  Arrow moves toward player and kills in 3 shrink cycles.
  DESTROY arrows by walking INTO them from NON-activation direction.
  Player at arrow position: pbknvgksfm shrinks arrow (3 hits = removed in one turn).

- Bouncers: oscillate along their axis, reverse at walls.
  Player CAN shrink by walking into them.
  Bouncers move 1 step per player turn.

- Delayed: follow player's movement with a lag. Activate when player is within range.

- Grid nodes at positions where BOTH x%6==0 AND y%6==0 relative to maze.
- Movement validation checks pixel 3px ahead (bsfndluqyd=3). Only pixel==2 is walkable.
- Each env.step() = one complete turn (player moves + entities settle).

Strategy: BFS with game state simulation (player pos, entity states, step count).
"""

import sys
import os
import numpy as np
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'environment_files', 'tu93', '2b534c15'))

from tu93 import levels as game_levels

UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4
DIRS = {
    'UP': (0, -6, 0, -3),
    'DOWN': (0, 6, 0, 3),
    'LEFT': (-6, 0, -3, 0),
    'RIGHT': (6, 0, 3, 0),
}
DIR_NAMES = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT'}
NAME_TO_ACT = {'UP': UP, 'DOWN': DOWN, 'LEFT': LEFT, 'RIGHT': RIGHT}


def analyze_level(idx):
    """Extract maze and entity data for a level."""
    lv = game_levels[idx]
    maze = [s for s in lv._sprites if 'vhlesexlqd' in getattr(s, 'tags', [])][0]
    pixels = np.array(maze.pixels)
    mx, my = maze.x, maze.y
    h, w = pixels.shape

    player = exit_p = None
    arrows = []
    bouncers = []
    delayed = []

    for s in lv._sprites:
        tags = getattr(s, 'tags', [])
        if 'albwnmiahg' in tags:
            player = (s.x - mx, s.y - my)
        elif 'xboyuzuyxv' in tags:
            exit_p = (s.x - mx, s.y - my)
        elif 'vllvfeggte' in tags:
            arrows.append((s.x - mx, s.y - my, s.rotation))
        elif 'zzuxulcort' in tags:
            bouncers.append((s.x - mx, s.y - my, s.rotation))
        elif 'natiyqayts' in tags:
            delayed.append((s.x - mx, s.y - my, s.rotation))

    return pixels, h, w, player, exit_p, arrows, bouncers, delayed


def activation_pos(ax, ay, rot):
    """Where must the player be for this arrow to activate?"""
    if rot == 0:    return (ax, ay - 6)
    elif rot == 180: return (ax, ay + 6)
    elif rot == 90:  return (ax + 6, ay)
    elif rot == 270: return (ax - 6, ay)


def can_move(pixels, h, w, x, y, dname):
    """Check if movement from (x,y) in direction is valid."""
    _, _, cx_off, cy_off = DIRS[dname]
    chk_x, chk_y = x + cx_off, y + cy_off
    if 0 <= chk_y < h and 0 <= chk_x < w:
        return pixels[chk_y, chk_x] == 2
    return False


def bouncer_step(pixels, h, w, bx, by, brot):
    """Simulate one bouncer step. Returns new (x, y, rot)."""
    # Check if next position in facing direction is walkable
    if brot == 0:    cx, cy = bx, by - 3
    elif brot == 180: cx, cy = bx, by + 3
    elif brot == 90:  cx, cy = bx + 3, by
    elif brot == 270: cx, cy = bx - 3, by

    if 0 <= cy < h and 0 <= cx < w and pixels[cy, cx] == 2:
        # Move forward
        if brot == 0:    return bx, by - 6, brot
        elif brot == 180: return bx, by + 6, brot
        elif brot == 90:  return bx + 6, by, brot
        elif brot == 270: return bx - 6, by, brot
    else:
        # Reverse direction, stay in place
        rev = {0: 180, 180: 0, 90: 270, 270: 90}
        return bx, by, rev[brot]


def solve_level(idx, max_steps=60):
    """Solve a level using BFS with entity simulation.

    State: (player_pos, alive_arrows_frozen, bouncer_states, step_count)
    """
    pixels, h, w, player, exit_p, arrows, bouncers, delayed = analyze_level(idx)

    n_arrows = len(arrows)
    n_bouncers = len(bouncers)

    # Initial bouncer state
    b_init = tuple((bx, by, br) for bx, by, br in bouncers)

    # For levels with no entities, simple BFS
    if n_arrows == 0 and n_bouncers == 0 and len(delayed) == 0:
        return simple_bfs(pixels, h, w, player, exit_p)

    # State: (player_pos, alive_arrows as frozenset, bouncer_states as tuple)
    start_alive = frozenset(range(n_arrows))
    start_state = (player, start_alive, b_init)

    q = deque([(start_state, [])])
    visited = {start_state}

    while q:
        ((px, py), alive, bstate), path = q.popleft()

        if (px, py) == exit_p:
            return path

        if len(path) >= max_steps:
            continue

        # Simulate bouncer movement for this turn
        new_bstate = tuple(bouncer_step(pixels, h, w, *b) for b in bstate)
        bouncer_positions_after = set((bx, by) for bx, by, _ in new_bstate)

        for dname, (dx, dy, cx_off, cy_off) in DIRS.items():
            if not can_move(pixels, h, w, px, py, dname):
                continue

            dest = (px + dx, py + dy)

            # 1. Player lands at dest. Destroy arrows there.
            new_alive = set(alive)
            for i in alive:
                if (arrows[i][0], arrows[i][1]) == dest:
                    new_alive.discard(i)

            # 2. Check arrow activation (lethal if player at activation pos)
            lethal = False
            for i in new_alive:
                if dest == activation_pos(*arrows[i]):
                    lethal = True
                    break
            if lethal:
                continue

            # 3. Check bouncer collision (bouncer at dest after their move)
            if dest in bouncer_positions_after:
                # Player and bouncer at same position — player walks into bouncer
                # This DESTROYS the bouncer (shrinks it). But for simplicity,
                # treat it as a valid move (bouncer gets damaged, path cleared).
                # In practice, the bouncer may still kill the player.
                # For safety, avoid bouncer positions.
                continue

            # 4. Also check: bouncer's CURRENT position (before move) at dest
            bouncer_current = set((bx, by) for bx, by, _ in bstate)
            if dest in bouncer_current:
                # Walking into a bouncer's current position — player might destroy it
                # or bouncer might kill player. Skip for safety.
                continue

            state = (dest, frozenset(new_alive), new_bstate)
            if state not in visited:
                visited.add(state)
                q.append((state, path + [dname]))

    # If safe BFS failed, try allowing bouncer positions (player destroys bouncer)
    return solve_level_aggressive(idx, max_steps)


def solve_level_aggressive(idx, max_steps=60):
    """More aggressive solve — allow walking into bouncers."""
    pixels, h, w, player, exit_p, arrows, bouncers, delayed = analyze_level(idx)

    n_arrows = len(arrows)
    b_init = tuple((bx, by, br) for bx, by, br in bouncers)
    start_alive = frozenset(range(n_arrows))
    start_state = (player, start_alive, b_init)

    q = deque([(start_state, [])])
    visited = {start_state}

    while q:
        ((px, py), alive, bstate), path = q.popleft()

        if (px, py) == exit_p:
            return path
        if len(path) >= max_steps:
            continue

        new_bstate = tuple(bouncer_step(pixels, h, w, *b) for b in bstate)

        for dname, (dx, dy, cx_off, cy_off) in DIRS.items():
            if not can_move(pixels, h, w, px, py, dname):
                continue

            dest = (px + dx, py + dy)

            new_alive = set(alive)
            for i in alive:
                if (arrows[i][0], arrows[i][1]) == dest:
                    new_alive.discard(i)

            lethal = False
            for i in new_alive:
                if dest == activation_pos(*arrows[i]):
                    lethal = True
                    break
            if lethal:
                continue

            # Allow bouncer positions — player destroys bouncer
            state = (dest, frozenset(new_alive), new_bstate)
            if state not in visited:
                visited.add(state)
                q.append((state, path + [dname]))

    return None


def simple_bfs(pixels, h, w, start, goal):
    """Simple BFS for levels with no entities."""
    q = deque([(start, [])])
    visited = {start}

    while q:
        (px, py), path = q.popleft()
        if (px, py) == goal:
            return path
        for dname, (dx, dy, cx_off, cy_off) in DIRS.items():
            if can_move(pixels, h, w, px, py, dname):
                dest = (px + dx, py + dy)
                if dest not in visited:
                    visited.add(dest)
                    q.append((dest, path + [dname]))
    return None


def main():
    """Solve all 9 levels and output action sequences."""
    print("tu93 Solver — Entity-Aware BFS\n")

    solutions = {}
    total = 0

    for idx in range(9):
        pixels, h, w, player, exit_p, arrows, bouncers, delayed = analyze_level(idx)
        n_entities = len(arrows) + len(bouncers) + len(delayed)

        print(f"L{idx+1}: player={player} exit={exit_p} "
              f"arrows={len(arrows)} bouncers={len(bouncers)} delayed={len(delayed)}")

        path = solve_level(idx)

        if path:
            actions = [NAME_TO_ACT[d] for d in path]
            solutions[idx] = {
                'path': path,
                'actions': actions,
                'moves': len(path),
            }
            total += len(path)
            print(f"  SOLVED in {len(path)} moves: {' '.join(path)}")
        else:
            print(f"  NO SOLUTION FOUND")
            break

    print(f"\nTotal: {total} moves across {len(solutions)} levels")
    return solutions


if __name__ == "__main__":
    solutions = main()
