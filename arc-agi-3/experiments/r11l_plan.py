#!/usr/bin/env python3
"""
Plan r11l moves using exact obstacle maps from source.

Given current ball position, endpoints, and target:
  - Finds optimal sequence of select+place moves
  - Uses exact bvzgd/qtwnv collision maps (not rendered grid approximation)
  - Outputs click coordinates ready to execute

Usage:
    python r11l_plan.py
    # Reads current state from /tmp/sage_solver/, outputs move plan
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from r11l_obstacles import (
    build_obstacle_set, check_endpoint_dest, check_ball_landing, SOURCE
)


def load_state():
    """Load current game state."""
    grid = np.load('/tmp/sage_solver/current_grid.npy')
    with open('/tmp/sage_solver/session.json') as f:
        session = json.load(f)
    return grid, session


def find_elements(grid):
    """Find balls, endpoints, targets from grid."""
    # Magenta (6) center pixels = ball positions
    balls = []
    magenta = [(int(r), int(c)) for r, c in zip(*np.where(grid == 6))
               if 5 <= r <= 58 and 5 <= c <= 58]
    for r, c in magenta:
        balls.append({'click': (c, r), 'color': int(grid[r - 1, c])})

    # Endpoints: white (0) or gray (3) crosses with colored center pixels
    endpoints = []
    for val, selected in [(0, True), (3, False)]:
        coords = [(int(r), int(c)) for r, c in zip(*np.where(grid == val))
                  if 5 <= r <= 58 and 5 <= c <= 58]
        # Cluster
        sorted_c = sorted(coords)
        clusters = []
        current = [sorted_c[0]] if sorted_c else []
        for i in range(1, len(sorted_c)):
            if abs(sorted_c[i][0] - current[-1][0]) <= 4 and \
               abs(sorted_c[i][1] - current[-1][1]) <= 4:
                current.append(sorted_c[i])
            else:
                clusters.append(current)
                current = [sorted_c[i]]
        if current:
            clusters.append(current)

        for cl in clusters:
            if 8 <= len(cl) <= 14:
                rs = [r for r, c in cl]
                cs = [c for r, c in cl]
                cy = sum(rs) // len(rs)
                cx = sum(cs) // len(cs)
                center_color = int(grid[cy, cx])
                if center_color not in (0, 3, 5):  # has a colored center
                    endpoints.append({
                        'click': (cx, cy),
                        'color': center_color,
                        'selected': selected,
                    })

    # Targets: ring shapes (colored pixels with different center)
    targets = []
    for val in [7, 8, 9, 11, 12, 14, 15]:
        coords = [(int(r), int(c)) for r, c in zip(*np.where(grid == val))
                  if 5 <= r <= 58 and 5 <= c <= 58]
        sorted_c = sorted(coords)
        clusters = []
        current = [sorted_c[0]] if sorted_c else []
        for i in range(1, len(sorted_c)):
            if abs(sorted_c[i][0] - current[-1][0]) <= 6 and \
               abs(sorted_c[i][1] - current[-1][1]) <= 6:
                current.append(sorted_c[i])
            else:
                clusters.append(current)
                current = [sorted_c[i]]
        if current:
            clusters.append(current)

        for cl in clusters:
            if 6 <= len(cl) <= 14:
                rs = [r for r, c in cl]
                cs = [c for r, c in cl]
                cy = sum(rs) // len(rs)
                cx = sum(cs) // len(cs)
                center_val = int(grid[cy, cx])
                if center_val != val:  # hollow = target ring
                    targets.append({'click': (cx, cy), 'color': val})

    return balls, endpoints, targets


def group_compasses(balls, endpoints):
    """Group endpoints by color → compass."""
    compasses = {}
    for b in balls:
        color = b['color']
        ends = [e for e in endpoints if e['color'] == color]
        compasses[color] = {
            'ball': b['click'],
            'ends': [e['click'] for e in ends],
            'n_legs': len(ends),
        }
    return compasses


def solve_compass(ball, ends, target, bvzgd, qtwnv, max_moves=10):
    """Find move sequence to walk ball to target.

    Returns list of (end_index, select_click, place_click) tuples.
    """
    n = len(ends)
    ball = list(ball)
    ends = [list(e) for e in ends]

    # Build valid endpoint destinations (precompute)
    valid_dests = []
    for ry in range(6, 58, 2):
        for cx in range(6, 58, 2):
            if check_endpoint_dest(bvzgd, cx, ry, margin=3):
                valid_dests.append((cx, ry))

    moves = []
    for step in range(max_moves):
        best = None
        for ei in range(n):
            ex, ey = ends[ei]
            for px, py in valid_dests:
                # New ball position
                bx = ball[0] + (px - ex) / n
                by = ball[1] + (py - ey) / n
                if not (5 < bx < 59 and 5 < by < 59):
                    continue
                if qtwnv and not check_ball_landing(qtwnv, bx, by, margin=2):
                    continue
                dist = ((bx - target[0])**2 + (by - target[1])**2)**0.5
                if best is None or dist < best[0]:
                    best = (dist, ei, ex, ey, px, py, bx, by)

        if best is None:
            print(f"  STUCK at step {step + 1}")
            break

        dist, ei, ex, ey, px, py, bx, by = best
        select_click = (int(ex), int(ey))
        place_click = (int(px), int(py))
        moves.append((select_click, place_click))
        ends[ei] = [px, py]
        ball = [bx, by]

        if dist < 3:
            break

    return moves, (ball[0], ball[1])


def main():
    grid, session = load_state()
    level = session.get('levels_completed', 0)
    step = session.get('step', 0)

    print(f"Level {level}, Step {step}, Budget: {60 - step}")

    with open(SOURCE) as f:
        src = f.read()

    bvzgd = build_obstacle_set(src, level, 'bvzgd')
    qtwnv = build_obstacle_set(src, level, 'qtwnv')

    balls, endpoints, targets = find_elements(grid)

    print(f"\nBalls: {[(b['click'], b['color']) for b in balls]}")
    print(f"Targets: {[(t['click'], t['color']) for t in targets]}")
    print(f"Endpoints: {[(e['click'], e['color'], 'SEL' if e['selected'] else '') for e in endpoints]}")

    compasses = group_compasses(balls, endpoints)

    # Match targets to compasses by color
    all_moves = []
    for t in targets:
        color = t['color']
        if color not in compasses:
            print(f"\n  No compass for target color {color} at {t['click']}")
            continue
        comp = compasses[color]
        print(f"\n  Compass color={color}: ball={comp['ball']} → target={t['click']}, {comp['n_legs']} legs")

        moves, final_ball = solve_compass(
            comp['ball'], comp['ends'], t['click'],
            bvzgd, qtwnv, max_moves=8)

        for sel, plc in moves:
            print(f"    select ({sel[0]},{sel[1]}) → place ({plc[0]},{plc[1]})")
            all_moves.append((sel, plc))

        print(f"    Ball lands at ({final_ball[0]:.0f},{final_ball[1]:.0f})")

    total_steps = len(all_moves) * 2
    print(f"\nTotal: {len(all_moves)} moves = {total_steps} steps")
    print(f"Budget after: {60 - step - total_steps}")

    # Output as executable commands
    print("\n# Commands:")
    for sel, plc in all_moves:
        print(f"python3 sage_solver.py --interactive step 6 {sel[0]} {sel[1]}")
        print(f"python3 sage_solver.py --interactive step 6 {plc[0]} {plc[1]}")


if __name__ == '__main__':
    main()
