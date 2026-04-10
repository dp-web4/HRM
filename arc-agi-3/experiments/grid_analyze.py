#!/usr/bin/env python3
"""Quick grid analysis for interactive r11l sessions.
Shows exact positions of all game elements in click coordinates (x,y).
Usage: python grid_analyze.py
"""
import numpy as np
import json

COLOR_NAMES = {
    0: 'WHITE', 1: 'off-white', 2: 'lt-gray', 3: 'GRAY',
    5: 'BLACK', 6: 'MAGENTA', 9: 'blue', 10: 'LT-BLUE',
    12: 'ORANGE', 15: 'PURPLE',
}

grid = np.load('/tmp/sage_solver/current_grid.npy')
if grid.ndim == 3:
    grid = grid[-1]

print(f"Grid: {grid.shape}")

# Session info
try:
    with open('/tmp/sage_solver/session.json') as f:
        s = json.load(f)
    print(f"Step: {s['step']}, Levels: {s['levels_completed']}/{s['win_levels']}, "
          f"State: {s['state']}")
except:
    pass

# Find elements (skip background: black=5, lt-gray=2 border)
for val in sorted(set(grid.flatten())):
    if val in (2, 5):  # background
        continue
    coords = [(int(r), int(c)) for r, c in zip(*np.where(grid == val))
              if 5 <= r <= 58 and 5 <= c <= 58]
    if not coords:
        continue

    name = COLOR_NAMES.get(val, f'color_{val}')

    # Cluster by proximity
    sorted_c = sorted(coords)
    clusters = []
    current = [sorted_c[0]]
    for i in range(1, len(sorted_c)):
        if abs(sorted_c[i][0] - current[-1][0]) <= 6 and \
           abs(sorted_c[i][1] - current[-1][1]) <= 6:
            current.append(sorted_c[i])
        else:
            clusters.append(current)
            current = [sorted_c[i]]
    clusters.append(current)

    for cl in clusters:
        rs = [r for r, c in cl]
        cs = [c for r, c in cl]
        cy = sum(rs) // len(rs)
        cx = sum(cs) // len(cs)
        # Click coords are (x=col, y=row)
        shape = "cross" if len(cl) in (10, 11, 12, 13, 14) else \
                "ring" if len(cl) in (8, 12) and max(rs)-min(rs) > 3 else \
                "block" if len(cl) > 5 else \
                "pixel" if len(cl) == 1 else f"{len(cl)}px"
        print(f"  {name:10s} click=({cx},{cy}) grid=({cy},{cx}) n={len(cl):2d} {shape}")
