#!/usr/bin/env python3
"""BP35 L5 BFS solver.

L5: player (3,23), gem (2,31), gravity UP
3 G blocks, 11 O blocks + 1 B block (toggleable), 0 D blocks, 17 spikes
Baseline: 48 actions
"""
import sys, time
from collections import deque
sys.stdout.reconfigure(line_buffering=True)

WALLS = set()
SPIKES = set()
GEM = (2, 31)

grid_rows = {
    0:  "..WWWWWWWWWWW....",
    1:  "..WWWWWWWWGWW....",
    2:  "..WWWWWWWWWWW....",
    3:  "..WWWWWWWWWWW....",
    4:  "..WWWWWWWWWWW....",
    5:  "..WWWWWWWWWWW....",
    6:  "..WWWWWWWWWWW....",
    7:  "..WWWWWWW..WW....",
    8:  "..WWWW.....WW....",
    9:  "..WWWW.WWWWWW....",
    10: "..WWWW.^^^^WW....",
    11: "..WW.......WW....",
    12: "..WW.......WW....",
    13: "..WWOOOOBOOWW....",
    14: "..WW.......WW....",
    15: "..WWvvvv...WW....",
    16: "..WWWWWW.WWWW....",
    17: "..WWWWWW.WWWW....",
    18: "..WW^^^....WW....",
    19: "..WW.....W.WW....",
    20: "..WWOOOWWW.WW....",
    21: "..WW.......WW....",
    22: "..WWWWWWGWWWW....",
    23: "..WW.......WW....",
    24: "..WW.......WW....",
    25: "..WW....OO.WW....",
    26: "..WWvvvvvv.WW....",
    27: "..WWWWWWWW.WW....",
    28: "..WWWWWWWW.WW....",
    29: "..WWWWWWWW.WW....",
    30: "..W...W....WW....",
    31: "..W.*.G....WW....",
    32: "..WWWWWWWWWWW....",
}

for y, row in grid_rows.items():
    for i, ch in enumerate(row):
        x = i - 2
        if ch == 'W':
            WALLS.add((x, y))
        elif ch in ('^', 'v'):
            SPIKES.add((x, y))

G_BLOCKS = [(8, 1), (6, 22), (4, 31)]
G_IDX = {pos: i for i, pos in enumerate(G_BLOCKS)}

# O/B toggleable: 0 = O (passable), 1 = B (solid)
# Initial state: all O except (6,13) which starts as B
OB_BLOCKS = [
    (2, 13), (3, 13), (4, 13), (5, 13), (6, 13), (7, 13), (8, 13),  # y=13
    (2, 20), (3, 20), (4, 20),  # y=20
    (6, 25), (7, 25),  # y=25
]
OB_IDX = {pos: i for i, pos in enumerate(OB_BLOCKS)}
# Initial OB state: (6,13) = B = 1, all others = O = 0
INITIAL_OB = 1 << OB_IDX[(6, 13)]

START = (3, 23)
START_GRAV = True


def is_solid(x, y, g_alive, ob_state):
    if (x, y) in WALLS or (x, y) in SPIKES:
        return True
    if (x, y) in G_IDX and (g_alive >> G_IDX[(x, y)]) & 1:
        return True
    if (x, y) in OB_IDX:
        return bool((ob_state >> OB_IDX[(x, y)]) & 1)
    return False


def fall(x, y, grav_up, g_alive, ob_state):
    dy = -1 if grav_up else 1
    last_safe = (x, y)
    cx, cy = x, y + dy
    while True:
        if (cx, cy) == GEM:
            return cx, cy, True
        if is_solid(cx, cy, g_alive, ob_state):
            return last_safe[0], last_safe[1], False
        if cy < -5 or cy > 40:
            return last_safe[0], last_safe[1], False
        last_safe = (cx, cy)
        cy += dy


def simulate_move(px, py, dx, grav_up, g_alive, ob_state):
    nx, ny = px + dx, py
    if (nx, ny) == GEM:
        return nx, ny, True
    if is_solid(nx, ny, g_alive, ob_state):
        return px, py, False
    fx, fy, won = fall(nx, ny, grav_up, g_alive, ob_state)
    return fx, fy, won


def simulate_click_g(px, py, gpos, grav_up, g_alive, ob_state):
    idx = G_IDX.get(gpos)
    if idx is None or not ((g_alive >> idx) & 1):
        return px, py, grav_up, g_alive, False
    new_grav = not grav_up
    new_g = g_alive & ~(1 << idx)
    grav_dy = -1 if new_grav else 1
    if is_solid(px, py + grav_dy, new_g, ob_state):
        return px, py, new_grav, new_g, False
    fx, fy, won = fall(px, py, new_grav, new_g, ob_state)
    return fx, fy, new_grav, new_g, won


def simulate_click_ob(px, py, obpos, grav_up, g_alive, ob_state):
    idx = OB_IDX.get(obpos)
    if idx is None:
        return px, py, ob_state, False
    new_ob = ob_state ^ (1 << idx)
    was_solid = (ob_state >> idx) & 1
    if was_solid:
        # Became passable - check if player falls
        grav_dy = -1 if grav_up else 1
        if obpos == (px, py + grav_dy):
            fx, fy, won = fall(px, py, grav_up, g_alive, new_ob)
            return fx, fy, new_ob, won
    return px, py, new_ob, False


def solve():
    initial_g = (1 << len(G_BLOCKS)) - 1
    start_state = (START[0], START[1], START_GRAV, initial_g, INITIAL_OB)

    queue = deque()
    queue.append((start_state, []))
    visited = {start_state}

    t0 = time.time()
    n_visited = 0

    while queue:
        state, path = queue.popleft()
        px, py, grav_up, g_alive, ob_state = state

        n_visited += 1
        if n_visited % 500000 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_visited}] visited, queue={len(queue)}, depth={len(path)}, {elapsed:.1f}s")

        if len(path) >= 50:  # baseline is 48
            continue

        actions = []

        # LEFT/RIGHT
        for dx, label in [(-1, 'L'), (1, 'R')]:
            nx, ny, won = simulate_move(px, py, dx, grav_up, g_alive, ob_state)
            if won:
                sol = path + [label]
                print(f"  WIN at depth {len(sol)}: {sol}")
                return sol
            ns = (nx, ny, grav_up, g_alive, ob_state)
            if ns not in visited:
                visited.add(ns)
                queue.append((ns, path + [label]))

        # Click G blocks
        for gpos in G_BLOCKS:
            if not ((g_alive >> G_IDX[gpos]) & 1):
                continue
            nx, ny, new_grav, new_g, won = simulate_click_g(px, py, gpos, grav_up, g_alive, ob_state)
            if won:
                sol = path + [('G', gpos)]
                print(f"  WIN at depth {len(sol)}: {sol}")
                return sol
            ns = (nx, ny, new_grav, new_g, ob_state)
            if ns not in visited:
                visited.add(ns)
                queue.append((ns, path + [('G', gpos)]))

        # Click O/B blocks
        for obpos in OB_BLOCKS:
            nx, ny, new_ob, won = simulate_click_ob(px, py, obpos, grav_up, g_alive, ob_state)
            if won:
                sol = path + [('T', obpos)]
                print(f"  WIN at depth {len(sol)}: {sol}")
                return sol
            ns = (nx, ny, grav_up, g_alive, new_ob)
            if ns not in visited:
                visited.add(ns)
                queue.append((ns, path + [('T', obpos)]))

    elapsed = time.time() - t0
    print(f"  No solution found! Visited {n_visited} states in {elapsed:.1f}s")
    return None


print("BP35 L5 BFS Solver")
print(f"G: {len(G_BLOCKS)}, O/B: {len(OB_BLOCKS)}")
print()

result = solve()
if result:
    print(f"\nSolution ({len(result)} steps): {result}")
