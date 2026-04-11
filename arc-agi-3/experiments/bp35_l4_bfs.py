#!/usr/bin/env python3
"""BP35 L4 BFS solver.

L4: player (3,12), gem (5,7), gravity UP, no rising floor
14 D blocks, 2 G blocks, 2 O/B toggleable blocks, 13 spikes
Baseline: 31 actions
"""
import sys, os, time
from collections import deque
sys.stdout.reconfigure(line_buffering=True)

WALLS = set()
SPIKES = set()
GEM = (5, 7)

# Grid from engine survey: x = char_index - 2, rows are 17 chars (x=-2 to x=14)
# P replaced with '.', trailing dots are empty space
grid_rows = {
    0:  "..WWWWWWWWWWW....",
    1:  "..WWWWWWWWWWW....",
    2:  "..WWWWWWWWWWW....",
    3:  "..WWWWWWWWWWW....",
    4:  "..WWWWWWWWWWW....",
    5:  "..WWWWWWWWWWW....",
    6:  "..WWWWWWWWWWW....",
    7:  "..WWWW.*.^^^W....",
    8:  "..WWWW...DDDW....",
    9:  "..WWWW...DDDW....",
    10: "..WWWWWWWWW.W....",
    11: "..WWWWWWWWW.W....",
    12: "..WW......G.W....",
    13: "..WW......WWW....",
    14: "..WWvvvv..WWW....",
    15: "..WWWWWW..WWW....",
    16: "..WWWWWWDDWWW....",
    17: "..WW......WWW....",
    18: "..WW.WWW..^^W....",
    19: "..WW.WWW....W....",
    20: "..WW.^^WDD..W....",
    21: "..WW.OOWWWDDW....",
    22: "..WW........W....",
    23: "..WW........W....",
    24: "..WWW.....WWW....",
    25: "..WWW..WWWWWW....",
    26: "..WWWDDWWWWWW....",
    27: "..WWW..WWWWWW....",
    28: "..WWWvvWWWWWW....",
    29: "..WWWWWWWWGWW....",
    30: "..WWWWWWWWWWW....",
}

for y, row in grid_rows.items():
    for i, ch in enumerate(row):
        x = i - 2
        if ch == 'W':
            WALLS.add((x, y))
        elif ch in ('^', 'v'):
            SPIKES.add((x, y))

D_BLOCKS = [
    (7, 8), (8, 8), (9, 8),
    (7, 9), (8, 9), (9, 9),
    (6, 16), (7, 16),
    (6, 20), (7, 20),
    (8, 21), (9, 21),
    (3, 26), (4, 26),
]
D_IDX = {pos: i for i, pos in enumerate(D_BLOCKS)}

G_BLOCKS = [(8, 12), (8, 29)]
G_IDX = {pos: i for i, pos in enumerate(G_BLOCKS)}

OB_BLOCKS = [(3, 21), (4, 21)]
OB_IDX = {pos: i for i, pos in enumerate(OB_BLOCKS)}

START = (3, 12)
START_GRAV = True


def is_solid(x, y, d_alive, g_alive, ob_state):
    if (x, y) in WALLS or (x, y) in SPIKES:
        return True
    if (x, y) in D_IDX and (d_alive >> D_IDX[(x, y)]) & 1:
        return True
    if (x, y) in G_IDX and (g_alive >> G_IDX[(x, y)]) & 1:
        return True
    if (x, y) in OB_IDX:
        return bool((ob_state >> OB_IDX[(x, y)]) & 1)
    return False


def fall(x, y, grav_up, d_alive, g_alive, ob_state):
    dy = -1 if grav_up else 1
    last_safe = (x, y)
    cx, cy = x, y + dy
    while True:
        if (cx, cy) == GEM:
            return cx, cy, True
        if is_solid(cx, cy, d_alive, g_alive, ob_state):
            return last_safe[0], last_safe[1], False
        if cy < -5 or cy > 35:
            return last_safe[0], last_safe[1], False
        last_safe = (cx, cy)
        cy += dy


def simulate_move(px, py, dx, grav_up, d_alive, g_alive, ob_state):
    nx, ny = px + dx, py
    if (nx, ny) == GEM:
        return nx, ny, True
    if is_solid(nx, ny, d_alive, g_alive, ob_state):
        return px, py, False
    fx, fy, won = fall(nx, ny, grav_up, d_alive, g_alive, ob_state)
    return fx, fy, won


def simulate_click_d(px, py, dpos, grav_up, d_alive, g_alive, ob_state):
    idx = D_IDX.get(dpos)
    if idx is None or not ((d_alive >> idx) & 1):
        return px, py, d_alive, False
    new_d = d_alive & ~(1 << idx)
    grav_dy = -1 if grav_up else 1
    if dpos == (px, py + grav_dy):
        fx, fy, won = fall(px, py, grav_up, new_d, g_alive, ob_state)
        return fx, fy, new_d, won
    return px, py, new_d, False


def simulate_click_g(px, py, gpos, grav_up, d_alive, g_alive, ob_state):
    idx = G_IDX.get(gpos)
    if idx is None or not ((g_alive >> idx) & 1):
        return px, py, grav_up, g_alive, False
    new_grav = not grav_up
    new_g = g_alive & ~(1 << idx)
    grav_dy = -1 if new_grav else 1
    if is_solid(px, py + grav_dy, d_alive, new_g, ob_state):
        return px, py, new_grav, new_g, False
    fx, fy, won = fall(px, py, new_grav, d_alive, new_g, ob_state)
    return fx, fy, new_grav, new_g, won


def simulate_click_ob(px, py, obpos, grav_up, d_alive, g_alive, ob_state):
    idx = OB_IDX.get(obpos)
    if idx is None:
        return px, py, ob_state, False
    new_ob = ob_state ^ (1 << idx)
    was_solid = (ob_state >> idx) & 1
    now_solid = (new_ob >> idx) & 1
    if was_solid and not now_solid:
        grav_dy = -1 if grav_up else 1
        if obpos == (px, py + grav_dy):
            fx, fy, won = fall(px, py, grav_up, d_alive, g_alive, new_ob)
            return fx, fy, new_ob, won
    return px, py, new_ob, False


def solve():
    initial_d = (1 << len(D_BLOCKS)) - 1
    initial_g = (1 << len(G_BLOCKS)) - 1
    initial_ob = 0  # all O (passable)
    start_state = (START[0], START[1], START_GRAV, initial_d, initial_g, initial_ob)

    queue = deque()
    queue.append((start_state, []))
    visited = {start_state}

    t0 = time.time()
    n_visited = 0

    while queue:
        state, path = queue.popleft()
        px, py, grav_up, d_alive, g_alive, ob_state = state

        n_visited += 1
        if n_visited % 500000 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_visited}] visited, queue={len(queue)}, depth={len(path)}, {elapsed:.1f}s")

        if len(path) >= 35:
            continue

        # 1. LEFT
        nx, ny, won = simulate_move(px, py, -1, grav_up, d_alive, g_alive, ob_state)
        if won:
            sol = path + ['L']
            print(f"  WIN at depth {len(sol)}: {sol}")
            return sol
        ns = (nx, ny, grav_up, d_alive, g_alive, ob_state)
        if ns not in visited:
            visited.add(ns)
            queue.append((ns, path + ['L']))

        # 2. RIGHT
        nx, ny, won = simulate_move(px, py, 1, grav_up, d_alive, g_alive, ob_state)
        if won:
            sol = path + ['R']
            print(f"  WIN at depth {len(sol)}: {sol}")
            return sol
        ns = (nx, ny, grav_up, d_alive, g_alive, ob_state)
        if ns not in visited:
            visited.add(ns)
            queue.append((ns, path + ['R']))

        # 3. Click D blocks
        for dpos in D_BLOCKS:
            if not ((d_alive >> D_IDX[dpos]) & 1):
                continue
            nx, ny, new_d, won = simulate_click_d(px, py, dpos, grav_up, d_alive, g_alive, ob_state)
            if won:
                sol = path + [('D', dpos)]
                print(f"  WIN at depth {len(sol)}: {sol}")
                return sol
            ns = (nx, ny, grav_up, new_d, g_alive, ob_state)
            if ns not in visited:
                visited.add(ns)
                queue.append((ns, path + [('D', dpos)]))

        # 4. Click G blocks
        for gpos in G_BLOCKS:
            if not ((g_alive >> G_IDX[gpos]) & 1):
                continue
            nx, ny, new_grav, new_g, won = simulate_click_g(px, py, gpos, grav_up, d_alive, g_alive, ob_state)
            if won:
                sol = path + [('G', gpos)]
                print(f"  WIN at depth {len(sol)}: {sol}")
                return sol
            ns = (nx, ny, new_grav, d_alive, new_g, ob_state)
            if ns not in visited:
                visited.add(ns)
                queue.append((ns, path + [('G', gpos)]))

        # 5. Click O/B blocks
        for obpos in OB_BLOCKS:
            nx, ny, new_ob, won = simulate_click_ob(px, py, obpos, grav_up, d_alive, g_alive, ob_state)
            if won:
                sol = path + [('T', obpos)]
                print(f"  WIN at depth {len(sol)}: {sol}")
                return sol
            ns = (nx, ny, grav_up, d_alive, g_alive, new_ob)
            if ns not in visited:
                visited.add(ns)
                queue.append((ns, path + [('T', obpos)]))

    elapsed = time.time() - t0
    print(f"  No solution found! Visited {n_visited} states in {elapsed:.1f}s")
    return None


print("BP35 L4 BFS Solver (fixed grid)")
print(f"D: {len(D_BLOCKS)}, G: {len(G_BLOCKS)}, O/B: {len(OB_BLOCKS)}")
print(f"Walls: {len(WALLS)}, Spikes: {len(SPIKES)}")
print()

result = solve()
if result:
    print(f"\nSolution ({len(result)} steps): {result}")
