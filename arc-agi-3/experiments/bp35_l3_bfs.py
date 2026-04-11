#!/usr/bin/env python3
"""BP35 L3 BFS solver: find optimal path through gravity-flip puzzle.

State: (player_x, player_y, gravity_up, d_blocks_alive, g_blocks_alive)
D blocks: 10 blocks encoded as bitmask
G blocks: 4 (including walled-off (4,31)) encoded as bitmask

Engine-verified mechanics:
- Spikes act as SOLID blocks (block movement and stop fall, no death)
- G blocks consumed on click, flip gravity globally, trigger fall check
- Off-screen clicks work (no bounds checking)
- D blocks removed on click; if directly in gravity direction from player, trigger fall
"""
import sys, os, time
from collections import deque
sys.stdout.reconfigure(line_buffering=True)

# Grid data from L3 survey
WALLS = set()
SPIKES = set()
GEM = (4, 27)

# Parse grid from the survey output
grid_rows = {
    0: "..WWWWWWWWWWW",
    1: "..WWWWWWWWWWW",
    2: "..WWWWWWWWWWW",
    3: "..WWWWWWWWWWW",
    4: "..WWWWWWWWWWW",
    5: "..WWWWWWWWWWW",
    6: "..WWWWWWWWWWW",
    7: "..WWWWWGWWWWW",
    8: "..WWWWWWWWWWW",
    9: "..WW.......WW",
    10: "..WW.......WW",
    11: "..WWWWWWWW.WW",
    12: "..WWWWWWW..WW",
    13: "..WWWWWW...WW",
    14: "..WW^^.....WW",  # player start (4,14), spikes at (2,14),(3,14)
    15: "..WW.......WW",
    16: "..WW.......WW",
    17: "..WWDDWWWWWWW",
    18: "..WW.......WW",
    19: "..WW...DDDDWW",
    20: "..WW.......WW",
    21: "..WW....W..WW",
    22: "..WWWWWWW..WW",
    23: "..WWWGWGWDDWW",
    24: "..WWWWWWWDDWW",
    25: "..WW^^.^^..WW",
    26: "..WW.......WW",
    27: "..WW..*....WW",
    28: "..WW.......WW",
    29: "..WW.......WW",
    30: "..WWWWWWWWWWW",
}

# Build static grid
for y, row in grid_rows.items():
    for i, ch in enumerate(row):
        x = i - 2  # offset: index 0 = x=-2
        if ch == 'W':
            WALLS.add((x, y))
        elif ch == '^':
            SPIKES.add((x, y))

# D blocks with indices
D_BLOCKS = [
    (2, 17), (3, 17),  # y=17
    (5, 19), (6, 19), (7, 19), (8, 19),  # y=19
    (7, 23), (8, 23),  # y=23
    (7, 24), (8, 24),  # y=24
]
D_IDX = {pos: i for i, pos in enumerate(D_BLOCKS)}

# G blocks with indices (all clickable, including walled-off (4,31))
G_BLOCKS = [(5, 7), (3, 23), (5, 23), (4, 31)]
G_IDX = {pos: i for i, pos in enumerate(G_BLOCKS)}

START = (4, 14)
START_GRAV = True  # UP


def is_solid(x, y, d_alive, g_alive):
    """Check if cell is solid (blocks movement and falling).
    Walls, alive D blocks, alive G blocks, and SPIKES are all solid."""
    if (x, y) in WALLS:
        return True
    if (x, y) in SPIKES:
        return True  # spikes act as solid walls
    if (x, y) in D_IDX and (d_alive >> D_IDX[(x, y)]) & 1:
        return True
    if (x, y) in G_IDX and (g_alive >> G_IDX[(x, y)]) & 1:
        return True
    return False


def is_passable(x, y, d_alive, g_alive):
    """Check if cell can be walked into (empty, gem)."""
    if is_solid(x, y, d_alive, g_alive):
        return False
    return True  # empty or gem


def fall(x, y, grav_up, d_alive, g_alive):
    """Simulate fall from position. Returns (final_x, final_y, won).
    No death from spikes (they're solid and stop the fall)."""
    dy = -1 if grav_up else 1
    last_safe = (x, y)
    cx, cy = x, y + dy

    while True:
        if (cx, cy) == GEM:
            return cx, cy, True  # WIN
        if is_solid(cx, cy, d_alive, g_alive):
            return last_safe[0], last_safe[1], False  # stop
        if cy < -5 or cy > 35:
            return last_safe[0], last_safe[1], False  # OOB

        # Cell is passable (empty)
        last_safe = (cx, cy)
        cy += dy


def simulate_move(px, py, dx, grav_up, d_alive, g_alive):
    """Simulate LEFT/RIGHT move. Returns (new_x, new_y, won)."""
    nx, ny = px + dx, py
    if (nx, ny) == GEM:
        return nx, ny, True
    if not is_passable(nx, ny, d_alive, g_alive):
        return px, py, False  # blocked
    # Entered passable cell, check fall
    fx, fy, won = fall(nx, ny, grav_up, d_alive, g_alive)
    return fx, fy, won


def simulate_click_d(px, py, dpos, grav_up, d_alive, g_alive):
    """Simulate clicking a D block to destroy it.
    Returns (new_px, new_py, new_d_alive, won).
    """
    idx = D_IDX.get(dpos)
    if idx is None or not ((d_alive >> idx) & 1):
        return px, py, d_alive, False

    new_d = d_alive & ~(1 << idx)

    # Check if destroyed block is directly in gravity direction from player
    grav_dy = -1 if grav_up else 1
    cell_in_grav_dir = (px, py + grav_dy)
    if dpos == cell_in_grav_dir:
        # Directly in gravity direction → player falls
        fx, fy, won = fall(px, py, grav_up, new_d, g_alive)
        return fx, fy, new_d, won
    return px, py, new_d, False


def simulate_click_g(px, py, gpos, grav_up, d_alive, g_alive):
    """Simulate clicking a G block. Flips gravity, removes G, triggers fall check.
    Returns (new_px, new_py, new_grav, new_g_alive, won).
    """
    idx = G_IDX.get(gpos)
    if idx is None or not ((g_alive >> idx) & 1):
        return px, py, grav_up, g_alive, False

    new_grav = not grav_up
    new_g = g_alive & ~(1 << idx)

    # Fall check: check cell in NEW gravity direction
    grav_dy = -1 if new_grav else 1
    check_pos = (px, py + grav_dy)

    # If check_pos is solid → no fall
    if is_solid(check_pos[0], check_pos[1], d_alive, new_g):
        return px, py, new_grav, new_g, False

    # Otherwise: full fall
    fx, fy, won = fall(px, py, new_grav, d_alive, new_g)
    return fx, fy, new_grav, new_g, won


# BFS
def solve():
    initial_d = (1 << len(D_BLOCKS)) - 1
    initial_g = (1 << len(G_BLOCKS)) - 1
    start_state = (START[0], START[1], START_GRAV, initial_d, initial_g)

    queue = deque()
    queue.append((start_state, []))
    visited = {start_state}

    t0 = time.time()
    n_visited = 0

    while queue:
        state, path = queue.popleft()
        px, py, grav_up, d_alive, g_alive = state

        n_visited += 1
        if n_visited % 100000 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_visited}] visited, queue={len(queue)}, depth={len(path)}, {elapsed:.1f}s")

        if len(path) >= 35:
            continue

        # 1. LEFT
        nx, ny, won = simulate_move(px, py, -1, grav_up, d_alive, g_alive)
        if won:
            print(f"  WIN at depth {len(path)+1}: {path + ['L']}")
            return path + ['L']
        ns = (nx, ny, grav_up, d_alive, g_alive)
        if ns not in visited:
            visited.add(ns)
            queue.append((ns, path + ['L']))

        # 2. RIGHT
        nx, ny, won = simulate_move(px, py, 1, grav_up, d_alive, g_alive)
        if won:
            print(f"  WIN at depth {len(path)+1}: {path + ['R']}")
            return path + ['R']
        ns = (nx, ny, grav_up, d_alive, g_alive)
        if ns not in visited:
            visited.add(ns)
            queue.append((ns, path + ['R']))

        # 3. Click each alive D block
        for dpos in D_BLOCKS:
            if not ((d_alive >> D_IDX[dpos]) & 1):
                continue
            nx, ny, new_d, won = simulate_click_d(px, py, dpos, grav_up, d_alive, g_alive)
            if won:
                print(f"  WIN at depth {len(path)+1}: {path + [('D', dpos)]}")
                return path + [('D', dpos)]
            ns = (nx, ny, grav_up, new_d, g_alive)
            if ns not in visited:
                visited.add(ns)
                queue.append((ns, path + [('D', dpos)]))

        # 4. Click each alive G block
        for gpos in G_BLOCKS:
            if not ((g_alive >> G_IDX[gpos]) & 1):
                continue
            nx, ny, new_grav, new_g, won = simulate_click_g(px, py, gpos, grav_up, d_alive, g_alive)
            if won:
                print(f"  WIN at depth {len(path)+1}: {path + [('G', gpos)]}")
                return path + [('G', gpos)]
            ns = (nx, ny, new_grav, d_alive, new_g)
            if ns not in visited:
                visited.add(ns)
                queue.append((ns, path + [('G', gpos)]))

    elapsed = time.time() - t0
    print(f"  No solution found! Visited {n_visited} states in {elapsed:.1f}s")
    return None


print("BP35 L3 BFS Solver (spikes=solid, 4 G blocks)")
print(f"D blocks: {len(D_BLOCKS)}, G blocks: {len(G_BLOCKS)}")
print()

result = solve()
if result:
    print(f"\nSolution ({len(result)} steps): {result}")
