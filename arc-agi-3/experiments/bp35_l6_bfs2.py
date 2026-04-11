#!/usr/bin/env python3
"""BP35 L6 BFS v2 - reduced block set + smart toggle pruning.

Key analysis:
- x=8 going DOWN is lethal (v-spike at (8,24))
- x=9 going UP is lethal (^-spike at (9,4))
- x=8 going UP is safe (wall at (8,14) stops fall at (8,15))
- x=9 going DOWN is safe (wall at (9,27) stops fall at (9,26))
- Solution must reach x=9 with grav DOWN to safely transit to lower section
- Then walk left to (7,26), flip UP to (7,23), walk to (3,23), flip DOWN to gem

Path likely goes through x=7 column (y=4-7) with careful block toggling.

Smart pruning: only allow block toggles that affect player's current column,
adjacent columns, or the fall path.
"""
import sys, time
from collections import deque
sys.stdout.reconfigure(line_buffering=True)

WALLS = set()
UP_SPIKES = set()
DN_SPIKES = set()
ALL_SPIKES = set()
GEM = (3, 25)

grid_rows = {
    0:  "..WWWWWWWWWWW....",
    1:  "..WWWWWWWWWWW....",
    2:  "..WWWWWWWWWWW....",
    3:  "..WWWWWWWWWWW....",
    4:  "..WWWWWWWOW^W....",
    5:  "..WWWWWWWOW.W....",
    6:  "..WWWWWWWO..W....",
    7:  "..WWWWWWWO..W....",
    8:  "..WW.O^.^OW.W....",
    9:  "..WW.OO.BOW.W....",
    10: "..WW.O.B.OW.W....",
    11: "..WW.O.v.vW.W....",
    12: "..WW.WWWWWW.W....",
    13: "..WW.WOOO.W.W....",
    14: "..WW.WOOO.W.W....",
    15: "..WW..OOO...W....",
    16: "..WW..OOO...W....",
    17: "..WW.vOOOv..W....",
    18: "..WWWWWWWW..W....",
    19: "..WW....OW..W....",
    20: "..WW........W....",
    21: "..WW....OW..W....",
    22: "..WWWWWWWW..W....",
    23: "..WW........W....",
    24: "..WW......v.W....",
    25: "..WW.*.WW.W.W....",
    26: "..WW...WW...W....",
    27: "..WW...WWWWWW....",
    28: "..WWWWWWWWWWW....",
}

for y, row in grid_rows.items():
    for i, ch in enumerate(row):
        x = i - 2
        if ch == 'W':
            WALLS.add((x, y))
        elif ch == '^':
            UP_SPIKES.add((x, y))
            ALL_SPIKES.add((x, y))
        elif ch == 'v':
            DN_SPIKES.add((x, y))
            ALL_SPIKES.add((x, y))

# ALL OB blocks in the level
ALL_OB_BLOCKS = [
    (7, 4), (7, 5), (7, 6), (7, 7),
    (3, 8), (7, 8),
    (3, 9), (4, 9), (6, 9), (7, 9),
    (3, 10), (5, 10), (7, 10),
    (3, 11),
    (4, 13), (5, 13), (6, 13),
    (4, 14), (5, 14), (6, 14),
    (4, 15), (5, 15), (6, 15),
    (4, 16), (5, 16), (6, 16),
    (4, 17), (5, 17), (6, 17),
    (6, 19),
    (6, 21),
]

# Only include blocks that could plausibly matter
# Upper area (y=4-11) + corridor (y=19,21) + platform connection (y=13-15)
OB_BLOCKS = [
    (7, 4), (7, 5), (7, 6), (7, 7),  # x=7 column
    (3, 8), (7, 8),                    # y=8 row
    (3, 9), (4, 9), (6, 9), (7, 9),   # y=9 row
    (3, 10), (5, 10), (7, 10),         # y=10 row
    (3, 11),                            # y=11
    (4, 13), (5, 13), (6, 13),         # y=13 platform (walking path)
    (4, 14), (5, 14), (6, 14),         # y=14 platform (possible floors)
    (4, 15), (5, 15), (6, 15),         # y=15 platform (x=2 shaft entry)
    (6, 19),                            # corridor
    (6, 21),                            # corridor
]
OB_IDX = {pos: i for i, pos in enumerate(OB_BLOCKS)}
N_OB = len(OB_BLOCKS)
OB_MASKS = {pos: 1 << OB_IDX[pos] for pos in OB_BLOCKS}

# Initial state: (6,9)=B, (5,10)=B, all others O
INITIAL_OB = 0
if (6, 9) in OB_IDX:
    INITIAL_OB |= OB_MASKS[(6, 9)]
if (5, 10) in OB_IDX:
    INITIAL_OB |= OB_MASKS[(5, 10)]

START = (3, 19)
START_GRAV = True

STATIC_SOLID = WALLS | ALL_SPIKES

# Also include OB blocks NOT in our set as permanent transparent (O) blocks
# They don't affect solidity since they're always O (transparent)
EXCLUDED_OB = set(ALL_OB_BLOCKS) - set(OB_BLOCKS)


def is_solid(x, y, ob_state):
    if (x, y) in STATIC_SOLID:
        return True
    if (x, y) in OB_IDX:
        return bool(ob_state & OB_MASKS[(x, y)])
    # Excluded OB blocks are always O (transparent) = not solid
    return False


def is_dead(x, y, grav_up):
    if grav_up:
        return (x, y - 1) in UP_SPIKES
    else:
        return (x, y + 1) in DN_SPIKES


def fall(x, y, grav_up, ob_state):
    dy = -1 if grav_up else 1
    lx, ly = x, y
    cx, cy = x, y + dy
    while True:
        if (cx, cy) == GEM:
            return cx, cy, True, False
        if is_solid(cx, cy, ob_state):
            return lx, ly, False, is_dead(lx, ly, grav_up)
        if cy < -5 or cy > 35:
            return lx, ly, False, False
        lx, ly = cx, cy
        cy += dy


def pack_state(px, py, grav_up, ob):
    # px range: -5 to 15 (needs 5 bits)
    # py range: -5 to 35 (needs 6 bits)
    return (px + 5) | ((py + 5) << 5) | ((1 if grav_up else 0) << 11) | (ob << 12)


def unpack_state(s):
    px = (s & 0x1F) - 5
    py = ((s >> 5) & 0x3F) - 5
    grav_up = bool((s >> 11) & 1)
    ob = s >> 12
    return px, py, grav_up, ob


def get_fall_column(px, py, grav_up, ob_state):
    """Return set of (x,y) cells in the fall path from this position."""
    cells = set()
    dy = -1 if grav_up else 1
    cx, cy = px, py + dy
    while True:
        if (cx, cy) in STATIC_SOLID or (cx, cy) in OB_IDX:
            break
        if cy < -5 or cy > 35:
            break
        cells.add((cx, cy))
        cy += dy
    return cells


def solve(max_depth=25):
    s0 = pack_state(START[0], START[1], START_GRAV, INITIAL_OB)

    visited = {s0}
    parent = {s0: None}
    queue = deque([s0])

    t0 = time.time()
    n = 0
    n_dead = 0
    depth_map = {s0: 0}

    while queue:
        state = queue.popleft()
        px, py, grav_up, ob = unpack_state(state)
        depth = depth_map[state]

        n += 1
        if n % 500000 == 0:
            elapsed = time.time() - t0
            mem_mb = len(visited) * 60 // (1024*1024)
            print(f"  [{n}] vis={len(visited)}, q={len(queue)}, d={depth}, dead={n_dead}, ~{mem_mb}MB, {elapsed:.1f}s")

        if depth >= max_depth:
            continue

        next_depth = depth + 1
        successors = []

        # LEFT/RIGHT
        for dx, label in [(-1, 'L'), (1, 'R')]:
            nx, ny = px + dx, py
            if (nx, ny) == GEM:
                path = [label]
                s = state
                while parent[s] is not None:
                    ps, act = parent[s]
                    path.append(act)
                    s = ps
                path.reverse()
                print(f"  WIN depth {len(path)} ({n} vis, {time.time()-t0:.1f}s)")
                return path
            if is_solid(nx, ny, ob):
                continue
            fx, fy, won, dead = fall(nx, ny, grav_up, ob)
            if won:
                path = [label]
                s = state
                while parent[s] is not None:
                    ps, act = parent[s]
                    path.append(act)
                    s = ps
                path.reverse()
                print(f"  WIN depth {len(path)} ({n} vis, {time.time()-t0:.1f}s)")
                return path
            if dead:
                n_dead += 1
                continue
            ns = pack_state(fx, fy, grav_up, ob)
            if ns not in visited:
                successors.append((ns, label))

        # Flip gravity
        new_grav = not grav_up
        grav_dy = -1 if new_grav else 1
        if is_solid(px, py + grav_dy, ob):
            if not is_dead(px, py, new_grav):
                ns = pack_state(px, py, new_grav, ob)
                if ns not in visited:
                    successors.append((ns, 'G'))
            else:
                n_dead += 1
        else:
            fx, fy, won, dead = fall(px, py, new_grav, ob)
            if won:
                path = ['G']
                s = state
                while parent[s] is not None:
                    ps, act = parent[s]
                    path.append(act)
                    s = ps
                path.reverse()
                print(f"  WIN depth {len(path)} ({n} vis, {time.time()-t0:.1f}s)")
                return path
            if not dead:
                ns = pack_state(fx, fy, new_grav, ob)
                if ns not in visited:
                    successors.append((ns, 'G'))
            else:
                n_dead += 1

        # Toggle O/B blocks — allow all blocks (screen-visible in game)
        for obpos in OB_BLOCKS:
            mask = OB_MASKS[obpos]
            new_ob = ob ^ mask
            was_solid = bool(ob & mask)
            npx, npy = px, py
            won = dead = False
            if was_solid:
                grav_dy2 = -1 if grav_up else 1
                if obpos == (px, py + grav_dy2):
                    npx, npy, won, dead = fall(px, py, grav_up, new_ob)
            else:
                # Toggling O→B: if the block is at player's position, player gets pushed?
                # Actually in the game, if player is at an O block position and it becomes B,
                # the player would be inside a solid block. Skip this case.
                if obpos == (px, py):
                    continue
            if won:
                path = [('T', obpos)]
                s = state
                while parent[s] is not None:
                    ps, act = parent[s]
                    path.append(act)
                    s = ps
                path.reverse()
                print(f"  WIN depth {len(path)} ({n} vis, {time.time()-t0:.1f}s)")
                return path
            if dead:
                n_dead += 1
                continue
            ns = pack_state(npx, npy, grav_up, new_ob)
            if ns not in visited:
                successors.append((ns, ('T', obpos)))

        for ns, act in successors:
            visited.add(ns)
            parent[ns] = (state, act)
            depth_map[ns] = next_depth
            queue.append(ns)

    elapsed = time.time() - t0
    print(f"  No solution! {n} visited, {n_dead} dead, {elapsed:.1f}s")
    return None


print(f"BP35 L6 BFS v2 ({N_OB} blocks, smart toggle pruning)")
print(f"Blocks: {OB_BLOCKS}")
print(f"Initial B: {[(p,i) for p,i in OB_IDX.items() if INITIAL_OB & OB_MASKS[p]]}")
print(f"^ spikes: {sorted(UP_SPIKES)}")
print(f"v spikes: {sorted(DN_SPIKES)}")
print()

result = solve(max_depth=25)
if result:
    print(f"\nSolution ({len(result)} steps): {result}")
else:
    print("\nNo solution found. May need more blocks or higher depth.")
