#!/usr/bin/env python3
"""
Build L7 solution trace by interactive probing.

Confirmed so far:
- Push L,L,U,U,R,R,R moves block from (5,5) to (6,3)
- Red@(6,1) can then jump DOWN to (6,3) [landing = wall+block = 2 objs = valid]
- Red is now on block at (6,3)

Next: push red on block through wall corridors to bottom row, then leapfrog.
"""
import os, sys, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT'}
DIR_ACTIONS = {(0, -1): GameAction.ACTION1, (0, 1): GameAction.ACTION2,
               (-1, 0): GameAction.ACTION3, (1, 0): GameAction.ACTION4}

def get_state(eq, rows=10, cols=25):
    grid = eq.hncnfaqaddg
    pieces = {}
    blocks = set()
    walls = set()
    pegs = set()
    walkable = set()
    for y in range(rows):
        for x in range(cols):
            objs = grid.ijpoqzvnjt(x, y)
            names = [o.name for o in objs]
            for n in names:
                if n == 'fozwvlovdui': pieces[(x,y)] = 'N'
                elif n == 'fozwvlovdui_red': pieces[(x,y)] = 'R'
                elif n == 'fozwvlovdui_blue': pieces[(x,y)] = 'B'
                elif n == 'hupkpseyuim2': blocks.add((x,y))
                elif 'kraubslpehi' in n: walls.add((x,y))
                elif n == 'dgxfozncuiz': pegs.add((x,y))
                elif n == 'hupkpseyuim': walkable.add((x,y))
    return {'pieces': pieces, 'blocks': blocks, 'walls': walls, 'pegs': pegs, 'walkable': walkable}

def cell_contents(eq, x, y):
    objs = eq.hncnfaqaddg.ijpoqzvnjt(x, y)
    return [o.name for o in objs]

def do_push(env, direction_name):
    d_map = {'L': GameAction.ACTION3, 'R': GameAction.ACTION4,
             'U': GameAction.ACTION1, 'D': GameAction.ACTION2}
    return env.step(d_map[direction_name])

def do_jump(env, eq, src, dst):
    """Execute a jump: click source piece, then click arrow direction."""
    grid = eq.hncnfaqaddg
    off = grid.cdpcbbnfdp
    sx, sy = src
    dx, dy = dst

    # Click source
    px = sx * 6 + off[0] + 3
    py = sy * 6 + off[1] + 3
    fd = env.step(GameAction.ACTION6, data={'x': px, 'y': py})

    # Click arrow
    off = grid.cdpcbbnfdp
    half_dx = (dx - sx) // 2
    half_dy = (dy - sy) // 2
    ax = sx * 6 + off[0] + half_dx * 12 + 3
    ay = sy * 6 + off[1] + half_dy * 12 + 3
    fd = env.step(GameAction.ACTION6, data={'x': ax, 'y': ay})

    # Check completion button
    if eq.zvcnglshzcx:
        fd = env.step(GameAction.ACTION6, data={'x': 8, 'y': 56})

    return fd

def check_all_jumps(eq, rows=10, cols=25):
    """Check all valid jumps available."""
    valid = []
    for y in range(rows):
        for x in range(cols):
            for d in DIRS:
                if eq.qikmikecdf((x,y), d):
                    dx, dy = d
                    lx, ly = x+2*dx, y+2*dy
                    valid.append(((x,y), (lx,ly), DIR_NAMES[d]))
    return valid

# Setup
arc = Arcade()
env = arc.make('lf52')
fd = env.reset()
game = env._game
game.set_level(6)  # L7
eq = game.ikhhdzfmarl

print("=== L7 Solution Trace Building ===")
print()

# Phase 1: Push blocks to create landing for red
print("Phase 1: Push blocks (L,L,U,U,R,R,R)")
phase1_pushes = ['L', 'L', 'U', 'U', 'R', 'R', 'R']
for p in phase1_pushes:
    do_push(env, p)

s = get_state(eq)
print(f"  Pieces: {s['pieces']}")
print(f"  Blocks: {sorted(s['blocks'])}")

# Phase 2: Red jumps onto block at (6,3)
print("\nPhase 2: Red jump (6,1) -> (6,3)")
valid_jumps = check_all_jumps(eq)
print(f"  Valid jumps: {valid_jumps}")

do_jump(env, eq, (6,1), (6,3))
s = get_state(eq)
print(f"  After jump: pieces={s['pieces']}, blocks={sorted(s['blocks'])}")
# Red should now be at (6,3) on the block

# Phase 3: Push red through wall corridors to bottom
# From investigation: (6,3) --L--> (5,3) --L--> (4,3) --L--> (3,3) --D--> (3,4) --D--> (3,5) --L--> (2,5) --L--> (1,5) --D--> (1,6)
print("\nPhase 3: Push red on block through corridors")
phase3_pushes = ['L', 'L', 'L', 'D', 'D', 'L', 'L', 'D']
for p in phase3_pushes:
    do_push(env, p)
    s = get_state(eq)
    print(f"  After {p}: R@{[pos for pos, n in s['pieces'].items() if n == 'R']}, blocks={sorted(s['blocks'])}")

# Check where everything is now
s = get_state(eq)
print(f"\nAfter Phase 3:")
print(f"  Pieces: {s['pieces']}")
print(f"  Blocks: {sorted(s['blocks'])}")

# Check valid jumps now
valid_jumps = check_all_jumps(eq)
print(f"  Valid jumps: {valid_jumps}")

# Phase 4: Red jumps off block to bottom row
# From (1,6) or wherever, red should jump DOWN through peg to landing
# Let's check cells around red's position
r_pos = [pos for pos, n in s['pieces'].items() if n == 'R']
if r_pos:
    rx, ry = r_pos[0]
    print(f"\nRed is at ({rx},{ry})")
    for d in DIRS:
        dx, dy = d
        mx, my = rx+dx, ry+dy
        lx, ly = rx+2*dx, ry+2*dy
        mid = cell_contents(eq, mx, my)
        land = cell_contents(eq, lx, ly)
        valid = eq.qikmikecdf((rx,ry), d)
        print(f"  {DIR_NAMES[d]}: mid({mx},{my})={mid}, land({lx},{ly})={land}, valid={valid}")

# Let's also check what N can do
print("\nLeft-N position and jumps:")
n_pos = [pos for pos, n in s['pieces'].items() if n == 'N']
for pos in n_pos:
    print(f"  N@{pos}")
    for d in DIRS:
        dx, dy = d
        mx, my = pos[0]+dx, pos[1]+dy
        lx, ly = pos[0]+2*dx, pos[1]+2*dy
        valid = eq.qikmikecdf(pos, d)
        if valid:
            mid = cell_contents(eq, mx, my)
            land = cell_contents(eq, lx, ly)
            print(f"    {DIR_NAMES[d]}: mid({mx},{my})={mid}, land({lx},{ly})={land}")

# Check if there's a peg at (1,7) for red to jump to (1,8)
print("\nChecking (1,7) and (1,8):")
print(f"  (1,7): {cell_contents(eq, 1, 7)}")
print(f"  (1,8): {cell_contents(eq, 1, 8)}")

print(f"\nStep count: {eq.asqvqzpfdi}")
print(f"Step limit: {eq.whtqurkphir}")

# Check walkable cells in bottom rows
print("\nBottom row (y=8) contents:")
for x in range(25):
    c = cell_contents(eq, x, 8)
    if c:
        print(f"  ({x},8): {c}")

print("\nRow y=7 contents:")
for x in range(25):
    c = cell_contents(eq, x, 7)
    if c:
        print(f"  ({x},7): {c}")

print("\nRow y=6 contents:")
for x in range(13):
    c = cell_contents(eq, x, 6)
    if c:
        print(f"  ({x},6): {c}")

print("\nDone.")
