#!/usr/bin/env python3
"""Continue exploration: need to get left-N to bottom row.
Key problem: N@(0,1) needs block at (0,3) so it can jump DOWN.
But (0,3) is kraubslpehi-L (a wall). Can we push a block there?

Blocks move to adjacent wall cells. So if there's a block at (1,3) and we push LEFT,
it would go to (0,3) IF (0,3) has a wall. It does! But (1,3) is also a wall...

Actually: blocks at (1,6) with wall. Push UP repeatedly?
Let's explore systematically."""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT'}

def cell_contents(eq, x, y):
    objs = eq.hncnfaqaddg.ijpoqzvnjt(x, y)
    return [o.name for o in objs]

def get_pieces(eq, rows=10, cols=25):
    grid = eq.hncnfaqaddg
    pieces = {}
    blocks = set()
    for y in range(rows):
        for x in range(cols):
            objs = grid.ijpoqzvnjt(x, y)
            for o in objs:
                if o.name == 'fozwvlovdui': pieces[(x,y)] = 'N'
                elif o.name == 'fozwvlovdui_red': pieces[(x,y)] = 'R'
                elif o.name == 'hupkpseyuim2': blocks.add((x,y))
    return pieces, blocks

def check_all_jumps(eq, rows=10, cols=25):
    valid = []
    for y in range(rows):
        for x in range(cols):
            for d in DIRS:
                if eq.qikmikecdf((x,y), d):
                    dx, dy = d
                    lx, ly = x+2*dx, y+2*dy
                    valid.append(((x,y), (lx,ly), DIR_NAMES[d]))
    return valid

def do_push(env, d):
    d_map = {'L': GameAction.ACTION3, 'R': GameAction.ACTION4,
             'U': GameAction.ACTION1, 'D': GameAction.ACTION2}
    return env.step(d_map[d])

def do_jump(env, eq, src, dst):
    grid = eq.hncnfaqaddg
    off = grid.cdpcbbnfdp
    sx, sy = src
    dx, dy = dst
    px = sx * 6 + off[0] + 3
    py = sy * 6 + off[1] + 3
    env.step(GameAction.ACTION6, data={'x': px, 'y': py})
    off = grid.cdpcbbnfdp
    half_dx = (dx - sx) // 2
    half_dy = (dy - sy) // 2
    ax = sx * 6 + off[0] + half_dx * 12 + 3
    ay = sy * 6 + off[1] + half_dy * 12 + 3
    fd = env.step(GameAction.ACTION6, data={'x': ax, 'y': ay})
    if eq.zvcnglshzcx:
        fd = env.step(GameAction.ACTION6, data={'x': 8, 'y': 56})
    return fd

# Setup
arc = Arcade()
env = arc.make('lf52')
env.reset()
game = env._game
game.set_level(6)
eq = game.ikhhdzfmarl

# Replicate Phase 1-4
for p in ['L', 'L', 'U', 'U', 'R', 'R', 'R']:
    do_push(env, p)
do_jump(env, eq, (6,1), (6,3))  # Phase 2
for p in ['L', 'L', 'L', 'D', 'D', 'L', 'L', 'D']:
    do_push(env, p)
do_jump(env, eq, (1,6), (1,8))  # Phase 4
do_jump(env, eq, (1,8), (3,8))  # Red to (3,8)
do_jump(env, eq, (3,8), (5,8))  # Red to (5,8)

pieces, blocks = get_pieces(eq)
print(f"State: pieces={pieces}")
print(f"Blocks: {sorted(blocks)}")
print(f"Steps: {eq.asqvqzpfdi}")

# The block at (1,6) has a wall (kraubslpehi-up).
# Push UP from (1,6): target (1,5) which is kraubslpehi-< (a wall). So block should move UP.
print(f"\n=== Trying to push block from (1,6) upward ===")
print(f"  (1,5): {cell_contents(eq, 1, 5)} -- is wall? kraubslpehi in name?")
print(f"  (1,4): {cell_contents(eq, 1, 4)}")
print(f"  (1,3): {cell_contents(eq, 1, 3)}")

# Push UP
do_push(env, 'U')
pieces, blocks = get_pieces(eq)
print(f"After U: blocks={sorted(blocks)}")
print(f"  (1,6): {cell_contents(eq, 1, 6)}")
print(f"  (1,5): {cell_contents(eq, 1, 5)}")

# Push UP again
do_push(env, 'U')
pieces, blocks = get_pieces(eq)
print(f"After UU: blocks={sorted(blocks)}")

# Push LEFT to move block to column 0
do_push(env, 'L')
pieces, blocks = get_pieces(eq)
print(f"After UUL: blocks={sorted(blocks)}")
print(f"  (0,3): {cell_contents(eq, 0, 3)}")
print(f"  (0,4): {cell_contents(eq, 0, 4)}")

# Check if N@(0,1) can now jump DOWN
print(f"\nValid jumps involving (0,1):")
for d in DIRS:
    valid = eq.qikmikecdf((0,1), d)
    if valid:
        dx, dy = d
        lx, ly = 0+2*dx, 1+2*dy
        print(f"  N@(0,1) {DIR_NAMES[d]} -> ({lx},{ly})")

pieces, blocks = get_pieces(eq)
print(f"\nFull state: pieces={pieces}")
print(f"Blocks: {sorted(blocks)}")
print(f"Steps: {eq.asqvqzpfdi}")

# Maybe more pushes needed. Let me understand where blocks ended up.
# Let me try different push sequences from the state after Phase 4

# Full grid dump for rows 0-9, cols 0-12
print("\n=== Full grid layout ===")
for y in range(10):
    row = []
    for x in range(13):
        c = cell_contents(eq, x, y)
        if not c:
            row.append(f"({x},{y}):_")
        else:
            short = []
            for n in c:
                if 'kraubslpehi' in n: short.append('W')
                elif n == 'hupkpseyuim': short.append('.')
                elif n == 'hupkpseyuim2': short.append('B')
                elif n == 'dgxfozncuiz': short.append('P')
                elif n == 'fozwvlovdui_red': short.append('R')
                elif n == 'fozwvlovdui': short.append('N')
                else: short.append('?')
            row.append(f"({x},{y}):{''.join(short)}")
    print("  ".join(row))
