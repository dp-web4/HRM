#!/usr/bin/env python3
"""Explore L7 from Phase 4 onwards. Red at (1,6) can jump DOWN to (1,8).
Need to figure out the full leapfrog sequence to get left-N to right-N."""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT'}
DIR_ACTIONS = {(0, -1): GameAction.ACTION1, (0, 1): GameAction.ACTION2,
               (-1, 0): GameAction.ACTION3, (1, 0): GameAction.ACTION4}

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

# Setup - get to L7 using set_level
arc = Arcade()
env = arc.make('lf52')
env.reset()
game = env._game
game.set_level(6)
eq = game.ikhhdzfmarl

# Phase 1: Push blocks
for p in ['L', 'L', 'U', 'U', 'R', 'R', 'R']:
    do_push(env, p)

# Phase 2: Red jump to (6,3)
do_jump(env, eq, (6,1), (6,3))

# Phase 3: Push red through corridors
for p in ['L', 'L', 'L', 'D', 'D', 'L', 'L', 'D']:
    do_push(env, p)

# Phase 4: Red jumps from (1,6) to (1,8)
pieces, blocks = get_pieces(eq)
print(f"Before Phase 4: pieces={pieces}, blocks={sorted(blocks)}")
print(f"Steps used: {eq.asqvqzpfdi}")

do_jump(env, eq, (1,6), (1,8))
pieces, blocks = get_pieces(eq)
print(f"After Phase 4 jump: pieces={pieces}, blocks={sorted(blocks)}")

# Now red at (1,8). Can it jump right?
print(f"\nValid jumps after Phase 4:")
for j in check_all_jumps(eq):
    print(f"  {j}")

# Phase 5: Red jumps right through pegs
# (1,8) -> (3,8) over peg at (2,8)
do_jump(env, eq, (1,8), (3,8))
pieces, blocks = get_pieces(eq)
print(f"\nAfter R@(1,8)->(3,8): pieces={pieces}")

# (3,8) -> (5,8) over peg at (4,8)
do_jump(env, eq, (3,8), (5,8))
pieces, blocks = get_pieces(eq)
print(f"After R@(3,8)->(5,8): pieces={pieces}")

# Now red is at (5,8). No more pegs in row 8 beyond this.
# Need left-N to come down. Can we push blocks to create a path for N?
print(f"\nValid jumps now:")
for j in check_all_jumps(eq):
    print(f"  {j}")

# Let's look at the full grid layout for planning
print(f"\nSteps used: {eq.asqvqzpfdi}")
print(f"\nBlock positions: {sorted(blocks)}")

# Can we push block to create jump path for left-N?
# Left-N is at (0,1). Need to get it to bottom row.
# Need: block at (0,3) so N can jump over peg at (0,2)? Check (0,2)
print(f"\n(0,2): {cell_contents(eq, 0, 2)}")
print(f"(0,3): {cell_contents(eq, 0, 3)}")

# What about getting N down through a different path?
# Check cells around N@(0,1)
for d in DIRS:
    dx, dy = d
    mx, my = 0+dx, 1+dy
    lx, ly = 0+2*dx, 1+2*dy
    mid = cell_contents(eq, mx, my) if mx >= 0 and my >= 0 else []
    land = cell_contents(eq, lx, ly) if lx >= 0 and ly >= 0 else []
    valid = eq.qikmikecdf((0,1), d)
    print(f"  N@(0,1) {DIR_NAMES[d]}: mid({mx},{my})={mid}, land({lx},{ly})={land}, valid={valid}")

# What about pushing to get a block near N?
# We need to understand: where are walls that blocks can be pushed to?
print("\nRow y=0-3 wall/walkable layout for x=0-7:")
for y in range(4):
    for x in range(8):
        c = cell_contents(eq, x, y)
        if c:
            short = []
            for n in c:
                if 'kraubslpehi' in n: short.append('W')
                elif n == 'hupkpseyuim': short.append('.')
                elif n == 'hupkpseyuim2': short.append('B')
                elif n == 'dgxfozncuiz': short.append('P')
                elif 'fozwvlovdui' in n: short.append(n.split('_')[-1][0].upper() if '_' in n else 'N')
                else: short.append('?')
            print(f"  ({x},{y}): {''.join(short)}", end="")
    print()

# Row 4-8
print("\nRow y=4-9 layout for x=0-12:")
for y in range(4, 10):
    for x in range(13):
        c = cell_contents(eq, x, y)
        if c:
            short = []
            for n in c:
                if 'kraubslpehi' in n: short.append('W')
                elif n == 'hupkpseyuim': short.append('.')
                elif n == 'hupkpseyuim2': short.append('B')
                elif n == 'dgxfozncuiz': short.append('P')
                elif 'fozwvlovdui' in n: short.append(n.split('_')[-1][0].upper() if '_' in n else 'N')
                else: short.append('?')
            print(f"  ({x},{y}): {''.join(short)}", end="")
    print()

# Check around right-N@(22,6) more carefully
print("\nAround right-N@(22,6):")
for y in range(4, 10):
    for x in range(19, 25):
        c = cell_contents(eq, x, y)
        if c:
            short = []
            for n in c:
                if 'kraubslpehi' in n: short.append('W')
                elif n == 'hupkpseyuim': short.append('.')
                elif n == 'hupkpseyuim2': short.append('B')
                elif n == 'dgxfozncuiz': short.append('P')
                elif 'fozwvlovdui' in n: short.append(n.split('_')[-1][0].upper() if '_' in n else 'N')
                else: short.append('?')
            print(f"  ({x},{y}): {''.join(short)}", end="")
    print()

# ddaguepwkt count
print(f"\nddaguepwkt (movable count): {eq.ddaguepwkt}")
print(f"Win condition: whtqurkphir={eq.whtqurkphir}, need ddaguepwkt==2")
