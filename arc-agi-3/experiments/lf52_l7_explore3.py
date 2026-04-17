#!/usr/bin/env python3
"""
L7 BFS using engine as oracle. Start from Phase 4 state (red at 5,8, N at 0,1, N at 22,6).
Goal: reduce movable count from 3 to 2 (N jumps over N).
Use qikmikecdf as validity oracle, engine state as ground truth.
"""
import os, sys, time, json
from collections import deque
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction
import copy

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT'}
DIR_ACTIONS = {'U': GameAction.ACTION1, 'D': GameAction.ACTION2,
               'L': GameAction.ACTION3, 'R': GameAction.ACTION4}

def get_state(eq, rows=10, cols=25):
    grid = eq.hncnfaqaddg
    pieces = {}
    blocks = set()
    pegs = set()
    walls = set()
    walkable = set()
    for y in range(rows):
        for x in range(cols):
            objs = grid.ijpoqzvnjt(x, y)
            for o in objs:
                if o.name == 'fozwvlovdui': pieces[(x,y)] = 'N'
                elif o.name == 'fozwvlovdui_red': pieces[(x,y)] = 'R'
                elif o.name == 'hupkpseyuim2': blocks.add((x,y))
                elif 'kraubslpehi' in o.name: walls.add((x,y))
                elif o.name == 'dgxfozncuiz': pegs.add((x,y))
                elif o.name == 'hupkpseyuim': walkable.add((x,y))
    return pieces, frozenset(blocks), frozenset(pegs), frozenset(walls), frozenset(walkable)

def state_key(pieces, blocks):
    return (frozenset(pieces.items()), blocks)

def cell_contents(eq, x, y):
    return [o.name for o in eq.hncnfaqaddg.ijpoqzvnjt(x, y)]

def do_push_sim(env, d):
    return env.step(DIR_ACTIONS[d])

def do_jump_sim(env, eq, src, dst):
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

def check_all_jumps(eq, rows=10, cols=25):
    valid = []
    for y in range(rows):
        for x in range(cols):
            for d in DIRS:
                if eq.qikmikecdf((x,y), d):
                    dx, dy = d
                    lx, ly = x+2*dx, y+2*dy
                    valid.append(((x,y), (lx,ly)))
    return valid

# Setup
arc = Arcade()
env = arc.make('lf52')
env.reset()
game = env._game
game.set_level(6)
eq = game.ikhhdzfmarl

# Replicate through Phase 4
for p in ['L', 'L', 'U', 'U', 'R', 'R', 'R']:
    do_push_sim(env, p)
do_jump_sim(env, eq, (6,1), (6,3))
for p in ['L', 'L', 'L', 'D', 'D', 'L', 'L', 'D']:
    do_push_sim(env, p)
do_jump_sim(env, eq, (1,6), (1,8))
do_jump_sim(env, eq, (1,8), (3,8))
do_jump_sim(env, eq, (3,8), (5,8))

pieces, blocks, pegs, walls, walkable = get_state(eq)
print(f"Starting BFS from: pieces={pieces}, blocks={sorted(blocks)}")
print(f"Steps used so far: {eq.asqvqzpfdi}")

# Now we need a model-based BFS using the simulator from lf52_solve_final.
# But the prior search exhausted at 2.2M states. The trick is: we now have red
# at (5,8) which changes the landscape entirely.
# Let me use the model-based solver with the current state.
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
import lf52_solve_final as solver

state_dict = solver.extract_state(eq)
ps = solver.make_puzzle_state(state_dict)

print(f"\nMovable count: {ps.movable_count()}")
print(f"Target: 2")
print(f"Pieces: {ps.piece_dict()}")
print(f"Blocks: {sorted(ps.blocks)}")

# The unified A* should be able to handle this now since the state is different
# Red is at (5,8) instead of (6,1)
print(f"\nRunning unified A* from current state (180s limit)...")
result = solver.solve_unified(ps, 2, time_limit=180, max_depth=240)
if result:
    print(f"FOUND! {len(result)} actions")
    for a in result:
        print(f"  {a}")
else:
    print("Unified failed.")
    print("Trying integrated solver (180s)...")
    result = solver.solve_integrated(ps, 2, max_steps=240, time_limit=180)
    if result:
        print(f"Integrated FOUND! {len(result)} actions")
        for a in result:
            print(f"  {a}")
    else:
        print("Integrated also failed.")
