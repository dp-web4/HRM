#!/usr/bin/env python3
"""Probe L7 and L8 layouts. Replay solver's L0-L6 solutions, then dump L7 state and simulate greedy to find failure mode."""
import sys, os, time
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")

from arc_agi import Arcade
from arcengine import GameAction
from collections import deque

arcade = Arcade(operation_mode='offline')
env = arcade.make('wa30-ee6fef47')
fd = env.reset()
game = env._game

CELL = 4
UP, DOWN, LEFT, RIGHT, ACT5 = (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5)
AM = {'U': UP, 'D': DOWN, 'L': LEFT, 'R': RIGHT, '5': ACT5}

# Import KNOWN_SOLUTIONS and helper plan_action from solver by exec'ing guarded portion
import runpy
# Simpler: hard-replicate what we need.
import importlib.util

# We'll just advance L0-L6 using solver's known solutions and its solve_no_ai / solve_with_ai logic.
# But exec'ing runs main loop. Do it by subprocess: run solver, stop at L7. Better: copy the known solutions.

KNOWN = {
    2: 'LLDDR5RRRRR5LLLLLLUUUUUUUR5RRRRRR5LLLDR5RRR5' + 'U'*56,
    3: 'UUL5L5DDDRDL5L5RUURRU5U5DDDD5D5UUULU5U5DDDD5DD5' + 'U'*10,
    6: 'U'*15 + 'DDRR5' + 'RU5LLLLLD5' + 'R'*9 + '5' + 'L'*9 + '5',
}

def s2a(s): return [AM[c] for c in s]

# Simple greedy to pass L0,1,4,5 - or import from solver. Let's just load solver's functions.
# Load solver module by modifying __name__ trick... the file runs top-level.
# Workaround: read file, strip the main loop, exec the rest.
src = open('arc-agi-3/experiments/wa30_solve_final.py').read()
# Main loop starts at "total_actions = 0"
cut = src.index('total_actions = 0')
header = src[:cut]
exec(header, globals())

# Now we have: save_frame, solve_no_ai, solve_with_ai, replay_from_solutions, plan_action, KNOWN_SOLUTIONS, string_to_actions, etc.
# Game env was created once above. But header re-creates arcade/env/game. Re-fetch.
# (The exec already did arcade=..., env=..., fd=env.reset(), game=env._game.)

all_solutions = []
for lv in range(7):
    fd = replay_from_solutions(all_solutions)
    print(f"L{lv}: solving...")
    player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    blues = game.current_level.get_sprites_by_tag("kdweefinfi")
    whites = game.current_level.get_sprites_by_tag("ysysltqlke")
    has_ai = len(blues) > 0 or len(whites) > 0
    if lv in KNOWN_SOLUTIONS:
        sol = string_to_actions(KNOWN_SOLUTIONS[lv])
    elif not has_ai:
        sol = solve_no_ai()
    else:
        sol = solve_with_ai(all_solutions)
    assert sol is not None, f"L{lv} failed"
    # Verify
    fd = replay_from_solutions(all_solutions)
    before = fd.levels_completed
    actual = []
    for m in sol:
        fd = env.step(m); actual.append(m)
        if fd.levels_completed > before or fd.state.name == 'WIN': break
    all_solutions.append(actual)
    print(f"  L{lv} done, {len(actual)} moves, completed={fd.levels_completed}")

# Now at L7
fd = replay_from_solutions(all_solutions)
print("\n=== L7 STATE ===")
player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
blues = game.current_level.get_sprites_by_tag("kdweefinfi")
whites = game.current_level.get_sprites_by_tag("ysysltqlke")
print(f"Player: ({player.x},{player.y}) rot={player.rotation}")
print(f"Pieces ({len(goals)}): {sorted((g.x,g.y) for g in goals)}")
print(f"Blues ({len(blues)}): {sorted((b.x,b.y) for b in blues)}")
print(f"Whites ({len(whites)}): {sorted((w.x,w.y) for w in whites)}")
print(f"Correct slots: {sorted(game.wyzquhjerd)}")
print(f"Wrong slots: {sorted(game.lqctaojiby)}")
print(f"Blocked ({len(game.qthdiggudy)}): {sorted(game.qthdiggudy)}")
collidable = set(game.pkbufziase)
# Get walls only (exclude dynamic)
dyn = {(player.x,player.y)} | {(g.x,g.y) for g in goals} | {(b.x,b.y) for b in blues} | {(w.x,w.y) for w in whites}
walls = collidable - dyn
print(f"Walls (static): {len(walls)}")
print(f"Steps: {game.kuncbnslnm.current_steps}")

# ASCII render 16x16
grid = [['.']*16 for _ in range(16)]
for x,y in walls:
    gx, gy = x//CELL, y//CELL
    if 0<=gx<16 and 0<=gy<16: grid[gy][gx] = '#'
for x,y in game.qthdiggudy:
    gx,gy = x//CELL, y//CELL
    if 0<=gx<16 and 0<=gy<16: grid[gy][gx] = '%'
for x,y in game.wyzquhjerd:
    gx,gy = x//CELL, y//CELL
    if 0<=gx<16 and 0<=gy<16 and grid[gy][gx]=='.': grid[gy][gx] = 'S'
for x,y in game.lqctaojiby:
    gx,gy = x//CELL, y//CELL
    if 0<=gx<16 and 0<=gy<16 and grid[gy][gx]=='.': grid[gy][gx] = 'X'
for g in goals:
    gx,gy = g.x//CELL, g.y//CELL
    grid[gy][gx] = 'P' if grid[gy][gx] in '.SX' else 'P'
for b in blues:
    gx,gy = b.x//CELL, b.y//CELL
    grid[gy][gx] = 'B'
for w in whites:
    gx,gy = w.x//CELL, w.y//CELL
    grid[gy][gx] = 'W'
gx,gy = player.x//CELL, player.y//CELL
grid[gy][gx] = '@'
print("\nGrid (16x16), #=wall %=blocked S=slot X=wrongslot P=piece B=blue W=white @=player")
print("    " + ''.join(str(i%10) for i in range(16)))
for i,row in enumerate(grid):
    print(f"{i:2d}: {''.join(row)}")
