#!/usr/bin/env python3
"""Debug: Why can't player reach white in L7?"""
import sys, os
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")

src = open('arc-agi-3/experiments/wa30_solve_final.py').read()
cut = src.index('total_actions = 0')
exec(src[:cut], globals())

KNOWN = KNOWN_SOLUTIONS
all_solutions = []
for lv in range(7):
    fd = replay_from_solutions(all_solutions)
    blues = game.current_level.get_sprites_by_tag("kdweefinfi")
    whites = game.current_level.get_sprites_by_tag("ysysltqlke")
    has_ai = len(blues)>0 or len(whites)>0
    if lv in KNOWN:
        sol = string_to_actions(KNOWN[lv])
    elif not has_ai:
        sol = solve_no_ai()
    else:
        sol = solve_with_ai(all_solutions)
    fd = replay_from_solutions(all_solutions)
    before = fd.levels_completed
    actual = []
    for m in sol:
        fd = env.step(m); actual.append(m)
        if fd.levels_completed > before or fd.state.name == 'WIN': break
    all_solutions.append(actual)

# Step forward in L7 until step ~20
fd = replay_from_solutions(all_solutions)
slots = game.wyzquhjerd
slots2 = game.lqctaojiby
slot_aligned = set((sx,sy) for sx,sy in slots if sx%4==0 and sy%4==0)

# Use the current solve_with_ai logic for 20 steps
for step in range(20):
    player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    whites = game.current_level.get_sprites_by_tag("ysysltqlke")
    blues = game.current_level.get_sprites_by_tag("kdweefinfi")
    collidable = set(game.pkbufziase)
    blocked = set(game.qthdiggudy)
    pc = game.nsevyuople.get(player, None)
    a = plan_action(player, goals, whites, blues, collidable, blocked, slots, slots2, slot_aligned, pc)
    if a is None: a = UP
    fd = env.step(a)

# Now inspect
player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
whites = game.current_level.get_sprites_by_tag("ysysltqlke")
goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
collidable = set(game.pkbufziase)
print(f"player=({player.x},{player.y})")
for w in whites:
    print(f"white=({w.x},{w.y})")
# What's around white?
w = whites[0]
for ddx,ddy in [(-4,0),(4,0),(0,-4),(0,4)]:
    nx,ny = w.x+ddx, w.y+ddy
    in_c = (nx,ny) in collidable
    print(f"  adj ({nx},{ny}) collidable={in_c}")
# BFS from player to adj cells (excluding player's own pos)
obs = collidable.copy()
obs.discard((player.x, player.y))
# Also consider: is the white itself collidable? Yes (sprites in pkbufziase). But adj cells don't include white pos.
# Target: cells adjacent to white that are NOT collidable
from collections import deque
adj_cells = set()
for ddx,ddy in [(-4,0),(4,0),(0,-4),(0,4)]:
    nx,ny = w.x+ddx, w.y+ddy
    if 0<=nx<64 and 0<=ny<64:
        adj_cells.add((nx,ny))
print(f"adj_cells={adj_cells}")
# Check which are collidable
print(f"free adj: {[c for c in adj_cells if c not in obs]}")

# BFS from player
path = bfs_to_targets((player.x,player.y), adj_cells, obs)
print(f"BFS path: {path}")

# Print 16x16 of collidables near white
print("\nGrid around white+player:")
print("    " + ''.join(str(i%10) for i in range(16)))
for y in range(16):
    row = ''
    for x in range(16):
        px,py = x*4, y*4
        if (px,py) == (player.x,player.y): row += '@'
        elif (px,py) == (w.x,w.y): row += 'W'
        elif (px,py) in collidable: row += '#'
        else: row += '.'
    print(f"{y:2d}: {row}")

# Which pieces are on wrong slots near white?
for g in goals:
    if abs(g.x-w.x)+abs(g.y-w.y) <= 8:
        print(f"piece near white: ({g.x},{g.y}) claimed={game.zmqreragji.get(g) is not None}")
