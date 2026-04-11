#!/usr/bin/env python3
"""Simulate L6 mechanics: enemy path, ghost teleportation, obs[0] shift.

Key questions:
1. obs[0] at (31,1) shifts LEFT (-1,0) → moves to (25,1). Does this open y=1 path?
2. Can a ghost on portal (13,25) get teleported to (13,13) when enemy hits mod[0]?
3. What's the enemy's actual path from (19,55) to waypoint (1,25)?
4. Does obs[1] at (1,37) block the enemy? Toggle via mod[2] at (19,37).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}
STEP = 6

SOLUTIONS = {
    0: 'RIGHT RIGHT RIGHT RIGHT UNDO DOWN DOWN DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT RIGHT',
    1: 'LEFT LEFT UNDO DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT UNDO UP UP UP LEFT LEFT LEFT LEFT LEFT LEFT LEFT DOWN DOWN RIGHT RIGHT RIGHT',
    2: 'UP UP RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN RIGHT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT LEFT LEFT UP UP UP RIGHT RIGHT UP UP',
    3: 'DOWN DOWN RIGHT DOWN UNDO DOWN DOWN RIGHT RIGHT UP UP RIGHT RIGHT DOWN DOWN DOWN UNDO LEFT LEFT LEFT DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT LEFT LEFT LEFT',
    4: 'UP DOWN DOWN RIGHT RIGHT RIGHT DOWN DOWN DOWN UNDO DOWN RIGHT RIGHT RIGHT UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN UNDO DOWN RIGHT RIGHT RIGHT UP UP RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN RIGHT LEFT DOWN LEFT LEFT LEFT LEFT LEFT UP UP',
    5: 'LEFT LEFT UP UNDO LEFT LEFT UP LEFT LEFT UNDO LEFT LEFT DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT LEFT DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT UP DOWN LEFT LEFT UP UP UP UP UP RIGHT RIGHT RIGHT DOWN DOWN RIGHT RIGHT DOWN DOWN RIGHT RIGHT',
}

arcade = Arcade()
env = arcade.make('g50t-5849a774')
fd = env.reset()
for lv in range(6):
    for name in SOLUTIONS[lv].split():
        fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
print(f'After L0-L5: completed={fd.levels_completed}')

game = env._game
lc = game.vgwycxsxjz

# === Test 1: What does obs[0] shifting LEFT do? ===
print('\n=== Test 1: obs[0] shift analysis ===')
obs0 = lc.uwxkstolmf[0]
obs1 = lc.uwxkstolmf[1]
print(f'obs[0] at ({obs0.x},{obs0.y}) rot={obs0.rotation} toggle={obs0.dpdubazedr}')
print(f'obs[1] at ({obs1.x},{obs1.y}) rot={obs1.rotation} toggle={obs1.dpdubazedr}')
# obs[0] shift is (-1,0) = LEFT in pixel terms
# With STEP=6, shift amount = 6 pixels LEFT
# So obs[0] moves from (31,1) to (31-6,1) = (25,1)
# This means (31,1) opens but (25,1) closes...
# Wait - let me check what shift actually means in the code
dx, dy = obs0.hluvhlvimq()
print(f'obs[0] shift vector: ({dx},{dy}) → moves to ({obs0.x + dx*STEP},{obs0.y + dy*STEP})')
dx1, dy1 = obs1.hluvhlvimq()
print(f'obs[1] shift vector: ({dx1},{dy1}) → moves to ({obs1.x + dx1*STEP},{obs1.y + dy1*STEP})')

# === Test 2: Check walkability on y=1 with and without obs[0] ===
print('\n=== Test 2: y=1 walkability ===')
player = lc.dzxunlkwxt
arena = lc.afbbgvkpip
orig_x, orig_y = player.x, player.y
offset = player.x % STEP  # = 1

print('Current y=1 walkability:')
for x in range(1, 56, STEP):
    player.set_position(x, 1)
    in_arena = lc.xvkyljflji(player, arena)
    blocked = lc.vjpujwqrto(player) if in_arena else False
    status = 'BLOCKED' if blocked else ('walk' if in_arena else 'wall')
    print(f'  ({x},1): {status}')
player.set_position(orig_x, orig_y)

# === Test 3: Check enemy path ===
print('\n=== Test 3: Enemy analysis ===')
enemies = list(lc.kgvnkyaimw.keys())
enemy = enemies[0]
wp = enemy.baygqyisjz
print(f'Enemy: ({enemy.x},{enemy.y}) dir={enemy.wzgvpxcawd}')
print(f'Waypoint: ({wp.x},{wp.y}) w={wp.width} h={wp.height}')
# Check waypoint bounds
print(f'Waypoint bounds: x=[{wp.x},{wp.x+wp.width}] y=[{wp.y},{wp.y+wp.height}]')

# === Test 4: What's blocking the enemy path? ===
print('\n=== Test 4: Enemy path column x=19 and x=1 ===')
for x_col in [19, 13, 7, 1]:
    print(f'Column x={x_col}:')
    for y in range(1, 62, STEP):
        player.set_position(x_col, y)
        in_arena = lc.xvkyljflji(player, arena)
        blocked = lc.vjpujwqrto(player) if in_arena else False
        status = 'BLOCKED' if blocked else ('walk' if in_arena else 'wall')
        if in_arena:
            print(f'  ({x_col},{y}): {status}')
player.set_position(orig_x, orig_y)

# === Test 5: Simulate enemy movement for many steps ===
print('\n=== Test 5: Enemy movement simulation (pacing player) ===')
obs_list = lc.uwxkstolmf
mods = lc.hamayflsib

# Just pace LEFT/RIGHT and watch enemy
for i in range(80):
    p = lc.dzxunlkwxt
    name = 'LEFT' if i % 2 == 0 else 'RIGHT'
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
    p = lc.dzxunlkwxt

    enemy_on_mod = ''
    for mi, mod in enumerate(mods):
        if enemy.x == mod.x and enemy.y == mod.y:
            enemy_on_mod = f' ON_MOD[{mi}]!'

    obs_pos = ' '.join(f'o{j}=({o.x},{o.y})' for j, o in enumerate(obs_list))

    if i < 20 or enemy_on_mod or i % 10 == 0:
        print(f'  {i+1:3d}. {name:5s} P=({p.x},{p.y}) E=({enemy.x},{enemy.y}) dir={enemy.wzgvpxcawd}{enemy_on_mod} {obs_pos}')

    if fd.levels_completed > 6:
        print('  LEVEL UP!')
        break
