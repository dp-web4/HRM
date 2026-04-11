#!/usr/bin/env python3
"""Simulate L5 enemy movement with ghosts clearing y=7 path."""
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
}

arcade = Arcade()
env = arcade.make('g50t-5849a774')
fd = env.reset()
for lv in range(5):
    for name in SOLUTIONS[lv].split():
        fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
print(f'After L0-L4: completed={fd.levels_completed}')

game = env._game
lc = game.vgwycxsxjz

# Get enemy info
enemies = list(lc.kgvnkyaimw.keys())
enemy = enemies[0]
print(f'Enemy start: ({enemy.x},{enemy.y})')
print(f'Enemy direction: {enemy.wzgvpxcawd}')
if enemy.baygqyisjz:
    wp = enemy.baygqyisjz
    print(f'Enemy waypoint: ({wp.x},{wp.y})')

# Phase 0: player to mod[0] at (43,25)
phase0 = 'LEFT LEFT UP'
print(f'\n=== Phase 0: {phase0} ===')
for i, name in enumerate(phase0.split()):
    p = lc.dzxunlkwxt
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
    p = lc.dzxunlkwxt
    print(f'  {i+1}. {name} P=({p.x},{p.y}) E=({enemy.x},{enemy.y}) dir={enemy.wzgvpxcawd}')

print(f'\nUNDO...')
fd = env.step(INT_TO_GA[NAME_TO_INT['UNDO']])
p = lc.dzxunlkwxt
print(f'After UNDO: P=({p.x},{p.y}) E=({enemy.x},{enemy.y}) ghosts={len(lc.rloltuowth)}')

# Phase 1: player to mod[1] at (31,25)
phase1 = 'LEFT LEFT UP LEFT LEFT'
print(f'\n=== Phase 1: {phase1} ===')
for i, name in enumerate(phase1.split()):
    p = lc.dzxunlkwxt
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
    p = lc.dzxunlkwxt
    # Ghost0 position
    g0_pos = [(g.x, g.y) for g in lc.rloltuowth.keys()]
    obs_status = ' '.join(f'o{j}=({o.x},{o.y})' for j, o in enumerate(lc.uwxkstolmf))
    print(f'  {i+1}. {name} P=({p.x},{p.y}) E=({enemy.x},{enemy.y}) dir={enemy.wzgvpxcawd} g={g0_pos} {obs_status}')

print(f'\nUNDO...')
fd = env.step(INT_TO_GA[NAME_TO_INT['UNDO']])
p = lc.dzxunlkwxt
print(f'After UNDO: P=({p.x},{p.y}) E=({enemy.x},{enemy.y}) ghosts={len(lc.rloltuowth)}')

# Phase 2: let player pace LEFT/RIGHT while tracking enemy
print(f'\n=== Phase 2: Pacing and watching enemy ===')
obs_list = lc.uwxkstolmf
for i in range(50):
    p = lc.dzxunlkwxt
    # Alternate LEFT/RIGHT to pace
    if i % 2 == 0:
        name = 'LEFT'
    else:
        name = 'RIGHT'

    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
    p = lc.dzxunlkwxt

    ghost_pos = [(g.x, g.y) for g in lc.rloltuowth.keys()]
    obs_status = ' '.join(f'o{j}=({o.x},{o.y})' for j, o in enumerate(obs_list))

    # Check if enemy is on a modifier
    enemy_on_mod = ''
    for mi, mod in enumerate(lc.hamayflsib):
        if enemy.x == mod.x and enemy.y == mod.y:
            enemy_on_mod = f' ON_MOD[{mi}]!'

    if i < 10 or enemy_on_mod or i % 5 == 0:
        print(f'  {i+1:3d}. {name:5s} P=({p.x},{p.y}) E=({enemy.x},{enemy.y}){enemy_on_mod} dir={enemy.wzgvpxcawd} {obs_status}')

    if fd.levels_completed > 5:
        print(f'  LEVEL UP!')
        break
