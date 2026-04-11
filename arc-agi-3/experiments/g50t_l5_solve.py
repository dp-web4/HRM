#!/usr/bin/env python3
"""Solve g50t L5: enemy-assisted toggle puzzle.

Grid:
         1   7  13  19  25  31  37  43  49  55
    7  M3   .   .  M2   .  #1   .  #0   .   .
   13                                       .
   19                                       .  (enemy)
   25  #3   .   .  #2      M1   .  M0
   31   .           .               .   .   P
   37   .       .   .   .   .   .   .       .
   43   .                   .               .
   49   .      M4          #4   .   G       .
   55   .   .   .

obs[0] (43,7) shift DOWN, mod[0] (43,25) | obs[1] (31,7) shift DOWN, mod[1] (31,25)
obs[2] (19,25) shift UP toggle, mod[2] (19,7) | obs[3] (1,25) shift UP toggle, mod[3] (1,7)
obs[4] (31,49) shift LEFT toggle, mod[4] (13,49)
Enemy at (55,19), waypoint (1,7)

Strategy:
  Phase 0: ghost→mod[0] (3 steps: LEFT LEFT UP)
  Phase 1: ghost→mod[1] via mod[0] (5 steps: LEFT LEFT UP LEFT LEFT)
  Phase 2: ghosts clear y=7 for enemy. Enemy reaches mod[2] at step 8, mod[3] at step 11.
    Player navigates through freed obs[2]/obs[3] windows to toggle obs[4] via mod[4],
    then reaches goal via cleared obs[4].
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}

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

phase0 = 'LEFT LEFT UP'  # 3 moves → mod[0] at (43,25)
phase1 = 'LEFT LEFT UP LEFT LEFT'  # 5 moves → mod[1] at (31,25)
phase2 = 'LEFT LEFT DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT LEFT DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT UP DOWN LEFT LEFT UP UP UP UP UP RIGHT RIGHT RIGHT DOWN DOWN RIGHT RIGHT DOWN DOWN RIGHT RIGHT'

L5 = f'{phase0} UNDO {phase1} UNDO {phase2}'
actions = L5.split()
print(f'\nL5 solution: {len(actions)} actions')
print(f'  Phase 0: {len(phase0.split())} + UNDO')
print(f'  Phase 1: {len(phase1.split())} + UNDO')
print(f'  Phase 2: {len(phase2.split())}')

prev_level = fd.levels_completed
enemies = list(lc.kgvnkyaimw.keys())
enemy = enemies[0] if enemies else None
obs_list = lc.uwxkstolmf
phase2_start = len(phase0.split()) + 1 + len(phase1.split()) + 1

for i, name in enumerate(actions):
    p = lc.dzxunlkwxt
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
    p = lc.dzxunlkwxt
    level_up = fd.levels_completed > prev_level

    e_pos = f'E=({enemy.x},{enemy.y})' if enemy else ''
    e_on_mod = ''
    if enemy:
        for mi, mod in enumerate(lc.hamayflsib):
            if enemy.x == mod.x and enemy.y == mod.y:
                e_on_mod = f' E_ON_M{mi}'

    obs = ' '.join(f'o{j}=({o.x},{o.y})' for j, o in enumerate(obs_list))
    sym = '★' if level_up else '·'

    step_in_p2 = i - phase2_start + 1 if i >= phase2_start else -1

    if (name == 'UNDO' or level_up or e_on_mod or i == 0 or i == len(actions)-1
        or (step_in_p2 in [8,9,10,11,12,20,21,28,29,30,31,36,37,38,39])):
        print(f'  {sym} {i+1:3d}. {name:5s} P=({p.x},{p.y}) {e_pos}{e_on_mod} | {obs}')

    if level_up:
        print(f'  ★ LEVEL UP! levels_completed={fd.levels_completed}')
        break

if fd.levels_completed < 6:
    p = lc.dzxunlkwxt
    g = lc.whftgckbcu
    print(f'\nFAILED: levels_completed={fd.levels_completed}, state={fd.state.name}')
    print(f'  Player: ({p.x},{p.y}), Goal: ({g.x+1},{g.y+1})')
    # Show obs status
    for i, obs in enumerate(obs_list):
        print(f'  obs[{i}]: ({obs.x},{obs.y})')
else:
    print(f'\n✓ L5 SOLVED!')
