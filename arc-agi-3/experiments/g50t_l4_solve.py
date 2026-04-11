#!/usr/bin/env python3
"""Solve g50t L4: portal + toggle puzzle.

Grid:
         1   7  13  19  25  31  37  43  49  55
    7  M3          #2   .   .   .   .   .   .
   13   P           .           .           .
   19   .   .   .   .           .           .
   25   .          M2          #1   .   .  M0
   31   .   .   .   .           .
   37              M1           .  M4(portal)
   43       G
   49       .                   .  M5(portal)
   55       .   .  #0   .   .   .

obs[0] (19,55): shift UP, activated by mod[1] (19,37) - NOT toggle
obs[1] (37,25): shift LEFT, toggle, activated by mod[2] (19,25)
obs[2] (19,7): shift LEFT, toggle, activated by mod[3] (1,7)
mod[0] (55,25) → swap gate → portals (43,37)↔(43,49)

Phase 0: ghost→mod[1] via mod[3](toggle obs[2]) and mod[2](toggle obs[1])
  UP DOWN DOWN RIGHT RIGHT RIGHT DOWN DOWN DOWN = 9 steps
Phase 1: ghost→mod[0] via y=7(obs[2] cleared by ghost0)
  DOWN RIGHT RIGHT RIGHT UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN = 15 steps
Phase 2: player→portal at step 15, teleport, navigate to goal
  DOWN RIGHT RIGHT RIGHT UP UP RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN RIGHT
  LEFT DOWN LEFT LEFT LEFT LEFT LEFT UP UP = 24 steps
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
}

arcade = Arcade()
env = arcade.make('g50t-5849a774')
fd = env.reset()
for lv in range(4):
    for name in SOLUTIONS[lv].split():
        fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
print(f'After L0-L3: completed={fd.levels_completed}')

game = env._game
lc = game.vgwycxsxjz

phase0 = 'UP DOWN DOWN RIGHT RIGHT RIGHT DOWN DOWN DOWN'  # 9 moves → mod[1]
phase1 = 'DOWN RIGHT RIGHT RIGHT UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN'  # 15 moves → mod[0]
phase2 = 'DOWN RIGHT RIGHT RIGHT UP UP RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN RIGHT LEFT DOWN LEFT LEFT LEFT LEFT LEFT UP UP'  # 24 moves

L4 = f'{phase0} UNDO {phase1} UNDO {phase2}'
actions = L4.split()
print(f'\nL4 solution: {len(actions)} actions')
print(f'  Phase 0: {len(phase0.split())} + UNDO')
print(f'  Phase 1: {len(phase1.split())} + UNDO')
print(f'  Phase 2: {len(phase2.split())}')

prev_level = fd.levels_completed
player = lc.dzxunlkwxt

for i, name in enumerate(actions):
    # Get ACTUAL player (re-read each time since UNDO changes it)
    p = lc.dzxunlkwxt
    p_before = (p.x, p.y)

    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])

    p = lc.dzxunlkwxt
    p_after = (p.x, p.y)
    ghosts = len(lc.rloltuowth)
    path_len = len(lc.areahjypvy)
    level_up = fd.levels_completed > prev_level

    ghost_pos = []
    for gi, (g, gp) in enumerate(lc.rloltuowth.items()):
        ghost_pos.append(f'g{gi}=({g.x},{g.y})')

    obs_info = ' '.join(f'o{oi}=({obs.x},{obs.y})' for oi, obs in enumerate(lc.uwxkstolmf))

    teleported = ''
    if abs(p_after[0] - p_before[0]) > STEP or abs(p_after[1] - p_before[1]) > STEP:
        teleported = f' TELEPORT!'

    sym = '★' if level_up else '·'

    # Print at key moments
    phase2_start = len(phase0.split()) + 1 + len(phase1.split()) + 1
    if (name == 'UNDO' or level_up or teleported or i == 0 or i == len(actions)-1
        or i == phase2_start or (i >= phase2_start and i % 5 == 0)):
        print(f'  {sym} {i+1:3d}. {name:5s} P={p_after} path={path_len} {obs_info}{teleported} {" ".join(ghost_pos)}')

    if level_up:
        print(f'  ★ LEVEL UP! levels_completed={fd.levels_completed}')
        break

if fd.levels_completed < 5:
    p = lc.dzxunlkwxt
    g = lc.whftgckbcu
    print(f'\nFAILED: levels_completed={fd.levels_completed}, state={fd.state.name}')
    print(f'  Player: ({p.x},{p.y}), Goal: ({g.x+1},{g.y+1})')
else:
    print(f'\n✓ L4 SOLVED!')
