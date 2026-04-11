#!/usr/bin/env python3
"""Solve g50t L3: portal teleportation puzzle.

Grid (step=6, offset=1):
         7  13  19  25  31  37  43  49
    7   .   .   .   P       .   .   .
   13   .           .       .       .
   19   .           .   .   .       .
   25   #              M1          M0
   31   .
   37   .   .   .  M2
   49   G   .   .  M3

Player=(25,7), Goal=(7,49), 2 undos
obs[0] at (7,25): rot=270 → shifts RIGHT, activated by mod[1] at (31,25)
mod[0] at (49,25) → swap gate at (23,40) connecting portal M2(25,37) ↔ M3(25,49)
mod[1] at (31,25) → obs[0]

Strategy:
  Phase 0: Record path to mod[1] at (31,25) → ghost0 holds obs clear
  Phase 1: Record path to mod[0] at (49,25) → ghost1 triggers swap gate
  Phase 2: Navigate LEFT to x=7, DOWN through cleared obs, RIGHT to portal,
           get teleported to (25,49), LEFT to goal at (7,49)

  Key timing: player reaches portal at step 11 = same step ghost1 reaches mod[0]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}
STEP = 6

L0 = 'RIGHT RIGHT RIGHT RIGHT UNDO DOWN DOWN DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT RIGHT'
L1 = 'LEFT LEFT UNDO DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT UNDO UP UP UP LEFT LEFT LEFT LEFT LEFT LEFT LEFT DOWN DOWN RIGHT RIGHT RIGHT'
L2 = 'UP UP RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN RIGHT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT LEFT LEFT UP UP UP RIGHT RIGHT UP UP'

arcade = Arcade()
env = arcade.make('g50t-5849a774')
fd = env.reset()
for name in (L0 + ' ' + L1 + ' ' + L2).split():
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
print(f'After L0-L2: completed={fd.levels_completed}')

game = env._game
lc = game.vgwycxsxjz
player = lc.dzxunlkwxt

# L3 Solution
phase0 = 'DOWN DOWN RIGHT DOWN'           # 4 moves → mod[1] at (31,25)
phase1 = 'DOWN DOWN RIGHT RIGHT UP UP RIGHT RIGHT DOWN DOWN DOWN'  # 11 moves → mod[0] at (49,25)
phase2 = 'LEFT LEFT LEFT DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT LEFT LEFT LEFT'  # 14 moves → goal

L3 = f'{phase0} UNDO {phase1} UNDO {phase2}'
actions = L3.split()
print(f'\nL3 solution: {len(actions)} actions')
print(f'  Phase 0: {len(phase0.split())} moves + UNDO')
print(f'  Phase 1: {len(phase1.split())} moves + UNDO')
print(f'  Phase 2: {len(phase2.split())} moves')

prev_level = fd.levels_completed
for i, name in enumerate(actions):
    p_before = (player.x, player.y)

    # Check portal states before step
    portals = [m for m in lc.hamayflsib if type(m).__name__ == 'ulhhdeoyok']
    portal_info = ''
    for pi, portal in enumerate(portals):
        if portal.vbqvjbxkfm:
            portal_info += f' P{pi}={len(portal.vbqvjbxkfm)}'

    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])

    p_after = (player.x, player.y)
    ghosts = len(lc.rloltuowth)
    path_len = len(lc.areahjypvy)
    level_up = fd.levels_completed > prev_level

    # Ghost positions
    ghost_pos = []
    for gi, (g, gp) in enumerate(lc.rloltuowth.items()):
        ghost_pos.append(f'g{gi}=({g.x},{g.y})')

    # Check obstacle status
    obs_info = ''
    for oi, obs in enumerate(lc.uwxkstolmf):
        obs_info += f' obs{oi}=({obs.x},{obs.y})'

    sym = '★' if level_up else '·'
    teleported = ''
    if abs(p_after[0] - p_before[0]) > STEP or abs(p_after[1] - p_before[1]) > STEP:
        teleported = f' TELEPORTED from {p_before}!'

    # Print key steps
    if name == 'UNDO' or level_up or i == 0 or i == len(actions) - 1 or teleported or (i >= len(phase0.split()) + 1 + len(phase1.split()) + 1):
        print(f'  {sym} {i+1:3d}. {name:5s} → P={p_after} g={ghosts} path={path_len}{obs_info}{portal_info}{teleported} {" ".join(ghost_pos)}')

    if level_up:
        prev_level = fd.levels_completed
        print(f'  ★ LEVEL UP! levels_completed={fd.levels_completed}')
        break

if fd.levels_completed < 4:
    print(f'\nFAILED: levels_completed={fd.levels_completed}, state={fd.state.name}')
    print(f'  Player: ({player.x},{player.y}), Goal: ({lc.whftgckbcu.x+1},{lc.whftgckbcu.y+1})')
else:
    print(f'\n✓ L3 SOLVED! levels_completed={fd.levels_completed}')
    print(f'  Solution: {L3}')
