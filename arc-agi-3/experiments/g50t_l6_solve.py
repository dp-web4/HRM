#!/usr/bin/env python3
"""Solve g50t L6: the most complex level.

Grid (6 connected components):
         1   7  13  19  25  31  37  43  49
    1          M3          #0   .   .  M1    ← comp4(3cells) | comp3(3cells)
    7           .           .                ← comp4 cont    | comp0 via (31,7)
   13          M4       .   .   .   .   .    ← comp4(portal) | comp0 (main, 22 cells)
   19                           .       .
   25  M0      M5   .   P       .       .    ← comp5(2cells) | comp0 cont
   31   .               .       .       .    ← comp5 cont
   37  #1          M2   .   .   .      M6    ← obs[1]       | comp0 cont (portal)
   43   .           .
   49   .                   G   .   .  M7    ← comp1(6cells) | comp2(goal,4cells, portal)
   55   .   .   .   .                        ← comp1 cont (enemy start)

Portal systems:
  LEFT:  mod[0]@(1,25) → swap gate → (13,13)↔(13,25) [comp4↔comp0]
  RIGHT: mod[1]@(49,1) → swap gate → (49,37)↔(49,49) [comp0↔comp2]

Obstacles:
  obs[0]@(31,1): shift LEFT, NOT toggle → bridges comp3↔comp0 via (31,7)
  obs[1]@(1,37): shift RIGHT, toggle → bridges comp1↔comp5

Enemy at (19,55), waypoint (1,25), bounds x=[1,20] y=[25,56]

Solution chain:
  1. Ghost1 toggles obs[1] (mod[2]) → enemy reaches mod[0] → left swap fires
  2. Ghost0 at portal (13,25) → teleported to (13,13) → walks UP UP to mod[3] (13,1)
  3. Ghost0 on mod[3] keeps obs[0] shifted → opens (31,1)
  4. Ghost1 walks through (31,1) to mod[1] (49,1) → right swap fires
  5. Player at portal (49,37) → teleported to (49,49) → walks to goal

Phase 0 (ghost0): 16 steps - timed to reach portal at step 8
  DOWN UP DOWN UP DOWN UP LEFT LEFT RIGHT RIGHT DOWN DOWN RIGHT LEFT UP UP
Phase 1 (ghost1): 24 steps - toggles obs[1] step 3, navigates to mod[1] after step 16
  DOWN DOWN LEFT RIGHT UP UP DOWN DOWN UP UP DOWN DOWN RIGHT RIGHT UP UP UP UP LEFT UP UP RIGHT RIGHT RIGHT
Phase 2 (player): 27 steps - reach portal, pace, teleport, walk to goal
  DOWN DOWN RIGHT RIGHT UP UP UP UP RIGHT RIGHT DOWN DOWN DOWN DOWN UP DOWN UP DOWN UP DOWN UP DOWN UP DOWN LEFT LEFT LEFT
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

# Phase 0: ghost0 path (16 steps)
phase0 = 'DOWN UP DOWN UP DOWN UP LEFT LEFT RIGHT RIGHT DOWN DOWN RIGHT LEFT UP UP'

# Phase 1: ghost1 path (24 steps)
phase1 = 'DOWN DOWN LEFT RIGHT UP UP DOWN DOWN UP UP DOWN DOWN RIGHT RIGHT UP UP UP UP LEFT UP UP RIGHT RIGHT RIGHT'

# Phase 2: player final run (27 steps)
phase2 = 'DOWN DOWN RIGHT RIGHT UP UP UP UP RIGHT RIGHT DOWN DOWN DOWN DOWN UP DOWN UP DOWN UP DOWN UP DOWN UP DOWN LEFT LEFT LEFT'

L6 = f'{phase0} UNDO {phase1} UNDO {phase2}'
actions = L6.split()
print(f'\nL6 solution: {len(actions)} actions')
print(f'  Phase 0: {len(phase0.split())} + UNDO')
print(f'  Phase 1: {len(phase1.split())} + UNDO')
print(f'  Phase 2: {len(phase2.split())}')

prev_level = fd.levels_completed
obs_list = lc.uwxkstolmf
enemies = list(lc.kgvnkyaimw.keys())
enemy = enemies[0] if enemies else None

phase0_len = len(phase0.split())
phase1_start = phase0_len + 1  # after first UNDO
phase1_len = len(phase1.split())
phase2_start = phase1_start + phase1_len + 1  # after second UNDO

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

    ghosts_pos = []
    for gi, (g, gp) in enumerate(lc.rloltuowth.items()):
        ghosts_pos.append(f'g{gi}=({g.x},{g.y})')
    ghost_str = ' '.join(ghosts_pos)

    obs = ' '.join(f'o{j}=({o.x},{o.y})' for j, o in enumerate(obs_list))
    sym = '★' if level_up else '·'

    # Determine phase
    if i < phase0_len:
        phase_label = f'P0.{i+1:2d}'
    elif i == phase0_len:
        phase_label = 'UNDO1 '
    elif i < phase1_start + phase1_len:
        step_in_p1 = i - phase1_start + 1
        phase_label = f'P1.{step_in_p1:2d}'
    elif i == phase1_start + phase1_len:
        phase_label = 'UNDO2 '
    else:
        step_in_p2 = i - phase2_start + 1
        phase_label = f'P2.{step_in_p2:2d}'

    # Print at key moments
    show = (name == 'UNDO' or level_up or e_on_mod or i == 0 or i == len(actions)-1)

    # Key Phase 2 steps
    if i >= phase2_start:
        step_in_p2 = i - phase2_start + 1
        if step_in_p2 in [1, 3, 8, 10, 14, 15, 16, 21, 24, 25, 26, 27]:
            show = True

    # Key Phase 1 steps (during recording)
    if phase1_start <= i < phase1_start + phase1_len:
        step_in_p1 = i - phase1_start + 1
        if step_in_p1 in [1, 3, 8, 16, 21, 24]:
            show = True

    if show:
        print(f'  {sym} {phase_label} {name:5s} P=({p.x},{p.y}) {e_pos}{e_on_mod} | {obs} | {ghost_str}')

    if level_up:
        print(f'  ★ LEVEL UP! levels_completed={fd.levels_completed}')
        break

if fd.levels_completed < 7:
    p = lc.dzxunlkwxt
    g = lc.whftgckbcu
    print(f'\nFAILED: levels_completed={fd.levels_completed}, state={fd.state.name}')
    print(f'  Player: ({p.x},{p.y}), Goal: ({g.x+1},{g.y+1})')
    for i, obs in enumerate(obs_list):
        print(f'  obs[{i}]: ({obs.x},{obs.y})')
    # Show ghost positions
    for gi, (g, gp) in enumerate(lc.rloltuowth.items()):
        print(f'  ghost{gi}: ({g.x},{g.y})')
else:
    print(f'\n✓ L6 SOLVED!')
