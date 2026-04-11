#!/usr/bin/env python3
"""Verify L2 solution for g50t with obstacle shift mechanics."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}

arcade = Arcade()
env = arcade.make('g50t-5849a774')
fd = env.reset()

# Solve L0 and L1
L0 = 'RIGHT RIGHT RIGHT RIGHT UNDO DOWN DOWN DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT RIGHT'
L1 = 'LEFT LEFT UNDO DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT UNDO UP UP UP LEFT LEFT LEFT LEFT LEFT LEFT LEFT DOWN DOWN RIGHT RIGHT RIGHT'

for name in L0.split():
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
print(f'After L0: completed={fd.levels_completed}')
for name in L1.split():
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
print(f'After L1: completed={fd.levels_completed}')

# L2 solution:
# Phase 0: Record path to mod[1] via mod[2] (toggle obs[2] on the way)
#   start(7,19) → UP UP → RIGHT×4 → DOWN DOWN DOWN DOWN → RIGHT → UNDO
# Phase 1: Record path to mod[0] (ghost0 replays, clearing obs[2] and obs[0])
#   start(7,19) → UP UP → RIGHT×7 → DOWN×7 → LEFT×5 → UNDO
# Phase 2: Navigate to goal (both ghosts replay)
#   start(7,19) → UP UP → RIGHT×7 → DOWN×7 → LEFT×8 → UP UP UP → RIGHT RIGHT → UP UP

phase0 = 'UP UP RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN RIGHT'
phase1 = 'UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT'
phase2 = 'UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT LEFT LEFT UP UP UP RIGHT RIGHT UP UP'

L2 = f'{phase0} UNDO {phase1} UNDO {phase2}'
actions = L2.split()
print(f'\nL2 solution: {len(actions)} actions')
print(f'  Phase 0: {len(phase0.split())} moves + UNDO')
print(f'  Phase 1: {len(phase1.split())} moves + UNDO')
print(f'  Phase 2: {len(phase2.split())} moves')

game = env._game
lc = game.vgwycxsxjz

prev_level = fd.levels_completed
for i, name in enumerate(actions):
    p = lc.dzxunlkwxt
    obs_status = []
    for j, obs in enumerate(lc.uwxkstolmf):
        obs_status.append(f'obs{j}=({obs.x},{obs.y}){"*" if obs.dijhfchobv else ""}')

    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])

    p = lc.dzxunlkwxt
    ghosts = len(lc.rloltuowth)
    path_len = len(lc.areahjypvy)
    level_up = fd.levels_completed > prev_level

    sym = '★' if level_up else '·'
    # Only print key steps
    if name == 'UNDO' or level_up or (i % 10 == 0) or i == len(actions) - 1:
        print(f'  {sym} {i+1:3d}. {name:5s} → P=({p.x},{p.y}) g={ghosts} path={path_len} {" ".join(obs_status)}')

    if level_up:
        prev_level = fd.levels_completed
        print(f'  ★ LEVEL UP! levels_completed={fd.levels_completed}')
        break

if fd.levels_completed < 3:
    print(f'\n  FAILED: levels_completed={fd.levels_completed}, state={fd.state.name}')
    p = lc.dzxunlkwxt
    g = lc.whftgckbcu
    print(f'  Player: ({p.x},{p.y}), Goal target: ({g.x+1},{g.y+1})')

    # Show where player is vs goal
    obs_status = []
    for j, obs in enumerate(lc.uwxkstolmf):
        obs_status.append(f'obs{j}=({obs.x},{obs.y}) active={obs.dijhfchobv}')
    print(f'  Obstacles: {", ".join(obs_status)}')
else:
    print(f'\n  ✓ L2 SOLVED! levels_completed={fd.levels_completed}')
