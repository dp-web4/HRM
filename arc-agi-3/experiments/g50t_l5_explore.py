#!/usr/bin/env python3
"""Explore g50t L5."""
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
    print(f'After L{lv}: completed={fd.levels_completed}')

game = env._game
lc = game.vgwycxsxjz
player = lc.dzxunlkwxt
goal = lc.whftgckbcu
arena = lc.afbbgvkpip

print(f'\n=== L5 State ===')
print(f'Player: ({player.x},{player.y})')
print(f'Goal: ({goal.x+1},{goal.y+1})')
print(f'Start: ({lc.yugzlzepkr},{lc.vgpdqizwwm})')
print(f'Max undos: {len(lc.drofvwhbxb)-1}')

# Classify
mods = lc.hamayflsib
obs_list = lc.uwxkstolmf
print(f'Obstacles: {len(obs_list)}')
for i, obs in enumerate(obs_list):
    dx, dy = obs.hluvhlvimq()
    print(f'  obs[{i}]: ({obs.x},{obs.y}) rot={obs.rotation} shift=({dx},{dy}) toggle={obs.dpdubazedr}')
print(f'Modifiers: {len(mods)}')
for i, mod in enumerate(mods):
    cls = type(mod).__name__
    print(f'  mod[{i}]: ({mod.x},{mod.y}) type={cls}')
    if cls == 'lqtxaumfed' and mod.nexhtmlmxh:
        pc = mod.nexhtmlmxh
        for j, out in enumerate(pc.ytztewxdin):
            out_cls = type(out).__name__
            if out_cls == 'yyzqramdhd':
                print(f'    → obs at ({out.x},{out.y}) toggle={out.dpdubazedr}')
            elif out_cls == 'crfcpstubm':
                print(f'    → SWAP GATE at ({out.x},{out.y})')
                if out.lzcbbrzwdn: print(f'      portal_a: ({out.lzcbbrzwdn.x},{out.lzcbbrzwdn.y})')
                if out.esbtgcjrfp: print(f'      portal_b: ({out.esbtgcjrfp.x},{out.esbtgcjrfp.y})')
    elif cls == 'ulhhdeoyok':
        print(f'    (portal)')

print(f'Enemies: {len(lc.kgvnkyaimw)}')
for enemy, path in lc.kgvnkyaimw.items():
    print(f'  enemy at ({enemy.x},{enemy.y})')

# Map grid
offset = player.x % STEP
walkable = set()
obs_cells = set()
orig_x, orig_y = player.x, player.y
for px in range(offset, 70, STEP):
    for py in range(offset, 70, STEP):
        player.set_position(px, py)
        if lc.xvkyljflji(player, arena):
            if lc.vjpujwqrto(player):
                obs_cells.add((px, py))
            else:
                walkable.add((px, py))
player.set_position(orig_x, orig_y)

all_cells = walkable | obs_cells
xs = sorted(set(p[0] for p in all_cells))
ys = sorted(set(p[1] for p in all_cells))
mod_set = {(m.x, m.y): i for i, m in enumerate(mods)}

print(f'\nWalkable: {len(walkable)}, Obs blocked: {len(obs_cells)}')
header = '      ' + ''.join(f'{x:4d}' for x in xs)
print(header)
for y in ys:
    row = f'  {y:3d} '
    for x in xs:
        if (x, y) == (player.x, player.y):
            row += '  P '
        elif (x, y) == (goal.x+1, goal.y+1):
            row += '  G '
        elif (x, y) in obs_cells:
            oi = next((i for i, o in enumerate(obs_list) if (o.x, o.y) == (x, y)), '?')
            row += f' #{oi} '
        elif (x, y) in mod_set:
            mi = mod_set[(x,y)]
            row += f' M{mi} '
        elif (x, y) in walkable:
            row += '  . '
        else:
            row += '    '
    print(row)
