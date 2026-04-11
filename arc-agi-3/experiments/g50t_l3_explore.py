#!/usr/bin/env python3
"""Explore g50t L3 portal mechanics."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}
STEP = 6

# Known solutions for L0-L2
L0 = 'RIGHT RIGHT RIGHT RIGHT UNDO DOWN DOWN DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT RIGHT'
L1 = 'LEFT LEFT UNDO DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT UNDO UP UP UP LEFT LEFT LEFT LEFT LEFT LEFT LEFT DOWN DOWN RIGHT RIGHT RIGHT'
L2 = 'UP UP RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN RIGHT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT LEFT LEFT UP UP UP RIGHT RIGHT UP UP'

arcade = Arcade()
env = arcade.make('g50t-5849a774')
fd = env.reset()

# Solve L0-L2
for name in L0.split():
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
print(f'After L0: completed={fd.levels_completed}')
for name in L1.split():
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
print(f'After L1: completed={fd.levels_completed}')
for name in L2.split():
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
print(f'After L2: completed={fd.levels_completed}')

game = env._game
lc = game.vgwycxsxjz
player = lc.dzxunlkwxt
goal = lc.whftgckbcu
arena = lc.afbbgvkpip
obstacles = lc.uwxkstolmf
modifiers = lc.hamayflsib
indicators = lc.drofvwhbxb

print(f'\n=== L3 State ===')
print(f'Player: ({player.x},{player.y})')
print(f'Goal: ({goal.x+1},{goal.y+1})')
print(f'Start: ({lc.yugzlzepkr},{lc.vgpdqizwwm})')
print(f'Max undos: {len(indicators)-1}')
print(f'Obstacles: {len(obstacles)}')
print(f'Modifiers: {len(modifiers)}')

# Classify each modifier
for i, mod in enumerate(modifiers):
    cls = type(mod).__name__
    print(f'  mod[{i}]: ({mod.x},{mod.y}) type={cls}')
    # Check for pressure pad attributes
    if hasattr(mod, 'nexhtmlmxh'):
        pc = mod.nexhtmlmxh
        if pc:
            targets = [(o.x, o.y) for o in pc.ytztewxdin]
            print(f'    → targets: {targets}')
    # Check for portal attributes
    if hasattr(mod, 'crfcpstubm'):
        sg = mod.crfcpstubm
        print(f'    → swap gate: {sg}')
        if sg:
            print(f'      portal_a: ({sg.lzcbbrzwdn.x},{sg.lzcbbrzwdn.y})')
            print(f'      portal_b: ({sg.esbtgcjrfp.x},{sg.esbtgcjrfp.y})')
    # Check for entity tracking
    if hasattr(mod, 'vbqvjbxkfm'):
        ents = mod.vbqvjbxkfm
        print(f'    → tracked entities: {ents}')
    if hasattr(mod, 'rqxibulywa'):
        print(f'    → rqxibulywa: {mod.rqxibulywa}')

# Map the full grid
print(f'\n=== Grid Map ===')
walkable = set()
obs_cells = set()
orig_x, orig_y = player.x, player.y
for px in range(arena.x - 12, arena.x + arena.width + 12, STEP):
    for py in range(arena.y - 12, arena.y + arena.height + 12, STEP):
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

# Print grid with column headers
print(f'  Walkable: {len(walkable)}, Obs blocked: {len(obs_cells)}')
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
            row += '  # '
        elif any((x, y) == (mod.x, mod.y) for mod in modifiers):
            row += '  M '
        elif (x, y) in walkable:
            row += '  . '
        else:
            row += '    '
    print(row)

# Now test: walk player toward portal at (25,37)
# First find a path from (25,7) to (25,37)
print(f'\n=== Path exploration ===')
print(f'Current position: ({player.x},{player.y})')

# Check what cells are accessible from player going DOWN
test_moves = 'DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN'
print(f'Testing: {test_moves}')
for i, name in enumerate(test_moves.split()):
    prev_x, prev_y = player.x, player.y
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
    new_x, new_y = player.x, player.y
    moved = (new_x != prev_x or new_y != prev_y)
    print(f'  {i+1}. {name} → ({new_x},{new_y}) {"moved" if moved else "BLOCKED"}')
    if not moved:
        break

# Reset by undoing
print(f'\nUndoing to reset...')
fd = env.step(INT_TO_GA[NAME_TO_INT['UNDO']])
print(f'After UNDO: ({player.x},{player.y}) ghosts={len(lc.rloltuowth)}')

# Try different route: go LEFT first, then DOWN
print(f'\n=== Try LEFT then DOWN ===')
test2 = 'LEFT LEFT LEFT DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN'
for i, name in enumerate(test2.split()):
    prev_x, prev_y = player.x, player.y
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
    new_x, new_y = player.x, player.y
    moved = (new_x != prev_x or new_y != prev_y)
    print(f'  {i+1}. {name} → ({new_x},{new_y}) {"moved" if moved else "BLOCKED"}')
