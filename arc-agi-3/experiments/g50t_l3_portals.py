#!/usr/bin/env python3
"""Deep dive into L3 portal/swap mechanics."""
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

# Search all attributes of lc for swap gates or portal-related objects
print('\n=== Level container attributes ===')
for attr in sorted(dir(lc)):
    if attr.startswith('_'):
        continue
    try:
        val = getattr(lc, attr)
        if callable(val):
            continue
        t = type(val).__name__
        if t in ('int', 'float', 'bool', 'str', 'NoneType'):
            continue
        if isinstance(val, (list, tuple)):
            if len(val) > 0:
                item_types = set(type(v).__name__ for v in val)
                print(f'  lc.{attr} = [{len(val)}] types={item_types}')
                for i, v in enumerate(val):
                    if hasattr(v, 'x'):
                        print(f'    [{i}]: ({v.x},{v.y}) type={type(v).__name__}')
        elif isinstance(val, dict):
            if len(val) > 0:
                print(f'  lc.{attr} = dict({len(val)})')
        elif isinstance(val, set):
            if len(val) > 0:
                print(f'  lc.{attr} = set({len(val)})')
        else:
            print(f'  lc.{attr} = {t}')
    except:
        pass

# Search game entities for crfcpstubm class
print('\n=== Searching for crfcpstubm (swap gates) ===')
# Check level sprites
level = game._Game__current_level if hasattr(game, '_Game__current_level') else None
if level:
    print(f'Level: {level}')
    for attr in sorted(dir(level)):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(level, attr)
            if isinstance(val, (list, tuple)) and len(val) > 0:
                item_types = set(type(v).__name__ for v in val)
                if any(t not in ('int', 'float', 'bool', 'str') for t in item_types):
                    print(f'  level.{attr} = [{len(val)}] types={item_types}')
        except:
            pass

# Check the game source for how portals connect
# The portals might auto-connect based on shared vgwycxsxjz
mods = lc.hamayflsib
portals = [m for m in mods if type(m).__name__ == 'ulhhdeoyok']
print(f'\n=== Portal analysis ({len(portals)} portals) ===')
for p in portals:
    print(f'Portal at ({p.x},{p.y}):')
    for attr in sorted(dir(p)):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(p, attr)
            if not callable(val):
                print(f'  .{attr} = {val} (type={type(val).__name__})')
        except Exception as e:
            print(f'  .{attr} ERROR: {e}')

# Test: walk to portal and see what happens
print(f'\n=== Walking to portal at (25,37) ===')
player = lc.dzxunlkwxt
print(f'Start: ({player.x},{player.y})')

# Navigate: DOWN DOWN (to 25,19), then LEFT LEFT LEFT (to 7,19), then DOWN DOWN (to 7,31), DOWN (to 7,37)
test = 'DOWN DOWN LEFT LEFT LEFT DOWN DOWN DOWN'
for i, name in enumerate(test.split()):
    prev_x, prev_y = player.x, player.y
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
    moved = (player.x != prev_x or player.y != prev_y)
    print(f'  {i+1}. {name} → ({player.x},{player.y}) {"moved" if moved else "BLOCKED"}')

# From (7,37), try to reach portal at (25,37)
print(f'\nFrom ({player.x},{player.y}), going RIGHT toward portal:')
test2 = 'RIGHT RIGHT RIGHT'
for i, name in enumerate(test2.split()):
    prev_x, prev_y = player.x, player.y
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
    moved = (player.x != prev_x or player.y != prev_y)
    print(f'  {i+1}. {name} → ({player.x},{player.y}) {"moved" if moved else "BLOCKED"}')

# Now step on portal — what happens?
print(f'\nPlayer on portal? ({player.x},{player.y})')
portal_pos = [(p.x, p.y) for p in portals]
print(f'Portal positions: {portal_pos}')
on_portal = (player.x, player.y) in portal_pos

# Try stepping DOWN from portal
if player.y == 37:
    print(f'\nTrying DOWN from y=37 (should be blocked normally):')
    prev_x, prev_y = player.x, player.y
    fd = env.step(INT_TO_GA[NAME_TO_INT['DOWN']])
    moved = (player.x != prev_x or player.y != prev_y)
    print(f'  DOWN → ({player.x},{player.y}) {"moved" if moved else "BLOCKED"}')
    if moved:
        print(f'  TELEPORTED? From ({prev_x},{prev_y}) to ({player.x},{player.y})')
