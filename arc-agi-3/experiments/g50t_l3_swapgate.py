#!/usr/bin/env python3
"""Find swap gates and signal chains in L3."""
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

# Check all sprites in the level for swap gate tag
level = lc.current_level
print(f'\n=== Level sprites by tag ===')

# Import the tag enum
g50t_module = type(lc)
# Get the evgpfjbmvf enum
import importlib
spec = importlib.util.spec_from_file_location("g50t", "/home/sprout/ai-workspace/SAGE/arc-agi-3/experiments/environment_files/g50t/5849a774/g50t.py")
g50t = importlib.util.module_from_spec(spec)

# Instead, just check known tags
tags_to_check = ['ugfrhsffov', 'hyztgkifvb', 'mxmeqbrmab', 'mcqullpcwz', 'dryrnuvljg']
for tag_name in tags_to_check:
    try:
        sprites = level.get_sprites_by_tag(tag_name)
        print(f'  Tag {tag_name}: {len(sprites)} sprites')
        for s in sprites:
            print(f'    ({s.x},{s.y}) vis={s.is_visible}')
    except:
        # Try with enum
        pass

# Try to enumerate all tags in the enum
print(f'\n=== Trying all tags ===')
# The evgpfjbmvf enum has these known values
known_tags = {
    'lrtamslcit': 'goal',
    'tddbvyfbvq': 'waypoints',
    'qeixtoeawu': 'enemies',
    'ugfrhsffov': 'portals',
    'hyztgkifvb': 'swap_gates',
    'mxmeqbrmab': 'pressure_pads',
    'mcqullpcwz': 'obstacles',
    'dryrnuvljg': 'path_constrainers',
    'inaylmmhhy': 'cursor',
    'ekfhaifjds': 'indicators',
}

for tag, desc in known_tags.items():
    try:
        sprites = level.get_sprites_by_tag(tag)
        if sprites:
            print(f'  {desc} ({tag}): {len(sprites)}')
            for s in sprites:
                print(f'    ({s.x},{s.y})')
    except Exception as e:
        print(f'  {desc} ({tag}): ERROR {e}')

# Check the path constrainers to understand signal routing
print(f'\n=== Path constrainer details ===')
# Reconstruct from setup
mods = lc.hamayflsib  # pressure pads + portals
for i, mod in enumerate(mods):
    cls = type(mod).__name__
    if cls == 'lqtxaumfed':
        pc = mod.nexhtmlmxh
        if pc:
            print(f'mod[{i}] ({mod.x},{mod.y}) → constrainer at ({pc.x},{pc.y})')
            print(f'  outputs ({len(pc.ytztewxdin)}):')
            for j, out in enumerate(pc.ytztewxdin):
                out_cls = type(out).__name__
                print(f'    [{j}]: ({out.x},{out.y}) type={out_cls}')
                if out_cls == 'crfcpstubm':
                    print(f'      SWAP GATE FOUND!')
                    if out.lzcbbrzwdn:
                        print(f'      portal_a: ({out.lzcbbrzwdn.x},{out.lzcbbrzwdn.y})')
                    if out.esbtgcjrfp:
                        print(f'      portal_b: ({out.esbtgcjrfp.x},{out.esbtgcjrfp.y})')

# Check lc.hjvvibklzv (animation queue)
print(f'\n=== Animation queue ===')
print(f'hjvvibklzv: {len(lc.hjvvibklzv)} items')

# Direct test: walk player to mod[0] at (49,25) and see what happens
print(f'\n=== Testing: walk to mod[0] at (49,25) ===')
print(f'Player start: ({player.x},{player.y})')

# Route: (25,7) → RIGHT×4 → (49,7) → DOWN→(49,13) → DOWN→(49,19) → DOWN→(49,25)
test = 'RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN'
for i, name in enumerate(test.split()):
    prev_x, prev_y = player.x, player.y
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
    moved = (player.x != prev_x or player.y != prev_y)
    print(f'  {i+1}. {name} → ({player.x},{player.y}) {"moved" if moved else "BLOCKED"}')

print(f'Player at mod[0]? ({player.x},{player.y}) == (49,25): {(player.x, player.y) == (49, 25)}')

# Check if portals got entities after stepping on mod[0]
portals = [m for m in mods if type(m).__name__ == 'ulhhdeoyok']
for i, p in enumerate(portals):
    print(f'Portal ({p.x},{p.y}): vbqvjbxkfm={p.vbqvjbxkfm}')
