#!/usr/bin/env python3
"""L7: Test undocumented mechanics - click on walls/spikes/empty, step-up behavior."""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
CLICK = GameAction.ACTION6
UNDO = GameAction.ACTION7

R = ('R',)
L = ('L',)
def C(x, y): return ('C', x, y)

L0_L6_SOLS = [
    [R,R,R,R, C(7,19), C(4,16), L,L,L, C(4,15), C(4,12), R, C(5,9), L,L],
    [R,R,R,R,R, C(8,36), C(8,35), L,L, C(5,29),L, C(4,29),L, C(3,29),L, C(2,29),L, C(2,28), R,R,R, C(5,24), C(5,23), L,L, C(3,20), C(3,17), C(3,16), C(4,16),R, C(5,16),R, C(6,16),R, C(7,16),R, C(8,16),R, C(8,15), C(8,14), L, L, L, C(5,9)],
    [C(5,28),R,R,R,C(6,27), C(5,23),C(4,23),C(3,23),L,L,L,L, R, C(5,17),C(6,17),C(5,18),C(6,18),R,R,R,R, C(6,12),C(5,12),C(4,12),C(3,12),L,L,L,L, C(5,7),R,R,R,R],
    [C(3,17),C(7,23),C(7,24),C(5,7), L, R,R, C(3,23), R,R, C(5,23), L,L,L, C(4,31)],
    [R,R,R,R, C(7,9),C(8,9),C(9,9), C(8,12), C(8,29), R,R, L,L,L,L],
    [R,R,R,R,R, C(4,31), L,L,L,L,L,L],
    [R,R,R, C(6,21), C(0,19), R, C(0,20), R, L, L,L,L,
     C(0,14), C(4,15), C(0,16), L, L,
     R, C(0,9), R, C(4,9), C(0,10), C(5,10), R, C(5,10), C(0,8), C(6,9), R, C(6,9), C(0,11), R,
     C(7,8), C(0,6), R, R,
     L, L, C(0,25), L,L,L,L, C(0,24)],
]

def make_l7_env():
    arcade = Arcade()
    env = arcade.make('bp35-0a0ad940')
    env.reset()
    for sol in L0_L6_SOLS:
        old_level = env._game.level_index
        for m in sol:
            if m[0] == 'R': env.step(RIGHT)
            elif m[0] == 'L': env.step(LEFT)
            elif m[0] == 'C':
                engine = env._game.oztjzzyqoek
                cy = engine.camera.rczgvgfsfb[1]
                env.step(CLICK, data={"x": m[1]*6, "y": m[2]*6 - cy})
            if env._game.level_index > old_level:
                break
        if env._game.level_index == old_level:
            for _ in range(10):
                env.step(LEFT)
                if env._game.level_index > old_level:
                    break
    return env

def click_act(env, gx, gy):
    engine = env._game.oztjzzyqoek
    cam_y = engine.camera.rczgvgfsfb[1]
    return env.step(CLICK, data={"x": gx*6, "y": gy*6 - cam_y})

def state(env):
    engine = env._game.oztjzzyqoek
    p = engine.twdpowducb
    return tuple(p.qumspquyus), engine.vivnprldht

def check_grid(env, x, y):
    engine = env._game.oztjzzyqoek
    grid = engine.hdnrlfmyrj
    items = grid.jhzcxkveiw(x, y)
    return [(i.name, i.collidable) for i in items] if items else []

# ============================================================
# TEST 1: Click on walls, spikes, empty space
# ============================================================
print("=== TEST 1: Click on various block types ===")
env = make_l7_env()

test_targets = [
    (3, 31, "wall at y=31"),
    (5, 27, "^-spike"),
    (5, 28, "empty space"),
    (5, 30, "empty space below gap"),
    (3, 33, "empty space below player"),
    (0, 32, "left boundary wall"),
    (10, 32, "right boundary wall"),
]

for gx, gy, desc in test_targets:
    items_before = check_grid(env, gx, gy)
    pos_before, grav_before = state(env)
    click_act(env, gx, gy)
    items_after = check_grid(env, gx, gy)
    pos_after, grav_after = state(env)
    changed = items_before != items_after or pos_before != pos_after or grav_before != grav_after
    print(f"  Click ({gx},{gy}) {desc}: {'CHANGED!' if changed else 'no effect'}")
    if changed:
        print(f"    before: {items_before} pos={pos_before}")
        print(f"    after:  {items_after} pos={pos_after}")
        env.step(UNDO)

# ============================================================
# TEST 2: Check if B blocks at y=22 interact with y=31 wall
# ============================================================
print("\n=== TEST 2: B→O toggle side effects ===")
env = make_l7_env()

# Toggle each B block and check if anything else changes
for bx in range(1, 10):
    # Check neighbors before toggle
    click_act(env, bx, 22)
    pos, grav = state(env)
    items = check_grid(env, bx, 22)
    # Check if player moved
    if pos != (3, 32):
        print(f"  Toggle ({bx},22): Player moved to {pos}!")
        env.step(UNDO)
    else:
        env.step(UNDO)  # undo toggle

# ============================================================
# TEST 3: Check game internals for move mechanics
# ============================================================
print("\n=== TEST 3: Game object inspection ===")
env = make_l7_env()
game = env._game
engine = game.oztjzzyqoek

# Check if there's a move list or step size
print("  Game attributes of interest:")
for attr in dir(game):
    if not attr.startswith('_'):
        try:
            val = getattr(game, attr)
            if not callable(val) and not isinstance(val, (dict, list)):
                if isinstance(val, (int, float, bool, str)):
                    print(f"    game.{attr} = {val}")
        except:
            pass

# Check player attributes more carefully
player = engine.twdpowducb
print("\n  Player state attributes:")
for attr in sorted(dir(player)):
    if not attr.startswith('_'):
        try:
            val = getattr(player, attr)
            if not callable(val):
                if isinstance(val, (int, float, bool, str, tuple)):
                    print(f"    .{attr} = {val}")
                elif isinstance(val, list) and len(val) < 5:
                    print(f"    .{attr} = {val}")
        except:
            pass

# ============================================================
# TEST 4: Check the game source code for move mechanics
# ============================================================
print("\n=== TEST 4: Game code inspection ===")
import inspect
# Check the game class
game_class = type(game)
print(f"  Game class: {game_class}")
print(f"  Game class file: {getattr(game_class, '__module__', 'unknown')}")

# Check if there's a perform_action method we can inspect
if hasattr(game, 'perform_action'):
    try:
        src = inspect.getsource(game.perform_action)
        # Print first 40 lines
        lines = src.split('\n')[:40]
        for line in lines:
            print(f"    {line}")
    except:
        print("    (cannot inspect perform_action source)")

# ============================================================
# TEST 5: Try toggling B to create new floor, then check fall
# ============================================================
print("\n=== TEST 5: Toggle B(5,22)→O→B mid-air test ===")
env = make_l7_env()

# Idea: what if we toggle B(5,22) to O, then toggle it BACK to B?
# Does the player "know" about the change?
click_act(env, 5, 22)  # O
items = check_grid(env, 5, 22)
print(f"  (5,22) after toggle: {items}")
click_act(env, 5, 22)  # B again
items = check_grid(env, 5, 22)
print(f"  (5,22) after retoggle: {items}")

# ============================================================
# TEST 6: Check E entity - can it be toggled like B?
# ============================================================
print("\n=== TEST 6: E entity re-click test ===")
env = make_l7_env()

items_before = check_grid(env, 3, 18)
print(f"  E(3,18) before: {items_before}")

click_act(env, 3, 18)
items_after = check_grid(env, 3, 18)
print(f"  E(3,18) after click: {items_after}")

click_act(env, 3, 18)
items_after2 = check_grid(env, 3, 18)
print(f"  E(3,18) after 2nd click: {items_after2}")

env.step(UNDO)
env.step(UNDO)
items_restored = check_grid(env, 3, 18)
print(f"  E(3,18) after 2 UNDOs: {items_restored}")

# ============================================================
# TEST 7: Walk left, then try to fall up at x=1 (outside spike zone?)
# ============================================================
print("\n=== TEST 7: x=1 exploration ===")
env = make_l7_env()

# Walk to x=1
env.step(LEFT); env.step(LEFT)
pos, grav = state(env)
print(f"  At: {pos}")

# x=1 at y=31: wall. x=1 at y=32: open. x=1 at y=0: wall (boundary).
# What's the column at x=1?
print(f"  Column x=1:")
for y in range(0, 36):
    items = check_grid(env, 1, y)
    if items:
        short = items[0][0][:3]
        print(f"    (1,{y:2d}): {short} col={items[0][1]}")

# ============================================================
# TEST 8: Read the game source to find movement logic
# ============================================================
print("\n=== TEST 8: Game source file ===")
import importlib
# The game class is loaded from environment_files/bp35/0a0ad940/bp35.py
game_file = os.path.join(os.path.dirname(__file__),
    '..', '.local', 'lib', 'python3.10', 'site-packages',
    'arcengine', 'base_game.py')
# Actually, let's find the bp35.py file
import glob
bp35_files = glob.glob('/home/sprout/.local/lib/python3.10/site-packages/arc_agi_3/**/bp35.py', recursive=True)
if not bp35_files:
    bp35_files = glob.glob('/tmp/**/bp35.py', recursive=True)
if not bp35_files:
    # Check the cache
    bp35_files = glob.glob('/home/sprout/**/*bp35*', recursive=False)
print(f"  BP35 game files found: {bp35_files[:5]}")

# Also check where the game was loaded from
print(f"  Game module: {type(game).__module__}")
mod = sys.modules.get(type(game).__module__)
if mod:
    print(f"  Module file: {getattr(mod, '__file__', 'N/A')}")

# Check the engine for step/movement methods
print("\n  Engine methods related to movement:")
for attr in sorted(dir(engine)):
    if not attr.startswith('_'):
        val = getattr(engine, attr)
        if callable(val):
            try:
                sig = inspect.signature(val)
                if len(str(sig)) < 50:
                    print(f"    engine.{attr}{sig}")
            except:
                pass
