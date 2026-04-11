#!/usr/bin/env python3
"""Test ALL game actions in L7 - are there UP/DOWN/JUMP actions we haven't used?"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

# List all available actions
print("=== All GameAction values ===")
for name in dir(GameAction):
    if name.startswith('ACTION'):
        val = getattr(GameAction, name)
        print(f"  {name} = {val}")

LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
CLICK = GameAction.ACTION6
UNDO = GameAction.ACTION7

def click_act(env, gx, gy):
    engine = env._game.oztjzzyqoek
    cam_y = engine.camera.rczgvgfsfb[1]
    return env.step(CLICK, data={"x": gx*6, "y": gy*6 - cam_y})

def state(env):
    engine = env._game.oztjzzyqoek
    p = engine.twdpowducb
    return tuple(p.qumspquyus), engine.vivnprldht

def execute_level(env, moves):
    old_level = env._game.level_index
    for m in moves:
        if m[0] == 'R': env.step(RIGHT)
        elif m[0] == 'L': env.step(LEFT)
        elif m[0] == 'C':
            engine = env._game.oztjzzyqoek
            cy = engine.camera.rczgvgfsfb[1]
            env.step(CLICK, data={"x": m[1]*6, "y": m[2]*6 - cy})
        if env._game.level_index > old_level:
            return True
    for _ in range(10):
        env.step(LEFT)
        if env._game.level_index > old_level:
            return True
    return False

R = ('R',)
L = ('L',)
def C(x, y): return ('C', x, y)

sols = [
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

arcade = Arcade()
env = arcade.make('bp35-0a0ad940')
fd = env.reset()

for i, sol in enumerate(sols):
    ok = execute_level(env, sol)
    if not ok:
        print(f"L{i} FAIL"); sys.exit(1)
print("L0-L6 solved, now on L7\n")

pos, grav = state(env)
print(f"Start: pos={pos} grav={'UP' if grav else 'DN'}")

# Test all actions (ACTION1 through ACTION8 or whatever exists)
print(f"\n=== Testing all actions from ({pos[0]},{pos[1]}) ===")
actions_to_test = []
for name in sorted(dir(GameAction)):
    if name.startswith('ACTION'):
        actions_to_test.append((name, getattr(GameAction, name)))

for name, action in actions_to_test:
    if name in ('ACTION6',):  # Skip click (needs data)
        continue
    pos_before, grav_before = state(env)
    try:
        env.step(action)
        pos_after, grav_after = state(env)
        moved = pos_before != pos_after
        grav_changed = grav_before != grav_after
        print(f"  {name}: ({pos_before[0]},{pos_before[1]})→({pos_after[0]},{pos_after[1]}) grav {'UP' if grav_after else 'DN'} {'MOVED!' if moved else ''} {'GRAV CHANGED!' if grav_changed else ''}")
        if moved or grav_changed:
            env.step(UNDO)
    except Exception as e:
        print(f"  {name}: ERROR - {e}")

# Also test ACTION6 (click) without data and with various data
print(f"\n=== Testing click variants ===")
pos_before, grav_before = state(env)
try:
    env.step(GameAction.ACTION6)  # no data
    pos_after, grav_after = state(env)
    print(f"  Click no data: ({pos_before[0]},{pos_before[1]})→({pos_after[0]},{pos_after[1]})")
except Exception as e:
    print(f"  Click no data: ERROR - {e}")

# Test: does ACTION1 = UP exist?
print(f"\n=== Testing UP/DOWN actions specifically ===")
for act_name, act in [('ACTION1', GameAction.ACTION1), ('ACTION2', GameAction.ACTION2)]:
    pos_before, grav_before = state(env)
    env.step(act)
    pos_after, grav_after = state(env)
    if pos_before != pos_after:
        print(f"  {act_name}: MOVED {pos_before}→{pos_after}")
        # Test further
        for i in range(5):
            p1, _ = state(env)
            env.step(act)
            p2, _ = state(env)
            print(f"    {act_name} again: {p1}→{p2}")
            if p1 == p2:
                break
        # Undo all
        for _ in range(6):
            env.step(UNDO)
    else:
        print(f"  {act_name}: no movement from {pos_before}")

# Check: what about ACTION5?
print(f"\n=== ACTION5 test ===")
pos_before, grav_before = state(env)
try:
    env.step(GameAction.ACTION5)
    pos_after, grav_after = state(env)
    print(f"  ACTION5: ({pos_before[0]},{pos_before[1]})→({pos_after[0]},{pos_after[1]}) grav {'UP' if grav_after else 'DN'}")
    if pos_before != pos_after:
        env.step(UNDO)
except Exception as e:
    print(f"  ACTION5: ERROR - {e}")

# Test: stepping on different cells (can we walk through E entities?)
print(f"\n=== Walking through level - all directions from start ===")
# Reset to known position
pos, _ = state(env)
print(f"  Current: {pos}")

# Walk left as far as possible
print("  Walking LEFT:")
for i in range(10):
    p1, _ = state(env)
    env.step(LEFT)
    p2, _ = state(env)
    if p1 == p2:
        print(f"    Blocked at {p2} after {i} steps")
        break
    print(f"    {p1}→{p2}")
# Undo
for _ in range(10):
    env.step(UNDO)

# Walk right as far as possible
print("  Walking RIGHT:")
for i in range(10):
    p1, _ = state(env)
    env.step(RIGHT)
    p2, _ = state(env)
    if p1 == p2:
        print(f"    Blocked at {p2} after {i} steps")
        break
    print(f"    {p1}→{p2}")
    if p1[1] != p2[1]:
        print(f"    Y changed! Likely fell through gap")
# Undo all
for _ in range(10):
    env.step(UNDO)

pos_final, _ = state(env)
print(f"\n  Final: {pos_final}")
