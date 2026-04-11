#!/usr/bin/env python3
"""Debug step 26: why does G flip fail at (5,8)?"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

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

def cam_info(env):
    c = env._game.oztjzzyqoek.camera.rczgvgfsfb
    return c[0], c[1]

def check_grid_at(env, x, y):
    """Check what entity is at (x,y) in the game grid."""
    engine = env._game.oztjzzyqoek
    grid = engine.hdnrlfmyrj
    items = grid.jhzcxkveiw(x, y)
    if items:
        return [(item.name, getattr(item, 'visible', '?')) for item in items]
    return []

def execute_level(env, moves):
    old_level = env._game.level_index
    for m in moves:
        if m[0] == 'R': env.step(RIGHT)
        elif m[0] == 'L': env.step(LEFT)
        elif m[0] == 'C': click_act(env, m[1], m[2])
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
]

arcade = Arcade()
env = arcade.make('bp35-0a0ad940')
fd = env.reset()

for i, sol in enumerate(sols):
    ok = execute_level(env, sol)
    if not ok:
        print(f"L{i} FAIL"); sys.exit(1)
print("L0-L5 solved")

# Execute steps 1-24 of L6 solution
print("\n=== Executing steps 1-24 ===")
pre_steps = [
    lambda: env.step(RIGHT),       # 1
    lambda: env.step(RIGHT),       # 2
    lambda: env.step(RIGHT),       # 3
    lambda: click_act(env, 6, 21), # 4
    lambda: click_act(env, 0, 19), # 5
    lambda: env.step(RIGHT),       # 6
    lambda: click_act(env, 0, 20), # 7
    lambda: env.step(RIGHT),       # 8
    lambda: env.step(LEFT),        # 9
    lambda: env.step(LEFT),        # 10
    lambda: env.step(LEFT),        # 11
    lambda: env.step(LEFT),        # 12
    lambda: click_act(env, 0, 13), # 13
    lambda: click_act(env, 4, 15), # 14
    lambda: click_act(env, 0, 17), # 15
    lambda: env.step(LEFT),        # 16
    lambda: env.step(LEFT),        # 17
    lambda: env.step(RIGHT),       # 18
    lambda: click_act(env, 0, 9),  # 19
    lambda: env.step(RIGHT),       # 20
    lambda: click_act(env, 4, 9),  # 21
    lambda: click_act(env, 0, 10), # 22
    lambda: click_act(env, 5, 10), # 23
    lambda: env.step(RIGHT),       # 24
]

for i, step in enumerate(pre_steps):
    step()
    pos, grav = state(env)

pos, grav = state(env)
cx, cy = cam_info(env)
print(f"After step 24: pos={pos} grav={'UP' if grav else 'DN'} cam=({cx},{cy})")

# Now we should be at (5,8) grav UP
# Check game state at key positions
print(f"\n=== Grid state at key positions ===")
for check_pos in [(5,7), (5,8), (5,9), (5,10), (5,11), (0,8), (0,9), (0,10)]:
    items = check_grid_at(env, *check_pos)
    print(f"  ({check_pos[0]:2d},{check_pos[1]:2d}): {items}")

# Try different G block positions
print(f"\n=== Testing G flip at different y values ===")
for test_y in [5, 6, 7, 8, 9, 10, 11, 12]:
    # Save state
    pos_before, grav_before = state(env)
    cy = cam_info(env)[1]
    sx, sy = 0*6, test_y*6 - cy

    click_act(env, 0, test_y)
    pos_after, grav_after = state(env)

    flipped = grav_before != grav_after
    print(f"  C(0,{test_y:2d}): screen=({sx},{sy:3d}) grav {'UP' if grav_before else 'DN'}→{'UP' if grav_after else 'DN'} {'FLIPPED!' if flipped else 'no flip'} pos={pos_after}")

    if flipped:
        # Undo the flip
        click_act(env, 0, test_y)
        pos_r, grav_r = state(env)
        print(f"    undo: grav {'UP' if grav_r else 'DN'} pos={pos_r}")

# Check: is the player actually dead?
print(f"\n=== Player liveness check ===")
pos, grav = state(env)
print(f"Current: pos={pos} grav={'UP' if grav else 'DN'}")

env.step(UNDO)
pos2, grav2 = state(env)
print(f"After UNDO: pos={pos2} grav={'UP' if grav2 else 'DN'}")

if pos == pos2:
    print("  Player might be dead!")
else:
    print("  Player alive, UNDO worked")
    # Redo to restore
    env.step(RIGHT)  # redo equivalent
