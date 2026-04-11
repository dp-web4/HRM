#!/usr/bin/env python3
"""Check camera x/y offsets at different player positions in L6."""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
CLICK = GameAction.ACTION6

def click_act(env, gx, gy):
    engine = env._game.oztjzzyqoek
    cam = engine.camera.rczgvgfsfb
    cam_x, cam_y = cam[0], cam[1]
    return env.step(CLICK, data={"x": gx*6 - cam_x, "y": gy*6 - cam_y})

def click_act_old(env, gx, gy):
    """Old version without cam_x adjustment."""
    engine = env._game.oztjzzyqoek
    cam_y = engine.camera.rczgvgfsfb[1]
    return env.step(CLICK, data={"x": gx*6, "y": gy*6 - cam_y})

def state(env):
    engine = env._game.oztjzzyqoek
    p = engine.twdpowducb
    return tuple(p.qumspquyus), engine.vivnprldht

def cam(env):
    engine = env._game.oztjzzyqoek
    c = engine.camera.rczgvgfsfb
    return c[0], c[1]

def execute_level(env, moves):
    old_level = env._game.level_index
    for m in moves:
        if m[0] == 'R': env.step(RIGHT)
        elif m[0] == 'L': env.step(LEFT)
        elif m[0] == 'C': click_act_old(env, m[1], m[2])
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
        print(f"L{i} FAIL")
        sys.exit(1)
print("L0-L5 solved")

# Now test camera in L6
print(f"\n=== L6 Camera Test ===")

# Check cam at start
pos, grav = state(env)
cx, cy = cam(env)
print(f"Start: pos={pos} grav={grav} cam=({cx},{cy})")

# Walk right and check camera
for step in range(6):
    env.step(RIGHT)
    pos, grav = state(env)
    cx, cy = cam(env)
    print(f"After R: pos={pos} cam=({cx},{cy})")

# Now test G flip with and without cam_x
print(f"\n--- Testing G flip at current position ---")
pos, grav = state(env)
cx, cy = cam(env)
print(f"Before: pos={pos} grav={grav} cam=({cx},{cy})")

# Try old click (no cam_x)
click_act_old(env, 0, pos[1])
pos2, grav2 = state(env)
print(f"Old click C(0,{pos[1]}): pos={pos2} grav={grav2} (changed={grav!=grav2})")

if grav == grav2:
    # Try new click (with cam_x)
    click_act(env, 0, pos[1])
    pos3, grav3 = state(env)
    print(f"New click C(0,{pos[1]}): pos={pos3} grav={grav3} (changed={grav!=grav3})")
