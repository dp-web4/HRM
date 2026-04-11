#!/usr/bin/env python3
"""Test L6 solution - try centered clicks and debug step 26 specifically."""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
CLICK = GameAction.ACTION6

def click_act(env, gx, gy):
    """Click with cell-center coordinates."""
    engine = env._game.oztjzzyqoek
    cam_y = engine.camera.rczgvgfsfb[1]
    return env.step(CLICK, data={"x": gx*6 + 3, "y": gy*6 + 3 - cam_y})

def state(env):
    engine = env._game.oztjzzyqoek
    p = engine.twdpowducb
    return tuple(p.qumspquyus), engine.vivnprldht

def cam_y(env):
    return env._game.oztjzzyqoek.camera.rczgvgfsfb[1]

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
]

arcade = Arcade()
env = arcade.make('bp35-0a0ad940')
fd = env.reset()

for i, sol in enumerate(sols):
    ok = execute_level(env, sol)
    if not ok:
        print(f"L{i} FAIL"); sys.exit(1)
print("L0-L5 solved")

print(f"\n=== L6 (centered clicks) ===")

steps = [
    # Phase 1
    ("R",        lambda: env.step(RIGHT)),
    ("R",        lambda: env.step(RIGHT)),
    ("R",        lambda: env.step(RIGHT)),
    ("C(6,21)",  lambda: click_act(env, 6, 21)),
    ("C(0,19)",  lambda: click_act(env, 0, 19)),
    ("R",        lambda: env.step(RIGHT)),
    ("C(0,20)",  lambda: click_act(env, 0, 20)),
    ("R",        lambda: env.step(RIGHT)),
    ("L",        lambda: env.step(LEFT)),
    ("L",        lambda: env.step(LEFT)),
    ("L",        lambda: env.step(LEFT)),
    ("L",        lambda: env.step(LEFT)),
    # Phase 2
    ("C(0,14)",  lambda: click_act(env, 0, 14)),  # G flip DN
    ("C(4,15)",  lambda: click_act(env, 4, 15)),
    ("C(0,16)",  lambda: click_act(env, 0, 16)),  # G flip UP
    ("L",        lambda: env.step(LEFT)),
    ("L",        lambda: env.step(LEFT)),
    # Phase 3
    ("R",        lambda: env.step(RIGHT)),
    ("C(0,9)",   lambda: click_act(env, 0, 9)),   # G flip DN
    ("R",        lambda: env.step(RIGHT)),
    ("C(4,9)",   lambda: click_act(env, 4, 9)),
    ("C(0,10)",  lambda: click_act(env, 0, 10)),  # G flip UP
    ("C(5,10)",  lambda: click_act(env, 5, 10)),
    ("R",        lambda: env.step(RIGHT)),
    # After arriving at (5,8): toggle + G flip with dummy L between
    ("C(5,10)",  lambda: click_act(env, 5, 10)),  # toggle back to B
    ("L_nop",    lambda: env.step(LEFT)),          # blocked, acts as tick
    ("C(0,9)",   lambda: click_act(env, 0, 9)),   # G flip DN
    ("C(6,9)",   lambda: click_act(env, 6, 9)),
    ("R",        lambda: env.step(RIGHT)),
    ("C(6,9)",   lambda: click_act(env, 6, 9)),
    ("C(0,10)",  lambda: click_act(env, 0, 10)),  # G flip UP
    ("R",        lambda: env.step(RIGHT)),
    # Phase 4
    ("C(7,8)",   lambda: click_act(env, 7, 8)),
    ("C(0,6)",   lambda: click_act(env, 0, 6)),   # G flip DN
    ("R",        lambda: env.step(RIGHT)),
    ("R",        lambda: env.step(RIGHT)),
    # Phase 5
    ("L",        lambda: env.step(LEFT)),
    ("L",        lambda: env.step(LEFT)),
    ("C(0,25)",  lambda: click_act(env, 0, 25)),  # G flip UP
    ("L",        lambda: env.step(LEFT)),
    ("L",        lambda: env.step(LEFT)),
    ("L",        lambda: env.step(LEFT)),
    ("L",        lambda: env.step(LEFT)),
    ("C(0,24)",  lambda: click_act(env, 0, 24)),  # G flip DN → gem
]

for i, (name, action) in enumerate(steps):
    old_pos, old_grav = state(env)
    cy = cam_y(env)
    fd = action()
    new_pos, new_grav = state(env)
    lvl = env._game.level_index

    grav_ch = ""
    if old_grav != new_grav:
        grav_ch = f" grav {'UP' if old_grav else 'DN'}→{'UP' if new_grav else 'DN'}"

    stuck = " STUCK!" if old_pos == new_pos and name in ('R', 'L', 'L_nop') else ""
    gfail = ""
    if name.startswith("C(0,") and old_grav == new_grav:
        gfail = " G-FAIL!"

    print(f"  {i+1:2d}. {name:12s}: ({old_pos[0]:2d},{old_pos[1]:2d})→({new_pos[0]:2d},{new_pos[1]:2d}) cy={cy}{grav_ch}{stuck}{gfail}")

    if lvl > 6:
        print(f"\n  L6 SOLVED in {i+1} steps!")
        break

    if stuck and name not in ('L_nop',):
        print(f"  FAILED at step {i+1}")
        break

print(f"\nFinal: pos={state(env)[0]} grav={state(env)[1]} level={env._game.level_index}")
