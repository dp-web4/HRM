#!/usr/bin/env python3
"""Test L6 solution in game engine with camera debug."""
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
    cam_y = engine.camera.rczgvgfsfb[1]
    sx, sy = gx*6, gy*6 - cam_y
    return env.step(CLICK, data={"x": sx, "y": sy})

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

print(f"\n=== L6 ===")

# Define solution with explicit G block y targets
# Use G blocks close to player's y for reliability
steps = [
    # Phase 1
    ("R",        lambda: env.step(RIGHT)),       # 1
    ("R",        lambda: env.step(RIGHT)),       # 2
    ("R",        lambda: env.step(RIGHT)),       # 3
    ("C(6,21)",  lambda: click_act(env, 6, 21)), # 4  toggle
    ("C(0,19)",  lambda: click_act(env, 0, 19)), # 5  G flip
    ("R",        lambda: env.step(RIGHT)),       # 6
    ("C(0,20)",  lambda: click_act(env, 0, 20)), # 7  G flip
    ("R",        lambda: env.step(RIGHT)),       # 8
    ("L",        lambda: env.step(LEFT)),        # 9
    ("L",        lambda: env.step(LEFT)),        # 10
    ("L",        lambda: env.step(LEFT)),        # 11
    ("L",        lambda: env.step(LEFT)),        # 12
    # Phase 2
    ("C(0,13)",  lambda: click_act(env, 0, 13)), # 13 G flip
    ("C(4,15)",  lambda: click_act(env, 4, 15)), # 14 toggle
    ("C(0,17)",  lambda: click_act(env, 0, 17)), # 15 G flip
    ("L",        lambda: env.step(LEFT)),        # 16
    ("L",        lambda: env.step(LEFT)),        # 17
    # Phase 3
    ("R",        lambda: env.step(RIGHT)),       # 18
    ("C(0,9)",   lambda: click_act(env, 0, 9)),  # 19 G flip (use y=9 near player at y=8)
    ("R",        lambda: env.step(RIGHT)),       # 20
    ("C(4,9)",   lambda: click_act(env, 4, 9)),  # 21 toggle
    ("C(0,10)",  lambda: click_act(env, 0, 10)), # 22 G flip (y=10 near player y=11)
    ("C(5,10)",  lambda: click_act(env, 5, 10)), # 23 toggle
    ("R",        lambda: env.step(RIGHT)),       # 24
    ("C(5,10)",  lambda: click_act(env, 5, 10)), # 25 toggle
    ("C(0,9)",   lambda: click_act(env, 0, 9)),  # 26 G flip (y=9 near player at y=8)
    ("C(6,9)",   lambda: click_act(env, 6, 9)),  # 27 toggle
    ("R",        lambda: env.step(RIGHT)),       # 28
    ("C(6,9)",   lambda: click_act(env, 6, 9)),  # 29 toggle
    ("C(0,10)",  lambda: click_act(env, 0, 10)), # 30 G flip (y=10 near player y=11)
    ("R",        lambda: env.step(RIGHT)),       # 31
    # Phase 4
    ("C(7,8)",   lambda: click_act(env, 7, 8)),  # 32 toggle
    ("C(0,6)",   lambda: click_act(env, 0, 6)),  # 33 G flip (y=6 near player y=4)
    ("R",        lambda: env.step(RIGHT)),       # 34
    ("R",        lambda: env.step(RIGHT)),       # 35
    # Phase 5
    ("L",        lambda: env.step(LEFT)),        # 36
    ("L",        lambda: env.step(LEFT)),        # 37
    ("C(0,25)",  lambda: click_act(env, 0, 25)), # 38 G flip (y=25 near player y=26)
    ("L",        lambda: env.step(LEFT)),        # 39
    ("L",        lambda: env.step(LEFT)),        # 40
    ("L",        lambda: env.step(LEFT)),        # 41
    ("L",        lambda: env.step(LEFT)),        # 42
    ("C(0,24)",  lambda: click_act(env, 0, 24)), # 43 G flip (y=24 near player y=23)
]

for i, (name, action) in enumerate(steps):
    old_pos, old_grav = state(env)
    cy = cam_y(env)

    # Calculate screen coords for clicks
    screen_info = ""
    if name.startswith("C("):
        parts = name[2:-1].split(",")
        gx, gy = int(parts[0]), int(parts[1])
        sx, sy = gx*6, gy*6 - cy
        screen_info = f" screen=({sx},{sy})"

    fd = action()
    new_pos, new_grav = state(env)
    lvl = env._game.level_index

    grav_ch = ""
    if old_grav != new_grav:
        grav_ch = f" grav {'UP' if old_grav else 'DN'}→{'UP' if new_grav else 'DN'}"

    stuck = " STUCK!" if old_pos == new_pos and name in ('R', 'L') else ""

    print(f"  {i+1:2d}. {name:12s}: ({old_pos[0]:2d},{old_pos[1]:2d})→({new_pos[0]:2d},{new_pos[1]:2d}) cam_y={cy}{grav_ch}{screen_info}{stuck}")

    if lvl > 6:
        print(f"\n  L6 SOLVED in {i+1} steps!")
        break

    if old_pos == new_pos and name in ('R', 'L'):
        print(f"  STUCK at step {i+1}!")
        break

    if name.startswith("C(0,") and old_grav == new_grav:
        print(f"  WARNING: G flip didn't work! cam_y={cy}{screen_info}")

print(f"\nFinal: pos={state(env)[0]} grav={state(env)[1]} level={env._game.level_index}")
