#!/usr/bin/env python3
"""Test L6 solution - fixed with unique G block usage (consumable!)."""
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
    return env.step(CLICK, data={"x": gx*6, "y": gy*6 - cam_y})

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

print(f"\n=== L6 (unique G blocks) ===")

# G blocks at x=0, y=5-27 (23 total, each consumable/single-use)
# Need 11 flips. Assign unique y values:
# Flip 1: y=19, Flip 2: y=20, Flip 3: y=14, Flip 4: y=16
# Flip 5: y=9,  Flip 6: y=10, Flip 7: y=8,  Flip 8: y=11
# Flip 9: y=6,  Flip 10: y=25, Flip 11: y=24

steps = [
    # Phase 1: Start to platform area
    ("R",        lambda: env.step(RIGHT)),       # 1
    ("R",        lambda: env.step(RIGHT)),       # 2
    ("R",        lambda: env.step(RIGHT)),       # 3  → (6,19)
    ("C(6,21)",  lambda: click_act(env, 6, 21)), # 4  toggle O→B
    ("C(0,19)",  lambda: click_act(env, 0, 19)), # 5  G flip #1 → DN
    ("R",        lambda: env.step(RIGHT)),       # 6  → (7,20)
    ("C(0,20)",  lambda: click_act(env, 0, 20)), # 7  G flip #2 → UP
    ("R",        lambda: env.step(RIGHT)),       # 8  → (8,15)
    ("L",        lambda: env.step(LEFT)),        # 9  → (7,13)
    ("L",        lambda: env.step(LEFT)),        # 10 → (6,13)
    ("L",        lambda: env.step(LEFT)),        # 11 → (5,13)
    ("L",        lambda: env.step(LEFT)),        # 12 → (4,13)

    # Phase 2: Platform to x=2 shaft
    ("C(0,14)",  lambda: click_act(env, 0, 14)), # 13 G flip #3 → DN  (was 0,13)
    ("C(4,15)",  lambda: click_act(env, 4, 15)), # 14 toggle O→B
    ("C(0,16)",  lambda: click_act(env, 0, 16)), # 15 G flip #4 → UP  (was 0,17)
    ("L",        lambda: env.step(LEFT)),        # 16 → (3,15)
    ("L",        lambda: env.step(LEFT)),        # 17 → (2,8)

    # Phase 3: Navigate y=8-11 to reach x=7
    ("R",        lambda: env.step(RIGHT)),       # 18 → (3,8)
    ("C(0,9)",   lambda: click_act(env, 0, 9)),  # 19 G flip #5 → DN
    ("R",        lambda: env.step(RIGHT)),       # 20 → (4,11)
    ("C(4,9)",   lambda: click_act(env, 4, 9)),  # 21 toggle O→B
    ("C(0,10)",  lambda: click_act(env, 0, 10)), # 22 G flip #6 → UP
    ("C(5,10)",  lambda: click_act(env, 5, 10)), # 23 toggle B→O
    ("R",        lambda: env.step(RIGHT)),       # 24 → (5,8)
    ("C(5,10)",  lambda: click_act(env, 5, 10)), # 25 toggle O→B
    ("C(0,8)",   lambda: click_act(env, 0, 8)),  # 26 G flip #7 → DN  *** FIXED ***
    ("C(6,9)",   lambda: click_act(env, 6, 9)),  # 27 toggle B→O
    ("R",        lambda: env.step(RIGHT)),       # 28 → (6,11)
    ("C(6,9)",   lambda: click_act(env, 6, 9)),  # 29 toggle O→B
    ("C(0,11)",  lambda: click_act(env, 0, 11)), # 30 G flip #8 → UP
    ("R",        lambda: env.step(RIGHT)),       # 31 → (7,4)

    # Phase 4: Through x=7 to x=9
    ("C(7,8)",   lambda: click_act(env, 7, 8)),  # 32 toggle O→B
    ("C(0,6)",   lambda: click_act(env, 0, 6)),  # 33 G flip #9 → DN
    ("R",        lambda: env.step(RIGHT)),       # 34 → (8,7)
    ("R",        lambda: env.step(RIGHT)),       # 35 → (9,26)

    # Phase 5: Navigate to gem
    ("L",        lambda: env.step(LEFT)),        # 36 → (8,26)
    ("L",        lambda: env.step(LEFT)),        # 37 → (7,26)
    ("C(0,25)",  lambda: click_act(env, 0, 25)), # 38 G flip #10 → UP
    ("L",        lambda: env.step(LEFT)),        # 39 → (6,23)
    ("L",        lambda: env.step(LEFT)),        # 40 → (5,23)
    ("L",        lambda: env.step(LEFT)),        # 41 → (4,23)
    ("L",        lambda: env.step(LEFT)),        # 42 → (3,23)
    ("C(0,24)",  lambda: click_act(env, 0, 24)), # 43 G flip #11 → DN → GEM!
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

    stuck = " STUCK!" if old_pos == new_pos and name in ('R', 'L') else ""
    gfail = ""
    if name.startswith("C(0,") and old_grav == new_grav:
        gfail = " G-FAIL!"

    print(f"  {i+1:2d}. {name:12s}: ({old_pos[0]:2d},{old_pos[1]:2d})→({new_pos[0]:2d},{new_pos[1]:2d}) cy={cy}{grav_ch}{stuck}{gfail}")

    if lvl > 6:
        print(f"\n  L6 SOLVED in {i+1} steps!")
        break

    if stuck:
        print(f"  FAILED at step {i+1}")
        break
    if gfail:
        print(f"  G flip failed at step {i+1}!")
        # Don't break, keep going to see full state

print(f"\nFinal: pos={state(env)[0]} grav={state(env)[1]} level={env._game.level_index}")
