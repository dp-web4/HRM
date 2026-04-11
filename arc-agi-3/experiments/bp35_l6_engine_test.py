#!/usr/bin/env python3
"""Test L6 solution in actual game engine."""
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

def execute_level(env, moves, name=""):
    old_level = env._game.level_index
    for i, m in enumerate(moves):
        if m[0] == 'R': env.step(RIGHT)
        elif m[0] == 'L': env.step(LEFT)
        elif m[0] == 'C': click_act(env, m[1], m[2])
        if env._game.level_index > old_level:
            return True
    for j in range(10):
        env.step(LEFT)
        if env._game.level_index > old_level:
            return True
    return False

R = ('R',)
L = ('L',)
def C(x, y): return ('C', x, y)

# L0-L5 solutions (verified)
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

# Solve L0-L5
for i, sol in enumerate(sols):
    ok = execute_level(env, sol, f"L{i}")
    print(f"L{i}: {'OK' if ok else 'FAIL'} -> level {env._game.level_index}")
    if not ok:
        sys.exit(1)

print(f"\n=== L6 (level {env._game.level_index}) ===")
pos, grav = state(env)
print(f"Start: {pos} grav_up={grav}")

# L6 solution - 43 steps
# Actions: R, L, C(gx,gy) for clicks (G blocks at x=0, OB blocks at their position)
steps = [
    # Phase 1: Start to platform area
    ("R",        lambda: env.step(RIGHT),        "(4,19)"),
    ("R",        lambda: env.step(RIGHT),        "(5,19)"),
    ("R",        lambda: env.step(RIGHT),        "(6,19) O-pass"),
    ("C(6,21)",  lambda: click_act(env, 6, 21),  "toggle O→B"),
    ("C(0,19)",  lambda: click_act(env, 0, 19),  "G flip DN → (6,20)"),
    ("R",        lambda: env.step(RIGHT),        "(7,20)"),
    ("C(0,20)",  lambda: click_act(env, 0, 20),  "G flip UP → stays"),
    ("R",        lambda: env.step(RIGHT),        "(8,15) fall UP"),
    ("L",        lambda: env.step(LEFT),         "(7,13) fall UP"),
    ("L",        lambda: env.step(LEFT),         "(6,13) O"),
    ("L",        lambda: env.step(LEFT),         "(5,13) O"),
    ("L",        lambda: env.step(LEFT),         "(4,13) O"),

    # Phase 2: Platform to x=2 shaft
    ("C(0,13)",  lambda: click_act(env, 0, 13),  "G flip DN → (4,17)"),
    ("C(4,15)",  lambda: click_act(env, 4, 15),  "toggle O→B"),
    ("C(0,17)",  lambda: click_act(env, 0, 17),  "G flip UP → (4,16)"),
    ("L",        lambda: env.step(LEFT),         "(3,15) fall UP"),
    ("L",        lambda: env.step(LEFT),         "(2,8) fall UP"),

    # Phase 3: Navigate y=8-11 to reach x=7
    ("R",        lambda: env.step(RIGHT),        "(3,8) O"),
    ("C(0,8)",   lambda: click_act(env, 0, 8),   "G flip DN → (3,11)"),
    ("R",        lambda: env.step(RIGHT),        "(4,11)"),
    ("C(4,9)",   lambda: click_act(env, 4, 9),   "toggle O→B"),
    ("C(0,11)",  lambda: click_act(env, 0, 11),  "G flip UP → (4,10)"),
    ("C(5,10)",  lambda: click_act(env, 5, 10),  "toggle B→O"),
    ("R",        lambda: env.step(RIGHT),        "(5,8) fall UP"),
    ("C(5,10)",  lambda: click_act(env, 5, 10),  "toggle O→B"),
    ("C(0,8)",   lambda: click_act(env, 0, 8),   "G flip DN → (5,9)"),
    ("C(6,9)",   lambda: click_act(env, 6, 9),   "toggle B→O"),
    ("R",        lambda: env.step(RIGHT),        "(6,11) fall DN"),
    ("C(6,9)",   lambda: click_act(env, 6, 9),   "toggle O→B"),
    ("C(0,11)",  lambda: click_act(env, 0, 11),  "G flip UP → (6,10)"),
    ("R",        lambda: env.step(RIGHT),        "(7,4) fall UP thru O col"),

    # Phase 4: Through x=7 to x=9
    ("C(7,8)",   lambda: click_act(env, 7, 8),   "toggle O→B"),
    ("C(0,5)",   lambda: click_act(env, 0, 5),   "G flip DN → (7,7)"),
    ("R",        lambda: env.step(RIGHT),        "(8,7)"),
    ("R",        lambda: env.step(RIGHT),        "(9,26) fall DN!"),

    # Phase 5: Navigate to gem
    ("L",        lambda: env.step(LEFT),         "(8,26)"),
    ("L",        lambda: env.step(LEFT),         "(7,26)"),
    ("C(0,26)",  lambda: click_act(env, 0, 26),  "G flip UP → (7,23)"),
    ("L",        lambda: env.step(LEFT),         "(6,23)"),
    ("L",        lambda: env.step(LEFT),         "(5,23)"),
    ("L",        lambda: env.step(LEFT),         "(4,23)"),
    ("L",        lambda: env.step(LEFT),         "(3,23)"),
    ("C(0,23)",  lambda: click_act(env, 0, 23),  "G flip DN → GEM!"),
]

print(f"\nExecuting {len(steps)} steps:")
for i, (name, action, desc) in enumerate(steps):
    old_pos, old_grav = state(env)
    fd = action()
    new_pos, new_grav = state(env)
    lvl = env._game.level_index

    grav_ch = ""
    if old_grav != new_grav:
        grav_ch = f" grav {'UP' if old_grav else 'DN'}->{'UP' if new_grav else 'DN'}"

    moved = ""
    if old_pos == new_pos and name in ('R', 'L'):
        moved = " STUCK!"

    print(f"  {i+1:2d}. {name:12s}: ({old_pos[0]:2d},{old_pos[1]:2d}) → ({new_pos[0]:2d},{new_pos[1]:2d}){grav_ch} {moved}  # {desc}")

    if lvl > 6:
        print(f"\n  LEVEL ADVANCED! L6 solved in {i+1} steps!")
        break

    # Check if player died (stuck on move)
    if old_pos == new_pos and name in ('R', 'L'):
        test_pos = state(env)[0]
        env.step(UNDO)
        test_pos2 = state(env)[0]
        if test_pos == test_pos2:
            print(f"  DEAD? Player unresponsive at {new_pos}")
            break
        else:
            # Redo
            action()

print(f"\nFinal: pos={state(env)[0]} grav_up={state(env)[1]} level={env._game.level_index}")
