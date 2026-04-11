#!/usr/bin/env python3
"""Debug L6 - test path that avoids v-spike at (8,24).

Key discovery: v-spikes KILL when player lands adjacent with gravity pointing at spike.
The player died at (8,23) from v-spike at (8,24) with gravity DOWN.

New path: go UP through (8,15) then to (9,5) via x=9 shaft, then flip DOWN
to (9,26), walk left to (3,27), flip UP to gem.

Question: does ^-spike at (9,4) also kill with gravity UP? Testing this.
"""
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
    ok = execute_level(env, sol, f"L{i}")
    print(f"L{i}: {'OK' if ok else 'FAIL'} -> level {env._game.level_index}")
    if not ok:
        sys.exit(1)

print(f"\n=== L6 (level {env._game.level_index}) ===")
pos, grav = state(env)
print(f"Start: {pos} grav_up={grav}")

# Path: avoid (8,23) by going UP through shaft first
# R,R,R → (6,19)
# C(6,21) toggle O→B
# C(0,10) flip DN → (6,20)
# R → (7,20)
# C(0,15) flip UP → (7,20) stays [wall at (7,19)]
# R → (8,20) fall UP → (8,15) [wall at (8,14)]
# R → (9,15) fall UP → (9,5) [^spike at (9,4)]  ← TEST: alive?
# C(0,20) flip DN → (9,26) [wall at (9,27)]
# L,L,L,L,L → walk to (4,27) via falling at x=4
# L → (3,27)
# C(0,25) flip UP → gem (3,25)!

steps = [
    ("R",        lambda: env.step(RIGHT),        "→ (4,19)"),
    ("R",        lambda: env.step(RIGHT),        "→ (5,19)"),
    ("R",        lambda: env.step(RIGHT),        "→ (6,19)"),
    ("C(6,21)",  lambda: click_act(env, 6, 21),  "toggle O→B"),
    ("C(0,10)",  lambda: click_act(env, 0, 10),  "flip DN → (6,20)"),
    ("R",        lambda: env.step(RIGHT),        "→ (7,20)"),
    ("C(0,15)",  lambda: click_act(env, 0, 15),  "flip UP → (7,20)"),
    ("R",        lambda: env.step(RIGHT),        "→ (8,15) via fall UP"),
    ("R",        lambda: env.step(RIGHT),        "→ (9,5) via fall UP - SPIKE TEST"),
    ("C(0,20)",  lambda: click_act(env, 0, 20),  "flip DN → (9,26)"),
    ("L",        lambda: env.step(LEFT),         "→ (8,26)"),
    ("L",        lambda: env.step(LEFT),         "→ (7,26)"),
    ("L",        lambda: env.step(LEFT),         "→ (6,26)"),
    ("L",        lambda: env.step(LEFT),         "→ (5,26)"),
    ("L",        lambda: env.step(LEFT),         "→ fall (4,27)"),
    ("L",        lambda: env.step(LEFT),         "→ (3,27)"),
    ("C(0,25)",  lambda: click_act(env, 0, 25),  "flip UP → gem!"),
]

for i, (name, action, desc) in enumerate(steps):
    old_pos, old_grav = state(env)
    fd = action()
    new_pos, new_grav = state(env)
    lvl = env._game.level_index

    grav_ch = ""
    if old_grav != new_grav:
        grav_ch = f" grav {'UP' if old_grav else 'DN'}->{'UP' if new_grav else 'DN'}"
    moved = "STUCK!" if old_pos == new_pos and 'flip' not in desc and 'toggle' not in desc else ""
    print(f"  {i+1:2d}. {name:12s}: ({old_pos[0]:2d},{old_pos[1]:2d}) -> ({new_pos[0]:2d},{new_pos[1]:2d}){grav_ch} {moved}  # {desc}")

    if lvl > 6:
        print(f"     LEVEL ADVANCED! Solved in {i+1} steps!")
        break

    # Check if player died (unresponsive)
    if old_pos == new_pos and name in ('R', 'L'):
        # Try another move to see if alive
        test_pos_before = state(env)[0]
        env.step(UNDO)
        test_pos_after = state(env)[0]
        if test_pos_before == test_pos_after:
            print(f"     DEAD? UNDO didn't work. Player unresponsive at {new_pos}")
            break
        else:
            print(f"     Alive - UNDO worked: {test_pos_before} -> {test_pos_after}")
            # Redo to restore state
            action()

print(f"\nFinal: pos={state(env)[0]} grav_up={state(env)[1]} level={env._game.level_index}")
