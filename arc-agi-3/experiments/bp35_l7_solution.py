#!/usr/bin/env python3
"""L7: Complete solution using E push-up mechanic."""
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

SYM = {'xcjjwqfzjfe':'W', 'yuuqpmlxorv':'B', 'oonshderxef':'O',
       'ubhhgljbnpu':'^', 'fjlzdjxhant':'*', 'etlsaqqtjvn':'E',
       'lrpkmzabbfa':'G'}

def print_grid(env, yr, xr=(0,11)):
    pp = tuple(env._game.oztjzzyqoek.twdpowducb.qumspquyus)
    grid = env._game.oztjzzyqoek.hdnrlfmyrj
    for y in range(yr[0], yr[1]):
        row = ""
        for x in range(xr[0], xr[1]):
            if (x, y) == pp: row += "P"
            else:
                items = grid.jhzcxkveiw(x, y)
                row += SYM.get(items[0].name, '?') if items else "."
        print(f"    y={y:3d}: {row}")

# ============================================================
# L7 SOLUTION
# ============================================================
env = make_l7_env()
print("=== L7 Solution ===\n")

L7_SOL = [
    # Phase 1: E bridge y=29-30 (5 clicks)
    C(2,29), C(3,29), C(4,29), C(5,29), C(6,29),
    # Phase 2: E ceiling y=17-19 (4 clicks)
    C(3,18), C(4,18), C(5,18), C(6,18),
    # Phase 3: B toggles (2 clicks)
    C(7,25), C(8,22),
    # Phase 4: Walk across bridge to x=8 shaft (5 walks)
    R, R, R, R, R,
    # Phase 5: Navigate to (8,21) (4 walks)
    L, L, R, R,
    # Phase 6: Navigate to (5,20) via E ceiling (3 walks)
    L, L, L,
    # Phase 7: Push up (3 clicks)
    C(5,19), C(5,18), C(5,17),
    # Phase 8: Walk east clearing E (3 clicks + 3 walks)
    C(6,17), R, C(7,17), R, C(8,17), R,
    # Phase 9: G flip + B toggle + gem (2 clicks + 1 walk)
    C(5,2), C(8,18), R,
]

step = 0
for move in L7_SOL:
    step += 1
    p1, g1 = state(env)
    if move[0] == 'R': env.step(RIGHT)
    elif move[0] == 'L': env.step(LEFT)
    elif move[0] == 'C': click_act(env, move[1], move[2])
    p2, g2 = state(env)

    dy = f" y:{p1[1]}→{p2[1]}" if p1[1] != p2[1] else ""
    gc = f" grav:{'UP' if g1 else 'DN'}→{'UP' if g2 else 'DN'}" if g1 != g2 else ""
    stuck = " STUCK!" if p1 == p2 and move[0] in ('R', 'L') else ""
    lvl = env._game.level_index
    won = " ** LEVEL WON! **" if lvl > 7 else ""

    print(f"  {step:3d}. {str(move):14s} ({p1[0]:2d},{p1[1]:2d})→({p2[0]:2d},{p2[1]:2d}){dy}{gc}{stuck}{won}")

    if stuck:
        print(f"\n  FAILED at step {step}!")
        print_grid(env, (max(0, p2[1]-5), min(36, p2[1]+5)))
        break

    if lvl > 7:
        print(f"\n  L7 SOLVED in {step} steps!")
        break

if env._game.level_index <= 7:
    p, g = state(env)
    print(f"\n  Final: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} level={env._game.level_index}")
    print_grid(env, (15, 22))
