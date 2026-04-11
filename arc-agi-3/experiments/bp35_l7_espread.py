#!/usr/bin/env python3
"""L7: Test E entity spreading mechanic and build a bridge."""
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

def print_e_map(env, y_range=(27,33), x_range=(0,11)):
    """Print E entity map for a region."""
    engine = env._game.oztjzzyqoek
    grid = engine.hdnrlfmyrj
    sym = {'xcjjwqfzjfe':'W', 'yuuqpmlxorv':'B', 'oonshderxef':'O',
           'ubhhgljbnpu':'^', 'fjlzdjxhant':'*', 'etlsaqqtjvn':'E',
           'lrpkmzabbfa':'G'}
    pp = tuple(engine.twdpowducb.qumspquyus)
    for y in range(y_range[0], y_range[1]):
        row = ""
        for x in range(x_range[0], x_range[1]):
            if (x, y) == pp:
                row += "P"
            else:
                items = grid.jhzcxkveiw(x, y)
                if items:
                    row += sym.get(items[0].name, '?')
                else:
                    row += "."
        print(f"    y={y:3d}: {row}")

# ============================================================
# TEST: E spreading from (2,29)
# ============================================================
print("=== E Spreading Test ===")
env = make_l7_env()
pos, grav = state(env)
print(f"  Start: {pos} grav={'UP' if grav else 'DN'}")

print(f"\n  Before spreading:")
print_e_map(env, (26, 33))

# Spread E rightward along y=29
spread_targets = [(2,29), (3,29), (4,29), (5,29), (6,29), (7,29), (8,29)]
for i, (ex, ey) in enumerate(spread_targets):
    click_act(env, ex, ey)
    pos, grav = state(env)
    print(f"\n  After click E({ex},{ey}) [step {i+1}]:")
    print_e_map(env, (27, 33))

# Now check: can we walk right and survive?
print(f"\n  Player: {state(env)[0]}")
print(f"\n=== Walking test with E bridge ===")

# Walk right from (3,32)
for i in range(8):
    p1, g1 = state(env)
    env.step(RIGHT)
    p2, g2 = state(env)
    dy = f" (y: {p1[1]}→{p2[1]})" if p1[1] != p2[1] else ""
    alive = p1 != p2 or i == 0
    print(f"  R{i+1}: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}){dy} {'ALIVE' if p1 != p2 else 'BLOCKED/DEAD'}")
    if p1 == p2:
        break

pos_final, grav_final = state(env)
print(f"\n  Final: {pos_final} grav={'UP' if grav_final else 'DN'}")

# Show map around player
print(f"\n  Grid around player:")
print_e_map(env, (26, 35), (0, 11))
