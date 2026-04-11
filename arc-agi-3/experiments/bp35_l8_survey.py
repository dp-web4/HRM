#!/usr/bin/env python3
"""L8: Detailed survey and pathfinding."""
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

L0_L7_SOLS = [
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
    [C(2,29), C(3,29), C(4,29), C(5,29), C(6,29),
     C(3,18), C(4,18), C(5,18), C(6,18),
     C(7,25), C(8,22),
     R, R, R, R, R, L, L, R, R, L, L, L,
     C(5,19), C(5,18), C(5,17),
     C(6,17), R, C(7,17), R, C(8,17), R,
     C(5,2), C(8,18), R],
]

def make_l8_env():
    arcade = Arcade()
    env = arcade.make('bp35-0a0ad940')
    env.reset()
    for sol in L0_L7_SOLS:
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
       'lrpkmzabbfa':'G', 'hzusueifitk':'v', 'qclfkhjnaac':'D',
       'aknlbboysnc':'c'}

env = make_l8_env()
engine = env._game.oztjzzyqoek
grid = engine.hdnrlfmyrj
print(f"=== L8 Survey ===")
print(f"Level: {env._game.level_index}")

p, g = state(env)
print(f"Player: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Full entity scan
entities = {}
for y in range(-5, 60):
    for x in range(-5, 20):
        items = grid.jhzcxkveiw(x, y)
        if items:
            entities[(x,y)] = items[0].name

# Categorize
by_type = {}
for (x,y), name in entities.items():
    short = SYM.get(name, '?')
    by_type.setdefault(short, []).append((x,y))

print(f"\nEntity counts:")
for t, positions in sorted(by_type.items()):
    print(f"  {t}: {len(positions)}")

print(f"\nEntity positions:")
for t in ['G', 'B', 'E', 'D', '*', 'O', '^', 'v']:
    if t in by_type:
        pos = sorted(by_type[t])
        print(f"  {t}: {pos}")

# Print grid with coordinates
print(f"\nGrid (with entity key: W=wall, G=grav, B=toggle_solid, O=toggle_pass,")
print(f"  E=spreader, D=destroyable, ^=up_spike, v=down_spike, *=gem, c=passable_deco):")
pp = p
x_min = min(x for x,y in entities) - 1
x_max = max(x for x,y in entities) + 2
y_min = min(y for x,y in entities) - 1
y_max = max(y for x,y in entities) + 2

# Header
print(f"       {''.join(str(x%10) for x in range(x_min, x_max))}")
for y in range(y_min, y_max):
    row = ""
    has = False
    for x in range(x_min, x_max):
        if (x, y) == pp:
            row += "P"; has = True
        elif (x,y) in entities:
            row += SYM.get(entities[(x,y)], '?'); has = True
        else:
            row += "."
    if has:
        print(f"  y={y:3d}: {row}")

# Column analysis for key positions
print(f"\n--- Column analysis ---")
for x in [2, 3, 8, 9]:
    print(f"\n  Column x={x}:")
    for y in range(y_min, y_max):
        if (x,y) in entities:
            print(f"    ({x},{y:2d}): {SYM.get(entities[(x,y)], '?')} ({entities[(x,y)][:8]})")

# Test basic movement
print(f"\n--- Movement test ---")
for i in range(8):
    p1, g1 = state(env)
    env.step(RIGHT)
    p2, g2 = state(env)
    print(f"  R: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]})")
    if p1 == p2: break

for _ in range(8): env.step(UNDO)

# Test left movement
p, _ = state(env)
print(f"\n  Back at: ({p[0]},{p[1]})")
for i in range(4):
    p1, g1 = state(env)
    env.step(LEFT)
    p2, g2 = state(env)
    print(f"  L: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]})")
    if p1 == p2: break

for _ in range(4): env.step(UNDO)

# Test G clicks
print(f"\n--- G block test ---")
p, g = state(env)
print(f"  Start: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Click first G at (0,35)
click_act(env, 0, 35)
p, g = state(env)
print(f"  After G(0,35): ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
env.step(UNDO)

# Click G at (0,33)
click_act(env, 0, 33)
p, g = state(env)
print(f"  After G(0,33): ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
env.step(UNDO)
