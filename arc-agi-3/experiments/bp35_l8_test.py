#!/usr/bin/env python3
"""L8: Test key mechanics — G flip from start, E spreading, shaft navigation."""
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

def sym(env, x, y):
    items = env._game.oztjzzyqoek.hdnrlfmyrj.jhzcxkveiw(x, y)
    return SYM.get(items[0].name, '?') if items else '.'

def print_grid(env, yr, xr=(-1,12)):
    pp = state(env)[0]
    for y in range(yr[0], yr[1]):
        row = ""
        for x in range(xr[0], xr[1]):
            if (x, y) == pp: row += "P"
            else: row += sym(env, x, y)
        print(f"    y={y:3d}: {row}")

# ============================================================
print("=== L8 Tests ===\n")
env = make_l8_env()
p, g = state(env)
print(f"Start: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} level={env._game.level_index}")

# ============================================================
# TEST 1: Consume all x=0 G blocks + observe player position
# ============================================================
print("\n--- TEST 1: Consume x=0 G blocks ---")
x0_gs = [(0,15), (0,19), (0,21), (0,25), (0,33), (0,35)]
for gx, gy in x0_gs:
    p1, g1 = state(env)
    click_act(env, gx, gy)
    p2, g2 = state(env)
    print(f"  G({gx},{gy}): ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}) grav:{'UP' if g1 else 'DN'}→{'UP' if g2 else 'DN'}")

p, g = state(env)
print(f"\n  After 6 G clicks: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
print(f"  x=0 column (should be clear):")
for y in [7, 14, 15, 16, 19, 21, 25, 33, 35, 38, 40]:
    print(f"    (0,{y:2d}): {sym(env,0,y)}")

# Click one y=1 G to get grav DOWN
click_act(env, 1, 1)
p, g = state(env)
print(f"\n  After G(1,1): ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Undo ALL G clicks
for _ in range(7): env.step(UNDO)
p, g = state(env)
print(f"  After 7 UNDOs: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# ============================================================
# TEST 2: E spreading at (6,31) — create ceiling for shaft
# ============================================================
print("\n--- TEST 2: E spreading at (6,31) ---")
# Spread E upward to create ceiling for y=34 gap crossing
clicks = [(6,31), (6,30), (6,29), (6,28)]
for ex, ey in clicks:
    click_act(env, ex, ey)
print_grid(env, (26, 35))

# Show E positions
print("\n  E positions by row:")
for y in range(27, 33):
    es = []
    for x in range(-1, 12):
        if sym(env, x, y) == 'E':
            es.append(x)
    if es: print(f"    y={y}: {es}")

# Undo E spreads
for _ in range(4): env.step(UNDO)

# ============================================================
# TEST 3: Can we cross y=34 gap with E ceiling?
# Walk R into gap → should land safely at y=29
# ============================================================
print("\n--- TEST 3: Cross y=34 gap with E ceiling ---")
# First spread E for ceiling
for ex, ey in [(6,31), (6,30), (6,29), (6,28)]:
    click_act(env, ex, ey)

# Walk R from (3,35) into the gap
p, g = state(env)
print(f"  Start: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
for i in range(4):
    p1, g1 = state(env)
    env.step(RIGHT)
    p2, g2 = state(env)
    dy = f" y:{p1[1]}→{p2[1]}" if p1[1] != p2[1] else ""
    print(f"  R{i+1}: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}){dy}")
    if p1 == p2: break

p, g = state(env)
print(f"\n  Position: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
if p[1] < 32:
    print("  SUCCESS: crossed the gap!")
    print_grid(env, (26, 36))
else:
    print("  FAILED or DEAD")
    print_grid(env, (26, 36))

# Undo walks
for _ in range(4): env.step(UNDO)

# ============================================================
# TEST 4: Navigate upward from y=29 area
# After crossing gap, try to reach y=26 (D blocks)
# ============================================================
print("\n--- TEST 4: Navigate from y=29 upward ---")
# Re-walk into gap
for _ in range(4): env.step(RIGHT)
p, g = state(env)
print(f"  At: ({p[0]},{p[1]})")

# Walk left from the shaft to x=4-2 area (below y=27 where x=2-4 are clear)
for i in range(5):
    p1 = state(env)[0]
    env.step(LEFT)
    p2 = state(env)[0]
    print(f"  L: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]})")
    if p1 == p2: break

p, g = state(env)
print(f"\n  At: ({p[0]},{p[1]})")

# From x=2-4 at y=29 area: walk UP? Check ceiling
# x=4 at y=27: open (no spike). x=4 at y=26: D block (solid).
# So at (4,29): ceiling ... let's check what's above
print(f"\n  Column above player at x={p[0]}:")
for y in range(p[1]-5, p[1]+1):
    print(f"    ({p[0]},{y}): {sym(env, p[0], y)}")

# Can we destroy D(4,26) to open a path?
# First, what's above D(4,26)?
print(f"\n  x=4 column y=19-27:")
for y in range(19, 28):
    print(f"    (4,{y}): {sym(env,4,y)}")

# Destroy D blocks at y=26 to open shaft
print(f"\n  Destroying D(4,26), D(3,26), D(2,26)...")
click_act(env, 4, 26)
click_act(env, 3, 26)
click_act(env, 2, 26)
print_grid(env, (24, 30), (-1, 12))

# Now from (4,29): ceiling should be further up
p, g = state(env)
print(f"\n  At: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Walk to x=4 if not there
while p[0] > 4:
    env.step(LEFT)
    p, g = state(env)
while p[0] < 4:
    env.step(RIGHT)
    p, g = state(env)

print(f"  At: ({p[0]},{p[1]})")
# Now check: from (4,29), can we push up using E?
# Or just walk left to x=3 or x=2 where ceiling might be different

# Walk to x=3
env.step(LEFT)
p, g = state(env)
print(f"  Walk L to: ({p[0]},{p[1]})")

# Walk to x=2
env.step(LEFT)
p, g = state(env)
print(f"  Walk L to: ({p[0]},{p[1]})")

# From x=2: ceiling is (2,26) was D, now destroyed. Above: (2,25) open, ..., (2,19) ^-spike
# With D destroyed: falls from y=27 through y=26 to... what stops them?
print(f"\n  x=2 above player:")
for y in range(14, p[1]+1):
    print(f"    (2,{y}): {sym(env,2,y)}")

# ============================================================
# TEST 5: Check if we can navigate the x=9 corridor
# ============================================================
print("\n--- TEST 5: x=9 corridor ---")
# x=9 appears open at y=17-24 (no wall or spike at y=17-18 for x=9)
print("  x=9 column:")
for y in range(15, 28):
    print(f"    (9,{y}): {sym(env,9,y)}")

# x=10 column (right wall):
print("  x=10 column:")
for y in range(15, 28):
    print(f"    (10,{y}): {sym(env,10,y)}")
