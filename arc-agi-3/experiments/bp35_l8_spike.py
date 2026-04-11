#!/usr/bin/env python3
"""L8: Test spike lethality on horizontal walk + E click coordinate debugging."""
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

def click_raw(env, sx, sy):
    """Click using raw screen coordinates."""
    return env.step(CLICK, data={"x": sx, "y": sy})

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

def is_alive(env):
    """Test if player is alive by trying L and R."""
    p = state(env)[0]
    env.step(LEFT)
    p2 = state(env)[0]
    if p2 != p:
        env.step(UNDO)
        return True
    env.step(RIGHT)
    p3 = state(env)[0]
    if p3 != p:
        env.step(UNDO)
        return True
    return False

# ============================================================
# TEST 1: E click debugging — verify coordinates
# ============================================================
print("=== TEST 1: E Click Coordinate Debug ===\n")
env = make_l8_env()
p, g = state(env)
print(f"Start: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Click E(6,31) — original E
engine = env._game.oztjzzyqoek
cam_y = engine.camera.rczgvgfsfb[1]
print(f"Camera Y: {cam_y}")
print(f"E(6,31) screen coords: x={6*6}, y={31*6 - cam_y}")
print(f"Before: (6,31)={sym(env,6,31)}")
click_act(env, 6, 31)
print(f"After click E(6,31): (6,31)={sym(env,6,31)}")
print(f"Spread check: (5,31)={sym(env,5,31)} (7,31)={sym(env,7,31)} (6,30)={sym(env,6,30)} (6,32)={sym(env,6,32)}")

# Undo
env.step(UNDO)
print(f"\nUndo: (6,31)={sym(env,6,31)}")

# Now do E spread for gap ceiling and push-up to (5,28)
print("\n--- Setup: E spread + walk + push-up ---")
for ex, ey in [(6,31), (6,30), (6,29), (6,28)]:
    click_act(env, ex, ey)

# Walk into gap
env.step(RIGHT)  # (3,35)→(4,35)
env.step(RIGHT)  # (4,35)→(5,32) via E ceiling
p, g = state(env)
print(f"After walk into gap: ({p[0]},{p[1]})")

# Push-up
for ey in [31, 30, 29, 28]:
    click_act(env, 5, ey)
    p, g = state(env)
    print(f"  Push-up E(5,{ey}): ({p[0]},{p[1]})")

print(f"\nAt ({p[0]},{p[1]}). Alive: {is_alive(env)}")

# Now try to click E(6,28) with detailed coordinate info
cam_y = env._game.oztjzzyqoek.camera.rczgvgfsfb[1]
print(f"\nCamera Y now: {cam_y}")
print(f"E(6,28) screen: x={6*6}, y={28*6 - cam_y}")
print(f"Before click: (6,28)={sym(env,6,28)}")

# Try clicking with precise coordinates
click_act(env, 6, 28)
print(f"After click_act(6,28): (6,28)={sym(env,6,28)}")

# Check spread results
print(f"  (5,28)={sym(env,5,28)} (7,28)={sym(env,7,28)} (6,27)={sym(env,6,27)} (6,29)={sym(env,6,29)}")

# Try raw pixel click
env.step(UNDO)  # undo the click
print(f"\nAfter undo click: (6,28)={sym(env,6,28)}")

# Try clicking with +3 offset (center of cell)
cam_y = env._game.oztjzzyqoek.camera.rczgvgfsfb[1]
sx = 6*6 + 3
sy = 28*6 - cam_y + 3
print(f"Raw click ({sx},{sy}) for E(6,28):")
click_raw(env, sx, sy)
print(f"  (6,28)={sym(env,6,28)}")

env.step(UNDO)

# Try clicking E(4,28) — the problematic one
print(f"\n--- E(4,28) click test ---")
print(f"(4,28)={sym(env,4,28)}")
cam_y = env._game.oztjzzyqoek.camera.rczgvgfsfb[1]
print(f"Camera Y: {cam_y}")
sx4 = 4*6
sy4 = 28*6 - cam_y
print(f"Screen coords: ({sx4},{sy4})")
click_act(env, 4, 28)
print(f"After click: (4,28)={sym(env,4,28)}")
print(f"  (3,28)={sym(env,3,28)} (5,28)={sym(env,5,28)} (4,27)={sym(env,4,27)} (4,29)={sym(env,4,29)}")

# Show what entities are at (4,28)
items = env._game.oztjzzyqoek.hdnrlfmyrj.jhzcxkveiw(4, 28)
print(f"Entities at (4,28): {[(i.name, id(i)) for i in items]}")

# Show ALL entities at y=28
print(f"\nAll entities at y=28:")
for x in range(-1, 12):
    items = env._game.oztjzzyqoek.hdnrlfmyrj.jhzcxkveiw(x, 28)
    if items:
        print(f"  ({x},28): {items[0].name}")

# ============================================================
# TEST 2: Horizontal walk under spike — direct test
# From (5,28) alive via push-up, can we reach (4,28)?
# Alternative: can we reach (4,27) where y=27 is open (no spike)?
# ============================================================
print("\n\n=== TEST 2: Navigate from push-up position ===\n")
env = make_l8_env()

# Spread E for gap ceiling
for ex, ey in [(6,31), (6,30), (6,29), (6,28)]:
    click_act(env, ex, ey)

# Walk R + push-up
env.step(RIGHT); env.step(RIGHT)
for ey in [31, 30, 29, 28]:
    click_act(env, 5, ey)

p, g = state(env)
print(f"At ({p[0]},{p[1]}). Alive: {is_alive(env)}")

# Show nearby grid
print_grid(env, (25, 33))

# Find ALL E positions near player
print("\nE positions y=26-32:")
for y in range(26, 33):
    es = [x for x in range(-1, 12) if sym(env, x, y) == 'E']
    if es: print(f"  y={y}: {es}")

# What if we UNDO one push-up to be at (5,29), then walk L?
print("\n--- Undo to (5,29) and try walking L ---")
env.step(UNDO)
p, g = state(env)
print(f"After undo: ({p[0]},{p[1]})")

# At (5,29): what's at (4,29)?
print(f"(4,29)={sym(env,4,29)}")
print(f"(4,28)={sym(env,4,28)}")

# Walk L
env.step(LEFT)
p, g = state(env)
print(f"Walk L: ({p[0]},{p[1]}). Alive: {is_alive(env)}")

# If at (4,29): what's above?
if p[0] == 4:
    for y in range(p[1]-5, p[1]+1):
        print(f"  ({p[0]},{y}): {sym(env, p[0], y)}")

# ============================================================
# TEST 3: G-flip approach — get to (7,32) via Test B path
# Then chain E westward to x=3,2 at y=31 to create ceiling
# Walk L along y=32 under E ceiling
# ============================================================
print("\n\n=== TEST 3: G-flip + E chain walk ===\n")
env = make_l8_env()

# Phase 0: Consume x=0 G blocks and destroy D at y=26
for gy in [15, 19, 21, 25, 33, 35]:
    click_act(env, 0, gy)
for dx in [2, 3, 4]:
    click_act(env, dx, 26)
click_act(env, 1, 8)  # Also destroy D(1,8) while we're at it

# Phase 1: E spread from (7,8) for upper ceiling
for ex, ey in [(7,8), (6,8), (5,8), (4,8), (3,8)]:
    click_act(env, ex, ey)

# Phase 2: E spread from (6,31) + chain westward
# Single click creates ceiling at y=31 for x=5,7 and (6,30),(6,32)
click_act(env, 6, 31)
# Now chain E(5,31) → (4,31) → (3,31) → (2,31)
click_act(env, 5, 31)
click_act(env, 4, 31)
click_act(env, 3, 31)

print("E at y=31 after chain:")
for x in range(-1, 12):
    if sym(env, x, 31) == 'E': print(f"  E({x},31)")

# Also chain at y=30 for more ceiling
# E(6,30) exists from first click
click_act(env, 6, 30)
click_act(env, 5, 30)
click_act(env, 4, 30)
click_act(env, 3, 30)

print("E at y=30 after chain:")
for x in range(-1, 12):
    if sym(env, x, 30) == 'E': print(f"  E({x},30)")

# Show the setup
print("\nGrid y=28-36:")
print_grid(env, (28, 37))

p, g = state(env)
print(f"\nSetup done. ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Phase 3: G flip DOWN → walk R to x=8
click_act(env, 1, 1)
p, g = state(env)
print(f"\nG flip DOWN: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Walk R to x=8
for i in range(7):
    p1 = state(env)[0]
    env.step(RIGHT)
    p2 = state(env)[0]
    print(f"  R: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]})")

# Phase 4: G flip UP → fly through gap
click_act(env, 2, 1)
p, g = state(env)
print(f"\nG flip UP at x={p[0]}: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
alive = is_alive(env)
print(f"Alive: {alive}")

if alive:
    # Walk L through E ceiling corridor at y=32
    print("\nWalk L along E ceiling:")
    for i in range(10):
        p1 = state(env)[0]
        lx = p1[0] - 1
        ly = p1[1]
        lsym = sym(env, lx, ly)
        print(f"  ({lx},{ly})={lsym}", end="")

        # Check ceiling at target
        for cy in range(ly-1, ly-5, -1):
            cs = sym(env, lx, cy)
            if cs != '.':
                print(f" ceil=({lx},{cy})={cs}", end="")
                break

        env.step(LEFT)
        p2, g2 = state(env)
        alive2 = True
        if p2 == p1:
            alive2 = is_alive(env)
        print(f" → ({p2[0]},{p2[1]}){'' if alive2 else ' DEAD!'}")
        if p2 == p1 or not alive2:
            break

    p, g = state(env)
    print(f"\nFinal: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
    print_grid(env, (p[1]-5, p[1]+5))
