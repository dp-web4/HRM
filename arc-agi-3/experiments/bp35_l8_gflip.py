#!/usr/bin/env python3
"""L8: G-flip corridor approach — walk along y=32 under E ceiling.
Key insight: push-up to ^-spike kills! Must avoid ^-spike adjacency entirely.
Use G flip DOWN→walk R→G flip UP→land on E ceiling→walk L to x=2."""
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

def track(env, action):
    p1, g1 = state(env)
    if action[0] == 'R': env.step(RIGHT)
    elif action[0] == 'L': env.step(LEFT)
    elif action[0] == 'C': click_act(env, action[1], action[2])
    p2, g2 = state(env)
    dy = f" y:{p1[1]}→{p2[1]}" if p1[1] != p2[1] else ""
    gc = f" grav:{'UP' if g1 else 'DN'}→{'UP' if g2 else 'DN'}" if g1 != g2 else ""
    stuck = " STUCK!" if p1 == p2 and action[0] in ('R','L') else ""
    lvl = env._game.level_index
    won = f" ** LEVEL {lvl}! **" if lvl > 8 else ""
    print(f"  {str(action):14s} ({p1[0]:2d},{p1[1]:2d})→({p2[0]:2d},{p2[1]:2d}){dy}{gc}{stuck}{won}")
    return p2, g2

# ============================================================
print("=== L8: G-flip Corridor Approach ===\n")
env = make_l8_env()
p, g = state(env)
print(f"Start: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# ============================================================
# SETUP PHASE: All clicks before movement
# ============================================================
print("\n--- SETUP ---")

# 1. E spread from (6,31) — create ceiling at y=31
#    Single click: E at (5,31),(7,31),(6,30),(6,32)
print("E(6,31) spread:")
click_act(env, 6, 31)

# Chain west: E(5,31)→E(4,31)→E(3,31)→E(2,31)
for ex in [5, 4, 3]:
    click_act(env, ex, 31)
# Check if E(2,31) was created
print(f"  E(2,31)={sym(env,2,31)}")

# If not, click one more
if sym(env, 2, 31) != 'E':
    click_act(env, 2, 31)
    # Chain created (2,31)? Check west side effects
    # Actually let me just trace what exists
    pass

print("E at y=31:", [x for x in range(-1,12) if sym(env,x,31) == 'E'])
print("E at y=32:", [x for x in range(-1,12) if sym(env,x,32) == 'E'])
print("E at y=30:", [x for x in range(-1,12) if sym(env,x,30) == 'E'])

# 2. Need E ceiling that extends from x=7 down to x=2 at y=31
# E at y=31 should be continuous from x=2 to x=7 for the walk
# Check for gaps
for x in range(2, 8):
    s = sym(env, x, 31)
    print(f"  ({x},31)={s}", end="")
print()

# If there are gaps, we need to fill them via more E chaining
# E chain oscillates: click creates E at adjacent, removing current
# After chain at y=31: odd-indexed x have E, even might not
# Let me also chain at y=30 to provide backup ceiling
print("\nChain E at y=30:")
# E(6,30) should exist from first click
print(f"  E(6,30)={sym(env,6,30)}")
if sym(env, 6, 30) == 'E':
    for ex in [6, 5, 4, 3]:
        if sym(env, ex, 30) == 'E':
            click_act(env, ex, 30)
    print("E at y=30:", [x for x in range(-1,12) if sym(env,x,30) == 'E'])

# E at y=29 chain too
print("\nChain E at y=29:")
for ex in range(6, 1, -1):
    if sym(env, ex, 29) == 'E':
        click_act(env, ex, 29)
        print(f"  Clicked E({ex},29)")
print("E at y=29:", [x for x in range(-1,12) if sym(env,x,29) == 'E'])

# Show the E layout
print("\nE ceiling layout y=28-33:")
for y in range(28, 34):
    row = ""
    for x in range(-1, 12):
        s = sym(env, x, y)
        if s == 'E': row += 'E'
        elif s == '.': row += '.'
        else: row += s
    print(f"  y={y}: {row}")

# 3. E spread from (7,8) for upper ceiling
print("\nE spread from (7,8):")
for ex, ey in [(7,8), (6,8), (5,8), (4,8), (3,8)]:
    click_act(env, ex, ey)
print("E at y=8:", [x for x in range(-1,12) if sym(env,x,8) == 'E'])

# 4. Consume x=0 G blocks (6 clicks = even = same grav)
print("\nConsume x=0 G blocks:")
for gy in [15, 19, 21, 25, 33, 35]:
    click_act(env, 0, gy)
p, g = state(env)
print(f"  After 6 G flips: grav={'UP' if g else 'DN'} (should be UP)")

# 5. Destroy D blocks
print("\nDestroy D blocks:")
click_act(env, 2, 26); click_act(env, 3, 26); click_act(env, 4, 26)
click_act(env, 1, 8)
print(f"  D(1,8)={sym(env,1,8)} D(2,26)={sym(env,2,26)} D(3,26)={sym(env,3,26)} D(4,26)={sym(env,4,26)}")

# 6. Toggle B(2,15) to O (passable)
print("\nToggle B(2,15):")
click_act(env, 2, 15)
print(f"  (2,15)={sym(env,2,15)}")

p, g = state(env)
print(f"\nSetup complete: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Verify critical path cells
print("\nPath verification:")
print(f"  x=0 column clear: (0,15)={sym(env,0,15)} (0,19)={sym(env,0,19)} (0,33)={sym(env,0,33)} (0,35)={sym(env,0,35)}")
print(f"  D(1,8) gone: (1,8)={sym(env,1,8)}")
print(f"  B(2,15)→O: (2,15)={sym(env,2,15)}")
print(f"  E(2,8) ceiling: (2,8)={sym(env,2,8)}")

# ============================================================
# MOVEMENT PHASE
# ============================================================
print("\n--- MOVEMENT ---")

# Step 1: G flip DOWN
print("\n1. G flip DOWN:")
track(env, C(1,1))

# Step 2: Walk R to x=7 (through y=34 gap zone, but grav DOWN = floor at y=38)
print("\n2. Walk R to x=7:")
for i in range(6):
    track(env, R)

p, g = state(env)
# Check: are we at x=7-9 at y=37?
print(f"   At ({p[0]},{p[1]})")

# Step 3: G flip UP — fly through y=34 gap
# At x=7: y=34 is open, so player flies UP through gap
# Need to land on E ceiling at y=31 → position at y=32
print("\n3. G flip UP (fly through gap):")
track(env, C(2,1))
p, g = state(env)
print(f"   At ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Verify alive
p_save = p
env.step(LEFT)
p2 = state(env)[0]
if p2 != p_save:
    print(f"   ALIVE! (L→({p2[0]},{p2[1]}))")
    env.step(UNDO)
else:
    print(f"   DEAD!")
    print_grid(env, (p[1]-5, p[1]+5))
    sys.exit(1)

# Step 4: Walk L along E ceiling to x=2
# Need continuous E ceiling at some y above player
# Player should be at y=32 with E at y=31 above
print(f"\n4. Walk L to x=2 under E ceiling:")
for i in range(10):
    p1, g1 = state(env)
    # Check what's at (p1[0]-1, p1[1])
    lx, ly = p1[0]-1, p1[1]
    ls = sym(env, lx, ly)
    # Check ceiling at target
    ceil = "?"
    for cy in range(ly-1, ly-5, -1):
        cs = sym(env, lx, cy)
        if cs != '.':
            ceil = f"({lx},{cy})={cs}"
            break

    env.step(LEFT)
    p2, g2 = state(env)
    moved = p2 != p1
    vert = f" y:{p1[1]}→{p2[1]}" if p1[1] != p2[1] else ""
    print(f"   ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}){vert} [{ls}] ceil:{ceil}")

    if not moved:
        print(f"   STUCK at ({p2[0]},{p2[1]})")
        break
    if p2[0] <= 2:
        break

p, g = state(env)
print(f"   At ({p[0]},{p[1]})")

# Step 5: From wherever we are, need to reach (2,9)
# The path: x=2 column, from current y up through y=26 (D destroyed)
# and y=15 (B toggled to O) to E ceiling at y=8
# With grav UP: player at y=32 or whatever → falls UP through all open cells
# to E(2,8) or whatever stops them
print(f"\n5. Check x=2 column above player:")
px, py = p
for y in range(py-1, 5, -1):
    s = sym(env, px, y)
    if s != '.':
        print(f"   ({px},{y})={s}")

# How far can we go? With grav UP at (2,32):
# Falls through y=31 (E?), y=30, ..., y=26 (D destroyed), ..., y=19 (^-spike)!
# We need E ceiling at x=2 to catch the player before y=19 spike

# Check E at x=2
print(f"\n   E at x=2:")
for y in range(8, 35):
    if sym(env, 2, y) == 'E':
        print(f"   E(2,{y})")

print(f"\nFull grid state:")
print_grid(env, (6, 38))
