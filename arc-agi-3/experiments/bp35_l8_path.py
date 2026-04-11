#!/usr/bin/env python3
"""L8: Complete path exploration — all clicks first, then walk.
Fresh env for each test to avoid death contamination."""
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

def track(env, action, label=""):
    p1, g1 = state(env)
    if action[0] == 'R': env.step(RIGHT)
    elif action[0] == 'L': env.step(LEFT)
    elif action[0] == 'C': click_act(env, action[1], action[2])
    p2, g2 = state(env)
    dy = f" y:{p1[1]}→{p2[1]}" if p1[1] != p2[1] else ""
    gc = f" grav:{'UP' if g1 else 'DN'}→{'UP' if g2 else 'DN'}" if g1 != g2 else ""
    stuck = " STUCK!" if p1 == p2 and action[0] in ('R','L') else ""
    print(f"  {label}{str(action):14s} ({p1[0]:2d},{p1[1]:2d})→({p2[0]:2d},{p2[1]:2d}){dy}{gc}{stuck}")
    return p2, g2

def find_e_positions(env, yr=(0,45), xr=(-1,12)):
    """Find all E positions in given range."""
    es = []
    for y in range(yr[0], yr[1]):
        for x in range(xr[0], xr[1]):
            if sym(env, x, y) == 'E':
                es.append((x,y))
    return es

# ============================================================
# TEST A: G flip approach — grav DOWN to walk past gap, then UP
# ============================================================
print("=== TEST A: G flip to bypass y=34 gap ===\n")
env = make_l8_env()
p, g = state(env)
print(f"Start: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Phase 1: E spread from (6,31) for gap ceiling
print("\nPhase 1: E spread for gap ceiling")
for ex, ey in [(6,31), (6,30), (6,29), (6,28)]:
    track(env, C(ex, ey), "  ")

# Phase 2: E spread from (7,8) for upper ceiling
print("\nPhase 2: E spread for upper ceiling")
for ex, ey in [(7,8), (6,8), (5,8), (4,8), (3,8)]:
    track(env, C(ex, ey), "  ")

print("\n  E at y=7-9:")
for y in [7,8,9]:
    es = [x for x in range(-1, 12) if sym(env,x,y) == 'E']
    print(f"    y={y}: x={es}")

# Phase 3: G flip DOWN
print("\nPhase 3: G flip → grav DOWN")
track(env, C(1,1), "  ")
p, g = state(env)
print(f"  At ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Phase 4: Walk R with grav DOWN (floor at y=38)
print("\nPhase 4: Walk R to x=7")
for i in range(6):
    track(env, R, f"  R{i+1}: ")
p, g = state(env)
print(f"  At ({p[0]},{p[1]})")

# Phase 5: G flip UP — fly through gap at x=7
print("\nPhase 5: G flip UP at x=7 (y=34 gap open)")
print(f"  x=7 column y=27-37:")
for y in range(27, 38):
    print(f"    (7,{y}): {sym(env,7,y)}")
track(env, C(2,1), "  ")
p, g = state(env)
print(f"  At ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
print_grid(env, (max(0,p[1]-5), p[1]+5))

# Check if alive
env.step(LEFT)
p2, _ = state(env)
if p2 != p:
    print(f"  ALIVE! Walk L to ({p2[0]},{p2[1]})")
    env.step(UNDO)
else:
    env.step(RIGHT)
    p3, _ = state(env)
    if p3 != p:
        print(f"  ALIVE! Walk R to ({p3[0]},{p3[1]})")
        env.step(UNDO)
    else:
        print(f"  DEAD at ({p[0]},{p[1]})")

# ============================================================
# TEST B: Same but at x=6 (also has y=34 gap open)
# Also test: does horizontal walk into spike-ceiling cell kill?
# ============================================================
print("\n\n=== TEST B: G flip at x=6 + walk under spike test ===\n")
env = make_l8_env()
p, g = state(env)

# E spread for gap
for ex, ey in [(6,31), (6,30), (6,29), (6,28)]:
    click_act(env, ex, ey)

# E spread for upper
for ex, ey in [(7,8), (6,8), (5,8), (4,8), (3,8)]:
    click_act(env, ex, ey)

# G flip DOWN
click_act(env, 1, 1)
p, g = state(env)
print(f"After G flip: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Walk R to x=6
for i in range(5):
    env.step(RIGHT)
p, g = state(env)
print(f"At x=6: ({p[0]},{p[1]})")

# G flip UP at x=6
click_act(env, 2, 1)
p, g = state(env)
print(f"G flip UP at x=6: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Show surroundings
print_grid(env, (max(0,p[1]-5), min(45,p[1]+5)))

# Check alive
p_before = p
env.step(LEFT)
p2, _ = state(env)
alive = p2 != p_before
if alive:
    print(f"ALIVE! Walk L to ({p2[0]},{p2[1]})")
    env.step(UNDO)
else:
    print(f"POSSIBLY DEAD at ({p[0]},{p[1]})")

# ============================================================
# TEST C: E spread from (6,31) + walk R from (3,35)
# (Same as working Test A from bp35_l8_nav.py)
# Then test: from (5,28) after push-up, can we walk L to x=4?
# ============================================================
print("\n\n=== TEST C: E push-up then navigate left ===\n")
env = make_l8_env()

# E spread for gap
for ex, ey in [(6,31), (6,30), (6,29), (6,28)]:
    click_act(env, ex, ey)

# E spread for upper ceiling
for ex, ey in [(7,8), (6,8), (5,8), (4,8), (3,8)]:
    click_act(env, ex, ey)

# Consume all x=0 G blocks (6 clicks, even = same grav)
for gy in [15, 19, 21, 25, 33, 35]:
    click_act(env, 0, gy)

# Destroy D blocks at y=26
click_act(env, 2, 26)
click_act(env, 3, 26)
click_act(env, 4, 26)

p, g = state(env)
print(f"After setup: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Show E positions
print(f"\nE positions:")
es = find_e_positions(env)
for e in sorted(es):
    print(f"  E{e}")

print(f"\n  Grid y=26-35:")
print_grid(env, (26, 36))

# Walk R into gap
print("\nWalk R:")
track(env, R, "")
track(env, R, "")
p, g = state(env)
print(f"Landed: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# E push-up sequence
print("\nE push-up:")
for i in range(6):
    p, g = state(env)
    # Find E directly above
    above_y = None
    for y in range(p[1]-1, p[1]-10, -1):
        s = sym(env, p[0], y)
        if s == 'E':
            above_y = y
            break
        elif s not in ('.', 'O', 'c'):
            break
    if above_y is None:
        print(f"  No E above ({p[0]},{p[1]})")
        break
    track(env, C(p[0], above_y), f"  push{i+1}: ")

p, g = state(env)
print(f"\nAfter push-up: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Show E around player
print(f"\nE and grid near player:")
print_grid(env, (p[1]-3, p[1]+5))

# Try walking L
print(f"\nWalk L attempts:")
for i in range(5):
    p_before = state(env)[0]
    # Check what's left
    lx = p_before[0] - 1
    ly = p_before[1]
    print(f"  ({lx},{ly}) = {sym(env, lx, ly)}", end="")

    # If E, click it first to clear
    if sym(env, lx, ly) == 'E':
        print(f" → clicking E({lx},{ly})")
        track(env, C(lx, ly), "    ")
        # Check again
        print(f"  ({lx},{ly}) after click = {sym(env, lx, ly)}")
    else:
        print()

    track(env, L, f"  L{i+1}: ")
    p2, g2 = state(env)
    if p2 == p_before:
        print(f"  STUCK! Check above: ({p2[0]},{p2[1]-1}) = {sym(env, p2[0], p2[1]-1)}")
        break

p, g = state(env)
print(f"\nFinal: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
print_grid(env, (p[1]-5, p[1]+5))

# ============================================================
# TEST D: From (3,35), test horizontal walk into spike-ceiling
# Walk LEFT to x=2, where ceiling is W(2,34). Safe.
# Check: can player survive at (2,28) with ^(2,19) far above?
# ============================================================
print("\n\n=== TEST D: Walk LEFT + spike adjacency test ===\n")
env = make_l8_env()
p, g = state(env)
print(f"Start: ({p[0]},{p[1]})")

# Walk LEFT from (3,35) → (2,35)
track(env, L, "  ")
p, g = state(env)
print(f"At ({p[0]},{p[1]}). Ceiling above: (2,{p[1]-1})={sym(env,2,p[1]-1)}")

# Walk LEFT again → (1,35)? x=1 at y=35 is W
track(env, L, "  ")
p, g = state(env)
