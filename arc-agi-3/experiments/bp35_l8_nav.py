#!/usr/bin/env python3
"""L8: Navigation tests — getting from y=32 area up past spike barriers."""
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
    dead = ""
    if p1 == p2 and action[0] == 'C':
        dead = " (no move)"
    print(f"  {label}{str(action):14s} ({p1[0]:2d},{p1[1]:2d})→({p2[0]:2d},{p2[1]:2d}){dy}{gc}{stuck}{dead}")
    return p2, g2

# ============================================================
print("=== L8 Navigation Tests ===\n")
env = make_l8_env()
p, g = state(env)
print(f"Start: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} level={env._game.level_index}")

# ============================================================
# TEST A: E spread from (6,31), then E push-up from (5,32)
# ============================================================
print("\n--- TEST A: E push-up approach from gap ---")

# Step 1: Spread E from (6,31) — creates E ceiling
print("Phase 1: E spread")
for ex, ey in [(6,31), (6,30), (6,29), (6,28)]:
    track(env, C(ex, ey), "  ")

# Show E positions
print("\n  E positions after spreading:")
for y in range(27, 33):
    es = [(x, sym(env, x, y)) for x in range(-1, 12) if sym(env, x, y) == 'E']
    if es: print(f"    y={y}: x={[e[0] for e in es]}")

# Step 2: Walk R into gap
print("\nPhase 2: Walk into gap")
for i in range(3):
    p2, g2 = track(env, R, "  ")

# Step 3: Try E push-up from current position
p, g = state(env)
print(f"\n  At ({p[0]},{p[1]}). What's above?")
for y in range(p[1]-5, p[1]+1):
    print(f"    ({p[0]},{y}): {sym(env, p[0], y)}")

# Try clicking the E above player
print("\nPhase 3: E push-up attempts")
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
        # Try walking L to find column with E above
        print(f"  Checking columns for E above:")
        for cx in range(p[0]-1, -2, -1):
            for cy in range(p[1]-1, p[1]-10, -1):
                s = sym(env, cx, cy)
                if s == 'E':
                    print(f"    E at ({cx},{cy})")
                    break
                elif s not in ('.', 'O', 'c'):
                    break
        break
    print(f"  Push-up: clicking E({p[0]},{above_y}) from ({p[0]},{p[1]})")
    track(env, C(p[0], above_y), "  ")

p, g = state(env)
print(f"\n  After push-ups: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
print_grid(env, (26, 36))

# Undo all
actions_a = 4 + 3 + 6  # E spreads + walks + push-ups (max)
for _ in range(20): env.step(UNDO)
p, g = state(env)
print(f"\n  After undo: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# ============================================================
# TEST B: Different E spread pattern — avoid catching player at y=31
# Spread E once (6,31), DON'T chain-spread. Check what (5,31) is.
# ============================================================
print("\n\n--- TEST B: Minimal E spread + controlled approach ---")

# Just click E(6,31) once — creates E at adjacent empties
print("Single E(6,31) click:")
track(env, C(6, 31), "  ")

print("\n  Cells around (6,31):")
for y in range(29, 34):
    row = ""
    for x in range(4, 9):
        row += f" ({x},{y})={sym(env,x,y)}"
    print(f"    {row}")

# Check if (5,31) is now E
s531 = sym(env, 5, 31)
print(f"\n  (5,31) = {s531}")

# Undo
env.step(UNDO)

# ============================================================
# TEST C: Walk R first, THEN spread E — player blocks E creation
# ============================================================
print("\n\n--- TEST C: Walk first, then E spread ---")
print("Walk R to get closer to gap:")
for i in range(2):
    track(env, R, "  ")

p, g = state(env)
print(f"\n  At ({p[0]},{p[1]}). Now spread E(6,31):")
track(env, C(6, 31), "  ")

print("\n  Check y=31 row:")
for x in range(-1, 12):
    s = sym(env, x, 31)
    if s != '.' and s != 'W':
        print(f"    ({x},31) = {s}")

# Now click E(5,31) if it exists, to push upward
p, g = state(env)
if sym(env, 5, 31) == 'E':
    print("\n  E at (5,31)! Try push-up after walking into gap:")
    track(env, R, "  ")  # Walk into gap
    p, g = state(env)
    print(f"  Landed at ({p[0]},{p[1]})")
    # What's above?
    for y in range(p[1]-3, p[1]+1):
        print(f"    ({p[0]},{y}): {sym(env, p[0], y)}")

# Undo all
for _ in range(10): env.step(UNDO)

# ============================================================
# TEST D: Alternative — go RIGHT further to x=9 corridor
# From (3,35), walk R to x=8-9, see where we end up
# ============================================================
print("\n\n--- TEST D: Walk to x=9 area ---")
p, g = state(env)
print(f"Start: ({p[0]},{p[1]})")

# Walk R many steps
for i in range(8):
    p2, g2 = track(env, R, "  ")
    if p2 == state(env)[0] and i > 0:
        break

p, g = state(env)
print(f"\n  Reached ({p[0]},{p[1]}). Grid around:")
print_grid(env, (p[1]-3, p[1]+3), (p[0]-3, p[0]+4))

# Check what's above at x=9
print(f"\n  x=9 column y=15-38:")
for y in range(15, 39):
    s = sym(env, 9, y)
    if s != '.': print(f"    (9,{y}): {s}")

# Undo walks
for _ in range(8): env.step(UNDO)

# ============================================================
# TEST E: Full column scan at key x positions y=15-40
# ============================================================
print("\n\n--- TEST E: Key column scans ---")
for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    solids = []
    for y in range(0, 45):
        s = sym(env, x, y)
        if s != '.':
            solids.append((y, s))
    if solids:
        print(f"  x={x}: {[(y,s) for y,s in solids if y >= 5]}")

# ============================================================
# TEST F: Can player reach x=9 at y=23 from (3,35)?
# Path idea: R,R,R,R,R,R → see if we reach x=9
# Then check if we can walk L/R along y=23-24 area
# ============================================================
print("\n\n--- TEST F: Direct path to x=9 corridor ---")

# First, what about using G to flip gravity at start?
# G(1,1) available — grav UP → grav DOWN
print("G flip test — click G(1,1) from start:")
track(env, C(1,1), "  ")
p, g = state(env)
print(f"  Now at ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# With grav DOWN, player falls... where?
# From (3,35) or wherever they are now
print("  Grid around player:")
print_grid(env, (p[1]-2, p[1]+5))

# Walk R with grav DOWN
print("\n  Walk R with grav DOWN:")
for i in range(8):
    p2, g2 = track(env, R, "  ")
    if p2[0] == p[0] and p2[1] == p[1]:
        p, g = p2, g2
        break
    p, g = p2, g2

print(f"\n  Final: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
print_grid(env, (p[1]-3, p[1]+3))

# Undo all
for _ in range(15): env.step(UNDO)

# ============================================================
# TEST G: Map full grid y=0-45 to understand layout
# ============================================================
print("\n\n--- TEST G: Full grid map ---")
p, g = state(env)
print(f"Player: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
print_grid(env, (0, 45), (-1, 12))
