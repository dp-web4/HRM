#!/usr/bin/env python3
"""L8: Test x=9 corridor approach — bypass both spike barriers."""
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

# ============================================================
print("=== L8: x=9 Corridor Test ===\n")
env = make_l8_env()
p, g = state(env)
print(f"Start: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} level={env._game.level_index}")

# ============================================================
# TEST 1: Walk R from (3,35) — how far can we go along y=35?
# y=34 wall at x=2-4 and x=8-10. Gap at x=5-7.
# ============================================================
print("\n--- TEST 1: Walk R along y=35 ---")
print("y=34 row (ceiling): ", end="")
for x in range(-1, 12):
    print(f"{sym(env,x,34)}", end="")
print()
print("y=35 row (player): ", end="")
for x in range(-1, 12):
    s = sym(env,x,35)
    if (x,35) == p: print("P", end="")
    else: print(f"{s}", end="")
print()

# Walk R step by step
actions = 0
for i in range(8):
    p1, g1 = state(env)
    env.step(RIGHT)
    actions += 1
    p2, g2 = state(env)
    alive = p2 != p1 or (p2 == p1 and i == 0)  # not necessarily dead
    dy = f" y:{p1[1]}→{p2[1]}" if p1[1] != p2[1] else ""
    stuck = " STUCK!" if p1 == p2 else ""
    print(f"  R{i+1}: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}){dy}{stuck}")
    if p1 == p2 and i > 0:
        # Check if dead
        env.step(LEFT)
        p3, _ = state(env)
        actions += 1
        if p3 == p2:
            print(f"  DEAD at ({p2[0]},{p2[1]})!")
        else:
            print(f"  Alive, wall at R. L→({p3[0]},{p3[1]})")
            env.step(UNDO)
            actions -= 1
        break

# Undo
for _ in range(actions): env.step(UNDO)
p, g = state(env)
print(f"\n  Reset: ({p[0]},{p[1]})")

# ============================================================
# TEST 2: Walk R past gap — but skip x=5-7 by not entering them
# What if there's a wall at x=4→x=5 that blocks?
# Actually, movement is 1 cell at a time — must pass through x=5.
# But y=34 gap at x=5 means falling UP. Need alternate path.
# Can we go BELOW the gap? With grav flip?
# ============================================================
print("\n--- TEST 2: Grav flip to reach x=8-9 from below ---")
print("Click G(1,1) to flip grav → DOWN:")
track(env, C(1,1))
p, g = state(env)

# Now with grav DOWN, walk R
print(f"  At ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
for i in range(8):
    p2, g2 = track(env, R, f"  R{i+1}: ")
    if p2 == state(env)[0] and g2 == state(env)[1]:
        # stuck check
        pp = state(env)[0]
        if pp == p2 and i > 0:
            break
    p, g = p2, g2

p, g = state(env)
print(f"\n  At ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
print_grid(env, (p[1]-3, p[1]+5))

# undo all
for _ in range(20): env.step(UNDO)
p, g = state(env)
print(f"\n  Reset: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# ============================================================
# TEST 3: With grav DOWN from start, what's the floor at y=36-38?
# Walk R with grav DOWN to reach x=8,9 at y=37
# Then flip grav back UP to fly up through x=9 corridor
# ============================================================
print("\n--- TEST 3: Grav DOWN → walk to x=9 → flip UP ---")

# Flip grav DOWN
track(env, C(1,1))
p, g = state(env)
print(f"  After G flip: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Walk R to x=9
for i in range(8):
    p2, g2 = track(env, R)
    if p2[0] >= 9 or (p2 == p):
        break
    p = p2

p, g = state(env)
print(f"\n  Reached ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Verify x=9 column
print(f"  x=9 column y=5-26:")
for y in range(5, 27):
    s = sym(env, 9, y)
    if s != '.': print(f"    (9,{y}): {s}")
    elif y in [7, 8, 14, 15, 16, 17, 18, 19, 20, 23, 24]:
        print(f"    (9,{y}): . (open)")

# Now flip grav back UP with another G
# Available: y=1 row has G blocks at x=2-9 (minus x=1 already consumed)
print(f"\n  Flipping grav UP with G(2,1):")
track(env, C(2,1))
p, g = state(env)
print(f"  After G flip: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# With grav UP at x=9: falls UP to... B(9,15) as ceiling → lands at (9,16)
print_grid(env, (14, 20), (7, 12))

# Walk L from x=9 along y=16 (B ceiling at y=15)
print(f"\n  Walk L along y=16:")
for i in range(8):
    p2, g2 = track(env, L)
    if p2 == p:
        break
    p = p2

p, g = state(env)
print(f"\n  At ({p[0]},{p[1]})")

# undo all
for _ in range(30): env.step(UNDO)
p, g = state(env)
print(f"\n  Reset: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# ============================================================
# TEST 4: Complete x=9 corridor path
# G flip DOWN → walk R to x=9 → G flip UP → fly to (9,16) →
# walk L to (2,16) → need E ceiling at y=8 area → toggle B → fly up
# ============================================================
print("\n--- TEST 4: Full corridor path ---")

# Phase 1: Spread E(7,8) to create ceiling at y=8-9 area
print("Phase 1: E spread from (7,8)")
# E(7,8) → E at (6,8),(8,8),(7,7),(7,9)
track(env, C(7,8))
# E(6,8) → E at (5,8),(7,8),(6,7),(6,9)
track(env, C(6,8))
# E(5,8) → E at (4,8),(6,8),(5,7),(5,9)
track(env, C(5,8))
# E(4,8) → E at (3,8),(5,8),(4,7),(4,9)
track(env, C(4,8))
# E(3,8) → E at (2,8),(4,8),(3,7),(3,9)
track(env, C(3,8))

print(f"\n  E positions at y=7-9:")
for y in range(7, 10):
    es = [x for x in range(-1, 12) if sym(env, x, y) == 'E']
    if es: print(f"    y={y}: x={es}")
print_grid(env, (6, 11))

# Phase 2: G flip DOWN
print("\nPhase 2: G flip → grav DOWN")
track(env, C(1,1))
p, g = state(env)
print(f"  At ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Phase 3: Walk R to x=9
print("\nPhase 3: Walk R to x=9")
for i in range(8):
    p2, g2 = track(env, R)
    if p2[0] >= 9:
        break

p, g = state(env)
print(f"  At ({p[0]},{p[1]})")

# Phase 4: G flip UP
print("\nPhase 4: G flip → grav UP")
track(env, C(2,1))
p, g = state(env)
print(f"  At ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
print_grid(env, (14, 20), (0, 12))

# Phase 5: Walk L to x=2
print("\nPhase 5: Walk L along y=16")
for i in range(8):
    p2, g2 = track(env, L)
    if p2 == p:
        break
    p = p2

p, g = state(env)
print(f"  At ({p[0]},{p[1]})")

# Phase 6: Toggle B(2,15) → fall UP through y=15 → E ceiling at y=8
print("\nPhase 6: Toggle B(2,15) → fly up to E ceiling")
print(f"  B(2,15) = {sym(env,2,15)}")
print(f"  x=2 column y=6-15:")
for y in range(6, 16):
    print(f"    (2,{y}): {sym(env,2,y)}")

# Click B(2,15) to toggle to O
track(env, C(2,15))
p, g = state(env)
print(f"  After B toggle: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
print_grid(env, (6, 11))

# Phase 7: From y=9 area, walk L toward D(1,8)
print(f"\nPhase 7: Walk toward D(1,8)")
print(f"  x=1 at y=8: {sym(env,1,8)}")
print(f"  x=2 at y=8: {sym(env,2,8)}")

# Walk L to x=2 if not there
while state(env)[0][0] > 2:
    track(env, L)
# What's at (1,8)?
print(f"\n  D(1,8) = {sym(env,1,8)}")
p, g = state(env)
print(f"  At ({p[0]},{p[1]})")

# Click D(1,8) to destroy
track(env, C(1,8))
p, g = state(env)
print(f"  After D destroy: ({p[0]},{p[1]})")

# Walk L to x=1
track(env, L)
p, g = state(env)
print(f"  After L: ({p[0]},{p[1]})")

# Walk L to x=0
track(env, L)
p, g = state(env)
print(f"  After L: ({p[0]},{p[1]})")

# Show x=0 column
print(f"\n  x=0 column y=5-42:")
for y in range(5, 43):
    s = sym(env, 0, y)
    if s != '.': print(f"    (0,{y}): {s}")

print(f"\n  Final: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
print_grid(env, (5, 10))
