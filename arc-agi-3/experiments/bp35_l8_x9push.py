#!/usr/bin/env python3
"""L8: Test x=9 push-up approach.
After reaching (8,23) grav UP, clear E(9,23), walk R, then push-up through x=9 to (9,16).
Then walk-and-clear L along y=16 to (4,16)."""
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

def alive(env):
    p1 = state(env)[0]
    env.step(LEFT)
    p2 = state(env)[0]
    if p2 != p1:
        env.step(UNDO)
        return True
    env.step(RIGHT)
    p3 = state(env)[0]
    if p3 != p1:
        env.step(UNDO)
        return True
    return False

# ============================================================
print("=== L8: x=9 Push-up Test ===\n")
env = make_l8_env()

# Verified working solution through step 55 (reaches (8,23) grav UP)
SETUP_AND_NAV = [
    # SETUP: E spread, G consume, D destroy
    C(7,8), C(6,8), C(5,8), C(4,8), C(3,8),
    C(6,31), C(5,31), C(4,31), C(3,31),
    C(0,15), C(0,19), C(0,21), C(0,25), C(0,33), C(0,35),
    C(2,26), C(3,26), C(4,26),
    # G flip DOWN, walk R, G flip UP → (7,32)
    C(1,1), R, R, R, R,
    C(2,1),
    # Walk-and-clear L to (2,32)
    C(6,32), L, C(5,32), L, C(4,32), L, C(3,32), L, C(2,32), L,
    # Push-up to (2,21)
    C(2,31), C(2,30), C(2,29), C(2,28), C(2,27),
    C(2,26), C(2,25), C(2,24), C(2,23), C(2,22), C(2,21),
    # Walk-and-clear R to (7,21)
    C(3,21), R, C(4,21), R, C(5,21), R, C(6,21), R, C(7,21), R,
    # G-flip descent to (8,23)
    C(7,22), C(3,1), C(7,23), C(8,23), R, C(4,1),
]

step = 0
for move in SETUP_AND_NAV:
    step += 1
    p1, g1 = state(env)
    if move[0] == 'R': env.step(RIGHT)
    elif move[0] == 'L': env.step(LEFT)
    elif move[0] == 'C': click_act(env, move[1], move[2])
    p2, g2 = state(env)
    stuck = " STUCK!" if p1 == p2 and move[0] in ('R', 'L') else ""
    if stuck:
        print(f"  {step:3d}. {str(move):14s} ({p1[0]:2d},{p1[1]:2d})→({p2[0]:2d},{p2[1]:2d}){stuck}")
        print(f"  FAILED at step {step}!")
        print_grid(env, (max(0, p2[1]-3), p2[1]+3))
        sys.exit(1)

p, g = state(env)
print(f"After nav (step {step}): ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
print(f"Alive: {alive(env)}")

# Now at (8,23) grav UP. E(9,23) exists as side-effect of C(8,23).
print(f"\nx=9 column y=15-25:")
for y in range(15, 26):
    print(f"  (9,{y})={sym(env,9,y)}")

print(f"\n--- x=9 PUSH-UP SEQUENCE ---")

# Step 1: Clear E(9,23)
print(f"\n1. Clear E(9,23):")
print(f"  Before: (9,23)={sym(env,9,23)} (9,22)={sym(env,9,22)}")
click_act(env, 9, 23)
print(f"  After:  (9,23)={sym(env,9,23)} (9,22)={sym(env,9,22)} (9,24)={sym(env,9,24)}")
print(f"  (8,23)={sym(env,8,23)}")  # side-effect back to x=8?

# Step 2: Walk R to (9,23)
print(f"\n2. Walk R to x=9:")
p1 = state(env)[0]
env.step(RIGHT)
p2, g2 = state(env)
print(f"  ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}) grav={'UP' if g2 else 'DN'}")
print(f"  Alive: {alive(env)}")

if p2[0] != 9:
    print(f"  FAILED to reach x=9!")
    print_grid(env, (p2[1]-3, p2[1]+3))
    sys.exit(1)

# Step 3: Push-up through x=9
print(f"\n3. Push-up sequence:")
for i in range(10):
    p, g = state(env)
    # Find ceiling (nearest solid above)
    ceil_y = None
    for y in range(p[1]-1, p[1]-10, -1):
        s = sym(env, 9, y)
        if s != '.':
            ceil_y = y
            break

    if ceil_y is None:
        print(f"  No ceiling above ({p[0]},{p[1]})!")
        break

    ceil_s = sym(env, 9, ceil_y)
    print(f"  Push {i+1}: player ({p[0]},{p[1]}), ceil ({9},{ceil_y})={ceil_s}", end="")

    if ceil_s == 'B':
        print(f" → B ceiling, done!")
        break
    elif ceil_s != 'E':
        print(f" → not E, can't push!")
        break

    click_act(env, 9, ceil_y)
    p2, g2 = state(env)
    print(f" → ({p2[0]},{p2[1]})")

    if not alive(env):
        print(f"  DEAD after push!")
        break

    if p2[1] <= 16:
        print(f"  Reached y=16!")
        break

p, g = state(env)
print(f"\nAfter push-up: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
print(f"Alive: {alive(env)}")

# Step 4: Walk-and-clear L along y=16
print(f"\n4. Walk-and-clear L along y=16:")
for i in range(8):
    p, g = state(env)
    lx = p[0] - 1
    ly = p[1]
    ls = sym(env, lx, ly)

    if ls == 'E':
        print(f"  Clear E({lx},{ly})", end="")
        click_act(env, lx, ly)
        ls2 = sym(env, lx, ly)
        print(f" → ({lx},{ly})={ls2}")
    elif ls == 'W':
        print(f"  W at ({lx},{ly}) — blocked!")
        break

    env.step(LEFT)
    p2, g2 = state(env)
    moved = p2[0] < p[0]
    print(f"  L: ({p[0]},{p[1]})→({p2[0]},{p2[1]})" + (" STUCK!" if not moved else ""))

    if not moved:
        break

    if not alive(env):
        print(f"  DEAD!")
        break

    if p2[0] <= 4:
        print(f"  Reached x=4!")
        break

p, g = state(env)
print(f"\nFinal position: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
print(f"Alive: {alive(env)}")
print(f"\nGrid around player:")
print_grid(env, (max(0, p[1]-5), min(45, p[1]+5)))

# Step 5: Continue from (4,16) — toggle B(4,15), fly to D(4,13)
if p == (4, 16):
    print(f"\n--- Phase 7: B toggle + D destroy ---")
    print(f"B(4,15)={sym(env,4,15)}")
    click_act(env, 4, 15)  # B→O
    p2, g2 = state(env)
    print(f"After B toggle: ({p2[0]},{p2[1]}) (4,15)={sym(env,4,15)}")
    print(f"Alive: {alive(env)}")

    if p2[1] < 16:
        # Player flew through O
        print(f"Player flew up! D(4,13)={sym(env,4,13)}")
        click_act(env, 4, 13)  # Destroy D
        p3, g3 = state(env)
        print(f"After D destroy: ({p3[0]},{p3[1]})")
        print(f"Alive: {alive(env)}")

        # Check what's above
        print(f"\nx=4 column:")
        for y in range(7, 16):
            print(f"  (4,{y})={sym(env,4,y)}")

        # Walk L toward x=2
        print(f"\nWalk L from ({p3[0]},{p3[1]}):")
        for i in range(5):
            p_before = state(env)[0]
            lx = p_before[0] - 1
            ly = p_before[1]
            ls = sym(env, lx, ly)
            if ls == 'E':
                click_act(env, lx, ly)
            env.step(LEFT)
            p_after = state(env)[0]
            print(f"  ({p_before[0]},{p_before[1]})→({p_after[0]},{p_after[1]}) [{ls}]")
            if p_after == p_before:
                break
            if not alive(env):
                print(f"  DEAD!")
                break

p, g = state(env)
print(f"\n=== END STATE: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} level={env._game.level_index} ===")
print_grid(env, (max(0, p[1]-5), min(45, p[1]+5)))
