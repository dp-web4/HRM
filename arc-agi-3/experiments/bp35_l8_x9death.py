#!/usr/bin/env python3
"""L8: Diagnose why push-up to (9,22) kills.
Also test: can player enter (9,23) with grav UP and fly straight to (9,16)?"""
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

def is_alive(env):
    engine = env._game.oztjzzyqoek
    return engine.twdpowducb.etquaizpmu

def all_entities(env, x, y):
    items = env._game.oztjzzyqoek.hdnrlfmyrj.jhzcxkveiw(x, y)
    return [(i.name, SYM.get(i.name, '?')) for i in items]

# ============================================================
# TEST 1: Direct flight test — enter (9,23) with clear column
# Can player at (9,23) with grav UP fly to (9,16)?
# ============================================================
print("=== TEST 1: Direct flight through x=9 ===\n")
env = make_l8_env()

# Minimal setup: just get to (8,23) grav UP
SETUP_NAV = [
    C(7,8), C(6,8), C(5,8), C(4,8), C(3,8),
    C(6,31), C(5,31), C(4,31), C(3,31),
    C(0,15), C(0,19), C(0,21), C(0,25), C(0,33), C(0,35),
    C(2,26), C(3,26), C(4,26),
    C(1,1), R, R, R, R, C(2,1),
    C(6,32), L, C(5,32), L, C(4,32), L, C(3,32), L, C(2,32), L,
    C(2,31), C(2,30), C(2,29), C(2,28), C(2,27),
    C(2,26), C(2,25), C(2,24), C(2,23), C(2,22), C(2,21),
    C(3,21), R, C(4,21), R, C(5,21), R, C(6,21), R, C(7,21), R,
    C(7,22), C(3,1), C(7,23), C(8,23), R, C(4,1),
]

for move in SETUP_NAV:
    if move[0] == 'R': env.step(RIGHT)
    elif move[0] == 'L': env.step(LEFT)
    elif move[0] == 'C': click_act(env, move[1], move[2])

p, g = state(env)
print(f"At ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} alive={is_alive(env)}")

# Examine x=9 column BEFORE any clearing
print(f"\nx=9 column:")
for y in range(14, 26):
    ents = all_entities(env, 9, y)
    s = sym(env, 9, y)
    if s != '.' or y in [15,16,22,23,24]:
        print(f"  (9,{y}): {s} {ents}")

# Clear E(9,23) to open path
print(f"\nClearing E(9,23):")
click_act(env, 9, 23)
print(f"  (9,22)={sym(env,9,22)} (9,23)={sym(env,9,23)} (9,24)={sym(env,9,24)}")
print(f"  (8,23)={sym(env,8,23)}")

# Now x=9 has E(9,22) blocking flight. Clear it too:
print(f"\nClearing E(9,22):")
click_act(env, 9, 22)
for y in range(15, 25):
    s = sym(env, 9, y)
    if s != '.': print(f"  (9,{y})={s}")

# Now clear E(9,21) if created:
if sym(env, 9, 21) == 'E':
    print(f"\nClearing E(9,21):")
    click_act(env, 9, 21)
    for y in range(15, 25):
        s = sym(env, 9, y)
        if s != '.': print(f"  (9,{y})={s}")

# Keep clearing until column is clean
for clear_y in range(20, 15, -1):
    if sym(env, 9, clear_y) == 'E':
        print(f"  Clearing E(9,{clear_y})")
        click_act(env, 9, clear_y)

print(f"\nx=9 final state:")
for y in range(14, 26):
    s = sym(env, 9, y)
    print(f"  (9,{y})={s}")

# Now try walking R to (9,23) — (9,23) should have E from oscillation
print(f"\n(9,23)={sym(env,9,23)}")
if sym(env, 9, 23) == 'E':
    click_act(env, 9, 23)
    print(f"After clear: (9,23)={sym(env,9,23)} (9,22)={sym(env,9,22)}")

# Walk R
p1 = state(env)[0]
env.step(RIGHT)
p2, g2 = state(env)
print(f"\nWalk R: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}) alive={is_alive(env)}")

if p2[0] == 9:
    # Check where gravity takes us
    print(f"  At (9,{p2[1]}). Entities above:")
    for y in range(p2[1]-1, 14, -1):
        s = sym(env, 9, y)
        if s != '.':
            print(f"    (9,{y})={s}")
            break

# ============================================================
# TEST 2: Check if there's a hidden entity or mechanic at (9,22)
# ============================================================
print(f"\n\n=== TEST 2: Detailed death diagnosis at (9,22) ===\n")
env2 = make_l8_env()

for move in SETUP_NAV:
    if move[0] == 'R': env2.step(RIGHT)
    elif move[0] == 'L': env2.step(LEFT)
    elif move[0] == 'C': click_act(env2, move[1], move[2])

p, g = state(env2)
print(f"At ({p[0]},{p[1]}) alive={is_alive(env2)}")

# Clear E(9,23) and walk R
click_act(env2, 9, 23)
env2.step(RIGHT)
p, g = state(env2)
print(f"At ({p[0]},{p[1]}) alive={is_alive(env2)}")

# Before push: check all entities at (9,22) and adjacent
print(f"\nBefore push-up:")
for x in range(8, 11):
    for y in range(20, 25):
        ents = all_entities(env2, x, y)
        if ents:
            print(f"  ({x},{y}): {ents}")

# Now push up
print(f"\nExecuting push-up C(9,22):")
print(f"  Player before: {state(env2)}")
click_act(env2, 9, 22)
p, g = state(env2)
print(f"  Player after: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} alive={is_alive(env2)}")

# Check what's around (9,22)
print(f"\nAfter push-up, entities around (9,22):")
for x in range(8, 11):
    for y in range(19, 25):
        ents = all_entities(env2, x, y)
        if ents:
            print(f"  ({x},{y}): {ents}")

# ============================================================
# TEST 3: Can we walk R into (9,23) with empty x=9 above?
# Pre-clear the entire x=9 column from setup
# ============================================================
print(f"\n\n=== TEST 3: Pre-clear x=9 then walk R ===\n")
env3 = make_l8_env()

# Modified setup: same but also clear x=9 column
for move in SETUP_NAV:
    if move[0] == 'R': env3.step(RIGHT)
    elif move[0] == 'L': env3.step(LEFT)
    elif move[0] == 'C': click_act(env3, move[1], move[2])

p, g = state(env3)
print(f"At ({p[0]},{p[1]}) alive={is_alive(env3)}")

# Clear E(9,23) from setup by clicking multiple times
# The goal: get x=9 from y=16 to y=23 completely clear
# Then walk R → player at (9,23) flies to (9,16) in one shot

# First clear E(9,23)
click_act(env3, 9, 23)  # removes E(9,23), creates E(9,22), E(9,24)
# Clear E(9,22)
click_act(env3, 9, 22)  # removes E(9,22), creates E(9,21), E(9,23)
# Clear E(9,23) again
click_act(env3, 9, 23)  # removes, creates E(9,22), E(9,24)
# Clear E(9,22) again
click_act(env3, 9, 22)  # removes, creates E(9,21), E(9,23)

# The oscillation fills E down from y=23. Let's see what we have:
print(f"After oscillation clearing:")
for y in range(15, 26):
    s = sym(env3, 9, y)
    if s != '.':
        print(f"  (9,{y})={s}")

# What if we clear ALL of them in sequence from top?
print(f"\nClearing all E in x=9 from top:")
for sweep in range(5):
    cleared_any = False
    for y in range(16, 25):
        if sym(env3, 9, y) == 'E':
            click_act(env3, 9, y)
            cleared_any = True
    if not cleared_any:
        break
    es = [y for y in range(15, 26) if sym(env3, 9, y) == 'E']
    print(f"  Sweep {sweep+1}: E at y={es}")

# Final state
print(f"\nx=9 final:")
for y in range(14, 26):
    print(f"  (9,{y})={sym(env3, 9, y)}")

# Try walking R
p1 = state(env3)[0]
if sym(env3, 9, 23) == 'E':
    click_act(env3, 9, 23)
    print(f"\nCleared E(9,23) one more time")
    print(f"  (9,22)={sym(env3,9,22)} (9,23)={sym(env3,9,23)}")

env3.step(RIGHT)
p2, g2 = state(env3)
print(f"\nWalk R: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}) grav={'UP' if g2 else 'DN'} alive={is_alive(env3)}")

if p2[0] == 9:
    print(f"  SUCCESS! Player at (9,{p2[1]})")
    print(f"  Grid:")
    print_grid(env3, (14, 26))

# ============================================================
# TEST 4: What if we use UNDO after E(8,23) creates E(9,23)?
# Click E(8,23), walk R, UNDO walk, UNDO click, then try different
# ============================================================
print(f"\n\n=== TEST 4: v-spike lethality check with grav DN ===\n")
env4 = make_l8_env()
p, g = state(env4)
print(f"Start: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Test: is v-spike at (7,25) lethal with grav DN?
# Place player at (7,24) grav DN → adjacent to v(7,25)
# Quick test: G flip DN, walk R to x=7
click_act(env4, 1, 1)  # G flip DN
p, g = state(env4)
print(f"G flip: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Walk R to x=7
for _ in range(4):
    env4.step(RIGHT)
p, g = state(env4)
print(f"Walk R: ({p[0]},{p[1]}) alive={is_alive(env4)}")
# Player should be at floor y=38 level with grav DN

# G flip UP: fly up to E ceiling
click_act(env4, 2, 1)
p, g = state(env4)
print(f"G flip UP: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} alive={is_alive(env4)}")

print(f"\n  Entities at v-spike row y=25:")
for x in range(5, 10):
    print(f"  ({x},25)={sym(env4,x,25)}", end="")
print()
print(f"  Entities at y=17:")
for x in range(2, 10):
    print(f"  ({x},17)={sym(env4,x,17)}", end="")
print()

# ============================================================
# TEST 5: Does walk into x=9 at y=23 cause flight if column is clean?
# Simpler test: fresh env, minimal setup, manually place player near x=9
# ============================================================
print(f"\n\n=== TEST 5: Flight through clean x=9 shaft ===\n")
env5 = make_l8_env()
p, g = state(env5)
print(f"L8 start: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Verify x=9 column is clean
print(f"x=9 original:")
for y in range(14, 26):
    s = sym(env5, 9, y)
    if s != '.':
        print(f"  (9,{y})={s}")

# Simple test: G flip DN, walk R to x=8, G flip UP
# Then check if we can reach y=16 area
click_act(env5, 1, 1)  # G flip DN
for _ in range(5):
    env5.step(RIGHT)
p, g = state(env5)
print(f"\nAfter walk R: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} alive={is_alive(env5)}")

click_act(env5, 2, 1)  # G flip UP
p, g = state(env5)
print(f"G flip UP: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} alive={is_alive(env5)}")

# Walk R toward x=9
env5.step(RIGHT)
p2, g2 = state(env5)
print(f"Walk R: ({p[0]},{p[1]})→({p2[0]},{p2[1]}) alive={is_alive(env5)}")

if p2[0] == 9:
    print(f"  At x=9! y={p2[1]}")
    print_grid(env5, (p2[1]-3, p2[1]+5))
