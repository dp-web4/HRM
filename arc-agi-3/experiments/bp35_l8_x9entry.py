#!/usr/bin/env python3
"""L8: Test x=9 corridor entry from y=8-9 area.
Plan: E chain from (7,8) to (9,8), chain down x=9 to create stepping stones,
then navigate player to (9,16) for the walk-L-to-x=2 path."""
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
print("=== TEST: x=9 entry from y=8 area ===\n")
env = make_l8_env()

# Setup: E chain from (7,8) east to (9,8)
print("E chain east from (7,8):")
click_act(env, 7, 8)  # E at (6,8),(8,8),(7,7),(7,9)
click_act(env, 8, 8)  # E at (7,8),(9,8),(8,7),(8,9)
print(f"  E(9,8)={sym(env,9,8)}")

# Chain E down x=9 from y=8
print("\nE chain down x=9:")
click_act(env, 9, 8)   # E at (8,8),(10,8)=W,(9,7),(9,9)
print(f"  After C(9,8): E(9,9)={sym(env,9,9)} E(9,7)={sym(env,9,7)}")

click_act(env, 9, 9)   # E at (8,9),(10,9)=W,(9,8),(9,10)
print(f"  After C(9,9): E(9,10)={sym(env,9,10)}")

click_act(env, 9, 10)  # E at (8,10),(9,9),(9,11)
click_act(env, 9, 11)  # E at (8,11),(9,10),(9,12)
click_act(env, 9, 12)  # E at (8,12),(9,11),(9,13)
click_act(env, 9, 13)  # E at (8,13),(9,12),(9,14)
click_act(env, 9, 14)  # E at (8,14),(9,13),(9,15)=B→blocked!

print(f"\nx=9 column after chain:")
for y in range(6, 20):
    s = sym(env, 9, y)
    if s != '.': print(f"  (9,{y})={s}")

# Check: can E spread to (9,15) if B is toggled?
print(f"\n  (9,15) = {sym(env,9,15)} [B or O?]")

# Toggle B(9,15) first, then try E chain
print("\nToggle B(9,15):")
click_act(env, 9, 15)
print(f"  (9,15) = {sym(env,9,15)}")

# Try placing E at (9,15) via chain from (9,14)
# (9,14) was removed by chain click. What E is at y=14?
print(f"  (9,14) = {sym(env,9,14)}")
print(f"  (9,13) = {sym(env,9,13)}")

# Re-chain from existing E at y=13
if sym(env, 9, 13) == 'E':
    click_act(env, 9, 13)
    print(f"  After C(9,13): (9,14)={sym(env,9,14)} (9,15)={sym(env,9,15)}")

if sym(env, 9, 14) == 'E':
    click_act(env, 9, 14)
    print(f"  After C(9,14): (9,15)={sym(env,9,15)} (9,16)={sym(env,9,16)}")

# Check full x=9 corridor
print(f"\nx=9 after all manipulation:")
for y in range(6, 26):
    s = sym(env, 9, y)
    if s != '.': print(f"  (9,{y})={s}")
    elif y in [7,8,9,14,15,16,17]: print(f"  (9,{y})=. (open)")

# Can player reach (9,9)? Need to walk there.
# Path: from (3,35), walk to (8,10) via y=14 or y=10 corridor
# First need to get from y=35 to y=10...

# Test: what if we just navigate player to (4,14) via D(4,13)?
# Setup already done: E at y=8-9 area
# Need westward E chain too for upper ceiling
print("\n\n--- E chain west for ceiling ---")
env2 = make_l8_env()

# Full E chain: west from (7,8)
for ex, ey in [(7,8), (6,8), (5,8), (4,8), (3,8)]:
    click_act(env2, ex, ey)
# East: (8,8) and chain down x=9
click_act(env2, 8, 8)  # need E(8,8) from west chain... it exists at even positions

print("E at y=8:", [x for x in range(-1,12) if sym(env2,x,8) == 'E'])
print("E at y=9:", [x for x in range(-1,12) if sym(env2,x,9) == 'E'])

# Now E(8,8) — does it exist?
print(f"E(8,8)={sym(env2,8,8)}")
if sym(env2, 8, 8) == 'E':
    click_act(env2, 8, 8)
    print(f"After C(8,8): E(9,8)={sym(env2,9,8)}")

# E chain down x=9 from (9,8)
for y in range(8, 15):
    if sym(env2, 9, y) == 'E':
        click_act(env2, 9, y)
        print(f"  C(9,{y}): E(9,{y+1})={sym(env2,9,y+1)}")

print(f"\nx=9 column:")
for y in range(6, 20):
    s = sym(env2, 9, y)
    if s != '.': print(f"  (9,{y})={s}")
    elif y in [7,8,9,14,15,16,17]: print(f"  (9,{y})=. (open)")

# Test: if player at (9,14) with grav DN, toggle B(9,15)→O,
# Does O count as "empty" for falling through? Or solid?
print(f"\n--- B toggle fall test ---")
# Reset with fresh env for clean test
env3 = make_l8_env()
# Simple test: from anywhere, click B to toggle
# Check if O is passable for player movement
print(f"B(9,15) = {sym(env3,9,15)}")
click_act(env3, 9, 15)
print(f"After toggle: (9,15) = {sym(env3,9,15)}")
# Check if O entity name
items = env3._game.oztjzzyqoek.hdnrlfmyrj.jhzcxkveiw(9, 15)
if items:
    print(f"Entity name: {items[0].name}")
else:
    print(f"No entity at (9,15)!")

# What about D blocks — when destroyed, do they leave anything?
print(f"\nD(1,8) = {sym(env3,1,8)}")
click_act(env3, 1, 8)
print(f"After destroy: (1,8) = {sym(env3,1,8)}")
items = env3._game.oztjzzyqoek.hdnrlfmyrj.jhzcxkveiw(1, 8)
print(f"Entity: {items}")

# Test: does E spread consider O (toggled B) as empty or occupied?
click_act(env3, 9, 15)  # toggle back to B
click_act(env3, 9, 15)  # toggle to O again
# Place E near (9,15) to test if spread goes through
# We'd need E at (9,14) first
# Quick test: spread E near B/O and see
print(f"\n--- E spread near O test ---")
# Create E at (8,15) by clicking the B
click_act(env3, 8, 15)  # toggle B(8,15) → O
print(f"(8,15) = {sym(env3,8,15)}")
# Now if we had E at (8,14), spreading would test O
# For now just check entity types
for x in range(7, 11):
    items = env3._game.oztjzzyqoek.hdnrlfmyrj.jhzcxkveiw(x, 15)
    name = items[0].name if items else 'empty'
    print(f"  ({x},15) = {SYM.get(name, name)}")
