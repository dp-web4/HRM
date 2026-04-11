#!/usr/bin/env python3
"""L7 path exploration: test UNDO from death, test x=8-9 shaft, strategic G flip."""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
CLICK = GameAction.ACTION6
UNDO = GameAction.ACTION7

def click_act(env, gx, gy):
    engine = env._game.oztjzzyqoek
    cam_y = engine.camera.rczgvgfsfb[1]
    return env.step(CLICK, data={"x": gx*6, "y": gy*6 - cam_y})

def state(env):
    engine = env._game.oztjzzyqoek
    p = engine.twdpowducb
    return tuple(p.qumspquyus), engine.vivnprldht

def check_grid(env, x, y):
    engine = env._game.oztjzzyqoek
    grid = engine.hdnrlfmyrj
    items = grid.jhzcxkveiw(x, y)
    return [i.name for i in items] if items else []

R = ('R',)
L = ('L',)
def C(x, y): return ('C', x, y)

sols = [
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
]

arcade = Arcade()
env = arcade.make('bp35-0a0ad940')
fd = env.reset()

for i, sol in enumerate(sols):
    ok = False
    old_level = env._game.level_index
    for m in sol:
        if m[0] == 'R': env.step(RIGHT)
        elif m[0] == 'L': env.step(LEFT)
        elif m[0] == 'C':
            engine = env._game.oztjzzyqoek
            cy = engine.camera.rczgvgfsfb[1]
            env.step(CLICK, data={"x": m[1]*6, "y": m[2]*6 - cy})
        if env._game.level_index > old_level:
            ok = True; break
    if not ok:
        for _ in range(10):
            env.step(LEFT)
            if env._game.level_index > old_level:
                ok = True; break
    if not ok:
        print(f"L{i} FAIL"); sys.exit(1)
print("L0-L6 solved, now on L7\n")

# === Test 1: UNDO from death ===
print("=== Test 1: Clean UNDO from death ===")
pos0, g0 = state(env)
print(f"  Start: {pos0} grav={'UP' if g0 else 'DN'}")

env.step(RIGHT)  # (3,32) → (4,32)
pos1, g1 = state(env)
print(f"  R1: {pos1}")

env.step(RIGHT)  # (4,32) → (5,28) spike death
pos2, g2 = state(env)
print(f"  R2: {pos2} {'DEAD?' if pos2[1] < 31 else ''}")

# Immediate UNDO
env.step(UNDO)
pos3, g3 = state(env)
print(f"  UNDO: {pos3}")

env.step(UNDO)
pos4, g4 = state(env)
print(f"  UNDO: {pos4}")

env.step(UNDO)
pos5, g5 = state(env)
print(f"  UNDO: {pos5}")

# === Test 2: G flip then walk into gap ===
print(f"\n=== Test 2: G flip first, then walk right ===")
# Reset
while state(env)[0] != (3, 32):
    env.step(UNDO)
pos, grav = state(env)
print(f"  Reset: {pos} grav={'UP' if grav else 'DN'}")

# Flip G first
click_act(env, 5, 2)
pos, grav = state(env)
print(f"  After G flip: {pos} grav={'UP' if grav else 'DN'}")

# Walk right
for i in range(9):
    p1, g1 = state(env)
    env.step(RIGHT)
    p2, g2 = state(env)
    moved = p1 != p2
    print(f"  R{i+1}: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}) {'MOVED' if moved else 'blocked'}")
    if not moved:
        break

# Undo all
for _ in range(15):
    env.step(UNDO)
pos, grav = state(env)
print(f"  Reset: {pos} grav={'UP' if grav else 'DN'}")

# === Test 3: Toggle B blocks, then G flip, then navigate ===
print(f"\n=== Test 3: Toggle B(7,25)→O, B(8,22)→O, then G flip + walk ===")
# Toggle setup
click_act(env, 7, 25)  # B→O at (7,25)
items = check_grid(env, 7, 25)
print(f"  (7,25) after toggle: {items}")

click_act(env, 8, 22)  # B→O at (8,22)
items = check_grid(env, 8, 22)
print(f"  (8,22) after toggle: {items}")

click_act(env, 8, 18)  # B→O at (8,18)
items = check_grid(env, 8, 18)
print(f"  (8,18) after toggle: {items}")

# Now flip G
click_act(env, 5, 2)
pos, grav = state(env)
print(f"  After G flip: {pos} grav={'UP' if grav else 'DN'}")

# Walk right from (3,34)
for i in range(9):
    p1, g1 = state(env)
    env.step(RIGHT)
    p2, g2 = state(env)
    print(f"  R{i+1}: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]})")
    if p1 == p2:
        break

# Undo all
for _ in range(20):
    env.step(UNDO)
pos, grav = state(env)
print(f"  Reset: {pos} grav={'UP' if grav else 'DN'}")

# === Test 4: What if we toggle B blocks at y=22 to create a floor gap, walk up with grav UP ===
# Key idea: toggle B blocks that are NOT in our landing column
# Use E(2,29) as a ceiling by going to x=2 area
print(f"\n=== Test 4: E(2,29) ceiling approach ===")
# E at (2,29) is solid. With grav UP, player at (2,30) has E ceiling.
# But how to reach (2,30)? x=2 at y=31 is wall (no gap).
# Unless... toggle some B to make a floor at y=29 level?
# No, we can't place new B blocks.

# What if we approach from the gap: enter at x=5-7, somehow get to x=2?
# Player enters at (5,28) dead. Can't walk to x=2.

# === Test 5: Check if player can SURVIVE at y=28 below spikes ===
# Maybe the death is at y=28 but we can UNDO?
print(f"\n=== Test 5: R,R then immediate extensive UNDO ===")
pos, _ = state(env)
print(f"  Start: {pos}")

env.step(RIGHT)
p1, _ = state(env)
print(f"  R: {p1}")

env.step(RIGHT)
p2, _ = state(env)
print(f"  R: {p2}")

# Check if actually dead by examining player object
engine = env._game.oztjzzyqoek
player = engine.twdpowducb
print(f"  Player attrs of interest:")
for attr in dir(player):
    if not attr.startswith('_'):
        try:
            val = getattr(player, attr)
            if not callable(val) and attr not in ('hdnrlfmyrj', 'welhuapdwo', 'hilopxwoqvn', 'dwzxeajrgkw', 'gimrsagplbc', 'esishrsguis', 'onfaxqmstk', 'vyicipsdbdd'):
                if 'dead' in str(val).lower() or 'alive' in str(val).lower() or 'kill' in str(val).lower():
                    print(f"    .{attr} = {val}")
                # Check boolean flags
                if isinstance(val, bool):
                    print(f"    .{attr} = {val}")
        except:
            pass

# Try UNDO extensively
for i in range(10):
    env.step(UNDO)
    p, g = state(env)
    if p != p2:
        print(f"  UNDO {i+1}: MOVED to {p} grav={'UP' if g else 'DN'}")
        break
    else:
        if i < 3:
            print(f"  UNDO {i+1}: still at {p}")

pos_final, grav_final = state(env)
print(f"  Final: {pos_final} grav={'UP' if grav_final else 'DN'}")

# === Test 6: What about toggling y=22 B blocks to create floor for a grav-DOWN approach ===
# Restore to start
for _ in range(15):
    env.step(UNDO)
pos, grav = state(env)
if pos != (3,32):
    print(f"\n  WARNING: Could not reset! pos={pos}")
else:
    print(f"\n=== Test 6: Strategic B toggle + G flip path ===")
    # Idea:
    # 1. Toggle B(5,22) to O
    # 2. Flip G → grav DOWN, fall to (3,34)
    # 3. Walk to x=5
    # 4. At (5,34) grav DOWN: floor at (5,35) wall
    # BUT: we need to go UP. With grav DOWN, we can't.
    #
    # Alternative: DON'T flip G. Toggle some B blocks.
    # Walk through y=31 gap...dies at y=28
    #
    # Key question: is there ANY B block we can toggle to
    # create a safe landing at x=5-7 between y=28 and y=31?
    print(f"  B blocks:")
    for pos_b in [(8,18), (1,22),(2,22),(3,22),(4,22),(5,22),(6,22),(7,22),(8,22),(9,22), (7,25)]:
        items = check_grid(env, *pos_b)
        print(f"    {pos_b}: {items}")

    # The only way: use GRAV change to navigate safely.
    # With grav DN and spikes being non-lethal:
    # 1. Flip G from start → grav DN, fall to (3,34)
    # 2. Walk to x=7 at y=34
    # 3. Toggle B blocks to create a "staircase" effect?
    #
    # With grav DN at (7,34): floor y=35. Can't go up.
    # UNLESS we toggle y=22 B to O and somehow there's a
    # floor above the player?

    # What about toggling B to O then BACK to B after
    # positioning? E.g., toggle B(7,22) to O, walk a player
    # onto it... but player can't reach y=22 with grav DN.

    # RADICAL IDEA: what if spikes DONT actually kill?
    # What if only landing on the exact spike cell kills?
    # (5,28) is one cell below spike. Maybe the death is
    # from something else entirely?

    # Let's check: what's at (5,28)?
    items_528 = check_grid(env, 5, 28)
    items_527 = check_grid(env, 5, 27)
    print(f"\n  (5,27): {items_527}")
    print(f"  (5,28): {items_528}")

    # Check ALL cells between y=28-31 at x=5
    for y in range(27, 33):
        items = check_grid(env, 5, y)
        print(f"  (5,{y}): {items}")

    # Maybe there's a HIDDEN entity at (5,28) or nearby?
    # Or maybe the death is from the MOVEMENT hitting a spike
    # during the fall?

# === Test 7: Can we create a B block ceiling BELOW spikes ===
# Toggle B(5,22) to O then back to B: it's still at y=22, far from y=28
# What if there's a way to move B blocks? NO - they toggle in place.

# === Test 8: Test walking from gap with grav DOWN ===
print(f"\n=== Test 8: G flip + walk through gap with grav DOWN ===")
for _ in range(20):
    env.step(UNDO)
pos, grav = state(env)
print(f"  Start: {pos} grav={'UP' if grav else 'DN'}")

if pos == (3, 32) and grav:
    # Flip gravity FIRST
    click_act(env, 5, 2)
    pos, grav = state(env)
    print(f"  After G: {pos} grav={'UP' if grav else 'DN'}")

    # Walk right carefully
    for i in range(8):
        p1, g1 = state(env)
        env.step(RIGHT)
        p2, g2 = state(env)
        print(f"  R{i+1}: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]})")
        if p1 == p2:
            break
        # If y changed, something interesting happened
        if p1[1] != p2[1]:
            print(f"    Y changed from {p1[1]} to {p2[1]}!")

    pos_now, grav_now = state(env)
    print(f"  Current: {pos_now} grav={'UP' if grav_now else 'DN'}")

    # Walk left
    for i in range(8):
        p1, _ = state(env)
        env.step(LEFT)
        p2, _ = state(env)
        print(f"  L{i+1}: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]})")
        if p1 == p2:
            break
        if p1[1] != p2[1]:
            print(f"    Y changed from {p1[1]} to {p2[1]}!")
