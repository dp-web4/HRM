#!/usr/bin/env python3
"""L7 path exploration v2: each test uses a fresh env to avoid death-state issues."""
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

L0_L6_SOLS = [
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

def make_l7_env():
    """Create env and solve L0-L6 to reach L7."""
    arcade = Arcade()
    env = arcade.make('bp35-0a0ad940')
    env.reset()
    for sol in L0_L6_SOLS:
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

def check_grid(env, x, y):
    engine = env._game.oztjzzyqoek
    grid = engine.hdnrlfmyrj
    items = grid.jhzcxkveiw(x, y)
    return [i.name for i in items] if items else []


# ============================================================
# TEST A: G flip first, then explore with grav DOWN
# ============================================================
print("=== TEST A: G flip → grav DOWN exploration ===")
env = make_l7_env()
pos, grav = state(env)
print(f"  Start: {pos} grav={'UP' if grav else 'DN'}")

# Flip gravity
click_act(env, 5, 2)
pos, grav = state(env)
print(f"  After G flip: {pos} grav={'UP' if grav else 'DN'}")

# Walk right extensively
print("  Walking RIGHT:")
for i in range(9):
    p1, _ = state(env)
    env.step(RIGHT)
    p2, _ = state(env)
    dy = f" (fell {'down' if p2[1]>p1[1] else 'up'} {abs(p2[1]-p1[1])} cells)" if p1[1] != p2[1] else ""
    print(f"    R{i+1}: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}){dy}")
    if p1 == p2:
        break

# Walk left from current position
print("  Walking LEFT:")
for i in range(12):
    p1, _ = state(env)
    env.step(LEFT)
    p2, _ = state(env)
    dy = f" (fell {'down' if p2[1]>p1[1] else 'up'} {abs(p2[1]-p1[1])} cells)" if p1[1] != p2[1] else ""
    print(f"    L{i+1}: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}){dy}")
    if p1 == p2:
        break


# ============================================================
# TEST B: Toggle B blocks then G flip
# ============================================================
print("\n=== TEST B: Toggle B + G flip exploration ===")
env = make_l7_env()

# Toggle B blocks to open paths
click_act(env, 7, 25)   # (7,25) B→O
click_act(env, 8, 22)   # (8,22) B→O
click_act(env, 9, 22)   # (9,22) B→O
click_act(env, 8, 18)   # (8,18) B→O (gem area access)
print(f"  Toggled: (7,25)→O, (8,22)→O, (9,22)→O, (8,18)→O")

# Flip G → grav DOWN
click_act(env, 5, 2)
pos, grav = state(env)
print(f"  After G flip: {pos} grav={'UP' if grav else 'DN'}")

# Walk right
print("  Walking RIGHT:")
for i in range(9):
    p1, _ = state(env)
    env.step(RIGHT)
    p2, _ = state(env)
    dy = f" (y: {p1[1]}→{p2[1]})" if p1[1] != p2[1] else ""
    print(f"    R{i+1}: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}){dy}")
    if p1 == p2:
        break

# Walk left from wherever we end up
print("  Walking LEFT:")
for i in range(12):
    p1, _ = state(env)
    env.step(LEFT)
    p2, _ = state(env)
    dy = f" (y: {p1[1]}→{p2[1]})" if p1[1] != p2[1] else ""
    print(f"    L{i+1}: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}){dy}")
    if p1 == p2:
        break


# ============================================================
# TEST C: NO G flip - toggle everything, try to navigate with grav UP
# ============================================================
print("\n=== TEST C: No G flip, toggle B blocks, grav UP exploration ===")
env = make_l7_env()

# Toggle ALL B blocks at y=22 to O
for bx in range(1, 10):
    click_act(env, bx, 22)
# Toggle B(7,25) to O
click_act(env, 7, 25)
# Toggle B(8,18) to O
click_act(env, 8, 18)
print(f"  Toggled all B to O")

pos, grav = state(env)
print(f"  Player: {pos} grav={'UP' if grav else 'DN'}")

# Walk left to wall
env.step(LEFT); env.step(LEFT)
pos, _ = state(env)
print(f"  After LL: {pos}")

# Walk right and log everything
print("  Walking RIGHT from (1,32):")
for i in range(9):
    p1, g1 = state(env)
    env.step(RIGHT)
    p2, g2 = state(env)
    dy = f" (y: {p1[1]}→{p2[1]})" if p1[1] != p2[1] else ""
    print(f"    R{i+1}: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}){dy}")
    if p1 == p2:
        break


# ============================================================
# TEST D: B toggles to create useful floor patterns with grav UP
# Only toggle specific B blocks to create a safe column
# ============================================================
print("\n=== TEST D: Selective B toggle for x=8 safe path ===")
env = make_l7_env()

# Key insight: x=8 column has NO spikes at y=27!
# x=8: y=18 B, y=22 B, rest open
# If we toggle B(8,22) to O, the x=8 shaft is clear from y=21 to y=28
# But we need to GET to x=8 in the y=28-31 region first

# What if we toggle ALL y=22 B blocks to O, creating a floor gap?
# Then with grav UP, from y=34 the player falls up through:
# (x,33)→(x,32)→(x,31)→ depends on y=31 wall

# Actually, the B floor at y=22 is ABOVE the y=31 wall.
# With grav UP, player at y=32 has ceiling at y=31 wall.
# The B floor at y=22 is even further up.

# Let me think about what happens at EACH x position going right:
# x=1-4: y=31 is wall (ceiling). Player stable at (x,32).
# x=5-7: y=31 is gap. Fall up through (x,30),(x,29),(x,28),(x,27) spike. DEAD.
# x=8-9: y=31 is wall. Player stable at (x,32).

# So player CAN reach x=8 or x=9 at y=32!
# But (8,32) has ceiling wall at y=31. And above that is... NOT accessible.

# What's BELOW (8,32)? With grav UP the player is pressed against ceiling.
# (8,33), (8,34) open, (8,35) wall floor. Player floats at (8,32).

# Key: the player at x=8, y=32 is stable. But stuck. Y=31 wall above.
# Above y=31 at x=8 is the spike-free shaft. But can't access it.

# WAIT - what if x=8 at y=31 can be passed through somehow?
# y=31 at x=8 is WALL. Walls can't be toggled.

# What about approaching from a different direction?
# What if we toggle something to create a gap at y=31?
# y=31 is all walls except x=5-7 gap. Can't create new gaps.

# I'm stuck. Let me check: are there any other entities I missed?
# Let me do a comprehensive check of the y=31-35 area
print(f"  Checking y=31-35, x=0-10:")
engine = env._game.oztjzzyqoek
grid = engine.hdnrlfmyrj
sym = {'xcjjwqfzjfe':'W', 'yuuqpmlxorv':'B', 'oonshderxef':'O',
       'ubhhgljbnpu':'^', 'fjlzdjxhant':'*', 'etlsaqqtjvn':'E',
       'lrpkmzabbfa':'G'}
for y in range(31, 36):
    row = ""
    for x in range(0, 11):
        items = grid.jhzcxkveiw(x, y)
        if items:
            row += sym.get(items[0].name, '?')
        else:
            row += "."
    print(f"    y={y}: {row}")

# Also check y=26-28 for completeness
print(f"  y=26-28, x=0-10:")
for y in range(26, 29):
    row = ""
    for x in range(0, 11):
        items = grid.jhzcxkveiw(x, y)
        if items:
            row += sym.get(items[0].name, '?')
        else:
            row += "."
    print(f"    y={y}: {row}")

# Test: walk to x=8 and confirm stable
env.step(RIGHT); env.step(RIGHT); env.step(RIGHT); env.step(RIGHT); env.step(RIGHT)
pos, _ = state(env)
print(f"\n  After RRRRR from (3,32): {pos}")

# Does x=5 death prevent reaching x=8?
# YES - walking R from x=4 goes to x=5 which is death.
# Player never reaches x=8 because x=5 kills them first!

# So the gap at x=5-7 is a TRAP. Any rightward path from x=1-4
# must cross x=5 and die.

# Wait: can player go LEFT from x=8-9? No, they start at x=3.
# Can player TELEPORT? No.

# The player is TRAPPED in x=1-4, y=32-34 with grav UP.
# The only exit (x=5-7 gap) is lethal.

# UNLESS: G flip makes the exit safe!
# G flip → grav DOWN: player at (3,34) [or (4,34)]
# With grav DN, walk right: at x=5, y=31 gap is ABOVE player.
# With grav DN, player is on y=35 floor. y=31 gap is above.
# Player walks freely at y=34 to x=9.

# But then stuck at y=34 with grav DN, can't go up.

# UNLESS we then get another G flip... but G is consumed.

# UNLESS G blocks DON'T get consumed when clicked off-screen?
# Let me test this!


# ============================================================
# TEST E: Is G block consumed after off-screen click?
# ============================================================
print("\n=== TEST E: G block consumption test ===")
env = make_l7_env()
pos, grav = state(env)
print(f"  Start: {pos} grav={'UP' if grav else 'DN'}")

# Check G block at (5,2) before click
items = check_grid(env, 5, 2)
print(f"  G at (5,2) before: {items}")

# Click G
click_act(env, 5, 2)
pos, grav = state(env)
print(f"  After G flip: {pos} grav={'UP' if grav else 'DN'}")

# Check G block after click
items = check_grid(env, 5, 2)
print(f"  G at (5,2) after first click: {items}")

# Try clicking G again
click_act(env, 5, 2)
pos, grav = state(env)
print(f"  After 2nd click: {pos} grav={'UP' if grav else 'DN'}")

items = check_grid(env, 5, 2)
print(f"  G at (5,2) after 2nd click: {items}")

# UNDO and check
env.step(UNDO)
items = check_grid(env, 5, 2)
grav = state(env)[1]
print(f"  After UNDO: grav={'UP' if grav else 'DN'} G at (5,2): {items}")

env.step(UNDO)
items = check_grid(env, 5, 2)
grav = state(env)[1]
print(f"  After UNDO2: grav={'UP' if grav else 'DN'} G at (5,2): {items}")


# ============================================================
# TEST F: G flip from (4,32), then walk right through level with grav DOWN
# Map the ENTIRE reachable area with grav DOWN
# ============================================================
print("\n=== TEST F: Full grav DOWN exploration ===")
env = make_l7_env()
pos, grav = state(env)

# Walk to (4,32) first
env.step(RIGHT)
pos, _ = state(env)
print(f"  At: {pos}")

# G flip
click_act(env, 5, 2)
pos, grav = state(env)
print(f"  After G flip: {pos} grav={'UP' if grav else 'DN'}")

# Map reachable positions
print("  Exploring RIGHT:")
positions_right = [(pos[0], pos[1])]
for i in range(12):
    p1, _ = state(env)
    env.step(RIGHT)
    p2, _ = state(env)
    if p1 == p2:
        print(f"    Blocked at ({p2[0]},{p2[1]})")
        break
    y_note = f" Y:{p1[1]}→{p2[1]}" if p1[1] != p2[1] else ""
    print(f"    ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}){y_note}")
    positions_right.append((p2[0], p2[1]))

# Now walk left from rightmost position
print("  Exploring LEFT from rightmost:")
for i in range(15):
    p1, _ = state(env)
    env.step(LEFT)
    p2, _ = state(env)
    if p1 == p2:
        print(f"    Blocked at ({p2[0]},{p2[1]})")
        break
    y_note = f" Y:{p1[1]}→{p2[1]}" if p1[1] != p2[1] else ""
    print(f"    ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}){y_note}")
