#!/usr/bin/env python3
"""BP35 L3 exploration: understand grid, test paths, verify off-screen clicks."""
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


def click_act(env, gx, gy):
    engine = env._game.oztjzzyqoek
    cam_y = engine.camera.rczgvgfsfb[1]
    return env.step(CLICK, data={"x": gx*6, "y": gy*6 - cam_y})


def execute_and_advance(env, moves, level_name=""):
    old_level = env._game.level_index
    for m in moves:
        if m[0] == 'R': env.step(RIGHT)
        elif m[0] == 'L': env.step(LEFT)
        elif m[0] == 'C': click_act(env, m[1], m[2])
        if env._game.level_index > old_level:
            return True
    for _ in range(20):
        env.step(LEFT)
        if env._game.level_index > old_level:
            return True
    return False


def get_pos(engine):
    return tuple(engine.twdpowducb.qumspquyus)


def survey(engine, y_range=(-2, 40)):
    grid = engine.hdnrlfmyrj
    sym = {'xcjjwqfzjfe':'W', 'qclfkhjnaac':'D', 'fjlzdjxhant':'*',
           'ubhhgljbnpu':'^', 'hzusueifitk':'v', 'oonshderxef':'O',
           'yuuqpmlxorv':'B', 'lrpkmzabbfa':'G', 'etlsaqqtjvn':'E',
           'aknlbboysnc':'c', 'jcyhkseuorf':'F'}
    pp = get_pos(engine)
    for y in range(y_range[0], y_range[1]):
        row = ""
        has = False
        for x in range(-2, 15):
            items = grid.jhzcxkveiw(x, y)
            if (x,y) == pp: row += "P"; has = True
            elif items: row += sym.get(items[0].name,'?'); has = True
            else: row += "."
        if has: print(f"  y={y:3d}: {row}")
    grav = engine.vivnprldht
    print(f"  Player: {pp}, Grav up: {grav}")


L0_SOL = [R,R,R,R, C(7,19), C(4,16), L,L,L, C(4,15), C(4,12), R, C(5,9), L,L]
L1_SOL = [
    R,R,R,R,R, C(8,36), C(8,35),
    L,L, C(5,29),L, C(4,29),L, C(3,29),L, C(2,29),L,
    C(2,28), R,R,R, C(5,24), C(5,23),
    L,L, C(3,20), C(3,17), C(3,16),
    C(4,16),R, C(5,16),R, C(6,16),R, C(7,16),R, C(8,16),R,
    C(8,15), C(8,14), L, L, L, C(5,9), L, L,
]
L2_SOL = [
    C(5,28), R,R,R, C(6,27),
    C(5,23), C(4,23), C(3,23), L,L,L,L,
    R,
    C(5,17), C(6,17), C(5,18), C(6,18), R,R,R,R,
    C(6,12), C(5,12), C(4,12), C(3,12), L,L,L,L,
    C(5,7), R,R,R,R,
]


def setup_l3():
    arcade = Arcade()
    env = arcade.make('bp35-0a0ad940')
    env.reset()
    assert execute_and_advance(env, L0_SOL, "L0")
    assert execute_and_advance(env, L1_SOL, "L1")
    assert execute_and_advance(env, L2_SOL, "L2")
    assert env._game.level_index == 3
    print("L0-L2 solved, on L3")
    return env


# ============================================================
# Test 1: Clean L3 grid survey
# ============================================================
print("="*60)
print("L3 Clean Grid Survey")
print("="*60)
env = setup_l3()
engine = env._game.oztjzzyqoek
survey(engine, y_range=(-2, 40))

# Also check what the G block positions really are
grid = engine.hdnrlfmyrj
print("\nG block positions:")
for y in range(-2, 40):
    for x in range(-2, 15):
        items = grid.jhzcxkveiw(x, y)
        if items and items[0].name == 'lrpkmzabbfa':
            print(f"  G at ({x},{y})")

print("\nD block positions:")
for y in range(-2, 40):
    for x in range(-2, 15):
        items = grid.jhzcxkveiw(x, y)
        if items and items[0].name == 'qclfkhjnaac':
            print(f"  D at ({x},{y})")

print("\nSpike positions:")
for y in range(-2, 40):
    for x in range(-2, 15):
        items = grid.jhzcxkveiw(x, y)
        if items and items[0].name in ('ubhhgljbnpu', 'hzusueifitk'):
            print(f"  ^ at ({x},{y}) [{items[0].name[:8]}]")


# ============================================================
# Test 2: Test off-screen click
# ============================================================
print("\n" + "="*60)
print("Test 2: Off-screen click behavior")
print("="*60)
env2 = setup_l3()
engine2 = env2._game.oztjzzyqoek
pp = get_pos(engine2)
cam_y = engine2.camera.rczgvgfsfb[1]
print(f"Player: {pp}, cam_y={cam_y}")
print(f"Screen shows pixel y: {cam_y} to {cam_y+63}")
print(f"G(5,7) screen_y = {7*6-cam_y} (pixel)")
print(f"G(3,23) screen_y = {23*6-cam_y} (pixel)")

# Try clicking G(3,23) with off-screen y
print("\nAttempting off-screen click on G(3,23)...")
grav_before = engine2.vivnprldht
click_act(env2, 3, 23)
grav_after = engine2.vivnprldht
pp_after = get_pos(engine2)
print(f"  Gravity before: {grav_before}, after: {grav_after}")
print(f"  Player: {pp_after}")
if grav_before == grav_after:
    print("  OFF-SCREEN CLICK DID NOT WORK (gravity unchanged)")
else:
    print("  OFF-SCREEN CLICK WORKED!")


# ============================================================
# Test 3: Staircase climb + G(5,7) flip
# ============================================================
print("\n" + "="*60)
print("Test 3: Climb staircase, flip gravity via G(5,7)")
print("="*60)
env3 = setup_l3()
engine3 = env3._game.oztjzzyqoek

steps = [R, R, R, R]  # Climb staircase
for i, m in enumerate(steps):
    env3.step(RIGHT)
    pp = get_pos(engine3)
    print(f"  Step {i+1}: R → {pp}")

pp = get_pos(engine3)
cam_y = engine3.camera.rczgvgfsfb[1]
print(f"\nAt {pp}, cam_y={cam_y}")
print(f"G(5,7) screen_y = {7*6-cam_y}")

# Click G(5,7)
print("\nClicking G(5,7)...")
grav_before = engine3.vivnprldht
click_act(env3, 5, 7)
grav_after = engine3.vivnprldht
pp_after = get_pos(engine3)
print(f"  Gravity: {grav_before} → {grav_after}")
print(f"  Player: {pp_after}")

# Show current grid
survey(engine3, y_range=(14, 24))


# ============================================================
# Test 4: Full descent path attempt
# ============================================================
print("\n" + "="*60)
print("Test 4: Full path - flip, descend, flip back, reach gem")
print("="*60)
env4 = setup_l3()
engine4 = env4._game.oztjzzyqoek

path = [
    ("R", R), ("R", R), ("R", R), ("R", R),
    ("C(5,7) flip gravity", C(5,7)),
    ("L to (7,16)", L),
    ("L to (6,16)", L),
    ("L to (5,16)", L),
    ("L to (4,16)", L),
    ("L to (3,16)", L),
    ("L to (2,16)", L),
    ("C(3,17) destroy D (remote)", C(3,17)),  # Don't destroy (2,17)!
    ("C(2,17) destroy D → fall", C(2,17)),
]

for label, m in path:
    if m[0] == 'R': env4.step(RIGHT)
    elif m[0] == 'L': env4.step(LEFT)
    elif m[0] == 'C': click_act(env4, m[1], m[2])
    pp = get_pos(engine4)
    grav = engine4.vivnprldht
    print(f"  {label:40s} → {pp} grav_up={grav}")

# Now at (2,21) with gravity DOWN
print("\nGrid around y=19-25:")
survey(engine4, y_range=(19, 26))

# Try walking right to (3,21) → falls to where?
print("\nWalking R from (2,21)...")
env4.step(RIGHT)
pp = get_pos(engine4)
print(f"  R → {pp}")

# Check if G blocks are visible
cam_y = engine4.camera.rczgvgfsfb[1]
print(f"  cam_y={cam_y}")
print(f"  G(3,23) screen_y = {23*6-cam_y}")
print(f"  G(5,23) screen_y = {23*6-cam_y}")

# Click G(3,23) to flip gravity UP
print("\nClicking G(3,23) to flip gravity UP...")
grav_before = engine4.vivnprldht
click_act(env4, 3, 23)
grav_after = engine4.vivnprldht
pp_after = get_pos(engine4)
print(f"  Gravity: {grav_before} → {grav_after}")
print(f"  Player: {pp_after}")

# Now walk right to reach x=7-8 at y=18
print("\nWalking RIGHT to reach x=7-8...")
for i in range(5):
    env4.step(RIGHT)
    pp = get_pos(engine4)
    grav = engine4.vivnprldht
    print(f"  R → {pp} grav_up={grav}")

# Check if G(5,23) is visible
cam_y = engine4.camera.rczgvgfsfb[1]
pp = get_pos(engine4)
print(f"\nPlayer: {pp}, cam_y={cam_y}")
print(f"G(5,23) screen_y = {23*6-cam_y}")

# Try clicking G(5,23) — might be off-screen
print("\nClicking G(5,23)...")
grav_before = engine4.vivnprldht
click_act(env4, 5, 23)
grav_after = engine4.vivnprldht
pp_after = get_pos(engine4)
print(f"  Gravity: {grav_before} → {grav_after}")
print(f"  Player: {pp_after}")

survey(engine4, y_range=(14, 30))
