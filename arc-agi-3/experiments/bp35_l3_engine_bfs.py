#!/usr/bin/env python3
"""BP35 L3 BFS using actual game engine for state transitions.
Uses save/restore via UNDO/RESET to explore the tree.
"""
import sys, os, time, copy, pickle, hashlib
from collections import deque
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


def execute_and_advance(env, moves):
    old = env._game.level_index
    for m in moves:
        if m[0] == 'R': env.step(RIGHT)
        elif m[0] == 'L': env.step(LEFT)
        elif m[0] == 'C': click_act(env, m[1], m[2])
        if env._game.level_index > old: return True
    for _ in range(20):
        env.step(LEFT)
        if env._game.level_index > old: return True
    return False


def get_state_key(engine):
    """Extract hashable state from engine."""
    p = engine.twdpowducb
    pp = tuple(p.qumspquyus)
    grav = engine.vivnprldht
    grid = engine.hdnrlfmyrj

    # Survey relevant cells for D and G blocks
    d_state = []
    g_state = []
    for pos in [(2,17),(3,17),(5,19),(6,19),(7,19),(8,19),(7,23),(8,23),(7,24),(8,24)]:
        items = grid.jhzcxkveiw(pos[0], pos[1])
        d_state.append(1 if items and items[0].name == 'qclfkhjnaac' else 0)
    for pos in [(5,7),(3,23),(5,23)]:
        items = grid.jhzcxkveiw(pos[0], pos[1])
        g_state.append(1 if items and items[0].name == 'lrpkmzabbfa' else 0)

    return (pp[0], pp[1], grav, tuple(d_state), tuple(g_state))


def get_clickable(engine):
    """Get list of clickable blocks (D and G)."""
    grid = engine.hdnrlfmyrj
    clickable = []
    for y in range(-2, 35):
        for x in range(-2, 12):
            items = grid.jhzcxkveiw(x, y)
            if items:
                name = items[0].name
                if name in ('qclfkhjnaac', 'lrpkmzabbfa'):
                    clickable.append((x, y, name))
    return clickable


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


# Setup and verify
print("Setting up L3...")
arcade = Arcade()
env = arcade.make('bp35-0a0ad940')
env.reset()
assert execute_and_advance(env, L0_SOL)
assert execute_and_advance(env, L1_SOL)
assert execute_and_advance(env, L2_SOL)
assert env._game.level_index == 3

engine = env._game.oztjzzyqoek
pp = tuple(engine.twdpowducb.qumspquyus)
grav = engine.vivnprldht
print(f"L3 start: player={pp}, grav_up={grav}")

# Get clickable blocks
clickable = get_clickable(engine)
print(f"Clickable blocks: {len(clickable)}")
for x, y, name in clickable:
    print(f"  ({x},{y}): {name[:12]}")

# State key
sk = get_state_key(engine)
print(f"State key: {sk}")

# Test: verify staircase positions match BFS grid
print("\nVerifying staircase (4 RIGHT moves):")
test_env = copy.deepcopy(env)
test_engine = test_env._game.oztjzzyqoek
for i in range(4):
    test_env.step(RIGHT)
    pp = tuple(test_engine.twdpowducb.qumspquyus)
    print(f"  R → {pp}")

# Test: click G(3,23) from start (off-screen)
print("\nTest: click G(3,23) from (4,14):")
test_env2 = copy.deepcopy(env)
test_engine2 = test_env2._game.oztjzzyqoek
cam_y = test_engine2.camera.rczgvgfsfb[1]
print(f"  cam_y={cam_y}, G(3,23) screen_y={23*6-cam_y}")
test_env2.step(CLICK, data={"x": 3*6, "y": 23*6 - cam_y})
pp = tuple(test_engine2.twdpowducb.qumspquyus)
grav = test_engine2.vivnprldht
print(f"  → player={pp}, grav_up={grav}")

# Test: verify fall through destroyed D
print("\nTest: destroy D(2,17) from (2,16) with gravity DOWN:")
test_env3 = copy.deepcopy(env)
test_engine3 = test_env3._game.oztjzzyqoek
# First flip gravity: click G(5,23)
cam_y = test_engine3.camera.rczgvgfsfb[1]
test_env3.step(CLICK, data={"x": 5*6, "y": 23*6 - cam_y})
pp = tuple(test_engine3.twdpowducb.qumspquyus)
grav = test_engine3.vivnprldht
print(f"  After G(5,23): player={pp}, grav_up={grav}")
# Walk left to (2,16)
for _ in range(2):
    test_env3.step(LEFT)
pp = tuple(test_engine3.twdpowducb.qumspquyus)
print(f"  At (2,16): player={pp}")
# Destroy (2,17)
cam_y = test_engine3.camera.rczgvgfsfb[1]
test_env3.step(CLICK, data={"x": 2*6, "y": 17*6 - cam_y})
pp = tuple(test_engine3.twdpowducb.qumspquyus)
print(f"  After C(2,17): player={pp}")
# Walk right to (3,21)
test_env3.step(RIGHT)
pp = tuple(test_engine3.twdpowducb.qumspquyus)
print(f"  R: player={pp}")
# Click G(3,23)
cam_y = test_engine3.camera.rczgvgfsfb[1]
test_env3.step(CLICK, data={"x": 3*6, "y": 23*6 - cam_y})
pp = tuple(test_engine3.twdpowducb.qumspquyus)
grav = test_engine3.vivnprldht
print(f"  After G(3,23): player={pp}, grav_up={grav}")
# Walk right to (8,18)
for i in range(5):
    test_env3.step(RIGHT)
    pp = tuple(test_engine3.twdpowducb.qumspquyus)
    print(f"  R: player={pp}")

# From here, check what G blocks remain
sk = get_state_key(test_engine3)
print(f"\nState key: {sk}")
print(f"G blocks alive: (5,7)={sk[4][0]}, (3,23)={sk[4][1]}, (5,23)={sk[4][2]}")

# Click G(5,7) from (8,18) — far off screen above
cam_y = test_engine3.camera.rczgvgfsfb[1]
print(f"\ncam_y={cam_y}, G(5,7) screen_y={7*6-cam_y}")
test_env3.step(CLICK, data={"x": 5*6, "y": 7*6 - cam_y})
pp = tuple(test_engine3.twdpowducb.qumspquyus)
grav = test_engine3.vivnprldht
print(f"After G(5,7): player={pp}, grav_up={grav}")

# Check if we died (check level status)
lvl = test_env3._game.level_index
print(f"Level: {lvl}")

# Show grid around player
grid = test_engine3.hdnrlfmyrj
sym = {'xcjjwqfzjfe':'W', 'qclfkhjnaac':'D', 'fjlzdjxhant':'*',
       'ubhhgljbnpu':'^', 'hzusueifitk':'v', 'oonshderxef':'O',
       'yuuqpmlxorv':'B', 'lrpkmzabbfa':'G'}
for y in range(16, 32):
    row = ""
    has = False
    for x in range(-2, 12):
        items = grid.jhzcxkveiw(x, y)
        if (x,y) == pp: row += "P"; has = True
        elif items: row += sym.get(items[0].name,'?'); has = True
        else: row += "."
    if has: print(f"  y={y:3d}: {row}")

# Now try to descend through D chain
print("\nDescending D chain at x=8:")
# Destroy (8,19)
cam_y = test_engine3.camera.rczgvgfsfb[1]
test_env3.step(CLICK, data={"x": 8*6, "y": 19*6 - cam_y})
pp = tuple(test_engine3.twdpowducb.qumspquyus)
print(f"  C(8,19): {pp}")

if pp[1] >= 19:  # fell down
    test_env3.step(CLICK, data={"x": 8*6, "y": 23*6 - test_engine3.camera.rczgvgfsfb[1]})
    pp = tuple(test_engine3.twdpowducb.qumspquyus)
    print(f"  C(8,23): {pp}")

    test_env3.step(CLICK, data={"x": 8*6, "y": 24*6 - test_engine3.camera.rczgvgfsfb[1]})
    pp = tuple(test_engine3.twdpowducb.qumspquyus)
    print(f"  C(8,24): {pp}")

    # Walk left to x=4
    for i in range(4):
        test_env3.step(LEFT)
        pp = tuple(test_engine3.twdpowducb.qumspquyus)
        print(f"  L: {pp}")

    grav = test_engine3.vivnprldht
    print(f"\nFinal: player={pp}, grav_up={grav}")
    print("Gem at (4,27). With gravity DOWN, can't reach.")

    # Check if UNDO works
    print("\nTesting UNDO from here:")
    test_env3.step(UNDO)
    pp = tuple(test_engine3.twdpowducb.qumspquyus)
    grav = test_engine3.vivnprldht
    print(f"  UNDO: player={pp}, grav_up={grav}")

    test_env3.step(UNDO)
    pp = tuple(test_engine3.twdpowducb.qumspquyus)
    grav = test_engine3.vivnprldht
    print(f"  UNDO: player={pp}, grav_up={grav}")
