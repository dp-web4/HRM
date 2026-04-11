#!/usr/bin/env python3
"""Test L7: can we click G(5,2) from various positions? Test camera bounds and off-screen clicks."""
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

def click_raw(env, sx, sy):
    """Click at raw screen coordinates."""
    return env.step(CLICK, data={"x": sx, "y": sy})

def state(env):
    engine = env._game.oztjzzyqoek
    p = engine.twdpowducb
    return tuple(p.qumspquyus), engine.vivnprldht

def cam(env):
    return env._game.oztjzzyqoek.camera.rczgvgfsfb

def execute_level(env, moves):
    old_level = env._game.level_index
    for m in moves:
        if m[0] == 'R': env.step(RIGHT)
        elif m[0] == 'L': env.step(LEFT)
        elif m[0] == 'C':
            engine = env._game.oztjzzyqoek
            cy = engine.camera.rczgvgfsfb[1]
            env.step(CLICK, data={"x": m[1]*6, "y": m[2]*6 - cy})
        if env._game.level_index > old_level:
            return True
    for _ in range(10):
        env.step(LEFT)
        if env._game.level_index > old_level:
            return True
    return False

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
    ok = execute_level(env, sol)
    if not ok:
        print(f"L{i} FAIL"); sys.exit(1)
print("L0-L6 solved, now on L7\n")

pos, grav = state(env)
c = cam(env)
print(f"Start: pos={pos} grav={'UP' if grav else 'DN'} cam=({c[0]},{c[1]})")

# Test 1: Try clicking G(5,2) from start position with negative screen coords
print(f"\n=== Test 1: Click G(5,2) from start ({pos}) ===")
sy = 2*6 - c[1]
print(f"  Screen coords for G(5,2): ({5*6}, {sy})")
pos_before, grav_before = state(env)
click_act(env, 5, 2)
pos_after, grav_after = state(env)
print(f"  Result: pos {pos_before}→{pos_after} grav {'UP' if grav_before else 'DN'}→{'UP' if grav_after else 'DN'}")
if grav_before != grav_after:
    print("  G FLIP WORKED FROM FAR AWAY!")
    env.step(UNDO)
else:
    print("  No effect (expected - off screen)")

# Test 2: Check camera behavior - what's the viewport size?
print(f"\n=== Test 2: Camera/viewport analysis ===")
engine = env._game.oztjzzyqoek
camera = engine.camera
for attr in dir(camera):
    if not attr.startswith('_'):
        try:
            val = getattr(camera, attr)
            if not callable(val):
                print(f"  camera.{attr} = {val}")
        except:
            pass

# Test 3: Check the game/engine for viewport size
print(f"\n=== Test 3: Engine viewport info ===")
game = env._game
for attr in ['width', 'height', 'screen_width', 'screen_height', 'viewport']:
    if hasattr(game, attr):
        print(f"  game.{attr} = {getattr(game, attr)}")
for attr in dir(engine):
    if not attr.startswith('_') and any(k in attr.lower() for k in ['width', 'height', 'view', 'screen', 'size']):
        try:
            val = getattr(engine, attr)
            if not callable(val):
                print(f"  engine.{attr} = {val}")
        except:
            pass

# Test 4: Navigate closer and test G click at various y positions
# First, understand: where can we be alive with grav UP?
print(f"\n=== Test 4: Safe positions with grav UP ===")
# Walk left from start
pos0, _ = state(env)
print(f"  Start: {pos0}")
env.step(LEFT)
pos1, _ = state(env)
print(f"  After L: {pos1}")
env.step(LEFT)
pos2, _ = state(env)
print(f"  After LL: {pos2}")

# Undo back
env.step(UNDO); env.step(UNDO)

# Test: toggle some B blocks at y=22, then navigate up
# First let's see what happens if we toggle B(5,22) and try to go through
print(f"\n=== Test 5: Toggle B blocks and explore upward path ===")
# Toggle B blocks in the gap area (x=5,6,7) to open path
for bx in [5, 6, 7]:
    click_act(env, bx, 22)
    engine = env._game.oztjzzyqoek
    grid = engine.hdnrlfmyrj
    items = grid.jhzcxkveiw(bx, 22)
    names = [i.name for i in items] if items else []
    print(f"  Toggled ({bx},22): now {names}")

# Now try going right through x=5 gap
pos, grav = state(env)
print(f"\n  After toggles: pos={pos} grav={'UP' if grav else 'DN'}")

# Check: from (3,32), can we reach x=5 and fall up through y=31 gap,
# through y=27 spikes, through y=22 O-blocks?
# x=5 column going up: y=31 empty, y=30 empty, y=29 empty, y=28 empty,
# y=27 ^-spike SOLID → death at y=28
# Even with y=22 toggled, spikes at y=27 still block and kill

# What if we toggle B(7,25) to O and try the x=8-9 shaft?
print(f"\n=== Test 6: Check x=8-9 shaft viability ===")
# x=8 column from y=28 up: y=27 EMPTY (no spike at x=8), y=26 EMPTY (no wall at x=8)
# y=25 has B(7,25)? No, check: B positions are (7,25) not (8,25)
# Actually let me verify: what's at x=8, y=21-28?
engine = env._game.oztjzzyqoek
grid = engine.hdnrlfmyrj
print(f"  x=8 column y=18-28:")
for y in range(18, 29):
    items = grid.jhzcxkveiw(8, y)
    names = [(i.name, i.collidable) for i in items] if items else []
    print(f"    (8,{y}): {names}")

print(f"  x=9 column y=18-28:")
for y in range(18, 29):
    items = grid.jhzcxkveiw(9, y)
    names = [(i.name, i.collidable) for i in items] if items else []
    print(f"    (9,{y}): {names}")

# Undo the B toggles
print(f"\n  Undoing toggles...")
env.step(UNDO); env.step(UNDO); env.step(UNDO)
pos, grav = state(env)
print(f"  After undo: pos={pos}")

# Test 7: What if we toggle B(8,22) to O and enter from x=8?
# From start, walk to (8,32). Above is (8,31) = wall. Stuck.
# What about (9,32)? (9,31) = wall. Stuck.
# Only exit from bottom is x=5-7 gap at y=31.

# Test 8: Check if E at (2,29) can be used as a stopping point
print(f"\n=== Test 7: E(2,29) as stopping point ===")
# If player is at x=2 and falls up, E(2,29) is at y=29.
# But (2,31) is wall, so player at (2,32) has wall ceiling → stuck.
# What if we walk right, enter x=5-7 gap, and somehow land on E?
# E is at x=2, gap is at x=5-7. Different columns.
# Unless we fall up from x=5-7, land somewhere, then walk to x=2

# Let's test: what's at x=5-7 y=28-31?
print(f"  x=5 y=27-31:")
for y in range(27, 32):
    items = grid.jhzcxkveiw(5, y)
    names = [i.name for i in items] if items else []
    print(f"    (5,{y}): {names}")

# Test 8: Try navigating without dying - go through gap and check
# The key question: is there ANY safe landing at x=5-7 between y=28 and y=31?
print(f"\n=== Test 8: Is (5,28) actually death? ===")
# Move right twice to trigger the fall
env.step(RIGHT)  # (3,32)→(4,32)
env.step(RIGHT)  # (4,32)→(5,28) via gap, spike death?
pos, grav = state(env)
print(f"  After R,R: pos={pos} grav={'UP' if grav else 'DN'}")

# Test if player is dead by trying to move
env.step(RIGHT)
pos2, _ = state(env)
print(f"  After R,R,R: pos={pos2} {'DEAD' if pos==pos2 else 'alive'}")
env.step(LEFT)
pos3, _ = state(env)
print(f"  After R,R,R,L: pos={pos3} {'DEAD' if pos==pos3 else 'alive'}")

# Check camera at death position
c = cam(env)
print(f"  Camera at death: ({c[0]},{c[1]})")
sy_g = 2*6 - c[1]
print(f"  G(5,2) screen_y from here: {sy_g}")

# Even if dead, try clicking G
click_act(env, 5, 2)
_, grav_after = state(env)
print(f"  Click G(5,2) after death: grav {'UP' if grav_after else 'DN'}")

# Undo everything
for _ in range(5):
    env.step(UNDO)
pos_reset, _ = state(env)
print(f"  After undos: {pos_reset}")

# Test 9: What if we toggle ALL B blocks at y=22, then go through x=5-7?
# Player would fall through y=27 spike → death. Unless the B block row
# somehow interferes? No, B blocks are at y=22, spikes at y=27.
# The spikes still kill.

# Test 10: What's the screen/viewport height?
print(f"\n=== Test 9: Frame analysis ===")
fd = env._last_frame_data if hasattr(env, '_last_frame_data') else None
frame = env._game.render()
print(f"  Frame shape: {frame.shape if hasattr(frame, 'shape') else type(frame)}")
if hasattr(frame, 'shape'):
    h, w = frame.shape[-2], frame.shape[-1]
    print(f"  Height: {h} pixels = {h/6} grid cells")
    print(f"  Width: {w} pixels = {w/6} grid cells")

    # Calculate: from y=11 (safe in open area), can we see G at y=2?
    cam_at_11 = 11*6 - 36  # = 30
    sy_g_at_11 = 2*6 - cam_at_11  # = -18
    print(f"  At y=11: cam_y={cam_at_11}, G screen_y={sy_g_at_11}, viewport_h={h}")

    # What y do we need to see G(5,2)?
    # Need: 2*6 - cam_y >= 0 → cam_y <= 12 → player_y*6 - 36 <= 12 → player_y <= 8
    # Need: 2*6 - cam_y < h → cam_y > 12 - h → player_y > (48 - h)/6
    print(f"  Need player_y <= 8 to see G on screen (cam_y <= 12)")
    print(f"  Or if viewport = {h}, player_y > {(48-h)/6:.1f}")
