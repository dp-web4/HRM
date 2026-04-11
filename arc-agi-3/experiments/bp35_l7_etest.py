#!/usr/bin/env python3
"""L7: Test E spreading into player cell + test grav DOWN path to (8,17)."""
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
U = ('U',)
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

def grid_at(env, x, y):
    engine = env._game.oztjzzyqoek
    items = engine.hdnrlfmyrj.jhzcxkveiw(x, y)
    return items[0].name if items else None

SYM = {'xcjjwqfzjfe':'W', 'yuuqpmlxorv':'B', 'oonshderxef':'O',
       'ubhhgljbnpu':'^', 'fjlzdjxhant':'*', 'etlsaqqtjvn':'E',
       'lrpkmzabbfa':'G', 'hzusueifitk':'v', 'qclfkhjnaac':'D',
       'aknlbboysnc':'c'}

def sym(env, x, y):
    g = grid_at(env, x, y)
    return SYM.get(g, '.') if g else '.'

def print_grid(env, yr, xr=(0,11), label=""):
    pp = tuple(env._game.oztjzzyqoek.twdpowducb.qumspquyus)
    if label: print(f"  {label}:")
    for y in range(yr[0], yr[1]):
        row = ""
        for x in range(xr[0], xr[1]):
            if (x, y) == pp: row += "P"
            else: row += sym(env, x, y)
        print(f"    y={y:3d}: {row}")

# ============================================================
# TEST 1: E spreading into player cell
# ============================================================
print("=== TEST 1: E spread into player cell ===")
env = make_l7_env()

# Setup: E ceiling at y=18-19, walk player to (5,20)
for ex in range(2, 7): click_act(env, ex, 29)  # E bridge
for ex in [3, 4, 5, 6]: click_act(env, ex, 18)  # E ceiling
click_act(env, 7, 25)  # B→O
click_act(env, 8, 22)  # B→O

# Walk to (5,20)
for _ in range(5): env.step(RIGHT)  # → (8,25)
env.step(LEFT); env.step(LEFT)  # → (6,23)
env.step(RIGHT); env.step(RIGHT)  # → (8,21)
env.step(LEFT)  # → (7,21)
env.step(LEFT)  # → (6,20)
env.step(LEFT)  # → (5,20)

p, g = state(env)
print(f"  Player at: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
print(f"  (5,19)={sym(env,5,19)} (5,20)={sym(env,5,20)}")
print_grid(env, (17, 22), label="Before E click")

# Click E(5,19) — should create E at (5,20) where player stands!
print("\n  Clicking E(5,19)...")
click_act(env, 5, 19)
p2, g2 = state(env)
print(f"  Player now: ({p2[0]},{p2[1]}) grav={'UP' if g2 else 'DN'}")
print(f"  (5,19)={sym(env,5,19)} (5,20)={sym(env,5,20)}")
print_grid(env, (15, 22), label="After clicking E(5,19)")

# Undo and try different E clicks
env.step(UNDO)
p3, g3 = state(env)
print(f"\n  After UNDO: ({p3[0]},{p3[1]})")

# ============================================================
# TEST 2: Can clicking E pull player upward?
# ============================================================
print("\n=== TEST 2: Click E(5,18) while at (5,20) ===")
# E(5,18) exists. Clicking it spreads. Where does (5,17) E go?
# After Phase 2: y=18 E at 2,3,4,5. Click E(5,18):
# Remove (5,18). Add: (4,18)E→skip, (6,18)empty→add, (5,17)E→skip, (5,19)E→skip
# Creates (6,18)E. Removes (5,18).
# Player at (5,20) ceiling was E(5,19). Still there. No change.

click_act(env, 5, 18)
p, g = state(env)
print(f"  After C(5,18): ({p[0]},{p[1]})")
print(f"  (5,17)={sym(env,5,17)} (5,18)={sym(env,5,18)} (5,19)={sym(env,5,19)}")
env.step(UNDO)

# ============================================================
# TEST 3: What if we remove E(5,19) from a DIFFERENT position?
# ============================================================
print("\n=== TEST 3: Remove ceiling while player elsewhere ===")
# Player at (5,20). Walk L to (4,20), then click E(5,19)
env.step(LEFT)  # to (4,20)
p, g = state(env)
print(f"  At: ({p[0]},{p[1]})")

click_act(env, 5, 19)
p, g = state(env)
print(f"  After C(5,19) from (4,20): ({p[0]},{p[1]})")
print(f"  (5,19)={sym(env,5,19)} (5,20)={sym(env,5,20)}")

# Now go back to x=5: what happens?
env.step(RIGHT)
p, g = state(env)
print(f"  Walk R to x=5: ({p[0]},{p[1]})")
# If (5,20) now has E, player can't enter
# If (5,19) empty, player at (5,20) would fall up through (5,19) to...
print_grid(env, (15, 22), (3, 8), label="x=3-7")

# Undo all
for _ in range(3): env.step(UNDO)

# ============================================================
# TEST 4: Approach (8,17) via grav DOWN from upper area
# Navigate player to (6,20), flip G, see where they end up
# ============================================================
print("\n=== TEST 4: G flip from (6,20) ===")
# Player should be at (5,20) after undos. Walk R to (6,20)
env.step(RIGHT)
p, g = state(env)
print(f"  At: ({p[0]},{p[1]})")

if p == (6, 20):
    # Flip G
    click_act(env, 5, 2)
    p, g = state(env)
    print(f"  After G flip: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

    # Walk to see where we end up
    for i in range(6):
        p1 = state(env)[0]
        env.step(RIGHT)
        p2 = state(env)[0]
        print(f"  R: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]})")
        if p1 == p2: break

    print_grid(env, (17, 26), label="After G flip + walk R")

    # Undo G flip and walks
    for _ in range(10): env.step(UNDO)

# ============================================================
# TEST 5: What if we DON'T do E ceiling spread, and instead:
# 1. Walk to (8,25) via E bridge
# 2. Navigate to (7,21) via B toggles
# 3. From (7,21), walk L to (6,21) — player falls to (6,7) DEAD
# BUT: what if G flip at (7,21)? Player goes to grav DOWN,
# floor (7,22) B → stays at (7,21). Walk L to (6,21),
# floor (6,22) B → stays at (6,21).
# Then walk L to... eventually reach position where toggling
# floor B causes fall through to y=17 area.
# ============================================================
print("\n=== TEST 5: G flip at (8,21) then walk along y=21 ===")
env = make_l7_env()

# E bridge only
for ex in range(2, 7): click_act(env, ex, 29)
# E ceiling for navigation
for ex in [3, 4, 5, 6]: click_act(env, ex, 18)
# B toggles
click_act(env, 7, 25)  # B→O
click_act(env, 8, 22)  # B→O

# Walk to (8,21)
for _ in range(5): env.step(RIGHT)  # → (8,25)
env.step(LEFT); env.step(LEFT)  # → (6,23)
env.step(RIGHT); env.step(RIGHT)  # → (8,21)

p, g = state(env)
print(f"  At: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Toggle B(8,22) BACK to B (so we have floor)
click_act(env, 8, 22)
print(f"  B(8,22) toggled back to B")

# G flip
click_act(env, 5, 2)
p, g = state(env)
print(f"  After G: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Walk left along y=21 with B floor at y=22
for i in range(8):
    p1 = state(env)[0]
    env.step(LEFT)
    p2, g2 = state(env)
    dy = f" y:{p1[1]}→{p2[0]}" if p1[1] != p2[1] else ""
    print(f"  L: ({p1[0]},{p1[1]})→({p2[0]},{p2[1]}){dy}")
    if p1 == p2: break

p, g = state(env)
print(f"\n  Final: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")

# Now what? We're at y=21 grav DOWN. Can't go up.
# BUT: what if we toggle B below us and fall through a column that has E?
# y=22 B → toggle to O → fall through y=23 → y=24 → ... → y=26 wall
# E at y=17 is ABOVE us. Can't reach.

# TEST: from (3,21) grav DOWN, toggle B(3,22)→O
if p[1] == 21:
    print(f"\n  Toggle B(3,22)→O at ({p[0]},21)...")
    click_act(env, p[0], 22)
    p2, g2 = state(env)
    print(f"  After toggle: ({p2[0]},{p2[1]}) grav={'UP' if g2 else 'DN'}")

    print_grid(env, (16, 27), label="After B toggle")

# ============================================================
# TEST 6: Read source code for E click handler more carefully
# What exactly happens after E spread? Does it re-check player?
# ============================================================
print("\n=== TEST 6: Source code check ===")
import inspect
game = env._game
engine = game.oztjzzyqoek

# Find the click handler
click_method = getattr(engine, 'gwfodrkvzx', None)
if click_method:
    try:
        src = inspect.getsource(click_method)
        # Find the E-related section
        lines = src.split('\n')
        for i, line in enumerate(lines):
            if 'etlsaqqtjvn' in line or 'fsvnqdbzrp' in line or 'player' in line.lower():
                start = max(0, i-2)
                end = min(len(lines), i+5)
                for j in range(start, end):
                    print(f"  {j:4d}: {lines[j]}")
                print("  ---")
    except:
        print("  (can't inspect)")
