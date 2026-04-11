#!/usr/bin/env python3
"""L7: Step-by-step exploration with grid visualization after every move."""
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

def print_grid(env, y_range, x_range=(0,11), label=""):
    engine = env._game.oztjzzyqoek
    grid = engine.hdnrlfmyrj
    sym = {'xcjjwqfzjfe':'W', 'yuuqpmlxorv':'B', 'oonshderxef':'O',
           'ubhhgljbnpu':'^', 'fjlzdjxhant':'*', 'etlsaqqtjvn':'E',
           'lrpkmzabbfa':'G', 'hzusueifitk':'v', 'qclfkhjnaac':'D',
           'aknlbboysnc':'c'}
    pp = tuple(engine.twdpowducb.qumspquyus)
    if label: print(f"  {label}:")
    for y in range(y_range[0], y_range[1]):
        row = ""
        for x in range(x_range[0], x_range[1]):
            if (x, y) == pp:
                row += "P"
            else:
                items = grid.jhzcxkveiw(x, y)
                if items:
                    row += sym.get(items[0].name, '?')
                else:
                    row += "."
        print(f"    y={y:3d}: {row}")

step_n = [0]
def do(env, move, desc=""):
    step_n[0] += 1
    p1, g1 = state(env)
    if move[0] == 'R': env.step(RIGHT)
    elif move[0] == 'L': env.step(LEFT)
    elif move[0] == 'C': click_act(env, move[1], move[2])
    elif move[0] == 'U': env.step(UNDO)
    p2, g2 = state(env)
    dy = f" y:{p1[1]}→{p2[1]}" if p1[1] != p2[1] else ""
    gc = f" grav:{'UP' if g1 else 'DN'}→{'UP' if g2 else 'DN'}" if g1 != g2 else ""
    stuck = " STUCK" if p1 == p2 and move[0] in ('R', 'L') else ""
    won = " **WIN**" if env._game.level_index > 7 else ""
    print(f"  {step_n[0]:3d}. {str(move):14s} ({p1[0]:2d},{p1[1]:2d})→({p2[0]:2d},{p2[1]:2d}){dy}{gc}{stuck}{won} {desc}")
    return p2, g2

# ============================================================
# START
# ============================================================
env = make_l7_env()
print("=== L7 Exploration ===\n")
print("Initial grid:")
print_grid(env, (0, 36))

p, g = state(env)
print(f"\nPlayer: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}\n")

# Phase 1: E bridge at y=29-30 (5 clicks from (2,29) rightward)
print("--- Phase 1: E bridge ---")
for ex in range(2, 7):  # 2,3,4,5,6
    do(env, C(ex, 29), f"E spread ({ex},29)")
print_grid(env, (27, 33), label="Bridge area")

# Phase 2: Toggle B blocks for navigation
print("\n--- Phase 2: B toggles ---")
do(env, C(7, 25), "B(7,25)→O")
do(env, C(8, 22), "B(8,22)→O")

# Phase 3: Walk right across bridge to x=8 shaft
print("\n--- Phase 3: Walk to x=8 shaft ---")
for i in range(5):
    p, g = do(env, R, f"walk R step {i+1}")
print_grid(env, (22, 33), label="After walking R×5")

# Phase 4: Navigate upper area
print("\n--- Phase 4: Navigate upper area ---")
do(env, L, "L from x=8")  # through B(7,25)=O
do(env, L, "L to x=6")    # falls up through y=22 B gap

p, g = state(env)
print(f"  At: ({p[0]},{p[1]})")

do(env, R, "R back to x=7")
do(env, R, "R to x=8")  # through B(8,22)=O → falls up to y=21

p, g = state(env)
print(f"  At: ({p[0]},{p[1]})")
print_grid(env, (17, 26), label="Upper area")

# Explore: can we reach y=19 area?
print("\n--- Phase 5: Explore y=19-21 ---")
do(env, L, "L to x=7")
do(env, L, "L to x=6")  # should fall up if E ceiling at y=19

p, g = state(env)
print(f"  At: ({p[0]},{p[1]})")

# Walk along y=20 using E ceiling at y=19 (if it exists)
# But wait - we haven't done E spreading at y=18 yet!
# Let's check what's at y=19
print("\n  Checking y=18-19 at x=3-8:")
for x in range(3, 9):
    g18 = grid_at(env, x, 18)
    g19 = grid_at(env, x, 19)
    print(f"    ({x},18)={'E' if g18 == 'etlsaqqtjvn' else 'W' if g18 == 'xcjjwqfzjfe' else 'B' if g18 == 'yuuqpmlxorv' else '.' if g18 is None else '?'}"
          f"  ({x},19)={'.' if g19 is None else '*' if g19 == 'fjlzdjxhant' else 'W' if g19 == 'xcjjwqfzjfe' else '?'}")

# We need E spreading at y=18 to create ceilings at y=19!
# Let's UNDO back and do it before walking
print("\n--- UNDO back to start for re-plan ---")
for i in range(20):
    env.step(UNDO)
    p, g = state(env)
    if p == (3, 32) and g:
        print(f"  Reset after {i+1} UNDOs: ({p[0]},{p[1]}) grav=UP")
        break

# ============================================================
# REVISED PLAN: Do all clicks first, then walk
# ============================================================
print("\n=== REVISED: All clicks first, then walk ===\n")
step_n[0] = 0

# E bridge y=29
print("--- E bridge (y=29-30) ---")
for ex in range(2, 7):
    do(env, C(ex, 29), f"E({ex},29)")

# E ceiling y=18-19
print("\n--- E ceiling (y=18-19) ---")
do(env, C(3, 18), "E(3,18)")
do(env, C(4, 18), "E(4,18)")
do(env, C(5, 18), "E(5,18)")
do(env, C(6, 18), "E(6,18)")
print_grid(env, (16, 21), label="y=16-20 after E spread")

# Check y=17 E positions
print("\n  y=17 E positions:")
for x in range(0, 11):
    g = grid_at(env, x, 17)
    if g == 'etlsaqqtjvn':
        print(f"    E at ({x},17)")

# E ceiling extension y=16-17 for x=8 approach
print("\n--- E ceiling extension (y=16-17) ---")
# y=17 has E at x=3,4,5,6. Spread rightward:
do(env, C(6, 17), "E(6,17) → creates (7,17),(6,16),(6,18)")
do(env, C(7, 17), "E(7,17) → creates (6,17),(8,17),(7,16)")
do(env, C(8, 17), "E(8,17) → creates (7,17),(9,17),(8,16)")

print_grid(env, (14, 21), label="y=14-20 after all E spreads")

# Check what (8,16) and (8,17) look like
print(f"\n  (8,16): {grid_at(env, 8, 16)}")
print(f"  (8,17): {grid_at(env, 8, 17)}")
print(f"  (7,17): {grid_at(env, 7, 17)}")
print(f"  (9,17): {grid_at(env, 9, 17)}")

# Can we clear (7,17) to create a path to (8,17)?
# Click (7,17): removes it, adds E at empty neighbors
# Neighbors: (6,17)E skip, (8,17)empty→add!, (7,16)E skip, (7,18)wall skip
# Result: (7,17) cleared but (8,17) gets E. BAD.

# What if we try a different E pattern?
# What if we DON'T spread to (7,17) and (8,17)?
# Just stop at (6,17) and create E(6,16) as intermediate?

print("\n--- UNDO E extension and try alternate ---")
do(env, ('U',), "undo C(8,17)")
do(env, ('U',), "undo C(7,17)")
do(env, ('U',), "undo C(6,17)")

print_grid(env, (15, 21), label="After undoing extension")

# What if we spread E at (3,18) differently to create ceiling at (7,16) or (8,16)?
# Current y=17 E: x=3,4,5,6
# We need solid ceiling for (8,17). Options:
# - E at (8,16) via spreading
# - B at (8,18) is already there (ceiling if approaching from y=19)
# Wait - B(8,18) IS solid. With grav DOWN, player at (8,17) has FLOOR at (8,18) B.
# With grav UP, player at (8,19) has CEILING at (8,18) B.

# KEY INSIGHT: We don't need E ceiling at (8,16) if we approach
# from (8,19) with grav UP and ceiling B(8,18)!
# But we can't reach (8,19) because it's boxed in by walls.

# Alternative: what if we toggle B(8,18) to O, approach from above (y=17)
# with grav DOWN, fall through O to (8,19), walk R to gem?

# For that we need:
# 1. Player at (8,17) or above in x=8 column
# 2. Grav DOWN
# 3. B(8,18) = O
# Player falls: (8,17) → (8,18) O → (8,19) → (8,20) wall → lands at (8,19)
# Walk R → gem!

# But with grav DOWN at (8,17), what's the CEILING (above in grav sense = y+1)?
# Grav DOWN means floor is below (higher y). Ceiling is above (lower y).
# With grav DOWN: player at (8,17), floor below = (8,18).
# If (8,18) O: falls through. Good.
# No ceiling needed for grav DOWN (ceiling would be toward y=0, but we're falling away).

# So the plan: get to (8,17) with grav DOWN, toggle B(8,18)→O, fall to (8,19).
# OR: get to (8,17) with grav UP (need ceiling at (8,16)), flip G → grav DOWN,
# (8,18) B is now floor → player stays at (8,17). Then toggle B(8,18)→O → falls.

# For getting to (8,17) with grav UP + E(8,16) ceiling:
# The problem is the oscillation between (7,17) and (8,17).

# What if we click E ONLY to (7,17) and NOT to (8,17)?
# After C(6,17): E at (7,17), (6,16), (6,18). (6,17) empty.
# After C(7,17): E at (6,17), (8,17), (7,16). (7,17) empty.
#
# Now (8,17) has E. Player at (7,17) can walk R to (8,17)? No, (8,17) is E=solid.
# But (7,17) is empty. Player at (7,17) with ceiling (7,16) E. Safe!
# From (7,17) walk R: (8,17) E = solid. Blocked.
#
# What if we click (8,17) to spread it away?
# C(8,17): remove (8,17), add (7,17)? (7,17) empty→yes, (9,17) empty→yes, (8,16)→add, (8,18) B→skip
# Now: (7,17) E, (9,17) E, (8,16) E, (8,17) empty.
# But (7,17) has E again!

# WHAT IF we approach (8,17) not from x=7 but from (8,18)?
# Toggle B(8,18) → O. Player below at (8,19) with grav UP falls through O up to...
# (8,18) O → through → (8,17) → (8,16) E → ceiling! Land at (8,17).
# But we can't reach (8,19)! That's the gem approach position.

# CIRCULAR AGAIN. Let me try approaching from x=9.
# (9,17) has E after C(8,17). What if we click (9,17)?
# C(9,17): remove (9,17), add (8,17) if empty→yes, (10,17) wall skip, (9,16) empty→add, (9,18) wall→skip
# Now (8,17) E, (9,17) empty, (9,16) E.
# STILL fills (8,17)!

# The fundamental issue: (8,17) is always filled because it's the only empty neighbor
# when its adjacent E is clicked.

# NEW IDEA: What if we DON'T need the player at (8,17)?
# What if we use E spreading at (3,18) to create a ceiling at y=19 level,
# then walk along y=20, and somehow DROP down to (8,19) by flipping gravity?

# With grav UP at (6,20), ceiling E(6,19):
# Flip G → grav DOWN. Floor below (6,20) is (6,21) open → falls to (6,22) B.
# If B(6,22) solid: lands at (6,21). Then what? Walk R to (7,21), (8,21).
# Still at y=21. Can't reach y=19.

# What if B(7,22) is toggled to O? Then from (7,21) grav DOWN: (7,22) O → through → (7,23), (7,24) wall → land (7,23). Wrong direction.

# WHAT IF the approach is to use grav DOWN from the very start?
# Flip G at start: player (3,32) → (3,34) grav DOWN. Walk to x=9 at y=34.
# At (9,34): floor (9,35) wall.
# Now how to get from y=34 to y=19? 15 cells up. Can't with grav DOWN.
# G is consumed. No way to flip back.

# RETHINK: The G flip must happen AFTER reaching a position where
# the gravity change causes the player to land at a useful position.

# What positions near x=8-9 could lead to (8,19) after G flip?
# Player at (8,21), grav UP, ceiling (8,20) wall.
# G flip → grav DOWN, floor (8,22) O → falls through to (8,23) → (8,24) wall → lands (8,23).
# Wrong direction entirely.

# Player at (8,25), grav UP, ceiling (8,24) wall.
# G flip → grav DOWN, floor (8,26) wall → stays at (8,25). Then walk left...
# Still y=25 level. Can't reach y=19.

# The ONLY way: reach (8,17) with grav UP, flip G → grav DOWN at (8,17).
# With grav DOWN: floor is (8,18) B. If solid, player stays at (8,17).
# Then toggle B(8,18)→O → player falls to (8,19) → R → gem!

# And to reach (8,17), we MUST solve the E oscillation problem.

# BREAKTHROUGH IDEA: What if we put the player at (8,17) BEFORE creating E(8,16)?
# 1. Navigate to (8,17) — player falls up and dies (no ceiling → hits spike at 8,9)
# Wait, that kills the player. UNLESS we toggle some B block to be solid above?
# B blocks are at y=22. Too far from y=17.

# What if there's a D block or other entity we missed?
# Let me check column x=8 more carefully.

print("\n--- Column x=8 detailed ---")
for y in range(0, 36):
    g = grid_at(env, 8, y)
    if g:
        sym = {'xcjjwqfzjfe':'W', 'yuuqpmlxorv':'B', 'oonshderxef':'O',
               'ubhhgljbnpu':'^', 'etlsaqqtjvn':'E', 'lrpkmzabbfa':'G',
               'fjlzdjxhant':'*'}
        print(f"    (8,{y:2d}): {sym.get(g, g)}")

print("\n--- Column x=9 detailed ---")
for y in range(0, 36):
    g = grid_at(env, 9, y)
    if g:
        sym = {'xcjjwqfzjfe':'W', 'yuuqpmlxorv':'B', 'oonshderxef':'O',
               'ubhhgljbnpu':'^', 'etlsaqqtjvn':'E', 'lrpkmzabbfa':'G',
               'fjlzdjxhant':'*'}
        print(f"    (9,{y:2d}): {sym.get(g, g)}")

# Also check what entities exist at y=17-20 for x=7-9
print("\n--- y=17-20, x=7-9 before any E spreading ---")
for y in range(17, 21):
    for x in range(7, 10):
        g = grid_at(env, x, y)
        sym_map = {'xcjjwqfzjfe':'W', 'yuuqpmlxorv':'B', 'oonshderxef':'O',
               'ubhhgljbnpu':'^', 'etlsaqqtjvn':'E', 'lrpkmzabbfa':'G',
               'fjlzdjxhant':'*'}
        sym = sym_map.get(g, '.') if g else '.'
        print(f"    ({x},{y}): {sym}")

print("\n  Full grid after all E spreads (phase 1+2, no extension):")
print_grid(env, (0, 36))
