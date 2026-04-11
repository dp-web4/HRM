#!/usr/bin/env python3
"""L7: Fresh env, all clicks first, then walk. Verify E spreading works."""
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

def sym_at(env, x, y):
    g = grid_at(env, x, y)
    return SYM.get(g, '.') if g else '.'

def print_grid(env, y_range, x_range=(0,11), label=""):
    pp = tuple(env._game.oztjzzyqoek.twdpowducb.qumspquyus)
    if label: print(f"  {label}:")
    for y in range(y_range[0], y_range[1]):
        row = ""
        for x in range(x_range[0], x_range[1]):
            if (x, y) == pp: row += "P"
            else: row += sym_at(env, x, y)
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
env = make_l7_env()
print("=== L7 Fresh Exploration ===\n")
p, g = state(env)
print(f"Start: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} level={env._game.level_index}\n")

# ============================================================
# Step 1: E bridge at y=29-30 (5 clicks)
# ============================================================
print("--- Step 1: E bridge ---")
for ex in range(2, 7):
    do(env, C(ex, 29))
print_grid(env, (27, 33), label="Bridge")

# ============================================================
# Step 2: E ceiling at y=18-19 (4 clicks)
# ============================================================
print("\n--- Step 2: E ceiling at y=18-19 ---")
do(env, C(3, 18), "spread E(3,18)")
# Verify E spread
print(f"  After C(3,18): E at y=17? {sym_at(env,3,17)}, y=18? {sym_at(env,3,18)}, y=19? {sym_at(env,3,19)}")
print(f"    (2,18)={sym_at(env,2,18)} (4,18)={sym_at(env,4,18)}")

do(env, C(4, 18), "spread E(4,18)")
print(f"  After C(4,18): (3,18)={sym_at(env,3,18)} (5,18)={sym_at(env,5,18)} (4,17)={sym_at(env,4,17)}")

do(env, C(5, 18), "spread E(5,18)")
do(env, C(6, 18), "spread E(6,18)")

print_grid(env, (16, 21), label="After E ceiling spread")

# Show E positions at y=17, 18, 19
print("\n  E positions:")
for y in [17, 18, 19]:
    es = [x for x in range(11) if grid_at(env, x, y) == 'etlsaqqtjvn']
    print(f"    y={y}: {es}")

# ============================================================
# Step 3: B toggles
# ============================================================
print("\n--- Step 3: B toggles ---")
do(env, C(7, 25), "B(7,25)→O")
do(env, C(8, 22), "B(8,22)→O")

# ============================================================
# Step 4: Walk across bridge to x=8 shaft
# ============================================================
print("\n--- Step 4: Walk to x=8 shaft ---")
for i in range(5):
    do(env, R, f"R{i+1}")
p, g = state(env)
print(f"  Position: ({p[0]},{p[1]})")

# ============================================================
# Step 5: Navigate to y=21 via B gaps
# ============================================================
print("\n--- Step 5: Navigate upper area ---")
do(env, L, "L to x=7 via O(7,25)")
do(env, L, "L to x=6 → fall up")
p, g = state(env)
print(f"  At: ({p[0]},{p[1]})")

do(env, R, "R to x=7")
do(env, R, "R to x=8 → fall up through O(8,22)")
p, g = state(env)
print(f"  At: ({p[0]},{p[1]})")

# ============================================================
# Step 6: Navigate using E ceiling at y=19
# ============================================================
print("\n--- Step 6: Navigate using E ceiling ---")
do(env, L, "L to x=7")
p, g = state(env)
print(f"  At: ({p[0]},{p[1]}) [ceiling at (7,20) should be wall]")

do(env, L, "L to x=6 → fall up to E ceiling")
p, g = state(env)
print(f"  At: ({p[0]},{p[1]}) [should be (6,20) with ceiling E(6,19)]")

# Continue west along E ceiling
for i in range(4):
    do(env, L, f"L along E ceiling")
    p, g = state(env)
    print(f"  At: ({p[0]},{p[1]})")

# Now walk right back along E ceiling to explore
print("\n--- Step 7: Walk R along E ceiling toward x=7 ---")
for i in range(6):
    do(env, R, f"R along E ceiling")
    p, g = state(env)
    print(f"  At: ({p[0]},{p[1]})")

# Where can we get to from the E ceiling level?
# At (7,21) ceiling (7,20) wall.
# Can we reach (8,21)? Walk R from (7,21).
# But earlier we showed (8,21) accessible from (7,23).

print(f"\n  Final: ({p[0]},{p[1]})")
print_grid(env, (6, 22), label="Upper area after navigation")
print(f"\n  Level: {env._game.level_index}")
