#!/usr/bin/env python3
"""L8: Quick diagnosis — why does push-up at (9,22) kill?
And: can we clear x=9 column before walking in?"""
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

def is_alive(env):
    return env._game.oztjzzyqoek.twdpowducb.etquaizpmu

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

def run_to_823(env):
    for move in SETUP_NAV:
        if move[0] == 'R': env.step(RIGHT)
        elif move[0] == 'L': env.step(LEFT)
        elif move[0] == 'C': click_act(env, move[1], move[2])

# ============================================================
print("=== TEST 1: Push-up death at (9,22) ===\n")
env = make_l8_env()
run_to_823(env)
p, g = state(env)
print(f"At ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} alive={is_alive(env)}")

# Clear E(9,23) and walk R
click_act(env, 9, 23)
env.step(RIGHT)
p, g = state(env)
print(f"Walk R to: ({p[0]},{p[1]}) alive={is_alive(env)}")

# Check entities before push
print(f"\nBefore push:")
for dy in [-2,-1,0,1,2]:
    y = p[1]+dy
    ents = [(i.name, SYM.get(i.name,'?')) for i in env._game.oztjzzyqoek.hdnrlfmyrj.jhzcxkveiw(9, y)]
    print(f"  (9,{y}): {ents}")
for dx in [-1,1]:
    x = 9+dx
    ents = [(i.name, SYM.get(i.name,'?')) for i in env._game.oztjzzyqoek.hdnrlfmyrj.jhzcxkveiw(x, p[1])]
    print(f"  ({x},{p[1]}): {ents}")

# Do push-up
print(f"\nPush-up C(9,22):")
click_act(env, 9, 22)
p, g = state(env)
print(f"  After: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} alive={is_alive(env)}")

# Check entities after
print(f"\nAfter push:")
for dy in [-2,-1,0,1,2]:
    y = p[1]+dy
    ents = [(i.name, SYM.get(i.name,'?')) for i in env._game.oztjzzyqoek.hdnrlfmyrj.jhzcxkveiw(9, y)]
    print(f"  (9,{y}): {ents}")

# ============================================================
print(f"\n\n=== TEST 2: Clear x=9 then walk R (direct flight) ===\n")
env2 = make_l8_env()
run_to_823(env2)
p, g = state(env2)
print(f"At ({p[0]},{p[1]}) alive={is_alive(env2)}")

# Clear E(9,23)
click_act(env2, 9, 23)  # creates E(9,22), E(9,24)
# Clear E(9,22)
click_act(env2, 9, 22)  # creates E(9,21), E(9,23)

# Check state
print(f"After 2 clears: E at x=9:")
for y in range(15, 26):
    s = sym(env2, 9, y)
    if s != '.': print(f"  (9,{y})={s}")

# E oscillation means we can't clear the column.
# But what about clearing from a different approach?
# What if E(9,23) is cleared and we walk R BEFORE E(9,22) matters?
# Player walks R to (9,23). If (9,22) has E, player stops at (9,23).
# That's fine IF we then clear (9,22) differently.

# Actually — key question: when player walks into (9,23) with grav UP,
# does the game check all cells above for the first solid?
# If (9,22)=E, player lands at (9,23). If (9,22)=empty, player flies up.

# So the approach: clear E(9,23) only (1 click), walk R.
# Player enters (9,23) with E(9,22) above → stops at (9,23).
# Then need to get from (9,23) to (9,16) without push-up.

# What if we G flip DN while at (9,23)?
# With grav DN at (9,23): fall to (9,24)=E → floor. Player at (9,23).
# Clear E(9,24): fall to (9,24), then (9,25)=v lethal with DN.

# What if we clear E(9,22) while player is at (9,23)?
# C(9,22) from (9,23): E is NOT adjacent to player?
# (9,22) is adjacent to (9,23). Clicking E(9,22):
# removes E(9,22). Creates E(9,21),(9,23)=player→no, (8,22)=W, (10,22)=W.
# Player at (9,23) with grav UP: ceiling was E(9,22), now removed.
# Player flies up through (9,22)(empty), (9,21)(E just created)→stops.
# Player at (9,22) with E(9,21) ceiling.
# But (9,23) is occupied by... nothing? E wasn't created there (player).
# Below: (9,23)=empty(player just left), (9,24)=E.
# So (9,23) gets E? Wait, player moved UP. (9,23) is now empty.
# E side-effects are: (9,21), (9,23)→was player there when E was created?

# The timing matters. Let me just test it:
print(f"\n--- Scenario: clear E(9,22) while player at (9,23) ---")
env3 = make_l8_env()
run_to_823(env3)

# Clear E(9,23) and walk R
click_act(env3, 9, 23)  # E(9,22) created
env3.step(RIGHT)         # walk to (9,23)
p, g = state(env3)
print(f"At ({p[0]},{p[1]}) alive={is_alive(env3)}")

# Now click E(9,22) — does it push or kill?
print(f"  (9,22)={sym(env3,9,22)} (9,21)={sym(env3,9,21)}")
print(f"  Clicking E(9,22) from (9,23)...")
click_act(env3, 9, 22)
p2, g2 = state(env3)
print(f"  After: ({p2[0]},{p2[1]}) grav={'UP' if g2 else 'DN'} alive={is_alive(env3)}")

# If alive, check position
if is_alive(env3):
    print(f"  ALIVE! Check column:")
    for y in range(15, 25):
        s = sym(env3, 9, y)
        if s != '.' or y == p2[1]:
            mark = " <-- PLAYER" if (9,y)==(p2[0],p2[1]) else ""
            print(f"    (9,{y})={s}{mark}")

    # Try clicking E(9,21) to push further
    if sym(env3, 9, 21) == 'E':
        print(f"\n  Push C(9,21):")
        click_act(env3, 9, 21)
        p3, g3 = state(env3)
        print(f"    ({p3[0]},{p3[1]}) alive={is_alive(env3)}")
