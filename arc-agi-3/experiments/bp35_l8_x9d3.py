#!/usr/bin/env python3
"""L8: Single test — clear E(9,22) while player at (9,23). Does it push or kill?"""
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

# ============================================================
print("=== L8: Push-up vs flight at x=9 ===\n")
env = make_l8_env()
print(f"L8 env ready")

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

for move in SETUP_NAV:
    if move[0] == 'R': env.step(RIGHT)
    elif move[0] == 'L': env.step(LEFT)
    elif move[0] == 'C': click_act(env, move[1], move[2])

p, g = state(env)
print(f"At ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} alive={is_alive(env)}")

# Step 1: Clear E(9,23) and walk R to (9,23)
click_act(env, 9, 23)  # creates E(9,22), E(9,24)
env.step(RIGHT)
p, g = state(env)
print(f"Walk R: ({p[0]},{p[1]}) alive={is_alive(env)}")

# Step 2: Now at (9,23) with E(9,22) ceiling.
# KEY TEST: click E(9,22) — push-up or just E-click?
# Push-up should work (player adjacent to E from below with grav UP)
# But earlier test showed death.

# First: detailed state before click
print(f"\nState before C(9,22):")
for y in range(20, 26):
    s = sym(env, 9, y)
    mark = " <PLAYER>" if (9,y) == p else ""
    print(f"  (9,{y})={s}{mark}")
print(f"  (8,{p[1]})={sym(env,8,p[1])}")
print(f"  (10,{p[1]})={sym(env,10,p[1])}")

# Try a different approach: DON'T push up.
# Instead, just clear E(9,22) to remove the ceiling so gravity can pull player up.
# With grav UP at (9,23), if we remove E(9,22), does the player automatically fly up?
# In push-up: player rises one cell. In gravity: player should fly until hitting next solid.

# Actually, the E click creates side-effects that might block.
# Let's trace: C(9,22) removes E(9,22). Side-effects:
# E(9,21), E(9,23)=player→no?, E(8,22)=W, E(10,22)=W.
# Then gravity: player at (9,23) with grav UP, ceiling gone at (9,22).
# Does player fly up? Or does E(9,21) block at (9,22)?
# Player rises from (9,23) to (9,22). Then (9,21)=E blocks. Player at (9,22).
# But E(9,23) gets created? Player was at (9,23)...
# IF player moves to (9,22) FIRST, then E(9,23) gets created in now-empty cell.
# IF E(9,23) gets created FIRST, player is trapped.

# The order matters. Let me test empirically:
print(f"\nClicking E(9,22)...")
click_act(env, 9, 22)
p2, g2 = state(env)
print(f"  After: ({p2[0]},{p2[1]}) grav={'UP' if g2 else 'DN'} alive={is_alive(env)}")

print(f"\nEntities after:")
for y in range(19, 25):
    items = env._game.oztjzzyqoek.hdnrlfmyrj.jhzcxkveiw(9, y)
    names = [(SYM.get(i.name, i.name)) for i in items]
    mark = " <PLAYER>" if (9,y) == (p2[0],p2[1]) else ""
    print(f"  (9,{y}): {names}{mark}")

# Check W walls
for y in [p2[1]-1, p2[1], p2[1]+1]:
    for x in [8, 10]:
        items = env._game.oztjzzyqoek.hdnrlfmyrj.jhzcxkveiw(x, y)
        names = [(SYM.get(i.name, i.name)) for i in items]
        if names:
            print(f"  ({x},{y}): {names}")

if not is_alive(env):
    print(f"\n  DEAD! Trying UNDO...")
    env.step(UNDO)
    p3, g3 = state(env)
    print(f"  After UNDO: ({p3[0]},{p3[1]}) alive={is_alive(env)}")

    if is_alive(env):
        print(f"  UNDO restored life!")
        # Now try a different approach: G flip instead of push-up
        print(f"\n  Alternative: G flip DN at (9,23) to go past E(9,22)")
        print(f"  Current: ({p3[0]},{p3[1]}) grav={'UP' if g3 else 'DN'}")
        print(f"  E(9,22)={sym(env,9,22)} E(9,24)={sym(env,9,24)}")

        # G flip DN: player at (9,23) falls down
        # Below: E(9,24) → floor. Stays at (9,23).
        # Clear E(9,24): fall to (9,24). Below: (9,25)=v → lethal with DN!

        # Alternative: click E(9,22) to clear but DON'T push up
        # Wait, the test just showed that clicking E(9,22) causes push-up death.
        # What if we use UNDO to go back to (8,23)?
        env.step(UNDO)  # undo walk R
        p4, g4 = state(env)
        print(f"  After 2nd UNDO: ({p4[0]},{p4[1]}) alive={is_alive(env)}")

        if p4[0] == 8:
            # Now at (8,23) again, E(9,23) should be back
            print(f"  E(9,23)={sym(env,9,23)}")

            # Try: clear E(9,23), clear E(9,22), THEN walk R
            # After clearing both: (9,23)=E (from clearing 9,22), (9,22)=empty
            # Nope, oscillation.

            # Try: clear E(9,23) only, then G flip DN, walk R to (9,23)
            # With grav DN at (9,23): floor = E(9,24) from clearing.
            # E(9,22) still there above. Not a problem with grav DN.
            # Then G flip UP: player at (9,23) flies through (9,22)=E → stops at (9,23).
            # Still stuck.

            # Try: what if the column above (9,22) is clean?
            # I.e., if we could remove E(9,22) WITHOUT creating E(9,21),
            # then the corridor from (9,22) to (9,16) would be open.
            # But E spread ALWAYS creates in 4 directions.

            # WHAT IF we put something at (9,21) first?
            # E can't be created at occupied cells.
            # If (9,21) has an entity, E won't form there.
            print(f"\n  (9,21) currently: {sym(env,9,21)}")
            print(f"  (9,20) currently: {sym(env,9,20)}")

            # Check: does W or other entity exist anywhere in x=9 y=16-22?
            print(f"\n  x=9 y=16-22:")
            for y in range(16, 23):
                s = sym(env, 9, y)
                print(f"    (9,{y})={s}")
