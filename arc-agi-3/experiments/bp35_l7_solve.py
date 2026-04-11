#!/usr/bin/env python3
"""L7 solver: E spreading + navigation to gem."""
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

def print_grid_region(env, y_range, x_range=(0,11)):
    engine = env._game.oztjzzyqoek
    grid = engine.hdnrlfmyrj
    sym = {'xcjjwqfzjfe':'W', 'yuuqpmlxorv':'B', 'oonshderxef':'O',
           'ubhhgljbnpu':'^', 'fjlzdjxhant':'*', 'etlsaqqtjvn':'E',
           'lrpkmzabbfa':'G'}
    pp = tuple(engine.twdpowducb.qumspquyus)
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

# ============================================================
# SOLVE L7
# ============================================================
env = make_l7_env()
print("L7 solving...\n")

steps = []
step_count = [0]

def do(move, desc=""):
    step_count[0] += 1
    p1, g1 = state(env)
    if move[0] == 'R':
        env.step(RIGHT)
    elif move[0] == 'L':
        env.step(LEFT)
    elif move[0] == 'C':
        click_act(env, move[1], move[2])
    p2, g2 = state(env)
    dy = f" y:{p1[1]}→{p2[1]}" if p1[1] != p2[1] else ""
    gc = f" grav:{'UP' if g1 else 'DN'}→{'UP' if g2 else 'DN'}" if g1 != g2 else ""
    stuck = " STUCK!" if p1 == p2 and move[0] in ('R', 'L') else ""
    gfail = ""
    if move[0] == 'C' and move[1:] == (5,2) and g1 == g2:
        gfail = " G-FAIL!"
    lvl = env._game.level_index
    won = " ** LEVEL ADVANCED! **" if lvl > 7 else ""
    print(f"  {step_count[0]:3d}. {str(move):12s} ({p1[0]:2d},{p1[1]:2d})→({p2[0]:2d},{p2[1]:2d}){dy}{gc}{stuck}{gfail}{won} {desc}")
    if stuck or gfail:
        return False
    if lvl > 7:
        return "WIN"
    return True

# Phase 1: Spread E from (2,29) rightward (5 clicks)
# Creates bridge at y=30 (x=2-6) and y=29 at various x
print("Phase 1: E bridge at y=29-30")
do(C(2,29), "E spread")
do(C(3,29), "E spread")
do(C(4,29), "E spread")
do(C(5,29), "E spread")
do(C(6,29), "E spread → bridge complete")

print("\n  Grid y=27-32:")
print_grid_region(env, (27, 33))

# Phase 2: Spread E from (3,18) for upper ceiling (4 clicks)
print("\nPhase 2: E ceiling at y=18-19")
do(C(3,18), "E spread")
do(C(4,18), "E spread")
do(C(5,18), "E spread")
do(C(6,18), "E spread → ceiling complete")

print("\n  Grid y=16-21:")
print_grid_region(env, (16, 22))

# Phase 3: Toggle B blocks
print("\nPhase 3: B block toggles")
do(C(7,25), "B→O (open passage)")
do(C(8,22), "B→O (open y=22)")

# Phase 4: Walk right across E bridge to x=8 shaft
print("\nPhase 4: Walk to x=8 shaft")
do(R, "walk")
do(R, "through y=31 gap → y=31")
do(R, "across E bridge")
do(R, "across E bridge")
do(R, "fall up x=8 shaft → y=25")

p, g = state(env)
print(f"\n  Current: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'}")
print("\n  Grid y=22-31:")
print_grid_region(env, (22, 32))

# Phase 5: Navigate from (8,25) to upper area
print("\nPhase 5: Navigate to upper area")
do(L, "to x=7 via B(7,25)=O")
do(L, "fall up → y=23 (ceiling B(6,22))")

p, g = state(env)
print(f"  At: ({p[0]},{p[1]})")

# Walk right along y=23 to x=8
do(R, "back to x=7")
do(R, "to x=8, fall up through B(8,22)=O → y=21")

p, g = state(env)
print(f"  At: ({p[0]},{p[1]})")
print("\n  Grid y=18-25:")
print_grid_region(env, (18, 26))

# Phase 6: Navigate from (8,21) westward using E ceiling at y=19
print("\nPhase 6: Navigate to gem approach position")
do(L, "to x=7")  # ceiling (7,20) wall
do(L, "to x=6 → fall up to y=20 (ceiling E(6,19))")

p, g = state(env)
print(f"  At: ({p[0]},{p[1]})")

# If at y=20, continue west along y=20 using E ceiling at y=19
do(L, "to x=5")
do(L, "to x=4")
do(L, "to x=3")

p, g = state(env)
print(f"  At: ({p[0]},{p[1]})")
print("\n  Grid y=16-22:")
print_grid_region(env, (16, 22))

# Now we need to get from here to (8,17) for the G flip approach
# We need E at (8,16) or (8,17) area for safe ceiling
# Spread E from y=17 rightward
print("\nPhase 7: Spread E at y=17 for x=8 ceiling")
# E at y=17: after Phase 2 we have E at (3,17), (4,17), (5,17), (6,17)
# Need to spread to x=7 and x=8
do(C(6,17), "E spread")  # should create E at (7,17)
do(C(7,17), "E spread")  # should create E at (8,17)

print("\n  Grid y=15-19:")
print_grid_region(env, (15, 20))

# Now we need to clear (8,17) so player can stand there
# and have E ceiling at (8,16)
do(C(8,17), "E spread → creates E at (8,16), removes (8,17)")

print("\n  Grid y=15-19:")
print_grid_region(env, (15, 20))

# Phase 8: Navigate to (8,17)
print("\nPhase 8: Navigate to (8,17)")
# From current position, walk right along y=20 (E ceiling at y=19)
p, g = state(env)
print(f"  Current: ({p[0]},{p[1]})")

# Need to reach (8,17). From y=20 area, walk right.
# (7,20): ceiling (7,19) → wall! So (7,20) ceiling is wall. Safe.
# Wait, y=19 at x=7 is wall. So (7,20) with grav UP has ceiling (7,19) wall. Can walk to x=7.
do(R, "right")
do(R, "right")
do(R, "right")
do(R, "right")  # to x=7

p, g = state(env)
print(f"  At: ({p[0]},{p[1]})")

# From x=7, we need to get to x=8 and up to y=17
# At x=7, y=20: ceiling (7,19) wall. Walk right to (8,20): (8,20) wall! Blocked.
# Hmm. Need another path.

# What about going from y=21? From (7,21) ceiling (7,20) wall.
# Walk right to (8,21) ceiling (8,20) wall.
# Then need to get from (8,21) to (8,17).
# (8,20) is wall. (8,19) is below that... Can't reach (8,18) or (8,17) from y=21.

# What if we remove E above at x=8?
# E at (8,16) exists (from Phase 7). Player at (8,21).
# Between y=21 and y=16: (8,20) wall, (8,19) open, (8,18) B, (8,17) open, (8,16) E.
# Player at (8,21) with ceiling (8,20) wall. Can't go up past wall.

# The wall at (8,20) blocks upward movement at x=8 in the y=19-21 zone.
# Need to approach (8,17) from a DIFFERENT direction.

# From x=7 at y=17-18: (7,18) wall, (7,17) open.
# Player at (7,17) with ceiling... what's above? (7,16) open, ..., (7,8) spike. Dead!
# Unless E ceiling at (7,16) exists.

# After Phase 7: E at (7,17) was clicked (step C(7,17)).
# Click C(7,17): removes (7,17), creates E at:
#   (6,17) E→skip, (8,17) empty→add, (7,16) empty→add, (7,18) wall→skip
# So E at (7,16) and (8,17).
# Then C(8,17): removes (8,17), creates E at:
#   (7,17) empty→add, (9,17) empty→add, (8,16) empty→add, (8,18) B→skip
# So after all Phase 7: E at (7,16) from C(7,17), E at (7,17)from C(8,17), E at (9,17), E at (8,16).

# So (7,17) has E (re-created by C(8,17)). Walking TO (7,17) means walking INTO a solid E block.
# (7,16) has E (ceiling for (7,17)).
# (8,17) is EMPTY (removed by C(8,17)).
# (8,16) has E (ceiling for (8,17)).

# We need to reach (8,17) which is empty. But (8,18) is B block.
# Toggle B(8,18) to O first? Or keep it?
# For the final approach: player at (8,17), grav UP, ceiling E(8,16), floor... hmm.
# With grav UP, player at (8,17) is held by ceiling. Below (8,18) doesn't matter for staying.

# Path to (8,17): from y=19 or y=18 area. But walls block at x=7-8.
# What about from x=9? E(9,17) exists. Can't walk onto it.

# What about from x=8, y=16? E(8,16) exists. Can't stand there (it's the ceiling).
# From x=8, approaching from above requires grav DOWN... which we haven't flipped yet.

# I think the path needs to go:
# 1. From (8,21) or similar, toggle B(8,18) to O
# 2. G flip → grav DOWN
# 3. Navigate with grav DOWN from (8,21) down to...
#    (8,22) B=O → fall through to (8,23), (8,24) wall. Land at (8,23).
#    No, that's going further from gem.

# Alternative: from the upper area (y=17-18), approach (8,17) with grav UP.
# Player needs to be at some position where walking right or left leads to (8,17).
# (7,17) has E → can't walk there.
# (9,17) has E → can't walk there.
# Can only reach (8,17) from (8,18) or (8,16). Both are solid (B or E).

# Hmm, (8,18) is B. If we toggle B(8,18) to O: (8,18) becomes O (passable).
# Player at (8,19) with grav UP: ceiling at (8,18) O → falls through!
# (8,17) empty → (8,16) E. Lands at (8,17). Ceiling E(8,16). ALIVE!

# So: player needs to reach (8,19) first with grav UP.
# But (8,19) is behind walls at x=7 and x=8 y=20.

# WAIT: We were navigating toward gem, but maybe the approach is different.
# What if B(8,18) is toggled to O, and the player approaches from (8,19)?
# How to reach (8,19)? Same problem as before — walled off.

# Let me reconsider. Maybe I should use grav DOWN from a different position.

# What if:
# 1. Navigate with grav UP to (8,21)
# 2. Toggle B(8,18) to O
# 3. Toggle B(8,22) to O (already done)
# 4. G flip → grav DOWN
# 5. Player at (8,21): floor below is B(8,22)=O → falls through to (8,23), (8,24) wall. Land (8,23).
# 6. Walk... not helpful.

# What if G flip happens at a position where falling DOWN leads to (8,19)?
# Need: player at (8,17) or (8,18) grav UP → flip → grav DOWN → fall to (8,19)
# At (8,17): flip → fall to (8,18) B/O. If O: (8,19), (8,20) wall. Land (8,19). → R → gem!
# At (8,18): if B, player is standing on it. Toggle to O first... but player would fall with grav UP.

# The cleanest approach: get player to (8,17) with ceiling E(8,16), toggle B(8,18) to O,
# G flip → fall to (8,19) → R → gem.

# To reach (8,17): need a column with no obstructions between player and (8,17).
# The only way seems to be from (8,19) going UP through (8,18) O.
# But (8,19) is where we want to end up!

# CIRCULAR. Let me try completely different path.

# What if the final approach uses E spreading at y=19?
# If we have E at (8,19), we can click it to create E at (8,18) B→skip, (8,20) wall→skip,
# (7,19) wall→skip, (9,19) gem→skip. Hmm, nothing useful.

# What about approaching from x=9 at y=21?
# (9,21): ceiling (9,20) wall. Walk right → (10,21) wall.
# From (9,21), toggle B(9,22) to O → player at (9,21), floor... wait, grav UP.
# With grav UP, toggling B below doesn't cause fall.

# I think I need to approach the gem from a COMPLETELY different angle.
# What if the path goes: E bridge → x=8 shaft → y=25 → left across y=23 →
# toggle B → up to y=21 → left → E ceiling at y=19 →
# eventually reach (3,19) → then what?

# (3,19) with E ceiling at (3,18) (re-created). Walk right to (4,19) E?
# After Phase 2: y=19 E at {3,4,5,6}. So (4,19) is E → can't walk there.

# Unless we click E(4,19) to spread it, clearing the path while creating ceiling above.
# Hmm this is very complex.

# Let me just print the current state and stop to reassess.
print(f"\n  Current state: pos={state(env)[0]} grav={'UP' if state(env)[1] else 'DN'}")
print("\n  Full grid:")
print_grid_region(env, (0, 36))
print(f"\n  Level: {env._game.level_index}")
