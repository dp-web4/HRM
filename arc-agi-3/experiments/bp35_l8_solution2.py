#!/usr/bin/env python3
"""L8: Complete solution v2 — x=9 push-up chain.
Fix from v1: instead of walking R at (8,23) into E(9,23),
clear E(9,23), walk R to (9,23), then push-up through x=9 shaft
from y=23 to y=16 using E ceiling chain.

Key insight: the x=9 shaft (y=18-22) is 1-cell wide (W at x=8 and x=10).
Push-up in a 1-wide shaft creates E above and below, but the player is
between them — NOT dead, just unable to move L/R. The earlier "death"
was a false positive from a movement-based alive check.

At (9,16): (8,16) is open, so walk-and-clear L to reach (4,16).
"""
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
    return not env._game.oztjzzyqoek.jrhqdvdwpsb

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
print("=== L8 Solution v2 ===\n")
env = make_l8_env()

L8_SOL = [
    # ---- SETUP PHASE (clicks only, no movement) ----
    # E spread from (7,8) west to (2,8): 5 clicks
    C(7,8), C(6,8), C(5,8), C(4,8), C(3,8),
    # E spread from (6,31) + chain west to create y=31 ceiling: 4 clicks
    C(6,31), C(5,31), C(4,31), C(3,31),
    # Consume x=0 G blocks (6 clicks, even = same grav): 6 clicks
    C(0,15), C(0,19), C(0,21), C(0,25), C(0,33), C(0,35),
    # Destroy D at y=26 (open vertical shaft): 3 clicks
    C(2,26), C(3,26), C(4,26),

    # ---- PHASE 1: G-flip to cross y=34 gap ----
    C(1,1),            # G flip DOWN (grav UP→DN)
    R, R, R, R,        # Walk R to x=7 (floor at y=38)
    C(2,1),            # G flip UP → fly through y=34 gap → (7,32)

    # ---- PHASE 2: Walk-and-clear L along y=32 to x=2 ----
    C(6,32), L,        # (7,32)→(6,32)
    C(5,32), L,        # (6,32)→(5,32)
    C(4,32), L,        # (5,32)→(4,32)
    C(3,32), L,        # (4,32)→(3,32)
    C(2,32), L,        # (3,32)→(2,32)

    # ---- PHASE 3: Push-up at x=2 from y=32 to y=21 ----
    C(2,31), C(2,30), C(2,29), C(2,28), C(2,27),
    C(2,26), C(2,25), C(2,24), C(2,23), C(2,22), C(2,21),

    # ---- PHASE 4: Walk-and-clear R at y=21 to x=7 ----
    C(3,21), R,        # → (3,21)
    C(4,21), R,        # → (4,21)
    C(5,21), R,        # → (5,21)
    C(6,21), R,        # → (6,21)
    C(7,21), R,        # → (7,21)

    # ---- PHASE 5: Navigate from (7,21) to (8,23) ----
    C(7,22),           # Clear E(7,22) → creates E(7,23),(6,22)
    C(3,1),            # G flip DOWN → falls to (7,22)
    C(7,23),           # Clear E(7,23) → creates E(7,24),(8,23),(6,23)
    C(8,23),           # Clear E(8,23) → creates E(9,23),(8,24)
    R,                 # Walk R to (8,23)
    C(4,1),            # G flip UP → (8,23) grav UP

    # ---- PHASE 6: Enter x=9 and push-up to y=16 ----
    # Clear E(9,23) then walk R → (9,23) with E(9,22) ceiling
    C(9,23),           # Clear E(9,23) → creates E(9,22),(9,24)
    R,                 # Walk R to (9,23) [E(9,22) ceiling stops flight]

    # Push-up through x=9 shaft: 7 pushes from y=23 to y=16
    # Each push: click E above → player rises 1, E created above+below
    # In 1-wide shaft (W at x=8,x=10 for y=18-22): player between E's
    # At (9,22): E(9,21) above, (9,23) below (player left, E fills)
    C(9,22),           # (9,23)→(9,22) [E(9,21) ceil, E(9,23) floor]
    C(9,21),           # (9,22)→(9,21) [E(9,20) ceil, E(9,22) floor]
    C(9,20),           # (9,21)→(9,20) [E(9,19) ceil, E(9,21) floor]
    C(9,19),           # (9,20)→(9,19) [E(9,18) ceil, E(9,20) floor]
    C(9,18),           # (9,19)→(9,18) [E(9,17) ceil, E(9,19) floor]
    C(9,17),           # (9,18)→(9,17) [E(9,16) ceil, E(9,18) floor]
    C(9,16),           # (9,17)→(9,16) [B(9,15) blocks E→no new ceil]

    # ---- PHASE 7: Walk-and-clear L along y=16 to x=4 ----
    # E(8,16) created as side-effect of C(9,16) push-up
    C(8,16), L,        # Clear E(8,16), walk to (8,16)
    C(7,16), L,        # Clear E(7,16) if exists, walk to (7,16)
    C(6,16), L,        # Clear E(6,16) if exists, walk to (6,16)
    C(5,16), L,        # → (5,16)
    C(4,16), L,        # → (4,16)

    # ---- PHASE 8: Toggle B(4,15), fly up, destroy D(4,13) ----
    C(4,15),           # B→O. Grav UP → fly through O(4,15) to D(4,13)→(4,14)
    C(4,13),           # Destroy D → fly through (4,13) up to E(4,9) ceiling → (4,10)
                       # (4,9) has E from setup E spread chain at y=9

    # ---- PHASE 9: Walk L to x=2 ----
    L,                 # (4,10)→(3,10) [E(3,9) ceiling from spread]
    L,                 # (3,10)→(2,10)→fly up to E(2,8) ceiling → (2,9)

    # ---- PHASE 10: Push-up to (2,8), enter x=0 ----
    C(2,8),            # Push-up: (2,9)→(2,8)
    C(1,8),            # Destroy D(1,8)
    L,                 # Walk to (1,8)
    L,                 # Walk to (0,8) → fly to (0,7) [W(0,6) ceiling]

    # ---- PHASE 11: G flip DOWN, fall to gem ----
    C(5,1),            # G flip UP→DN
    R,                 # (0,40)→(1,40) [W(0,41) floor in DN]
    R,                 # (1,40)→(2,40) = GEM!
]

step = 0
for move in L8_SOL:
    step += 1
    p1, g1 = state(env)
    alive = is_alive(env)
    if move[0] == 'R': env.step(RIGHT)
    elif move[0] == 'L': env.step(LEFT)
    elif move[0] == 'C': click_act(env, move[1], move[2])
    p2, g2 = state(env)
    alive2 = is_alive(env)

    dy = f" y:{p1[1]}→{p2[1]}" if p1[1] != p2[1] else ""
    gc = f" grav:{'UP' if g1 else 'DN'}→{'UP' if g2 else 'DN'}" if g1 != g2 else ""
    stuck = " STUCK!" if p1 == p2 and move[0] in ('R', 'L') else ""
    dead = " DEAD!" if alive and not alive2 else ""
    lvl = env._game.level_index
    won = " ** LEVEL WON! **" if lvl > 8 else ""

    print(f"  {step:3d}. {str(move):14s} ({p1[0]:2d},{p1[1]:2d})→({p2[0]:2d},{p2[1]:2d}){dy}{gc}{stuck}{dead}{won}")

    if dead:
        print(f"\n  DIED at step {step}!")
        print(f"  Grid around ({p2[0]},{p2[1]}):")
        print_grid(env, (max(0, p2[1]-5), min(45, p2[1]+5)))
        break

    if stuck:
        print(f"\n  STUCK at step {step}!")
        print(f"  Grid around ({p2[0]},{p2[1]}):")
        print_grid(env, (max(0, p2[1]-5), min(45, p2[1]+5)))
        lx, ly = p2[0]-1 if move[0]=='L' else p2[0]+1, p2[1]
        print(f"  Target ({lx},{ly}) = {sym(env, lx, ly)}")
        break

    if won:
        print(f"\n  L8 SOLVED in {step} steps!")
        break

if env._game.level_index <= 8:
    p, g = state(env)
    print(f"\n  Current: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} alive={is_alive(env)} level={env._game.level_index}")
    print_grid(env, (max(0, p[1]-5), min(45, p[1]+5)))
