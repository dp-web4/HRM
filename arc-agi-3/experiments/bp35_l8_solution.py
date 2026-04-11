#!/usr/bin/env python3
"""L8: Complete solution attempt.
Path: Setup clicks → G-flip corridor → E walk-and-clear → x=9 corridor → gem.

Key mechanics used:
- E walk-and-clear: click E ahead to clear path, walk into cleared cell
- E side-effect ceiling: clicking E(x,y) creates E(x,y-1) which protects from ^-spikes
- G flip alternation: use multiple G blocks to navigate vertically
- D blocking E spread: leave D(1,8) intact to block E side-effect at x=0 entrance
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
print("=== L8 Solution ===\n")
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
    # DO NOT destroy D(1,8) yet! D blocks E side-effect.
    # DO NOT destroy D(4,13) yet! D provides ceiling.

    # ---- PHASE 1: G-flip to cross y=34 gap ----
    # G flip DOWN (grav UP→DN): player falls to y=37
    C(1,1),
    # Walk R to x=7 (floor at y=38 with grav DN)
    R, R, R, R,
    # G flip UP (grav DN→UP): fly through y=34 gap at x=7
    # E ceiling at y=31 catches player → lands at (7,32)
    C(2,1),

    # ---- PHASE 2: Walk-and-clear L along y=32 to x=2 ----
    # E at y=32 blocks: click to clear, walk into cleared cell
    # Each click creates E at y=33 (harmless) and removes blocking E
    C(6,32), L,  # (7,32)→(6,32) [ceil E(6,31)]
    C(5,32), L,  # (6,32)→(5,32) [ceil E(5,31)]
    C(4,32), L,  # (5,32)→(4,32) [ceil E(4,31)]
    C(3,32), L,  # (4,32)→(3,32) [ceil E(3,31)]
    # (3,32)→(2,32): E(2,32) may have been created by C(3,32) side effect
    C(2,32), L,  # → (2,32) [ceil E(2,31)]

    # ---- PHASE 3: Push-up at x=2 from y=32 to y=21 ----
    # E ceiling chain: push-up creates E above and below, allowing continued push
    # Each push-up also creates E(3,y) side-effect
    C(2,31),  # (2,32)→(2,31)
    C(2,30),  # (2,31)→(2,30) [E(2,30) from chain side-effects]
    C(2,29),  # (2,30)→(2,29)
    C(2,28),  # (2,29)→(2,28)
    C(2,27),  # (2,28)→(2,27) [y=27 open at x=2, no spike]
    C(2,26),  # (2,27)→(2,26) [D destroyed, now empty→E→push]
    C(2,25),  # (2,26)→(2,25)
    C(2,24),  # (2,25)→(2,24)
    C(2,23),  # (2,24)→(2,23)
    C(2,22),  # (2,23)→(2,22)
    C(2,21),  # (2,22)→(2,21) [STOP here, y=20→^(2,19) deadly]

    # ---- PHASE 4: Walk-and-clear R at y=21 to x=7 ----
    # Side-effect E at x=3 from push-up. Clear with walk-and-clear.
    # Each clear also creates E(x,20) which protects from ^(x,19)!
    C(3,21), R,  # (2,21)→(3,21) [ceil E(3,20)]
    C(4,21), R,  # (3,21)→(4,21) [ceil E(4,20)]
    C(5,21), R,  # (4,21)→(5,21) [ceil E(5,20) protects from ^(5,19)]
    C(6,21), R,  # (5,21)→(6,21) [ceil E(6,20) protects from ^(6,19)]
    C(7,21), R,  # (6,21)→(7,21) [ceil E(7,20) protects from ^(7,19)]

    # ---- PHASE 5: Navigate from (7,21) to (8,23) via G flips ----
    # x=8 wall at y=21-22 blocks direct R. Must go to y=23.
    # E(7,22) from walk-and-clear side-effect
    C(7,22),       # Clear E(7,22). Creates E(7,23),(6,22). (7,22) now empty.
    C(3,1),        # G flip DOWN. E(7,20) above (floor with grav DN). Player stays (7,21).
    # Hmm, with grav DN at (7,21): floor is E(7,22)→removed. Falls to... need floor.
    # Actually E(7,22) was just cleared. What's below? (7,23) E from click.
    # Falls from (7,21) through (7,22)(empty) to E(7,23) floor → (7,22)

    # Clear E(7,23) to fall further
    C(7,23),       # Clear E(7,23). Creates E(7,24),(8,23),(6,23). Falls to (7,23) [floor E(7,24)]

    # Now at (7,23) grav DN. Click E(8,23) to clear, then walk R.
    C(8,23),       # Clear E(8,23). Creates E(8,24),(9,23),(8,22)=W blocked.
    R,             # Walk R to (8,23). Floor E(8,24). Safe with grav DN.

    # G flip UP at (8,23)
    C(4,1),        # Grav DN→UP. (8,22)=W ceiling. Player stays (8,23).

    # Walk R to (9,23) → fly up to B(9,15) ceiling → (9,16)
    R,             # (8,23)→(9,16) [flies through open x=9 corridor]

    # ---- PHASE 6: Walk L along y=16 to x=4 ----
    L, L, L, L, L,  # (9,16)→(8,16)→...→(4,16) [B ceiling at y=15]

    # ---- PHASE 7: Toggle B(4,15), fly to D(4,13), then to E ceiling ----
    C(4,15),       # B→O. Falls UP through y=15 to D(4,13) → (4,14)

    # Destroy D(4,13) to continue UP
    C(4,13),       # D destroyed. Falls UP through y=13 to E(4,9) ceiling → (4,10)

    # Walk L to x=2 → fall to E(2,8) ceiling → (2,9)
    L,             # (4,10)→(3,10) [E(3,9) ceiling]
    L,             # (3,10)→(2,9) [falls through (2,9)→E(2,8) ceiling] wait...
    # Actually (2,10)→what? Ceiling at x=2: (2,9) open, (2,8) E. Falls to (2,9).

    # ---- PHASE 8: Push-up to (2,8), enter x=0 ----
    C(2,8),        # Push-up: (2,9)→(2,8). D(1,8) blocks E side-effect!
    C(1,8),        # Destroy D(1,8). (1,8) now empty.
    L,             # Walk L to (1,8). Ceiling W(1,7).
    L,             # Walk L to (0,8)→falls to (0,7) [W(0,6) ceiling]

    # ---- PHASE 9: G flip DOWN, fall through x=0 shaft to gem ----
    C(5,1),        # Grav UP→DN. Falls from (0,7) to (0,40) [W(0,41) floor]
    R,             # (0,40)→(1,40)
    R,             # (1,40)→(2,40) = GEM!
]

step = 0
for move in L8_SOL:
    step += 1
    p1, g1 = state(env)
    if move[0] == 'R': env.step(RIGHT)
    elif move[0] == 'L': env.step(LEFT)
    elif move[0] == 'C': click_act(env, move[1], move[2])
    p2, g2 = state(env)

    dy = f" y:{p1[1]}→{p2[1]}" if p1[1] != p2[1] else ""
    gc = f" grav:{'UP' if g1 else 'DN'}→{'UP' if g2 else 'DN'}" if g1 != g2 else ""
    stuck = " STUCK!" if p1 == p2 and move[0] in ('R', 'L') else ""
    lvl = env._game.level_index
    won = " ** LEVEL WON! **" if lvl > 8 else ""

    print(f"  {step:3d}. {str(move):14s} ({p1[0]:2d},{p1[1]:2d})→({p2[0]:2d},{p2[1]:2d}){dy}{gc}{stuck}{won}")

    if stuck:
        print(f"\n  FAILED at step {step}!")
        print(f"  Grid around ({p2[0]},{p2[1]}):")
        print_grid(env, (max(0, p2[1]-5), min(45, p2[1]+5)))
        # Check what's blocking
        lx, ly = p2[0]-1 if move[0]=='L' else p2[0]+1, p2[1]
        print(f"  Target ({lx},{ly}) = {sym(env, lx, ly)}")
        break

    if won:
        print(f"\n  L8 SOLVED in {step} steps!")
        break

if env._game.level_index <= 8:
    p, g = state(env)
    print(f"\n  Current: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} level={env._game.level_index}")
    print_grid(env, (max(0, p[1]-5), min(45, p[1]+5)))
