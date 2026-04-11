#!/usr/bin/env python3
"""L8: Fixed x=9 push-up test — use proper is_alive check.
Player at (9,22) between W walls can't move L/R but may still be alive.
Push-up chain through x=9 from y=23 to y=16."""
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
    """Check actual alive flag, not movement-based."""
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
print("=== L8: x=9 Push-up with proper alive check ===\n")
env = make_l8_env()
print("L8 ready")

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

# Clear E(9,23) and walk R
click_act(env, 9, 23)
env.step(RIGHT)
p, g = state(env)
print(f"Walk R: ({p[0]},{p[1]}) alive={is_alive(env)}")

# Push-up through x=9
print(f"\nPush-up chain:")
for i in range(10):
    p, g = state(env)
    alive = is_alive(env)

    # Find ceiling
    ceil_y = None
    for y in range(p[1]-1, p[1]-10, -1):
        s = sym(env, 9, y)
        if s != '.':
            ceil_y = y
            break

    if ceil_y is None:
        print(f"  {i+1}: at ({p[0]},{p[1]}) alive={alive} — no ceiling!")
        break

    ceil_s = sym(env, 9, ceil_y)
    print(f"  {i+1}: ({p[0]},{p[1]}) alive={alive} ceil=({9},{ceil_y})={ceil_s}", end="")

    if not alive:
        print(f" DEAD!")
        break

    if ceil_s == 'B':
        print(f" — B ceiling, done!")
        break

    if ceil_s != 'E':
        print(f" — not E!")
        break

    click_act(env, 9, ceil_y)
    p2, g2 = state(env)
    print(f" → ({p2[0]},{p2[1]}) alive={is_alive(env)}")

    if p2[1] <= 16:
        break

p, g = state(env)
alive = is_alive(env)
print(f"\nAfter push-up: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} alive={alive}")

if alive and p[1] <= 17:
    # Walk-and-clear L along y=16
    print(f"\nWalk-and-clear L:")
    for i in range(8):
        p, g = state(env)
        lx = p[0] - 1
        ly = p[1]
        ls = sym(env, lx, ly)

        if ls == 'E':
            click_act(env, lx, ly)
            print(f"  Clear E({lx},{ly})", end="")
        elif ls == 'W':
            print(f"  W({lx},{ly}) — blocked!")
            break
        else:
            print(f"  ({lx},{ly})={ls}", end="")

        env.step(LEFT)
        p2, g2 = state(env)
        moved = p2[0] < p[0]
        print(f" → L: ({p[0]},{p[1]})→({p2[0]},{p2[1]}) alive={is_alive(env)}")

        if not moved or not is_alive(env):
            break
        if p2[0] <= 4:
            break

    p, g = state(env)
    print(f"\nAt ({p[0]},{p[1]})")

    # Continue: B toggle, D destroy, etc.
    if p[0] <= 5 and p[1] == 16:
        print(f"\n--- Phase 7: B→O, D destroy, navigate to x=0 ---")

        # Walk L to (4,16) if not there
        while p[0] > 4:
            ls = sym(env, p[0]-1, p[1])
            if ls == 'E':
                click_act(env, p[0]-1, p[1])
            env.step(LEFT)
            p, g = state(env)
        print(f"At ({p[0]},{p[1]})")

        # Toggle B(4,15)
        click_act(env, 4, 15)
        p, g = state(env)
        print(f"B toggle: ({p[0]},{p[1]}) alive={is_alive(env)}")

        # Destroy D(4,13)
        if sym(env, 4, 13) == 'D':
            click_act(env, 4, 13)
            p, g = state(env)
            print(f"D destroy: ({p[0]},{p[1]}) alive={is_alive(env)}")

        # Walk L toward x=2
        print(f"\nWalk L to x=2:")
        for i in range(5):
            p, g = state(env)
            lx, ly = p[0]-1, p[1]
            ls = sym(env, lx, ly)
            if ls == 'E':
                click_act(env, lx, ly)
            elif ls == 'W':
                print(f"  W({lx},{ly}) blocked")
                break
            env.step(LEFT)
            p2, g2 = state(env)
            print(f"  ({p[0]},{p[1]})→({p2[0]},{p2[1]}) alive={is_alive(env)}")
            if not is_alive(env) or p2 == p:
                break
            if p2[0] <= 2:
                break

        p, g = state(env)
        print(f"\nAt ({p[0]},{p[1]})")

        # Push-up to E(2,8) ceiling
        if p[0] == 2:
            print(f"\nPush-up at x=2:")
            for i in range(5):
                p, g = state(env)
                ceil_y = None
                for y in range(p[1]-1, p[1]-10, -1):
                    s = sym(env, 2, y)
                    if s != '.':
                        ceil_y = y
                        break
                if ceil_y is None:
                    break
                cs = sym(env, 2, ceil_y)
                if cs != 'E':
                    print(f"  Ceiling ({2},{ceil_y})={cs} — not E")
                    break
                click_act(env, 2, ceil_y)
                p2, g2 = state(env)
                print(f"  C(2,{ceil_y}): ({p[0]},{p[1]})→({p2[0]},{p2[1]}) alive={is_alive(env)}")
                if not is_alive(env):
                    break

            p, g = state(env)
            # Destroy D(1,8) and walk to x=0
            if p == (2,8) or p[1] <= 9:
                print(f"\nNavigate to x=0:")
                if sym(env, 1, 8) == 'D':
                    click_act(env, 1, 8)
                env.step(LEFT)
                p, g = state(env)
                print(f"  L: ({p[0]},{p[1]}) alive={is_alive(env)}")
                env.step(LEFT)
                p, g = state(env)
                print(f"  L: ({p[0]},{p[1]}) alive={is_alive(env)}")

                # G flip DN → fall to (0,40)
                print(f"\nG flip DN + fall:")
                click_act(env, 5, 1)  # 5th G flip
                p, g = state(env)
                print(f"  Flip: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} alive={is_alive(env)}")

                # Walk R to gem
                env.step(RIGHT)
                p, g = state(env)
                print(f"  R: ({p[0]},{p[1]})")
                env.step(RIGHT)
                p, g = state(env)
                print(f"  R: ({p[0]},{p[1]})")

                lvl = env._game.level_index
                if lvl > 8:
                    print(f"\n*** L8 SOLVED! Level={lvl} ***")
                else:
                    print(f"\n  Level={lvl}. Grid:")
                    print_grid(env, (max(0,p[1]-3), min(45,p[1]+3)))

p, g = state(env)
print(f"\n=== END: ({p[0]},{p[1]}) grav={'UP' if g else 'DN'} alive={is_alive(env)} level={env._game.level_index} ===")
