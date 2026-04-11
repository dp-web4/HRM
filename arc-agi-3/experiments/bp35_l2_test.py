#!/usr/bin/env python3
"""BP35 L2 solution test: 34 actions (baseline 36)."""
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


def click_act(env, gx, gy):
    engine = env._game.oztjzzyqoek
    cam_y = engine.camera.rczgvgfsfb[1]
    return env.step(CLICK, data={"x": gx*6, "y": gy*6 - cam_y})


def execute_and_advance(env, moves, level_name=""):
    old_level = env._game.level_index
    step = 0
    for m in moves:
        step += 1
        if m[0] == 'R': env.step(RIGHT)
        elif m[0] == 'L': env.step(LEFT)
        elif m[0] == 'C': click_act(env, m[1], m[2])
        if env._game.level_index > old_level:
            return True, step
    for i in range(20):
        step += 1
        env.step(LEFT)
        if env._game.level_index > old_level:
            return True, step
    return False, step


def get_pos(engine):
    return tuple(engine.twdpowducb.qumspquyus)


def survey_brief(engine):
    grid = engine.hdnrlfmyrj
    sym = {'xcjjwqfzjfe':'W', 'qclfkhjnaac':'D', 'fjlzdjxhant':'*',
           'ubhhgljbnpu':'^', 'hzusueifitk':'v', 'oonshderxef':'O',
           'yuuqpmlxorv':'B', 'lrpkmzabbfa':'G', 'etlsaqqtjvn':'E',
           'aknlbboysnc':'c', 'jcyhkseuorf':'F'}
    pp = get_pos(engine)
    for y in range(-2, 40):
        row = ""
        has = False
        for x in range(-2, 15):
            items = grid.jhzcxkveiw(x, y)
            if (x,y) == pp: row += "P"; has = True
            elif items: row += sym.get(items[0].name,'?'); has = True
            else: row += "."
        if has: print(f"  y={y:3d}: {row}")
    print(f"  Player: {pp}")


# L0 and L1 solutions
L0_SOL = [R,R,R,R, C(7,19), C(4,16), L,L,L, C(4,15), C(4,12), R, C(5,9), L,L]
L1_SOL = [
    R,R,R,R,R, C(8,36), C(8,35),
    L,L, C(5,29),L, C(4,29),L, C(3,29),L, C(2,29),L,
    C(2,28), R,R,R, C(5,24), C(5,23),
    L,L, C(3,20), C(3,17), C(3,16),
    C(4,16),R, C(5,16),R, C(6,16),R, C(7,16),R, C(8,16),R,
    C(8,15), C(8,14),
    L, L, L, C(5,9), L, L,
]

# L2 solution: 34 actions
# Player (3,28) → gem (7,7), gravity UP, rising floor at y=34
L2_SOL = [
    # Phase A: get to (6,24) via D destruction (5 actions)
    C(5,28),             # toggle B→O at (5,28)
    R, R, R,             # walk to (6,28) via (5,28)=O
    C(6,27),             # destroy D, fall up to (6,24)

    # Phase B: toggle y=23 O→B, walk left, fall to (2,19) (7 actions)
    C(5,23), C(4,23), C(3,23),  # make y=23 solid floor
    L, L, L,             # → (5,24) → (4,24) → (3,24)
    L,                   # → (2,24) → fall through empty y=23-19 to (2,19) (y=18 W)

    # Phase C: cross right to (3,18) (1 action)
    R,                   # → (3,19) → fall up to (3,18) (y=17 W)

    # Phase D: toggle y=17/18, cross to (7,13) (8 actions)
    C(5,17), C(6,17),   # toggle O→B (make ceiling)
    C(5,18), C(6,18),   # toggle B→O (make passable)
    R, R, R, R,          # (4,18) → (5,18) → (6,18) → (7,18)→fall to (7,13)

    # Phase E: toggle y=12, navigate to (3,7) (8 actions)
    C(6,12), C(5,12), C(4,12),  # toggle O→B (safe ceiling)
    C(3,12),             # toggle B→O (fall chute)
    L, L, L,             # → (6,13) → (5,13) → (4,13)
    L,                   # → (3,13) → fall through (3,12)=O all way to (3,7)

    # Phase F: reach gem (5 actions)
    C(5,7),              # toggle B→O
    R, R, R, R,          # → (4,7) → (5,7)=O → (6,7) → (7,7)=gem WIN!
]

print(f"L2 solution: {len(L2_SOL)} actions (baseline 36)")

# Run
arcade = Arcade()
env = arcade.make('bp35-0a0ad940')
env.reset()

# Solve L0
ok, _ = execute_and_advance(env, L0_SOL, "L0")
print(f"L0: {'OK' if ok else 'FAIL'} (level={env._game.level_index})")

# Solve L1
ok, _ = execute_and_advance(env, L1_SOL, "L1")
print(f"L1: {'OK' if ok else 'FAIL'} (level={env._game.level_index})")

# Now L2 — trace step by step
print(f"\n{'='*60}")
print(f"L2 (baseline=36, ours={len(L2_SOL)})")
print(f"{'='*60}")

engine = env._game.oztjzzyqoek
pp = get_pos(engine)
print(f"Start: player={pp}")

old_level = env._game.level_index
step = 0
labels = [
    "C(5,28) toggle", "R", "R", "R", "C(6,27) destroy→fall",
    "C(5,23) toggle", "C(4,23) toggle", "C(3,23) toggle", "L", "L", "L", "L→fall(2,19)",
    "R→fall(3,18)",
    "C(5,17) toggle", "C(6,17) toggle", "C(5,18) toggle", "C(6,18) toggle",
    "R(4,18)", "R(5,18)", "R(6,18)", "R→fall(7,13)",
    "C(6,12) O→B", "C(5,12) O→B", "C(4,12) O→B", "C(3,12) B→O",
    "L(6,13)", "L(5,13)", "L(4,13)", "L→fall(3,7)",
    "C(5,7) B→O", "R(4,7)", "R(5,7)", "R(6,7)", "R→gem(7,7)",
]

for i, m in enumerate(L2_SOL):
    step += 1
    if m[0] == 'R': env.step(RIGHT)
    elif m[0] == 'L': env.step(LEFT)
    elif m[0] == 'C': click_act(env, m[1], m[2])

    pp = get_pos(engine)
    label = labels[i] if i < len(labels) else ""
    lvl = env._game.level_index

    if lvl > old_level:
        print(f"  {step:2d}. {label:20s} → {pp} *** LEVEL ADVANCED! ***")
        break
    else:
        print(f"  {step:2d}. {label:20s} → {pp}")

if env._game.level_index == old_level:
    print("\n  Processing animation...")
    for i in range(20):
        step += 1
        env.step(LEFT)
        if env._game.level_index > old_level:
            print(f"  Animation step {i+1}: LEVEL ADVANCED! (total {step})")
            break

if env._game.level_index > old_level:
    print(f"\nL2 SOLVED in {len(L2_SOL)} actions! (baseline 36)")
else:
    print(f"\nL2 FAILED. Player at {get_pos(engine)}")
    survey_brief(engine)
