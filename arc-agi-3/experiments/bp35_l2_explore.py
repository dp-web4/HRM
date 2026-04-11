#!/usr/bin/env python3
"""BP35 L2 exploration: verify grid state after Phase 1-3 and test Phase 4 routes."""
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


def execute(env, moves):
    """Execute moves, return step count."""
    steps = 0
    for m in moves:
        steps += 1
        if m[0] == 'R':
            env.step(RIGHT)
        elif m[0] == 'L':
            env.step(LEFT)
        elif m[0] == 'C':
            click_act(env, m[1], m[2])
    return steps


def survey_detailed(engine, y_range=(-2, 45), x_range=(-2, 15)):
    """Print grid with entity names for debugging."""
    grid = engine.hdnrlfmyrj
    player = engine.twdpowducb
    pp = tuple(player.qumspquyus)
    grav = engine.vivnprldht

    sym = {'xcjjwqfzjfe':'W', 'qclfkhjnaac':'D', 'fjlzdjxhant':'*',
           'ubhhgljbnpu':'^', 'hzusueifitk':'v', 'oonshderxef':'O',
           'yuuqpmlxorv':'B', 'lrpkmzabbfa':'G', 'etlsaqqtjvn':'E',
           'aknlbboysnc':'c', 'jcyhkseuorf':'F'}

    gem = None
    for y in range(y_range[0], y_range[1]):
        row = ""
        has = False
        for x in range(x_range[0], x_range[1]):
            items = grid.jhzcxkveiw(x, y)
            if (x,y) == pp:
                row += "P"; has = True
            elif items:
                name = items[0].name
                s = sym.get(name, '?')
                row += s; has = True
                if name == 'fjlzdjxhant': gem = (x,y)
            else:
                row += "."
        if has:
            print(f"  y={y:3d}: {row}")
    print(f"  Player: {pp}, Gem: {gem}, Grav up: {grav}")
    return pp, gem


def get_pos(engine):
    p = engine.twdpowducb
    return tuple(p.qumspquyus)


def execute_and_advance(env, moves, level_name=""):
    """Execute moves and process animation to advance level."""
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


def setup_l2(env):
    """Solve L0 and L1 to reach L2."""
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

    ok, _ = execute_and_advance(env, L0_SOL, "L0")
    if not ok:
        print("L0 FAILED")
        return False
    print(f"L0 solved, level={env._game.level_index}")

    ok, _ = execute_and_advance(env, L1_SOL, "L1")
    if not ok:
        print("L1 FAILED")
        return False
    print(f"L1 solved, level={env._game.level_index}")

    assert env._game.level_index == 2, f"Expected level 2, got {env._game.level_index}"
    print("Now on L2")
    return True


# Phase 1-3 moves (proven to work, gets to (7,13) in 24 steps)
PHASE_1_3 = [
    # Phase 1: Toggle (5,28) B→O, move right, destroy C(6,27)
    C(5,28), R,R,R,R, C(6,27),
    # Phase 2: Toggle 3 blocks at y=23 O→B, move left → fall to (2,19)
    C(5,23), C(4,23), C(3,23), L,L,L,L,
    # Phase 3: Toggle y=17/18 to create crossing, move right → fall to (7,13)
    C(5,17), C(6,17), C(5,18), C(6,18), R,R,R,R,R,
]


# ============================================================
# Test 1: Verify state after Phase 1-3
# ============================================================
print("="*60)
print("Test 1: Verify grid state after Phase 1-3")
print("="*60)
arcade = Arcade()
env = arcade.make('bp35-0a0ad940')
env.reset()
setup_l2(env)

print("\nL2 initial grid:")
engine = env._game.oztjzzyqoek
survey_detailed(engine, y_range=(0, 35))

print(f"\nExecuting Phase 1-3 ({len(PHASE_1_3)} moves)...")
execute(env, PHASE_1_3)

engine = env._game.oztjzzyqoek
pp = get_pos(engine)
print(f"\nAfter Phase 1-3: player at {pp}")
print("\nGrid state (y=5 to y=20):")
survey_detailed(engine, y_range=(5, 20))

# Check what's at each cell in the path from (7,13) toward gem
print("\n--- Path analysis from player to gem ---")
grid = engine.hdnrlfmyrj
for y in range(13, 4, -1):
    for x in range(-1, 10):
        items = grid.jhzcxkveiw(x, y)
        if items:
            name = items[0].name
            sym = {'xcjjwqfzjfe':'W', 'qclfkhjnaac':'D', 'fjlzdjxhant':'*',
                   'ubhhgljbnpu':'^', 'hzusueifitk':'v', 'oonshderxef':'O',
                   'yuuqpmlxorv':'B', 'lrpkmzabbfa':'G', 'etlsaqqtjvn':'E',
                   'aknlbboysnc':'c'}
            s = sym.get(name, '?')
            if (x,y) == pp:
                print(f"  ({x},{y}): PLAYER + {s}({name[:8]})")
            else:
                print(f"  ({x},{y}): {s} ({name[:8]})")


# ============================================================
# Test 2: Try Phase 4 - toggle y=12 and navigate left
# ============================================================
print("\n" + "="*60)
print("Test 2: Phase 4 - toggle y=12 blocks, navigate to gem")
print("="*60)

# Fresh env
arcade2 = Arcade()
env2 = arcade2.make('bp35-0a0ad940')
env2.reset()
setup_l2(env2)
execute(env2, PHASE_1_3)

engine2 = env2._game.oztjzzyqoek
pp = get_pos(engine2)
print(f"Player at {pp}")

# Try: toggle (6,12) B→? and (5,12) B→?
# First check what's at these positions
grid2 = engine2.hdnrlfmyrj
for x in range(-1, 10):
    items = grid2.jhzcxkveiw(x, 12)
    if items:
        name = items[0].name
        print(f"  ({x},12): {name[:12]}")

# Phase 4 attempt: make y=12 traversable left of player
# Toggle B→O at positions to create passable layer
phase4_test = [
    C(6,12),  # toggle whatever's at (6,12)
    C(5,12),  # toggle whatever's at (5,12)
    L,        # walk to (6,13)
    L,        # walk to (5,13) — does player fall through (5,12)=O?
]

print("\nExecuting Phase 4 test...")
for i, m in enumerate(phase4_test):
    if m[0] == 'C':
        print(f"  Step {i}: Click ({m[1]},{m[2]})")
        click_act(env2, m[1], m[2])
    else:
        print(f"  Step {i}: {'LEFT' if m[0]=='L' else 'RIGHT'}")
        env2.step(LEFT if m[0]=='L' else RIGHT)

    pp = get_pos(engine2)
    print(f"    -> Player at {pp}")

    # Check level transition
    if env2._game.level_index > 2:
        print("    LEVEL ADVANCED!")
        break

print("\nGrid after Phase 4 test:")
survey_detailed(engine2, y_range=(5, 16))


# ============================================================
# Test 3: Systematic - what blocks are at y=7-12 column by column
# ============================================================
print("\n" + "="*60)
print("Test 3: Detailed column analysis y=5 to y=14")
print("="*60)
arcade3 = Arcade()
env3 = arcade3.make('bp35-0a0ad940')
env3.reset()
setup_l2(env3)
execute(env3, PHASE_1_3)
engine3 = env3._game.oztjzzyqoek

sym = {'xcjjwqfzjfe':'W', 'qclfkhjnaac':'D', 'fjlzdjxhant':'*',
       'ubhhgljbnpu':'^', 'hzusueifitk':'v', 'oonshderxef':'O',
       'yuuqpmlxorv':'B', 'lrpkmzabbfa':'G', 'etlsaqqtjvn':'E',
       'aknlbboysnc':'c'}
grid3 = engine3.hdnrlfmyrj

print("Column analysis (gravity UP = fall upward):")
for x in range(0, 10):
    col = []
    for y in range(5, 15):
        items = grid3.jhzcxkveiw(x, y)
        if items:
            name = items[0].name
            s = sym.get(name, '?')
            col.append(f"y{y}={s}")
        elif (x,y) == get_pos(engine3):
            col.append(f"y{y}=P")
        else:
            col.append(f"y{y}=.")
    print(f"  x={x}: {' '.join(col)}")

# What's the gem exactly?
gem_items = grid3.jhzcxkveiw(7, 7)
if gem_items:
    print(f"\nGem at (7,7): {gem_items[0].name}")
else:
    # Search for gem
    for y in range(0, 30):
        for x in range(-2, 15):
            items = grid3.jhzcxkveiw(x, y)
            if items and items[0].name == 'fjlzdjxhant':
                print(f"\nGem found at ({x},{y})")

# Check what's directly above player at each step going up column x=7
print("\nColumn x=7 going up from player:")
for y in range(13, 4, -1):
    items = grid3.jhzcxkveiw(7, y)
    s = "empty"
    if items:
        name = items[0].name
        s = f"{sym.get(name,'?')}({name[:10]})"
    if (7,y) == get_pos(engine3):
        s = f"PLAYER ({s})"
    print(f"  (7,{y}): {s}")
