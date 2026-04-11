#!/usr/bin/env python3
"""BP35 solver: platformer with gravity, destroyable blocks, spikes, gem target.

Actions: LEFT(3), RIGHT(4), CLICK(6 with x,y), UNDO(7)
Gravity UP: player falls upward through empty space.
Click destroyable block: removes it. If directly above player, triggers fall.
Reach gem = WIN. Land on spike = LOSE.
"""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
CLICK = GameAction.ACTION6
UNDO = GameAction.ACTION7


def click_act(env, gx, gy):
    engine = env._game.oztjzzyqoek
    cam_y = engine.camera.rczgvgfsfb[1]
    return env.step(CLICK, data={"x": gx*6, "y": gy*6 - cam_y})


def execute_and_advance(env, moves, level_name=""):
    """Execute a move sequence and process animation to advance level.
    moves: list of ('R',), ('L',), ('C', gx, gy)
    Returns (success, actions_used)
    """
    old_level = env._game.level_index
    step = 0

    for m in moves:
        step += 1
        if m[0] == 'R':
            fd = env.step(RIGHT)
        elif m[0] == 'L':
            fd = env.step(LEFT)
        elif m[0] == 'C':
            fd = click_act(env, m[1], m[2])
        else:
            print(f"  Unknown: {m}")
            continue

        engine = env._game.oztjzzyqoek
        p = engine.twdpowducb
        lvl = env._game.level_index

        if lvl > old_level:
            print(f"  {level_name} step {step}/{len(moves)}: Level advanced! ({step} actions)")
            return True, step

    # Process animation frames (level transition needs extra steps)
    for i in range(10):
        step += 1
        fd = env.step(LEFT)
        lvl = env._game.level_index
        if lvl > old_level:
            print(f"  {level_name} animation step {i+1}: Level advanced! ({step} actions)")
            return True, step

    # Check if this was the last level
    if env._game.level_index == old_level:
        engine = env._game.oztjzzyqoek
        p = engine.twdpowducb
        print(f"  {level_name} FAILED after {step} actions. player=({p.qumspquyus[0]},{p.qumspquyus[1]})")
    return False, step


def survey(engine):
    """Return grid dict and key positions."""
    grid = engine.hdnrlfmyrj
    cells = {}
    gem = None
    destroyables = set()
    spikes = set()
    walls = set()

    for y in range(-5, 60):
        for x in range(-5, 20):
            items = grid.jhzcxkveiw(x, y)
            if items:
                name = items[0].name
                cells[(x,y)] = name
                if name == 'fjlzdjxhant': gem = (x,y)
                elif name == 'qclfkhjnaac': destroyables.add((x,y))
                elif name in ('ubhhgljbnpu', 'hzusueifitk'): spikes.add((x,y))
                elif name == 'xcjjwqfzjfe': walls.add((x,y))

    player = engine.twdpowducb
    pp = tuple(player.qumspquyus)
    grav = engine.vivnprldht
    return cells, gem, destroyables, spikes, walls, pp, grav


def print_grid(engine, y_range=(-2, 45)):
    cells, gem, destroyables, spikes, walls, pp, grav = survey(engine)
    sym = {'xcjjwqfzjfe':'W', 'qclfkhjnaac':'D', 'fjlzdjxhant':'*',
           'ubhhgljbnpu':'^', 'hzusueifitk':'v', 'oonshderxef':'O',
           'yuuqpmlxorv':'B', 'lrpkmzabbfa':'G', 'etlsaqqtjvn':'E',
           'aknlbboysnc':'c'}
    for y in range(y_range[0], y_range[1]):
        row = ""
        has = False
        for x in range(-2, 15):
            if (x,y) == pp: row += "P"; has = True
            elif (x,y) in cells: row += sym.get(cells[(x,y)],'?'); has = True
            else: row += "."
        if has:
            print(f"  y={y:3d}: {row}")
    print(f"  Player: {pp}, Gem: {gem}, Grav up: {grav}")
    print(f"  D: {len(destroyables)}, Spikes: {len(spikes)}")


# ============================================================
# Solutions
# ============================================================

# Helper to build move lists
R = ('R',)
L = ('L',)
def C(x, y): return ('C', x, y)

# L0: player (3,23), gem (3,7), gravity UP
# Path: 4R → gap at x=7 y=22 → fall to (7,20) → destroy up → navigate to gem
L0_SOL = [
    R, R, R, R,          # → (7,20) via fall through y=22 gap
    C(7,19),             # destroy, fall to (7,16)
    C(4,16),             # destroy (no fall)
    L, L, L,             # → (4,16)
    C(4,15),             # destroy, fall to (4,13)
    C(4,12),             # destroy, fall to (4,10)
    R,                   # → (5,10)
    C(5,9),              # destroy, fall to (5,7) but gem at (3,7) — need to check
    L, L,                # → (3,7) = gem! WIN
]

# L1: player (3,37), gem (5,7), gravity UP
# Navigate: right side (avoid y=32 spikes) → cross left (avoid y=25 spikes) →
# x=3 shaft (y=18 safe) → right side (avoid y=12 spikes) → gem
L1_SOL = [
    # Phase 1: right side, past D ceiling
    R, R, R, R, R,       # → (8,37)
    C(8,36),             # fall to (8,36)
    C(8,35),             # fall to (8,29)

    # Phase 2: cross left through y=29 D blocks
    L, L,                # → (6,29)
    C(5,29), L,          # destroy, walk to (5,29)
    C(4,29), L,          # destroy, walk to (4,29)
    C(3,29), L,          # destroy, walk to (3,29)
    C(2,29), L,          # destroy, walk to (2,29)

    # Phase 3: up through y=28 to safe left side
    C(2,28),             # destroy, fall to (2,25)

    # Phase 4: navigate to x=5 and up through y=24-23
    R, R, R,             # → (5,25)
    C(5,24),             # destroy, fall to (5,24)
    C(5,23),             # destroy, fall to (5,21)

    # Phase 5: x=3 safe passage through y=18
    L, L,                # → (3,21)
    C(3,20),             # destroy, fall to (3,18)

    # Phase 6: through D shaft and right
    C(3,17),             # destroy, fall to (3,17)
    C(3,16),             # destroy, fall to (3,16)

    # Phase 6b: traverse y=16 D blocks rightward
    C(4,16), R,          # → (4,16)
    C(5,16), R,          # → (5,16)
    C(6,16), R,          # → (6,16)
    C(7,16), R,          # → (7,16)
    C(8,16), R,          # → (8,16)

    # Phase 6c: up through right side
    C(8,15),             # destroy, fall to (8,15)
    C(8,14),             # destroy, fall to (8,11)

    # Phase 6d: navigate left to gem column
    L,                   # → (7,11)
    L,                   # → (6,11) → falls to (6,10)
    L,                   # → (5,10)
    C(5,9),              # destroy, fall to gem (5,7)! WIN
]


# L2: player (3,28), gem (7,7), gravity UP, rising floor y=34
# Route: destroy D→fall(6,24) → toggle y=23 → fall(2,19) → cross y=17/18 →
# fall(7,13) → toggle y=12 → fall(3,7) → toggle (5,7) → walk to gem
L2_SOL = [
    # Phase A: get to (6,24) via D destruction (5 actions)
    C(5,28), R, R, R, C(6,27),
    # Phase B: toggle y=23 O→B, walk left, fall to (2,19) (7 actions)
    C(5,23), C(4,23), C(3,23), L, L, L, L,
    # Phase C: cross right to (3,18) (1 action)
    R,
    # Phase D: toggle y=17/18, cross to (7,13) (8 actions)
    C(5,17), C(6,17), C(5,18), C(6,18), R, R, R, R,
    # Phase E: toggle y=12, navigate to (3,7) (8 actions)
    C(6,12), C(5,12), C(4,12), C(3,12), L, L, L, L,
    # Phase F: reach gem (5 actions)
    C(5,7), R, R, R, R,
]

# L3: player (4,14), gem (4,27), gravity UP, no rising floor
# Key: 4th hidden G block at (4,31) behind walls, clickable remotely
# Destroy D blocks to create fall paths, use all 4 G flips to navigate
L3_SOL = [
    C(3,17),        # destroy D
    C(7,23),        # destroy D
    C(7,24),        # destroy D
    C(5,7),         # G flip → grav DOWN, fall to (4,16)
    L,              # fall to (3,21)
    R, R,           # → (5,21)
    C(3,23),        # G flip → grav UP, rise to (5,20)
    R, R,           # → (7,20)
    C(5,23),        # G flip → grav DOWN, fall to (7,29)
    L, L, L,        # → (4,29)
    C(4,31),        # G flip → grav UP, fall up to gem (4,27)!
]


# ============================================================
# Main
# ============================================================
arcade = Arcade()
env = arcade.make('bp35-0a0ad940')
fd = env.reset()

print("=" * 60)
print("BP35 Solver")
print("=" * 60)

# Solve levels
solutions = [
    ("L0", L0_SOL, 15),
    ("L1", L1_SOL, 72),
    ("L2", L2_SOL, 36),
    ("L3", L3_SOL, 31),
]

for name, sol, baseline in solutions:
    print(f"\n{'='*40}")
    print(f"{name} (baseline={baseline}, our={len(sol)} actions)")
    print(f"{'='*40}")

    engine = env._game.oztjzzyqoek
    print_grid(engine)

    success, steps_used = execute_and_advance(env, sol, name)
    if not success:
        print(f"\n{name} failed!")
        break
    print(f"  {name} solved in {len(sol)} actions (baseline {baseline})")

    # Check if there are more levels
    current_level = env._game.level_index
    print(f"  Now on level {current_level}")

# Survey next level if we have one
current_level = env._game.level_index
if current_level >= len(solutions):
    print(f"\n{'='*40}")
    print(f"L{current_level} survey")
    print(f"{'='*40}")
    engine = env._game.oztjzzyqoek
    print_grid(engine)
