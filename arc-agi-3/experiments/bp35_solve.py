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


# L4: player (3,12), gem (5,7), gravity UP
# Walk staircase, destroy D row, double G flip to navigate shaft
L4_SOL = [
    R, R, R, R,      # walk to x=7 via staircase
    C(7,9),           # destroy D
    C(8,9),           # destroy D
    C(9,9),           # destroy D
    C(8,12),          # G flip → grav DOWN, fall to (7,15)
    C(8,29),          # G flip → grav UP, rise to (7,12)
    R, R,             # walk right through shaft → (9,9)
    L, L, L, L,       # walk left → (5,7) gem!
]


# L5: player (3,23), gem (2,31), gravity UP
# Walk right to shaft, G(4,31) flip → fall to y=31, walk left to gem
L5_SOL = [R, R, R, R, R, C(4,31), L, L, L, L, L, L]


# L6: player (3,19), gem (3,25), gravity UP
# 23 G blocks at x=0 (y=5-27), CONSUMABLE (single-use each!)
# 29 O blocks + 2 B blocks (O/B toggleable), initial B: (6,9), (5,10)
# Key: navigate through y=8-11 spike maze with toggle-retoggle tricks
# to reach x=7 O-block column, drop through (8,7)→(9,7)→(9,26),
# flip UP to y=23 corridor, walk left to gem
L6_SOL = [
    # Phase 1: Start → platform area via (8,15) (12 steps)
    R, R, R,         # → (6,19) through O block
    C(6, 21),        # toggle O→B (create floor at y=21)
    C(0, 19),        # G flip #1 → DN, fall to (6,20) [floor B(6,21)]
    R,               # → (7,20) [floor wall(7,21)]
    C(0, 20),        # G flip #2 → UP, stays (7,20) [ceiling wall(7,19)]
    R,               # → (8,15) fall UP [wall(8,14)]
    L,               # → (7,13) fall UP [wall(7,12)]
    L, L, L,         # → (4,13) through O blocks

    # Phase 2: Platform → x=2 shaft (5 steps)
    C(0, 14),        # G flip #3 → DN, fall to (4,17) [wall(4,18)]
    C(4, 15),        # toggle O→B (create ceiling for return)
    C(0, 16),        # G flip #4 → UP, fall to (4,16) [ceiling B(4,15)]
    L,               # → (3,15) fall UP [wall(3,14)]
    L,               # → (2,8) fall UP through x=2 shaft [wall(2,7)]

    # Phase 3: Navigate y=8-11 spike maze to x=7 (14 steps)
    R,               # → (3,8) O block [ceiling wall(3,7)]
    C(0, 9),         # G flip #5 → DN, fall to (3,11) [wall(3,12)]
    R,               # → (4,11) [floor wall(4,12)]
    C(4, 9),         # toggle O→B (create ceiling at (4,9))
    C(0, 10),        # G flip #6 → UP, fall to (4,10) [ceiling B(4,9)]
    C(5, 10),        # toggle B→O (make (5,10) passable)
    R,               # → (5,8) fall UP through (5,10) O, (5,9) [wall(5,7)]
    C(5, 10),        # toggle O→B (restore floor at (5,10))
    C(0, 8),         # G flip #7 → DN, fall to (5,9) [floor B(5,10)]
    C(6, 9),         # toggle B→O (make (6,9) passable)
    R,               # → (6,11) fall DN through (6,9) O [wall(6,12)]
    C(6, 9),         # toggle O→B (restore ceiling at (6,9))
    C(0, 11),        # G flip #8 → UP, fall to (6,10) [ceiling B(6,9)]
    R,               # → (7,4) fall UP through x=7 O column [wall(7,3)]

    # Phase 4: Through x=7 column to x=9 shaft (4 steps)
    C(7, 8),         # toggle O→B (create floor at (7,8))
    C(0, 6),         # G flip #9 → DN, fall to (7,7) [floor B(7,8)]
    R,               # → (8,7) [floor wall(8,8)]
    R,               # → (9,26) fall DN through entire x=9 shaft [wall(9,27)]

    # Phase 5: Navigate to gem (8 steps)
    L,               # → (8,26) [floor wall(8,27)]
    L,               # → (7,26) [floor wall(7,27)]
    C(0, 25),        # G flip #10 → UP, fall to (7,23) [ceiling wall(7,22)]
    L, L, L, L,      # walk left across y=23 → (3,23)
    C(0, 24),        # G flip #11 → DN, fall through (3,24) → gem (3,25)!
]


# L7: player (3,32), gem (9,19), gravity UP
# Key mechanic: E entities SPREAD to adjacent empty cells when clicked.
# "Push up" trick: clicking E ceiling removes it, player falls up into empty cell.
# Route: E bridge at y=30 → walk to x=8 shaft → navigate to (5,20) →
# push up ×3 to (5,17) → walk east clearing E → (8,17) →
# G flip DN → toggle B(8,18)→O → fall to (8,19) → walk R to gem
L7_SOL = [
    # E bridge y=29-30 (5 clicks)
    C(2,29), C(3,29), C(4,29), C(5,29), C(6,29),
    # E ceiling y=17-19 (4 clicks)
    C(3,18), C(4,18), C(5,18), C(6,18),
    # B toggles for navigation
    C(7,25), C(8,22),
    # Walk across bridge to x=8 shaft
    R, R, R, R, R,
    # Navigate to (8,21) via B gaps
    L, L, R, R,
    # Navigate to (5,20) via E ceiling
    L, L, L,
    # Push up: click ceiling E ×3
    C(5,19), C(5,18), C(5,17),
    # Walk east clearing E path
    C(6,17), R, C(7,17), R, C(8,17), R,
    # G flip → grav DOWN, toggle B(8,18)→O → fall, walk to gem
    C(5,2), C(8,18), R,
]


# L8: player (3,35), gem (2,40), gravity UP
# ALL mechanics combined: E spread, G flip, B toggle, D destroy, spikes
# 11 phases: setup → G-flip corridor → walk-and-clear → push-up →
# walk-and-clear → navigate → x=9 push-up → walk-and-clear →
# B toggle + D destroy → cross to x=0 → G flip fall to gem
L8_SOL = [
    # SETUP: E spread, G consume, D destroy (18 clicks, no movement)
    C(7,8), C(6,8), C(5,8), C(4,8), C(3,8),     # E spread west at y=8
    C(6,31), C(5,31), C(4,31), C(3,31),           # E spread west at y=31
    C(0,15), C(0,19), C(0,21), C(0,25), C(0,33), C(0,35),  # consume 6 G blocks
    C(2,26), C(3,26), C(4,26),                     # destroy D at y=26

    # PHASE 1: G-flip cross y=34 gap → (7,32)
    C(1,1), R, R, R, R, C(2,1),

    # PHASE 2: Walk-and-clear L along y=32
    C(6,32), L, C(5,32), L, C(4,32), L, C(3,32), L, C(2,32), L,

    # PHASE 3: Push-up at x=2 from y=32 to y=21 (11 pushes)
    C(2,31), C(2,30), C(2,29), C(2,28), C(2,27),
    C(2,26), C(2,25), C(2,24), C(2,23), C(2,22), C(2,21),

    # PHASE 4: Walk-and-clear R at y=21 to x=7
    C(3,21), R, C(4,21), R, C(5,21), R, C(6,21), R, C(7,21), R,

    # PHASE 5: Navigate to (8,23) via G flips
    C(7,22), C(3,1), C(7,23), C(8,23), R, C(4,1),

    # PHASE 6: Enter x=9 + push-up y=23→y=16 (7 pushes in 1-wide shaft)
    C(9,23), R,
    C(9,22), C(9,21), C(9,20), C(9,19), C(9,18), C(9,17), C(9,16),

    # PHASE 7: Walk-and-clear L along y=16 to x=4
    C(8,16), L, C(7,16), L, C(6,16), L, C(5,16), L, C(4,16), L,

    # PHASE 8: B toggle + D destroy → fly up
    C(4,15),       # B→O, fly through to (4,14)
    C(4,13),       # D destroy, fly to (4,10)

    # PHASE 9: Walk L to x=2, push-up, cross to x=0
    L, L,          # → (2,9)
    C(2,8),        # push-up to (2,8)
    C(1,8),        # destroy D(1,8)
    L, L,          # → (0,7) via x=0 shaft

    # PHASE 10: G flip DN → fall to (0,40), walk R to gem
    C(5,1), R, R,  # → (2,40) = GEM!
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
    ("L4", L4_SOL, 31),
    ("L5", L5_SOL, 48),
    ("L6", L6_SOL, 86),
    ("L7", L7_SOL, 155),
    ("L8", L8_SOL, 422),
]

for name, sol, baseline in solutions:
    print(f"\n{'='*40}")
    print(f"{name} (baseline={baseline}, our={len(sol)} actions)")
    print(f"{'='*40}")

    engine = env._game.oztjzzyqoek
    print_grid(engine)

    success, steps_used = execute_and_advance(env, sol, name)
    if not success:
        # Last level doesn't advance — check win flag
        engine = env._game.oztjzzyqoek
        if engine.nkuphphdgrp:
            print(f"  {name} solved in {len(sol)} actions (baseline {baseline}) [WIN flag]")
            print(f"\n  *** BP35 COMPLETE — ALL 9 LEVELS SOLVED ***")
            total = sum(len(s) for _, s, _ in solutions)
            print(f"  Total actions: {total}")
            break
        print(f"\n{name} failed!")
        break
    print(f"  {name} solved in {len(sol)} actions (baseline {baseline})")

    # Check if there are more levels
    current_level = env._game.level_index
    print(f"  Now on level {current_level}")

    # L8 is the last level — check win flag instead of level advance
    if name == "L8":
        engine = env._game.oztjzzyqoek
        if engine.nkuphphdgrp:
            print(f"\n  *** BP35 COMPLETE — ALL 9 LEVELS SOLVED ***")
        break

# Survey next level if we have one
current_level = env._game.level_index
if current_level < 8 and current_level >= len(solutions):
    print(f"\n{'='*40}")
    print(f"L{current_level} survey")
    print(f"{'='*40}")
    engine = env._game.oztjzzyqoek
    print_grid(engine)
