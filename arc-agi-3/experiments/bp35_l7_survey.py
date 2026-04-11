#!/usr/bin/env python3
"""Detailed L7 survey: map every entity, test E blocks, test B toggleability."""
import sys, os
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

def state(env):
    engine = env._game.oztjzzyqoek
    p = engine.twdpowducb
    return tuple(p.qumspquyus), engine.vivnprldht

def execute_level(env, moves):
    old_level = env._game.level_index
    for m in moves:
        if m[0] == 'R': env.step(RIGHT)
        elif m[0] == 'L': env.step(LEFT)
        elif m[0] == 'C': click_act(env, m[1], m[2])
        if env._game.level_index > old_level:
            return True
    for _ in range(10):
        env.step(LEFT)
        if env._game.level_index > old_level:
            return True
    return False

R = ('R',)
L = ('L',)
def C(x, y): return ('C', x, y)

sols = [
    [R,R,R,R, C(7,19), C(4,16), L,L,L, C(4,15), C(4,12), R, C(5,9), L,L],
    [R,R,R,R,R, C(8,36), C(8,35), L,L, C(5,29),L, C(4,29),L, C(3,29),L, C(2,29),L, C(2,28), R,R,R, C(5,24), C(5,23), L,L, C(3,20), C(3,17), C(3,16), C(4,16),R, C(5,16),R, C(6,16),R, C(7,16),R, C(8,16),R, C(8,15), C(8,14), L, L, L, C(5,9)],
    [C(5,28),R,R,R,C(6,27), C(5,23),C(4,23),C(3,23),L,L,L,L, R, C(5,17),C(6,17),C(5,18),C(6,18),R,R,R,R, C(6,12),C(5,12),C(4,12),C(3,12),L,L,L,L, C(5,7),R,R,R,R],
    [C(3,17),C(7,23),C(7,24),C(5,7), L, R,R, C(3,23), R,R, C(5,23), L,L,L, C(4,31)],
    [R,R,R,R, C(7,9),C(8,9),C(9,9), C(8,12), C(8,29), R,R, L,L,L,L],
    [R,R,R,R,R, C(4,31), L,L,L,L,L,L],
    # L6
    [R,R,R, C(6,21), C(0,19), R, C(0,20), R, L, L,L,L,
     C(0,14), C(4,15), C(0,16), L, L,
     R, C(0,9), R, C(4,9), C(0,10), C(5,10), R, C(5,10), C(0,8), C(6,9), R, C(6,9), C(0,11), R,
     C(7,8), C(0,6), R, R,
     L, L, C(0,25), L,L,L,L, C(0,24)],
]

arcade = Arcade()
env = arcade.make('bp35-0a0ad940')
fd = env.reset()

for i, sol in enumerate(sols):
    ok = execute_level(env, sol)
    if not ok:
        print(f"L{i} FAIL"); sys.exit(1)
print("L0-L6 solved, now on L7")

# === Detailed L7 Survey ===
engine = env._game.oztjzzyqoek
grid = engine.hdnrlfmyrj
player = engine.twdpowducb
pp = tuple(player.qumspquyus)
grav = engine.vivnprldht
cam = engine.camera.rczgvgfsfb

print(f"\nPlayer: {pp}, Grav UP: {grav}, Camera: ({cam[0]},{cam[1]})")

# Map every entity
entities_by_type = {}
all_entities = {}
for y in range(-5, 60):
    for x in range(-5, 20):
        items = grid.jhzcxkveiw(x, y)
        if items:
            for item in items:
                name = item.name
                if name not in entities_by_type:
                    entities_by_type[name] = []
                entities_by_type[name].append((x, y))
                all_entities[(x, y)] = name

print(f"\n=== Entity counts ===")
sym = {'xcjjwqfzjfe':'W(wall)', 'qclfkhjnaac':'D(destroy)', 'fjlzdjxhant':'*(gem)',
       'ubhhgljbnpu':'^(up-spike)', 'hzusueifitk':'v(dn-spike)', 'oonshderxef':'O(pass)',
       'yuuqpmlxorv':'B(solid)', 'lrpkmzabbfa':'G(gravity)', 'etlsaqqtjvn':'E(?)',
       'aknlbboysnc':'c(?)'}
for name, positions in sorted(entities_by_type.items(), key=lambda x: -len(x[1])):
    label = sym.get(name, name)
    print(f"  {label}: {len(positions)}")
    if len(positions) <= 30:
        # Group by y for readability
        by_y = {}
        for x, y in sorted(positions):
            by_y.setdefault(y, []).append(x)
        for y in sorted(by_y):
            xs = by_y[y]
            print(f"    y={y:3d}: x={xs}")

# Print full grid
print(f"\n=== Full Grid ===")
sym2 = {'xcjjwqfzjfe':'W', 'qclfkhjnaac':'D', 'fjlzdjxhant':'*',
        'ubhhgljbnpu':'^', 'hzusueifitk':'v', 'oonshderxef':'O',
        'yuuqpmlxorv':'B', 'lrpkmzabbfa':'G', 'etlsaqqtjvn':'E',
        'aknlbboysnc':'c'}
for y in range(-2, 50):
    row = ""
    has = False
    for x in range(-2, 15):
        if (x, y) == pp:
            row += "P"; has = True
        elif (x, y) in all_entities:
            row += sym2.get(all_entities[(x,y)], '?'); has = True
        else:
            row += "."
    if has:
        print(f"  y={y:3d}: {row}")

# === Test E entity properties ===
print(f"\n=== E entity investigation ===")
e_positions = entities_by_type.get('etlsaqqtjvn', [])
for ex, ey in e_positions:
    items = grid.jhzcxkveiw(ex, ey)
    for item in items:
        print(f"  E at ({ex},{ey}):")
        # Check all attributes
        for attr in dir(item):
            if not attr.startswith('_'):
                try:
                    val = getattr(item, attr)
                    if not callable(val):
                        print(f"    .{attr} = {val}")
                except:
                    pass

# === Test clicking E entities ===
print(f"\n=== Click test on E entities ===")
for ex, ey in e_positions:
    pos_before, grav_before = state(env)
    click_act(env, ex, ey)
    pos_after, grav_after = state(env)
    # Check if E entity still exists
    items_after = grid.jhzcxkveiw(ex, ey)
    names_after = [i.name for i in items_after] if items_after else []
    print(f"  Click E({ex},{ey}): pos {pos_before}→{pos_after} grav {grav_before}→{grav_after} entities_after={names_after}")
    # Undo if something changed
    if pos_before != pos_after or grav_before != grav_after:
        env.step(UNDO)
        print(f"    (undone)")

# === Test clicking B blocks ===
print(f"\n=== B block click test ===")
b_positions = entities_by_type.get('yuuqpmlxorv', [])
for bx, by in b_positions[:5]:  # test first 5
    pos_before, grav_before = state(env)
    items_before = [i.name for i in grid.jhzcxkveiw(bx, by)]
    click_act(env, bx, by)
    pos_after, grav_after = state(env)
    items_after = [i.name for i in grid.jhzcxkveiw(bx, by)] if grid.jhzcxkveiw(bx, by) else []
    changed = items_before != items_after
    print(f"  Click B({bx},{by}): {items_before}→{items_after} {'CHANGED!' if changed else 'same'} pos {pos_before}→{pos_after}")
    if changed or pos_before != pos_after:
        env.step(UNDO)
        print(f"    (undone)")

# === Test movement from start ===
print(f"\n=== Movement test from ({pp[0]},{pp[1]}) ===")
# Try moving right
for i in range(8):
    pos_before, _ = state(env)
    env.step(RIGHT)
    pos_after, _ = state(env)
    print(f"  R{i+1}: {pos_before}→{pos_after}")
    if pos_before == pos_after:
        print(f"    blocked!")
        break
# Undo all
for i in range(8):
    env.step(UNDO)
pos_check, _ = state(env)
print(f"  After undo: {pos_check}")

# Try moving left
for i in range(3):
    pos_before, _ = state(env)
    env.step(LEFT)
    pos_after, _ = state(env)
    print(f"  L{i+1}: {pos_before}→{pos_after}")
    if pos_before == pos_after:
        break
for i in range(3):
    env.step(UNDO)
