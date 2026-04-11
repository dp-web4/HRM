#!/usr/bin/env python3
"""Dump all L6 walkable cells and find connected components."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
from collections import deque

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}
STEP = 6

SOLUTIONS = {
    0: 'RIGHT RIGHT RIGHT RIGHT UNDO DOWN DOWN DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT RIGHT',
    1: 'LEFT LEFT UNDO DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT UNDO UP UP UP LEFT LEFT LEFT LEFT LEFT LEFT LEFT DOWN DOWN RIGHT RIGHT RIGHT',
    2: 'UP UP RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN RIGHT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT LEFT LEFT UP UP UP RIGHT RIGHT UP UP',
    3: 'DOWN DOWN RIGHT DOWN UNDO DOWN DOWN RIGHT RIGHT UP UP RIGHT RIGHT DOWN DOWN DOWN UNDO LEFT LEFT LEFT DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT LEFT LEFT LEFT',
    4: 'UP DOWN DOWN RIGHT RIGHT RIGHT DOWN DOWN DOWN UNDO DOWN RIGHT RIGHT RIGHT UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN UNDO DOWN RIGHT RIGHT RIGHT UP UP RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN RIGHT LEFT DOWN LEFT LEFT LEFT LEFT LEFT UP UP',
    5: 'LEFT LEFT UP UNDO LEFT LEFT UP LEFT LEFT UNDO LEFT LEFT DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT LEFT DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT UP DOWN LEFT LEFT UP UP UP UP UP RIGHT RIGHT RIGHT DOWN DOWN RIGHT RIGHT DOWN DOWN RIGHT RIGHT',
}

arcade = Arcade()
env = arcade.make('g50t-5849a774')
fd = env.reset()
for lv in range(6):
    for name in SOLUTIONS[lv].split():
        fd = env.step(INT_TO_GA[NAME_TO_INT[name]])

game = env._game
lc = game.vgwycxsxjz
player = lc.dzxunlkwxt
arena = lc.afbbgvkpip
offset = player.x % STEP
orig_x, orig_y = player.x, player.y

walkable = set()
obs_cells = {}
for px in range(offset, 80, STEP):
    for py in range(offset, 80, STEP):
        player.set_position(px, py)
        if lc.xvkyljflji(player, arena):
            if lc.vjpujwqrto(player):
                obs_list = lc.uwxkstolmf
                oi = next((i for i, o in enumerate(obs_list) if o.x == px and o.y == py), '?')
                obs_cells[(px, py)] = oi
            else:
                walkable.add((px, py))
player.set_position(orig_x, orig_y)

# Print all cells sorted by y, then x
print('All walkable cells:')
all_cells = sorted(walkable | set(obs_cells.keys()), key=lambda c: (c[1], c[0]))
for y_val in sorted(set(c[1] for c in all_cells)):
    row_cells = sorted(c for c in all_cells if c[1] == y_val)
    labels = []
    for c in row_cells:
        if c in obs_cells:
            labels.append(f'#{obs_cells[c]}({c[0]})')
        else:
            labels.append(f'{c[0]}')
    print(f'  y={y_val:2d}: {", ".join(labels)}')

# Find connected components (walkable only, no obs)
def neighbors(pos):
    x, y = pos
    for dx, dy in [(STEP,0),(-STEP,0),(0,STEP),(0,-STEP)]:
        nb = (x+dx, y+dy)
        if nb in walkable:
            yield nb

visited = set()
components = []
for cell in walkable:
    if cell in visited:
        continue
    comp = set()
    queue = deque([cell])
    while queue:
        p = queue.popleft()
        if p in comp:
            continue
        comp.add(p)
        visited.add(p)
        for nb in neighbors(p):
            if nb not in comp:
                queue.append(nb)
    components.append(comp)

components.sort(key=lambda c: -len(c))
mods = lc.hamayflsib
mod_map = {(m.x, m.y): (i, type(m).__name__) for i, m in enumerate(mods)}
goal = (31, 49)

print(f'\nConnected components: {len(components)}')
for ci, comp in enumerate(components):
    special = []
    for c in sorted(comp):
        if c in mod_map:
            mi, mtype = mod_map[c]
            short = 'pad' if mtype == 'lqtxaumfed' else 'portal'
            special.append(f'mod[{mi}]({c[0]},{c[1]})={short}')
        if c == (25, 25):
            special.append(f'PLAYER({c[0]},{c[1]})')
        if c == goal:
            special.append(f'GOAL({c[0]},{c[1]})')
    print(f'\n  Component {ci} ({len(comp)} cells): {", ".join(special) if special else "no specials"}')
    for c in sorted(comp, key=lambda c: (c[1], c[0])):
        print(f'    {c}')

# Show obs blocking analysis
print(f'\n=== Obstacle blocking analysis ===')
for pos, oi in obs_cells.items():
    # What does this obs connect?
    x, y = pos
    adj = []
    for dx, dy, d in [(STEP,0,'RIGHT'),(-STEP,0,'LEFT'),(0,STEP,'DOWN'),(0,-STEP,'UP')]:
        nb = (x+dx, y+dy)
        if nb in walkable:
            # Find which component
            for ci, comp in enumerate(components):
                if nb in comp:
                    adj.append(f'{d}→comp{ci}({nb})')
    print(f'  obs[{oi}] at {pos}: {", ".join(adj)}')
