#!/usr/bin/env python3
"""Build L6 adjacency graph and find paths."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
from collections import deque

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}
STEP = 6
DIRS = {'UP': (0, -STEP), 'DOWN': (0, STEP), 'LEFT': (-STEP, 0), 'RIGHT': (STEP, 0)}

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
obs_list = lc.uwxkstolmf
mods = lc.hamayflsib

offset = player.x % STEP
orig_x, orig_y = player.x, player.y

# Build walkable set
walkable = set()
obs_cells = set()
for px in range(offset, 80, STEP):
    for py in range(offset, 80, STEP):
        player.set_position(px, py)
        if lc.xvkyljflji(player, arena):
            if lc.vjpujwqrto(player):
                obs_cells.add((px, py))
            else:
                walkable.add((px, py))
player.set_position(orig_x, orig_y)

print(f'Walkable: {len(walkable)}, Obs blocked: {len(obs_cells)}')
print(f'Obs cells: {obs_cells}')

# Build adjacency (walkable cells only, no obs)
def neighbors(pos, extra_walkable=set()):
    x, y = pos
    for dx, dy in DIRS.values():
        nb = (x+dx, y+dy)
        if nb in walkable or nb in extra_walkable:
            yield nb

def bfs(start, targets, extra_walkable=set()):
    """BFS from start, returns {target: path} for each reachable target."""
    visited = {start: None}
    queue = deque([start])
    found = {}
    while queue:
        pos = queue.popleft()
        if pos in targets:
            # Reconstruct path
            path = []
            p = pos
            while visited[p] is not None:
                prev = visited[p]
                dx, dy = p[0]-prev[0], p[1]-prev[1]
                if dx > 0: path.append('RIGHT')
                elif dx < 0: path.append('LEFT')
                elif dy > 0: path.append('DOWN')
                else: path.append('UP')
                p = prev
            path.reverse()
            found[pos] = path
        for nb in neighbors(pos, extra_walkable):
            if nb not in visited:
                visited[nb] = pos
                queue.append(nb)
    return found, visited

# Key locations
start = (25, 25)
mod_locs = {i: (m.x, m.y) for i, m in enumerate(mods)}
goal = (31, 49)

print(f'\nKey locations:')
for i, loc in mod_locs.items():
    print(f'  mod[{i}]: {loc} ({type(mods[i]).__name__})')
print(f'  Goal: {goal}')

# BFS from player start (no obs cleared)
print(f'\n=== Reachability from {start} (no obs cleared) ===')
targets = set(mod_locs.values()) | {goal}
found, visited = bfs(start, targets)
for t, path in sorted(found.items()):
    name = next((f'mod[{i}]' for i, l in mod_locs.items() if l == t), str(t))
    print(f'  {name} at {t}: {len(path)} steps = {" ".join(path)}')

unreachable = targets - set(found.keys())
if unreachable:
    print(f'  UNREACHABLE: {unreachable}')

# What cells are reachable?
print(f'  Total reachable cells: {len(visited)}')

# === Scenario: obs[1] toggled (removed from blocking) ===
print(f'\n=== Reachability from {start} with obs[1] cleared ===')
obs1_pos = (obs_list[1].x, obs_list[1].y)
found2, visited2 = bfs(start, targets, extra_walkable={obs1_pos})
for t, path in sorted(found2.items()):
    name = next((f'mod[{i}]' for i, l in mod_locs.items() if l == t), str(t))
    print(f'  {name} at {t}: {len(path)} steps = {" ".join(path)}')
unreachable2 = targets - set(found2.keys())
if unreachable2:
    print(f'  UNREACHABLE: {unreachable2}')

# === Scenario: obs[0] shifted (31,1) opens, (25,1) blocks (but was already wall) ===
print(f'\n=== Reachability from {start} with obs[0] shifted (31,1 open) ===')
obs0_pos = (obs_list[0].x, obs_list[0].y)  # (31,1)
found3, visited3 = bfs(start, targets, extra_walkable={obs0_pos})
for t, path in sorted(found3.items()):
    name = next((f'mod[{i}]' for i, l in mod_locs.items() if l == t), str(t))
    print(f'  {name} at {t}: {len(path)} steps = {" ".join(path)}')
unreachable3 = targets - set(found3.keys())
if unreachable3:
    print(f'  UNREACHABLE: {unreachable3}')

# === Scenario: BOTH obs cleared/shifted ===
print(f'\n=== Reachability from {start} with BOTH obs cleared ===')
found4, visited4 = bfs(start, targets, extra_walkable={obs0_pos, obs1_pos})
for t, path in sorted(found4.items()):
    name = next((f'mod[{i}]' for i, l in mod_locs.items() if l == t), str(t))
    print(f'  {name} at {t}: {len(path)} steps = {" ".join(path)}')
unreachable4 = targets - set(found4.keys())
if unreachable4:
    print(f'  UNREACHABLE: {unreachable4}')

# === Portal teleport paths ===
# If we're at portal (13,25) and get teleported to (13,13), what can we reach?
print(f'\n=== Reachability from (13,13) [left portal dest] with obs[0] shifted ===')
found5, visited5 = bfs((13, 13), targets, extra_walkable={obs0_pos})
for t, path in sorted(found5.items()):
    name = next((f'mod[{i}]' for i, l in mod_locs.items() if l == t), str(t))
    print(f'  {name} at {t}: {len(path)} steps = {" ".join(path)}')

# From (13,13) UP to (13,7) UP to (13,1) = mod[3]. From mod[3], obs[0] shifts.
# Then (31,1) opens. But can we reach (31,1) from (13,1)?
# (13,1) → (19,1) wall → can't go right
print(f'\n=== Can (13,1) reach (31,1) on y=1? ===')
found6, _ = bfs((13, 1), {(31, 1), (37, 1), (49, 1)}, extra_walkable={obs0_pos})
if found6:
    for t, path in found6.items():
        print(f'  {t}: {" ".join(path)}')
else:
    print(f'  NO — (13,1) cannot reach right y=1 cluster even with obs[0] shifted')
    print(f'  y=1 walkable: {sorted(c for c in walkable if c[1]==1)}')
    print(f'  y=1 obs: {sorted(c for c in obs_cells if c[1]==1)}')

# === Key question: what if we teleport via RIGHT portal? ===
# Portal (49,49) → what's reachable from there?
print(f'\n=== Reachability from (49,49) [right portal] ===')
found7, _ = bfs((49, 49), {goal}, extra_walkable=set())
if found7:
    for t, path in found7.items():
        print(f'  goal at {t}: {len(path)} steps = {" ".join(path)}')
else:
    print(f'  Cannot reach goal from (49,49)')

# Try from (49,37) - the other right portal
print(f'\n=== Reachability from (49,37) [right portal alt] ===')
found8, visited8 = bfs((49, 37), targets | {goal})
for t, path in sorted(found8.items()):
    name = next((f'mod[{i}]' for i, l in mod_locs.items() if l == t), str(t))
    print(f'  {name} at {t}: {len(path)} steps = {" ".join(path)}')
