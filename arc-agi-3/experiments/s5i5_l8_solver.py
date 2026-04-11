#!/usr/bin/env python3
"""s5i5 L8 solver — target (18,3) to (18,42)"""

import sys
sys.path.insert(0, 'environment_files/s5i5/a48e4b1d')
import s5i5
from arcengine.enums import GameAction
import heapq

def click(g, x, y):
    g.action.id = GameAction.ACTION6
    g.action.data = {'x': x, 'y': y}
    g.step()

def get_state_key(g):
    chains = g.current_level.get_sprites_by_tag('agujdcrunq')
    parts = []
    for c in sorted(chains, key=lambda s: s.name):
        # Skip walls
        if c.name in ('tvidpllbtw', 'pfqqmzqpef'):
            continue
        parts.append((c.x, c.y, c.width, c.height))
    return tuple(parts)

def tgt_pos(g):
    for t in g.current_level.get_sprites_by_tag('zylvdxoiuq'):
        return (t.x, t.y)

def check_win(g):
    tgt_pos_set = set()
    for t in g.current_level.get_sprites_by_tag('zylvdxoiuq'):
        tgt_pos_set.add((t.x, t.y))
    goal_pos = set()
    for gl in g.current_level.get_sprites_by_tag('cpdhnkdobh'):
        goal_pos.add((gl.x, gl.y))
    return tgt_pos_set == goal_pos

GOAL = (18, 42)

ACTIONS = [
    ('S1-R', 14, 56), ('S1-L', 8, 56),
    ('S2-R', 59, 11), ('S2-L', 53, 11),
    ('S3-R', 45, 4),  ('S3-L', 39, 4),
    ('S4-R', 45, 11), ('S4-L', 39, 11),
    ('S5-R', 28, 56), ('S5-L', 22, 56),
    ('S6-R', 59, 4),  ('S6-L', 53, 4),
    ('WHT+', 48, 17),
]

def replay(actions):
    g = s5i5.S5i5()
    g.set_level(7)
    for label, cx, cy in actions:
        click(g, cx, cy)
    return g

def heuristic(pos):
    return abs(pos[0] - GOAL[0]) + abs(pos[1] - GOAL[1])

MAX_DEPTH = 30

g0 = s5i5.S5i5()
g0.set_level(7)
init_key = get_state_key(g0)
init_pos = tgt_pos(g0)

print(f'L8: target {init_pos} -> goal {GOAL}')
print(f'Manhattan distance: {heuristic(init_pos)}')
print(f'Actions: {len(ACTIONS)}')

counter = 0
pq = []
heapq.heappush(pq, (heuristic(init_pos), 0, counter, []))
visited = {init_key: 0}

best_dist = heuristic(init_pos)
best_seq = []
states_checked = 0

while pq:
    f_score, depth, _, seq = heapq.heappop(pq)
    if depth >= MAX_DEPTH:
        continue
    states_checked += 1
    if states_checked % 2000 == 0:
        print(f'  {states_checked} states, d={depth}, f={f_score}, best={best_dist}, q={len(pq)}')
    if states_checked > 500000:
        print('Search limit reached')
        break

    for label, cx, cy in ACTIONS:
        new_seq = seq + [(label, cx, cy)]
        new_depth = depth + 1
        g = replay(new_seq)
        new_key = get_state_key(g)
        if new_key in visited and visited[new_key] <= new_depth:
            continue
        visited[new_key] = new_depth
        pos = tgt_pos(g)
        dist = heuristic(pos)
        if dist < best_dist:
            best_dist = dist
            best_seq = [a[0] for a in new_seq]
            print(f'  New best: dist={dist}, pos={pos}, depth={new_depth}, seq={best_seq}')
        if pos == GOAL and check_win(g):
            print(f'\n*** L8 SOLVED in {new_depth} clicks! ***')
            print(f'Sequence: {[a[0] for a in new_seq]}')
            sys.exit(0)
        counter += 1
        heapq.heappush(pq, (new_depth + dist, new_depth, counter, new_seq))

print(f'\nNot solved. Best: dist={best_dist}, seq={best_seq}')
print(f'States: {states_checked}')
