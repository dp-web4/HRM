#!/usr/bin/env python3
"""s5i5 L7 solver - route chain around barrier from (54,15) to (24,15)"""

import sys
sys.path.insert(0, 'environment_files/s5i5/a48e4b1d')
import s5i5
from arcengine.enums import GameAction
import heapq
import pickle

def click(g, x, y):
    g.action.id = GameAction.ACTION6
    g.action.data = {'x': x, 'y': y}
    g.step()

def get_state_key(g):
    """Compact hashable state"""
    chains = g.current_level.get_sprites_by_tag('agujdcrunq')
    parts = []
    for c in sorted(chains, key=lambda s: s.name):
        if c.name == 'qivjsoaoda':  # skip wall
            continue
        if c.name == 'pcovmzgxdv':  # skip orange (don't touch)
            continue
        if c.name == 'ruktfcnnju':  # skip azure (isolated)
            continue
        parts.append((c.x, c.y, c.width, c.height))
    return tuple(parts)

def tgt_a_pos(g):
    """Get target A position (the one that needs to move)"""
    for t in g.current_level.get_sprites_by_tag('zylvdxoiuq'):
        if not (t.x == 21 and t.y == 6):
            return (t.x, t.y)
    # If both are at (21,6) somehow
    tgts = list(g.current_level.get_sprites_by_tag('zylvdxoiuq'))
    return (tgts[0].x, tgts[0].y)

def tgt_b_pos(g):
    """Check target B hasn't moved"""
    for t in g.current_level.get_sprites_by_tag('zylvdxoiuq'):
        if t.x == 21 and t.y == 6:
            return (21, 6)
    return None

def check_win(g):
    tgt_pos = set()
    for t in g.current_level.get_sprites_by_tag('zylvdxoiuq'):
        tgt_pos.add((t.x, t.y))
    goal_pos = set()
    for gl in g.current_level.get_sprites_by_tag('cpdhnkdobh'):
        goal_pos.add((gl.x, gl.y))
    return tgt_pos == goal_pos

GOAL = (24, 15)

# All actions (excluding ORG controls and AZU+)
ACTIONS = [
    ('WHT-R', 10, 51),
    ('WHT-L', 4, 51),
    ('MRN-R', 32, 51),
    ('MRN-L', 26, 51),
    ('GRY-R', 10, 58),
    ('GRY-L', 4, 58),
    ('CYN-R', 32, 58),
    ('CYN-L', 26, 58),
    ('WHT+', 17, 51),
    ('MRN+', 39, 51),
    ('GRY+', 17, 58),
]

def replay(actions):
    """Replay action sequence from fresh game"""
    g = s5i5.S5i5()
    g.set_level(6)
    for label, cx, cy in actions:
        click(g, cx, cy)
    return g

def heuristic(pos):
    return abs(pos[0] - GOAL[0]) + abs(pos[1] - GOAL[1])

# A* search
MAX_DEPTH = 30
g0 = s5i5.S5i5()
g0.set_level(6)
init_key = get_state_key(g0)
init_pos = tgt_a_pos(g0)

print(f'Initial target A: {init_pos}, goal: {GOAL}')
print(f'Manhattan distance: {heuristic(init_pos)}')
print(f'Actions available: {len(ACTIONS)}')
print()

# Priority queue: (f_score, depth, counter, actions, state_key)
counter = 0
pq = []
heapq.heappush(pq, (heuristic(init_pos), 0, counter, []))
visited = {init_key: 0}  # state -> min depth

best_dist = heuristic(init_pos)
best_seq = []
states_checked = 0

while pq:
    f_score, depth, _, seq = heapq.heappop(pq)

    if depth >= MAX_DEPTH:
        continue

    states_checked += 1
    if states_checked % 1000 == 0:
        print(f'  Checked {states_checked} states, depth={depth}, f={f_score}, best_dist={best_dist}, queue={len(pq)}')

    if states_checked > 200000:
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

        pos = tgt_a_pos(g)
        dist = heuristic(pos)

        if dist < best_dist:
            best_dist = dist
            best_seq = [a[0] for a in new_seq]
            print(f'  New best: dist={dist}, pos={pos}, depth={new_depth}, seq={best_seq}')

        if pos == GOAL:
            # Check target B is still ok
            tb = tgt_b_pos(g)
            if tb == (21, 6) and check_win(g):
                print(f'\n*** SOLVED in {new_depth} clicks! ***')
                print(f'Sequence: {[a[0] for a in new_seq]}')
                sys.exit(0)
            elif tb != (21, 6):
                print(f'  Target A at goal but B moved to {tb}')
            else:
                print(f'  Target A at goal, B ok, but win check failed')
                # Print all positions
                for c in g.current_level.get_sprites_by_tag('agujdcrunq'):
                    if c.name not in ('qivjsoaoda', 'pcovmzgxdv', 'ruktfcnnju'):
                        print(f'    {c.name}: ({c.x},{c.y},{c.width},{c.height})')

        counter += 1
        heapq.heappush(pq, (new_depth + dist, new_depth, counter, new_seq))

print(f'\nSearch complete. Best distance: {best_dist}')
print(f'Best sequence ({len(best_seq)} moves): {best_seq}')
print(f'Total states: {states_checked}')
