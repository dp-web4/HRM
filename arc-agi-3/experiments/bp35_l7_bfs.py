#!/usr/bin/env python3
"""BFS over L7 state space using deepcopy for state snapshots.

State signature: (px, py, grav_up, level, tuple(g_positions), tuple(ones), tuple(twos))
Goal: reach level > 7 (L7 won) via any action sequence.
"""
import sys, os, copy, json, time
sys.path.insert(0, '/mnt/c/exe/projects/ai-agents/ARC-SAGE/arc-agi-3/experiments')
os.chdir('/mnt/c/exe/projects/ai-agents/ARC-SAGE/arc-agi-3/experiments')
sys.setrecursionlimit(50000)
from collections import deque
from arc_agi import Arcade
from arcengine import GameAction

ACT = {'UP': GameAction.ACTION1, 'DOWN': GameAction.ACTION2, 'LEFT': GameAction.ACTION3,
       'RIGHT': GameAction.ACTION4, 'CLICK': GameAction.ACTION6}


def bootstrap():
    """Run all prefix actions to reach L7 spawn, return env snapshot."""
    arc = Arcade(operation_mode='offline', environments_dir='environment_files')
    env = arc.make('bp35-0a0ad940')
    env.reset()
    game = env._game
    # L1-L4
    trace_in = json.load(open('/mnt/c/exe/projects/ai-agents/ARC-SAGE/knowledge/visual-memory/bp35/run_legitimate_chain/run.json'))
    for s in [t for t in trace_in['steps'] if t['step'] <= 143]:
        if s['action'] == 'CLICK':
            env.step(ACT['CLICK'], data={'x': s['x'], 'y': s['y']})
        elif s['action'] == 'CLICK_OOB_SKIPPED':
            continue
        else:
            env.step(ACT[s['action']])
    # L5 win (LEFT)
    env.step(ACT['LEFT'])
    # L6 win
    for _ in range(5): env.step(ACT['RIGHT'])
    def click(gx, gy):
        sc = game.oztjzzyqoek
        cam_y = sc.camera.rczgvgfsfb[1]
        return env.step(ACT['CLICK'], data={'x': gx*6, 'y': gy*6 - cam_y})
    click(6, 22)
    for _ in range(3): env.step(ACT['LEFT'])
    click(4, 31)
    click(8, 1)
    for _ in range(3): env.step(ACT['LEFT'])
    assert game.oztjzzyqoek.qswcochjodb == 7, f'expected L7, got L{game.oztjzzyqoek.qswcochjodb}'
    return env


def signature(game):
    sc = game.oztjzzyqoek
    p = sc.twdpowducb.qumspquyus
    grav = sc.vivnprldht
    lvl = sc.qswcochjodb
    g_list = []
    one_list = []
    two_list = []
    for y in range(40):
        for x in range(11):
            names = [e.name for e in sc.hdnrlfmyrj.jhzcxkveiw(x, y)]
            if 'lrpkmzabbfa' in names: g_list.append((x, y))
            if 'yuuqpmlxorv' in names: one_list.append((x, y))
            if 'oonshderxef' in names: two_list.append((x, y))
    return (p, grav, lvl, tuple(g_list), tuple(one_list), tuple(two_list))


def apply_click(env, gx, gy):
    game = env._game
    sc = game.oztjzzyqoek
    cam_y = sc.camera.rczgvgfsfb[1]
    return env.step(ACT['CLICK'], data={'x': gx*6, 'y': gy*6 - cam_y})


def enumerate_actions(game):
    """Return list of (type, params) for all candidate actions.

    Prune heuristically: clicks on g tiles (any), clicks on 1/2 tiles directly above/below player
    (grav direction), and remote clicks on 1/2 in cols 4-8 rows 13-21 (the key area for row 20 trick).
    """
    actions = [('LEFT', None), ('RIGHT', None)]
    sc = game.oztjzzyqoek
    px, py = sc.twdpowducb.qumspquyus
    grav_up = sc.vivnprldht
    dy = -1 if grav_up else 1
    adjacent = (px, py + dy)
    # Relevant conversion cells for L7 route planning
    relevant_relocate = set()
    for y in range(13, 22):
        for x in range(3, 10):
            relevant_relocate.add((x, y))
    # Also add (6, 9), (5, 10) 1 tiles
    relevant_relocate.add((6, 9))
    relevant_relocate.add((5, 10))
    for (x, y) in relevant_relocate:
        names = [e.name for e in sc.hdnrlfmyrj.jhzcxkveiw(x, y)]
        if 'yuuqpmlxorv' in names or 'oonshderxef' in names:
            actions.append(('CLICK', (x, y)))
    # g clicks: try just a few (all go to col 0, but pick one per row to reduce duplicates)
    for y in range(40):
        names = [e.name for e in sc.hdnrlfmyrj.jhzcxkveiw(0, y)]
        if 'lrpkmzabbfa' in names:
            actions.append(('CLICK', (0, y)))
            break  # one g click is enough for flip
    # adjacent 1/2 click if not already
    ax, ay = adjacent
    if 0 <= ax < 11 and 0 <= ay < 40:
        names = [e.name for e in sc.hdnrlfmyrj.jhzcxkveiw(ax, ay)]
        if 'yuuqpmlxorv' in names or 'oonshderxef' in names:
            if ('CLICK', adjacent) not in actions:
                actions.append(('CLICK', adjacent))
    return actions


def bfs_l7(max_states=50000, max_time=300):
    start_env = bootstrap()
    start_sig = signature(start_env._game)
    print(f'L7 start: p={start_sig[0]}, grav={start_sig[1]}, L={start_sig[2]}')

    start_t = time.time()
    queue = deque([(start_env, [])])
    visited = {start_sig: 0}
    best_level = start_sig[2]
    expansions = 0

    while queue:
        env, path = queue.popleft()
        game = env._game
        expansions += 1
        if expansions % 50 == 0:
            elapsed = time.time() - start_t
            print(f'  exp={expansions}, queue={len(queue)}, visited={len(visited)}, best_L={best_level}, t={elapsed:.1f}s')
            if elapsed > max_time:
                print('TIMEOUT')
                return None
            if len(visited) > max_states:
                print('STATE CAP')
                return None
        # Try each action
        actions = enumerate_actions(game)
        for act_type, param in actions:
            env_new = copy.deepcopy(env)
            try:
                if act_type == 'CLICK':
                    fd = apply_click(env_new, param[0], param[1])
                else:
                    fd = env_new.step(ACT[act_type])
            except Exception as e:
                continue
            if fd.state.name == 'GAME_OVER':
                continue
            new_sig = signature(env_new._game)
            if new_sig[2] > best_level:
                best_level = new_sig[2]
                print(f'  LEVEL UP! path_len={len(path)+1}, new_L={new_sig[2]}, p={new_sig[0]}')
                return path + [(act_type, param)]
            if new_sig in visited:
                continue
            visited[new_sig] = len(path) + 1
            queue.append((env_new, path + [(act_type, param)]))
    return None


if __name__ == '__main__':
    result = bfs_l7(max_states=30000, max_time=300)
    if result:
        print(f'\nWIN PATH: {result}')
    else:
        print('\nNo path found.')
