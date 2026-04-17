#!/usr/bin/env python3
"""L7 BFS using real engine + deepcopy. Time-limited search."""
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
    arc = Arcade(operation_mode='offline', environments_dir='environment_files')
    env = arc.make('bp35-0a0ad940')
    env.reset()
    game = env._game
    trace_in = json.load(open('/mnt/c/exe/projects/ai-agents/ARC-SAGE/knowledge/visual-memory/bp35/run_legitimate_chain/run.json'))
    for s in [t for t in trace_in['steps'] if t['step'] <= 143]:
        if s['action'] == 'CLICK':
            env.step(ACT['CLICK'], data={'x': s['x'], 'y': s['y']})
        elif s['action'] == 'CLICK_OOB_SKIPPED':
            continue
        else:
            env.step(ACT[s['action']])
    env.step(ACT['LEFT'])
    for _ in range(5): env.step(ACT['RIGHT'])
    def click(gx, gy):
        sc = game.oztjzzyqoek
        cam_y = sc.camera.rczgvgfsfb[1]
        return env.step(ACT['CLICK'], data={'x': gx*6, 'y': gy*6 - cam_y})
    click(6, 22)
    for _ in range(3): env.step(ACT['LEFT'])
    click(4, 31); click(8, 1)
    for _ in range(3): env.step(ACT['LEFT'])
    assert game.oztjzzyqoek.qswcochjodb == 7
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
            names = {e.name for e in sc.hdnrlfmyrj.jhzcxkveiw(x, y)}
            if 'lrpkmzabbfa' in names: g_list.append((x, y))
            if 'yuuqpmlxorv' in names: one_list.append((x, y))
            if 'oonshderxef' in names: two_list.append((x, y))
    return (p, grav, lvl, tuple(g_list), tuple(one_list), tuple(two_list))


def apply_click(env, gx, gy):
    sc = env._game.oztjzzyqoek
    cam_y = sc.camera.rczgvgfsfb[1]
    return env.step(ACT['CLICK'], data={'x': gx*6, 'y': gy*6 - cam_y})


def enumerate_actions(game):
    """Smart action enum: LEFT, RIGHT, g-click (one, since all equivalent),
    adjacent 1/2, and TOP remote click candidates."""
    sc = game.oztjzzyqoek
    actions = [('LEFT', None), ('RIGHT', None)]
    # Find one g (any) — they all flip gravity
    g_found = None
    for y in range(40):
        for x in range(11):
            if 'lrpkmzabbfa' in {e.name for e in sc.hdnrlfmyrj.jhzcxkveiw(x, y)}:
                g_found = (x, y)
                break
        if g_found: break
    if g_found:
        actions.append(('CLICK', g_found))
    # Adjacent 1/2
    px, py = sc.twdpowducb.qumspquyus
    dy = -1 if sc.vivnprldht else 1
    ax, ay = px, py + dy
    if 0 <= ax < 11 and 0 <= ay < 40:
        names = {e.name for e in sc.hdnrlfmyrj.jhzcxkveiw(ax, ay)}
        if 'yuuqpmlxorv' in names or 'oonshderxef' in names:
            actions.append(('CLICK', (ax, ay)))
    # All remote 1/2 clicks (game-changing)
    seen = set(a[1] for a in actions if a[0] == 'CLICK')
    for y in range(40):
        for x in range(11):
            if (x, y) in seen: continue
            names = {e.name for e in sc.hdnrlfmyrj.jhzcxkveiw(x, y)}
            if 'yuuqpmlxorv' in names or 'oonshderxef' in names:
                actions.append(('CLICK', (x, y)))
    return actions


def bfs(max_time=900, max_states=100000):
    start_env = bootstrap()
    start_sig = signature(start_env._game)
    print(f'L7 start: p={start_sig[0]}, grav={start_sig[1]}, L={start_sig[2]}')

    visited = {start_sig: None}
    parent = {start_sig: (None, None, None)}  # (prev_sig, action, param)
    start_env.deepcopy_cached = True
    queue = deque([(start_env, start_sig)])
    start_t = time.time()
    expansions = 0
    last_log = 0

    while queue:
        env, sig = queue.popleft()
        game = env._game
        expansions += 1
        if time.time() - start_t - last_log > 5:
            last_log = time.time() - start_t
            print(f'  exp={expansions}, queue={len(queue)}, visited={len(visited)}, t={last_log:.1f}s')
        if time.time() - start_t > max_time:
            print('TIMEOUT')
            return None
        if len(visited) > max_states:
            print('CAP')
            return None

        for act, param in enumerate_actions(game):
            try:
                env_new = copy.deepcopy(env)
                if act == 'CLICK':
                    fd = apply_click(env_new, param[0], param[1])
                else:
                    fd = env_new.step(ACT[act])
            except Exception as e:
                continue
            if fd.state.name == 'GAME_OVER':
                continue
            new_sig = signature(env_new._game)
            if new_sig[2] > sig[2]:
                # Level up!
                print(f'LEVEL UP! new L={new_sig[2]}, p={new_sig[0]}')
                parent[new_sig] = (sig, act, param)
                # Reconstruct path
                path = []
                s = new_sig
                while s is not None and parent.get(s, (None, None, None))[0] is not None:
                    prev_s, a, p = parent[s]
                    path.append((a, p))
                    s = prev_s
                path.reverse()
                return path
            if new_sig in visited:
                continue
            visited[new_sig] = None
            parent[new_sig] = (sig, act, param)
            queue.append((env_new, new_sig))
    print('No more states')
    return None


if __name__ == '__main__':
    result = bfs(max_time=600, max_states=50000)
    if result:
        print(f'\nWIN PATH ({len(result)} actions):')
        for i, (a, p) in enumerate(result):
            print(f'  {i+1}. {a} {p if p else ""}')
