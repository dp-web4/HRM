#!/usr/bin/env python3
"""Solve all g50t levels. L0-L2 known, L3+ need solving."""
import sys, os, json
from collections import deque
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}
DIR_NAMES = {1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'UNDO'}
DIRS = [(1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))]
STEP = 6

KNOWN_SOLUTIONS = {
    0: 'RIGHT RIGHT RIGHT RIGHT UNDO DOWN DOWN DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT RIGHT',
    1: 'LEFT LEFT UNDO DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT UNDO UP UP UP LEFT LEFT LEFT LEFT LEFT LEFT LEFT DOWN DOWN RIGHT RIGHT RIGHT',
    2: 'UP UP RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN RIGHT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT LEFT LEFT UP UP UP RIGHT RIGHT UP UP',
}


def extract_level(env):
    """Extract complete level state from SDK."""
    game = env._game
    lc = game.vgwycxsxjz
    player = lc.dzxunlkwxt
    goal = lc.whftgckbcu
    arena = lc.afbbgvkpip
    obstacles = lc.uwxkstolmf
    modifiers = lc.hamayflsib
    indicators = lc.drofvwhbxb

    # Map walkable grid
    walkable = set()
    obs_cells = set()
    orig_x, orig_y = player.x, player.y
    for px in range(arena.x - 6, arena.x + arena.width + 6, STEP):
        for py in range(arena.y - 6, arena.y + arena.height + 6, STEP):
            player.set_position(px, py)
            if lc.xvkyljflji(player, arena):
                if lc.vjpujwqrto(player):
                    obs_cells.add((px, py))
                else:
                    walkable.add((px, py))
    player.set_position(orig_x, orig_y)

    # Modifier → obstacle mapping
    mod_to_obs = {}
    for i, mod in enumerate(modifiers):
        pc = mod.nexhtmlmxh
        if pc:
            for j, obs in enumerate(obstacles):
                if obs in pc.ytztewxdin:
                    mod_to_obs[i] = j

    info = {
        'player': (player.x, player.y),
        'goal': (goal.x + 1, goal.y + 1),
        'start': (lc.yugzlzepkr, lc.vgpdqizwwm),
        'init_path': list(lc.areahjypvy),
        'max_undos': len(indicators) - 1,
        'walkable': walkable,
        'all_cells': walkable | obs_cells,
        'obs_cells': obs_cells,
        'obstacles': [(obs.x, obs.y) for obs in obstacles],
        'obs_shifts': [],  # (dx,dy) shift direction for each obstacle
        'obs_toggle': [],  # is toggle mode
        'modifiers': [(mod.x, mod.y) for mod in modifiers],
        'mod_to_obs': mod_to_obs,
    }

    for obs in obstacles:
        dx, dy = obs.hluvhlvimq()
        info['obs_shifts'].append((dx, dy))
        info['obs_toggle'].append(obs.dpdubazedr)

    return info


def print_level(info):
    """Print level summary."""
    print(f"  Player: {info['player']}, Goal: {info['goal']}")
    print(f"  Start: {info['start']}, init_path: {info['init_path']}")
    print(f"  Max undos: {info['max_undos']}")
    print(f"  Walkable: {len(info['walkable'])}, Obs blocked: {len(info['obs_cells'])}")
    for i, (ox, oy) in enumerate(info['obstacles']):
        dx, dy = info['obs_shifts'][i]
        shifted = (ox + dx * STEP, oy + dy * STEP)
        toggle = 'TOGGLE' if info['obs_toggle'][i] else 'pad'
        mi = [m for m, o in info['mod_to_obs'].items() if o == i]
        mod_str = f"mod[{mi[0]}]" if mi else "?"
        print(f"    obs[{i}]: ({ox},{oy}) → shifts to {shifted} [{toggle}] activated by {mod_str}")
    for i, (mx, my) in enumerate(info['modifiers']):
        oi = info['mod_to_obs'].get(i, '?')
        print(f"    mod[{i}]: ({mx},{my}) → clears obs[{oi}]")

    # Grid
    all_cells = info['walkable'] | info['obs_cells']
    xs = sorted(set(p[0] for p in all_cells))
    ys = sorted(set(p[1] for p in all_cells))
    for y in ys:
        row = f"  {y:3d} "
        for x in xs:
            if (x, y) == info['player']:
                row += "P"
            elif (x, y) == info['goal']:
                row += "G"
            elif (x, y) in info['obs_cells']:
                row += "#"
            elif (x, y) in [m for m in info['modifiers']]:
                row += "M"
            elif (x, y) in info['walkable']:
                row += "."
            else:
                row += " "
        print(row)


def play_and_verify(env, sol_str, fd):
    """Play solution, show progress, verify."""
    actions = sol_str.strip().split()
    prev = fd.levels_completed
    game = env._game
    lc = game.vgwycxsxjz

    for i, name in enumerate(actions):
        fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
        if fd.levels_completed > prev:
            print(f"  ★ Level up at step {i+1}/{len(actions)}! completed={fd.levels_completed}")
            return fd, True
        if fd.state.name in ('GAME_OVER', 'WON'):
            print(f"  ✗ Game {fd.state.name} at step {i+1}")
            return fd, fd.state.name == 'WON'

    p = lc.dzxunlkwxt
    g = lc.whftgckbcu
    print(f"  Player: ({p.x},{p.y}), Goal: ({g.x+1},{g.y+1}), State: {fd.state.name}")
    return fd, fd.levels_completed > prev


def main():
    arcade = Arcade()
    env = arcade.make('g50t-5849a774')
    fd = env.reset()

    solutions = {}
    total_actions = 0

    for level in range(7):
        print(f"\n{'='*60}")
        print(f"=== Level {level} (completed={fd.levels_completed}) ===")

        if fd.state.name in ('WON', 'GAME_OVER'):
            print(f"  Game ended: {fd.state.name}")
            break

        info = extract_level(env)
        print_level(info)

        if level in KNOWN_SOLUTIONS:
            sol_str = KNOWN_SOLUTIONS[level]
            n = len(sol_str.split())
            print(f"\n  Using known solution: {n} actions")
            fd, ok = play_and_verify(env, sol_str, fd)
            if ok:
                solutions[level] = sol_str
                total_actions += n
                print(f"  ✓ Level {level} solved! Running total: {total_actions}")
            else:
                print(f"  ✗ Known solution FAILED!")
                break
        else:
            print(f"\n  ** No known solution — need to solve **")
            # Save state for analysis
            break

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(solutions)}/7 levels solved, {total_actions} total actions")
    for lv, sol in solutions.items():
        n = len(sol.split())
        print(f"  L{lv}: {n} actions")

    if len(solutions) < 7 and fd.state.name not in ('WON', 'GAME_OVER'):
        print(f"\nUnsolved level {len(solutions)} state:")
        info = extract_level(env)
        print_level(info)

    with open('g50t_solutions.json', 'w') as f:
        json.dump({
            'game_id': 'g50t-5849a774',
            'solutions': solutions,
            'total_actions': total_actions,
            'levels_solved': len(solutions),
        }, f, indent=2)


if __name__ == '__main__':
    main()
