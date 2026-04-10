#!/usr/bin/env python3
"""
ls20 General Solver — Extracts level data from SDK and solves via BFS.

Handles: walls, arrows/conveyors, modifiers (shape/color/rotation), refills, goals.
Handles mobile modifiers that walk along indicator tracks.
Uses the game's actual collision model (5px grid, mrznumynfe box check, txnfzvzetn interaction).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from arc_agi import Arcade
from arcengine import GameAction
from collections import deque
from typing import Dict, List, Set, Tuple, Optional

NAME_TO_GA = {'UP': GameAction.ACTION1, 'DOWN': GameAction.ACTION2,
              'LEFT': GameAction.ACTION3, 'RIGHT': GameAction.ACTION4}
DIRS = {'UP': (0, -5), 'DOWN': (0, 5), 'LEFT': (-5, 0), 'RIGHT': (5, 0)}


def compute_modifier_trajectory(mob, max_steps=200):
    """Pre-compute a mobile modifier's position trajectory."""
    init_x, init_y = mob._sprite.x, mob._sprite.y
    init_dir = mob._dir

    positions = [(init_x, init_y)]
    for _ in range(max_steps):
        mob.step()
        positions.append((mob._sprite.x, mob._sprite.y))

    # Find cycle
    period = None
    for p in range(1, max_steps):
        if positions[p] == positions[0]:
            # Check that the next few match too
            if all(positions[p + i] == positions[i] for i in range(min(p, max_steps - p))):
                period = p
                break

    # Reset
    mob._sprite.set_position(init_x, init_y)
    mob._dir = init_dir

    return positions[:period] if period else positions, period


def extract_level_data(game) -> dict:
    """Extract all relevant data from the current game level."""
    level = game.current_level
    player = game.gudziatsk
    step_x = game.gisrhqpee
    step_y = game.tbwnoxqgc

    data = {
        'start': (player.x, player.y),
        'step_x': step_x, 'step_y': step_y,
        'step_budget': game._step_counter_ui.osgviligwp,
        'dec': game._step_counter_ui.efipnixsvl,
        'lives': game.aqygnziho,
        'shape': game.fwckfzsyc,
        'color': game.hiaauhahz,
        'rot': game.cklxociuu,
        'goal_shape': game.ldxlnycps,
        'goal_color': game.yjdexjsoa,
        'goal_rot': game.ehwheiwsk,
        'goal_positions': [(s.x, s.y) for s in game.plrpelhym],
        'n_shapes': len(game.ijessuuig),
        'n_colors': len(game.tnkekoeuk),
        'n_rots': 4,
    }

    # Walls
    walls = set()
    for s in level.get_sprites_by_tag('ihdgageizm'):
        walls.add((s.x, s.y))
    data['walls'] = walls

    # Goals
    goals_set = set()
    for s in level.get_sprites_by_tag('rjlbuycveu'):
        goals_set.add((s.x, s.y))
    data['goals_set'] = goals_set

    # Refills
    data['refills'] = [(s.x, s.y) for s in level.get_sprites_by_tag('npxgalaybz')]

    # Static modifiers (not in wsoslqeku)
    mobile_mod_sprites = set()
    for mob in game.wsoslqeku:
        mobile_mod_sprites.add(id(mob._sprite))

    data['static_shape_mods'] = []
    data['static_color_mods'] = []
    data['static_rot_mods'] = []
    for s in level.get_sprites_by_tag('ttfwljgohq'):
        if id(s) not in mobile_mod_sprites:
            data['static_shape_mods'].append((s.x, s.y))
    for s in level.get_sprites_by_tag('soyhouuebz'):
        if id(s) not in mobile_mod_sprites:
            data['static_color_mods'].append((s.x, s.y))
    for s in level.get_sprites_by_tag('rhsxkxzdjz'):
        if id(s) not in mobile_mod_sprites:
            data['static_rot_mods'].append((s.x, s.y))

    # Mobile modifiers — pre-compute trajectories
    # Important: modifiers step FIRST, then interaction is checked at their NEW position
    mobile_mods = []
    for mob in game.wsoslqeku:
        tags = list(mob._sprite.tags) if mob._sprite.tags else []
        mod_type = None
        if 'ttfwljgohq' in tags:
            mod_type = 'shape'
        elif 'soyhouuebz' in tags:
            mod_type = 'color'
        elif 'rhsxkxzdjz' in tags:
            mod_type = 'rotation'

        trajectory, period = compute_modifier_trajectory(mob)
        mobile_mods.append({
            'type': mod_type,
            'trajectory': trajectory,
            'period': period,
        })
    data['mobile_mods'] = mobile_mods

    # Arrows/Conveyors
    walls_and_goals = walls | goals_set
    conveyor_map = {}
    for h in game.hasivfwip:
        sx, sy = h.sprite.x, h.sprite.y
        w, ht = h.width, h.height
        dx, dy = h.dx, h.dy

        # Calculate push distance
        wall_cx, wall_cy = sx + dx, sy + dy
        dist = 0
        for i in range(1, 12):
            nx = wall_cx + dx * w * i
            ny = wall_cy + dy * ht * i
            if (nx, ny) in walls_and_goals:
                dist = max(0, i - 1)
                break

        # Find trigger grid positions
        grid_x_base = data['start'][0] % step_x
        grid_y_base = data['start'][1] % step_y
        for gx in range(grid_x_base, 60, step_x):
            for gy in range(grid_y_base, 60, step_y):
                if gx < sx + w and gx + step_x > sx and gy < sy + ht and gy + step_y > sy:
                    is_walled = any(gx <= wx < gx + step_x and gy <= wy < gy + step_y for wx, wy in walls)
                    if not is_walled:
                        dest_x = gx + dx * w * dist
                        dest_y = gy + dy * ht * dist
                        conveyor_map[(gx, gy)] = (dest_x, dest_y)

    data['conveyor_map'] = conveyor_map

    return data


def build_grid(data) -> Set[Tuple[int, int]]:
    sx, sy = data['start']
    step_x, step_y = data['step_x'], data['step_y']
    return set((x, y) for x in range(sx % step_x, 60, step_x)
                       for y in range(sy % step_y, 60, step_y))


def is_wall(x, y, walls, step_x, step_y):
    return any(x <= wx < x + step_x and y <= wy < y + step_y for wx, wy in walls)


def get_mobile_mod_positions(data, time):
    """Get positions of all mobile modifiers at given time step."""
    positions = {}
    for i, mob in enumerate(data['mobile_mods']):
        traj = mob['trajectory']
        period = mob['period']
        if period:
            idx = (time + 1) % period  # +1 because mods step BEFORE interaction check
        else:
            idx = min(time + 1, len(traj) - 1)
        positions[i] = traj[idx]
    return positions


def check_interactions(x, y, data, shape, color, rot, ref_mask, time):
    """Process interactions at position. Returns (new_shape, new_color, new_rot, new_ref_mask, refilled)."""
    step_x, step_y = data['step_x'], data['step_y']
    new_shape, new_color, new_rot = shape, color, rot
    new_ref = ref_mask
    refilled = False

    # Static modifiers
    for mx, my in data['static_shape_mods']:
        if x <= mx < x + step_x and y <= my < y + step_y:
            new_shape = (new_shape + 1) % data['n_shapes']
    for mx, my in data['static_color_mods']:
        if x <= mx < x + step_x and y <= my < y + step_y:
            new_color = (new_color + 1) % data['n_colors']
    for mx, my in data['static_rot_mods']:
        if x <= mx < x + step_x and y <= my < y + step_y:
            new_rot = (new_rot + 1) % data['n_rots']

    # Mobile modifiers (at their position at this time)
    mob_positions = get_mobile_mod_positions(data, time)
    for i, mob in enumerate(data['mobile_mods']):
        mx, my = mob_positions[i]
        if x <= mx < x + step_x and y <= my < y + step_y:
            if mob['type'] == 'shape':
                new_shape = (new_shape + 1) % data['n_shapes']
            elif mob['type'] == 'color':
                new_color = (new_color + 1) % data['n_colors']
            elif mob['type'] == 'rotation':
                new_rot = (new_rot + 1) % data['n_rots']

    # Refills
    for i, (rx, ry) in enumerate(data['refills']):
        if not (ref_mask & (1 << i)):
            if x <= rx < x + step_x and y <= ry < y + step_y:
                new_ref = ref_mask | (1 << i)
                refilled = True

    return new_shape, new_color, new_rot, new_ref, refilled


def solve_level(data, max_states=15_000_000) -> Optional[List[str]]:
    """BFS solver for a level. Uses parent pointers to save memory."""
    grid = build_grid(data)
    walls = data['walls']
    step_x, step_y = data['step_x'], data['step_y']
    step_budget = data['step_budget']
    dec = data['dec']
    n_goals = len(data['goal_positions'])

    start = data['start']
    has_mobile = len(data['mobile_mods']) > 0

    # Pre-compute wall check set for faster lookup
    wall_set = set()
    for gx, gy in grid:
        if any(gx <= wx < gx + step_x and gy <= wy < gy + step_y for wx, wy in walls):
            wall_set.add((gx, gy))

    # Pre-compute shortest distances from each goal to all grid cells (BFS flood fill)
    # This enables pruning states that can't possibly reach remaining goals
    goal_dists = []
    walkable = grid - wall_set
    for gx, gy in data['goal_positions']:
        dist = {(gx, gy): 0}
        q = deque([(gx, gy)])
        while q:
            px, py = q.popleft()
            for _, (ddx, ddy) in DIRS.items():
                npx, npy = px + ddx, py + ddy
                if (npx, npy) in walkable and (npx, npy) not in dist:
                    dist[(npx, npy)] = dist[(px, py)] + 1
                    q.append((npx, npy))
        goal_dists.append(dist)

    # Determine time modulus
    if has_mobile:
        periods = [m['period'] for m in data['mobile_mods'] if m['period']]
        if periods:
            from math import lcm
            time_mod = periods[0]
            for p in periods[1:]:
                time_mod = lcm(time_mod, p)
        else:
            time_mod = 200
    else:
        time_mod = 1

    # State: (x, y, shape, color, rot, ref_mask, collected, steps_left, time_mod)
    init_state = (start[0], start[1], data['shape'], data['color'], data['rot'],
                  0, 0, step_budget, 0)

    # BFS with parent pointers (state -> (parent_state, action_name))
    parent = {init_state: None}
    queue = deque([init_state])
    explored = 0

    while queue:
        state = queue.popleft()
        explored += 1
        if explored % 500000 == 0:
            print(f'    BFS: {explored:,} states explored, queue={len(queue):,}')
        if explored > max_states:
            print(f'    BFS: hit state limit ({max_states:,})')
            return None

        x, y, shape, color, rot, ref_mask, collected, steps, time = state

        for dname, (ddx, ddy) in DIRS.items():
            nx, ny = x + ddx, y + ddy

            if (nx, ny) not in grid or (nx, ny) in wall_set:
                continue

            # Goal blocking check (skip already-collected goals)
            goal_blocked = False
            for i, (gx, gy) in enumerate(data['goal_positions']):
                if collected & (1 << i):
                    continue  # already collected, sprite removed
                if nx <= gx < nx + step_x and ny <= gy < ny + step_y:
                    if not (shape == data['goal_shape'][i] and
                           color == data['goal_color'][i] and
                           rot == data['goal_rot'][i]):
                        goal_blocked = True
                        break
            if goal_blocked:
                continue

            new_shape, new_color, new_rot, new_ref, refilled = check_interactions(
                nx, ny, data, shape, color, rot, ref_mask, time)

            new_steps = step_budget if refilled else steps - dec

            final_x, final_y = nx, ny
            if (nx, ny) in data['conveyor_map']:
                final_x, final_y = data['conveyor_map'][(nx, ny)]
                new_shape, new_color, new_rot, new_ref, refilled2 = check_interactions(
                    final_x, final_y, data, new_shape, new_color, new_rot, new_ref, time)
                if refilled2:
                    new_steps = step_budget

            new_collected = collected
            for i, (gx, gy) in enumerate(data['goal_positions']):
                if not (collected & (1 << i)):
                    if final_x == gx and final_y == gy:
                        if (new_shape == data['goal_shape'][i] and
                            new_color == data['goal_color'][i] and
                            new_rot == data['goal_rot'][i]):
                            new_collected = collected | (1 << i)

            if new_collected == (1 << n_goals) - 1:
                # Reconstruct path
                path = [dname]
                s = state
                while parent[s] is not None:
                    ps, act = parent[s]
                    path.append(act)
                    s = ps
                path.reverse()
                return path

            if new_steps < 0:
                continue

            # Distance pruning: can we reach all remaining goals with steps left?
            # Only prune if no refills remain (conservative)
            remaining_refills = sum(1 for ri in range(len(data['refills']))
                                    if not (new_ref & (1 << ri)))
            if remaining_refills == 0 and new_collected != (1 << n_goals) - 1:
                max_moves = new_steps // dec
                reachable = True
                for gi in range(n_goals):
                    if not (new_collected & (1 << gi)):
                        gd = goal_dists[gi].get((final_x, final_y), 999)
                        if gd > max_moves:
                            reachable = False
                            break
                if not reachable:
                    continue

            new_time = (time + 1) % time_mod
            new_state = (final_x, final_y, new_shape, new_color, new_rot,
                        new_ref, new_collected, new_steps, new_time)
            if new_state not in parent:
                parent[new_state] = (state, dname)
                queue.append(new_state)

    print(f'    BFS: exhausted all {explored:,} reachable states')
    return None


def main():
    arcade = Arcade()
    env = arcade.make('ls20-9607627b')
    fd = env.reset()

    solutions = {
        1: 'LEFT LEFT LEFT UP UP UP UP RIGHT RIGHT RIGHT UP UP UP'.split(),
        2: 'UP RIGHT UP UP UP UP UP RIGHT RIGHT DOWN RIGHT DOWN DOWN DOWN DOWN DOWN DOWN UP DOWN DOWN LEFT LEFT RIGHT UP RIGHT UP UP UP UP UP UP UP LEFT LEFT LEFT LEFT LEFT LEFT DOWN LEFT DOWN DOWN DOWN DOWN DOWN'.split(),
        3: 'UP UP UP UP UP UP UP UP LEFT DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN UP UP UP LEFT LEFT UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT UP UP UP LEFT UP DOWN UP RIGHT DOWN'.split(),
    }

    for level_num in range(1, 8):
        if level_num in solutions:
            print(f'\n=== Level {level_num} (known: {len(solutions[level_num])} moves) ===')
            for name in solutions[level_num]:
                fd = env.step(NAME_TO_GA[name])
            print(f'  levels={fd.levels_completed}')
            continue

        print(f'\n=== Level {level_num} (solving...) ===')
        game = env._game
        data = extract_level_data(game)

        print(f'  Start: {data["start"]}, Config: s={data["shape"]} c={data["color"]} r={data["rot"]}')
        print(f'  Goals: {data["goal_positions"]} (s={data["goal_shape"]} c={data["goal_color"]} r={data["goal_rot"]})')
        print(f'  Steps: {data["step_budget"]}, Dec: {data["dec"]}')
        print(f'  Walls: {len(data["walls"])}, Conveyors: {len(data["conveyor_map"])}')
        print(f'  Static mods: shape={data["static_shape_mods"]} color={data["static_color_mods"]} rot={data["static_rot_mods"]}')
        print(f'  Mobile mods: {len(data["mobile_mods"])}')
        for m in data['mobile_mods']:
            print(f'    {m["type"]}: period={m["period"]}, first 5 pos={m["trajectory"][:5]}')
        print(f'  Refills: {data["refills"]}')

        solution = solve_level(data)

        if solution:
            print(f'  SOLUTION: {len(solution)} moves')
            print(f'  {" ".join(solution)}')
            solutions[level_num] = solution

            for name in solution:
                fd = env.step(NAME_TO_GA[name])
            print(f'  Executed! levels={fd.levels_completed}, state={fd.state.name}')

            if fd.levels_completed < level_num:
                print(f'  WARNING: Solution failed!')
                break
        else:
            print(f'  NO SOLUTION FOUND')
            break

    print(f'\nFinal: levels={fd.levels_completed}/{fd.win_levels}, state={fd.state.name}')
    if fd.state.name == 'WON':
        print('\n*** GAME WON! ***')

    print('\n=== All Solutions ===')
    total = 0
    for l in sorted(solutions.keys()):
        n = len(solutions[l])
        total += n
        print(f'L{l} ({n}): {" ".join(solutions[l])}')
    print(f'Total: {total} actions')


if __name__ == '__main__':
    main()
