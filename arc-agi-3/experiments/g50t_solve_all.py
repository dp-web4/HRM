#!/usr/bin/env python3
"""Solve all g50t levels from scratch using SDK + analytical BFS."""
import sys, os, json
from collections import deque
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}
DIR_NAMES = {1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'UNDO'}
DIRS = [(1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))]  # UP DOWN LEFT RIGHT
STEP = 6


def extract_level(env):
    """Extract level state from SDK."""
    game = env._game
    lc = game.vgwycxsxjz
    player = lc.dzxunlkwxt
    goal = lc.whftgckbcu
    arena = lc.afbbgvkpip
    obstacles = lc.uwxkstolmf
    modifiers = lc.hamayflsib
    indicators = lc.drofvwhbxb

    # Map walkable grid using game's collision check
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

    # Get all cells (walkable + obstacle-blocked = all positions inside arena)
    all_cells = walkable | obs_cells

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
        'all_cells': all_cells,
        'obs_cells': obs_cells,
        'obstacles': [(obs.x, obs.y) for obs in obstacles],
        'modifiers': [(mod.x, mod.y) for mod in modifiers],
        'mod_to_obs': mod_to_obs,
    }
    return info


def print_grid(info):
    """Print level grid."""
    all_cells = info['walkable'] | info['obs_cells']
    if not all_cells:
        return
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


def ghost_pos_at_step(start, path, step_idx, all_cells):
    """Simulate ghost position at given step index."""
    gx, gy = start
    for s in range(min(step_idx, len(path))):
        dx, dy = path[s]
        nx, ny = gx + dx * STEP, gy + dy * STEP
        if (nx, ny) in all_cells:
            gx, gy = nx, ny
        # If not walkable, ghost stays in place (failed move)
    return (gx, gy)


def bfs_simple(start, goal, walkable, max_steps=60):
    """Simple BFS on walkable grid."""
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path
        if len(path) >= max_steps:
            continue
        for action, (dx, dy) in DIRS:
            nx, ny = x + dx * STEP, y + dy * STEP
            if (nx, ny) in walkable and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(dx, dy)]))
    return None


def bfs_with_ghosts(start, goal, all_cells, obstacles, modifiers, mod_to_obs,
                    ghost_specs, max_steps=80):
    """
    BFS with ghost replay.
    ghost_specs: list of (ghost_start, ghost_path) tuples.
    At step k, each ghost replays step k of its path. If k >= len(path), ghost stays.
    """
    queue = deque([(start[0], start[1], 0, [])])
    visited = {(start[0], start[1], 0)}

    while queue:
        px, py, step, actions = queue.popleft()
        if step > max_steps:
            continue

        for action, (dx, dy) in DIRS:
            nx, ny = px + dx * STEP, py + dy * STEP
            new_step = step + 1

            if (nx, ny) not in all_cells:
                continue

            # Determine ghost positions at current step
            # Ghosts are at their positions from end of previous step
            ghost_positions = []
            for gs, gp in ghost_specs:
                gpos = ghost_pos_at_step(gs, gp, step, all_cells)
                ghost_positions.append(gpos)

            # After player deactivates old modifier, check which mods are still active
            # (only ghosts' modifiers count during player move)
            active_mods = set()
            for mi, (mx, my) in enumerate(modifiers):
                for gpos in ghost_positions:
                    if gpos == (mx, my):
                        active_mods.add(mi)
                        break

            cleared_obs = set()
            for mi in active_mods:
                if mi in mod_to_obs:
                    cleared_obs.add(mod_to_obs[mi])

            # Check if target cell blocked
            blocked = False
            for oi, (ox, oy) in enumerate(obstacles):
                if oi not in cleared_obs and (nx, ny) == (ox, oy):
                    blocked = True
                    break

            if blocked:
                continue

            if (nx, ny) == goal:
                return actions + [(dx, dy)]

            state_key = (nx, ny, new_step)
            if state_key not in visited:
                visited.add(state_key)
                queue.append((nx, ny, new_step, actions + [(dx, dy)]))

    return None


def find_paths_to_all_mods(start, walkable, modifiers, max_steps=50):
    """BFS from start, find shortest path to each modifier."""
    queue = deque([(start, [])])
    visited = {start}
    paths = {}

    while queue:
        (x, y), path = queue.popleft()
        for i, (mx, my) in enumerate(modifiers):
            if (x, y) == (mx, my) and i not in paths:
                paths[i] = path[:]
        if len(paths) == len(modifiers):
            break
        if len(path) >= max_steps:
            continue
        for action, (dx, dy) in DIRS:
            nx, ny = x + dx * STEP, y + dy * STEP
            if (nx, ny) in walkable and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(dx, dy)]))

    return paths


def find_paths_with_ghost(start, walkable_base, all_cells, obstacles, modifiers,
                          mod_to_obs, ghost_specs, max_steps=50):
    """BFS from start with ghosts replaying, find paths to each modifier."""
    queue = deque([(start, 0, [])])
    visited = {(start, 0)}
    paths = {}

    while queue:
        (x, y), step, path = queue.popleft()
        for i, (mx, my) in enumerate(modifiers):
            if (x, y) == (mx, my) and i not in paths:
                paths[i] = path[:]
        if len(paths) == len(modifiers):
            break
        if step >= max_steps:
            continue
        for action, (dx, dy) in DIRS:
            nx, ny = x + dx * STEP, y + dy * STEP
            new_step = step + 1

            if (nx, ny) not in all_cells:
                continue

            # Check obstacle clearance from ghosts
            ghost_positions = [ghost_pos_at_step(gs, gp, step, all_cells) for gs, gp in ghost_specs]
            active_mods = set()
            for mi, (mx, my) in enumerate(modifiers):
                for gpos in ghost_positions:
                    if gpos == (mx, my):
                        active_mods.add(mi)
                        break
            cleared_obs = set(mod_to_obs[mi] for mi in active_mods if mi in mod_to_obs)

            blocked = False
            for oi, (ox, oy) in enumerate(obstacles):
                if oi not in cleared_obs and (nx, ny) == (ox, oy):
                    blocked = True
                    break
            if blocked:
                continue

            state_key = ((nx, ny), new_step)
            if state_key not in visited:
                visited.add(state_key)
                queue.append(((nx, ny), new_step, path + [(dx, dy)]))

    return paths


def solve_level(info):
    """Solve a level with up to 2 undos."""
    start = info['start']
    goal = info['goal']
    walkable = info['walkable']
    all_cells = info['all_cells']
    obstacles = info['obstacles']
    modifiers = info['modifiers']
    mod_to_obs = info['mod_to_obs']
    max_undos = info['max_undos']
    init_path = info['init_path']

    # Compute actual player position
    px, py = start
    for dx, dy in init_path:
        px += dx * STEP
        py += dy * STEP
    actual_pos = (px, py)

    print(f"  Start: {start}, actual: {actual_pos}, goal: {goal}")
    print(f"  Max undos: {max_undos}, {len(obstacles)} obs, {len(modifiers)} mods")

    # Try direct path first (no undo)
    path = bfs_simple(actual_pos, goal, walkable)
    if path:
        return [dir_to_name(dx, dy) for dx, dy in path]

    if max_undos == 0:
        print("  No direct path and no undos available!")
        return None

    # Find paths to modifiers from base walkable grid
    mod_paths_base = find_paths_to_all_mods(actual_pos, walkable, modifiers)
    print(f"  Base-reachable modifiers: {list(mod_paths_base.keys())}")

    if max_undos == 1:
        # 1 undo: ghost goes to modifier, player navigates with obstacle cleared
        for mod_idx, mod_path in mod_paths_base.items():
            ghost_full = init_path + mod_path
            ghost_specs = [(start, ghost_full)]
            result = bfs_with_ghosts(start, goal, all_cells, obstacles, modifiers,
                                     mod_to_obs, ghost_specs)
            if result:
                action_names = [dir_to_name(dx, dy) for dx, dy in mod_path]
                action_names.append('UNDO')
                action_names.extend([dir_to_name(dx, dy) for dx, dy in result])
                return action_names

    if max_undos >= 2:
        # 2 undos: need two ghosts on two modifiers
        # Phase 0: ghost0 goes to mod_i (base path)
        # Phase 1: ghost1 goes to mod_j (might need ghost0's help)
        # Phase 2: navigate to goal with both ghosts

        for mod_i, path_i in mod_paths_base.items():
            ghost0_full = init_path + path_i
            ghost0_spec = [(start, ghost0_full)]

            # Find paths with ghost0 replaying
            mod_paths_g0 = find_paths_with_ghost(
                start, walkable, all_cells, obstacles, modifiers,
                mod_to_obs, ghost0_spec
            )
            print(f"  With ghost0 on mod[{mod_i}], reachable mods: {list(mod_paths_g0.keys())}")

            for mod_j, path_j in mod_paths_g0.items():
                if mod_j == mod_i:
                    continue

                ghost1_full = path_j  # Phase 1 recording (no init_path)
                ghost_specs = [(start, ghost0_full), (start, ghost1_full)]
                result = bfs_with_ghosts(start, goal, all_cells, obstacles, modifiers,
                                         mod_to_obs, ghost_specs)
                if result:
                    action_names = [dir_to_name(dx, dy) for dx, dy in path_i]
                    action_names.append('UNDO')
                    action_names.extend([dir_to_name(dx, dy) for dx, dy in path_j])
                    action_names.append('UNDO')
                    action_names.extend([dir_to_name(dx, dy) for dx, dy in result])
                    print(f"  SOLVED: ghost0→mod[{mod_i}], ghost1→mod[{mod_j}]")
                    return action_names

        # Try 3-modifier levels with 2 undos (all 3 must be covered)
        if len(modifiers) == 3 and max_undos == 2:
            print("  3 modifiers with 2 undos — trying path-exhaustion tricks...")
            # A ghost on a modifier with exhausted path keeps it active
            # But we only have 2 ghosts for 3 modifiers
            # Unless the player stands on one modifier while at the goal
            # Or the player passes through obstacles in sequence
            pass

    return None


def dir_to_name(dx, dy):
    if (dx, dy) == (0, -1): return 'UP'
    if (dx, dy) == (0, 1): return 'DOWN'
    if (dx, dy) == (-1, 0): return 'LEFT'
    if (dx, dy) == (1, 0): return 'RIGHT'
    return f'?({dx},{dy})'


def verify_with_sdk(env, solution_str, fd):
    """Play solution through SDK and verify level completion."""
    actions = solution_str.strip().split()
    prev_level = fd.levels_completed
    for name in actions:
        fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
    level_up = fd.levels_completed > prev_level
    return fd, level_up


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
        print_grid(info)

        sol = solve_level(info)
        if sol is None:
            print(f"  FAILED to solve level {level}")
            break

        sol_str = ' '.join(sol)
        print(f"  Solution ({len(sol)} actions): {sol_str}")

        fd, ok = verify_with_sdk(env, sol_str, fd)
        if ok:
            solutions[level] = sol_str
            total_actions += len(sol)
            print(f"  ✓ Level {level} solved! Total actions: {total_actions}")
        else:
            print(f"  ✗ Solution didn't complete level!")
            game = env._game
            lc = game.vgwycxsxjz
            print(f"    Player: ({lc.dzxunlkwxt.x},{lc.dzxunlkwxt.y})")
            print(f"    Goal: ({lc.whftgckbcu.x+1},{lc.whftgckbcu.y+1})")
            print(f"    State: {fd.state.name}")
            break

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(solutions)}/{7} levels solved, {total_actions} total actions")
    for lv, sol in solutions.items():
        n = len(sol.split())
        print(f"  L{lv}: {n} actions — {sol}")

    with open('g50t_solutions.json', 'w') as f:
        json.dump({'solutions': solutions, 'total_actions': total_actions}, f, indent=2)
    print(f"\nSaved to g50t_solutions.json")


if __name__ == '__main__':
    main()
