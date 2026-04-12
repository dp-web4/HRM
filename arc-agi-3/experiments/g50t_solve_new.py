#!/usr/bin/env python3
"""g50t full solver: L0-L4 known solutions + DFS for L5-L6.

Multi-phase clone maze puzzle mechanics:
- Player moves on 6px grid
- ACTION5 = PHASE transition: records moves, creates clone, resets player to start
- Clones replay recorded moves simultaneously with each player move
- Modifiers (buttons) toggle obstacles when stepped on
- Win: player reaches goal position (goal.x+1, goal.y+1)
- Timer ticks every 2 steps — runs out = lose
"""
import sys, os, time
from collections import deque
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

UP = GameAction.ACTION1
DOWN = GameAction.ACTION2
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
PHASE = GameAction.ACTION5

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}
DIR_NAMES = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT', PHASE: 'UNDO'}
REVERSE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
DIRS = [UP, DOWN, LEFT, RIGHT]
STEP = 6

KNOWN = {
    0: 'RIGHT RIGHT RIGHT RIGHT UNDO DOWN DOWN DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT RIGHT',
    1: 'LEFT LEFT UNDO DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT UNDO UP UP UP LEFT LEFT LEFT LEFT LEFT LEFT LEFT DOWN DOWN RIGHT RIGHT RIGHT',
    2: 'UP UP RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN RIGHT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT UNDO UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT LEFT LEFT UP UP UP RIGHT RIGHT UP UP',
    3: 'DOWN DOWN RIGHT DOWN UNDO DOWN DOWN RIGHT RIGHT UP UP RIGHT RIGHT DOWN DOWN DOWN UNDO LEFT LEFT LEFT DOWN DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT LEFT LEFT LEFT',
    4: 'UP DOWN DOWN RIGHT RIGHT RIGHT DOWN DOWN DOWN UNDO DOWN RIGHT RIGHT RIGHT UP UP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT DOWN DOWN DOWN UNDO DOWN RIGHT RIGHT RIGHT UP UP RIGHT RIGHT RIGHT DOWN DOWN DOWN DOWN DOWN RIGHT LEFT DOWN LEFT LEFT LEFT LEFT LEFT UP UP',
}


def extract_level(env):
    """Extract level state."""
    game = env._game
    lc = game.vgwycxsxjz
    player = lc.dzxunlkwxt
    goal = lc.whftgckbcu
    arena = lc.afbbgvkpip
    obstacles = lc.uwxkstolmf
    modifiers = lc.hamayflsib
    indicators = lc.drofvwhbxb

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

    mod_to_obs = {}
    for i, mod in enumerate(modifiers):
        pc = mod.nexhtmlmxh
        if pc:
            for j, obs in enumerate(obstacles):
                if obs in pc.ytztewxdin:
                    mod_to_obs[i] = j

    return {
        'player': (player.x, player.y),
        'goal': (goal.x + 1, goal.y + 1),
        'start': (lc.yugzlzepkr, lc.vgpdqizwwm),
        'n_phases': len(indicators),
        'walkable': walkable,
        'obs_cells': obs_cells,
        'obstacles': [(obs.x, obs.y) for obs in obstacles],
        'obs_shifts': [obs.hluvhlvimq() for obs in obstacles],
        'obs_toggle': [obs.dpdubazedr for obs in obstacles],
        'modifiers': [(mod.x, mod.y) for mod in modifiers],
        'mod_to_obs': mod_to_obs,
    }


def bfs_path(start, goal, walkable, max_depth=50):
    """Simple BFS on walkable grid."""
    if start == goal:
        return []
    if goal not in walkable:
        return None

    visited = {start}
    queue = deque([(start, [])])
    deltas = {UP: (0, -STEP), DOWN: (0, STEP), LEFT: (-STEP, 0), RIGHT: (STEP, 0)}

    while queue:
        (px, py), path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for act, (dx, dy) in deltas.items():
            nx, ny = px + dx, py + dy
            if (nx, ny) == goal:
                return path + [act]
            if (nx, ny) in walkable and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [act]))
    return None


def solve_with_modifiers(env, info, max_depth_per_phase=30):
    """Solve a level that may require modifiers.

    Strategy:
    For each phase:
    1. Find which modifiers, when activated, open paths to the goal
    2. Plan a path to the modifier, then from modifier to goal
    3. If multi-phase: record path, transition, plan next phase
    """
    game = env._game
    lc = game.vgwycxsxjz

    goal = info['goal']
    start = info['start']
    n_phases = info['n_phases']
    walkable = info['walkable']
    obs_cells = info['obs_cells']
    modifiers = info['modifiers']
    obstacles = info['obstacles']
    mod_to_obs = info['mod_to_obs']
    obs_toggle = info['obs_toggle']
    obs_shifts = info['obs_shifts']

    # First check: can we reach goal directly?
    player_pos = info['player']
    direct = bfs_path(player_pos, goal, walkable)
    if direct is not None:
        return direct

    # The goal might be behind an obstacle. Find which obstacles block the goal
    # and which modifiers clear those obstacles.
    print(f"  Direct path blocked. Trying modifier combinations...")

    # For each modifier, compute what walkable set looks like after activation
    def compute_walkable_after_mods(active_mods):
        """Compute walkable set after activating given modifiers."""
        new_walkable = set(walkable)
        new_obs = set(obs_cells)

        for mi in active_mods:
            if mi in mod_to_obs:
                oi = mod_to_obs[mi]
                ox, oy = obstacles[oi]
                dx, dy = obs_shifts[oi]

                if obs_toggle[oi]:
                    # Toggle: obstacle moves to shifted position, original becomes walkable
                    new_walkable.add((ox, oy))
                    new_obs.discard((ox, oy))
                    shifted = (ox + dx * STEP, oy + dy * STEP)
                    new_obs.add(shifted)
                    new_walkable.discard(shifted)
                else:
                    # Pad: obstacle moves away, original becomes walkable
                    new_walkable.add((ox, oy))
                    new_obs.discard((ox, oy))

        return new_walkable

    # Try single modifier activations
    for mi, mod_pos in enumerate(modifiers):
        if mod_pos not in walkable:
            continue

        # After stepping on modifier, check if goal is reachable
        new_walkable = compute_walkable_after_mods([mi])
        path_from_mod = bfs_path(mod_pos, goal, new_walkable)

        if path_from_mod is not None:
            # Can reach goal after this modifier
            path_to_mod = bfs_path(player_pos, mod_pos, walkable)
            if path_to_mod is not None:
                print(f"    Mod[{mi}] at {mod_pos} opens path! {len(path_to_mod)} + {len(path_from_mod)} moves")

                # Check if we need a phase transition or can do it in one shot
                if len(path_to_mod) + len(path_from_mod) <= max_depth_per_phase:
                    return path_to_mod + path_from_mod

    # Try two-modifier combinations
    for mi in range(len(modifiers)):
        for mj in range(mi + 1, len(modifiers)):
            if modifiers[mi] not in walkable or modifiers[mj] not in walkable:
                continue

            new_walkable = compute_walkable_after_mods([mi, mj])
            # Check if goal is reachable from either modifier
            for last_mod in [mi, mj]:
                first_mod = mj if last_mod == mi else mi
                first_pos = modifiers[first_mod]
                last_pos = modifiers[last_mod]

                path_to_first = bfs_path(player_pos, first_pos, walkable)
                if path_to_first is None:
                    continue

                # After first mod, walkable changes
                walkable_after_first = compute_walkable_after_mods([first_mod])
                path_first_to_last = bfs_path(first_pos, last_pos, walkable_after_first)
                if path_first_to_last is None:
                    continue

                path_to_goal = bfs_path(last_pos, goal, new_walkable)
                if path_to_goal is not None:
                    total = path_to_first + path_first_to_last + path_to_goal
                    print(f"    Mods[{first_mod},{last_mod}] open path! {len(total)} moves")
                    return total

    # Multi-phase approach: use PHASE transition
    if n_phases > 1:
        print(f"  Trying multi-phase approach ({n_phases} phases)...")

        # Phase 1: go to a modifier to activate it, then PHASE
        for mi, mod_pos in enumerate(modifiers):
            if mod_pos not in walkable:
                continue

            path_to_mod = bfs_path(player_pos, mod_pos, walkable)
            if path_to_mod is None:
                continue

            # Execute phase 1 moves
            for act in path_to_mod:
                env.step(act)

            # Press PHASE
            env.step(PHASE)

            # Now back at start with clone replaying our path
            # Extract new level state (obstacle state changed by clone)
            info2 = extract_level(env)

            # Try to reach goal in phase 2
            path_to_goal = bfs_path(info2['player'], goal, info2['walkable'])
            if path_to_goal is not None:
                print(f"    Phase 1→mod[{mi}], Phase 2→goal: {len(path_to_mod)} + {len(path_to_goal)} moves")
                return path_to_mod + [PHASE] + path_to_goal

            # Need to try more modifiers in phase 2
            for mj, mod_pos2 in enumerate(info2['modifiers']):
                if mj == mi or mod_pos2 not in info2['walkable']:
                    continue
                path_to_mod2 = bfs_path(info2['player'], mod_pos2, info2['walkable'])
                if path_to_mod2 is None:
                    continue

                new_walkable2 = compute_walkable_after_mods([mi, mj])
                path_from_mod2 = bfs_path(mod_pos2, goal, new_walkable2)
                if path_from_mod2 is not None:
                    print(f"    Phase 1→mod[{mi}], Phase 2→mod[{mj}]→goal")
                    # Need to undo phase transition and replay properly
                    # For now, execute directly
                    for act in path_to_mod2:
                        env.step(act)

                    # Check if we need another PHASE
                    if n_phases > 2:
                        env.step(PHASE)
                        info3 = extract_level(env)
                        path_final = bfs_path(info3['player'], goal, info3['walkable'])
                        if path_final:
                            return path_to_mod + [PHASE] + path_to_mod2 + [PHASE] + path_final
                    else:
                        for act in path_from_mod2:
                            env.step(act)
                        return path_to_mod + [PHASE] + path_to_mod2 + path_from_mod2

            # Backtrack phase transition — this is tricky with the engine
            # Reset and replay to undo
            # For now, we can't easily undo PHASE. Report failure.
            print(f"    Phase 2 after mod[{mi}] failed, can't undo PHASE cleanly")
            return None

    print("  All solver strategies exhausted")
    return None


def main():
    print("=" * 60)
    print("g50t Full Solver")
    print("=" * 60)

    arcade = Arcade()
    env = arcade.make('g50t-5849a774')
    fd = env.reset()
    game = env._game

    total_actions = 0
    results = {}

    for level in range(7):
        print(f"\n{'='*50}")
        print(f"Level {level} (engine={game.level_index})")
        print(f"{'='*50}")

        if fd.state.name in ('WON', 'GAME_OVER'):
            print(f"  Game ended: {fd.state.name}")
            break

        if level in KNOWN:
            sol_str = KNOWN[level]
            n = len(sol_str.split())
            print(f"  Using known solution: {n} actions")
            for name in sol_str.split():
                fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
            if game.level_index > level:
                print(f"  L{level} SOLVED!")
                results[level] = n
                total_actions += n
            else:
                print(f"  Known solution FAILED!")
                break
        else:
            info = extract_level(env)
            print(f"  Player: {info['player']}, Goal: {info['goal']}")
            print(f"  Start: {info['start']}, Phases: {info['n_phases']}")
            print(f"  Walkable: {len(info['walkable'])}, Obs: {len(info['obs_cells'])}")
            print(f"  Modifiers: {len(info['modifiers'])}, Obstacles: {len(info['obstacles'])}")

            t0 = time.time()
            solution = solve_with_modifiers(env, info)
            dt = time.time() - t0
            print(f"  Solver time: {dt:.1f}s")

            if solution is None:
                print(f"  FAILED to solve L{level}")
                break

            # Execute solution
            print(f"  Executing {len(solution)} actions...")
            for act in solution:
                fd = env.step(act)
                if game.level_index > level:
                    break

            if game.level_index > level:
                print(f"  L{level} SOLVED!")
                results[level] = len(solution)
                total_actions += len(solution)
            else:
                print(f"  Execution failed")
                lc = game.vgwycxsxjz
                p = lc.dzxunlkwxt
                g = lc.whftgckbcu
                print(f"    Player: ({p.x},{p.y}), Goal: ({g.x+1},{g.y+1})")
                break

    print(f"\n{'='*60}")
    print(f"FINAL: {len(results)}/7 levels solved, {total_actions} total actions")


if __name__ == '__main__':
    main()
