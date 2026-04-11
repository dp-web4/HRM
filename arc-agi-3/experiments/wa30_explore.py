#!/usr/bin/env python3
"""Explore wa30 game — understand levels then solve with greedy/BFS."""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
from collections import deque

CELL = 4
UP, DOWN, LEFT, RIGHT, ACT5 = (
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
    GameAction.ACTION4, GameAction.ACTION5
)
ACTIONS = [UP, DOWN, LEFT, RIGHT, ACT5]
ACTION_NAMES = {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R', ACT5: '5'}

arcade = Arcade()
env = arcade.make('wa30-ee6fef47')
fd = env.reset()
game = env._game


def show_level():
    """Show level layout as grid."""
    player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    reds = game.current_level.get_sprites_by_tag("kdweefinfi")
    whites = game.current_level.get_sprites_by_tag("ysysltqlke")

    # Show wyzquhjerd positions (slots where goals need to go)
    slot_cells = set()
    for (x, y) in game.wyzquhjerd:
        slot_cells.add((x // CELL, y // CELL))

    # Show lqctaojiby (secondary slots)
    slot2_cells = set()
    for (x, y) in game.lqctaojiby:
        slot2_cells.add((x // CELL, y // CELL))

    # Walls
    wall_cells = set()
    for (x, y) in game.pkbufziase:
        if 0 <= x < 64 and 0 <= y < 64:
            wall_cells.add((x // CELL, y // CELL))

    print(f"  Player: ({player.x},{player.y}) = cell ({player.x//CELL},{player.y//CELL})")
    print(f"  Goals: {[(g.x, g.y, g.x//CELL, g.y//CELL) for g in goals]}")
    print(f"  Reds: {[(r.x, r.y, r.x//CELL, r.y//CELL) for r in reds]}")
    print(f"  Whites: {[(w.x, w.y, w.x//CELL, w.y//CELL) for w in whites]}")
    print(f"  Slot cells (fsjj): {sorted(slot_cells)}")
    print(f"  Slot2 cells (zqxw): {sorted(slot2_cells)}")
    print(f"  Steps: {game.kuncbnslnm.current_steps}")
    print(f"  Carrying: {len(game.nsevyuople)}")
    print(f"  Win: {game.ymzfopzgbq()}")

    # Grid visualization (16x16 cells)
    grid = [['.' for _ in range(16)] for _ in range(16)]
    for (cx, cy) in wall_cells:
        if 0 <= cx < 16 and 0 <= cy < 16:
            grid[cy][cx] = '#'
    for (cx, cy) in slot_cells:
        if 0 <= cx < 16 and 0 <= cy < 16:
            grid[cy][cx] = 'O'
    for (cx, cy) in slot2_cells:
        if 0 <= cx < 16 and 0 <= cy < 16:
            grid[cy][cx] = 'o'
    for g in goals:
        cx, cy = g.x // CELL, g.y // CELL
        if 0 <= cx < 16 and 0 <= cy < 16:
            grid[cy][cx] = 'G'
    for r in reds:
        cx, cy = r.x // CELL, r.y // CELL
        if 0 <= cx < 16 and 0 <= cy < 16:
            grid[cy][cx] = 'R'
    for w in whites:
        cx, cy = w.x // CELL, w.y // CELL
        if 0 <= cx < 16 and 0 <= cy < 16:
            grid[cy][cx] = 'W'
    pcx, pcy = player.x // CELL, player.y // CELL
    if 0 <= pcx < 16 and 0 <= pcy < 16:
        grid[pcy][pcx] = 'P'

    for row in grid:
        print('  ' + ''.join(row))


def state_key():
    player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    reds = game.current_level.get_sprites_by_tag("kdweefinfi")
    whites = game.current_level.get_sprites_by_tag("ysysltqlke")
    carrying = tuple(sorted((k.name, v.name) for k, v in game.nsevyuople.items()))
    return (player.x, player.y,
            tuple(sorted((g.x, g.y) for g in goals)),
            tuple(sorted((r.x, r.y) for r in reds)),
            tuple(sorted((w.x, w.y) for w in whites)),
            carrying)


def replay_to_level(lv):
    """Reset and advance to given level."""
    env.reset()
    for _ in range(lv):
        game.next_level()


def solve_bfs(lv, max_depth=50, max_states=50000):
    """BFS solver using replay from level start."""
    replay_to_level(lv)
    t0 = time.time()
    initial_sk = state_key()
    visited = {initial_sk}
    queue = deque([[]])
    total_expanded = 0

    while queue and total_expanded < max_states:
        moves = queue.popleft()
        depth = len(moves)
        if depth >= max_depth:
            continue

        # Replay to state
        replay_to_level(lv)
        for m in moves:
            env.step(m)

        # Check win
        if game.ymzfopzgbq():
            elapsed = time.time() - t0
            print(f"  SOLVED! {len(moves)} moves, {total_expanded} expanded, {elapsed:.1f}s")
            return moves

        if game.kuncbnslnm.current_steps <= 0:
            continue

        # Get current state to compare
        before_sk = state_key()

        for action in ACTIONS:
            # Replay and do action
            replay_to_level(lv)
            for m in moves:
                env.step(m)
            env.step(action)
            total_expanded += 1

            sk = state_key()
            if sk == before_sk or sk in visited:
                continue

            visited.add(sk)
            new_moves = moves + [action]

            if game.ymzfopzgbq():
                elapsed = time.time() - t0
                print(f"  SOLVED! {len(new_moves)} moves, {total_expanded} expanded, {elapsed:.1f}s")
                return new_moves

            queue.append(new_moves)

        if total_expanded % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    d={depth}, visited={len(visited)}, expanded={total_expanded}, {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  No solution. {total_expanded} expanded, {len(visited)} visited, {elapsed:.1f}s")
    return None


# Explore all levels
total_actions = 0
for lv in range(20):
    replay_to_level(lv)
    if fd.state.name == 'WIN':
        break

    print(f"\n=== Level {lv} ===")
    show_level()

    solution = solve_bfs(lv, max_depth=30, max_states=100_000)
    if solution is None:
        print(f"  FAILED on level {lv}")
        break

    # Execute solution
    replay_to_level(lv)
    for m in solution:
        fd = env.step(m)

    # Win - advance
    if game.ymzfopzgbq():
        game.next_level()
        fd = env._make_frame_data()

    sol_names = [ACTION_NAMES[a] for a in solution]
    print(f"  Solution: {''.join(sol_names)}")
    total_actions += len(solution)
    print(f"  completed={fd.levels_completed}, state={fd.state.name}")

print(f"\nFinal: completed={fd.levels_completed}, state={fd.state.name}, total={total_actions}")
