#!/usr/bin/env python3
"""dc22 L6 - BFS over macro actions: clicks, walks to itkis/zbhis, teleports.
Each node is a (player_pos, board_config) state. Transitions are macro actions."""
import os, sys, json, time
os.chdir("/home/dp/ai-workspace/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
import numpy as np
from collections import deque
from PIL import Image

VIS = "/home/dp/ai-workspace/shared-context/arc-agi-3/visual-memory/dc22"

def player_reachable_cells(game):
    player = game.fdvakicpimr
    start = (player.x, player.y)
    orig_x, orig_y = player.x, player.y
    parents = {start: None}
    order = [start]
    head = 0
    deltas = [
        (0, -2, GameAction.ACTION1),
        (0,  2, GameAction.ACTION2),
        (-2, 0, GameAction.ACTION3),
        ( 2, 0, GameAction.ACTION4),
    ]
    while head < len(order):
        cx, cy = order[head]
        head += 1
        for dx, dy, act in deltas:
            player.set_position(cx, cy)
            collisions = game.try_move_sprite(player, dx, dy)
            if collisions:
                continue
            nx, ny = player.x, player.y
            if (nx, ny) == (cx, cy):
                continue
            if game.uxwpppoljm(nx, ny, player) is None:
                continue
            if (nx, ny) in parents:
                continue
            parents[(nx, ny)] = (cx, cy, act)
            order.append((nx, ny))
    player.set_position(orig_x, orig_y)
    return parents

def reconstruct_moves(parents, goal):
    moves = []
    cur = goal
    while parents.get(cur) is not None:
        px, py, act = parents[cur]
        moves.append(act)
        cur = (px, py)
    moves.reverse()
    return moves

def replay_to_L6_get_actions():
    cache_path = f"{VIS}/solutions.json"
    with open(cache_path) as f:
        raw = json.load(f)
    am = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
          3: GameAction.ACTION3, 4: GameAction.ACTION4,
          6: GameAction.ACTION6}
    actions = []
    for lvl_idx in range(5):
        for m in raw[lvl_idx]:
            a = am[m['action']]
            actions.append((a, m.get('data', {})))
    return actions

def get_board_key(game):
    """Compact state key for dedup."""
    parts = []
    for s in game.current_level.get_sprites():
        if s.interaction == InteractionMode.REMOVED:
            continue
        for t in s.tags:
            if t in ('wbze', 'jpug', 'itki', 'hhxv'):
                parts.append((s.name, s.x, s.y, s.interaction.value))
                break
    return (game.fdvakicpimr.x, game.fdvakicpimr.y, tuple(sorted(parts)))

def main():
    arcade = Arcade(operation_mode='offline')
    replay_actions = replay_to_L6_get_actions()

    def make_fresh():
        env = arcade.make('dc22-4c9bff3e')
        env.reset()
        for a, d in replay_actions:
            env.step(a, data=d)
        return env

    env = make_fresh()
    game = env._game

    # Get click targets
    cam_h = game.camera._height
    y_offset = (64 - cam_h) // 2
    ct = {}
    for s in game.current_level.get_sprites():
        if 'sys_click' in s.tags:
            cx = s.x + s.width // 2
            cy = s.y + s.height // 2 + y_offset
            ct[s.name] = {'x': cx, 'y': cy}

    # The available click actions from any board state
    click_names = ['jpug-bjuk', 'sprite-6']  # c and f buttons (always visible)
    # Pressure plate buttons are invisible initially. They become visible only
    # when player overlaps the plate. We'll handle those dynamically.

    print(f"Player: ({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
    print(f"Click targets: {list(ct.keys())}")

    # BFS approach: each node is a sequence of L6 actions.
    # We replay from scratch for each node (slow but correct).
    # Each transition: either a click or a walk-to-target.

    # State fingerprints for dedup
    t0 = time.time()
    visited = set()
    init_key = get_board_key(game)
    visited.add(init_key)

    # BFS queue: list of L6 actions
    queue = deque()
    queue.append([])

    best_reach = 0
    nodes = 0
    max_depth = 25  # max L6 action sequence length

    while queue:
        l6_actions = queue.popleft()
        nodes += 1

        if nodes % 100 == 0:
            elapsed = time.time() - t0
            print(f"  nodes={nodes}, queue={len(queue)}, depth={len(l6_actions)}, "
                  f"visited={len(visited)}, best_reach={best_reach}, {elapsed:.1f}s")

        if len(l6_actions) >= max_depth:
            continue

        if time.time() - t0 > 600:  # 10 min timeout
            break

        # Replay to get current state
        env = make_fresh()
        game = env._game
        replay_ok = True
        for action_data in l6_actions:
            if action_data[0] == 'click':
                fd = env.step(GameAction.ACTION6, data=action_data[1])
            elif action_data[0] == 'walk':
                for m in action_data[1]:
                    fd = env.step(m)
                    if fd.state.name == 'LOSE':
                        replay_ok = False
                        break
            if not replay_ok:
                break
            if fd.state.name == 'LOSE':
                replay_ok = False
                break

        if not replay_ok:
            continue

        obs = env.observation_space
        if obs.levels_completed > 5 or obs.state.name == 'WIN':
            print(f"\n*** SOLVED! {len(l6_actions)} macro actions ***")
            print(f"Actions: {l6_actions}")
            return

        # Compute reach
        parents = player_reachable_cells(game)
        if len(parents) > best_reach:
            best_reach = len(parents)
            elapsed = time.time() - t0
            print(f"  New best reach: {best_reach} at depth {len(l6_actions)}, {elapsed:.1f}s")
            ys = sorted(set(y for _, y in parents.keys()))
            for y in ys:
                xs = sorted(x for x, yy in parents.keys() if yy == y)
                print(f"    y={y}: x={xs}")

        # Check for goal
        goal = (game.bqxa.x, game.bqxa.y)
        if goal in parents:
            print(f"\n*** GOAL REACHABLE! Walking to {goal} ***")
            walk = reconstruct_moves(parents, goal)
            full_actions = l6_actions + [('walk', walk)]
            # Verify by replaying
            env2 = make_fresh()
            for action_data in full_actions:
                if action_data[0] == 'click':
                    fd = env2.step(GameAction.ACTION6, data=action_data[1])
                elif action_data[0] == 'walk':
                    for m in action_data[1]:
                        fd = env2.step(m)
            print(f"Final state: {fd.state.name}, levels_completed={env2.observation_space.levels_completed}")
            return

        # Generate transitions: clicks
        for click_name in click_names:
            if click_name in ct:
                new_actions = l6_actions + [('click', ct[click_name])]
                # Quick check: would this click cause a fall?
                # We'll find out when we replay
                queue.append(new_actions)

        # Generate transitions: walk to special cells then click
        # Walk to itki + click c (teleport)
        for s in game.current_level.get_sprites():
            if 'itki' in s.tags and s.interaction != InteractionMode.REMOVED:
                if (s.x, s.y) in parents and (s.x, s.y) != (game.fdvakicpimr.x, game.fdvakicpimr.y):
                    walk = reconstruct_moves(parents, (s.x, s.y))
                    new_actions = l6_actions + [('walk', walk), ('click', ct['jpug-bjuk'])]
                    queue.append(new_actions)

        # Walk to zbhi
        for s in game.current_level.get_sprites():
            if 'zbhi' in s.tags and s.interaction != InteractionMode.REMOVED:
                for dx in range(s.width):
                    for dy in range(s.height):
                        target = (s.x + dx, s.y + dy)
                        if target in parents and target != (game.fdvakicpimr.x, game.fdvakicpimr.y):
                            walk = reconstruct_moves(parents, target)
                            new_actions = l6_actions + [('walk', walk)]
                            queue.append(new_actions)
                            break
                    else:
                        continue
                    break

        # Walk to pressure plate + click direction button
        for s in game.current_level.get_sprites():
            if 'pressure-plate' in s.tags and s.interaction != InteractionMode.REMOVED:
                if (s.x, s.y) in parents and (s.x, s.y) != (game.fdvakicpimr.x, game.fdvakicpimr.y):
                    walk = reconstruct_moves(parents, (s.x, s.y))
                    # Check which buttons become visible on this plate
                    letter = next((t for t in s.tags if len(t) == 1), None)
                    if letter:
                        for btn_name, btn_data in ct.items():
                            for bs in game.current_level.get_sprites():
                                if bs.name == btn_name and letter in bs.tags:
                                    new_actions = l6_actions + [('walk', walk), ('click', btn_data)]
                                    queue.append(new_actions)

    elapsed = time.time() - t0
    print(f"\nBFS complete: {nodes} nodes, {len(visited)} visited, {elapsed:.1f}s")
    print(f"Best reach: {best_reach}")

if __name__ == "__main__":
    main()
