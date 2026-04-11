#!/usr/bin/env python3
"""Solve sk48: chain/snake color-matching puzzle.

BFS using game engine simulation. State = (head positions, chain lengths, target positions).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
from collections import deque

arcade = Arcade()
env = arcade.make('sk48-41055498')
fd = env.reset()
game = env._game

ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
ACTION_NAMES = {GameAction.ACTION1: 'UP', GameAction.ACTION2: 'DOWN',
                GameAction.ACTION3: 'LEFT', GameAction.ACTION4: 'RIGHT'}


def state_key():
    """Compact state: head positions + chain lengths + target positions."""
    parts = []
    for head, segs in game.mwfajkguqx.items():
        parts.append((head.x, head.y, len(segs)))
    for t in game.vbelzuaian:
        parts.append((t.x, t.y))
    return tuple(sorted(parts))


def save_state():
    """Save full game state for BFS."""
    data = {}
    # Save head positions
    data['heads'] = []
    for head in game.mwfajkguqx:
        data['heads'].append((head, head.x, head.y))
    # Save segments: positions + pixels
    data['segs'] = {}
    for head, segs in game.mwfajkguqx.items():
        data['segs'][id(head)] = [(s, s.x, s.y, s.pixels.copy(), s.rotation) for s in segs]
    # Save targets
    data['targets'] = [(t, t.x, t.y) for t in game.vbelzuaian]
    # Budget
    data['budget'] = game.qiercdohl
    # Selected head
    data['selected'] = game.vzvypfsnt
    # History length
    data['history_len'] = len(game.seghobzez)
    return data


def restore_state(data):
    """Restore game state."""
    for head, x, y in data['heads']:
        head.set_position(x, y)
    for head in game.mwfajkguqx:
        segs_data = data['segs'][id(head)]
        # Remove extra segments
        current_segs = game.mwfajkguqx[head]
        while len(current_segs) > len(segs_data):
            removed = current_segs.pop()
            game.current_level.remove_sprite(removed)
        # Add missing segments
        while len(current_segs) < len(segs_data):
            from arcengine import Sprite
            new_seg = segs_data[len(current_segs)][0]
            if new_seg not in [s for s in game.current_level.get_sprites()]:
                game.current_level.add_sprite(new_seg)
            current_segs.append(new_seg)
        # Restore positions
        for i, (s, x, y, px, rot) in enumerate(segs_data):
            s.set_position(x, y)
            s.pixels = px.copy()
            s.set_rotation(rot)
            game.mwfajkguqx[head][i] = s
    for t, x, y in data['targets']:
        t.set_position(x, y)
    game.qiercdohl = data['budget']
    game.vzvypfsnt = data['selected']
    # Trim history
    while len(game.seghobzez) > data['history_len']:
        game.seghobzez.pop()


total_actions = 0

for lv in range(len(game._levels)):
    level = game.current_level
    print(f"\n=== Level {lv} ===")

    # Show state
    for head, segs in game.mwfajkguqx.items():
        mapped = head in game.xpmcmtbcv
        print(f"  Head({head.x},{head.y}) rot={head.rotation}° segs={len(segs)} mapped={mapped}")
    targets = game.vbelzuaian
    board_targets = [(t.x, t.y, int(t.pixels[1,1])) for t in targets if t.y < 53]
    print(f"  Board targets: {board_targets}")
    print(f"  Budget: {game.qiercdohl}")

    # Check reference pattern
    for head, ref in game.xpmcmtbcv.items():
        ref_matches = game.vjfbwggsd.get(ref, [])
        ref_colors = [int(m.pixels[1,1]) for m in ref_matches]
        print(f"  Reference pattern: {ref_colors}")

    # BFS
    init_save = save_state()
    init_key = state_key()

    if game.gvtmoopqgy():
        print("  Already solved!")
        lv_actions = 0
    else:
        queue = deque([(init_key, [])])
        visited = {init_key}
        solution = None
        max_states = 200_000
        explored = 0

        while queue and len(visited) < max_states:
            skey, moves = queue.popleft()
            if len(moves) >= game.qiercdohl - 35:  # need budget for win animation
                continue
            explored += 1
            if explored % 5000 == 0:
                print(f"    BFS: {explored} expanded, {len(visited)} visited, depth={len(moves)}")

            # Try each action
            for action in ACTIONS:
                # Restore to initial state, then replay moves + new action
                restore_state(init_save)
                valid = True
                for m in moves:
                    old_budget = game.qiercdohl
                    fd_tmp = env.step(m)
                    # Need an extra step for animation/push resolution
                    while game.ljprkjlji or game.pzzwlsmdt:
                        fd_tmp = env.step(m)
                    if game.qiercdohl == old_budget:
                        valid = False  # action had no effect
                        break

                if not valid:
                    continue

                old_budget = game.qiercdohl
                fd_tmp = env.step(action)
                while game.ljprkjlji or game.pzzwlsmdt:
                    fd_tmp = env.step(action)

                if game.qiercdohl == old_budget:
                    continue  # action had no effect

                new_key = state_key()
                if new_key in visited:
                    continue
                visited.add(new_key)

                new_moves = moves + [action]
                if game.gvtmoopqgy():
                    solution = new_moves
                    break

                queue.append((new_key, new_moves))

            if solution:
                break

        restore_state(init_save)

        if solution:
            print(f"  BFS: {len(solution)} moves ({explored} expanded, {len(visited)} visited)")
            sol_names = [ACTION_NAMES[a] for a in solution]
            print(f"  Solution: {sol_names}")

            # Execute
            lv_actions = 0
            for action in solution:
                fd = env.step(action)
                lv_actions += 1
                # Wait for animations
                while game.ljprkjlji or game.pzzwlsmdt:
                    fd = env.step(action)
                    lv_actions += 1
        else:
            print(f"  No solution found! ({explored} expanded, {len(visited)} visited)")
            break

    total_actions += lv_actions
    print(f"  {lv_actions} actions, completed={fd.levels_completed}, state={fd.state.name}")

    if fd.state.name == 'WIN':
        print(f"\n=== ALL SOLVED! Total: {total_actions} ===")
        break

    if fd.levels_completed <= lv:
        print(f"  Level not completed!")
        break

print(f"\nFinal: completed={fd.levels_completed}, state={fd.state.name}, total={total_actions}")
