#!/usr/bin/env python3
"""Solve lp85: cyclic rotation puzzle.

Buttons rotate all pieces in a group's cycle path. Multiple buttons at same position
trigger compound rotations. State = goal tile positions. BFS finds optimal clicks.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'environment_files', 'lp85', '305b61c3'))
import lp85 as lp85_mod

CELL = 3

arcade = Arcade()
env = arcade.make('lp85-305b61c3')
fd = env.reset()
game = env._game

def grid_to_display(gx, gy):
    cam_w = game.camera.width
    cam_h = game.camera.height
    scale = 64 // cam_w
    x_offset = (64 - cam_w * scale) // 2
    y_offset = (64 - cam_h * scale) // 2
    return gx * scale + x_offset, gy * scale + y_offset

total_actions = 0

for lv in range(8):
    level = game.current_level
    level_name = level.get_data("level_name")
    step_limit = level.get_data("StepCounter")
    print(f"\n=== Level {lv} (name={level_name}, steps={step_limit}) ===")

    if level_name not in lp85_mod.izutyjcpih:
        print(f"  ERROR: no rotation data for {level_name}")
        break

    groups_data = lp85_mod.izutyjcpih[level_name]
    group_names = sorted(groups_data.keys())

    # Parse rotation maps
    group_info = {}
    for gname in group_names:
        grid = groups_data[gname]
        pos_to_num = {}
        num_to_pos = {}
        for gy, row in enumerate(grid):
            for gx, val in enumerate(row):
                if val != -1:
                    pos_to_num[(gx, gy)] = val
                    num_to_pos[val] = (gx, gy)
        cycle_len = max(num_to_pos.keys())
        group_info[gname] = {
            'pos_to_num': pos_to_num,
            'num_to_pos': num_to_pos,
            'cycle_len': cycle_len
        }

    # Find goal tiles and required positions
    circle_goals = level.get_sprites_by_tag("bghvgbtwcb")
    square_goals = level.get_sprites_by_tag("fdgmtkfrxl")

    required = []
    for g in circle_goals:
        required.append(((g.x + 1) // CELL, (g.y + 1) // CELL, "goal"))
    for g in square_goals:
        required.append(((g.x + 1) // CELL, (g.y + 1) // CELL, "goal-o"))

    goal_tiles = []
    for s in level.get_sprites():
        tags = s.tags if s.tags else []
        for t in tags:
            if t in ("goal", "goal-o"):
                goal_tiles.append({'sprite': s, 'grid': (s.x // CELL, s.y // CELL), 'tag': t})

    print(f"  Goal tiles: {[(g['tag'], g['grid']) for g in goal_tiles]}")
    print(f"  Required: {required}")

    n_goals = len(goal_tiles)
    initial_state = tuple(g['grid'] for g in goal_tiles)
    goal_tags = [g['tag'] for g in goal_tiles]

    req_set = set()
    for gx, gy, tag in required:
        req_set.add(((gx, gy), tag))

    def is_win(state):
        satisfied = set()
        for i, pos in enumerate(state):
            tag = goal_tags[i]
            key = (pos, tag)
            if key in req_set:
                satisfied.add(key)
        return satisfied == req_set

    def apply_single_rotation(state, gname, direction):
        info = group_info[gname]
        pos_to_num = info['pos_to_num']
        num_to_pos = info['num_to_pos']
        clen = info['cycle_len']
        new_state = list(state)
        for i, pos in enumerate(state):
            if pos in pos_to_num:
                n = pos_to_num[pos]
                if direction == 1:
                    new_n = 1 if n == clen else n + 1
                else:
                    new_n = clen if n == 1 else n - 1
                new_state[i] = num_to_pos[new_n]
        return tuple(new_state)

    # Build click actions: group buttons by position
    # Each unique position triggers all button rotations at that position
    all_sprites = level.get_sprites()
    position_buttons = {}  # (px, py) -> list of (gname, delta)
    button_at_pos = {}     # (px, py) -> first button sprite (for clicking)

    for s in all_sprites:
        if not s.tags:
            continue
        for t in s.tags:
            if t.startswith("button_"):
                parts = t.split("_")
                if len(parts) == 3:
                    gname, dir_char = parts[1], parts[2]
                    delta = 1 if dir_char == 'R' else -1
                    key = (s.x, s.y)
                    if key not in position_buttons:
                        position_buttons[key] = []
                        button_at_pos[key] = s
                    position_buttons[key].append((gname, delta))

    # Build click actions for BFS
    # Each click = compound action (multiple rotations)
    click_actions = []  # list of (position, rotations_list)
    for pos, rotations in position_buttons.items():
        click_actions.append((pos, rotations))

    # Also provide individual rotations for positions with single buttons
    # (for BFS efficiency, single-rotation clicks are already covered)

    print(f"  Click positions: {len(click_actions)}")
    for pos, rots in click_actions:
        if len(rots) > 1:
            print(f"    {pos}: {rots}")

    def apply_click(state, rotations):
        """Apply compound rotation (multiple groups simultaneously)."""
        # In the game, each button at the position is processed sequentially.
        # But for goal tiles, the effect is the compound of all rotations.
        for gname, delta in rotations:
            state = apply_single_rotation(state, gname, delta)
        return state

    if is_win(initial_state):
        print(f"  Already solved!")
        lv_actions = 0
    else:
        # BFS over states using click actions
        queue = deque([(initial_state, [])])
        visited = {initial_state}
        solution = None
        max_states = 500000

        while queue and len(visited) < max_states:
            state, moves = queue.popleft()
            for ci, (pos, rotations) in enumerate(click_actions):
                new_state = apply_click(state, rotations)
                if new_state not in visited:
                    new_moves = moves + [ci]
                    if is_win(new_state):
                        solution = new_moves
                        break
                    visited.add(new_state)
                    queue.append((new_state, new_moves))
            if solution:
                break

        if not solution:
            print(f"    ERROR: BFS found no solution! Visited {len(visited)} states")
            break

        print(f"  BFS: {len(solution)} clicks (visited {len(visited)} states)")

        # Execute
        lv_actions = 0
        for ci in solution:
            pos, rotations = click_actions[ci]
            btn = button_at_pos[pos]
            gcx = btn.x + btn.width // 2
            gcy = btn.y + btn.height // 2
            dx, dy = grid_to_display(gcx, gcy)
            fd = env.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
            lv_actions += 1

    total_actions += lv_actions
    print(f"  {lv_actions} actions, completed={fd.levels_completed}, state={fd.state.name}")

    if fd.state.name == 'WIN':
        print(f"\n=== ALL SOLVED! Total: {total_actions} ===")
        break

    if fd.levels_completed <= lv:
        print(f"  Level not completed! Debugging...")
        for gx, gy, tag in required:
            at = level.get_sprite_at(gx * CELL, gy * CELL, tag)
            print(f"    {tag} at grid ({gx},{gy}) pixel ({gx*CELL},{gy*CELL}): {at is not None}")
        for gt in goal_tiles:
            print(f"    tile {gt['tag']}: now at ({gt['sprite'].x},{gt['sprite'].y}) grid ({gt['sprite'].x//CELL},{gt['sprite'].y//CELL})")
        break

print(f"\nFinal: completed={fd.levels_completed}, state={fd.state.name}, total={total_actions}")
