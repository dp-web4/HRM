#!/usr/bin/env python3
"""Solve lp85: cyclic rotation puzzle.

Buttons rotate all pieces in a group's cycle path. Groups can share grid positions.
State = positions of goal tiles. BFS to find optimal button sequence.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'environment_files', 'lp85', '305b61c3'))
import lp85 as lp85_mod

CELL = 3  # pixels per grid cell

arcade = Arcade()
env = arcade.make('lp85-305b61c3')
fd = env.reset()
game = env._game

def grid_to_display(gx, gy):
    cam_w = game.camera.width
    cam_h = game.camera.height
    scale = 64 // cam_w
    y_offset = (64 - cam_h * scale) // 2
    return gx * scale, gy * scale + y_offset

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
    # For each group: build grid_pos -> cycle_number and cycle_number -> grid_pos
    group_info = {}
    for gname in group_names:
        grid = groups_data[gname]
        pos_to_num = {}  # (gx, gy) -> cycle_number
        num_to_pos = {}  # cycle_number -> (gx, gy)
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
    print(f"  Groups: {[(g, group_info[g]['cycle_len']) for g in group_names]}")

    # Find goal markers and their required positions
    circle_goals = level.get_sprites_by_tag("bghvgbtwcb")
    square_goals = level.get_sprites_by_tag("fdgmtkfrxl")

    required = []  # (grid_x, grid_y, tag_to_match)
    for g in circle_goals:
        required.append(((g.x + 1) // CELL, (g.y + 1) // CELL, "goal"))
    for g in square_goals:
        required.append(((g.x + 1) // CELL, (g.y + 1) // CELL, "goal-o"))

    # Find goal/goal-o tiles and their current positions
    goal_tiles = []
    for s in level.get_sprites():
        tags = s.tags if s.tags else []
        for t in tags:
            if t in ("goal", "goal-o"):
                goal_tiles.append({'sprite': s, 'grid': (s.x // CELL, s.y // CELL), 'tag': t})

    print(f"  Goal tiles: {[(g['tag'], g['grid']) for g in goal_tiles]}")
    print(f"  Required: {required}")

    # State = tuple of goal tile grid positions (matching order of goal_tiles list)
    # Win = for each required position+tag, some goal tile with that tag is at that position

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

    def apply_action(state, gname, direction):
        """Apply rotation to state. direction: 1=R, -1=L."""
        info = group_info[gname]
        pos_to_num = info['pos_to_num']
        num_to_pos = info['num_to_pos']
        clen = info['cycle_len']

        new_state = list(state)
        for i, pos in enumerate(state):
            if pos in pos_to_num:
                n = pos_to_num[pos]
                if direction == 1:  # R: n → n+1
                    new_n = 1 if n == clen else n + 1
                else:  # L: n → n-1
                    new_n = clen if n == 1 else n - 1
                new_state[i] = num_to_pos[new_n]
        return tuple(new_state)

    if is_win(initial_state):
        print(f"  Already solved!")
        lv_actions = 0
    else:
        # BFS
        queue = deque([(initial_state, [])])
        visited = {initial_state}
        solution = None

        while queue:
            state, moves = queue.popleft()
            for gname in group_names:
                for delta in [-1, 1]:
                    new_state = apply_action(state, gname, delta)
                    if new_state not in visited:
                        new_moves = moves + [(gname, delta)]
                        if is_win(new_state):
                            solution = new_moves
                            break
                        visited.add(new_state)
                        queue.append((new_state, new_moves))
                if solution:
                    break
            if solution:
                break

        if not solution:
            print(f"    ERROR: BFS found no solution! Visited {len(visited)} states")
            break

        print(f"  BFS: {len(solution)} moves (visited {len(visited)} states)")

        # Compress consecutive same-button presses
        compressed = []
        for gname, delta in solution:
            if compressed and compressed[-1][0] == gname and compressed[-1][1] == delta:
                compressed[-1] = (gname, delta, compressed[-1][2] + 1)
            else:
                compressed.append((gname, delta, 1))

        # Execute
        lv_actions = 0
        for gname, delta, count in compressed:
            direction = 'R' if delta == 1 else 'L'
            tag = f"button_{gname}_{direction}"
            btns = [s for s in level.get_sprites_by_tag("sys_click")
                   if any(tag in t for t in (s.tags or []))]
            if not btns:
                print(f"    ERROR: no button for {tag}")
                break
            btn = btns[0]
            gcx = btn.x + btn.width // 2
            gcy = btn.y + btn.height // 2
            dx, dy = grid_to_display(gcx, gcy)
            print(f"    {gname} {direction} x{count}")
            for _ in range(count):
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
