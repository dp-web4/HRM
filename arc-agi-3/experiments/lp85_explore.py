#!/usr/bin/env python3
"""Explore lp85: cyclic rotation puzzle."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

arcade = Arcade()
env = arcade.make('lp85-305b61c3')
fd = env.reset()
game = env._game

for lv in range(min(3, len(game._levels))):
    level = game.current_level
    level_name = level.get_data("level_name")
    step_counter = level.get_data("StepCounter")
    print(f"\n=== Level {lv} (name={level_name}, steps={step_counter}) ===")

    # Find buttons
    buttons = level.get_sprites_by_tag("sys_click")
    button_info = []
    for b in buttons:
        for tag in b.tags:
            if tag.startswith("button_"):
                parts = tag.split("_")
                if len(parts) == 3:
                    group, direction = parts[1], parts[2]
                    button_info.append((group, direction, b.x, b.y, b.name))
    print(f"  Buttons: {[(g, d, x, y) for g, d, x, y, n in button_info]}")

    # Find goals
    circle_goals = level.get_sprites_by_tag("bghvgbtwcb")
    square_goals = level.get_sprites_by_tag("fdgmtkfrxl")
    print(f"  Circle goals (bghvgbtwcb): {[(g.x, g.y, g.name) for g in circle_goals]}")
    print(f"  Square goals (fdgmtkfrxl): {[(g.x, g.y, g.name) for g in square_goals]}")

    # Find all sprites with their tags
    all_sprites = level.get_sprites()
    pieces = []
    for s in all_sprites:
        tags = s.tags if s.tags else []
        is_button = any("button" in t for t in tags)
        is_goal = any(t in ["bghvgbtwcb", "fdgmtkfrxl", "goal", "goal-o"] for t in tags)
        if not is_button and not is_goal:
            pieces.append(s)

    print(f"  All sprites ({len(all_sprites)}):")
    for s in all_sprites:
        tags = s.tags if s.tags else []
        colors = {int(c) for c in np.unique(s.pixels) if c >= 0}
        print(f"    {s.name}: pos=({s.x},{s.y}) size={s.width}x{s.height} tags={tags} colors={colors}")

    # Check win condition
    print(f"\n  Win check:")
    for g in circle_goals:
        at_pos = level.get_sprite_at(g.x + 1, g.y + 1, "goal")
        print(f"    circle goal at ({g.x},{g.y}): goal sprite at ({g.x+1},{g.y+1}) = {at_pos is not None}")
    for g in square_goals:
        at_pos = level.get_sprite_at(g.x + 1, g.y + 1, "goal-o")
        print(f"    square goal at ({g.x},{g.y}): goal-o sprite at ({g.x+1},{g.y+1}) = {at_pos is not None}")

    # Look at the rotation map for this level
    print(f"\n  Rotation groups (from izutyjcpih):")
    from environment_files.lp85 import lp85 as lp85_mod
    if level_name in lp85_mod.izutyjcpih:
        for group_name, grid in lp85_mod.izutyjcpih[level_name].items():
            positions = {}
            for y, row in enumerate(grid):
                for x, val in enumerate(row):
                    if val != -1:
                        positions[val] = (x, y)
            max_pos = max(positions.keys())
            print(f"    Group {group_name}: {max_pos} positions")
            # Show the path
            path = [positions[i] for i in range(1, max_pos + 1)]
            print(f"      path: {path}")

    # Skip to next level if not last
    if lv < 2:
        # Need to solve this level or skip somehow
        # For now just break
        break
