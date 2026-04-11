#!/usr/bin/env python3
"""Explore m0r0 levels: understand maze layout, cursor positions, mechanics."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

arcade = Arcade()
env = arcade.make('m0r0-dadda488')
fd = env.reset()
game = env._game

for lv_idx in range(6):
    game.set_level(lv_idx)
    game.on_set_level(game.current_level)
    level = game.current_level
    grid_w, grid_h = level.grid_size

    print(f"\n=== Level {lv_idx} (grid {grid_w}x{grid_h}) ===")

    # Find maze sprite
    maze_sprites = level.get_sprites_by_tag("jggua")
    for ms in maze_sprites:
        rendered = ms.render()
        print(f"  Maze: {ms.name} at ({ms.x},{ms.y}) rot={ms.rotation} size={rendered.shape}")

        # Build walkability map
        walkable = set()
        for y in range(rendered.shape[0]):
            for x in range(rendered.shape[1]):
                if rendered[y, x] < 0:  # -1 = walkable
                    world_x, world_y = ms.x + x, ms.y + y
                    walkable.add((world_x, world_y))

        print(f"  Walkable cells: {len(walkable)}")

        # Print maze visually
        for y in range(grid_h):
            row = ''
            for x in range(grid_w):
                if (x, y) in walkable:
                    row += '.'
                else:
                    row += '#'
            print(f"    {row}")

    # Find cursors
    cursor_names = ['qzfkx-ubwff-idtiq', 'qzfkx-ubwff-crkfz',
                    'qzfkx-kncqr-idtiq', 'qzfkx-kncqr-crkfz']
    cursors = []
    for cn in cursor_names:
        sprites = level.get_sprites_by_name(cn)
        for s in sprites:
            mirror = 'SAME' if 'idtiq' in cn else 'MIRROR_X' if 'ubwff' in cn else ('MIRROR_Y' if 'kncqr-idtiq' in cn else 'MIRROR_XY')
            print(f"  Cursor: {cn} at ({s.x},{s.y}) → {mirror}")
            cursors.append((cn, s.x, s.y, mirror))

    # Find movable blocks
    blocks = level.get_sprites_by_name("cvcer")
    for b in blocks:
        print(f"  Block: cvcer at ({b.x},{b.y})")

    # Find hazards
    hazards = level.get_sprites_by_name("wyiex")
    if hazards:
        print(f"  Hazards: {len(hazards)} wyiex sprites")
        for h in hazards[:5]:
            print(f"    wyiex at ({h.x},{h.y})")
        if len(hazards) > 5:
            print(f"    ... and {len(hazards)-5} more")

    # Find toggle walls
    for color in ['raixb', 'ujcze', 'qeazm']:
        keys = level.get_sprites_by_name(f"hnutp-{color}")
        walls = level.get_sprites_by_name(f"dfnuk-{color}")
        if keys or walls:
            print(f"  Toggle {color}: {len(keys)} keys, {len(walls)} walls")
            for k in keys:
                print(f"    Key: ({k.x},{k.y})")
            for w in walls:
                print(f"    Wall: ({w.x},{w.y}) rot={w.rotation}")

    print(f"  npwxa: {level.get_data('npwxa')}")
