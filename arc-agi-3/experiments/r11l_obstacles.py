#!/usr/bin/env python3
"""
Extract exact obstacle maps per level from r11l source.

Two collision systems:
  bvzgd-* sprites: block ENDPOINT placement (destination check only)
  qtwnv-* sprites: block BALL landing (final position check only)

Ball CAN fly over obstacles. Only landing matters.
Endpoints CAN teleport. Only destination matters.

Usage:
    python r11l_obstacles.py [level_index]  # 0-5, default current level
"""

import numpy as np
import re
import sys
import json
import os

SOURCE = os.path.join(os.path.dirname(__file__),
                      'environment_files/r11l/aa269680/r11l.py')


def parse_sprite_pixels(src, sprite_name):
    """Extract pixel array from a named sprite definition."""
    pattern = f'"{sprite_name}": Sprite('
    idx = src.find(pattern)
    if idx == -1:
        return None
    # Find pixels=[ ... ]
    pix_start = src.index('pixels=[', idx)
    # Find matching closing bracket — count nesting
    depth = 0
    i = pix_start + len('pixels=')
    while i < len(src):
        if src[i] == '[':
            depth += 1
        elif src[i] == ']':
            depth -= 1
            if depth == 0:
                break
        i += 1
    pix_str = src[pix_start + len('pixels='):i + 1]
    try:
        rows = eval(pix_str)  # Safe-ish: only contains ints and lists
        return rows
    except:
        return None


def parse_level_sprites(src, level_index):
    """Extract sprite placements for a level."""
    # Find Level( blocks by counting them
    levels = []
    pos = 0
    while True:
        idx = src.find('Level(\n', pos)
        if idx == -1:
            idx = src.find('Level(sprites', pos)
        if idx == -1:
            break
        levels.append(idx)
        pos = idx + 1

    if level_index >= len(levels):
        return []

    level_start = levels[level_index]
    level_end = levels[level_index + 1] if level_index + 1 < len(levels) else len(src)
    level_block = src[level_start:level_end]

    # Find all sprite placements: sprites["name"].clone().set_position(x, y)
    placements = re.findall(
        r'sprites\["([^"]+)"\]\.clone\(\)\.set_position\((-?\d+),\s*(-?\d+)\)'
        r'(?:\.set_mirror_lr\((True|False)\))?',
        level_block)

    result = []
    for name, x, y, mirror in placements:
        result.append({
            'name': name,
            'x': int(x),
            'y': int(y),
            'mirror_lr': mirror == 'True',
        })
    return result


def build_obstacle_set(src, level_index, sprite_prefix):
    """Build the set of grid positions covered by sprites matching prefix."""
    placements = parse_level_sprites(src, level_index)
    obs = set()

    for p in placements:
        if not p['name'].startswith(sprite_prefix):
            continue
        pixels = parse_sprite_pixels(src, p['name'])
        if pixels is None:
            continue
        x_off = p['x']
        y_off = p['y']

        for row_idx, row in enumerate(pixels):
            for col_idx, val in enumerate(row):
                if val != -1:  # -1 = transparent
                    if p.get('mirror_lr'):
                        c = x_off + (len(row) - 1 - col_idx)
                    else:
                        c = x_off + col_idx
                    r = y_off + row_idx
                    if 0 <= r < 64 and 0 <= c < 64:
                        obs.add((r, c))
    return obs


def check_endpoint_dest(bvzgd_set, cx, cy, margin=3):
    """Can an endpoint (7x7 cross) be placed at click (cx, cy)?"""
    for dr in range(-margin, margin + 1):
        for dc in range(-margin, margin + 1):
            if (int(cy) + dr, int(cx) + dc) in bvzgd_set:
                return False
    return True


def check_ball_landing(qtwnv_set, cx, cy, margin=2):
    """Can a ball (5x5 diamond) land at click (cx, cy)?"""
    for dr in range(-margin, margin + 1):
        for dc in range(-margin, margin + 1):
            if (int(cy) + dr, int(cx) + dc) in qtwnv_set:
                return False
    return True


def main():
    with open(SOURCE) as f:
        src = f.read()

    # Get current level from session
    level = 0
    try:
        with open('/tmp/sage_solver/session.json') as f:
            s = json.load(f)
        level = s.get('levels_completed', 0)
    except:
        pass

    if len(sys.argv) > 1:
        level = int(sys.argv[1])

    print(f"=== r11l Level {level} Obstacle Map ===")

    bvzgd = build_obstacle_set(src, level, 'bvzgd')
    qtwnv = build_obstacle_set(src, level, 'qtwnv')

    print(f"bvzgd (endpoint blocking): {len(bvzgd)} pixels")
    print(f"qtwnv (ball landing blocking): {len(qtwnv)} pixels")

    # Show valid endpoint destinations
    print(f"\nValid ENDPOINT destinations (margin=3):")
    for ry in range(6, 58, 4):
        valid = [cx for cx in range(6, 58, 4)
                 if check_endpoint_dest(bvzgd, cx, ry)]
        if valid:
            print(f"  row {ry}: {valid}")

    # Show valid ball landing positions
    print(f"\nValid BALL landing positions (margin=2):")
    for ry in range(6, 58, 4):
        valid = [cx for cx in range(6, 58, 4)
                 if check_ball_landing(qtwnv, cx, ry)]
        if len(valid) < 13:  # only show constrained rows
            print(f"  row {ry}: {valid}")
        else:
            print(f"  row {ry}: ALL CLEAR ({len(valid)} positions)")


if __name__ == '__main__':
    main()
