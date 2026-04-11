#!/usr/bin/env python3
"""Explore sk48: chain/snake color-matching puzzle."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

arcade = Arcade()
env = arcade.make('sk48-41055498')
fd = env.reset()
game = env._game

CELL = 6  # movement unit

for lv in range(min(3, len(game._levels))):
    level = game.current_level
    step_limit = level.get_data("StepCounter")
    print(f"\n=== Level {lv} (steps={step_limit}) ===")

    # Camera info
    cam = game.camera
    print(f"  Camera: {cam.width}x{cam.height}")

    # Chain heads
    print(f"\n  Chain heads ({len(game.mwfajkguqx)}):")
    for head, segments in game.mwfajkguqx.items():
        orient = head.pixels.shape  # approximate orient from pixels
        tags = head.tags if head.tags else []
        print(f"    Head at ({head.x},{head.y}) size={head.width}x{head.height} tags={tags}")
        print(f"      Segments: {len(segments)}")
        for s in segments:
            color = int(s.pixels[1, 1]) if s.height >= 2 and s.width >= 2 else -1
            print(f"        seg ({s.x},{s.y}) {s.width}x{s.height} color={color}")

    # Targets
    targets = level.get_sprites_by_tag("elmjchdqcn")
    print(f"\n  Targets ({len(targets)}):")
    for t in targets:
        color = int(t.pixels[1, 1]) if t.height >= 2 and t.width >= 2 else -1
        print(f"    ({t.x},{t.y}) {t.width}x{t.height} color={color}")

    # Target-to-head mapping
    print(f"\n  Target mapping:")
    for head, target_pos in game.xpmcmtbcv.items():
        print(f"    head({head.x},{head.y}) → target({target_pos.x},{target_pos.y})")

    # Match state
    print(f"\n  Match state:")
    for head, matches in game.vjfbwggsd.items():
        print(f"    head({head.x},{head.y}): {len(matches)} matches")

    # Walls
    walls = level.get_sprites_by_tag("irkeobngyh")
    obstacles = level.get_sprites_by_tag("mkgqjopcjn")
    print(f"\n  Walls: {len(walls)}, Obstacles: {len(obstacles)}")
    for w in walls[:5]:
        print(f"    wall ({w.x},{w.y}) {w.width}x{w.height}")

    # Budget
    print(f"\n  Budget: {game.qiercdohl}/{game.vhzjwcpmk}")

    # Win check
    print(f"  Win: {game.gvtmoopqgy()}")

    # Currently selected head
    if game.vzvypfsnt:
        print(f"  Selected: ({game.vzvypfsnt.x},{game.vzvypfsnt.y})")

    # Try to understand the grid layout
    print(f"\n  All sprites by tag:")
    all_sprites = level.get_sprites()
    tag_counts = {}
    for s in all_sprites:
        for t in (s.tags or []):
            tag_counts[t] = tag_counts.get(t, 0) + 1
    for tag, count in sorted(tag_counts.items()):
        print(f"    {tag}: {count}")

    # Don't advance
    break
