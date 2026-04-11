#!/usr/bin/env python3
"""Explore s5i5: arrow rotation/resizing puzzle."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

arcade = Arcade()
env = arcade.make('s5i5-a48e4b1d')
fd = env.reset()
game = env._game

for lv in range(min(3, len(game._levels))):
    level = game.current_level
    step_limit = level.get_data("StepCounter")
    children = level.get_data("Children")
    print(f"\n=== Level {lv} (steps={step_limit}) ===")

    # Categorize sprites by tag
    tracks = level.get_sprites_by_tag("gdgcpukdrl")
    arrows = level.get_sprites_by_tag("agujdcrunq")
    targets = level.get_sprites_by_tag("zylvdxoiuq")
    goals = level.get_sprites_by_tag("cpdhnkdobh")
    rotate_btns = level.get_sprites_by_tag("myzmclysbl")

    print(f"  Tracks: {len(tracks)}, Arrows: {len(arrows)}, Targets: {len(targets)}")
    print(f"  Goals: {len(goals)}, Rotate buttons: {len(rotate_btns)}")
    if children:
        print(f"  Children: {children}")

    for s in tracks:
        colors = set(int(c) for c in np.unique(s.pixels) if c >= 0)
        print(f"  Track: pos=({s.x},{s.y}) size={s.width}x{s.height} colors={colors}")
    for s in arrows:
        color = int(s.pixels[1, 1])
        orient = game.fhkoulsvoi(s)
        size_px = max(s.width, s.height)
        print(f"  Arrow: pos=({s.x},{s.y}) size={s.width}x{s.height} color={color} orient={orient}° children={len(game.enplxxgoja.get(s, set()))}")
    for s in targets:
        print(f"  Target: pos=({s.x},{s.y}) size={s.width}x{s.height}")
    for s in goals:
        print(f"  Goal marker: pos=({s.x},{s.y}) size={s.width}x{s.height}")
    for s in rotate_btns:
        color = int(s.pixels[s.height//2, s.height//2])
        print(f"  Rotate btn: pos=({s.x},{s.y}) size={s.width}x{s.height} color={color}")

    # Check win
    print(f"  Win check: {game.vodebmynqs()}")
    for g in goals:
        at_target = any(t.x == g.x and t.y == g.y for t in targets)
        print(f"    goal ({g.x},{g.y}) at target: {at_target}")

    # Camera info
    cam = game.camera
    print(f"  Camera: {cam.width}x{cam.height}")

    # Don't advance past L0 for now
    break
