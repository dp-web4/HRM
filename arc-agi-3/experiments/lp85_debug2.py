#!/usr/bin/env python3
"""Debug lp85 Level 1 click mechanics."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

arcade = Arcade()
env = arcade.make('lp85-305b61c3')
fd = env.reset()
game = env._game

# Solve L0 first (5 L presses)
level = game.current_level
btns = [s for s in level.get_sprites_by_tag("sys_click") if any("button_A_L" in t for t in s.tags)]
btn = btns[0]
cam_w, cam_h = game.camera.width, game.camera.height
scale = 64 // cam_w
y_off = (64 - cam_h * scale) // 2
print(f"L0: cam={cam_w}x{cam_h} scale={scale} y_off={y_off}")
dx = (btn.x + btn.width // 2) * scale
dy = (btn.y + btn.height // 2) * scale + y_off
for _ in range(5):
    fd = env.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
print(f"L0: completed={fd.levels_completed}")

# Now Level 1
level = game.current_level
cam_w, cam_h = game.camera.width, game.camera.height
grid = level.grid_size
print(f"\nL1: cam={cam_w}x{cam_h} grid={grid}")

# Find buttons
for s in level.get_sprites_by_tag("sys_click"):
    tags = [t for t in s.tags if "button" in t]
    print(f"  Button {tags}: pos=({s.x},{s.y}) size={s.width}x{s.height}")

# Try different scale factors
btns_a_r = [s for s in level.get_sprites_by_tag("sys_click") if any("button_A_R" in t for t in s.tags)]
btn = btns_a_r[0]
print(f"\nTarget button A_R at grid ({btn.x},{btn.y})")

# Find a goal tile to track
goal_tiles = [s for s in level.get_sprites() if s.tags and "goal" in s.tags]
gt = goal_tiles[0]
print(f"Goal tile at ({gt.x},{gt.y})")

# The camera dimensions may have changed
cam_w2, cam_h2 = game.camera.width, game.camera.height
print(f"Camera now: {cam_w2}x{cam_h2}")

# Try scale based on grid_size
# If grid is 41x41 and frame is 64x64, scale might be different
for scale_try in [1, 2]:
    for y_off_try in [0, 6, 8, 10, 12, 13]:
        gx = btn.x + btn.width // 2
        gy = btn.y + btn.height // 2
        dx = gx * scale_try
        dy = gy * scale_try + y_off_try
        result = game.camera.display_to_grid(dx, dy)
        if result and abs(result[0] - gx) < 3 and abs(result[1] - gy) < 3:
            print(f"  scale={scale_try} y_off={y_off_try}: display({dx},{dy}) → grid={result} (target grid ({gx},{gy}))")

# Just try clicking at the raw grid positions
for attempt in [(btn.x + 1, btn.y + 2), (btn.x + btn.width//2, btn.y + btn.height//2)]:
    gx, gy = attempt
    # Try multiple scale/offset combos
    for s in [1, 2]:
        for yo in range(0, 20, 2):
            dx, dy = gx * s, gy * s + yo
            if dx < 64 and dy < 64:
                result = game.camera.display_to_grid(dx, dy)
                if result:
                    rx, ry = result
                    if rx == gx and ry == gy:
                        gt_before = (gt.x, gt.y)
                        fd = env.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
                        moved = gt.x != gt_before[0] or gt.y != gt_before[1]
                        if moved:
                            print(f"  CLICK WORKED! display({dx},{dy}) scale={s} yo={yo}")
                            print(f"  Goal moved from {gt_before} to ({gt.x},{gt.y})")
                            break
            if moved:
                break
        if moved:
            break
