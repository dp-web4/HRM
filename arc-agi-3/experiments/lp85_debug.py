#!/usr/bin/env python3
"""Debug lp85 click mechanics."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

arcade = Arcade()
env = arcade.make('lp85-305b61c3')
fd = env.reset()
game = env._game

level = game.current_level
print(f"Camera: {game.camera.width}x{game.camera.height}")
print(f"Grid: {level.grid_size}")
frame = np.array(fd.frame)
print(f"Frame shape: {frame.shape}")

# Find the L button
btn = None
for s in level.get_sprites_by_tag("sys_click"):
    for t in s.tags:
        if "button_A_L" in t:
            btn = s
            break

print(f"\nL button: pos=({btn.x},{btn.y}) size={btn.width}x{btn.height}")

# Find the goal tile
goal_tile = None
for s in level.get_sprites():
    if s.tags and "goal" in s.tags:
        goal_tile = s
        break

print(f"Goal tile: pos=({goal_tile.x},{goal_tile.y}) name={goal_tile.name}")

# Try clicking at different coordinates
# The camera is 16x16, frame is 64x64, so scale factor = 4
# display_to_grid: display_coord / 4 = grid_coord
# So grid (2, 10) = display (8, 40)

# Button is at grid (1,8), size 3x4. Center grid = (2.5, 10)
# Display center = (10, 40)
print(f"\nTrying click at display coords (10, 40)...")
print(f"  Before: goal at ({goal_tile.x},{goal_tile.y})")

fd = env.step(GameAction.ACTION6, data={'x': 10, 'y': 40})
print(f"  After: goal at ({goal_tile.x},{goal_tile.y})")

# Try again
fd = env.step(GameAction.ACTION6, data={'x': 10, 'y': 40})
print(f"  After 2: goal at ({goal_tile.x},{goal_tile.y})")

# Check: what does display_to_grid return for our coords?
result = game.camera.display_to_grid(10, 40)
print(f"\n  display_to_grid(10, 40) = {result}")
result = game.camera.display_to_grid(2, 8)
print(f"  display_to_grid(2, 8) = {result}")

# Camera 32x19, frame 64x64. Scale = 64/32 = 2. Letterbox: (64-38)/2 = 13 top/bottom
# display_x = grid_x * 2, display_y = grid_y * 2 + 13
# Button center grid = (2.5, 10) → display = (5, 33)

for dx, dy in [(5, 33), (4, 32), (3, 30), (5, 30)]:
    goal_pos_before = (goal_tile.x, goal_tile.y)
    result = game.camera.display_to_grid(dx, dy)
    fd = env.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
    moved = goal_tile.x != goal_pos_before[0] or goal_tile.y != goal_pos_before[1]
    print(f"  click({dx},{dy}) → grid={result}, goal moved={moved}, now at ({goal_tile.x},{goal_tile.y})")
    if moved:
        break
