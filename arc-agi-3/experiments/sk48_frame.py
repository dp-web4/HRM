#!/usr/bin/env python3
"""Render sk48 frame to understand layout."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

arcade = Arcade()
env = arcade.make('sk48-41055498')
fd = env.reset()
game = env._game

# Get the rendered frame
frame = fd.frame[0]  # (1, H, W) → (H, W)
print(f"Frame shape: {frame.shape}")
print(f"Unique colors: {np.unique(frame)}")

# Print a compact view of the frame
# Map colors to characters
char_map = {0: '.', 1: '#', 2: 'G', 3: 'R', 4: 'B', 5: 'Y', 6: 'M', 7: 'C',
            8: 'O', 9: 'P', 10: 'g', 11: 'b', 12: 'y', 13: 'r', 14: 'p', 15: 'W'}
for y in range(0, 64, 3):
    row = ""
    for x in range(0, 64, 3):
        c = int(frame[y, x])
        row += char_map.get(c, str(c%10))
    print(row)

# Also show the chain and target details
level = game.current_level
all_sprites = level.get_sprites()
print(f"\n{len(all_sprites)} total sprites")

# Show all sprites with their positions
for s in sorted(all_sprites, key=lambda s: (s.y, s.x)):
    tags = s.tags if s.tags else []
    if 'jtteddgeyl' in tags or 'irkeobngyh' in tags:
        continue  # skip boundaries/walls
    if s.y < 53:  # only board sprites
        c = int(s.pixels[1,1]) if s.height >= 2 and s.width >= 2 else -1
        print(f"  ({s.x:2d},{s.y:2d}) {s.width}x{s.height} c={c:2d} tags={tags} name={s.name} rot={s.rotation}")
