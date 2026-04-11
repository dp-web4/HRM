#!/usr/bin/env python3
"""Play sk48 interactively to understand mechanics."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

arcade = Arcade()
env = arcade.make('sk48-41055498')
fd = env.reset()
game = env._game

def print_state():
    print(f"  Budget: {game.qiercdohl}/{game.vhzjwcpmk}")
    print(f"  Selected: ({game.vzvypfsnt.x},{game.vzvypfsnt.y})" if game.vzvypfsnt else "  Selected: None")
    for head, segs in game.mwfajkguqx.items():
        matches = game.vjfbwggsd.get(head, [])
        seg_info = [(s.x, s.y, int(s.pixels[1,1]) if s.height>=2 and s.width>=2 else -1) for s in segs]
        print(f"  Head({head.x},{head.y}): {len(segs)} segs, {len(matches)} matches: {seg_info}")
    print(f"  Win: {game.gvtmoopqgy()}")

print("=== Initial ===")
print_state()

# Try moving right
print("\n=== Move RIGHT ===")
fd = env.step(GameAction.ACTION4)
print_state()

print("\n=== Move RIGHT ===")
fd = env.step(GameAction.ACTION4)
print_state()

print("\n=== Move RIGHT ===")
fd = env.step(GameAction.ACTION4)
print_state()

print("\n=== Move UP ===")
fd = env.step(GameAction.ACTION1)
print_state()

print("\n=== Move UP ===")
fd = env.step(GameAction.ACTION1)
print_state()

print("\n=== Move UP ===")
fd = env.step(GameAction.ACTION1)
print_state()

print("\n=== Move UP ===")
fd = env.step(GameAction.ACTION1)
print_state()
