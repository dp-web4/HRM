#!/usr/bin/env python3
"""Deeper exploration of sk48 mechanics."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

arcade = Arcade()
env = arcade.make('sk48-41055498')
fd = env.reset()
game = env._game

CELL = 6

def print_state(label=""):
    if label: print(f"\n=== {label} ===")
    print(f"  Budget: {game.qiercdohl}")
    head = game.vzvypfsnt
    rot = head.rotation
    print(f"  Head: ({head.x},{head.y}) rot={rot}° dir={game.ghcqtpzzlq.__doc__}")
    segs = game.mwfajkguqx[head]
    for i, s in enumerate(segs):
        tgt = game.ebribtrdgw(s.x, s.y) if hasattr(game, 'ebribtrdgw') else None
        tgt_color = int(tgt.pixels[1,1]) if tgt else None
        print(f"    seg[{i}] ({s.x},{s.y}) → target color={tgt_color}")

    # Reference chain
    if head in game.xpmcmtbcv:
        ref = game.xpmcmtbcv[head]
        ref_segs = game.mwfajkguqx[ref]
        ref_matches = game.vjfbwggsd.get(ref, [])
        print(f"  Reference at ({ref.x},{ref.y}):")
        for i, s in enumerate(ref_segs):
            tgt = game.ebribtrdgw(s.x, s.y) if hasattr(game, 'ebribtrdgw') else None
            tgt_color = int(tgt.pixels[1,1]) if tgt else None
            print(f"    ref[{i}] ({s.x},{s.y}) → target color={tgt_color}")

    print(f"  Win: {game.gvtmoopqgy()}")

# Check ebribtrdgw function
print("Available methods:")
for name in dir(game):
    if 'ebr' in name.lower():
        print(f"  {name}")

# Find the target lookup function
targets = game.vbelzuaian
print(f"\nTarget positions:")
for t in targets:
    print(f"  ({t.x},{t.y}) color={int(t.pixels[1,1])}")

print_state("Initial")

# What direction does head 1 face?
head = game.vzvypfsnt
from environment_files.sk48 import sk48 as sk48_mod
dir_map = sk48_mod.hhvuoijeua
print(f"\nDirection map: {dir_map}")
print(f"Head rotation: {head.rotation}, direction: {dir_map[head.rotation]}")

# Move forward (right for rotation=0)
for i in range(5):
    fd = env.step(GameAction.ACTION4)
    print_state(f"RIGHT #{i+1}")

# Now try to go up at a wall
fd = env.step(GameAction.ACTION1)
print_state("UP #1")

fd = env.step(GameAction.ACTION1)
print_state("UP #2")
