#!/usr/bin/env python3
"""Check reference chain layout and win condition for sk48 L2."""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

CELL = 6
UP, DOWN, LEFT, RIGHT = GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4

arcade = Arcade()
env = arcade.make('sk48-41055498')
fd = env.reset()
game = env._game

L0_SOL = [UP, UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, LEFT, DOWN, DOWN, RIGHT, LEFT, UP, RIGHT]
L1_SOL = [UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, UP, LEFT, LEFT, UP, RIGHT, RIGHT, DOWN, DOWN,
          RIGHT, UP, RIGHT, LEFT, LEFT, UP, RIGHT, RIGHT, LEFT, LEFT, UP, RIGHT, RIGHT]

def step_action(action):
    f = env.step(action)
    while game.ljprkjlji or game.pzzwlsmdt:
        f = env.step(action)
    while game.lgdrixfno >= 0:
        f = env.step(action)
    return f

# Complete L0, L1
for a in L0_SOL:
    fd = step_action(a)
for a in L1_SOL:
    fd = step_action(a)

# L2 reference analysis
head = game.vzvypfsnt
print(f"Head: ({head.x},{head.y}) rot={head.rotation} tags={head.tags}")

ref = game.xpmcmtbcv.get(head)
print(f"Ref head: ({ref.x},{ref.y}) rot={ref.rotation} tags={ref.tags}")

# Head chain segments
head_segs = game.mwfajkguqx[head]
print(f"\nHead chain ({len(head_segs)} segs):")
for i, s in enumerate(head_segs):
    print(f"  [{i}] ({s.x},{s.y}) {s.width}x{s.height}")

# Reference chain segments
ref_segs = game.mwfajkguqx[ref]
print(f"\nRef chain ({len(ref_segs)} segs):")
for i, s in enumerate(ref_segs):
    print(f"  [{i}] ({s.x},{s.y}) {s.width}x{s.height}")

# Reference chain matches (vjfbwggsd)
ref_matches = game.vjfbwggsd.get(ref, [])
print(f"\nRef matches (vjfbwggsd[ref]): {len(ref_matches)}")
for i, m in enumerate(ref_matches):
    print(f"  [{i}] ({m.x},{m.y}) color={int(m.pixels[1,1])}")

# Head chain matches
head_matches = game.vjfbwggsd.get(head, [])
print(f"\nHead matches (vjfbwggsd[head]): {len(head_matches)}")
for i, m in enumerate(head_matches):
    print(f"  [{i}] ({m.x},{m.y}) color={int(m.pixels[1,1])}")

# Check marks (jdojcthkf)
for key in game.jdojcthkf:
    marks = game.jdojcthkf[key]
    print(f"\nCheck marks for ({key.x},{key.y}): {len(marks)}")

# All targets
print("\nAll targets (vbelzuaian):")
for t in sorted(game.vbelzuaian, key=lambda t: (t.y, t.x)):
    print(f"  ({t.x},{t.y}) color={int(t.pixels[1,1])}")

# Play area
pa = game.lqwkgffeb
print(f"\nPlay area: ({pa.x},{pa.y}) size=({pa.pixels.shape[1]*CELL},{pa.pixels.shape[0]*CELL})")
print(f"Play area bounds: x=[{pa.x}, {pa.x + pa.pixels.shape[1]*CELL}), y=[{pa.y}, {pa.y + pa.pixels.shape[0]*CELL})")
