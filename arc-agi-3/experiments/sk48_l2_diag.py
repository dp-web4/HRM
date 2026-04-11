#!/usr/bin/env python3
"""Diagnose sk48 L2 state and target push behavior."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

CELL = 6
UP, DOWN, LEFT, RIGHT = GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4
ACTION_NAMES = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT'}

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


def show():
    head = game.vzvypfsnt
    segs = game.mwfajkguqx[head]
    seg_pos = [(s.x, s.y) for s in segs]
    targets = [(t.x, t.y, int(t.pixels[1,1])) for t in game.vbelzuaian if t.y < 53]
    print(f"  head=({head.x},{head.y}) rot={head.rotation} L={len(segs)} budget={game.qiercdohl}")
    print(f"  segs={seg_pos}")
    print(f"  targets={sorted(targets)}")

    ref = game.xpmcmtbcv.get(head)
    if ref:
        ref_segs = game.mwfajkguqx.get(ref, [])
        ref_targets = game.vjfbwggsd.get(ref, [])
        ref_colors = [int(t.pixels[1,1]) for t in ref_targets] if ref_targets else [int(s.pixels[1,1]) for s in ref_segs]
        print(f"  ref_colors={ref_colors}")

    # Walls
    walls = []
    for s in game.current_level.get_sprites():
        if s.tags and 'irkeobngyh' in s.tags:
            walls.append((s.x, s.y))
    print(f"  walls={sorted(walls)}")

    # Play area
    pa = game.lqwkgffeb
    print(f"  play_area=({pa.x},{pa.y}) size=({pa.pixels.shape[1]*CELL},{pa.pixels.shape[0]*CELL})")


# Complete L0 and L1
for a in L0_SOL:
    fd = step_action(a)
print(f"L0 done: completed={fd.levels_completed}")
for a in L1_SOL:
    fd = step_action(a)
print(f"L1 done: completed={fd.levels_completed}")

# L2 state
print("\n=== Level 2 ===")
show()

# Try each action one at a time and see effect
for action in [UP, DOWN, LEFT, RIGHT]:
    # Save state for undo
    budget_save = game.qiercdohl
    hist_len = len(game.seghobzez)

    # Do one step manually to see intermediate state
    old_budget = game.qiercdohl
    f = env.step(action)
    has_anim = bool(game.ljprkjlji)
    has_slide = bool(game.pzzwlsmdt)
    new_budget = game.qiercdohl

    if new_budget == old_budget and not has_anim:
        print(f"\n{ACTION_NAMES[action]}: no effect")
        continue

    # Check what's in animation queue
    anim_sprites = [(s.x, s.y, tx, ty) for s, (tx, ty) in game.ljprkjlji]
    slide_sprites = [(s.x, s.y, int(s.pixels[1,1])) for s in game.pzzwlsmdt]
    print(f"\n{ACTION_NAMES[action]}: budget {old_budget}->{new_budget}")
    print(f"  ljprkjlji({len(anim_sprites)}): {anim_sprites}")
    print(f"  pzzwlsmdt({len(slide_sprites)}): {slide_sprites}")

    # Complete animation
    while game.ljprkjlji or game.pzzwlsmdt:
        f = env.step(action)
    while game.lgdrixfno >= 0:
        f = env.step(action)

    # Show resulting state
    head = game.vzvypfsnt
    segs = game.mwfajkguqx[head]
    seg_pos = [(s.x, s.y) for s in segs]
    targets = [(t.x, t.y, int(t.pixels[1,1])) for t in game.vbelzuaian if t.y < 53]
    print(f"  result: L={len(segs)} segs={seg_pos}")
    print(f"  targets={sorted(targets)}")

    # Undo
    game.uqclctlhyh()
    game.qiercdohl = budget_save

print("\n\n=== Exploring L2 move sequences ===")
# Try a few initial moves to understand the puzzle
# Try: RIGHT (extend), then UP, DOWN to see sliding behavior
show()

print("\n--- Extend RIGHT ---")
fd = step_action(RIGHT)
show()

print("\n--- Extend RIGHT again ---")
fd = step_action(RIGHT)
show()

print("\n--- Try UP (slide?) ---")
budget_save = game.qiercdohl
f = env.step(UP)
anim_sprites = [(s.x, s.y, tx, ty) for s, (tx, ty) in game.ljprkjlji]
slide_sprites = [(s.x, s.y, int(s.pixels[1,1])) for s in game.pzzwlsmdt]
print(f"  ljprkjlji({len(anim_sprites)}): {anim_sprites}")
print(f"  pzzwlsmdt({len(slide_sprites)}): {slide_sprites}")
while game.ljprkjlji or game.pzzwlsmdt:
    f = env.step(UP)
show()
