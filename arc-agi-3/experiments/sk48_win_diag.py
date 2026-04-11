#!/usr/bin/env python3
"""Diagnose why h=0 doesn't win: explore the exact win condition for L2."""
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
print(f"At L2, completed={fd.levels_completed}")

# === Analyze the win condition precisely ===
head = game.vzvypfsnt
ref = game.xpmcmtbcv.get(head)
print(f"\nHead: ({head.x},{head.y}) rot={head.rotation}")
print(f"Ref:  ({ref.x},{ref.y}) rot={ref.rotation}")

# Reference chain: segments and their target overlaps
ref_segs = game.mwfajkguqx[ref]
print(f"\nRef chain ({len(ref_segs)} segs):")
for i, s in enumerate(ref_segs):
    target = game.ebribtrdgw(s.x, s.y)
    tc = int(target.pixels[1,1]) if target else None
    print(f"  seg[{i}] ({s.x},{s.y}) -> target color={tc}")

# Build vjfbwggsd[ref] manually
game.gvtmoopqgy()  # This rebuilds vjfbwggsd
ref_vjf = game.vjfbwggsd[ref]
print(f"\nvjfbwggsd[ref] ({len(ref_vjf)} entries):")
for i, t in enumerate(ref_vjf):
    print(f"  [{i}] ({t.x},{t.y}) color={int(t.pixels[1,1])}")

# jdojcthkf[ref] — the checkmarks that must all be visible
ref_checks = game.jdojcthkf[ref]
print(f"\njdojcthkf[ref] ({len(ref_checks)} entries):")
for i, c in enumerate(ref_checks):
    print(f"  [{i}] ({c.x},{c.y}) visible={c.is_visible}")

# Head chain current state
head_segs = game.mwfajkguqx[head]
print(f"\nHead chain ({len(head_segs)} segs):")
for i, s in enumerate(head_segs):
    target = game.ebribtrdgw(s.x, s.y)
    tc = int(target.pixels[1,1]) if target else None
    print(f"  seg[{i}] ({s.x},{s.y}) -> target color={tc}")

head_vjf = game.vjfbwggsd[head]
print(f"\nvjfbwggsd[head] ({len(head_vjf)} entries):")
for i, t in enumerate(head_vjf):
    print(f"  [{i}] ({t.x},{t.y}) color={int(t.pixels[1,1])}")

# === What the win condition actually needs ===
print("\n=== WIN CONDITION REQUIREMENTS ===")
print(f"Need {len(ref_vjf)} target overlaps on head chain segments, matching colors:")
for i, rt in enumerate(ref_vjf):
    rc = int(rt.pixels[1,1])
    print(f"  position {i}: color {rc}")

print(f"\nHead chain extends RIGHT from head at x={head.x}")
print(f"Segments at: head.x, head.x+6, head.x+12, ...")
print(f"With head.x={head.x}:")
for L in range(2, 8):
    positions = [head.x + i*CELL for i in range(L)]
    in_play = [x for x in positions if 11 <= x < 53]
    print(f"  L={L}: seg positions={positions}, in-play={in_play} ({len(in_play)} could have targets)")

# What if chain is at various y levels?
print(f"\nTargets currently at:")
for t in sorted(game.vbelzuaian, key=lambda t: (t.y, t.x)):
    if t.y < 53:
        print(f"  ({t.x},{t.y}) color={int(t.pixels[1,1])}")

print(f"\nFor win: need chain at some y-level where:")
print(f"  First in-play segments have targets with colors matching ref: ", end="")
print([int(t.pixels[1,1]) for t in ref_vjf])
print(f"  ref vjfbwggsd builds by scanning segments IN ORDER, keeping only those with target overlap")
print(f"  So if chain has L=5 segs at x=[5,11,17,23,29], seg at x=5 is outside play and has no target")
print(f"  vjfbwggsd[head] would collect targets at x=11,17,23,29 → need colors 8,12,9,14 there")

# Verify: what if chain is L=6?
print(f"\n  If L=6: segs at x=[5,11,17,23,29,35]")
print(f"  In play: [11,17,23,29,35] - need 4 of these to have targets with right colors")
print(f"  vjfbwggsd collects ALL segments with targets, not just first 4")
print(f"  So if 5 segments have targets, vjfbwggsd[head] has 5 entries but ref only has 4")
print(f"  Would that cause fail? Check i >= len(vjfbwggsd[head]) at line 853...")
print(f"  No - it checks if i >= len(vjfbwggsd[bcwsrdcswp]) which is head's list")
print(f"  Actually the loop is over jdojcthkf[ref] which has len = len(vjfbwggsd[ref]) = 4")
print(f"  So as long as vjfbwggsd[head] has >= 4 entries, the first 4 must match colors")

# Critical: what about the ordering?
print(f"\n=== ORDERING MATTERS ===")
print(f"vjfbwggsd is built LEFT to RIGHT (iterating segments in order)")
print(f"So if L=5, segs=[5,11,17,23,29]:")
print(f"  Skip x=5 (no target), then targets at x=11,17,23,29")
print(f"  vjfbwggsd[head] = [target@11, target@17, target@23, target@29]")
print(f"  Must match ref:   [color 8,    color 12,   color 9,    color 14]")
print(f"  So: target@(11,Y)=c8, target@(17,Y)=c12, target@(23,Y)=c9, target@(29,Y)=c14")
print(f"\nIf L=6, segs=[5,11,17,23,29,35]:")
print(f"  Skip x=5, targets at x=11,17,23,29 (x=35 might not have a target)")
print(f"  If only 4 targets overlap: same as L=5 case")
print(f"  If x=35 also has a target: vjfbwggsd has 5 entries, match first 4 → same req")

# What y-level works? Targets start at y=6,12,18,24
# Chain must be at same y as targets for overlap
print(f"\n=== REQUIRED TARGET POSITIONS ===")
print(f"Need targets at these exact positions (on chain at y=Y):")
print(f"  (11, Y, 8), (17, Y, 12), (23, Y, 9), (29, Y, 14)")
print(f"\nCurrent targets:")
print(f"  (29, 6, 14) → needs to go to (29, Y, 14) — already at x=29!")
print(f"  (29, 12, 9) → needs to go to (23, Y, 9)")
print(f"  (29, 18, 8) → needs to go to (11, Y, 8)")
print(f"  (29, 24, 12) → needs to go to (17, Y, 12)")
print(f"\nMinimum horizontal moves needed:")
print(f"  c14: 0 cells left (stay at x=29)")
print(f"  c9: 1 cell left (29→23)")
print(f"  c8: 3 cells left (29→11)")
print(f"  c12: 2 cells left (29→17)")
print(f"  Total: 6 cells of horizontal pushing")
print(f"\nBUT all must end at the same Y! Current y-levels differ (6,12,18,24)")
print(f"Chain can only be at one y-level → need to bring targets to same y")
