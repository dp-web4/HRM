#!/usr/bin/env python3
"""Test game undo mechanism for search reliability."""
import sys, os
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


def state_snapshot():
    """Quick state description."""
    head = game.vzvypfsnt
    segs = game.mwfajkguqx[head]
    seg_pos = [(s.x, s.y) for s in segs]
    targets = sorted([(t.x, t.y, int(t.pixels[1,1])) for t in game.vbelzuaian if t.y < 53])
    return head.y, len(segs), tuple(seg_pos), tuple(targets)


# Complete L0 and L1
for a in L0_SOL:
    fd = step_action(a)
for a in L1_SOL:
    fd = step_action(a)
print(f"At L2, completed={fd.levels_completed}")

# Test undo reliability
s0 = state_snapshot()
b0 = game.qiercdohl
h0 = len(game.seghobzez)
print(f"Initial: {s0}, budget={b0}, hist={h0}")

# Do 3 moves
step_action(RIGHT)  # extend
step_action(RIGHT)  # extend
step_action(UP)     # slide up
s1 = state_snapshot()
b1 = game.qiercdohl
h1 = len(game.seghobzez)
print(f"After 3 moves: {s1}, budget={b1}, hist={h1}")

# Undo all 3
for i in range(3):
    game.uqclctlhyh()
game.qiercdohl = b0  # restore budget
s2 = state_snapshot()
b2 = game.qiercdohl
h2 = len(game.seghobzez)
print(f"After undo:    {s2}, budget={b2}, hist={h2}")
print(f"States match: {s0 == s2}")

# Do the same 3 moves again and check reproducibility
step_action(RIGHT)
step_action(RIGHT)
step_action(UP)
s3 = state_snapshot()
print(f"Redo same:     {s3}")
print(f"Reproduce:     {s1 == s3}")

# Undo 2 and try different path
game.uqclctlhyh()
game.uqclctlhyh()
step_action(UP)
step_action(RIGHT)
s4 = state_snapshot()
print(f"Alt path:      {s4}")

# Undo back to start
while len(game.seghobzez) > h0:
    game.uqclctlhyh()
game.qiercdohl = b0
s5 = state_snapshot()
print(f"Back to start: {s5}")
print(f"Match start:   {s0 == s5}")

# Stress test: do 10 moves, undo all, check
import random
random.seed(42)
actions = [UP, DOWN, LEFT, RIGHT]
done_moves = []
for i in range(10):
    a = random.choice(actions)
    old_budget = game.qiercdohl
    step_action(a)
    if game.qiercdohl < old_budget:
        done_moves.append(a)
s6 = state_snapshot()
print(f"\nAfter {len(done_moves)} random moves: {s6}")

# Undo all
while len(game.seghobzez) > h0:
    game.uqclctlhyh()
game.qiercdohl = b0
s7 = state_snapshot()
print(f"After undo all: {s7}")
print(f"Match original: {s0 == s7}")
