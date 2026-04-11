#!/usr/bin/env python3
"""Track target positions as chain moves in sk48."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

arcade = Arcade()
env = arcade.make('sk48-41055498')
fd = env.reset()
game = env._game

def state():
    targets = game.vbelzuaian
    tpos = [(t.x, t.y, int(t.pixels[1,1])) for t in targets]
    head = game.vzvypfsnt
    segs = game.mwfajkguqx[head]
    spos = [(s.x, s.y) for s in segs]
    matches = game.vjfbwggsd.get(head, [])
    win = game.gvtmoopqgy()
    return tpos, spos, len(matches), head.x, head.y, win, game.qiercdohl

def show(label):
    tpos, spos, nm, hx, hy, win, budget = state()
    print(f"{label}: head=({hx},{hy}) segs={spos} matches={nm} budget={budget} win={win}")
    # Only show targets that moved
    board_targets = [(x,y,c) for x,y,c in tpos if y < 53]
    print(f"  board targets: {board_targets}")

show("init")

# The head is at (11,36) facing right (rotation=0)
# Board targets: (41,30)=14, (41,24)=9, (41,18)=8
# Chain has 2 segments at (11,36) and (17,36)
# Walls at (13,32), (13,26), (13,20), (13,14)

# Let's first extend chain right to reach x=41
# Need to go from x=17 (last seg) to x=41 = 4 steps of 6px
for i in range(4):
    fd = env.step(GameAction.ACTION4)
show("after 4 RIGHT")

# Now last segment should be at x=41, y=36
# But targets are at x=41, y=30/24/18 — different y
# Try extending more to see if we hit anything
fd = env.step(GameAction.ACTION4)
show("after 5 RIGHT")

# Try going up (lateral) — needs wall at head's position
# Walls are at x=13, but head is at x=11. Let me go back left first.
# Actually let me try going UP to see if it works
fd = env.step(GameAction.ACTION1)
show("after UP")

# Let me try going left (retract)
fd = env.step(GameAction.ACTION3)
show("after LEFT (retract)")

# More retracts
for i in range(3):
    fd = env.step(GameAction.ACTION3)
show("after 4 total LEFT")

# Now the chain should be shorter. Let me check undo
fd = env.step(GameAction.ACTION7)
show("after UNDO")
