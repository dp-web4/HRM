#!/usr/bin/env python3
"""Debug g50t L1 step by step."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GA = {a.value: a for a in GameAction}
NAME_TO_INT = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4, 'UNDO': 5}

arcade = Arcade()
env = arcade.make('g50t-5849a774')
fd = env.reset()

game = env._game
lc = game.vgwycxsxjz

print(f"Initial: player=({lc.dzxunlkwxt.x},{lc.dzxunlkwxt.y})")
print(f"  goal=({lc.whftgckbcu.x},{lc.whftgckbcu.y})")
print(f"  win target: ({lc.whftgckbcu.x+1},{lc.whftgckbcu.y+1})")
print(f"  start: ({lc.yugzlzepkr},{lc.vgpdqizwwm})")
print(f"  areahjypvy: {list(lc.areahjypvy)}")
print(f"  indicators: {len(lc.drofvwhbxb)}")
print(f"  obstacles: {len(lc.uwxkstolmf)}")
for i, obs in enumerate(lc.uwxkstolmf):
    print(f"    obs[{i}]: ({obs.x},{obs.y})")
print(f"  modifiers: {len(lc.hamayflsib)}")
for i, mod in enumerate(lc.hamayflsib):
    pc = mod.nexhtmlmxh
    targets = [(o.x,o.y) for o in pc.ytztewxdin] if pc else []
    print(f"    mod[{i}]: ({mod.x},{mod.y}) → {targets}")

# Play L1 step by step
L1 = 'LEFT LEFT UNDO DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT UP UP LEFT LEFT LEFT LEFT UP'
actions = L1.split()
for i, name in enumerate(actions):
    prev_lc_count = fd.levels_completed
    fd = env.step(INT_TO_GA[NAME_TO_INT[name]])
    p = lc.dzxunlkwxt
    g = lc.whftgckbcu
    level_up = fd.levels_completed > prev_lc_count
    sym = '★' if level_up else '·'
    ghosts = len(lc.rloltuowth)
    path = list(lc.areahjypvy)
    print(f"  {sym} {i+1:2d}. {name:5s} → P=({p.x},{p.y}) G=({g.x+1},{g.y+1}) ghosts={ghosts} path_len={len(path)} L={fd.levels_completed}")

print(f"\nFinal: player=({p.x},{p.y}), goal_target=({g.x+1},{g.y+1})")
print(f"Match: {p.x == g.x+1 and p.y == g.y+1}")
print(f"State: {fd.state.name}")
