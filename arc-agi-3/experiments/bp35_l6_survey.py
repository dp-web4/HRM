#!/usr/bin/env python3
"""Quick L6 survey to find G block positions for gravity flipping."""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
CLICK = GameAction.ACTION6

def click_act(env, gx, gy):
    engine = env._game.oztjzzyqoek
    cam_y = engine.camera.rczgvgfsfb[1]
    return env.step(CLICK, data={"x": gx*6, "y": gy*6 - cam_y})

R = ('R',)
L = ('L',)
def C(x, y): return ('C', x, y)

# Quick solutions for L0-L5 (verified)
L0_SOL = [R,R,R,R, C(7,19), C(4,16), L,L,L, C(4,15), C(4,12), R, C(5,9), L,L]
L1_SOL = [R,R,R,R,R, C(8,36), C(8,35), L,L, C(5,29),L, C(4,29),L, C(3,29),L, C(2,29),L, C(2,28), R,R,R, C(5,24), C(5,23), L,L, C(3,20), C(3,17), C(3,16), C(4,16),R, C(5,16),R, C(6,16),R, C(7,16),R, C(8,16),R, C(8,15), C(8,14), L, L, L, C(5,9)]
L2_SOL = [C(5,28),R,R,R,C(6,27), C(5,23),C(4,23),C(3,23),L,L,L,L, R, C(5,17),C(6,17),C(5,18),C(6,18),R,R,R,R, C(6,12),C(5,12),C(4,12),C(3,12),L,L,L,L, C(5,7),R,R,R,R]
L3_SOL = [C(3,17),C(7,23),C(7,24),C(5,7), L, R,R, C(3,23), R,R, C(5,23), L,L,L, C(4,31)]
L4_SOL = [R,R,R,R, C(7,9),C(8,9),C(9,9), C(8,12), C(8,29), R,R, L,L,L,L]
L5_SOL = [R,R,R,R,R, C(4,31), L,L,L,L,L,L]

solutions = [L0_SOL, L1_SOL, L2_SOL, L3_SOL, L4_SOL, L5_SOL]

arcade = Arcade()
env = arcade.make('bp35-0a0ad940')
fd = env.reset()

# Solve L0-L5
for i, sol in enumerate(solutions):
    old_level = env._game.level_index
    for m in sol:
        if m[0] == 'R': env.step(RIGHT)
        elif m[0] == 'L': env.step(LEFT)
        elif m[0] == 'C': click_act(env, m[1], m[2])
    # Animation
    for _ in range(10):
        env.step(LEFT)
        if env._game.level_index > old_level:
            break
    print(f"L{i}: level now {env._game.level_index}")

# L6 survey
print(f"\n=== L6 Survey ===")
engine = env._game.oztjzzyqoek
grid = engine.hdnrlfmyrj
player = engine.twdpowducb
pp = tuple(player.qumspquyus)
grav = engine.vivnprldht
print(f"Player: {pp}, Gravity UP: {grav}")

# Find all G blocks
g_blocks = []
ob_blocks = []
for y in range(-2, 35):
    for x in range(-5, 15):
        items = grid.jhzcxkveiw(x, y)
        if items:
            name = items[0].name
            if name == 'lrpkmzabbfa':  # G block
                g_blocks.append((x, y))
            elif name == 'oonshderxef':  # O block
                ob_blocks.append((x, y))
            elif name == 'yuuqpmlxorv':  # B block
                ob_blocks.append((x, y))

print(f"\nG blocks ({len(g_blocks)}):")
for pos in sorted(g_blocks, key=lambda p: (p[0], p[1])):
    print(f"  {pos}")

print(f"\nO/B blocks ({len(ob_blocks)}):")
for pos in sorted(ob_blocks, key=lambda p: (p[0], p[1])):
    items = grid.jhzcxkveiw(pos[0], pos[1])
    btype = 'O' if items[0].name == 'oonshderxef' else 'B'
    print(f"  {pos} = {btype}")
