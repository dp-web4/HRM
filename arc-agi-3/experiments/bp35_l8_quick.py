#!/usr/bin/env python3
"""Quick timing test for make_l8_env"""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
CLICK = GameAction.ACTION6

t0 = time.time()
arcade = Arcade()
print(f"Arcade: {time.time()-t0:.1f}s")

t1 = time.time()
env = arcade.make('bp35-0a0ad940')
print(f"make: {time.time()-t1:.1f}s")

t2 = time.time()
env.reset()
print(f"reset: {time.time()-t2:.1f}s")

print(f"Level: {env._game.level_index}")
engine = env._game.oztjzzyqoek
p = engine.twdpowducb
print(f"Player: {tuple(p.qumspquyus)} grav={engine.vivnprldht}")
print(f"Total: {time.time()-t0:.1f}s")
