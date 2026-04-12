#!/usr/bin/env python3
"""Dump full L7 frame and find actual pixel positions of pieces."""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
import lf52_solve_final as solver

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game
    for lvl in range(6):
        solver.solve_level(env, game, lvl)

    eq = game.ikhhdzfmarl
    state = solver.extract_state(eq)
    print(f"L7 offset={state['offset']}")
    frame = np.array(env.observation_space.frame[0])
    print(f"frame shape: {frame.shape}")
    print(f"Full frame (colors):")
    for y in range(frame.shape[0]):
        row = ""
        for x in range(frame.shape[1]):
            c = int(frame[y, x])
            row += f"{c:2d} " if c >= 10 else f" {c} "
        print(f"{y:2}: {row}")

    # Probe click at several candidate positions using action count changes
    import sys as _s
    steps_before = eq.asqvqzpfdi
    for px, py in [(8,14),(9,13),(7,13),(10,14),(8,12),(6,14)]:
        s_b = eq.asqvqzpfdi
        fd = env.step(GameAction.ACTION6, data={'x': px, 'y': py})
        eq2 = game.ikhhdzfmarl
        s_a = eq2.asqvqzpfdi
        print(f"click ({px},{py}): steps {s_b}->{s_a}, zvcnglshzcx={eq2.zvcnglshzcx}")

if __name__ == "__main__":
    main()
