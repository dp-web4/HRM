#!/usr/bin/env python3
"""L7 intro-screen dismissal test. After L6, frame shows intro not L7 grid.
Try clicking icons/buttons to dismiss and reach gameplay."""
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

def frame_sig(env):
    return np.array(env.observation_space.frame[0]).tobytes()

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game
    for lvl in range(6):
        solver.solve_level(env, game, lvl)

    eq = game.ikhhdzfmarl
    before = frame_sig(env)
    print(f"L7 start. frame_sig before={len(before)}")

    # Try clicking at icon centers (rows 23-26 cols 2-5 and 38-41 have the 15/7 icon patterns)
    candidates = [
        (4, 25), (40, 25),  # 15/7 icon centers
        (32, 32),  # center
        (3, 59),  # bottom-left button area? (row 59 cols 0-5 are 10)
        (30, 30),
        (27, 26),  # near color-7 patches
        (40, 25),
    ]
    for px, py in candidates:
        b = frame_sig(env)
        fd = env.step(GameAction.ACTION6, data={'x': px, 'y': py})
        a = frame_sig(env)
        eq2 = game.ikhhdzfmarl
        state = solver.extract_state(eq2)
        changed = b != a
        print(f"click ({px},{py}): frame_changed={changed} pieces={len(state['pieces'])} offset={state['offset']} state={fd.state.name}")

    # Try ACTION5 / ACTION7 to dismiss
    print("\nACTION7:")
    b = frame_sig(env)
    fd = env.step(GameAction.ACTION7)
    a = frame_sig(env)
    print(f"  frame_changed={b!=a}")

    print("\nACTION5:")
    b = frame_sig(env)
    fd = env.step(GameAction.ACTION5)
    a = frame_sig(env)
    print(f"  frame_changed={b!=a}")

    # Spam all 4 direction pushes
    print("\n5 pushes each direction:")
    for act, nm in [(GameAction.ACTION1,'UP'),(GameAction.ACTION2,'DOWN'),(GameAction.ACTION3,'LEFT'),(GameAction.ACTION4,'RIGHT')]:
        b = frame_sig(env)
        fd = env.step(act)
        a = frame_sig(env)
        print(f"  {nm}: frame_changed={b!=a}")

if __name__ == "__main__":
    main()
