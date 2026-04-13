#!/usr/bin/env python3
"""Detailed inspection of cells near right-N@(22,6) on pristine L7."""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)
from arc_agi import Arcade
from arcengine import GameAction
import lf52_solve_final as solver


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game
    for lvl in range(6):
        solver.solve_level(env, game, lvl)
    eq = game.ikhhdzfmarl
    grid = eq.hncnfaqaddg

    # Enumerate every cell in the right-N region
    print("=== Cells 18-25 x 3-9 ===")
    for y in range(3, 10):
        row = []
        for x in range(18, 26):
            objs = grid.ijpoqzvnjt(x, y)
            names = '|'.join(o.name for o in objs) or '.'
            row.append(f"{x},{y}:{names[:28]:28}")
        print('  '.join(row))
        print()

    # Entire L7 scene graph dump
    print("\n=== L7 full scene graph ===")
    for y in range(10):
        print(f"\nrow {y}:")
        for x in range(25):
            objs = grid.ijpoqzvnjt(x, y)
            if objs:
                names = [o.name for o in objs]
                print(f"  ({x},{y}): {names}")

if __name__ == "__main__":
    main()
