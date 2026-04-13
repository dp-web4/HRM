#!/usr/bin/env python3
"""Find valid click coordinates for L7 pieces using aufxjsaidrw as the
selection signal. Scan cell sizes, offsets, and pixel regions."""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)
from arc_agi import Arcade
from arcengine import GameAction
import lf52_solve_final as solver


def sel(eq):
    return getattr(eq, 'aufxjsaidrw', None)


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game

    for lvl in range(6):
        solver.solve_level(env, game, lvl)

    eq = game.ikhhdzfmarl
    st = solver.extract_state(eq)
    print(f"L7 pieces: {st['pieces']} offset={st['offset']}")

    # Baseline: no click, should be None
    print(f"baseline aufxjsaidrw: {sel(eq)}")

    # First click a piece at the solver's formula location
    piece = (0, 1)
    off = st['offset']
    cs = 6
    px = piece[0]*cs + off[0] + cs//2
    py = piece[1]*cs + off[1] + cs//2
    print(f"\nclick at ({px},{py}) cs=6:")
    fd = env.step(GameAction.ACTION6, data={'x': px, 'y': py})
    eq = game.ikhhdzfmarl
    print(f"  aufxjsaidrw: {sel(eq)}")

    # UNDO
    env.step(GameAction.ACTION7)
    eq = game.ikhhdzfmarl

    # Now scan: for piece (0,1), try every pixel in a 20x20 region around formula
    print(f"\n=== Scan pixels to find N@(0,1) select coords ===")
    hits = []
    base_sel = sel(eq)
    for y in range(0, 30):
        for x in range(0, 30):
            env.step(GameAction.ACTION6, data={'x': x, 'y': y})
            eq = game.ikhhdzfmarl
            s = sel(eq)
            if s is not None and s != base_sel:
                hits.append((x, y, s))
                print(f"  ({x},{y}) -> {s}")
            # Reset selection by clicking empty
            env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})
            eq = game.ikhhdzfmarl
            if eq.asqvqzpfdi > 400:
                print(f"STEP LIMIT near {eq.asqvqzpfdi}")
                return
    print(f"\nHits for N@(0,1): {len(hits)}")

    # Now: what about piece at (6,1)? Scan roughly right-side of frame
    print(f"\n=== Scan pixels 30-60 x 0-30 for R@(6,1) ===")
    # restore
    env.step(GameAction.ACTION7)
    eq = game.ikhhdzfmarl
    hits2 = []
    for y in range(0, 30):
        for x in range(30, 64):
            env.step(GameAction.ACTION6, data={'x': x, 'y': y})
            eq = game.ikhhdzfmarl
            s = sel(eq)
            if s is not None:
                hits2.append((x, y, s))
                print(f"  ({x},{y}) -> {s}")
            env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})
            eq = game.ikhhdzfmarl
            if eq.asqvqzpfdi > 1500:
                print(f"STEP LIMIT {eq.asqvqzpfdi}")
                return
    print(f"\nHits for R@(6,1): {len(hits2)}")


if __name__ == "__main__":
    main()
