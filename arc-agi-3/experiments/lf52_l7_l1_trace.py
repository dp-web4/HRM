#!/usr/bin/env python3
"""Trace L1 execution step-by-step — verify pieces actually move."""
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

    eq = game.ikhhdzfmarl
    st = solver.extract_state(eq)
    print(f"L1 pristine: pieces={st['pieces']} off={st['offset']}")

    ps = solver.make_puzzle_state(st)
    jumps = solver.solve_jumps_only(ps, 1, time_limit=10)
    print(f"Solver found jumps: {jumps}")

    # Now execute manually, one click at a time, with logging
    off = st['offset']
    for jump in jumps:
        src, dst = jump
        sx, sy = src
        dx, dy = dst
        px = sx*6 + off[0] + 3
        py = sy*6 + off[1] + 3
        print(f"\n--- JUMP {src}->{dst} ---")
        pre = solver.extract_state(game.ikhhdzfmarl)
        print(f"  pre pieces: {pre['pieces']}")
        print(f"  click piece at ({px},{py})")
        fd = env.step(GameAction.ACTION6, data={'x': px, 'y': py})
        mid = solver.extract_state(game.ikhhdzfmarl)
        print(f"  after piece click: {mid['pieces']} sel={getattr(game.ikhhdzfmarl, 'aufxjsaidrw', None)}")

        half_dx = (dx - sx) // 2
        half_dy = (dy - sy) // 2
        ax = sx*6 + off[0] + half_dx*12 + 3
        ay = sy*6 + off[1] + half_dy*12 + 3
        print(f"  click arrow at ({ax},{ay})")
        fd = env.step(GameAction.ACTION6, data={'x': ax, 'y': ay})
        post = solver.extract_state(game.ikhhdzfmarl)
        print(f"  after arrow click: {post['pieces']} state={fd.state.name} levels={fd.levels_completed}")

        if fd.levels_completed > 0:
            print("  LEVEL 1 SOLVED!")
            break


if __name__ == "__main__":
    main()
