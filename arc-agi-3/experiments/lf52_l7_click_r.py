#!/usr/bin/env python3
"""Clean test: on fresh L7, click ONLY R@(6,1) at (44,14). Does it select?"""
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
    print(f"sel baseline: {getattr(eq,'aufxjsaidrw',None)}")

    # Click R@(6,1) at (44, 14)
    env.step(GameAction.ACTION6, data={'x': 44, 'y': 14})
    eq = game.ikhhdzfmarl
    print(f"sel after (44,14): {getattr(eq,'aufxjsaidrw',None)}")

    # Now: try jump R. Red has only self-jumps possible if same-colored middle exists.
    # On pristine: middle candidates for R@(6,1):
    # UP: (6,0) empty. DOWN: (6,2) has peg (dgxfozncuiz). LEFT: (5,1) empty. RIGHT: (7,1) empty.
    # So DOWN is the only option (jump over peg at (6,2) to landing (6,3)?)
    # (6,3) has kraubslpehi-3 which is a wall variant — is it a valid landing?
    # rule: len==1 and 'hupkpseyuim' in name → No. len==2 and hupkpseyuim2 in names → No.
    # So (6,3) is NOT a valid landing. Red has NO moves.
    # But let's test empirically.
    print("\n=== Try red's 4 directions ===")
    # UP
    env.step(GameAction.ACTION7)  # undo
    env.step(GameAction.ACTION6, data={'x': 44, 'y': 14})
    st0 = solver.extract_state(game.ikhhdzfmarl)
    env.step(GameAction.ACTION6, data={'x': 44, 'y': 2})  # UP arrow
    st1 = solver.extract_state(game.ikhhdzfmarl)
    print(f"R UP:   moved={st0['pieces']!=st1['pieces']}")

    env.step(GameAction.ACTION7)
    env.step(GameAction.ACTION7)
    env.step(GameAction.ACTION6, data={'x': 44, 'y': 14})
    st0 = solver.extract_state(game.ikhhdzfmarl)
    env.step(GameAction.ACTION6, data={'x': 44, 'y': 26})  # DOWN arrow
    st1 = solver.extract_state(game.ikhhdzfmarl)
    print(f"R DOWN: moved={st0['pieces']!=st1['pieces']}")

    # Actually — try all cell_size options too
    while game.ikhhdzfmarl.asqvqzpfdi > 0:
        env.step(GameAction.ACTION7)
    print(f"\n=== Clean reset steps={game.ikhhdzfmarl.asqvqzpfdi} ===")

    # For each cell_size, click at that cell_size's prediction of (6,1)
    print("=== click each cs for piece (6,1) ===")
    for cs in [4, 5, 6, 7, 8, 10]:
        px = 6*cs + 5 + cs//2
        py = 1*cs + 5 + cs//2
        if not (0<=px<64 and 0<=py<64):
            print(f"  cs={cs} ({px},{py}) off-screen")
            continue
        # Get baseline sel
        eq = game.ikhhdzfmarl
        base_sel = getattr(eq,'aufxjsaidrw',None)
        env.step(GameAction.ACTION6, data={'x': px, 'y': py})
        eq = game.ikhhdzfmarl
        new_sel = getattr(eq,'aufxjsaidrw',None)
        changed = base_sel is None and new_sel is not None
        print(f"  cs={cs} ({px},{py}): base_sel={base_sel is not None}, new_sel={new_sel is not None}, changed={changed}")
        env.step(GameAction.ACTION7)

if __name__ == "__main__":
    main()
