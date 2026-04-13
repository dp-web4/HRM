#!/usr/bin/env python3
"""
Key insight: the selected piece has attribute `wslpmugjcyi` which lists
valid directions. On pristine L7 for N@(0,1), it's empty — zero moves.

This is the AUTHORITATIVE source of which jumps the engine allows.
Previously the solver computed valid jumps from its own model.
Use the engine's own list!

Plan:
1. After each state transition, click each piece and read wslpmugjcyi.
2. Compare to the solver's get_valid_jumps() — find any discrepancies.
3. Use engine's view as ground truth and drive a search over it.
"""
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


def dirs_for_click(env, game, cell, cell_size=6, offset=None):
    eq = game.ikhhdzfmarl
    if offset is None:
        offset = eq.hncnfaqaddg.cdpcbbnfdp
    # Deselect first
    env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})
    px = cell[0]*cell_size + offset[0] + cell_size//2
    py = cell[1]*cell_size + offset[1] + cell_size//2
    if not (0 <= px < 64 and 0 <= py < 64):
        return None, "off-screen"
    env.step(GameAction.ACTION6, data={'x': px, 'y': py})
    eq = game.ikhhdzfmarl
    s = sel(eq)
    if s is None:
        return None, "no-select"
    return getattr(s, 'wslpmugjcyi', None), "ok"


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game

    for lvl in range(6):
        solver.solve_level(env, game, lvl)

    eq = game.ikhhdzfmarl
    st = solver.extract_state(eq)
    print(f"Pristine L7 pieces: {st['pieces']}")

    # Query each piece
    for pos in list(st['pieces'].keys()):
        dirs, status = dirs_for_click(env, game, pos)
        print(f"  piece@{pos}: status={status} dirs={dirs}")

    # Now push UP and re-check
    env.step(GameAction.ACTION7)  # clear any selection/reset
    env.step(GameAction.ACTION1)  # push UP
    eq = game.ikhhdzfmarl
    st = solver.extract_state(eq)
    print(f"\nAfter UP push: blocks={sorted(st['pushable'])}")
    for pos in list(st['pieces'].keys()):
        dirs, status = dirs_for_click(env, game, pos)
        print(f"  piece@{pos}: status={status} dirs={dirs}")

    # After LEFT LEFT UP UP
    env.step(GameAction.ACTION7)
    env.step(GameAction.ACTION3)  # LEFT
    env.step(GameAction.ACTION3)  # LEFT
    env.step(GameAction.ACTION1)  # UP
    env.step(GameAction.ACTION1)  # UP
    eq = game.ikhhdzfmarl
    st = solver.extract_state(eq)
    print(f"\nAfter L L U U: blocks={sorted(st['pushable'])}")
    for pos in list(st['pieces'].keys()):
        dirs, status = dirs_for_click(env, game, pos)
        print(f"  piece@{pos}: status={status} dirs={dirs}")

    # Back to fresh L7
    while eq.asqvqzpfdi > 0:
        env.step(GameAction.ACTION7)
        eq = game.ikhhdzfmarl
    print(f"\nReset steps={eq.asqvqzpfdi}")
    st = solver.extract_state(eq)
    print(f"pieces: {st['pieces']} blocks: {sorted(st['pushable'])}")

    # Now look at what dir objects contain — decode them
    # Select N@(0,1) — actually after LLUU should be a valid jump
    env.step(GameAction.ACTION3)
    env.step(GameAction.ACTION3)
    env.step(GameAction.ACTION1)
    env.step(GameAction.ACTION1)
    env.step(GameAction.ACTION3)  # L
    env.step(GameAction.ACTION3)  # L
    env.step(GameAction.ACTION3)  # L
    eq = game.ikhhdzfmarl
    st = solver.extract_state(eq)
    print(f"\nAfter L L U U L L L: blocks={sorted(st['pushable'])}")
    for pos in list(st['pieces'].keys()):
        dirs, status = dirs_for_click(env, game, pos)
        print(f"  piece@{pos}: status={status} dirs={dirs}")
        if dirs:
            for d in dirs:
                print(f"    dir obj: {d} type={type(d).__name__}")
                for attr in dir(d):
                    if attr.startswith('_'): continue
                    try:
                        v = getattr(d, attr)
                        if callable(v): continue
                        print(f"      {attr}: {repr(v)[:200]}")
                    except Exception:
                        pass

if __name__ == "__main__":
    main()
