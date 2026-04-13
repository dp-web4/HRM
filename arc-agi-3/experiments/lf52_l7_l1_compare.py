#!/usr/bin/env python3
"""Compare selection state on L1 (working) vs L7 (not-working) to see if
wslpmugjcyi is really the direction list or something else."""
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


def dump(obj, prefix=""):
    if obj is None:
        print(f"{prefix}None")
        return
    for attr in dir(obj):
        if attr.startswith('_'): continue
        try:
            v = getattr(obj, attr)
            if callable(v): continue
            print(f"{prefix}{attr}: {repr(v)[:300]}")
        except Exception:
            pass


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game

    # L1: fresh
    eq = game.ikhhdzfmarl
    st = solver.extract_state(eq)
    print(f"L1 fresh pieces: {st['pieces']} offset={st['offset']}")
    # Click a piece — (1,2) at cell_size=6, offset (10,5)
    # pixel = (1*6+10+3, 2*6+5+3) = (19, 20)
    env.step(GameAction.ACTION6, data={'x': 19, 'y': 20})
    eq = game.ikhhdzfmarl
    s = sel(eq)
    print(f"\nL1 selected piece obj: {s}")
    dump(s, "  ")

    # Now try DOWN arrow click on L1 - cell (1,4) at pixel (19, 32)
    print(f"\nL1: clicking DOWN arrow at (19,32)")
    st0 = solver.extract_state(eq)
    env.step(GameAction.ACTION6, data={'x': 19, 'y': 32})
    eq = game.ikhhdzfmarl
    st1 = solver.extract_state(eq)
    print(f"L1 before: {st0['pieces']}")
    print(f"L1 after:  {st1['pieces']}")
    print(f"L1 moved: {st0['pieces']!=st1['pieces']}")

    # Reset
    while eq.asqvqzpfdi > 0:
        env.step(GameAction.ACTION7)
        eq = game.ikhhdzfmarl

    # Now inspect: what IS the selected object's structure? Click a piece and
    # enumerate all attributes including nested
    st = solver.extract_state(eq)
    p = list(st['pieces'].keys())[0]
    off = st['offset']
    px = p[0]*6 + off[0] + 3
    py = p[1]*6 + off[1] + 3
    env.step(GameAction.ACTION6, data={'x': px, 'y': py})
    eq = game.ikhhdzfmarl
    s = sel(eq)
    print(f"\n=== L1 fresh selected: piece={p} px=({px},{py}) ===")
    dump(s, "  ")
    print(f"\n wslpmugjcyi len = {len(s.wslpmugjcyi) if hasattr(s,'wslpmugjcyi') else 'N/A'}")


if __name__ == "__main__":
    main()
