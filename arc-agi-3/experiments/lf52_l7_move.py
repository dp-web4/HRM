#!/usr/bin/env python3
"""Test moving pieces on L7 now that we know clicks work (aufxjsaidrw
signals selection)."""
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


def st(eq):
    s = solver.extract_state(eq)
    return dict(pieces=dict(s['pieces']), blocks=sorted(s['pushable']),
                pegs=sorted(s['fixed_pegs']), off=s['offset'])


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game

    for lvl in range(6):
        solver.solve_level(env, game, lvl)

    eq = game.ikhhdzfmarl
    print(f"L7 initial pieces: {st(eq)['pieces']}")

    # Select N@(0,1) at px (8,14)
    env.step(GameAction.ACTION6, data={'x': 8, 'y': 14})
    eq = game.ikhhdzfmarl
    print(f"Selected: {sel(eq) is not None}")

    # Try clicking at each of the 4 potential arrow targets
    # DOWN: jump (0,1)->(0,3), middle (0,2) has peg, landing (0,3) is walkable
    # UP: jump (0,1)->(0,-1), landing off-grid INVALID
    # LEFT: (-2,1) off-grid
    # RIGHT: (2,1) middle (1,1), no piece or peg at (1,1) - INVALID
    # So only DOWN should work
    # Arrow click position for DOWN: cell (0,3) = pixel (8, 26)
    st0 = st(eq)
    print(f"before arrow click: pieces={st0['pieces']}")
    fd = env.step(GameAction.ACTION6, data={'x': 8, 'y': 26})
    eq = game.ikhhdzfmarl
    st1 = st(eq)
    print(f"after DOWN-arrow click (8,26): pieces={st1['pieces']} steps={eq.asqvqzpfdi} selected={sel(eq) is not None}")
    if st1['pieces'] != st0['pieces']:
        print("!!! PIECE MOVED !!!")

    # Regardless — let's see what the scene graph says about arrows after selection
    env.step(GameAction.ACTION7)  # undo
    env.step(GameAction.ACTION7)
    eq = game.ikhhdzfmarl
    print(f"\nAfter 2x UNDO: pieces={st(eq)['pieces']} steps={eq.asqvqzpfdi}")

    # Re-select, look for valid directions
    env.step(GameAction.ACTION6, data={'x': 8, 'y': 14})
    eq = game.ikhhdzfmarl
    selected = sel(eq)
    print(f"selected object: {selected}")
    if selected:
        print(f"  type: {type(selected).__name__}")
        for attr in dir(selected):
            if attr.startswith('_'): continue
            try:
                v = getattr(selected, attr)
                if callable(v): continue
                r = repr(v)[:200]
                print(f"  {attr}: {r}")
            except Exception as e:
                pass

    # Check all 4 direction arrows — click each and see what happens
    for name, (arrow_x, arrow_y) in [('UP',(8,2)), ('DOWN',(8,26)), ('LEFT',(-4,14)), ('RIGHT',(20,14))]:
        # We can't click negative or OOB, skip those
        if arrow_x < 0 or arrow_x > 63 or arrow_y < 0 or arrow_y > 63:
            print(f"  {name} arrow at ({arrow_x},{arrow_y}) OOB")
            continue
        # Re-select piece first
        env.step(GameAction.ACTION7)
        env.step(GameAction.ACTION6, data={'x': 8, 'y': 14})
        eq = game.ikhhdzfmarl
        st0 = st(eq)
        fd = env.step(GameAction.ACTION6, data={'x': arrow_x, 'y': arrow_y})
        eq = game.ikhhdzfmarl
        st1 = st(eq)
        print(f"  {name} @({arrow_x},{arrow_y}): pieces moved? {st0['pieces']!=st1['pieces']}"
              f" new pieces={st1['pieces']}")


if __name__ == "__main__":
    main()
