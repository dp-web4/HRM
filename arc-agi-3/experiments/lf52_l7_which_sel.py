#!/usr/bin/env python3
"""Which piece does clicking at various pixels actually select? Inspect the
selected piece's grid position via the scene graph."""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)
from arc_agi import Arcade
from arcengine import GameAction
import lf52_solve_final as solver


def find_piece_grid_pos_for_obj(eq, target_id):
    grid = eq.hncnfaqaddg
    for y in range(30):
        for x in range(30):
            for o in grid.ijpoqzvnjt(x, y):
                if id(o) == target_id:
                    return (x, y), o.name
    return None, None


def get_piece_obj_at(eq, grid_pos):
    grid = eq.hncnfaqaddg
    for o in grid.ijpoqzvnjt(*grid_pos):
        if 'fozwvlovdui' in o.name:
            return o
    return None


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game
    for lvl in range(6):
        solver.solve_level(env, game, lvl)
    eq = game.ikhhdzfmarl

    # The selected piece is stored on aufxjsaidrw — it's a 'bjwicxpwbxc' object.
    # That's probably a wrapper around the scene graph piece. Let's find its real
    # grid position by comparing ids.

    # First: get object ids of each piece
    st = solver.extract_state(eq)
    ids = {}
    for p in st['pieces']:
        obj = get_piece_obj_at(eq, p)
        if obj:
            ids[id(obj)] = p
    print(f"piece ids: {ids}")

    # But aufxjsaidrw is type bjwicxpwbxc which wraps the piece.
    # Check its oegtnpbqims attr (scene graph ref) and look for any attr that
    # points to the piece.

    # Click at various pixels, inspect what aufxjsaidrw wraps
    def inspect_sel(label):
        sel = getattr(eq, 'aufxjsaidrw', None)
        if sel is None:
            print(f"  [{label}] sel=None")
            return
        print(f"  [{label}] sel type={type(sel).__name__}")
        for a in dir(sel):
            if a.startswith('_'): continue
            try:
                v = getattr(sel, a)
                if callable(v): continue
                r = repr(v)[:100]
                if 'fozwvlovdui' in r or 'object' in r:
                    print(f"    {a}: {r}")
            except Exception:
                pass

    for (px, py, label) in [
        (8, 14, "N@(0,1)"),
        (44, 14, "R@(6,1)"),
        (137, 41, "off-screen right-N"),
        (63, 41, "right edge y=41"),
        (0, 0, "corner"),
        (63, 63, "bot-right corner"),
    ]:
        # Reset selection first by clicking empty
        env.step(GameAction.ACTION7)
        # Actually use UNDO to not waste steps — or skip: clicks that don't
        # hit anything leave the last selection. Let me just step UNDO.
        env.step(GameAction.ACTION7)
        eq = game.ikhhdzfmarl
        # Click
        env.step(GameAction.ACTION6, data={'x': px, 'y': py})
        eq = game.ikhhdzfmarl
        print(f"\nClicked ({px},{py}) [{label}] — selection:")
        inspect_sel(label)
        # Check if click actually matched something via state diff of all
        # piece-like attrs on eq

        # Brute: scan every attr on eq looking for refs
        for a in dir(eq):
            if a.startswith('_'): continue
            try:
                v = getattr(eq, a)
                if callable(v): continue
                r = repr(v)
                if 'bjwicxpwbxc' in r:
                    print(f"    eq.{a}: {r[:150]}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
