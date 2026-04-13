#!/usr/bin/env python3
"""Verify: on pristine L7 (no prior clicks), do pushes mutate state?
Print block positions before/after each direction."""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)
from arc_agi import Arcade
from arcengine import GameAction
import lf52_solve_final as solver

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
    print(f"\npost-L6 eq lvl={eq.whtqurkphir}")
    print("pristine:", st(eq))

    # Isolated push tests — each after an UNDO to restore
    print("\n=== ACTION1 (UP) ===")
    fd = env.step(GameAction.ACTION1)
    eq = game.ikhhdzfmarl
    print("after:", st(eq), "steps=", eq.asqvqzpfdi)

    print("\n=== ACTION7 (UNDO) ===")
    fd = env.step(GameAction.ACTION7)
    eq = game.ikhhdzfmarl
    print("after:", st(eq), "steps=", eq.asqvqzpfdi)

    print("\n=== ACTION2 (DOWN) ===")
    fd = env.step(GameAction.ACTION2)
    eq = game.ikhhdzfmarl
    print("after:", st(eq), "steps=", eq.asqvqzpfdi)

    print("\n=== ACTION7 (UNDO) ===")
    fd = env.step(GameAction.ACTION7)
    eq = game.ikhhdzfmarl
    print("after:", st(eq), "steps=", eq.asqvqzpfdi)

    print("\n=== ACTION3 (LEFT) ===")
    fd = env.step(GameAction.ACTION3)
    eq = game.ikhhdzfmarl
    print("after:", st(eq), "steps=", eq.asqvqzpfdi)

    print("\n=== ACTION4 (RIGHT) ===")
    fd = env.step(GameAction.ACTION4)
    eq = game.ikhhdzfmarl
    print("after:", st(eq), "steps=", eq.asqvqzpfdi)

    # Now try clicking the blue button FIRST thing
    env.step(GameAction.ACTION7)  # undo
    env.step(GameAction.ACTION7)  # undo to be safe
    eq = game.ikhhdzfmarl
    print("\n=== after double UNDO ===")
    print("state:", st(eq), "steps=", eq.asqvqzpfdi)

    # Click pieces at (0,1), (6,1) — cell_size 6, offset (5,5) → pixels (8,14), (44,14)
    # Before click: look at scene graph for arrow objects
    print("\n=== BEFORE click (0,1): scene graph arrows? ===")
    grid = eq.hncnfaqaddg
    arrows = []
    for yy in range(30):
        for xx in range(30):
            for o in grid.ijpoqzvnjt(xx, yy):
                if 'lgbyiaitpdi' in o.name:
                    arrows.append(((xx, yy), o.name))
    print(f"arrows: {arrows}")

    print("\n=== click (8, 14) — piece N@(0,1) cell_size=6 ===")
    fd = env.step(GameAction.ACTION6, data={'x': 8, 'y': 14})
    eq = game.ikhhdzfmarl
    print("after:", st(eq), "steps=", eq.asqvqzpfdi)
    grid = eq.hncnfaqaddg
    arrows = []
    for yy in range(30):
        for xx in range(30):
            for o in grid.ijpoqzvnjt(xx, yy):
                if 'lgbyiaitpdi' in o.name:
                    arrows.append(((xx, yy), o.name))
    print(f"arrows after click: {arrows}")

    # Check scalars on eq that might track selection
    for attr in dir(eq):
        if attr.startswith('_'): continue
        try:
            v = getattr(eq, attr)
            if callable(v): continue
            if v is not None and not isinstance(v, (int, float, bool, str, tuple, list, dict)):
                # check if it looks piece-like
                t = type(v).__name__
                if 'piece' in t.lower() or 'selected' in attr.lower() or 'pick' in attr.lower():
                    print(f"  {attr}: {t} = {v}")
        except Exception:
            pass

    # List ALL attributes that changed from "unselected" baseline
    print("\n=== Look for selection state markers ===")
    # Easier: sample many attrs, look for anything pointing at (0,1)
    for attr in dir(eq):
        if attr.startswith('_'): continue
        try:
            v = getattr(eq, attr)
            if callable(v): continue
            s_v = repr(v)
            if "'fozwvlovdui'" in s_v[:200] or "(0, 1)" in s_v[:200]:
                print(f"  {attr}: {s_v[:200]}")
        except Exception:
            pass

if __name__ == "__main__":
    main()
