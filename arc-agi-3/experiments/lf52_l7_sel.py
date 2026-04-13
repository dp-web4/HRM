#!/usr/bin/env python3
"""Find what 'piece selected' looks like on eq, by diffing a known-good
click. Strategy: on L1 or L2, where we KNOW the solver produces successful
clicks (execution of jumps works), check if any eq attribute changes after
selecting a piece."""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)
from arc_agi import Arcade
from arcengine import GameAction
import lf52_solve_final as solver


def snap(eq):
    """Snapshot all non-callable attrs as repr strings."""
    out = {}
    for name in dir(eq):
        if name.startswith('_'): continue
        try:
            v = getattr(eq, name)
            if callable(v): continue
            out[name] = repr(v)[:500]
        except Exception:
            pass
    return out


def diff(a, b):
    out = []
    for k in set(a)|set(b):
        if a.get(k) != b.get(k):
            out.append((k, a.get(k, '_'), b.get(k, '_')))
    return out


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game

    # L1: fresh state
    eq = game.ikhhdzfmarl
    print(f"L1 fresh: level={eq.whtqurkphir} pieces_grid_off={eq.hncnfaqaddg.cdpcbbnfdp}")
    st = solver.extract_state(eq)
    print(f"L1 pieces: {st['pieces']}")

    # Pick first piece, compute click coord
    p = list(st['pieces'].keys())[0]
    cs = 6
    off = st['offset']
    px = p[0]*cs + off[0] + cs//2
    py = p[1]*cs + off[1] + cs//2
    print(f"Clicking piece {p} at px({px},{py})")

    sb = snap(eq)
    fd = env.step(GameAction.ACTION6, data={'x': px, 'y': py})
    eq = game.ikhhdzfmarl
    sa = snap(eq)
    d = diff(sb, sa)
    print(f"\n{len(d)} diffs after click:")
    for k, a, b in d:
        print(f"  {k}:")
        print(f"    before: {a[:300]}")
        print(f"    after:  {b[:300]}")

    # Also check for arrows in scene graph
    grid = eq.hncnfaqaddg
    arrows = [((x,y), o.name) for y in range(30) for x in range(30) for o in grid.ijpoqzvnjt(x,y) if 'lgbyiaitpdi' in o.name]
    print(f"\narrows: {arrows}")


if __name__ == "__main__":
    main()
