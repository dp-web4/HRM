#!/usr/bin/env python3
"""Exhaustively diff ALL attributes on eq + its children between different
click positions. Find the one that stores WHICH piece is selected."""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)
from arc_agi import Arcade
from arcengine import GameAction
import lf52_solve_final as solver
import numpy as np


def deep_snap(obj, path="", depth=0, max_depth=3, visited=None):
    if visited is None: visited = set()
    if depth > max_depth: return {}
    out = {}
    for a in dir(obj):
        if a.startswith('_'): continue
        try:
            v = getattr(obj, a)
        except Exception:
            continue
        if callable(v): continue
        p = f"{path}.{a}" if path else a
        try:
            r = repr(v)
            if len(r) < 500:
                out[p] = r
        except Exception:
            continue
        if id(v) in visited: continue
        visited.add(id(v))
        t = type(v).__name__
        if t.startswith('bjwicx') or t == 'equnaohchtj' or 'qoljprchp' in t or 'wwnfrkbz' in t or 'fozwvlov' in t:
            out.update(deep_snap(v, p, depth+1, max_depth, visited))
    return out


def diff(a, b):
    return [(k, a.get(k, '_'), b.get(k, '_')) for k in set(a)|set(b) if a.get(k) != b.get(k)]


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game
    for lvl in range(6):
        solver.solve_level(env, game, lvl)
    eq = game.ikhhdzfmarl

    # Snapshot fresh L7 (no selection)
    s0 = deep_snap(eq, "eq")
    print(f"fresh snap: {len(s0)} attrs")

    # Click N@(0,1)
    env.step(GameAction.ACTION6, data={'x': 8, 'y': 14})
    eq = game.ikhhdzfmarl
    s1 = deep_snap(eq, "eq")
    d1 = diff(s0, s1)
    print(f"\n=== diffs after click (8,14) N@(0,1): {len(d1)} ===")
    for k, a, b in d1[:40]:
        print(f"  {k}: {a[:80]} -> {b[:80]}")

    # Undo, click R@(6,1)
    env.step(GameAction.ACTION7)
    env.step(GameAction.ACTION7)
    eq = game.ikhhdzfmarl
    s0b = deep_snap(eq, "eq")

    env.step(GameAction.ACTION6, data={'x': 44, 'y': 14})
    eq = game.ikhhdzfmarl
    s2 = deep_snap(eq, "eq")
    d2 = diff(s0b, s2)
    print(f"\n=== diffs after click (44,14) R@(6,1): {len(d2)} ===")
    for k, a, b in d2[:40]:
        print(f"  {k}: {a[:80]} -> {b[:80]}")

    # Click at off-screen px
    env.step(GameAction.ACTION7)
    env.step(GameAction.ACTION7)
    s0c = deep_snap(eq, "eq")

    env.step(GameAction.ACTION6, data={'x': 137, 'y': 41})
    eq = game.ikhhdzfmarl
    s3 = deep_snap(eq, "eq")
    d3 = diff(s0c, s3)
    print(f"\n=== diffs after click (137,41) off-screen: {len(d3)} ===")
    for k, a, b in d3[:40]:
        print(f"  {k}: {a[:80]} -> {b[:80]}")

    # diff between clicking N and R
    diff_nr = diff(s1, s2)
    print(f"\n=== diffs BETWEEN click-N and click-R: {len(diff_nr)} ===")
    for k, a, b in diff_nr[:40]:
        print(f"  {k}: {a[:80]} -> {b[:80]}")


if __name__ == "__main__":
    main()
