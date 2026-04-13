#!/usr/bin/env python3
"""L7 probe: the post-L6 frame shows an intermission screen with a blue button
sprite (colors 11='b'/12='c') in the middle-right area. Hypothesis: clicking
it advances to real L7. Also: scan ALL sprite-like rectangles in the frame,
click each, report state diffs.
"""
import os, sys
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.path.insert(0, "/mnt/c/exe/projects/ai-agents/SAGE/arc-agi-3/experiments")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
from PIL import Image
import lf52_solve_final as solver

OUT = "/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/lf52/l7_probe"
os.makedirs(OUT, exist_ok=True)


def raw(env):
    return np.array(env.observation_space.frame[0])


def save_raw(env, name, scale=10):
    arr = raw(env)
    h, w = arr.shape
    img = Image.new('RGB', (w*scale, h*scale))
    pix = img.load()
    for y in range(h):
        for x in range(w):
            c = solver.PALETTE.get(int(arr[y, x]), (0, 0, 0))
            for dy in range(scale):
                for dx in range(scale):
                    pix[x*scale+dx, y*scale+dy] = c
    img.save(f"{OUT}/{name}.png")


def state_sig(eq):
    """State signature from scene graph."""
    grid = eq.hncnfaqaddg
    out = {}
    for y in range(30):
        for x in range(30):
            names = tuple(sorted(o.name for o in grid.ijpoqzvnjt(x, y)))
            if names:
                out[(x, y)] = names
    return out


def arrow_positions(eq):
    grid = eq.hncnfaqaddg
    out = []
    for y in range(30):
        for x in range(30):
            for o in grid.ijpoqzvnjt(x, y):
                if 'lgbyiaitpdi' in o.name:
                    out.append(((x, y), o.name))
    return out


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game

    for lvl in range(6):
        fd = solver.solve_level(env, game, lvl)
        if fd is None or fd.levels_completed <= lvl:
            print(f"FAIL at L{lvl+1}")
            return

    eq = game.ikhhdzfmarl
    print(f"\npost-L6 level={eq.whtqurkphir} offset={eq.hncnfaqaddg.cdpcbbnfdp}")

    arr0 = raw(env)
    save_raw(env, "bbtn_00_post_L6")

    # Find all blue (11, 12) pixels — map possible button regions
    blue_mask = (arr0 == 11) | (arr0 == 12)
    ys, xs = np.where(blue_mask)
    print(f"blue pixels: {len(xs)}")
    if len(xs):
        print(f"  x range {xs.min()}-{xs.max()}, y range {ys.min()}-{ys.max()}")
        # Cluster: find rectangles
        visited = np.zeros_like(blue_mask)
        clusters = []
        for y in range(arr0.shape[0]):
            for x in range(arr0.shape[1]):
                if blue_mask[y, x] and not visited[y, x]:
                    # flood fill
                    stack = [(y, x)]
                    pts = []
                    while stack:
                        cy, cx = stack.pop()
                        if cy < 0 or cy >= arr0.shape[0] or cx < 0 or cx >= arr0.shape[1]:
                            continue
                        if visited[cy, cx] or not blue_mask[cy, cx]:
                            continue
                        visited[cy, cx] = True
                        pts.append((cy, cx))
                        stack.extend([(cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)])
                    if len(pts) >= 3:
                        clusters.append(pts)
        print(f"  clusters: {len(clusters)}")
        for i, cl in enumerate(clusters):
            ys_c = [p[0] for p in cl]
            xs_c = [p[1] for p in cl]
            cy = sum(ys_c)/len(ys_c)
            cx = sum(xs_c)/len(xs_c)
            print(f"  cluster {i}: size={len(cl)} center=({cx:.0f},{cy:.0f}) bbox=({min(xs_c)},{min(ys_c)})-({max(xs_c)},{max(ys_c)})")

    # Click each blue cluster center
    base_sig = state_sig(eq)
    print(f"\nBase state sig size: {len(base_sig)}")
    print(f"Base pieces (scene graph):")
    for k, v in base_sig.items():
        if any('fozwvlovdui' in n for n in v):
            print(f"  {k}: {v}")

    clicked_positions = []
    for i, cl in enumerate(clusters if len(xs) else []):
        ys_c = [p[0] for p in cl]
        xs_c = [p[1] for p in cl]
        cx = int(sum(xs_c)/len(xs_c))
        cy = int(sum(ys_c)/len(ys_c))
        clicked_positions.append((cx, cy, f"blue_cluster_{i}"))

    # Also: click at each L7 piece position under cell_size=6 assuming they
    # exist as real pieces
    # And: click at every grid cell under cell_size=6 within the 64x64 frame
    # (those pixels inside the frame)

    print(f"\n=== Click each blue cluster ===")
    for (cx, cy, label) in clicked_positions:
        sb = state_sig(eq)
        arb = raw(env).copy()
        fd = env.step(GameAction.ACTION6, data={'x': cx, 'y': cy})
        eq = game.ikhhdzfmarl
        sa = state_sig(eq)
        ara = raw(env)
        diff = [(k, sb.get(k), sa.get(k)) for k in set(sb)|set(sa) if sb.get(k) != sa.get(k)]
        pixdiff = int(np.sum(arb != ara))
        print(f"  {label} ({cx},{cy}): state_diffs={len(diff)} pixdiff={pixdiff} game_state={fd.state.name} levels={fd.levels_completed}")
        for d in diff[:6]:
            print(f"    {d}")
        if pixdiff > 200:
            save_raw(env, f"bbtn_click_{label}")
        if fd.levels_completed > 6 or fd.state.name == 'WIN':
            print("  !!! ADVANCE / WIN !!!")
            return

    # === Click every 4x4 block of the frame (systematic scan) ===
    # too many — only scan px where color is nonzero AND not background
    print(f"\n=== Systematic scan of non-background pixels ===")
    bg = 10  # blue background
    sb0 = state_sig(eq)
    interesting_clicks = []
    for y in range(0, 64, 3):
        for x in range(0, 64, 3):
            v = int(arr0[y, x])
            if v in (10, 11):  # skip background blue
                continue
            interesting_clicks.append((x, y, v))
    print(f"  {len(interesting_clicks)} candidates")

    # Sample every 4th
    for x, y, v in interesting_clicks[::2]:
        sb = state_sig(eq)
        fd = env.step(GameAction.ACTION6, data={'x': x, 'y': y})
        eq = game.ikhhdzfmarl
        sa = state_sig(eq)
        diff = [(k, sb.get(k), sa.get(k)) for k in set(sb)|set(sa) if sb.get(k) != sa.get(k)]
        arrows = arrow_positions(eq)
        if arrows or len(diff) > 1:
            print(f"  click({x},{y}) val={v}: diffs={len(diff)} arrows={len(arrows)}")
            for d in diff[:4]:
                print(f"    {d}")
            if arrows:
                print(f"    ARROWS: {arrows[:6]}")
                save_raw(env, f"bbtn_click_{x}_{y}_arrows")
        if fd.levels_completed > 6 or fd.state.name == 'WIN':
            print(f"  !!! ADVANCE from click({x},{y}) !!!")
            save_raw(env, f"bbtn_advance_{x}_{y}")
            return

    # Still stuck? Print final state
    print("\n=== Final state ===")
    state = solver.extract_state(eq)
    print(f"pieces: {state['pieces']}")
    print(f"offset: {state['offset']}")
    print(f"steps: {eq.asqvqzpfdi}")
    print(f"level: {eq.whtqurkphir}")
    save_raw(env, "bbtn_99_final")


if __name__ == "__main__":
    main()
