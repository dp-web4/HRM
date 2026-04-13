#!/usr/bin/env python3
"""L7 camera / cell-size / click calibration probe.

Approach: engine-as-oracle. For each candidate cell-size and for each piece
position, compute the pixel coordinate under that cell-size and offset,
click it, and check:
  - state_signature changed (selection state visible via lgbyiaitpdi arrows?)
  - frame pixels changed
  - game.ikhhdzfmarl.ssoftrulee (or whatever selection var exists — we don't
    know the name; just diff the engine dict from before/after)

Also probes camera offset changes from all four push directions, and captures
frames at the raw (unupscaled) 64x64 resolution so we can inspect pixel-for-pixel
what the actual rendering is.
"""
import os, sys, time, json
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


def raw_frame(env):
    f = env.observation_space.frame
    arr = np.array(f[0])
    return arr


def save_raw(env, name, scale=10):
    arr = raw_frame(env)
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
    return arr


def engine_dict_snapshot(eq):
    """Snapshot all scalar state on eq that looks selection-ish."""
    snap = {}
    for name in dir(eq):
        if name.startswith('_'):
            continue
        try:
            v = getattr(eq, name)
        except Exception:
            continue
        if callable(v):
            continue
        if isinstance(v, (int, float, bool, str, tuple)):
            snap[name] = v
        elif v is None:
            snap[name] = None
    return snap


def diff_snaps(a, b):
    out = []
    for k in set(a) | set(b):
        if a.get(k, '_') != b.get(k, '_'):
            out.append((k, a.get(k), b.get(k)))
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

    # Refetch eq post-L6
    eq = game.ikhhdzfmarl
    print(f"\npost-L6 eq level={eq.whtqurkphir} offset={eq.hncnfaqaddg.cdpcbbnfdp}")

    # Raw frame BEFORE any action
    arr0 = save_raw(env, "00_post_L6_raw")
    print(f"raw frame shape: {arr0.shape} dtype={arr0.dtype}")
    print(f"unique vals: {sorted(set(arr0.flatten().tolist()))}")

    # Print frame as ASCII
    def print_frame(arr, label):
        print(f"\n--- {label} ---")
        for y in range(arr.shape[0]):
            row = ''.join(f'{int(v):x}' if v > 0 else '.' for v in arr[y])
            print(f"{y:2}: {row}")

    print_frame(arr0, "post-L6 raw frame (before any action)")

    # First push to kick L7 rendering
    sigb = engine_dict_snapshot(eq)
    fd = env.step(GameAction.ACTION1)
    eq = game.ikhhdzfmarl
    arr1 = save_raw(env, "01_after_push_up")
    print(f"\nafter UP push: offset={eq.hncnfaqaddg.cdpcbbnfdp} level={eq.whtqurkphir}")
    print_frame(arr1, "after ACTION1 (UP)")

    state = solver.extract_state(eq)
    print(f"\npieces={state['pieces']}")
    print(f"offset={state['offset']}")
    print(f"blocks={sorted(state['pushable'])}")

    # Look at where pieces should be rendered at each candidate cell_size and offset
    off = state['offset']
    print(f"\n=== Cell size calibration ===")
    for cs in [4, 5, 6, 7, 8]:
        print(f"cs={cs}:")
        for (x, y), n in state['pieces'].items():
            px = x*cs + off[0]
            py = y*cs + off[1]
            print(f"  {n}@({x},{y}) -> pixel({px},{py})"
                  f" {'(visible)' if 0<=px<64 and 0<=py<64 else '(OFF-SCREEN)'}"
                  f" value@px: {int(arr1[py,px]) if 0<=px<64 and 0<=py<64 else 'N/A'}")

    # Try UNDO to restore pristine, then click each piece at cell_size=6 and cell_size=8
    env.step(GameAction.ACTION7)
    eq = game.ikhhdzfmarl
    state = solver.extract_state(eq)
    print(f"\nAfter UNDO: pieces={state['pieces']} offset={state['offset']}")
    if len(state['pieces']) != 3:
        print("UNDO didn't restore, re-pushing to get to L7 state anyway")

    base_arr = raw_frame(env)
    base_snap = engine_dict_snapshot(eq)

    # Show which coord-systems produce different results
    print(f"\n=== Click scan: each piece, each cell_size ===")
    for (gx, gy), pname in list(state['pieces'].items()):
        for cs in [6, 7, 8]:
            off = state['offset']
            px = gx*cs + off[0] + cs//2
            py = gy*cs + off[1] + cs//2
            if not (0 <= px < 64 and 0 <= py < 64):
                print(f"  {pname}@({gx},{gy}) cs={cs} px({px},{py}) OFF-SCREEN")
                continue
            sigb = engine_dict_snapshot(eq)
            arrb = raw_frame(env).copy()
            fd = env.step(GameAction.ACTION6, data={'x': px, 'y': py})
            eq = game.ikhhdzfmarl
            siga = engine_dict_snapshot(eq)
            arra = raw_frame(env)
            sd = diff_snaps(sigb, siga)
            pixdiff = int(np.sum(arrb != arra))
            interesting = [s for s in sd if s[0] not in ('asqvqzpfdi',)]
            print(f"  {pname}@({gx},{gy}) cs={cs} px({px},{py}):"
                  f" engine-diffs={len(sd)} (non-step={len(interesting)})"
                  f" pixdiff={pixdiff}")
            for s in interesting[:6]:
                print(f"      {s[0]}: {s[1]!r} -> {s[2]!r}")
            # click empty to deselect
            env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})
            eq = game.ikhhdzfmarl

    # Also: do ONE push (UP), then probe again — maybe the piece arrows only
    # appear after selection, and we need to inspect the frame to see if any
    # arrow sprites showed up.
    print(f"\n=== Scan frame for lgbyiaitpdi (arrow sprites) after selecting each piece ===")
    state = solver.extract_state(eq)
    for (gx, gy), pname in list(state['pieces'].items()):
        off = state['offset']
        for cs in [6, 7, 8]:
            px = gx*cs + off[0] + cs//2
            py = gy*cs + off[1] + cs//2
            if not (0 <= px < 64 and 0 <= py < 64):
                continue
            # Click to select
            env.step(GameAction.ACTION6, data={'x': px, 'y': py})
            # Inspect scene graph for arrow objects
            eq = game.ikhhdzfmarl
            grid = eq.hncnfaqaddg
            arrow_positions = []
            for yy in range(30):
                for xx in range(30):
                    for o in grid.ijpoqzvnjt(xx, yy):
                        if 'lgbyiaitpdi' in o.name:
                            arrow_positions.append(((xx, yy), o.name))
            if arrow_positions:
                print(f"  !!! {pname}@({gx},{gy}) cs={cs}: {len(arrow_positions)} arrows {arrow_positions[:8]}")
            # deselect
            env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})
            eq = game.ikhhdzfmarl

    print(f"\n=== Done ===")
    save_raw(env, "99_final")

if __name__ == "__main__":
    main()
