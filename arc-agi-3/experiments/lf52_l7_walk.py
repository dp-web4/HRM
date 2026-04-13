#!/usr/bin/env python3
"""L7 walk-through: click the blue btn, save frame at each step, try more
clicks, map out the transition sequence. Goal: reach the actual interactive
L7 playfield and see where L7's pieces render."""
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


def save_raw(env, name, scale=10):
    arr = np.array(env.observation_space.frame[0])
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


def describe(eq, label):
    st = solver.extract_state(eq)
    print(f"[{label}] lvl={st['level']} off={st['offset']} pieces={len(st['pieces'])}"
          f" blocks={len(st['pushable'])} pegs={len(st['fixed_pegs'])} walls={len(st['walls'])} steps={eq.asqvqzpfdi}")
    return st


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
    describe(eq, "post-L6 post-drain")
    arr = save_raw(env, "walk_00_post_L6")

    # Does game.zvcnglshzcx (the cwyrzsciwms button flag) show the intermission?
    try:
        print(f"zvcnglshzcx flag: {eq.zvcnglshzcx}")
    except Exception as e:
        print(f"no flag: {e}")

    # Click the blue button at (39,43)
    print("\n>>> Click blue btn (39,43)")
    fd = env.step(GameAction.ACTION6, data={'x': 39, 'y': 43})
    eq = game.ikhhdzfmarl
    describe(eq, "after btn click 1")
    save_raw(env, "walk_01_after_btn1")
    try:
        print(f"zvcnglshzcx flag: {eq.zvcnglshzcx}")
    except Exception as e:
        print(f"no flag: {e}")

    # Try clicking at pixel (8,56) (bottom-left) — the doc mentioned this
    print("\n>>> Click (8,56) bottom-left")
    fd = env.step(GameAction.ACTION6, data={'x': 8, 'y': 56})
    eq = game.ikhhdzfmarl
    describe(eq, "after (8,56)")
    save_raw(env, "walk_02_after_8_56")

    # Also try (4, 60) / (8, 60) etc — other corner positions
    for (x, y) in [(4, 60), (4, 56), (8, 60), (0, 60), (0, 63), (32, 32), (39, 43), (20, 20)]:
        print(f"\n>>> Click ({x},{y})")
        fd = env.step(GameAction.ACTION6, data={'x': x, 'y': y})
        eq = game.ikhhdzfmarl
        describe(eq, f"after ({x},{y})")
        save_raw(env, f"walk_click_{x}_{y}")
        if fd.levels_completed > 6 or fd.state.name == 'WIN':
            print("  !!!")
            return

    # Try push actions (after clicks) — maybe engine needs a push to transition
    for nm, act in [("UP", GameAction.ACTION1), ("DOWN", GameAction.ACTION2),
                    ("LEFT", GameAction.ACTION3), ("RIGHT", GameAction.ACTION4)]:
        print(f"\n>>> Push {nm}")
        fd = env.step(act)
        eq = game.ikhhdzfmarl
        describe(eq, f"after push {nm}")
        save_raw(env, f"walk_push_{nm}")
        if fd.levels_completed > 6 or fd.state.name == 'WIN':
            print("  !!!")
            return


if __name__ == "__main__":
    main()
