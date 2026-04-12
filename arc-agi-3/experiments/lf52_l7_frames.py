#!/usr/bin/env python3
"""Save frames at each step after reaching L7 to visually identify intro screen."""
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

OUT = "/tmp/lf52_l7_frames"
os.makedirs(OUT, exist_ok=True)

def save(env, name):
    frame = np.array(env.observation_space.frame[0])
    h, w = frame.shape
    s = 10
    img = Image.new('RGB', (w*s, h*s))
    pix = img.load()
    for y in range(h):
        for x in range(w):
            c = solver.PALETTE.get(int(frame[y,x]), (0,0,0))
            for dy in range(s):
                for dx in range(s):
                    pix[x*s+dx, y*s+dy] = c
    img.save(f"{OUT}/{name}.png")

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('lf52-271a04aa')
    env.reset()
    game = env._game
    for lvl in range(6):
        solver.solve_level(env, game, lvl)

    save(env, "00_after_L6")
    # Refetch to trigger L7 transition
    eq = game.ikhhdzfmarl
    save(env, "01_after_refetch")

    # Try: push UP once
    env.step(GameAction.ACTION1)
    save(env, "02_push_up")
    # Another push
    env.step(GameAction.ACTION2)
    save(env, "03_push_down")
    # ACTION5
    env.step(GameAction.ACTION5)
    save(env, "04_action5")
    # ACTION7
    env.step(GameAction.ACTION7)
    save(env, "05_action7")

    # Click at (8,14) (pseudo-piece)
    env.step(GameAction.ACTION6, data={'x':8,'y':14})
    save(env, "06_click_8_14")
    # Click at (8,56)
    env.step(GameAction.ACTION6, data={'x':8,'y':56})
    save(env, "07_click_8_56")

if __name__ == "__main__":
    main()
    print(f"Saved to {OUT}")

