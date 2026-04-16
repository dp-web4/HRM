#!/usr/bin/env python3
"""L6: brute-force reach changes under all click combinations of c and f."""
import os, sys, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
sys.path.insert(0, "arc-agi-3/experiments")
from dc22_solve_final import player_reachable_cells, save_frame

VIS = "/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/dc22"

def setup(arcade):
    env = arcade.make('dc22-4c9bff3e')
    env.reset()
    with open(f"{VIS}/solutions.json") as f:
        raw = json.load(f)
    am = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
          3: GameAction.ACTION3, 4: GameAction.ACTION4,
          6: GameAction.ACTION6}
    for lvl_idx in range(5):
        for m in raw[lvl_idx]:
            env.step(am[m['action']], data=m.get('data', {}))
    return env, am

def render_map(game, reach):
    print("  reach map (x=6..40, y=44..62):")
    for y in range(44, 63, 2):
        row = ''
        for x in range(6, 42, 2):
            if (x,y) in reach:
                row += '.'
            else:
                row += ' '
        print(f"   y={y}: {row}")

def main():
    arcade = Arcade(operation_mode='offline')
    # Click c 0,1 times × click f 0..5 times × (click c first or f first)
    for n_c in range(0, 2):
        for n_f in range(0, 6):
            env, am = setup(arcade)
            game = env._game
            for _ in range(n_c):
                env.step(am[6], data={'x': 51, 'y': 25})
            for _ in range(n_f):
                env.step(am[6], data={'x': 56, 'y': 8})
            reach = player_reachable_cells(game)
            print(f"\nc={n_c} f={n_f}: reach={len(reach)}")
            if len(reach) != 18:
                render_map(game, reach)

if __name__ == "__main__":
    main()
