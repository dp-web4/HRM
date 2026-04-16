#!/usr/bin/env python3
"""Extended reach probe: more f cycles, all c combos, walk-based itki triggers."""
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

def render_map(game, reach, y_lo=40, y_hi=64, x_lo=0, x_hi=40):
    for y in range(y_lo, y_hi, 2):
        row = ''
        for x in range(x_lo, x_hi, 2):
            if (x,y) in reach:
                row += '.'
            else:
                row += ' '
        print(f"   y={y:3d}: {row}")

def main():
    arcade = Arcade(operation_mode='offline')
    # Test f 0..11 (might have period > 6 due to differing initial indices)
    print("=== f only, 0..11 ===")
    for n_f in range(0, 12):
        env, am = setup(arcade)
        game = env._game
        for _ in range(n_f):
            env.step(am[6], data={'x': 56, 'y': 8})
        reach = player_reachable_cells(game)
        print(f"\nf={n_f}: reach={len(reach)}")
        if len(reach) != 18:
            render_map(game, reach)

    # c×n first, then f×m
    print("\n=== c varied with f=4 ===")
    for n_c in range(0, 6):
        env, am = setup(arcade)
        game = env._game
        for _ in range(n_c):
            env.step(am[6], data={'x': 51, 'y': 25})
        for _ in range(4):
            env.step(am[6], data={'x': 56, 'y': 8})
        reach = player_reachable_cells(game)
        print(f"\nc={n_c} f=4: reach={len(reach)}")
        if len(reach) > 18:
            render_map(game, reach)

if __name__ == "__main__":
    main()
