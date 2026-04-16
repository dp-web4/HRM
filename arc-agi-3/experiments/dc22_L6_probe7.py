#!/usr/bin/env python3
"""Full-map reach after f=4, f=5 — including player repositioning."""
import os, sys, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
sys.path.insert(0, "arc-agi-3/experiments")
from dc22_solve_final import player_reachable_cells, save_frame, reconstruct_moves

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

def render_full(reach):
    xs = sorted({x for x,_ in reach})
    ys = sorted({y for _,y in reach})
    print(f"  bbox: x {xs[0]}..{xs[-1]} y {ys[0]}..{ys[-1]}")
    for y in range(0, 64, 2):
        row = ''
        for x in range(0, 64, 2):
            if (x,y) in reach:
                row += '.'
            else:
                row += ' '
        if row.strip():
            print(f"  y={y:3d}: {row}")

def main():
    arcade = Arcade(operation_mode='offline')
    am = {1:GameAction.ACTION1, 2:GameAction.ACTION2, 3:GameAction.ACTION3, 4:GameAction.ACTION4, 6:GameAction.ACTION6}

    # f=4: walk to SW-most point then click f again and see if more opens
    for nf in [4, 5]:
        print(f"\n=== f={nf} baseline ===")
        env, _ = setup(arcade)
        game = env._game
        for _ in range(nf):
            env.step(am[6], data={'x':56,'y':8})
        reach = player_reachable_cells(game)
        render_full(reach)

    # At f=4, walk player to (16,58) (westmost-southmost), then try clicking f more
    print("\n=== f=4 then walk SW then click f more ===")
    env, _ = setup(arcade)
    game = env._game
    for _ in range(4):
        env.step(am[6], data={'x':56,'y':8})
    reach = player_reachable_cells(game)
    # Walk to (16,58)
    target = (16, 58)
    if target in reach:
        moves = reconstruct_moves(reach, target)
        for mv in moves:
            env.step(mv)
        print(f"  walked to (16,58), now at ({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
        # Try clicking f
        for i in range(6):
            env.step(am[6], data={'x':56,'y':8})
            r2 = player_reachable_cells(game)
            print(f"  +f #{i+1}: reach={len(r2)}")
        render_full(player_reachable_cells(game))
    else:
        print(f"  (16,58) not reachable: nearest are {sorted(reach)[:5]}")

if __name__ == "__main__":
    main()
