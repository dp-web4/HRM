#!/usr/bin/env python3
"""Walk SW under f=4, then click f repeatedly, tracking cumulative reach."""
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

def render_full(reach, player=None):
    xs = sorted({x for x,_ in reach})
    ys = sorted({y for _,y in reach})
    print(f"  bbox: x {xs[0]}..{xs[-1]} y {ys[0]}..{ys[-1]}  count={len(reach)}")
    for y in range(0, 64, 2):
        row = ''
        for x in range(0, 64, 2):
            if player and (x,y) == player:
                row += '@'
            elif (x,y) in reach:
                row += '.'
            else:
                row += ' '
        if row.strip():
            print(f"  y={y:3d}: {row}")

def main():
    arcade = Arcade(operation_mode='offline')
    am = {1:GameAction.ACTION1, 2:GameAction.ACTION2, 3:GameAction.ACTION3, 4:GameAction.ACTION4, 6:GameAction.ACTION6}

    # Strategy: click f repeatedly from starting position and track reach monotonically
    # But also: walk between clicks
    env, _ = setup(arcade)
    game = env._game

    # Approach 1: stay put, click f many times, track reach at each step
    print("=== stay put, 20 f clicks ===")
    for i in range(20):
        env.step(am[6], data={'x':56,'y':8})
        reach = player_reachable_cells(game)
        pxy = (game.fdvakicpimr.x, game.fdvakicpimr.y)
        xs = sorted({x for x,_ in reach}) if reach else []
        ys = sorted({y for _,y in reach}) if reach else []
        bb = f"x[{xs[0]}..{xs[-1]}] y[{ys[0]}..{ys[-1]}]" if reach else ""
        print(f"  f×{i+1}: reach={len(reach):3d} {bb}")

    # Dump final state
    print("\nFinal (f×20):")
    render_full(player_reachable_cells(game), (game.fdvakicpimr.x, game.fdvakicpimr.y))
    save_frame(env.observation_space.frame, f"{VIS}/L6_f20.png")

    # Approach 2: walk furthest before each click, then click, repeat
    print("\n=== walk-then-click loop ===")
    env, _ = setup(arcade)
    game = env._game
    seen_cells = set()
    for cycle in range(15):
        # click f
        env.step(am[6], data={'x':56,'y':8})
        # walk to furthest reachable point (SW)
        reach = player_reachable_cells(game)
        # pick maximum (x+y) or min x, max y
        best = max(reach.keys(), key=lambda c: (c[1], -c[0]))  # max y, min x
        moves = reconstruct_moves(reach, best)
        for mv in moves:
            env.step(mv)
        pxy = (game.fdvakicpimr.x, game.fdvakicpimr.y)
        reach2 = player_reachable_cells(game)
        seen_cells |= set(reach2.keys())
        print(f"  cycle {cycle+1}: player={pxy} reach={len(reach2)} cumulative={len(seen_cells)}")

    print("\nFinal cumulative reach seen across walk-then-click loop:")
    render_full(seen_cells)

if __name__ == "__main__":
    main()
