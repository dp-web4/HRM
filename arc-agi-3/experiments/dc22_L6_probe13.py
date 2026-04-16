#!/usr/bin/env python3
"""Test: walk to (18,48), click c, does player teleport?"""
import os, sys, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
sys.path.insert(0, "arc-agi-3/experiments")
from dc22_solve_final import player_reachable_cells, reconstruct_moves, save_frame

VIS = "/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/dc22"

def render(sup, label, player=None):
    print(f"\n{label}")
    for y in range(0, 64, 2):
        row = ''
        for x in range(0, 64, 2):
            if player and (x,y) == player: row += '@'
            elif (x,y) in sup: row += '.'
            else: row += ' '
        if row.strip():
            print(f"  y={y:3d}: {row}")

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('dc22-4c9bff3e')
    env.reset()
    with open(f"{VIS}/solutions.json") as f:
        raw = json.load(f)
    am = {1:GameAction.ACTION1, 2:GameAction.ACTION2, 3:GameAction.ACTION3, 4:GameAction.ACTION4, 6:GameAction.ACTION6}
    for lvl_idx in range(5):
        for m in raw[lvl_idx]:
            env.step(am[m['action']], data=m.get('data', {}))
    game = env._game

    print(f"Start player=({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
    reach = player_reachable_cells(game)
    render(reach, "initial reach", (game.fdvakicpimr.x, game.fdvakicpimr.y))

    # Walk to (18,48)
    moves = reconstruct_moves(reach, (18,48))
    print(f"\nWalk path to (18,48): {len(moves)} steps")
    for mv in moves:
        env.step(mv)
    print(f"Now at ({game.fdvakicpimr.x},{game.fdvakicpimr.y})")

    # Click c
    env.step(am[6], data={'x':51,'y':25})
    print(f"After click c: player=({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
    reach2 = player_reachable_cells(game)
    render(reach2, f"reach after teleport (len={len(reach2)})", (game.fdvakicpimr.x, game.fdvakicpimr.y))
    save_frame(env.observation_space.frame, f"{VIS}/L6_after_teleport.png")

if __name__ == "__main__":
    main()
