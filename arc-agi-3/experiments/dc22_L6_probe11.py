#!/usr/bin/env python3
"""Full-board support map under all c×f combos. Look for connectivity from player."""
import os, sys, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
sys.path.insert(0, "arc-agi-3/experiments")
from dc22_solve_final import player_reachable_cells

VIS = "/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/dc22"

def setup(arcade):
    env = arcade.make('dc22-4c9bff3e')
    env.reset()
    with open(f"{VIS}/solutions.json") as f:
        raw = json.load(f)
    am = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
          3: GameAction.ACTION3, 4: GameAction.ACTION4, 6: GameAction.ACTION6}
    for lvl_idx in range(5):
        for m in raw[lvl_idx]:
            env.step(am[m['action']], data=m.get('data', {}))
    return env, am

def support_map(game):
    p = game.fdvakicpimr
    ox, oy = p.x, p.y
    sup = set()
    for y in range(0, 64, 2):
        for x in range(0, 64, 2):
            p.set_position(x, y)
            if game.uxwpppoljm(x, y, p) is not None:
                sup.add((x,y))
    p.set_position(ox, oy)
    return sup

def render(sup, label, player=None):
    print(f"\n{label}: {len(sup)} supported cells")
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
    am = {1:GameAction.ACTION1, 2:GameAction.ACTION2, 3:GameAction.ACTION3, 4:GameAction.ACTION4, 6:GameAction.ACTION6}

    for nc in range(2):
        for nf in range(6):
            env, _ = setup(arcade)
            game = env._game
            for _ in range(nc):
                env.step(am[6], data={'x':51,'y':25})
            for _ in range(nf):
                env.step(am[6], data={'x':56,'y':8})
            sup = support_map(game)
            reach = player_reachable_cells(game)
            render(sup, f"c={nc} f={nf} (reach={len(reach)})", (game.fdvakicpimr.x, game.fdvakicpimr.y))

if __name__ == "__main__":
    main()
