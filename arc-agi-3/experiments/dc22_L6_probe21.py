#!/usr/bin/env python3
"""After teleport + zbhi-g step: check FULL walkability for everything including resurrected sprites."""
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

def render_full(game):
    p = game.fdvakicpimr
    ox, oy = p.x, p.y
    sup = set()
    col = set()
    for y in range(0, 64, 2):
        for x in range(0, 64, 2):
            p.set_position(x, y)
            if game.uxwpppoljm(x, y, p) is not None:
                sup.add((x,y))
            for other in game.current_level.get_sprites():
                if other is p or 'ignore' in other.tags or 'nxhz' in other.tags: continue
                if not (p.is_collidable and other.is_collidable): continue
                if game.collides_with(p, other):
                    col.add((x,y)); break
    p.set_position(ox, oy)
    print("    " + "".join(f'{(x//2)%10}' for x in range(0, 64, 2)))
    for y in range(0, 64, 2):
        row = f"{y:3d}:"
        for x in range(0, 64, 2):
            pp = (x,y)
            if pp == (p.x, p.y): row += '@'
            elif pp in col: row += 'X'
            elif pp in sup: row += '.'
            else: row += ' '
        print(row)

def setup(arcade):
    env = arcade.make('dc22-4c9bff3e')
    env.reset()
    with open(f"{VIS}/solutions.json") as f:
        raw = json.load(f)
    am = {1:GameAction.ACTION1,2:GameAction.ACTION2,3:GameAction.ACTION3,4:GameAction.ACTION4,6:GameAction.ACTION6}
    for lvl_idx in range(5):
        for m in raw[lvl_idx]:
            env.step(am[m['action']], data=m.get('data', {}))
    return env, am

def main():
    arcade = Arcade(operation_mode='offline')
    env, am = setup(arcade)
    game = env._game

    # teleport chain
    for mv in reconstruct_moves(player_reachable_cells(game), (18,48)):
        env.step(mv)
    env.step(am[6], data={'x':51,'y':25})  # teleport to (32,52)
    for mv in reconstruct_moves(player_reachable_cells(game), (34,48)):
        env.step(mv)
    print("After walking onto zbhi-g at (34,48):")
    print(f"  Player: ({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
    render_full(game)

    # Click jpug-bjuk (c) from here -- might trigger new state
    env.step(am[6], data={'x':51,'y':25})
    print("\nAfter click c from (34,48):")
    print(f"  Player: ({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
    render_full(game)

if __name__ == "__main__":
    main()
