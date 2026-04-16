#!/usr/bin/env python3
"""Cleaner walkability map (1 char per cell)."""
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

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('dc22-4c9bff3e')
    env.reset()
    with open(f"{VIS}/solutions.json") as f:
        raw = json.load(f)
    am = {1:GameAction.ACTION1,2:GameAction.ACTION2,3:GameAction.ACTION3,4:GameAction.ACTION4,6:GameAction.ACTION6}
    for lvl_idx in range(5):
        for m in raw[lvl_idx]:
            env.step(am[m['action']], data=m.get('data', {}))
    game = env._game
    p = game.fdvakicpimr
    ox, oy = p.x, p.y

    def walkable(x, y):
        # supported and no collision
        p.set_position(x, y)
        if game.uxwpppoljm(x, y, p) is None:
            return False
        for other in game.current_level.get_sprites():
            if other is p or 'ignore' in other.tags or 'nxhz' in other.tags: continue
            if not (p.is_collidable and other.is_collidable): continue
            if game.collides_with(p, other):
                return False
        return True

    # Header
    print("    " + "".join(f'{(x//2)%10}' for x in range(0, 64, 2)))
    for y in range(0, 64, 2):
        row = f"{y:3d}:"
        for x in range(0, 64, 2):
            if walkable(x, y):
                row += '.'
            else:
                row += ' '
        print(row)
    p.set_position(ox, oy)

    # Now test reach
    reach = player_reachable_cells(game)
    print(f"\nReach from player: {len(reach)} cells, bbox: {min(reach)}..{max(reach)}")

if __name__ == "__main__":
    main()
