#!/usr/bin/env python3
"""Direct support check for the top-right region and the goal."""
import os, sys, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
sys.path.insert(0, "arc-agi-3/experiments")

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

    def has_support(x, y):
        p.set_position(x, y)
        return game.uxwpppoljm(x, y, p) is not None

    def has_collision(x, y):
        p.set_position(x, y)
        for other in game.current_level.get_sprites():
            if other is p or 'ignore' in other.tags or 'nxhz' in other.tags: continue
            if not (p.is_collidable and other.is_collidable): continue
            if game.collides_with(p, other):
                return other.name
        return None

    # Print combined walkability map for full board
    print("Walkability (. supported AND no collision, X collision, ' ' no support)")
    print("       " + " ".join(f'{x:2d}' for x in range(0, 64, 2)))
    for y in range(0, 64, 2):
        row = f"y={y:3d}: "
        for x in range(0, 64, 2):
            sup = has_support(x, y)
            col = has_collision(x, y)
            if col:
                row += ' X'
            elif sup:
                row += ' .'
            else:
                row += '  '
        print(row)
    p.set_position(ox, oy)

if __name__ == "__main__":
    main()
