#!/usr/bin/env python3
"""What's blocking player movement at reach boundaries?"""
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
    am = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
          3: GameAction.ACTION3, 4: GameAction.ACTION4,
          6: GameAction.ACTION6}
    for lvl_idx in range(5):
        for m in raw[lvl_idx]:
            env.step(am[m['action']], data=m.get('data', {}))
    game = env._game

    # Dump ALL sprites in the area around player (y=44..60, x=14..50)
    print("All sprites in region y=44..62 x=14..52:")
    for s in sorted(game.current_level.get_sprites(), key=lambda s:(s.y,s.x)):
        if s.interaction == InteractionMode.REMOVED: continue
        if s.y+s.height < 44 or s.y > 62: continue
        if s.x+s.width < 14 or s.x > 52: continue
        if 'ignore' in s.tags: continue
        print(f"  {s.name:25s} ({s.x:3d},{s.y:3d}) {s.width}x{s.height} "
              f"{s.interaction.name:<10s} vis={s.is_visible} coll={s.is_collidable} "
              f"tags={s.tags}")

    # Detailed test: try stepping onto each neighbor of player one by one
    player = game.fdvakicpimr
    px, py = player.x, player.y
    print(f"\nPlayer at ({px},{py})")
    print("Testing every 2x2 cell in region x=14..50 y=44..62 for whether uxwpppoljm supports it:")
    for y in range(44, 63, 2):
        row = []
        for x in range(14, 52, 2):
            player.set_position(x, y)
            supp = game.uxwpppoljm(x, y, player)
            # Also check collisions at that position
            collides = False
            for other in game.current_level.get_sprites():
                if other is player or 'ignore' in other.tags or 'nxhz' in other.tags: continue
                if not (player.is_collidable and other.is_collidable): continue
                if game.collides_with(player, other):
                    collides = True; break
            if supp is not None and not collides:
                row.append('.')
            elif collides:
                row.append('X')
            else:
                row.append(' ')
        print(f"y={y:3d}: |{''.join(row)}|")
    player.set_position(px, py)

if __name__ == "__main__":
    main()
