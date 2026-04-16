#!/usr/bin/env python3
"""What sprite provides support at (10,48)? And at other mysterious cells?"""
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
          3: GameAction.ACTION3, 4: GameAction.ACTION4, 6: GameAction.ACTION6}
    for lvl_idx in range(5):
        for m in raw[lvl_idx]:
            env.step(am[m['action']], data=m.get('data', {}))
    game = env._game

    # All sprites that span the mystery zones
    print("Sprites overlapping y=46..50 area:")
    for s in sorted(game.current_level.get_sprites(), key=lambda s:(s.y,s.x)):
        if s.interaction == InteractionMode.REMOVED: continue
        if s.y+s.height <= 46 or s.y >= 52: continue
        if 'ignore' in s.tags: continue
        print(f"  {s.name:28s} ({s.x:3d},{s.y:3d}) {s.width}x{s.height} {s.interaction.name:<10s} "
              f"vis={s.is_visible} tags={s.tags}")

    print("\nSprites overlapping y=56..62 area:")
    for s in sorted(game.current_level.get_sprites(), key=lambda s:(s.y,s.x)):
        if s.interaction == InteractionMode.REMOVED: continue
        if s.y+s.height <= 56 or s.y >= 62: continue
        if 'ignore' in s.tags: continue
        print(f"  {s.name:28s} ({s.x:3d},{s.y:3d}) {s.width}x{s.height} {s.interaction.name:<10s} "
              f"vis={s.is_visible} tags={s.tags}")

    print("\nAll zbhi sprites:")
    for s in game.current_level.get_sprites():
        if 'zbhi' in s.tags and s.interaction != InteractionMode.REMOVED:
            print(f"  {s.name} ({s.x},{s.y}) tags={s.tags}")

    print("\nALL tangible/collidable sprites in L6:")
    for s in game.current_level.get_sprites():
        if s.interaction == InteractionMode.REMOVED: continue
        if 'ignore' in s.tags: continue
        if s.is_collidable and s.interaction == InteractionMode.TANGIBLE:
            print(f"  {s.name:28s} ({s.x:3d},{s.y:3d}) {s.width}x{s.height}")

if __name__ == "__main__":
    main()
