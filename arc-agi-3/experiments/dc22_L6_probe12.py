#!/usr/bin/env python3
"""List all tagged sprites in L6 by letter."""
import os, sys, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode

VIS = "/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/dc22"

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

    by_letter = {}
    for s in game.current_level.get_sprites():
        if s.interaction == InteractionMode.REMOVED: continue
        if 'ignore' in s.tags: continue
        for t in s.tags:
            if len(t) == 1:
                by_letter.setdefault(t, []).append(s)
    for k in sorted(by_letter):
        print(f"\nLetter {k}: {len(by_letter[k])} sprites")
        for s in by_letter[k]:
            print(f"  {s.name:28s} ({s.x:3d},{s.y:3d}) {s.width}x{s.height} "
                  f"{s.interaction.name:<10s} tags={s.tags}")

if __name__ == "__main__":
    main()
