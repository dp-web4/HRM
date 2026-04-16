#!/usr/bin/env python3
"""Brute: click at EVERY grid cell, see which clicks change game state."""
import os, sys, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
sys.path.insert(0, "arc-agi-3/experiments")
from dc22_solve_final import player_reachable_cells, save_game_state, restore_game_state

VIS = "/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/dc22"

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

def state_sig(game):
    s = []
    for sp in game.current_level.get_sprites():
        s.append((sp.name, sp.x, sp.y, sp.interaction.value))
    s.append(('player', game.fdvakicpimr.x, game.fdvakicpimr.y))
    return tuple(sorted(s))

def main():
    arcade = Arcade(operation_mode='offline')
    env, am = setup(arcade)
    game = env._game
    base_state = save_game_state(game)
    base_sig = state_sig(game)
    base_reach = player_reachable_cells(game)

    # Click at every cell
    effects = {}
    for y in range(0, 64):
        for x in range(0, 64):
            restore_game_state(game, base_state)
            env.step(am[6], data={'x':x,'y':y})
            sig = state_sig(game)
            if sig != base_sig:
                r = player_reachable_cells(game)
                key = (len(r), game.fdvakicpimr.x, game.fdvakicpimr.y)
                effects.setdefault(key, []).append((x,y))

    print(f"Unique effect classes: {len(effects)}")
    for key, coords in sorted(effects.items()):
        print(f"  reach={key[0]} player=({key[1]},{key[2]}): {len(coords)} click cells, sample={coords[:3]}")

if __name__ == "__main__":
    main()
