#!/usr/bin/env python3
"""L6 probe: cycle itki colors, track reach at each phase."""
import os, sys, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
sys.path.insert(0, "arc-agi-3/experiments")
from dc22_solve_final import (
    save_game_state, restore_game_state, player_reachable_cells,
    find_click_targets, save_frame,
)

VIS = "/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/dc22"

def dump_state(game, label):
    p = game.fdvakicpimr
    print(f"\n=== {label} ===")
    print(f"Player: ({p.x},{p.y})")
    parents = player_reachable_cells(game)
    print(f"Reach: {len(parents)} cells")
    ys = sorted(set(y for _,y in parents.keys()))
    for y in ys[:60]:
        xs = sorted(x for x,yy in parents.keys() if yy==y)
        print(f"  y={y:3d}: x={xs}")
    # Report itki names (tracks cycling)
    itkis = [s for s in game.current_level.get_sprites()
             if 'itki' in s.tags and s.interaction != InteractionMode.REMOVED]
    print("itki sprites:")
    for s in itkis:
        print(f"  {s.name:15s} ({s.x},{s.y}) interact={s.interaction.name}")

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

    dump_state(game, "phase 0 (initial)")
    save_frame(env.observation_space.frame, f"{VIS}/L6_phase0.png")

    # Click gkrr-jpug (at 49,46) to cycle itki
    for cycle in range(1, 5):
        env.step(am[6], data={'x': 51, 'y': 48})
        dump_state(game, f"phase {cycle}")
        save_frame(env.observation_space.frame, f"{VIS}/L6_phase{cycle}.png")

if __name__ == "__main__":
    main()
