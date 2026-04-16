#!/usr/bin/env python3
"""Explore: walk-to-c-itki → click c → repeat; also track f usage."""
import os, sys, json, copy
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
sys.path.insert(0, "arc-agi-3/experiments")
from dc22_solve_final import (
    player_reachable_cells, reconstruct_moves, save_frame,
    save_game_state, restore_game_state,
)

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

def setup(arcade):
    env = arcade.make('dc22-4c9bff3e')
    env.reset()
    with open(f"{VIS}/solutions.json") as f:
        raw = json.load(f)
    am = {1:GameAction.ACTION1, 2:GameAction.ACTION2, 3:GameAction.ACTION3, 4:GameAction.ACTION4, 6:GameAction.ACTION6}
    for lvl_idx in range(5):
        for m in raw[lvl_idx]:
            env.step(am[m['action']], data=m.get('data', {}))
    return env, am

def main():
    arcade = Arcade(operation_mode='offline')
    env, am = setup(arcade)
    game = env._game

    # Sequence: walk to (18,48), click c, walk to (32,52) already there, click c again...
    reach = player_reachable_cells(game)
    for mv in reconstruct_moves(reach, (18,48)):
        env.step(mv)
    env.step(am[6], data={'x':51,'y':25})  # teleport to (32,52)
    print(f"After c1: player=({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
    render(player_reachable_cells(game), "r1", (game.fdvakicpimr.x, game.fdvakicpimr.y))

    # From (32,52), can we walk to some other itki? itkiupry1 is now at (32,52) (swapped).
    # After swap, (18,48) has itkiupry2. Let me print itki state.
    print("\nitki after c1:")
    for s in game.current_level.get_sprites():
        if 'itki' in s.tags and s.interaction != InteractionMode.REMOVED:
            print(f"  {s.name} ({s.x},{s.y})")

    # Click c again while standing on (32,52) = now itkiupry1 (swapped to 1 here? or?)
    env.step(am[6], data={'x':51,'y':25})
    print(f"After c2: player=({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
    render(player_reachable_cells(game), "r2", (game.fdvakicpimr.x, game.fdvakicpimr.y))

    print("\nitki after c2:")
    for s in game.current_level.get_sprites():
        if 'itki' in s.tags and s.interaction != InteractionMode.REMOVED:
            print(f"  {s.name} ({s.x},{s.y})")

    # Let's try walking from r1 to the plate cross or further east
    # Also try walking to other c-itkis: itkijbyz2 at (34,58), itkizfrq2 at (4,4)
    # Reset and do a deeper exploration
    env, am = setup(arcade)
    game = env._game

    # Walk to (18,48), click c → at (32,52)
    for mv in reconstruct_moves(player_reachable_cells(game), (18,48)):
        env.step(mv)
    env.step(am[6], data={'x':51,'y':25})
    # Now at (32,52), reach is 9 cells in plate corridor
    r = player_reachable_cells(game)
    render(r, "state after c1", (game.fdvakicpimr.x, game.fdvakicpimr.y))
    # Try walking to (34,58) — the itkijbyz at plate cross?
    if (34,58) in r:
        for mv in reconstruct_moves(r, (34,58)):
            env.step(mv)
        print(f"Reached (34,58)! player=({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
        # Click c to teleport somewhere
        env.step(am[6], data={'x':51,'y':25})
        print(f"After c from (34,58): player=({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
        render(player_reachable_cells(game), "r3", (game.fdvakicpimr.x, game.fdvakicpimr.y))

if __name__ == "__main__":
    main()
