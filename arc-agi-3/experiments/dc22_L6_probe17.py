#!/usr/bin/env python3
"""From (34,48) post-zbhi-g, click c, f, grab. See all reach changes."""
import os, sys, json
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

def render(r, player=None):
    for y in range(0, 64, 2):
        row = ''
        for x in range(0, 64, 2):
            if player and (x,y)==player: row += '@'
            elif (x,y) in r: row += '.'
            else: row += ' '
        if row.strip():
            print(f"  y={y:3d}: {row}")

def main():
    arcade = Arcade(operation_mode='offline')
    env, am = setup(arcade)
    game = env._game

    # Get to (34,48) via c-teleport
    for mv in reconstruct_moves(player_reachable_cells(game), (18,48)):
        env.step(mv)
    env.step(am[6], data={'x':51,'y':25})  # teleport to (32,52)
    for mv in reconstruct_moves(player_reachable_cells(game), (34,48)):
        env.step(mv)
    print(f"At ({game.fdvakicpimr.x},{game.fdvakicpimr.y}); zbhi-g triggered")

    # Explore: click combinations and see what opens
    base_state = save_game_state(game)

    # Test: click c from (34,48)
    env.step(am[6], data={'x':51,'y':25})
    p = game.fdvakicpimr
    r = player_reachable_cells(game)
    print(f"\n-- click c -- player=({p.x},{p.y}) reach={len(r)}")
    render(r, (p.x,p.y))

    # Restore and test click f
    restore_game_state(game, base_state)
    env.step(am[6], data={'x':56,'y':8})
    p = game.fdvakicpimr
    r = player_reachable_cells(game)
    print(f"\n-- click f -- player=({p.x},{p.y}) reach={len(r)}")
    render(r, (p.x,p.y))

    # Restore and test click grab
    restore_game_state(game, base_state)
    env.step(am[6], data={'x':51,'y':18})
    p = game.fdvakicpimr
    r = player_reachable_cells(game)
    print(f"\n-- click grab -- player=({p.x},{p.y}) reach={len(r)} nxhz_attached={game.nxhz_attached_kind}")
    render(r, (p.x,p.y))

    # Test: click c multiple times in sequence
    restore_game_state(game, base_state)
    for i in range(4):
        env.step(am[6], data={'x':51,'y':25})
        env.step(am[6], data={'x':56,'y':8})  # also f
    p = game.fdvakicpimr
    r = player_reachable_cells(game)
    print(f"\n-- c,f × 4 -- player=({p.x},{p.y}) reach={len(r)}")
    render(r, (p.x,p.y))

if __name__ == "__main__":
    main()
