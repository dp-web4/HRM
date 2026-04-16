#!/usr/bin/env python3
"""Full BFS over (click_sequence, position) — find ANY way to reach plates or zbhi-d.
Uses short click sequences only."""
import os, sys, json, time
from collections import deque
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
sys.path.insert(0, "arc-agi-3/experiments")
from dc22_solve_final import (
    player_reachable_cells, reconstruct_moves, save_frame,
    save_game_state, restore_game_state, get_state_key,
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

def main():
    arcade = Arcade(operation_mode='offline')
    env, am = setup(arcade)
    game = env._game

    CLICK_C = (am[6], {'x':51,'y':25})
    CLICK_F = (am[6], {'x':56,'y':8})
    CLICK_GRAB = (am[6], {'x':51,'y':18})  # After zbhi-g triggered

    # Helper: walk to a target within current reach
    def walk_to(target):
        r = player_reachable_cells(game)
        if target not in r:
            return False
        for mv in reconstruct_moves(r, target):
            env.step(mv)
        return True

    # Strategy:
    #  Step 1: walk to (18,48), click c (teleport to 32,52).
    #  Step 2: walk to (34,48), activates zbhi-g (grab button live).
    #  Step 3: click c again (back to 18,48? player leaves (34,48) but grab button stays).
    #  Step 4: walk to (32,52), click c... cycle.
    # Then from each state, snapshot reach.
    env, am = setup(arcade)
    game = env._game

    print(f"Init: player=({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
    walk_to((18,48))
    env.step(*CLICK_C)
    print(f"After c1: ({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
    walk_to((34,48))  # zbhi-g
    print(f"After zbhi-g: ({game.fdvakicpimr.x},{game.fdvakicpimr.y})")

    # Check which sprites are now active
    vis = [s.name for s in game.current_level.get_sprites()
           if s.interaction in (InteractionMode.TANGIBLE, InteractionMode.INTANGIBLE)
           and 'sys_click' in s.tags]
    print(f"Clickable sprites: {vis}")

    # Now — can we click DIRECTION BUTTONS? They're at (53,35)(a), (49,31)(b), (45,35)(e), (49,39)(h)
    # All should still be INVISIBLE since we're not on plates. Let me verify.
    for name in ['nxhz-ghqmfnmmgrz-1', 'nxhz-zmjbupyjfyb-1', 'nxhz-vbdduyutyiw-1', 'nxhz-vbdduyutyiw-2', 'nxhz-bynyvtuepbt-1', 'gkrr-jpug']:
        for s in game.current_level.get_sprites():
            if s.name == name:
                print(f"  {name}: {s.interaction.name} vis={s.is_visible}")
                break

    # Try clicking each direction button — they should be invisible, but let's see
    before_reach = player_reachable_cells(game)
    print(f"\nReach before: {len(before_reach)}")
    for name, x, y in [('a', 55, 37), ('b', 51, 33), ('e', 47, 37), ('h', 51, 41), ('g', 51, 18)]:
        state = save_game_state(game)
        env.step(am[6], data={'x':x,'y':y})
        after = player_reachable_cells(game)
        changed = (after != before_reach) or game.nxhz_x != 0 or game.nxhz_y != 0
        print(f"  click {name}@({x},{y}): reach={len(after)} nxhz=({game.nxhz_x},{game.nxhz_y}) "
              f"attach={game.nxhz_attached_kind} changed={changed}")
        restore_game_state(game, state)

if __name__ == "__main__":
    main()
