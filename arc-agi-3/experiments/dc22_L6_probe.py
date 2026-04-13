#!/usr/bin/env python3
"""Probe L6: dump state, click targets, initial reach."""
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
    reconstruct_moves, find_click_targets, save_frame,
)

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
    save_frame(env.observation_space.frame, f"{VIS}/L6_start.png")

    print(f"\nPlayer: ({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
    print(f"Goal: ({game.bqxa.x},{game.bqxa.y})")
    print(f"Steps: {game.step_counter_ui.rdnpeqedga}")
    print(f"nxhz_list: {[(n.name,n.x,n.y) for n in game.nxhz_list]}")
    print(f"uzungcfexgf (hhxv mode): {game.uzungcfexgf}")
    print(f"dzpheqwbxaj: {game.dzpheqwbxaj}")

    click_targets = find_click_targets(game)
    print(f"\n{len(click_targets)} click targets:")
    for k, n, cx, cy in click_targets:
        letter = '?'
        for s in game.current_level.get_sprites():
            if s.name == n:
                for t in s.tags:
                    if len(t) == 1: letter = t; break
                break
        print(f"  {k}: {n} ({cx},{cy}) letter={letter}")

    print("\nRelevant sprites:")
    for s in sorted(game.current_level.get_sprites(), key=lambda s:(s.y, s.x)):
        if s.interaction == InteractionMode.REMOVED: continue
        if 'ignore' in s.tags: continue
        if s.name.startswith('bg'): continue
        if any(t in s.tags for t in ('wbze','jpug','zbhi','itki','kbqq','bqxa','hhxv','pressure-plate','bynyvtuepbt-object','path','nxhz','pcxjvnmybet')):
            print(f"  {s.name:30s} ({s.x:3d},{s.y:3d}) {s.width}x{s.height:<2d} {s.interaction.name:<10s} tags={s.tags}")

    parents = player_reachable_cells(game)
    print(f"\nInitial reach: {len(parents)} cells")
    ys = sorted(set(y for _,y in parents.keys()))
    for y in ys:
        xs = sorted(x for x,yy in parents.keys() if yy==y)
        print(f"  y={y:3d}: x={xs}")

if __name__ == "__main__":
    main()
