#!/usr/bin/env python3
"""Chain: walk→c-teleport→walk to zbhi-g→activate g→click g (need plate first)."""
import os, sys, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
sys.path.insert(0, "arc-agi-3/experiments")
from dc22_solve_final import player_reachable_cells, reconstruct_moves, save_frame

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

def dump_visible_clicks(game):
    out = []
    for s in game.current_level.get_sprites():
        if 'sys_click' in s.tags and s.interaction in (InteractionMode.TANGIBLE, InteractionMode.INTANGIBLE):
            out.append(f"{s.name}@({s.x},{s.y}) {s.interaction.name}")
    return out

def main():
    arcade = Arcade(operation_mode='offline')
    env, am = setup(arcade)
    game = env._game

    # Walk to (18,48)
    for mv in reconstruct_moves(player_reachable_cells(game), (18,48)):
        env.step(mv)
    # Click c → teleport to (32,52)
    env.step(am[6], data={'x':51,'y':25})
    assert (game.fdvakicpimr.x, game.fdvakicpimr.y) == (32,52)

    # Walk to (34,48) — zbhi-g
    r = player_reachable_cells(game)
    assert (34,48) in r, f"(34,48) not reachable, reach={sorted(r)}"
    for mv in reconstruct_moves(r, (34,48)):
        env.step(mv)
    print(f"At (34,48) standing on zbhi-g. Player=({game.fdvakicpimr.x},{game.fdvakicpimr.y})")
    print(f"Visible clicks: {dump_visible_clicks(game)}")
    # Check zbhi-g was removed, nxhz-bynyvtuepbt-1 now INTANGIBLE
    for s in game.current_level.get_sprites():
        if 'bynyvtuepbt' in s.name or 'bgeg-1' in s.name:
            print(f"  {s.name} ({s.x},{s.y}) {s.interaction.name}")

    # Now reach map
    render(player_reachable_cells(game), "reach after zbhi-g", (game.fdvakicpimr.x, game.fdvakicpimr.y))

    # Try clicking nxhz-bynyvtuepbt-1 (the grab button)
    # It's at (47,17) size 9x3, center (51,18)
    env.step(am[6], data={'x':51,'y':18})
    p = game.fdvakicpimr
    print(f"\nAfter clicking grab button (51,18): player=({p.x},{p.y})")
    print(f"  nxhz_attached_kind={game.nxhz_attached_kind}")
    print(f"  attached_hhxv_prefix={game.attached_hhxv_prefix}")
    print(f"  nxhz_x={game.nxhz_x} nxhz_y={game.nxhz_y}")
    print(f"  Visible clicks: {dump_visible_clicks(game)}")
    render(player_reachable_cells(game), "reach after grab click", (p.x, p.y))
    save_frame(env.observation_space.frame, f"{VIS}/L6_after_grab.png")

if __name__ == "__main__":
    main()
