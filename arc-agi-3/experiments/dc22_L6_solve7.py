#!/usr/bin/env python3
"""dc22 L6 - exhaustive check of ALL possible player support positions."""
import os, sys, json
os.chdir("/home/dp/ai-workspace/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
import numpy as np

VIS = "/home/dp/ai-workspace/shared-context/arc-agi-3/visual-memory/dc22"

def player_reachable_cells(game):
    player = game.fdvakicpimr
    start = (player.x, player.y)
    orig_x, orig_y = player.x, player.y
    parents = {start: None}
    order = [start]
    head = 0
    deltas = [
        (0, -2, GameAction.ACTION1),
        (0,  2, GameAction.ACTION2),
        (-2, 0, GameAction.ACTION3),
        ( 2, 0, GameAction.ACTION4),
    ]
    while head < len(order):
        cx, cy = order[head]
        head += 1
        for dx, dy, act in deltas:
            player.set_position(cx, cy)
            collisions = game.try_move_sprite(player, dx, dy)
            if collisions:
                continue
            nx, ny = player.x, player.y
            if (nx, ny) == (cx, cy):
                continue
            if game.uxwpppoljm(nx, ny, player) is None:
                continue
            if (nx, ny) in parents:
                continue
            parents[(nx, ny)] = (cx, cy, act)
            order.append((nx, ny))
    player.set_position(orig_x, orig_y)
    return parents

def reconstruct_moves(parents, goal):
    moves = []
    cur = goal
    while parents.get(cur) is not None:
        px, py, act = parents[cur]
        moves.append(act)
        cur = (px, py)
    moves.reverse()
    return moves

def replay_to_L6(env):
    cache_path = f"{VIS}/solutions.json"
    with open(cache_path) as f:
        raw = json.load(f)
    am = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
          3: GameAction.ACTION3, 4: GameAction.ACTION4,
          6: GameAction.ACTION6}
    for lvl_idx in range(5):
        for m in raw[lvl_idx]:
            a = am[m['action']]
            env.step(a, data=m.get('data', {}))

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('dc22-4c9bff3e')
    env.reset()
    replay_to_L6(env)
    game = env._game

    # Map ALL supported positions for EVERY possible board configuration
    # (each combination of c-clicks and f-clicks)
    # c-click toggles hhxv bridge between variant 1 and 2
    # f-click cycles qgdz variants (period 6)

    cam_h = game.camera._height
    y_offset = (64 - cam_h) // 2
    ct_data = {}
    for s in game.current_level.get_sprites():
        if 'sys_click' in s.tags:
            cx = s.x + s.width // 2
            cy = s.y + s.height // 2 + y_offset
            ct_data[s.name] = {'x': cx, 'y': cy}

    # For each config, build a complete support map and check reachability
    # from ALL supported positions (not just (28,52))
    for nc in range(2):  # 0 or 1 c-clicks
        for nf in range(6):  # 0-5 f-clicks
            env2 = arcade.make('dc22-4c9bff3e')
            env2.reset()
            replay_to_L6(env2)
            g = env2._game

            for i in range(nc):
                env2.step(GameAction.ACTION6, data=ct_data['jpug-bjuk'])
            for i in range(nf):
                env2.step(GameAction.ACTION6, data=ct_data['sprite-6'])

            # Build FULL support map: all positions where player would be supported
            support_map = {}
            for x in range(0, 64, 2):
                for y in range(0, 64, 2):
                    supp = g.uxwpppoljm(x, y, g.fdvakicpimr)
                    if supp:
                        support_map[(x, y)] = supp.name

            # Check if there's support at the critical gap: x=12-17, y=48-53
            gap_cells = [(x, y) for x in range(12, 18, 2) for y in range(48, 54, 2)
                        if (x, y) in support_map]

            # Check if (6,18) is reachable from (28,52)
            parents = player_reachable_cells(g)
            has_6_18 = (6, 18) in parents
            has_34_58 = (34, 58) in parents

            # Check if there's any path from right to left at y=48-53
            right_cells = set(c for c in parents if c[0] >= 18)
            left_cells = set(c for c in support_map if c[0] <= 12 and 48 <= c[1] <= 53)
            connected = bool(right_cells & set(parents.keys()) and
                           any(c in parents for c in left_cells))

            print(f"c={nc} f={nf}: reach={len(parents)}, gap_cells={len(gap_cells)}, "
                  f"(6,18)={has_6_18}, (34,58)={has_34_58}, "
                  f"gap={gap_cells[:3]}")

            if gap_cells:
                for cell in gap_cells:
                    print(f"  GAP at {cell}: {support_map[cell]}")

    # Also check: with c-click making bridge walkable, what's the FULL
    # support map at y=24-53?
    print("\n\n=== BRIDGE VARIANT 2 FULL SUPPORT MAP ===")
    env3 = arcade.make('dc22-4c9bff3e')
    env3.reset()
    replay_to_L6(env3)
    g3 = env3._game
    env3.step(GameAction.ACTION6, data=ct_data['jpug-bjuk'])  # c-click

    # Check all positions
    print("Support at x=0..22, y=20..54:")
    for y in range(20, 56, 2):
        row = []
        for x in range(0, 24, 2):
            supp = g3.uxwpppoljm(x, y, g3.fdvakicpimr)
            if supp:
                if 'kbqq' in supp.name:
                    row.append('K')
                elif 'hhxv' in supp.name:
                    row.append('H')
                elif 'qgdz' in supp.name:
                    row.append('Q')
                elif 'itki' in supp.name:
                    row.append('I')
                else:
                    row.append('?')
            else:
                row.append('.')
        print(f"  y={y:2d}: {''.join(row)}  x=0,2,4,...,22")

    # Now check: with the bridge, compute reach from (4,4) (as if we teleported there)
    print("\n\n=== REACH FROM (4,4) WITH BRIDGE ===")
    g3.fdvakicpimr.set_position(4, 4)
    parents_from_4_4 = player_reachable_cells(g3)
    print(f"Reach from (4,4) with bridge: {len(parents_from_4_4)} cells")
    ys = sorted(set(y for _, y in parents_from_4_4.keys()))
    for y in ys:
        xs = sorted(x for x, yy in parents_from_4_4.keys() if yy == y)
        print(f"  y={y:2d}: x={xs}")

    # Check if (6,18) is reachable from (4,4) with bridge
    has_6_18 = (6, 18) in parents_from_4_4
    print(f"\n(6,18) reachable from (4,4) with bridge: {has_6_18}")

    # Check if (18,48) is reachable from (4,4) with bridge
    has_18_48 = (18, 48) in parents_from_4_4
    print(f"(18,48) reachable from (4,4) with bridge: {has_18_48}")

if __name__ == "__main__":
    main()
