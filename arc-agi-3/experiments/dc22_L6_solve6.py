#!/usr/bin/env python3
"""dc22 L6 - full BFS approach: enumerate all click sequences from player positions,
tracking actual game state via env.step."""
import os, sys, json
os.chdir("/home/dp/ai-workspace/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
import numpy as np
from collections import deque
from PIL import Image

VIS = "/home/dp/ai-workspace/shared-context/arc-agi-3/visual-memory/dc22"

PALETTE = {
    0:(255,255,255), 1:(220,220,220), 2:(255,0,0), 3:(128,128,128),
    4:(255,255,0), 5:(100,100,100), 6:(255,0,255), 7:(255,192,203),
    8:(200,0,0), 9:(128,0,0), 10:(0,0,255), 11:(135,206,250),
    12:(0,0,200), 13:(255,165,0), 14:(0,255,0), 15:(128,0,128),
}

def save_frame(frame_data, path):
    frame = np.array(frame_data[0])
    h, w = frame.shape
    scale = 8
    img = Image.new('RGB', (w*scale, h*scale))
    pix = img.load()
    for y in range(h):
        for x in range(w):
            c = PALETTE.get(int(frame[y,x]), (0,0,0))
            for dy in range(scale):
                for dx in range(scale):
                    pix[x*scale+dx, y*scale+dy] = c
    img.save(path)

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

def get_board_fingerprint(game):
    """Quick fingerprint of board state for dedup."""
    parts = []
    for s in game.current_level.get_sprites():
        if s.interaction == InteractionMode.REMOVED:
            continue
        if any(t in s.tags for t in ('wbze', 'jpug', 'zbhi', 'itki', 'kbqq', 'hhxv', 'pressure-plate')):
            parts.append((s.name, s.x, s.y, s.interaction.value))
    parts.sort()
    return (game.fdvakicpimr.x, game.fdvakicpimr.y,
            game.nxhz_x, game.nxhz_y, game.nxhz_attached_kind,
            tuple(parts))

def main():
    # Instead of save/restore (which seems unreliable for this game),
    # let's use a replay-from-scratch approach.
    # For each sequence of actions, we replay L1-L5 then execute L6 actions fresh.
    # This is slow but correct.

    # First, let me just try specific handcrafted sequences end-to-end.

    arcade = Arcade(operation_mode='offline')

    def fresh_L6():
        env = arcade.make('dc22-4c9bff3e')
        env.reset()
        replay_to_L6(env)
        return env

    # Sequence to test: 4 f-clicks, walk down staircase, check qgdz states
    print("=== TEST: 4 f-clicks + walk down ===")
    env = fresh_L6()
    game = env._game
    ct_data = {}
    cam_h = game.camera._height
    y_offset = (64 - cam_h) // 2
    for s in game.current_level.get_sprites():
        if 'sys_click' in s.tags:
            cx = s.x + s.width // 2
            cy = s.y + s.height // 2 + y_offset
            ct_data[s.name] = {'x': cx, 'y': cy}

    # 4 f-clicks
    for i in range(4):
        env.step(GameAction.ACTION6, data=ct_data['sprite-6'])
    print(f"After 4f: player ({game.fdvakicpimr.x},{game.fdvakicpimr.y}), steps={game.step_counter_ui.current_steps}")

    # Walk to (8,58)
    parents = player_reachable_cells(game)
    if (8, 58) not in parents:
        print("Can't reach (8,58)!")
        return
    moves = reconstruct_moves(parents, (8, 58))
    for m in moves:
        env.step(m)
    print(f"At (8,58): steps={game.step_counter_ui.current_steps}")

    # Now check ACTUAL qgdz states at this point
    print("\nQgdz states AFTER walk to (8,58):")
    for s in game.current_level.get_sprites():
        if 'qgdz' in s.name and s.interaction != InteractionMode.REMOVED:
            pixels = s.render()
            nonblank = [col + s.x for col in range(pixels.shape[1])
                       if any(pixels[row][col] >= 0 for row in range(pixels.shape[0]))]
            print(f"  {s.name} y={s.y} nonblank_x={nonblank}")

    # Now do 1 f-click and check
    env.step(GameAction.ACTION6, data=ct_data['sprite-6'])
    print(f"\nAfter +1 f-click (5 total): player ({game.fdvakicpimr.x},{game.fdvakicpimr.y}), steps={game.step_counter_ui.current_steps}")
    for s in game.current_level.get_sprites():
        if 'qgdz' in s.name and s.interaction != InteractionMode.REMOVED:
            pixels = s.render()
            nonblank = [col + s.x for col in range(pixels.shape[1])
                       if any(pixels[row][col] >= 0 for row in range(pixels.shape[0]))]
            print(f"  {s.name} y={s.y} nonblank_x={nonblank}")

    # Check support at current player pos
    supp = game.uxwpppoljm(game.fdvakicpimr.x, game.fdvakicpimr.y, game.fdvakicpimr)
    print(f"Support at ({game.fdvakicpimr.x},{game.fdvakicpimr.y}): {supp.name if supp else 'NONE'}")

    parents2 = player_reachable_cells(game)
    print(f"Reach: {len(parents2)} cells")
    ys = sorted(set(y for _, y in parents2.keys()))
    for y in ys:
        xs = sorted(x for x, yy in parents2.keys() if yy == y)
        print(f"  y={y}: x={xs}")

    # Another f-click (6 total)
    env.step(GameAction.ACTION6, data=ct_data['sprite-6'])
    print(f"\nAfter +2 f-clicks (6 total): player ({game.fdvakicpimr.x},{game.fdvakicpimr.y}), steps={game.step_counter_ui.current_steps}")
    supp = game.uxwpppoljm(game.fdvakicpimr.x, game.fdvakicpimr.y, game.fdvakicpimr)
    print(f"Support at ({game.fdvakicpimr.x},{game.fdvakicpimr.y}): {supp.name if supp else 'NONE'}")
    parents3 = player_reachable_cells(game)
    print(f"Reach: {len(parents3)} cells")
    ys = sorted(set(y for _, y in parents3.keys()))
    for y in ys:
        xs = sorted(x for x, yy in parents3.keys() if yy == y)
        print(f"  y={y}: x={xs}")

    # Try c-click + 4f + walk down + 2 more f-clicks
    print("\n\n=== TEST: c + 4f + walk + 2f ===")
    env = fresh_L6()
    game = env._game
    ct_data = {}
    for s in game.current_level.get_sprites():
        if 'sys_click' in s.tags:
            cx = s.x + s.width // 2
            cy = s.y + s.height // 2 + y_offset
            ct_data[s.name] = {'x': cx, 'y': cy}

    # c-click (bridge walkable)
    env.step(GameAction.ACTION6, data=ct_data['jpug-bjuk'])
    # 4 f-clicks
    for i in range(4):
        env.step(GameAction.ACTION6, data=ct_data['sprite-6'])
    # Walk to (8,58)
    parents = player_reachable_cells(game)
    moves = reconstruct_moves(parents, (8, 58))
    for m in moves:
        env.step(m)
    print(f"At (8,58): steps={game.step_counter_ui.current_steps}")

    # c-click (bridge goes to variant 1, itkis swap, etc)
    env.step(GameAction.ACTION6, data=ct_data['jpug-bjuk'])
    print(f"After c: player ({game.fdvakicpimr.x},{game.fdvakicpimr.y}), steps={game.step_counter_ui.current_steps}")
    supp = game.uxwpppoljm(game.fdvakicpimr.x, game.fdvakicpimr.y, game.fdvakicpimr)
    print(f"Support: {supp.name if supp else 'NONE'}")

    # 2 more f-clicks (total 6 = back to original)
    env.step(GameAction.ACTION6, data=ct_data['sprite-6'])
    env.step(GameAction.ACTION6, data=ct_data['sprite-6'])
    print(f"After 2 more f: player ({game.fdvakicpimr.x},{game.fdvakicpimr.y}), steps={game.step_counter_ui.current_steps}")
    supp = game.uxwpppoljm(game.fdvakicpimr.x, game.fdvakicpimr.y, game.fdvakicpimr)
    print(f"Support: {supp.name if supp else 'NONE'}")

    parents4 = player_reachable_cells(game)
    print(f"Reach: {len(parents4)} cells")
    ys = sorted(set(y for _, y in parents4.keys()))
    for y in ys:
        xs = sorted(x for x, yy in parents4.keys() if yy == y)
        print(f"  y={y}: x={xs}")

    # Check: is (6,48) reachable? If so, bridge connects to top
    has_6_48 = (6, 48) in parents4
    has_6_18 = (6, 18) in parents4
    has_34_58 = (34, 58) in parents4
    print(f"\n(6,48)={has_6_48}, (6,18)={has_6_18}, (34,58)={has_34_58}")

    # If reach includes kbqq areas on left side, we can walk to zbhi-d!
    if has_6_18:
        print("CAN REACH ZBHI-D!")

    # Also check: what happened to the hhxv bridge after c-clicks?
    print("\nhhxv state:")
    for s in game.current_level.get_sprites():
        if 'hhxv-dmxj' in s.name and s.interaction != InteractionMode.REMOVED:
            print(f"  {s.name} ({s.x},{s.y}) int={s.interaction.name}")

    # Check if the kbqq(6,44) connects to qgdz at y=54 with variant 1
    # kbqq(6,44) at x=6..9, y=44..47
    # kbqq(6,48) at x=6..11, y=48..53
    # qgdz variant 1 at y=54 at x=6..11
    # If all three are active and overlapping in x, they connect!
    print(f"\nChecking vertical connections on left side:")
    for y in range(44, 60, 2):
        for x in [6, 8]:
            supp = game.uxwpppoljm(x, y, game.fdvakicpimr)
            if supp:
                print(f"  ({x},{y}): {supp.name} ({supp.x},{supp.y})")

if __name__ == "__main__":
    main()
