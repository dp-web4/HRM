#!/usr/bin/env python3
"""dc22 L6 solver — explore and solve the final level."""
import os, sys, json, copy
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

def save_game_state(game):
    safe_sprites = []
    for s in game.current_level.get_sprites():
        c = s.clone()
        c.set_position(s.x, s.y)
        c.set_interaction(s.interaction)
        c._blocking = s._blocking
        c._tags = list(s.tags)
        safe_sprites.append(c)
    return {
        'sprites': safe_sprites,
        'nxhz_x': game.nxhz_x, 'nxhz_y': game.nxhz_y,
        'nxhz_attached_kind': game.nxhz_attached_kind,
        'attached_hhxv_prefix': game.attached_hhxv_prefix,
        'attached_hhxv_x': game.attached_hhxv_x,
        'attached_hhxv_y': game.attached_hhxv_y,
        'step_counter': game.step_counter_ui.rdnpeqedga,
        'current_steps': game.step_counter_ui.current_steps,
        'uuehztercxf': game.uuehztercxf,
        'pxicvzkjuui': game.pxicvzkjuui,
        'prbjhwkkxth': game.prbjhwkkxth,
        'fgxfjbqnmgt': game.fgxfjbqnmgt,
        'zemyudjnnqd': game.zemyudjnnqd,
        'jnmawhhrfhh': game.jnmawhhrfhh,
        'dimvmykkjbg': game.dimvmykkjbg,
    }

def restore_game_state(game, state):
    fresh_sprites = []
    for s in state['sprites']:
        c = s.clone()
        c.set_position(s.x, s.y)
        c.set_interaction(s.interaction)
        c._blocking = s._blocking
        c._tags = list(s.tags)
        fresh_sprites.append(c)
    game.lvnwxszdcv = {
        'sprites': fresh_sprites,
        'nxhz_x': state['nxhz_x'], 'nxhz_y': state['nxhz_y'],
        'nxhz_attached_kind': state['nxhz_attached_kind'],
        'attached_hhxv_prefix': state['attached_hhxv_prefix'],
        'attached_hhxv_x': state['attached_hhxv_x'],
        'attached_hhxv_y': state['attached_hhxv_y'],
    }
    game.ycfbtkckze()
    game.step_counter_ui.rdnpeqedga = state['step_counter']
    game.step_counter_ui.current_steps = state['current_steps']
    game.uuehztercxf = state['uuehztercxf']
    game.pxicvzkjuui = state['pxicvzkjuui']
    game.prbjhwkkxth = state['prbjhwkkxth']
    game.fgxfjbqnmgt = state['fgxfjbqnmgt']
    game.zemyudjnnqd = state['zemyudjnnqd']
    game.jnmawhhrfhh = state['jnmawhhrfhh']
    game.dimvmykkjbg = state['dimvmykkjbg']
    game.nxhz_x = state['nxhz_x']
    game.nxhz_y = state['nxhz_y']
    game.nxhz_attached_kind = state['nxhz_attached_kind']
    game.attached_hhxv_prefix = state['attached_hhxv_prefix']
    game.attached_hhxv_x = state['attached_hhxv_x']
    game.attached_hhxv_y = state['attached_hhxv_y']
    if state['nxhz_attached_kind'] == 'hhxv':
        try: game.dldhnotovw()
        except ValueError: pass
    elif state['nxhz_attached_kind'] == 'bynyvtuepbt-object' and game.ciuxrvkyndj:
        game.euqqhkqayni = game.ciuxrvkyndj

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

def find_click_targets(game):
    targets = []
    cam_h = game.camera._height
    y_offset = (64 - cam_h) // 2
    seen_coords = set()
    for s in game.current_level.get_sprites():
        if 'jpug' in s.tags and 'sys_click' in s.tags:
            cx = s.x + s.width // 2
            cy = s.y + s.height // 2 + y_offset
            if (cx, cy) not in seen_coords:
                seen_coords.add((cx, cy))
                targets.append(('jpug', s.name, cx, cy))
        elif 'sys_click' in s.tags and 'jpug' not in s.tags:
            cx = s.x + s.width // 2
            cy = s.y + s.height // 2 + y_offset
            if (cx, cy) not in seen_coords:
                seen_coords.add((cx, cy))
                targets.append(('sys_click', s.name, cx, cy))
    return targets

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

def dump_reach(game, label=""):
    parents = player_reachable_cells(game)
    print(f"\n--- Reach {label}: {len(parents)} cells ---")
    ys = sorted(set(y for _, y in parents.keys()))
    for y in ys:
        xs = sorted(x for x, yy in parents.keys() if yy == y)
        print(f"  y={y:3d}: x={xs}")
    return parents

def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('dc22-4c9bff3e')
    env.reset()
    replay_to_L6(env)
    game = env._game

    save_frame(env.observation_space.frame, f"{VIS}/L6_start_fresh.png")

    p = game.fdvakicpimr
    print(f"Player: ({p.x},{p.y})")
    print(f"Goal: ({game.bqxa.x},{game.bqxa.y})")
    print(f"Steps: {game.step_counter_ui.current_steps}/{game.step_counter_ui.rdnpeqedga}")
    print(f"Crane logical: ({game.nxhz_x},{game.nxhz_y})")
    print(f"Crane pixel: ({game.nxhz.x},{game.nxhz.y})")
    print(f"Crane start: {game.nxhz_start}")
    print(f"hhxv mode: {game.uzungcfexgf}")
    print(f"Crane sprite: {game.nxhz.name} {game.nxhz.width}x{game.nxhz.height}")

    # List all non-removed sprites
    print("\nAll active sprites:")
    for s in sorted(game.current_level.get_sprites(), key=lambda s: (s.y, s.x)):
        if s.interaction == InteractionMode.REMOVED:
            continue
        if 'ignore' in s.tags:
            continue
        letter = next((t for t in s.tags if len(t) == 1), '')
        print(f"  {s.name:35s} ({s.x:3d},{s.y:3d}) {s.width:2d}x{s.height:<2d} "
              f"int={s.interaction.name:<12s} vis={s.is_visible:<5} ltr={letter} tags={s.tags}")

    # Path grid
    print("\nPath grid (valid crane positions):")
    valid_positions = []
    for ny in range(-3, 10):
        for nx in range(-3, 10):
            if game.ogwbggfvor(nx, ny):
                px = game.nxhz_start[0] + nx * 4
                py = game.nxhz_start[1] - ny * 4
                valid_positions.append((nx, ny, px, py))
                print(f"  crane ({nx:2d},{ny:2d}) -> pixel ({px:3d},{py:3d})")
    print(f"  Total valid: {len(valid_positions)}")

    # Initial reach
    parents0 = dump_reach(game, "INITIAL")

    # Check reachable targets
    print("\nReachable targets from start:")
    for s in game.current_level.get_sprites():
        if s.interaction == InteractionMode.REMOVED:
            continue
        for tag in ('itki', 'zbhi', 'pressure-plate'):
            if tag in s.tags:
                letter = next((t for t in s.tags if len(t) == 1), '?')
                if (s.x, s.y) in parents0:
                    print(f"  {tag} {s.name} ({s.x},{s.y}) ltr={letter} DIRECT")
                else:
                    # check overlapping cells
                    found = False
                    for dx in range(-4, s.width+4, 2):
                        for dy in range(-4, s.height+4, 2):
                            if (s.x+dx, s.y+dy) in parents0:
                                if not found:
                                    print(f"  {tag} {s.name} ({s.x},{s.y}) ltr={letter} NEARBY at ({s.x+dx},{s.y+dy})")
                                    found = True

    # Save initial state
    state0 = save_game_state(game)

    # EXPERIMENT: Teleport via c-click from itkiupry1 at (18,48)
    print("\n\n=== TELEPORT EXPERIMENT ===")
    if (18, 48) in parents0:
        print("Walking to itkiupry1 at (18,48)...")
        moves = reconstruct_moves(parents0, (18, 48))
        for m in moves:
            env.step(m)
        print(f"Player now at ({game.fdvakicpimr.x},{game.fdvakicpimr.y})")

        # Find c button
        c_data = None
        for k, n, cx, cy in find_click_targets(game):
            if n == 'jpug-bjuk':
                c_data = {'x': cx, 'y': cy}
                break

        if c_data:
            fd = env.step(GameAction.ACTION6, data=c_data)
            print(f"After c-click: player at ({game.fdvakicpimr.x},{game.fdvakicpimr.y}), state={fd.state.name}")

            parents1 = dump_reach(game, "AFTER TELEPORT")

            # Check zbhi access
            print("\nZbhi reachability after teleport:")
            for s in game.current_level.get_sprites():
                if 'zbhi' in s.tags and s.interaction != InteractionMode.REMOVED:
                    letter = next((t for t in s.tags if len(t) == 1), '?')
                    if (s.x, s.y) in parents1:
                        print(f"  ZBHI {s.name} ({s.x},{s.y}) ltr={letter} REACHABLE")
                    else:
                        for dx in range(-4, s.width+4, 2):
                            for dy in range(-4, s.height+4, 2):
                                if (s.x+dx, s.y+dy) in parents1:
                                    print(f"  ZBHI {s.name} ({s.x},{s.y}) ltr={letter} NEARBY at ({s.x+dx},{s.y+dy})")
                                    break

            # Check pressure plates
            print("\nPressure plate reachability after teleport:")
            for s in game.current_level.get_sprites():
                if 'pressure-plate' in s.tags and s.interaction != InteractionMode.REMOVED:
                    letter = next((t for t in s.tags if len(t) == 1), '?')
                    if (s.x, s.y) in parents1:
                        print(f"  PLATE {s.name} ({s.x},{s.y}) ltr={letter} REACHABLE")
                    else:
                        found_any = False
                        for dx in range(-4, s.width+4, 2):
                            for dy in range(-4, s.height+4, 2):
                                if (s.x+dx, s.y+dy) in parents1:
                                    if not found_any:
                                        print(f"  PLATE {s.name} ({s.x},{s.y}) ltr={letter} NEARBY at ({s.x+dx},{s.y+dy})")
                                        found_any = True

            # Try to reach zbhi at (34,48)
            if (34, 48) in parents1:
                print("\n\n=== ZBHI-BGEG EXPERIMENT ===")
                moves2 = reconstruct_moves(parents1, (34, 48))
                for m in moves2:
                    env.step(m)
                print(f"Player at ({game.fdvakicpimr.x},{game.fdvakicpimr.y})")

                # What changed? Check zbhi removal and jpug changes
                for s in game.current_level.get_sprites():
                    if s.name.startswith('zbhi') or ('g' in s.tags and 'jpug' in s.tags):
                        print(f"  {s.name} int={s.interaction.name} vis={s.is_visible}")

                parents2 = dump_reach(game, "AFTER ZBHI-G PICKUP")
                save_frame(env.observation_space.frame, f"{VIS}/L6_after_zbhi_g.png")

                # Now check if we can reach kbqq-efzv-1 at (32,56) area
                # which connects to pressure plates
                print("\nChecking access to plate area:")
                for target in [(32, 56), (32, 58), (34, 56), (34, 58), (34, 60), (36, 58)]:
                    if target in parents2:
                        print(f"  {target} REACHABLE")

                # Check kbqq-efzv-1 positions
                print("\nkbqq-efzv-1 tiles:")
                for s in game.current_level.get_sprites():
                    if s.name == 'kbqq-efzv-1' and s.interaction != InteractionMode.REMOVED:
                        reachable = (s.x, s.y) in parents2
                        print(f"  ({s.x},{s.y}) {s.width}x{s.height} int={s.interaction.name} reach={reachable}")

                # What's between kbqq at (32,48) and kbqq at (32,56)?
                # There's a gap at y=52..55 or similar
                print("\nAll floor/walkable in y=48..62 range:")
                for s in sorted(game.current_level.get_sprites(), key=lambda s: (s.y, s.x)):
                    if s.interaction == InteractionMode.REMOVED:
                        continue
                    if s.y >= 44 and s.y <= 62:
                        if any(t in s.tags for t in ('kbqq', 'wbze', 'itki', 'pressure-plate', 'hhxv')):
                            print(f"  {s.name:35s} ({s.x:3d},{s.y:3d}) {s.width:2d}x{s.height:<2d} int={s.interaction.name}")

                # Check qgdz sprites (floor extensions via f-cycle)
                print("\nqgdz sprites (f-cycle floor):")
                for s in game.current_level.get_sprites():
                    if 'qgdz' in s.name and s.interaction != InteractionMode.REMOVED:
                        print(f"  {s.name:35s} ({s.x:3d},{s.y:3d}) {s.width:2d}x{s.height:<2d} int={s.interaction.name} tags={s.tags}")

    # Restore
    restore_game_state(game, state0)
    print(f"\nRestored to initial. Player at ({game.fdvakicpimr.x},{game.fdvakicpimr.y})")

if __name__ == "__main__":
    main()
