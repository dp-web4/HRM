#!/usr/bin/env python3
"""
dc22 World-Model Solver — Engine-replay BFS.

WORLD MODEL (decoded from obfuscated dc22.py):
==============================================
Grid: variable size per level (camera 64x44, padded to 64x64 frame)
Actions: 1=up, 2=down, 3=left, 4=right, 6=click
Move step: qqsswktiui=2 pixels per directional action

OBJECTS:
- Player (tag "pcxjvnmybet"): 2x2 sprite, moves by 2px, must stay on intangible sprites
  If player steps off intangible ground -> death animation -> lose a life (step_counter -= 20)
- Goal (tag "bqxa"): 2x2 light blue sprite. Win = player reaches goal position
- Buttons (tag "jpug" + "sys_click"): clickable colored buttons, toggle sprite groups
  Each button has a single-char tag (a,b,c,...) linking it to a group
- wbze sprites: multi-state obstacles/platforms. Named prefix+number (e.g. wbze-efzv1, wbze-efzv2)
  Toggling cycles through states: current becomes REMOVED, next becomes TANGIBLE/INTANGIBLE
- bgeg sprites: similar toggleable platform/obstacle groups
- nxhz (cursor): appears in later levels. Has a separate position grid (nxhz_x, nxhz_y)
  Controlled by sys_click arrow buttons (zmjbupyjfyb=up, jffakyxiury=down, etc.)
  Can pick up/carry hhxv bridge pieces
- Pressure plates: when player stands on them, activate linked jpug sprites
- Path: intangible sprites that form walkable ground

KEY MECHANICS:
- Player MUST always be on an intangible sprite (uxwpppoljm check)
- Step counter decrements on each action. Lose when reaches 0.
- Falling costs 20 extra steps (yepbymuune)
- Clicking jpug buttons: cycles sprite states in that button's group
  - bpnwmawiuv ("itki-color-jpug"): cycles through 4 itki color variants
  - Otherwise: rotates between numbered sprite variants (1->2->3...->1)
- Win: player position == bqxa position

STRATEGY:
BFS with engine replay. State = (player_x, player_y, wbze/toggle states).
For levels without nxhz, this is straightforward.
For levels with nxhz, state includes cursor position and attached object.
"""

import os, sys, time, json
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
import numpy as np
from collections import deque
from PIL import Image

VISUAL_DIR = "shared-context/arc-agi-3/visual-memory/dc22"
os.makedirs(VISUAL_DIR, exist_ok=True)

# ARC color palette
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


def get_state_key(game):
    """Hashable state: player pos + all toggle-relevant sprite states."""
    p = game.fdvakicpimr
    toggles = []
    for s in game.current_level.get_sprites():
        if 'wbze' in s.tags or 'ordebgeg' in s.tags:
            toggles.append((s.name, s.x, s.y, s.interaction.value))
    # Also include nxhz position if present
    nxhz_state = ()
    if hasattr(game, 'nxhz_list') and game.nxhz_list:
        nxhz_state = (game.nxhz_x, game.nxhz_y, game.nxhz_attached_kind)
    return (p.x, p.y, tuple(sorted(toggles)), nxhz_state)


def find_click_targets(game):
    """Find all clickable elements and their display coordinates."""
    targets = []
    cam_h = game.camera._height  # typically 44
    frame_h = 64
    y_offset = (frame_h - cam_h) // 2

    # jpug buttons
    for s in game.current_level.get_sprites():
        if 'jpug' in s.tags and 'sys_click' in s.tags:
            cx = s.x + s.width // 2
            cy = s.y + s.height // 2 + y_offset
            targets.append(('jpug', s.name, cx, cy))

    # sys_click arrows (for nxhz cursor)
    for s in game.current_level.get_sprites():
        if 'sys_click' in s.tags and 'jpug' not in s.tags:
            cx = s.x + s.width // 2
            cy = s.y + s.height // 2 + y_offset
            tags_str = ','.join(s.tags)
            targets.append(('arrow', tags_str, cx, cy))

    return targets


def solve_level(env, game, level_num, all_flat_moves, max_depth=80):
    """Solve one dc22 level using BFS with engine replay.

    all_flat_moves: flat list of ALL moves executed so far (all previous levels).
    We replay this prefix, then explore from there.
    """
    t0 = time.time()
    prefix = list(all_flat_moves)  # copy

    def replay(moves):
        """Reset and replay prefix + moves."""
        env.reset()
        for m in prefix:
            if isinstance(m, tuple):
                env.step(m[0], data=m[1])
            else:
                env.step(m)
        for m in moves:
            if isinstance(m, tuple):
                env.step(m[0], data=m[1])
            else:
                env.step(m)

    replay([])

    # Save initial frame
    fd = env.observation_space
    save_frame(fd.frame, f"{VISUAL_DIR}/L{level_num+1}_start.png")

    p = game.fdvakicpimr
    print(f"  Player: ({p.x},{p.y}), Goal: ({game.bqxa.x},{game.bqxa.y})")
    print(f"  Steps: {game.step_counter_ui.current_steps}")

    # Find click targets
    click_targets = find_click_targets(game)
    print(f"  Click targets: {len(click_targets)}")
    for ct in click_targets:
        print(f"    {ct[0]}: {ct[1]} at display ({ct[2]},{ct[3]})")

    # Build action list
    move_actions = [
        GameAction.ACTION1,  # UP
        GameAction.ACTION2,  # DOWN
        GameAction.ACTION3,  # LEFT
        GameAction.ACTION4,  # RIGHT
    ]

    click_actions = []
    for _, name, cx, cy in click_targets:
        click_actions.append((GameAction.ACTION6, {'x': cx, 'y': cy}))

    all_actions = list(move_actions) + click_actions

    # BFS
    replay([])
    init_key = get_state_key(game)
    visited = {init_key}
    queue = deque([([], init_key)])
    expanded = 0

    while queue:
        moves, _ = queue.popleft()
        if len(moves) >= max_depth:
            continue

        expanded += 1
        if expanded % 2000 == 0:
            elapsed = time.time() - t0
            print(f"    expanded={expanded}, visited={len(visited)}, depth={len(moves)}, {elapsed:.1f}s")
            if elapsed > 600:  # 10 min timeout per level
                break

        for action in all_actions:
            replay(moves)

            if isinstance(action, tuple):
                fd = env.step(action[0], data=action[1])
                new_move = action
            else:
                fd = env.step(action)
                new_move = action

            if fd.levels_completed > level_num or fd.state.name == 'WIN':
                elapsed = time.time() - t0
                new_moves = moves + [new_move]
                print(f"  SOLVED! {len(new_moves)} moves, {expanded} expanded, {elapsed:.1f}s")
                save_frame(fd.frame, f"{VISUAL_DIR}/L{level_num+1}_solved.png")
                return new_moves

            if fd.state.name == 'LOSE':
                continue

            sk = get_state_key(game)
            if sk in visited:
                continue
            visited.add(sk)

            queue.append((moves + [new_move], sk))

    elapsed = time.time() - t0
    print(f"  No solution. {expanded} expanded, {len(visited)} visited, {elapsed:.1f}s")
    return None


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('dc22-4c9bff3e')
    obs = env.reset()
    game = env._game

    print(f"DC22: {obs.win_levels} levels")
    save_frame(obs.frame, f"{VISUAL_DIR}/initial.png")

    all_flat_moves = []  # ALL moves across all levels
    action_names = {
        GameAction.ACTION1: 'U', GameAction.ACTION2: 'D',
        GameAction.ACTION3: 'L', GameAction.ACTION4: 'R',
    }
    levels_solved = 0

    for level in range(obs.win_levels):
        print(f"\n=== DC22 Level {level+1} ===")

        sol = solve_level(env, game, level, all_flat_moves, max_depth=80)

        if sol is None:
            print(f"  FAILED on level {level+1}")
            break

        sol_str = ''
        for m in sol:
            if isinstance(m, tuple):
                sol_str += 'C'
            else:
                sol_str += action_names.get(m, '?')
        print(f"  Solution ({len(sol)} moves): {sol_str}")

        all_flat_moves.extend(sol)
        levels_solved += 1

        # Replay to verify
        env.reset()
        for m in all_flat_moves:
            if isinstance(m, tuple):
                fd = env.step(m[0], data=m[1])
            else:
                fd = env.step(m)

        print(f"  After replay: completed={fd.levels_completed}, state={fd.state.name}")

        if fd.state.name == 'WIN':
            print("  GAME WON!")
            break

    print(f"\nFinal: {levels_solved}/{obs.win_levels} levels solved")

    # Save flat moves
    moves_encoded = []
    for m in all_flat_moves:
        if isinstance(m, tuple):
            moves_encoded.append({'action': m[0].value, 'data': m[1]})
        else:
            moves_encoded.append({'action': m.value})

    with open(f"{VISUAL_DIR}/solutions.json", 'w') as f:
        json.dump({'levels_solved': levels_solved, 'moves': moves_encoded}, f, indent=2)


if __name__ == "__main__":
    main()
