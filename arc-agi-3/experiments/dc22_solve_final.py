#!/usr/bin/env python3
"""
dc22 Final Solver — Online A* with crane decomposition.

For simple levels (no crane): standard A* with manhattan heuristic.
For crane levels: decomposed search — enumerate crane configurations,
then player-only A* for each.
"""

import os, sys, time, json, copy, heapq
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
import numpy as np
from collections import deque
from PIL import Image

VISUAL_DIR = "/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/dc22"
os.makedirs(VISUAL_DIR, exist_ok=True)

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
    """Save game state using the game's native snapshot mechanism (fast clone).

    The returned dict owns its sprite clones — it will not be affected by any
    future game operation, even an internal checkpoint restore (ycfbtkckze).
    """
    # Use a direct snapshot of the level's current sprites, clone each out
    safe_sprites = []
    for s in game.current_level.get_sprites():
        c = s.clone()
        c.set_position(s.x, s.y)
        c.set_interaction(s.interaction)
        c._blocking = s._blocking
        c._tags = list(s.tags)
        safe_sprites.append(c)
    native = {
        'nxhz_x': game.nxhz_x,
        'nxhz_y': game.nxhz_y,
        'nxhz_attached_kind': game.nxhz_attached_kind,
        'attached_hhxv_prefix': game.attached_hhxv_prefix,
        'attached_hhxv_x': game.attached_hhxv_x,
        'attached_hhxv_y': game.attached_hhxv_y,
    }
    return {
        'sprites': safe_sprites,  # independent clones
        'nxhz_x': native['nxhz_x'],
        'nxhz_y': native['nxhz_y'],
        'nxhz_attached_kind': native['nxhz_attached_kind'],
        'attached_hhxv_prefix': native['attached_hhxv_prefix'],
        'attached_hhxv_x': native['attached_hhxv_x'],
        'attached_hhxv_y': native['attached_hhxv_y'],
        'step_counter': game.step_counter_ui.rdnpeqedga,
        'uuehztercxf': game.uuehztercxf,
        'pxicvzkjuui': game.pxicvzkjuui,
        'prbjhwkkxth': game.prbjhwkkxth,
        'fgxfjbqnmgt': game.fgxfjbqnmgt,
        'zemyudjnnqd': game.zemyudjnnqd,
        'jnmawhhrfhh': game.jnmawhhrfhh,
        'dimvmykkjbg': game.dimvmykkjbg,
    }


def restore_game_state(game, state):
    """Restore game state. Clones saved sprites into fresh copies so the
    saved state is not mutated by subsequent env.step calls."""
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
        'nxhz_x': state['nxhz_x'],
        'nxhz_y': state['nxhz_y'],
        'nxhz_attached_kind': state['nxhz_attached_kind'],
        'attached_hhxv_prefix': state['attached_hhxv_prefix'],
        'attached_hhxv_x': state['attached_hhxv_x'],
        'attached_hhxv_y': state['attached_hhxv_y'],
    }
    game.ycfbtkckze()  # replaces level sprites with our fresh clones, calls qgjulrdonb
    game.step_counter_ui.rdnpeqedga = state['step_counter']
    game.step_counter_ui.inqipvidhq()
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
        try:
            game.dldhnotovw()
        except ValueError:
            pass
    elif state['nxhz_attached_kind'] == 'bynyvtuepbt-object' and game.ciuxrvkyndj:
        game.euqqhkqayni = game.ciuxrvkyndj


def get_state_key(game):
    """Compact hashable state."""
    p = game.fdvakicpimr
    sprite_states = []
    for s in game.current_level.get_sprites():
        if s.interaction == InteractionMode.REMOVED:
            continue
        for tag in ('wbze', 'jpug', 'zbhi', 'itki', 'pressure-plate'):
            if tag in s.tags:
                sprite_states.append((s.name, s.x, s.y, s.interaction.value, s.is_visible))
                break
    nxhz_state = ()
    if hasattr(game, 'nxhz_list') and game.nxhz_list:
        nxhz_state = (game.nxhz_x, game.nxhz_y, game.nxhz_attached_kind,
                      game.attached_hhxv_prefix, game.attached_hhxv_x, game.attached_hhxv_y)
    return (p.x, p.y, tuple(sorted(sprite_states)), nxhz_state)


def find_click_targets(game):
    """Find ALL clickable targets including invisible ones."""
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


def manhattan_dist(game):
    return abs(game.fdvakicpimr.x - game.bqxa.x) + abs(game.fdvakicpimr.y - game.bqxa.y)


def solve_level_astar(env, level_num, actions, max_nodes=200000, max_depth=250,
                      timeout=900, h_weight=1.0, initial_state=None):
    """A* solver with configurable action set and heuristic weight."""
    game = env._game
    t0 = time.time()

    if initial_state:
        restore_game_state(game, initial_state)

    init_key = get_state_key(game)
    init_h = manhattan_dist(game)
    counter = 0
    state0 = save_game_state(game)
    heap = [(h_weight * init_h, counter, [], state0)]
    visited = {init_key}
    expanded = 0
    best_h = init_h

    while heap:
        f_score, _, moves, saved_state = heapq.heappop(heap)

        if len(moves) >= max_depth:
            continue

        expanded += 1
        if expanded % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    expanded={expanded}, visited={len(visited)}, depth={len(moves)}, "
                  f"f={f_score:.1f}, best_h={best_h}, heap={len(heap)}, {elapsed:.1f}s")
            if elapsed > timeout:
                break

        if expanded > max_nodes:
            break

        for action in actions:
            restore_game_state(game, saved_state)

            if isinstance(action, tuple):
                fd = env.step(action[0], data=action[1])
                new_move = action
            else:
                fd = env.step(action)
                new_move = action

            if fd.levels_completed > level_num or fd.state.name == 'WIN':
                elapsed = time.time() - t0
                new_moves = moves + [new_move]
                print(f"  SOLVED L{level_num+1}! {len(new_moves)} moves, "
                      f"{expanded} expanded, {elapsed:.1f}s")
                save_frame(fd.frame, f"{VISUAL_DIR}/L{level_num+1}_solved.png")
                return new_moves

            if fd.state.name == 'LOSE':
                continue

            sk = get_state_key(game)
            if sk in visited:
                continue
            visited.add(sk)

            new_moves = moves + [new_move]
            g_cost = len(new_moves)
            h = manhattan_dist(game)
            if h < best_h:
                best_h = h
            f = g_cost + h_weight * h

            counter += 1
            heapq.heappush(heap, (f, counter, new_moves, save_game_state(game)))

    elapsed = time.time() - t0
    return None


def player_reachable_cells(game):
    """Pure-position floodfill: compute all cells the player can reach via
    movement alone (no clicks) from the current player position in the current
    board state. Does NOT call env.step, does NOT mutate game state (we save
    and restore player position).

    Returns: dict mapping (x,y) -> parent move (None for start), so that the
    move sequence to reach any cell can be reconstructed.
    """
    player = game.fdvakicpimr
    start = (player.x, player.y)

    # Save original position
    orig_x, orig_y = player.x, player.y

    # BFS. At each node: try 4 directions via try_move_sprite. On success,
    # check support at new pos; if no support, revert. Then record child and
    # continue from parent (we'll set_position to child when expanding it).
    parents = {start: None}
    order = [start]
    head = 0
    # Move deltas in pixels (2 per step) and corresponding GameAction
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
            # Move from (cx,cy)
            player.set_position(cx, cy)
            # Attempt move — try_move_sprite auto-reverts on collision
            collisions = game.try_move_sprite(player, dx, dy)
            if collisions:
                continue
            # Check support at new pos
            nx, ny = player.x, player.y
            if (nx, ny) == (cx, cy):
                continue
            if game.uxwpppoljm(nx, ny, player) is None:
                continue
            if (nx, ny) in parents:
                continue
            parents[(nx, ny)] = (cx, cy, act)
            order.append((nx, ny))

    # Restore player position
    player.set_position(orig_x, orig_y)
    return parents


def reconstruct_moves(parents, goal):
    """Given parent map from player_reachable_cells, reconstruct move list."""
    moves = []
    cur = goal
    while parents.get(cur) is not None:
        px, py, act = parents[cur]
        moves.append(act)
        cur = (px, py)
    moves.reverse()
    return moves


def solve_macro(env, level_num, timeout=1800, max_click_depth=8):
    """Macro-action A* solver for crane levels.

    State: board configuration after a sequence of clicks (crane buttons + jpug).
    Movement inside a configuration is FREE — we only branch on click actions.
    For each configuration we compute player-reachable cells via pure-position
    floodfill. If goal is reachable, we emit clicks + movement path.
    """
    game = env._game
    t0 = time.time()

    fd = env.observation_space
    save_frame(fd.frame, f"{VISUAL_DIR}/L{level_num+1}_start.png")

    p = game.fdvakicpimr
    goal_x, goal_y = game.bqxa.x, game.bqxa.y
    print(f"  [macro] Player: ({p.x},{p.y}), Goal: ({goal_x},{goal_y})")
    print(f"  [macro] Steps budget: {game.step_counter_ui.rdnpeqedga}")

    click_targets = find_click_targets(game)
    click_actions = []
    for kind, name, cx, cy in click_targets:
        click_actions.append((GameAction.ACTION6, {'x': cx, 'y': cy}, name))
    print(f"  [macro] {len(click_actions)} click targets")

    initial_state = save_game_state(game)

    # Check initial reachability first
    parents0 = player_reachable_cells(game)
    print(f"  [macro] Initial reach: {len(parents0)} cells")
    if (goal_x, goal_y) in parents0:
        moves = reconstruct_moves(parents0, (goal_x, goal_y))
        print(f"  [macro] Goal reachable without clicks: {len(moves)} moves")
        # Replay via env.step to leave state correct
        restore_game_state(game, initial_state)
        for m in moves:
            env.step(m)
        return moves

    def board_key(g):
        state = []
        for s in g.current_level.get_sprites():
            if s.interaction == InteractionMode.REMOVED:
                continue
            for tag in ('wbze', 'jpug', 'zbhi', 'itki', 'bynyvtuepbt-object'):
                if tag in s.tags:
                    state.append((s.name, s.x, s.y, s.interaction.value, s.is_visible))
                    break
        nxhz_state = (g.nxhz_x, g.nxhz_y, g.nxhz_attached_kind,
                      g.attached_hhxv_prefix, g.attached_hhxv_x, g.attached_hhxv_y)
        # Include player pos (changes what's reachable)
        return ((g.fdvakicpimr.x, g.fdvakicpimr.y), tuple(sorted(state)), nxhz_state)

    # Macro BFS. Each node is a (sequence, state). Moves in sequence are
    # either raw actions (clicks or walk directions). The transitions are:
    #
    #   A) Click from current player pos (affects board)
    #   B) Walk-to macro: walk to a reachable cell that contains zbhi or itki,
    #      executing real env.step to trigger any side effects
    #   C) Walk-to-itki + immediately click a matching-letter jpug (teleport)
    #
    # Pruning by board_key (which includes player pos) ensures we don't revisit.

    # Pre-index the special cells: zbhi positions and itki positions with letter
    def find_zbhi_cells():
        out = []
        for s in game.current_level.get_sprites():
            if 'zbhi' in s.tags and s.interaction != InteractionMode.REMOVED:
                # Player 2x2 must overlap zbhi; pick any 2x2 position inside
                for dx in range(max(1, s.width - 1)):
                    for dy in range(max(1, s.height - 1)):
                        out.append((s.x + dx, s.y + dy))
        return out

    def find_itki_info():
        """Return list of (pos, letter_tag, matching_jpug_click_data)."""
        letter_to_click = {}
        for kind, name, cx, cy in click_targets:
            if kind != 'jpug':
                continue
            for s in game.current_level.get_sprites():
                if s.name == name:
                    for tag in s.tags:
                        if len(tag) == 1:
                            letter_to_click[tag] = (cx, cy)
                    break
        out = []
        for s in game.current_level.get_sprites():
            if 'itki' not in s.tags or s.interaction == InteractionMode.REMOVED:
                continue
            letter = None
            for tag in s.tags:
                if len(tag) == 1:
                    letter = tag
                    break
            if letter and letter in letter_to_click:
                out.append(((s.x, s.y), letter, letter_to_click[letter]))
        return out

    restore_game_state(game, initial_state)
    init_board = board_key(game)
    special_zbhi = find_zbhi_cells()
    special_itki = find_itki_info()
    print(f"  [macro] zbhi cells: {special_zbhi}")
    print(f"  [macro] itki info: {special_itki}")

    frontier = deque()
    frontier.append(([], initial_state, parents0))
    visited_boards = {init_board}
    nodes = 0
    last_report = t0

    def try_solution(click_seq_and_walks, reach_parents):
        """If goal in reach_parents, emit full move list and return it."""
        if (goal_x, goal_y) in reach_parents:
            final_walk = reconstruct_moves(reach_parents, (goal_x, goal_y))
            combined = list(click_seq_and_walks) + final_walk
            print(f"  [macro] SOLVED! {len(combined)} total actions")
            restore_game_state(game, initial_state)
            for m in combined:
                if isinstance(m, tuple):
                    env.step(m[0], data=m[1])
                else:
                    env.step(m)
            fd_final = env.observation_space
            save_frame(fd_final.frame, f"{VISUAL_DIR}/L{level_num+1}_solved.png")
            return combined
        return None

    while frontier and time.time() - t0 < timeout:
        moves_so_far, saved_state, reach_parents = frontier.popleft()
        nodes += 1

        now = time.time()
        if now - last_report > 10:
            last_report = now
            print(f"    [macro] nodes={nodes}, queue={len(frontier)}, depth={len(moves_so_far)}, visited={len(visited_boards)}, {now-t0:.1f}s")

        if len(moves_so_far) >= max_click_depth * 4:
            continue

        # Transition A: try each click from current state
        for (act, data, cname) in click_actions:
            restore_game_state(game, saved_state)
            fd_c = env.step(act, data=data)
            if fd_c.state.name == 'LOSE':
                continue
            if fd_c.levels_completed > level_num:
                combined = list(moves_so_far) + [(act, data)]
                print(f"  [macro] SOLVED by click: {len(combined)} moves")
                save_frame(fd_c.frame, f"{VISUAL_DIR}/L{level_num+1}_solved.png")
                return combined

            bk = board_key(game)
            if bk in visited_boards:
                continue
            visited_boards.add(bk)

            new_state = save_game_state(game)
            new_reach = player_reachable_cells(game)
            new_moves = list(moves_so_far) + [(act, data)]

            sol = try_solution(new_moves, new_reach)
            if sol:
                return sol

            frontier.append((new_moves, new_state, new_reach))

        # Transition B0: walk to any "stable" cell (supported by a sprite that
        # is not the crane-attached object and not a toggleable wbze). Allows
        # the player to park safely while the crane moves the object elsewhere.
        stable_targets = set()
        for cell in reach_parents.keys():
            if cell == reach_parents and cell in (None,):
                continue
            # classify by support sprite
            restore_game_state(game, saved_state)
            game.fdvakicpimr.set_position(*cell)
            supp = game.uxwpppoljm(cell[0], cell[1], game.fdvakicpimr)
            if supp is None:
                continue
            name = supp.name
            tags = supp.tags
            # Stable = kbqq (plain floor tiles), not wbze, not bynyvtuepbt-object
            if name.startswith('kbqq') and 'bynyvtuepbt-object' not in tags and 'wbze' not in tags:
                stable_targets.add(cell)
        # restore state after the probing
        restore_game_state(game, saved_state)
        for stable in stable_targets:
            if stable == (game.fdvakicpimr.x, game.fdvakicpimr.y):
                continue
            restore_game_state(game, saved_state)
            walk = reconstruct_moves(reach_parents, stable)
            walk_ok = True
            for m in walk:
                fd_w = env.step(m)
                if fd_w.state.name == 'LOSE':
                    walk_ok = False
                    break
                if fd_w.levels_completed > level_num:
                    combined = list(moves_so_far) + walk
                    return combined
            if not walk_ok:
                continue
            bk = board_key(game)
            if bk in visited_boards:
                continue
            visited_boards.add(bk)
            new_state = save_game_state(game)
            new_reach = player_reachable_cells(game)
            new_moves = list(moves_so_far) + walk
            sol = try_solution(new_moves, new_reach)
            if sol:
                return sol
            frontier.append((new_moves, new_state, new_reach))

        # Transition B: walk to any zbhi cell in reach
        for zcell in special_zbhi:
            if zcell not in reach_parents:
                continue
            # Execute walk in real env
            restore_game_state(game, saved_state)
            walk = reconstruct_moves(reach_parents, zcell)
            walk_ok = True
            for m in walk:
                fd_w = env.step(m)
                if fd_w.state.name == 'LOSE':
                    walk_ok = False
                    break
                if fd_w.levels_completed > level_num:
                    combined = list(moves_so_far) + walk
                    print(f"  [macro] SOLVED by walk: {len(combined)} moves")
                    save_frame(fd_w.frame, f"{VISUAL_DIR}/L{level_num+1}_solved.png")
                    return combined
            if not walk_ok:
                continue
            bk = board_key(game)
            if bk in visited_boards:
                continue
            visited_boards.add(bk)
            new_state = save_game_state(game)
            new_reach = player_reachable_cells(game)
            new_moves = list(moves_so_far) + walk
            sol = try_solution(new_moves, new_reach)
            if sol:
                return sol
            frontier.append((new_moves, new_state, new_reach))

        # Transition C: walk to itki cell + click matching jpug (teleport macro)
        for (ipos, letter, (cx, cy)) in special_itki:
            if ipos not in reach_parents:
                continue
            restore_game_state(game, saved_state)
            walk = reconstruct_moves(reach_parents, ipos)
            walk_ok = True
            for m in walk:
                fd_w = env.step(m)
                if fd_w.state.name == 'LOSE':
                    walk_ok = False
                    break
                if fd_w.levels_completed > level_num:
                    combined = list(moves_so_far) + walk
                    return combined
            if not walk_ok:
                continue
            # Click the letter jpug (player on itki → teleport)
            fd_c = env.step(GameAction.ACTION6, data={'x': cx, 'y': cy})
            if fd_c.state.name == 'LOSE':
                continue
            if fd_c.levels_completed > level_num:
                combined = list(moves_so_far) + walk + [(GameAction.ACTION6, {'x': cx, 'y': cy})]
                return combined
            bk = board_key(game)
            if bk in visited_boards:
                continue
            visited_boards.add(bk)
            new_state = save_game_state(game)
            new_reach = player_reachable_cells(game)
            new_moves = list(moves_so_far) + walk + [(GameAction.ACTION6, {'x': cx, 'y': cy})]
            sol = try_solution(new_moves, new_reach)
            if sol:
                return sol
            frontier.append((new_moves, new_state, new_reach))

    elapsed = time.time() - t0
    print(f"  [macro] Failed. nodes={nodes}, visited={len(visited_boards)}, {elapsed:.1f}s")
    restore_game_state(game, initial_state)
    return None


def solve_crane_level(env, level_num, timeout=1800):
    """Solve a crane level by BFS over crane+click states, then player A* for each.

    Strategy:
    1. BFS over (crane_actions + jpug_clicks) to explore board configurations
    2. For each configuration, try player-only A* to reach the goal
    3. Return combined crane_sequence + player_sequence
    """
    game = env._game
    t0 = time.time()

    fd = env.observation_space
    save_frame(fd.frame, f"{VISUAL_DIR}/L{level_num+1}_start.png")

    p = game.fdvakicpimr
    goal = (game.bqxa.x, game.bqxa.y)
    print(f"  Player: ({p.x},{p.y}), Goal: {goal}")
    print(f"  Steps budget: {game.step_counter_ui.rdnpeqedga}")

    click_targets = find_click_targets(game)

    # Separate crane controls from jpug buttons
    crane_actions = []
    jpug_actions = []
    for kind, name, cx, cy in click_targets:
        action = (GameAction.ACTION6, {'x': cx, 'y': cy})
        if kind == 'sys_click':
            crane_actions.append(action)
            print(f"    crane: {name} at ({cx},{cy})")
        else:
            jpug_actions.append(action)
            print(f"    jpug: {name} at ({cx},{cy})")

    move_actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
    player_actions = list(move_actions) + jpug_actions  # player can move + click jpug buttons

    # All non-player actions for board setup
    setup_actions = crane_actions + jpug_actions

    initial_state = save_game_state(game)

    # Phase 1: BFS over board configurations (crane + jpug only, no player movement)
    # Each node is a sequence of setup actions applied from the initial state
    print(f"\n  Phase 1: Enumerating board configurations...")

    def get_board_key(g):
        """Board state without player position."""
        sprite_states = []
        for s in g.current_level.get_sprites():
            if s.interaction == InteractionMode.REMOVED:
                continue
            for tag in ('wbze', 'jpug', 'zbhi', 'itki'):
                if tag in s.tags:
                    sprite_states.append((s.name, s.x, s.y, s.interaction.value, s.is_visible))
                    break
        nxhz_state = ()
        if hasattr(g, 'nxhz_list') and g.nxhz_list:
            nxhz_state = (g.nxhz_x, g.nxhz_y, g.nxhz_attached_kind,
                          g.attached_hhxv_prefix, g.attached_hhxv_x, g.attached_hhxv_y)
        return (tuple(sorted(sprite_states)), nxhz_state)

    # BFS over setup actions
    restore_game_state(game, initial_state)
    init_board = get_board_key(game)
    board_configs = [([], initial_state, init_board)]  # (setup_moves, state, board_key)
    board_visited = {init_board}
    setup_queue = deque([([], initial_state)])
    configs_found = 0
    max_setup_depth = 10  # limit crane sequence length
    last_report = t0

    while setup_queue and time.time() - t0 < timeout * 0.5:
        setup_moves, setup_state = setup_queue.popleft()

        if len(setup_moves) >= max_setup_depth:
            continue

        now = time.time()
        if now - last_report > 10:
            last_report = now
            print(f"    Phase1: configs={len(board_configs)}, queue={len(setup_queue)}, depth={len(setup_moves)}, {now-t0:.1f}s")

        for action in setup_actions:
            restore_game_state(game, setup_state)
            fd = env.step(action[0], data=action[1])

            if fd.state.name == 'LOSE':
                continue

            bk = get_board_key(game)
            if bk in board_visited:
                continue
            board_visited.add(bk)

            new_setup = setup_moves + [action]
            new_state = save_game_state(game)
            board_configs.append((new_setup, new_state, bk))
            setup_queue.append((new_setup, new_state))
            configs_found += 1

    elapsed = time.time() - t0
    print(f"  Found {len(board_configs)} board configurations in {elapsed:.1f}s")

    # Phase 2: For each board config, check reachability then try player A*
    print(f"\n  Phase 2: Reachability check + Player A* for each configuration...")

    # Sort configs by setup length (try simpler setups first)
    board_configs.sort(key=lambda x: len(x[0]))

    def check_goal_reachable(env_ref, game_ref, goal_x, goal_y, state_ref):
        """Quick flood-fill reachability check using actual movement simulation.

        Tests all 4 directions from each position. Ignores clicks — just movement.
        Returns (reachable, min_distance) for early termination.
        """
        restore_game_state(game_ref, state_ref)
        px, py = game_ref.fdvakicpimr.x, game_ref.fdvakicpimr.y
        visited_pos = {(px, py)}
        queue_pos = [(px, py)]
        min_dist = abs(px - goal_x) + abs(py - goal_y)

        while queue_pos:
            cx, cy = queue_pos.pop(0)

            for act in move_actions:
                restore_game_state(game_ref, state_ref)
                game_ref.fdvakicpimr.set_position(cx, cy)
                fd_check = env_ref.step(act)
                nx = game_ref.fdvakicpimr.x
                ny = game_ref.fdvakicpimr.y

                if (nx, ny) not in visited_pos and (nx != cx or ny != cy):
                    visited_pos.add((nx, ny))
                    d = abs(nx - goal_x) + abs(ny - goal_y)
                    if d < min_dist:
                        min_dist = d
                    if nx == goal_x and ny == goal_y:
                        return True, 0
                    queue_pos.append((nx, ny))

        return (min_dist == 0), min_dist

    best_solution = None
    configs_tested = 0
    reachable_count = 0

    for setup_moves, setup_state, bk in board_configs:
        if time.time() - t0 > timeout:
            break

        configs_tested += 1
        restore_game_state(game, setup_state)

        # Quick reachability check via actual movement simulation
        goal_x, goal_y = game.bqxa.x, game.bqxa.y
        reachable, min_d = check_goal_reachable(env, game, goal_x, goal_y, setup_state)
        if not reachable:
            if configs_tested % 200 == 0:
                elapsed = time.time() - t0
                print(f"    Tested {configs_tested}/{len(board_configs)} configs, "
                      f"{reachable_count} reachable, {elapsed:.1f}s")
            continue

        reachable_count += 1
        print(f"    Config #{configs_tested} ({len(setup_moves)} setup moves): "
              f"GOAL REACHABLE! Trying A*...")

        remaining_time = timeout - (time.time() - t0)
        sol = solve_level_astar(
            env, level_num, player_actions,
            max_nodes=50000,
            max_depth=200 - len(setup_moves),
            timeout=min(remaining_time, 120),
            h_weight=1.0,
            initial_state=setup_state,
        )

        if sol:
            combined = setup_moves + sol
            elapsed = time.time() - t0
            print(f"  SOLVED with config #{configs_tested} "
                  f"({len(setup_moves)} setup + {len(sol)} player = {len(combined)} total), "
                  f"{elapsed:.1f}s")
            save_frame(fd.frame, f"{VISUAL_DIR}/L{level_num+1}_solved.png")
            best_solution = combined
            break

        if configs_tested % 50 == 0:
            elapsed = time.time() - t0
            print(f"    Tested {configs_tested}/{len(board_configs)} configs, {elapsed:.1f}s")

    if best_solution:
        # Restore and replay the solution to leave game in correct state
        restore_game_state(game, initial_state)
        for m in best_solution:
            if isinstance(m, tuple):
                fd = env.step(m[0], data=m[1])
            else:
                fd = env.step(m)
        return best_solution

    elapsed = time.time() - t0
    print(f"  Crane level failed. Tested {configs_tested} configs in {elapsed:.1f}s")
    restore_game_state(game, initial_state)
    return None


def main():
    arcade = Arcade(operation_mode='offline')
    env = arcade.make('dc22-4c9bff3e')
    obs = env.reset()

    print(f"DC22: {obs.win_levels} levels")
    save_frame(obs.frame, f"{VISUAL_DIR}/initial.png")

    action_names = {
        GameAction.ACTION1: 'U', GameAction.ACTION2: 'D',
        GameAction.ACTION3: 'L', GameAction.ACTION4: 'R',
    }

    all_solutions = []
    levels_solved = 0

    # Load cached solutions if available — replay to skip already-solved levels
    cache_path = f"{VISUAL_DIR}/solutions.json"
    cached = []
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                raw = json.load(f)
            act_map = {1: GameAction.ACTION1, 2: GameAction.ACTION2,
                       3: GameAction.ACTION3, 4: GameAction.ACTION4,
                       6: GameAction.ACTION6}
            for lvl in raw:
                decoded = []
                for m in lvl:
                    a = act_map[m['action']]
                    if 'data' in m:
                        decoded.append((a, m['data']))
                    else:
                        decoded.append(a)
                cached.append(decoded)
            print(f"Loaded {len(cached)} cached level solutions")
        except Exception as e:
            print(f"Cache load failed: {e}")
            cached = []

    def save_solutions_incremental():
        enc = []
        for sol in all_solutions:
            lvl = []
            for m in sol:
                if isinstance(m, tuple):
                    lvl.append({'action': m[0].value, 'data': m[1]})
                else:
                    lvl.append({'action': m.value})
            enc.append(lvl)
        with open(cache_path, 'w') as f:
            json.dump(enc, f, indent=2)

    for level in range(obs.win_levels):
        print(f"\n{'='*60}")
        print(f"=== DC22 Level {level+1} (index {level}) ===")
        print(f"{'='*60}")

        game = env._game

        # Try cached replay first
        if level < len(cached):
            print(f"  Replaying cached solution ({len(cached[level])} moves)...")
            replay_ok = True
            for m in cached[level]:
                if isinstance(m, tuple):
                    fd = env.step(m[0], data=m[1])
                else:
                    fd = env.step(m)
                if fd.state.name == 'LOSE':
                    replay_ok = False
                    break
            fd = env.observation_space
            if replay_ok and fd.levels_completed > level:
                print(f"  Cached replay succeeded")
                all_solutions.append(cached[level])
                levels_solved += 1
                save_solutions_incremental()
                if fd.state.name == 'WIN':
                    save_frame(fd.frame, f"{VISUAL_DIR}/game_won.png")
                    print("  GAME WON!")
                    break
                continue
            else:
                print(f"  Cached replay failed — will solve fresh (levels_completed={fd.levels_completed})")
                # Note: env state is now corrupted. Reset and re-replay earlier levels.
                obs = env.reset()
                for i in range(level):
                    for m in all_solutions[i]:
                        if isinstance(m, tuple):
                            env.step(m[0], data=m[1])
                        else:
                            env.step(m)
                game = env._game

        has_crane = hasattr(game, 'nxhz_list') and len(game.nxhz_list) > 0

        if has_crane:
            # First try the macro-action solver (pure-position reachability +
            # click BFS). Cheap and expressive — clicks become the only
            # branching factor, movement is free within a reach zone.
            print(f"  Crane level — trying macro solver first")
            fd = env.observation_space
            save_frame(fd.frame, f"{VISUAL_DIR}/L{level+1}_start.png")
            level_start_state = save_game_state(game)
            sol = solve_macro(env, level, timeout=900, max_click_depth=10)
            if sol is None:
                restore_game_state(game, level_start_state)
                print(f"  Macro solver failed — falling back to A*")
            else:
                all_solutions.append(sol)
                levels_solved += 1
                save_solutions_incremental()
                sol_str = ''
                for m in sol:
                    if isinstance(m, tuple):
                        sol_str += f'C({m[1]["x"]},{m[1]["y"]})'
                    else:
                        sol_str += action_names.get(m, '?')
                print(f"  Solution ({len(sol)} moves): {sol_str[:200]}")
                obs = env.observation_space
                print(f"  Current state: levels_completed={obs.levels_completed}")
                if obs.state.name == 'WIN':
                    print("  GAME WON!")
                    save_frame(obs.frame, f"{VISUAL_DIR}/game_won.png")
                    break
                continue

            # Fast path: try plain A* with move+jpug only (skip crane actions)
            # Many "crane" levels are solvable without actually moving the crane.
            print(f"  Crane level detected — trying plain A* (move+jpug only) first")
            click_targets = find_click_targets(game)
            plain_actions = [GameAction.ACTION1, GameAction.ACTION2,
                            GameAction.ACTION3, GameAction.ACTION4]
            for kind, name, cx, cy in click_targets:
                if kind == 'jpug':
                    plain_actions.append((GameAction.ACTION6, {'x': cx, 'y': cy}))
            fd = env.observation_space
            save_frame(fd.frame, f"{VISUAL_DIR}/L{level+1}_start.png")
            p = game.fdvakicpimr
            print(f"  Player: ({p.x},{p.y}), Goal: ({game.bqxa.x},{game.bqxa.y})")
            print(f"  Steps budget: {game.step_counter_ui.rdnpeqedga}")
            print(f"  Plain action set: {len(plain_actions)} actions")
            # Save pristine level-start state so we can restore after a failed attempt
            level_start_state = save_game_state(game)
            # Try greedy plain A* (weight=2) first for speed
            sol = solve_level_astar(env, level, plain_actions,
                                   max_nodes=200000, max_depth=400,
                                   timeout=300, h_weight=2.0,
                                   initial_state=level_start_state)

            if sol is None:
                print(f"  Greedy plain A* failed — trying with crane actions included")
                restore_game_state(game, level_start_state)
                full_actions = list(plain_actions)
                for kind, name, cx, cy in click_targets:
                    if kind == 'sys_click':
                        full_actions.append((GameAction.ACTION6, {'x': cx, 'y': cy}))
                sol = solve_level_astar(env, level, full_actions,
                                       max_nodes=500000, max_depth=600,
                                       timeout=1500, h_weight=2.0,
                                       initial_state=level_start_state)

            if sol is None:
                print(f"  Weighted joint A* failed — falling back to crane decomposition")
                restore_game_state(game, level_start_state)
                sol = solve_crane_level(env, level, timeout=1200)
            else:
                # Replay the solution to leave game in post-solve state
                restore_game_state(game, level_start_state)
                for m in sol:
                    if isinstance(m, tuple):
                        env.step(m[0], data=m[1])
                    else:
                        env.step(m)

            if sol is None:
                # Fallback: brute force A* with low h_weight
                print(f"  Decomposed search failed, trying weighted A*...")
                game = env._game  # re-get in case state changed
                all_actions = [GameAction.ACTION1, GameAction.ACTION2,
                              GameAction.ACTION3, GameAction.ACTION4]
                click_targets = find_click_targets(game)
                for _, _, cx, cy in click_targets:
                    all_actions.append((GameAction.ACTION6, {'x': cx, 'y': cy}))
                sol = solve_level_astar(env, level, all_actions,
                                       max_nodes=300000, max_depth=250,
                                       timeout=1800, h_weight=0.3)
        else:
            print(f"  No crane — using A*")
            click_targets = find_click_targets(game)
            all_actions = [GameAction.ACTION1, GameAction.ACTION2,
                          GameAction.ACTION3, GameAction.ACTION4]
            for _, _, cx, cy in click_targets:
                all_actions.append((GameAction.ACTION6, {'x': cx, 'y': cy}))

            fd = env.observation_space
            save_frame(fd.frame, f"{VISUAL_DIR}/L{level+1}_start.png")

            p = game.fdvakicpimr
            print(f"  Player: ({p.x},{p.y}), Goal: ({game.bqxa.x},{game.bqxa.y})")
            print(f"  Steps budget: {game.step_counter_ui.rdnpeqedga}")
            print(f"  Manhattan: {manhattan_dist(game)}")
            print(f"  Click targets: {len(click_targets)}")
            for ct in click_targets:
                print(f"    {ct[0]}: {ct[1]} at ({ct[2]},{ct[3]})")

            sol = solve_level_astar(env, level, all_actions, timeout=900)

            if sol is None:
                print(f"  A* failed, trying BFS...")
                sol = solve_level_astar(env, level, all_actions,
                                       timeout=900, h_weight=0.0)

        if sol is None:
            print(f"  FAILED on level {level+1}")
            break

        sol_str = ''
        for m in sol:
            if isinstance(m, tuple):
                sol_str += f'C({m[1]["x"]},{m[1]["y"]})'
            else:
                sol_str += action_names.get(m, '?')
        print(f"  Solution ({len(sol)} moves): {sol_str}")

        all_solutions.append(sol)
        levels_solved += 1
        save_solutions_incremental()

        obs = env.observation_space
        print(f"  Current state: levels_completed={obs.levels_completed}")

        if obs.state.name == 'WIN':
            print("  GAME WON!")
            save_frame(obs.frame, f"{VISUAL_DIR}/game_won.png")
            break

    print(f"\n{'='*60}")
    print(f"Final: {levels_solved}/{obs.win_levels} levels solved")

    solutions_encoded = []
    for sol in all_solutions:
        level_moves = []
        for m in sol:
            if isinstance(m, tuple):
                level_moves.append({'action': m[0].value, 'data': m[1]})
            else:
                level_moves.append({'action': m.value})
        solutions_encoded.append(level_moves)

    with open(f"{VISUAL_DIR}/solutions.json", 'w') as f:
        json.dump(solutions_encoded, f, indent=2)

    print(f"Solutions saved to {VISUAL_DIR}/solutions.json")


if __name__ == "__main__":
    main()
