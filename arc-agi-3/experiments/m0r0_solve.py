#!/usr/bin/env python3
"""Solve m0r0: mirrored cursor maze puzzle.

Two cursors move simultaneously with mirrored directions.
Win when all cursors pair up at the same position.
- ubwff-idtiq: same direction (dx, dy)
- ubwff-crkfz: mirror X (-dx, dy)

Mechanics:
- Blocks (cvcer): click to select, UDLR to move, click empty to deselect
- Hazards (wyiex): landing on one resets ALL cursors to start positions
- Toggle walls (dfnuk/hnutp): cursor on key opens same-color walls
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction, InteractionMode
import numpy as np
from collections import deque
from itertools import product

arcade = Arcade()

DIRS = {'UP': (0,-1), 'DOWN': (0,1), 'LEFT': (-1,0), 'RIGHT': (1,0)}
ACTION_MAP = {'UP': GameAction.ACTION1, 'DOWN': GameAction.ACTION2,
              'LEFT': GameAction.ACTION3, 'RIGHT': GameAction.ACTION4}

def get_maze_walls(level):
    """Get wall cells from jggua maze sprites."""
    walls = set()
    for ms in level.get_sprites_by_tag("jggua"):
        rendered = ms.render()
        for y in range(rendered.shape[0]):
            for x in range(rendered.shape[1]):
                if rendered[y, x] >= 0:
                    walls.add((ms.x + x, ms.y + y))
    return walls

def get_walkable(level):
    """Build walkability set."""
    grid_w, grid_h = level.grid_size
    walls = get_maze_walls(level)
    return {(x, y) for x in range(grid_w) for y in range(grid_h) if (x, y) not in walls}

def apply_move(pos, direction, mirror_type, walkable, extra_walls=frozenset()):
    """Apply a move to a cursor position, respecting walls."""
    x, y = pos
    dx, dy = DIRS[direction]
    if mirror_type == 'MIRROR_X':
        dx = -dx
    elif mirror_type == 'MIRROR_Y':
        dy = -dy
    elif mirror_type == 'MIRROR_XY':
        dx, dy = -dx, -dy
    nx, ny = x + dx, y + dy
    if (nx, ny) in walkable and (nx, ny) not in extra_walls:
        return (nx, ny)
    return (x, y)

def bfs_2cursor(walkable, start1, start2, mirror1, mirror2, hazards=frozenset(), extra_walls=frozenset()):
    """BFS to find path for 2 cursors to meet."""
    initial = (start1, start2)
    visited = {initial: None}
    queue = deque([initial])
    while queue:
        state = queue.popleft()
        pos1, pos2 = state
        for direction in DIRS:
            np1 = apply_move(pos1, direction, mirror1, walkable, extra_walls)
            np2 = apply_move(pos2, direction, mirror2, walkable, extra_walls)
            if np1 in hazards or np2 in hazards:
                continue
            new_state = (np1, np2)
            if np1 == np2:
                path = [direction]
                s = state
                while visited[s] is not None:
                    prev, d = visited[s]
                    path.append(d)
                    s = prev
                path.reverse()
                return path
            if new_state not in visited:
                visited[new_state] = (state, direction)
                queue.append(new_state)
    return None

def bfs_2cursor_with_toggles(walkable, start1, start2, mirror1, mirror2,
                              hazards, toggle_keys, toggle_walls):
    """BFS with toggle wall state tracking.

    toggle_keys: dict color -> set of (x,y) key positions
    toggle_walls: dict color -> set of (x,y) wall positions

    State: (pos1, pos2, toggle_state) where toggle_state is frozenset of open colors
    A color is open when either cursor is on one of its keys.
    """
    colors = sorted(toggle_keys.keys())

    def get_toggle_state(p1, p2):
        open_colors = set()
        for color in colors:
            for kx, ky in toggle_keys[color]:
                if p1 == (kx, ky) or p2 == (kx, ky):
                    open_colors.add(color)
                    break
        return frozenset(open_colors)

    def get_extra_walls(open_colors):
        walls = set()
        for color in colors:
            if color not in open_colors:
                walls.update(toggle_walls.get(color, set()))
        return frozenset(walls)

    initial_toggles = get_toggle_state(start1, start2)
    initial = (start1, start2, initial_toggles)
    visited = {initial: None}
    queue = deque([initial])

    while queue:
        state = queue.popleft()
        pos1, pos2, open_colors = state
        extra_walls = get_extra_walls(open_colors)

        for direction in DIRS:
            np1 = apply_move(pos1, direction, mirror1, walkable, extra_walls)
            np2 = apply_move(pos2, direction, mirror2, walkable, extra_walls)
            if np1 in hazards or np2 in hazards:
                continue
            new_toggles = get_toggle_state(np1, np2)
            new_state = (np1, np2, new_toggles)
            if np1 == np2:
                path = [direction]
                s = state
                while visited[s] is not None:
                    prev, d = visited[s]
                    path.append(d)
                    s = prev
                path.reverse()
                return path
            if new_state not in visited:
                visited[new_state] = (state, direction)
                queue.append(new_state)
    return None

def find_block_targets(walkable, blocks, start1, start2, mirror1, mirror2, hazards, grid_w, grid_h,
                        collidable_obstacles=frozenset()):
    """Find block positions that allow cursor BFS to succeed.

    Try moving each block to every reachable position and check if cursor BFS works.
    Returns (block_moves, cursor_path) or None.
    """
    block_list = sorted(blocks)
    n_blocks = len(block_list)

    # For each block, find all reachable positions (BFS of block movement)
    # Blocks collide with: jggua walls (not in walkable), nhiae (other blocks),
    # AND all other collidable sprites (hazards, dfnuk walls, cursors, keys)
    def block_reachable(block_pos, other_blocks, obstacles):
        """Find all positions a block can reach via UDLR moves."""
        visited = {block_pos}
        queue = deque([block_pos])
        while queue:
            bx, by = queue.popleft()
            for dx, dy in DIRS.values():
                nx, ny = bx + dx, by + dy
                if (nx, ny) in walkable and (nx, ny) not in other_blocks and (nx, ny) not in obstacles and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return visited

    # Get reachable positions for each block
    reachable = []
    for i, bp in enumerate(block_list):
        others = frozenset(block_list[:i] + block_list[i+1:])
        reach = block_reachable(bp, others, collidable_obstacles)
        reachable.append(sorted(reach))
        print(f"    Block {bp}: {len(reach)} reachable positions")

    # Try all combinations of block target positions
    best = None
    best_cost = float('inf')

    # Limit search if too many combinations
    total_combos = 1
    for r in reachable:
        total_combos *= len(r)

    if total_combos > 50000:
        # Too many combos - use heuristic: only try positions near cursor paths
        # First find path ignoring blocks entirely
        path_no_blocks = bfs_2cursor(walkable, start1, start2, mirror1, mirror2, hazards)
        if not path_no_blocks:
            return None

        # Get cells used by cursors on the path
        cursor_cells = set()
        p1, p2 = start1, start2
        for d in path_no_blocks:
            p1 = apply_move(p1, d, mirror1, walkable)
            p2 = apply_move(p2, d, mirror2, walkable)
            cursor_cells.add(p1)
            cursor_cells.add(p2)

        # For each block, only try positions NOT on cursor path
        for i in range(n_blocks):
            reachable[i] = [p for p in reachable[i] if p not in cursor_cells]
            if not reachable[i]:
                reachable[i] = [block_list[i]]  # keep original if no alternatives

    for combo in product(*reachable):
        target_blocks = frozenset(combo)
        # Check no two blocks at same position
        if len(target_blocks) < n_blocks:
            continue

        # Check cursor BFS with these block positions
        effective_walkable = walkable - target_blocks
        path = bfs_2cursor(effective_walkable, start1, start2, mirror1, mirror2, hazards)
        if path:
            # Calculate block movement cost
            block_cost = 0
            for i, (orig, tgt) in enumerate(zip(block_list, combo)):
                if orig != tgt:
                    # BFS to find shortest path for this block
                    others_at_target = frozenset(c for j, c in enumerate(combo) if j != i)
                    bv = {orig}
                    bq = deque([(orig, 0)])
                    found = False
                    while bq:
                        bp, cost = bq.popleft()
                        if bp == tgt:
                            block_cost += cost + 2  # +2 for click select + click deselect
                            found = True
                            break
                        for dx, dy in DIRS.values():
                            nx, ny = bp[0] + dx, bp[1] + dy
                            if (nx, ny) in walkable and (nx, ny) not in others_at_target and (nx, ny) not in collidable_obstacles and (nx, ny) not in bv:
                                bv.add((nx, ny))
                                bq.append(((nx, ny), cost + 1))
                    if not found:
                        block_cost = float('inf')
                        break

            total_cost = block_cost + len(path)
            if total_cost < best_cost:
                best_cost = total_cost
                best = (list(zip(block_list, combo)), path)

    return best

def plan_block_moves(orig, target, walkable, other_blocks, obstacles=frozenset()):
    """BFS to find path to move a block from orig to target, avoiding walls, blocks, and obstacles."""
    if orig == target:
        return []
    visited = {orig: None}
    queue = deque([orig])
    while queue:
        pos = queue.popleft()
        if pos == target:
            path = []
            while visited[pos] is not None:
                prev, d = visited[pos]
                path.append(d)
                pos = prev
            path.reverse()
            return path
        for dname, (dx, dy) in DIRS.items():
            nx, ny = pos[0] + dx, pos[1] + dy
            if (nx, ny) in walkable and (nx, ny) not in other_blocks and (nx, ny) not in obstacles and (nx, ny) not in visited:
                visited[(nx, ny)] = (pos, dname)
                queue.append((nx, ny))
    return None

def click_at_grid(env, game, gx, gy):
    """Click at a grid position. Returns fd."""
    cam = game.camera
    for dx_d in range(64):
        for dy_d in range(64):
            result = cam.display_to_grid(dx_d, dy_d)
            if result and result[0] == gx and result[1] == gy:
                return env.step(GameAction.ACTION6, data={'x': dx_d, 'y': dy_d})
    return None

def click_empty(env, game, walkable, occupied):
    """Click an empty walkable cell to deselect."""
    cam = game.camera
    # Try to find an empty walkable cell
    for dx_d in range(64):
        for dy_d in range(64):
            result = cam.display_to_grid(dx_d, dy_d)
            if result:
                gx, gy = result
                if (gx, gy) in walkable and (gx, gy) not in occupied:
                    # Make sure no sprite at this position
                    sprite_at = game.current_level.get_sprite_at(gx, gy, tag="sys_click")
                    if sprite_at is None:
                        return env.step(GameAction.ACTION6, data={'x': dx_d, 'y': dy_d})
    # Fallback: click corner
    return env.step(GameAction.ACTION6, data={'x': 0, 'y': 0})


# === Solve each level ===
env = arcade.make('m0r0-dadda488')
fd = env.reset()
game = env._game

total_actions = 0

for lv in range(6):
    level = game.current_level
    grid_w, grid_h = level.grid_size
    print(f"\n=== Level {lv} (grid {grid_w}x{grid_h}) ===")

    walkable = get_walkable(level)

    # Find cursors
    cursor_types = [
        ('qzfkx-ubwff-idtiq', 'SAME'),
        ('qzfkx-ubwff-crkfz', 'MIRROR_X'),
        ('qzfkx-kncqr-idtiq', 'MIRROR_Y'),
        ('qzfkx-kncqr-crkfz', 'MIRROR_XY'),
    ]
    cursors = []
    for name, mirror in cursor_types:
        for s in level.get_sprites_by_name(name):
            if s.is_visible:
                cursors.append((name, s.x, s.y, mirror))
                print(f"  Cursor: {name} at ({s.x},{s.y}) mirror={mirror}")

    # Find hazards
    hazards = frozenset((h.x, h.y) for h in level.get_sprites_by_name("wyiex"))
    if hazards:
        print(f"  Hazards: {len(hazards)}")

    # Find movable blocks
    blocks = set()
    for b in level.get_sprites_by_name("cvcer"):
        blocks.add((b.x, b.y))
        print(f"  Block: ({b.x},{b.y})")

    # Find toggle walls
    toggle_keys = {}  # color -> set of (x,y)
    toggle_walls = {}  # color -> set of (x,y)
    for color in ['raixb', 'ujcze', 'qeazm']:
        keys = level.get_sprites_by_name(f"hnutp-{color}")
        walls = level.get_sprites_by_name(f"dfnuk-{color}")
        if keys or walls:
            key_positions = set()
            wall_positions = set()
            for k in keys:
                key_positions.add((k.x, k.y))
            for w in walls:
                # dfnuk walls are 1x3 or 3x1 depending on rotation
                rendered = w.render()
                for wy in range(rendered.shape[0]):
                    for wx in range(rendered.shape[1]):
                        if rendered[wy, wx] >= 0:
                            wall_positions.add((w.x + wx, w.y + wy))
            if key_positions:
                toggle_keys[color] = key_positions
            if wall_positions:
                toggle_walls[color] = wall_positions
            print(f"  Toggle {color}: {len(key_positions)} keys at {key_positions}, {len(wall_positions)} wall cells")

    if len(cursors) != 2:
        print(f"  {len(cursors)} cursors - need multi-cursor solver")
        break

    c1_name, c1x, c1y, c1m = cursors[0]
    c2_name, c2x, c2y, c2m = cursors[1]
    start1, start2 = (c1x, c1y), (c2x, c2y)

    # Collect all collidable sprite positions for block movement
    # try_move_sprite checks collision with ALL collidable sprites
    all_collidable = set(hazards)
    all_collidable.add(start1)
    all_collidable.add(start2)
    for color in toggle_keys:
        all_collidable.update(toggle_keys[color])
    for color in toggle_walls:
        all_collidable.update(toggle_walls[color])

    lv_actions = 0
    path = None

    # Determine solver strategy based on level features
    has_blocks = len(blocks) > 0
    has_toggles = len(toggle_keys) > 0

    if has_toggles and not has_blocks:
        # Toggle wall solver
        print("  Strategy: BFS with toggle walls")
        path = bfs_2cursor_with_toggles(walkable, start1, start2, c1m, c2m,
                                         hazards, toggle_keys, toggle_walls)
        if path:
            print(f"  Cursor path: {len(path)} steps")
            for d in path:
                fd = env.step(ACTION_MAP[d])
                lv_actions += 1

    elif has_blocks and not has_toggles:
        # Block puzzle solver
        print("  Strategy: block movement + cursor BFS")

        # First try without moving blocks
        effective_walkable = walkable - blocks
        path = bfs_2cursor(effective_walkable, start1, start2, c1m, c2m, hazards)

        if path:
            print(f"  No block moves needed: {len(path)} steps")
            for d in path:
                fd = env.step(ACTION_MAP[d])
                lv_actions += 1
        else:
            # Need to move blocks
            result = find_block_targets(walkable, blocks, start1, start2, c1m, c2m, hazards, grid_w, grid_h,
                                        collidable_obstacles=frozenset(all_collidable))
            if result:
                block_moves, cursor_path = result
                print(f"  Block moves: {block_moves}")
                print(f"  Cursor path: {len(cursor_path)} steps")

                # Execute block moves
                current_blocks = set(blocks)
                for orig, target in block_moves:
                    if orig == target:
                        continue

                    # Plan path for this block
                    other_blocks = current_blocks - {orig}
                    bmoves = plan_block_moves(orig, target, walkable, other_blocks, frozenset(all_collidable))
                    if bmoves is None:
                        print(f"    FAILED to plan block {orig} -> {target}")
                        break

                    print(f"    Moving block {orig} -> {target}: {len(bmoves)} moves")

                    # Click block to select
                    fd = click_at_grid(env, game, orig[0], orig[1])
                    lv_actions += 1

                    # Move block
                    for d in bmoves:
                        fd = env.step(ACTION_MAP[d])
                        lv_actions += 1

                    # Click empty to deselect
                    current_blocks.discard(orig)
                    current_blocks.add(target)
                    fd = click_empty(env, game, walkable, current_blocks | {start1, start2})
                    lv_actions += 1

                # Execute cursor path
                path = cursor_path
                for d in cursor_path:
                    fd = env.step(ACTION_MAP[d])
                    lv_actions += 1
            else:
                print(f"  NO block configuration found")

    elif has_blocks and has_toggles:
        # Combined solver: interleaved block+cursor BFS
        # State: (cursor1, cursor2, block_pos, toggle_state, block_selected)
        print("  Strategy: interleaved block+cursor+toggle BFS")

        colors = sorted(toggle_keys.keys())

        def get_toggle_st(p1, p2):
            open_c = set()
            for color in colors:
                for kx, ky in toggle_keys[color]:
                    if p1 == (kx, ky) or p2 == (kx, ky):
                        open_c.add(color)
                        break
            return frozenset(open_c)

        def get_closed_walls(open_c):
            w = set()
            for color in colors:
                if color not in open_c:
                    w.update(toggle_walls.get(color, set()))
            return frozenset(w)

        # Static collidable for block movement (hazards + keys)
        # Toggle walls are dynamic — checked per-state based on toggle state
        block_static_obstacles = set(hazards)
        for color in toggle_keys:
            block_static_obstacles.update(toggle_keys[color])

        init_toggles = get_toggle_st(start1, start2)
        block_pos = sorted(blocks)[0]  # single block
        # State: (c1, c2, block, toggles, selected)
        # selected: False = mirror mode, True = block selected
        init_state = (start1, start2, block_pos, init_toggles, False)
        visited = {init_state: None}
        queue = deque([init_state])

        # Actions: UP/DOWN/LEFT/RIGHT (move cursors or block depending on selected)
        #          CLICK_BLOCK, CLICK_EMPTY (toggle selection)
        found_path = None

        while queue:
            state = queue.popleft()
            c1, c2, bp, toggles, selected = state

            # Try movement actions
            for dname in DIRS:
                dx, dy = DIRS[dname]
                if selected:
                    # Block movement mode - only block moves
                    nbx, nby = bp[0] + dx, bp[1] + dy
                    if (nbx, nby) not in walkable:
                        continue
                    # Block collides with: static obstacles + closed toggle walls + cursors
                    closed_walls_for_block = get_closed_walls(toggles)
                    if (nbx, nby) in block_static_obstacles or (nbx, nby) in closed_walls_for_block or (nbx, nby) == c1 or (nbx, nby) == c2:
                        continue
                    new_state = (c1, c2, (nbx, nby), toggles, True)
                else:
                    # Mirror cursor movement
                    closed_walls = get_closed_walls(toggles)
                    all_walls = closed_walls | frozenset({bp})  # block is also a wall for cursors
                    nc1 = apply_move(c1, dname, c1m, walkable, all_walls)
                    nc2 = apply_move(c2, dname, c2m, walkable, all_walls)
                    if nc1 in hazards or nc2 in hazards:
                        continue
                    if nc1 == nc2:
                        # WIN! Reconstruct path
                        found_path = [dname]
                        s = state
                        while visited[s] is not None:
                            prev, action = visited[s]
                            found_path.append(action)
                            s = prev
                        found_path.reverse()
                        break
                    new_toggles = get_toggle_st(nc1, nc2)
                    new_state = (nc1, nc2, bp, new_toggles, False)

                if new_state not in visited:
                    visited[new_state] = (state, dname)
                    queue.append(new_state)

            if found_path:
                break

            # Try click actions
            if not selected:
                # Click block to select
                new_state = (c1, c2, bp, toggles, True)
                if new_state not in visited:
                    visited[new_state] = (state, 'CLICK_BLOCK')
                    queue.append(new_state)
            else:
                # Click empty to deselect - toggle state recomputed
                new_toggles = get_toggle_st(c1, c2)
                new_state = (c1, c2, bp, new_toggles, False)
                if new_state not in visited:
                    visited[new_state] = (state, 'CLICK_EMPTY')
                    queue.append(new_state)

        if found_path:
            path = found_path
            print(f"  Solution: {len(path)} actions = {' '.join(path)}")

            # Execute
            for action in path:
                if action == 'CLICK_BLOCK':
                    # Find current block position from game state
                    block_sprites = level.get_sprites_by_name("cvcer")
                    bs = block_sprites[0]
                    fd = click_at_grid(env, game, bs.x, bs.y)
                    lv_actions += 1
                elif action == 'CLICK_EMPTY':
                    current_block_pos = set()
                    for bs in level.get_sprites_by_name("cvcer"):
                        current_block_pos.add((bs.x, bs.y))
                    fd = click_empty(env, game, walkable, current_block_pos)
                    lv_actions += 1
                else:
                    fd = env.step(ACTION_MAP[action])
                    lv_actions += 1
        else:
            print(f"  NO solution found ({len(visited)} states explored)")

    else:
        # Simple BFS
        print("  Strategy: simple BFS")
        path = bfs_2cursor(walkable, start1, start2, c1m, c2m, hazards)
        if path:
            print(f"  Cursor path: {len(path)} steps")
            for d in path:
                fd = env.step(ACTION_MAP[d])
                lv_actions += 1

    if path:
        total_actions += lv_actions
        print(f"  {lv_actions} actions, completed={fd.levels_completed}, state={fd.state.name}")
    else:
        print(f"  NO SOLUTION")
        break

    if fd.state.name == 'WIN':
        print(f"\n=== ALL SOLVED! Total: {total_actions} ===")
        break

    if fd.levels_completed <= lv:
        print(f"  LEVEL NOT COMPLETED - debugging...")
        break

print(f"\nFinal: completed={fd.levels_completed}, state={fd.state.name}, total={total_actions}")
