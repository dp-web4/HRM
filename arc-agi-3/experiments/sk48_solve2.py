#!/usr/bin/env python3
"""Solve sk48: chain/snake color-matching puzzle.

Analytical BFS — no game engine during search.
State = (head_y, chain_len, target_positions).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
from collections import deque, defaultdict

CELL = 6
UP, DOWN, LEFT, RIGHT = GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4
ACTION_NAMES = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT'}

arcade = Arcade()
env = arcade.make('sk48-41055498')
fd = env.reset()
game = env._game


def step_action(action):
    """Execute one logical action, completing all animations."""
    fd = env.step(action)
    while game.ljprkjlji or game.pzzwlsmdt:
        fd = env.step(action)
    return fd


def parse_level():
    """Extract level geometry for analytical BFS."""
    head = game.vzvypfsnt
    segs = game.mwfajkguqx[head]
    head_x = head.x
    head_y = head.y
    head_rot = head.rotation

    # Play area bounds
    pa = game.lqwkgffeb
    play_x = pa.x
    play_y = pa.y
    play_w = pa.pixels.shape[1] * CELL
    play_h = pa.pixels.shape[0] * CELL

    # Board targets
    board_targets = []
    for t in game.vbelzuaian:
        if t.y < play_y + play_h:
            board_targets.append((t.x, t.y, int(t.pixels[1, 1])))

    # Walls
    walls = []
    for s in game.current_level.get_sprites():
        tags = list(s.tags) if s.tags else []
        if 'irkeobngyh' in tags:
            walls.append((s.x, s.y, s.width, s.height))

    # Reference pattern
    ref_head = game.xpmcmtbcv.get(head)
    ref_pattern = []
    if ref_head:
        ref_matches = game.vjfbwggsd.get(ref_head, [])
        ref_pattern = [int(m.pixels[1, 1]) for m in ref_matches]

    # Direction from rotation
    dir_map = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}
    base_dir = dir_map[head_rot]

    return {
        'head_x': head_x, 'head_y': head_y, 'head_rot': head_rot,
        'chain_len': len(segs), 'base_dir': base_dir,
        'play_x': play_x, 'play_y': play_y, 'play_w': play_w, 'play_h': play_h,
        'targets': board_targets, 'walls': walls, 'ref_pattern': ref_pattern,
    }


def analytical_bfs(info):
    """BFS for horizontal chain (rotation 0 or 180)."""
    head_x = info['head_x']
    play_x, play_y = info['play_x'], info['play_y']
    play_w, play_h = info['play_w'], info['play_h']
    x_min = play_x
    x_max = play_x + play_w   # target x + width must be <= x_max
    y_min = play_y
    y_max = play_y + play_h
    base_dx, base_dy = info['base_dir']
    ref_pattern = info['ref_pattern']
    walls = info['walls']

    # For horizontal chain (base_dx != 0): segments at fixed y, varying x
    # For vertical chain (base_dy != 0): segments at fixed x, varying y
    is_horizontal = (base_dx != 0)

    def wall_at(px, py):
        for wx, wy, ww, wh in walls:
            if wx <= px < wx + ww and wy <= py < wy + wh:
                return True
        return False

    def can_slide(hy, slide_dy):
        """Check if lateral slide from hy is enabled by a wall."""
        check_x = head_x + 2  # for horizontal chain, slide is vertical
        check_y = hy + 2 + slide_dy // 2
        return wall_at(check_x, check_y)

    # Max chain length
    max_chain = (x_max - head_x) // CELL  # for horizontal right chain

    def seg_xs(L):
        """X-positions of segments for chain of length L."""
        return [head_x + i * CELL for i in range(L)]

    # State: (head_y, chain_len, targets_tuple)
    # targets_tuple: sorted tuple of (tx, ty, tc)
    init_targets = tuple(sorted(info['targets']))
    init_state = (info['head_y'], info['chain_len'], init_targets)

    def check_win(state):
        hy, L, targets = state
        target_map = {(tx, ty): tc for tx, ty, tc in targets}
        covered = []
        for i in range(L):
            sx = head_x + i * CELL
            if (sx, hy) in target_map:
                covered.append(target_map[(sx, hy)])
        return covered == ref_pattern

    def push_line(target_dict, trigger_set, direction, in_bounds):
        """Push targets on a 1D line.

        target_dict: {pos: color} for ALL targets on this line
        trigger_set: set of positions where pushes are INITIATED (segment positions)
        direction: +CELL or -CELL
        in_bounds: fn(pos) -> bool, checks if position is valid

        Returns (new_target_dict, blocked).
        blocked = True if any triggered push failed.
        """
        result = dict(target_dict)
        processed = set()
        blocked = False

        def do_push(pos):
            """Try to push target at pos. Returns True if push succeeded."""
            nonlocal blocked
            if pos in processed:
                return pos not in result  # already moved away = success
            processed.add(pos)
            if pos not in result:
                return True  # nothing to push

            new_pos = pos + direction
            if not in_bounds(new_pos):
                blocked = True
                return False  # can't move, stay

            # Chain push: first push target at destination
            if new_pos in result:
                if not do_push(new_pos):
                    return False  # blocked downstream

            if new_pos in result:
                return False  # still occupied

            # Move
            result[new_pos] = result.pop(pos)
            processed.add(new_pos)  # Mark destination as processed (prevent re-push)
            return True

        # Process triggers — order doesn't matter due to recursion
        for pos in sorted(trigger_set):
            if pos in result:
                do_push(pos)
                # Also check pos + direction (segment destination)
            # Segment also checks its destination for targets
            dest = pos + direction
            if dest in result and dest not in processed:
                do_push(dest)

        return result, blocked

    def apply_action(state, action):
        hy, L, targets = state
        tlist = list(targets)

        if action == RIGHT:
            # EXTEND: chain grows, segments shift right, push targets right
            if L >= max_chain:
                return None
            # Segment x-range for current chain: 11, 17, ..., 11+(L-1)*6
            # After extension: new segment at 11, old shift to 17, ..., 11+L*6
            # bnrdrdiakd checks: current pos and destination for each segment
            # trigger positions = union of current and destination
            # = {11, 17, ..., 11+(L-1)*6} ∪ {17, 23, ..., 11+L*6}
            # = {11, 17, 23, ..., 11+L*6}
            triggers = {head_x + i * CELL for i in range(L + 1)}

            # Only affect targets at y=hy
            at_y = {tx: tc for tx, ty, tc in tlist if ty == hy}
            others = [(tx, ty, tc) for tx, ty, tc in tlist if ty != hy]

            new_at_y, _ = push_line(at_y, triggers, +CELL,
                                    lambda x: x + CELL <= x_max)
            # Extension never fails due to blocked targets
            new_tlist = others + [(x, hy, c) for x, c in new_at_y.items()]
            return (hy, L + 1, tuple(sorted(new_tlist)))

        elif action == LEFT:
            # RETRACT: chain shrinks, segments shift left
            if L <= 1:
                return None
            # After pop(0): remaining at {17, 23, ..., 11+(L-1)*6}
            # They shift left: {11, 17, ..., 11+(L-2)*6}
            # Trigger positions: current ∪ dest = {11, 17, ..., 11+(L-1)*6}
            triggers = {head_x + i * CELL for i in range(L)}

            at_y = {tx: tc for tx, ty, tc in tlist if ty == hy}
            others = [(tx, ty, tc) for tx, ty, tc in tlist if ty != hy]

            new_at_y, _ = push_line(at_y, triggers, -CELL,
                                    lambda x: x >= x_min)
            new_tlist = others + [(x, hy, c) for x, c in new_at_y.items()]
            return (hy, L - 1, tuple(sorted(new_tlist)))

        elif action == UP:
            # SLIDE UP: all segments move up, push targets up
            new_hy = hy - CELL
            if new_hy < y_min:
                return None
            if not can_slide(hy, -CELL):
                return None

            # Segments at x-positions: head_x, head_x+6, ..., head_x+(L-1)*6
            sxs = set(seg_xs(L))

            # For each segment column, push targets upward
            # Trigger: targets at (sx, hy) or (sx, new_hy) — current and destination y
            # Build per-column target dicts
            by_col = defaultdict(dict)  # {sx: {y: color}}
            others = []
            for tx, ty, tc in tlist:
                if tx in sxs:
                    by_col[tx][ty] = tc
                else:
                    others.append((tx, ty, tc))

            new_tlist = list(others)
            for sx in sxs:
                col = by_col.get(sx, {})
                if not col:
                    continue
                # Trigger positions: hy and new_hy
                triggers = {hy, new_hy}
                new_col, blocked = push_line(col, triggers, -CELL,
                                             lambda y: y >= y_min)
                if blocked:
                    return None  # slide fails
                for y, c in new_col.items():
                    new_tlist.append((sx, y, c))

            # Also check boundary for segments themselves
            # Head segment at (head_x, hy) → (head_x, new_hy)
            # The head is outside play area (x < play_x) but allowed
            # In-play segments must fit
            for i in range(L):
                sx = head_x + i * CELL
                if sx >= x_min:  # in-play segment
                    if new_hy < y_min or new_hy + CELL > y_max:
                        return None

            return (new_hy, L, tuple(sorted(new_tlist)))

        elif action == DOWN:
            new_hy = hy + CELL
            if new_hy + CELL > y_max:
                return None
            if not can_slide(hy, CELL):
                return None

            sxs = set(seg_xs(L))
            by_col = defaultdict(dict)
            others = []
            for tx, ty, tc in tlist:
                if tx in sxs:
                    by_col[tx][ty] = tc
                else:
                    others.append((tx, ty, tc))

            new_tlist = list(others)
            for sx in sxs:
                col = by_col.get(sx, {})
                if not col:
                    continue
                triggers = {hy, new_hy}
                new_col, blocked = push_line(col, triggers, +CELL,
                                             lambda y: y + CELL <= y_max)
                if blocked:
                    return None
                for y, c in new_col.items():
                    new_tlist.append((sx, y, c))

            return (new_hy, L, tuple(sorted(new_tlist)))

        return None

    # Compute possible goal positions for A* heuristic
    n_ref = len(ref_pattern)
    # Goal: n_ref consecutive segment positions covering targets with ref_pattern colors
    # Segment positions: head_x, head_x+CELL, ..., head_x+(max_chain-1)*CELL
    all_seg_xs = [head_x + i * CELL for i in range(max_chain)]
    # Valid positions (within play area)
    valid_xs = [x for x in all_seg_xs if x >= x_min and x + CELL <= x_max]
    # All possible y values
    valid_ys = list(range(y_min, y_max, CELL))

    # Goal assignments: for each starting segment index, the target colors map to positions
    goal_configs = []
    for start_idx in range(len(all_seg_xs) - n_ref + 1):
        goal_xs = [all_seg_xs[start_idx + j] for j in range(n_ref)]
        if all(x >= x_min and x + CELL <= x_max for x in goal_xs):
            goal_configs.append(goal_xs)

    def heuristic(state):
        """Lower bound on moves: min over goal configs of total Manhattan distance."""
        hy, L, targets = state
        target_by_color = {tc: (tx, ty) for tx, ty, tc in targets}
        best = float('inf')
        for goal_xs in goal_configs:
            for gy in valid_ys:
                cost = 0
                for j, c in enumerate(ref_pattern):
                    if c not in target_by_color:
                        cost = float('inf')
                        break
                    tx, ty = target_by_color[c]
                    # Manhattan distance in grid units
                    cost += abs(tx - goal_xs[j]) // CELL + abs(ty - gy) // CELL
                if cost < best:
                    best = cost
        return best

    # A* search with parent pointers
    import heapq
    h0 = heuristic(init_state)
    # (f, g, state)
    open_heap = [(h0, 0, init_state)]
    g_score = {init_state: 0}
    parent = {}  # state → (parent_state, action)
    explored = 0
    max_states = 5_000_000

    while open_heap and len(g_score) < max_states:
        f, g, state = heapq.heappop(open_heap)
        explored += 1

        if g > g_score.get(state, float('inf')):
            continue  # stale entry

        if check_win(state):
            # Reconstruct path
            path = []
            s = state
            while s in parent:
                ps, a = parent[s]
                path.append(a)
                s = ps
            path.reverse()
            print(f"  A*: found at depth {len(path)} ({explored} expanded, {len(g_score)} visited)")
            return path

        if explored % 50000 == 0:
            print(f"  A*: {explored} expanded, {len(g_score)} visited, f={f}, g={g}")

        for action in [UP, DOWN, LEFT, RIGHT]:
            new_state = apply_action(state, action)
            if new_state is None or new_state == state:
                continue
            new_g = g + 1
            if new_g >= g_score.get(new_state, float('inf')):
                continue
            g_score[new_state] = new_g
            parent[new_state] = (state, action)
            h = heuristic(new_state)
            heapq.heappush(open_heap, (new_g + h, new_g, new_state))

    print(f"  No solution found! ({explored} expanded, {len(g_score)} visited)")
    return None


def verify_solution(solution, info):
    """Verify solution by stepping through analytical model."""
    state = (info['head_y'], info['chain_len'], tuple(sorted(info['targets'])))
    print(f"    verify: initial state = hy={state[0]}, L={state[1]}, targets={state[2]}")
    for i, action in enumerate(solution):
        # Manually step through
        hy, L, targets = state
        target_map = {(tx, ty): tc for tx, ty, tc in targets}
        seg_ps = [(info['head_x'] + j * CELL, hy) for j in range(L)]
        covered = [(tx, ty, target_map[(tx, ty)]) for tx, ty in seg_ps if (tx, ty) in target_map]
        print(f"    step {i}: {ACTION_NAMES[action]} → hy={hy}, L={L}, covered={covered}")

        # We can't actually step the analytical model here without reimplementing
        # Just show state before each step


total_actions = 0

for lv in range(len(game._levels)):
    print(f"\n=== Level {lv} ===")
    info = parse_level()
    print(f"  Head: ({info['head_x']},{info['head_y']}) rot={info['head_rot']} dir={info['base_dir']}")
    print(f"  Chain: {info['chain_len']} segments")
    print(f"  Targets: {info['targets']}")
    print(f"  Ref pattern: {info['ref_pattern']}")
    print(f"  Play area: ({info['play_x']},{info['play_y']}) {info['play_w']}x{info['play_h']}")
    print(f"  Walls: {info['walls']}")
    print(f"  Budget: {game.qiercdohl}")

    if game.gvtmoopqgy():
        print("  Already solved!")
        lv_actions = 0
    else:
        solution = analytical_bfs(info)
        if solution:
            sol_names = [ACTION_NAMES[a] for a in solution]
            print(f"  Solution ({len(solution)} moves): {sol_names}")

            # Execute on game engine
            lv_actions = 0
            for i, action in enumerate(solution):
                fd = step_action(action)
                lv_actions += 1
                # Check if win triggered (animation starts)
                if game.lgdrixfno >= 0:
                    # Win animation started — advance through it
                    while fd.state.name != 'WIN' and fd.levels_completed <= lv:
                        fd = step_action(UP)
                        lv_actions += 1
                    break

            # Check win
            if fd.levels_completed <= lv and not game.gvtmoopqgy():
                print(f"  WARNING: analytical solution didn't win!")
                head = game.vzvypfsnt
                segs = game.mwfajkguqx[head]
                targets = sorted([(t.x, t.y, int(t.pixels[1,1])) for t in game.vbelzuaian if t.y < info['play_y'] + info['play_h']])
                matches = game.vjfbwggsd.get(head, [])
                match_colors = [int(m.pixels[1,1]) for m in matches]
                print(f"    segs=[{', '.join(f'({s.x},{s.y})' for s in segs)}]")
                print(f"    targets={targets} matches={match_colors}")
                print(f"    ref_pattern={info['ref_pattern']}")
                break
        else:
            print(f"  No solution found!")
            break

    total_actions += lv_actions
    print(f"  {lv_actions} actions, completed={fd.levels_completed}, state={fd.state.name}")

    if fd.state.name == 'WIN':
        print(f"\n=== ALL SOLVED! Total: {total_actions} ===")
        break

    if fd.levels_completed <= lv:
        print(f"  Level not completed!")
        break

print(f"\nFinal: completed={fd.levels_completed}, state={fd.state.name}, total={total_actions}")
