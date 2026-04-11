#!/usr/bin/env python3
"""Solve r11l: click-based creature movement puzzle.

Click leg to select, click empty space to move it. Body = centroid of legs.
Win when all bodies overlap targets. Avoid obstacles (5 collisions = lose).
Yukft bodies must absorb paint sprites to color-match targets.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
from collections import deque
from itertools import permutations

arcade = Arcade()

def leg_center(leg):
    return leg.x + leg.width // 2, leg.y + leg.height // 2

def sprite_center(s):
    return s.x + s.width // 2, s.y + s.height // 2

def compute_centroid(legs):
    tx = sum(l.x + l.width // 2 for l in legs)
    ty = sum(l.y + l.height // 2 for l in legs)
    return tx // len(legs), ty // len(legs)

def get_body_colors(body):
    return {int(c) for c in np.unique(body.pixels) if c > 0}

def get_target_colors(target):
    return {int(c) for c in np.unique(target.pixels) if c > 0}

def check_body_obstacle_collision(body, cx, cy, obstacles):
    """Check if body at centroid (cx,cy) would collide with any obstacle."""
    bx = cx - body.width // 2
    by = cy - body.height // 2
    orig_x, orig_y = body.x, body.y
    body.set_position(bx, by)
    collides = False
    for obs in obstacles:
        if body.collides_with(obs):
            collides = True
            break
    body.set_position(orig_x, orig_y)
    return collides

def build_centroid_safe_map(body, obstacles, lo=0, hi=63):
    """Return set of centroid positions where body doesn't collide with obstacles."""
    safe = set()
    for cy in range(lo, hi + 1):
        for cx in range(lo, hi + 1):
            if not check_body_obstacle_collision(body, cx, cy, obstacles):
                safe.add((cx, cy))
    return safe

def centroid_bfs(safe_map, start, end):
    """BFS from start to end through safe centroid positions."""
    if start == end:
        return [start]
    if start not in safe_map or end not in safe_map:
        return None
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        pos, path = queue.popleft()
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = pos[0]+dx, pos[1]+dy
            if (nx, ny) in safe_map and (nx, ny) not in visited:
                new_path = path + [(nx, ny)]
                if (nx, ny) == end:
                    return new_path
                visited.add((nx, ny))
                queue.append(((nx, ny), new_path))
    return None

def check_leg_terrain(leg, cx, cy, terrain):
    """Check if leg at center (cx,cy) collides with terrain."""
    orig = (leg.x, leg.y)
    leg.set_position(cx - leg.width // 2, cy - leg.height // 2)
    col = any(leg.collides_with(t) for t in terrain)
    leg.set_position(*orig)
    return col

def compute_target_legs(legs, target_cx, target_cy, terrain=None):
    """Compute target center positions for each leg to achieve target centroid.
    Handles boundary clamping AND terrain collision by redistributing."""
    n = len(legs)
    cur_cx, cur_cy = compute_centroid(legs)
    dx = target_cx - cur_cx
    dy = target_cy - cur_cy

    # Start with uniform delta, apply clamping
    targets = []
    for leg in legs:
        lc = leg_center(leg)
        targets.append([max(2, min(61, lc[0] + dx)), max(2, min(61, lc[1] + dy))])

    # Check terrain collision and fix
    if terrain:
        for i, leg in enumerate(legs):
            if check_leg_terrain(leg, targets[i][0], targets[i][1], terrain):
                # Find nearest terrain-safe position (prefer y adjustment)
                found = False
                for adj in range(1, 40):
                    for new_y in [targets[i][1] - adj, targets[i][1] + adj]:
                        new_y = max(2, min(61, new_y))
                        if not check_leg_terrain(leg, targets[i][0], new_y, terrain):
                            targets[i][1] = new_y
                            found = True
                            break
                    if found:
                        break
                if not found:
                    # Try x adjustment too
                    for adj in range(1, 40):
                        for new_x in [targets[i][0] - adj, targets[i][0] + adj]:
                            new_x = max(2, min(61, new_x))
                            if not check_leg_terrain(leg, new_x, targets[i][1], terrain):
                                targets[i][0] = new_x
                                found = True
                                break
                        if found:
                            break

    # Adjust centroid to match target (redistribute after clamping/terrain fixes)
    for axis in [0, 1]:
        target_val = target_cx if axis == 0 else target_cy
        needed_sum = target_val * n
        current_sum = sum(t[axis] for t in targets)
        diff = needed_sum - current_sum

        for i in range(n):
            if diff == 0:
                break
            lo, hi = 2, 61
            if targets[i][axis] > lo and targets[i][axis] < hi:
                adj = max(-abs(diff), min(abs(diff), diff))
                new_val = max(lo, min(hi, targets[i][axis] + adj))
                # Verify terrain safety after adjustment
                if terrain:
                    test = list(targets[i])
                    test[axis] = new_val
                    if check_leg_terrain(legs[i], test[0], test[1], terrain):
                        continue  # Skip this leg, try next
                actual_adj = new_val - targets[i][axis]
                targets[i][axis] = new_val
                diff -= actual_adj

    return [tuple(t) for t in targets]

def find_safe_order(legs, targets, safe_map):
    """Find a leg movement order where all intermediate centroids are safe.
    Returns the order (list of indices) or None."""
    n = len(legs)
    cur_xs = [l.x + l.width // 2 for l in legs]
    cur_ys = [l.y + l.height // 2 for l in legs]

    for order in permutations(range(n)):
        xs = cur_xs[:]
        ys = cur_ys[:]
        safe = True
        for idx in order:
            xs[idx] = targets[idx][0]
            ys[idx] = targets[idx][1]
            cx = sum(xs) // n
            cy = sum(ys) // n
            if (cx, cy) not in safe_map:
                safe = False
                break
        if safe:
            return list(order)
    return None

def move_creature_to(env, game, creature_data, target_cx, target_cy, obstacles, safe_map=None):
    """Move legs to put body at target centroid, avoiding obstacles.
    Returns (actions_count, last_fd)."""
    body = creature_data['body']
    legs = creature_data['legs']
    n = len(legs)
    actions = 0
    fd = None

    cur_cx, cur_cy = compute_centroid(legs)
    if cur_cx == target_cx and cur_cy == target_cy:
        return 0, None

    dx = target_cx - cur_cx
    dy = target_cy - cur_cy

    terrain = game.yhfnp

    if not obstacles:
        # No obstacles — compute terrain-aware targets
        targets = compute_target_legs(legs, target_cx, target_cy, terrain)
        for i, leg in enumerate(legs):
            lc = leg_center(leg)
            fd = env.step(GameAction.ACTION6, data={'x': lc[0], 'y': lc[1]})
            actions += 1
            fd = env.step(GameAction.ACTION6, data={'x': targets[i][0], 'y': targets[i][1]})
            actions += 1
        return actions, fd

    # With obstacles
    if safe_map is None:
        safe_map = build_centroid_safe_map(body, obstacles)

    # Compute target leg positions (terrain-aware)
    targets = compute_target_legs(legs, target_cx, target_cy, terrain)

    # Try to find a safe ordering for direct move
    order = find_safe_order(legs, targets, safe_map)
    if order:
        for idx in order:
            lc = leg_center(legs[idx])
            fd = env.step(GameAction.ACTION6, data={'x': lc[0], 'y': lc[1]})
            actions += 1
            fd = env.step(GameAction.ACTION6, data={'x': targets[idx][0], 'y': targets[idx][1]})
            actions += 1
        # Verify success
        actual = compute_centroid(legs)
        if actual == (target_cx, target_cy):
            return actions, fd
        # Direct move didn't fully work — fall through to waypoint strategy

    # Use waypoint strategy
    path = centroid_bfs(safe_map, compute_centroid(legs), (target_cx, target_cy))
    if not path:
        print(f"    ERROR: no centroid path to ({target_cx},{target_cy})!")
        return actions, fd

    print(f"    centroid BFS: {len(path)} steps, using waypoints")

    # Move through waypoints iteratively
    max_rounds = 50
    for round_i in range(max_rounds):
        cur = compute_centroid(legs)
        if cur == (target_cx, target_cy):
            break

        # Recompute target legs and try direct ordering
        tgts = compute_target_legs(legs, target_cx, target_cy, terrain)
        order = find_safe_order(legs, tgts, safe_map)
        if order:
            for idx in order:
                lc = leg_center(legs[idx])
                fd = env.step(GameAction.ACTION6, data={'x': lc[0], 'y': lc[1]})
                actions += 1
                fd = env.step(GameAction.ACTION6, data={'x': tgts[idx][0], 'y': tgts[idx][1]})
                actions += 1
            break

        # BFS from current to target
        repath = centroid_bfs(safe_map, cur, (target_cx, target_cy))
        if not repath or len(repath) < 2:
            print(f"    ERROR: no path from {cur}")
            break

        # Try jumping to each waypoint on the path (farthest first)
        jumped = False
        for j in range(len(repath) - 1, 0, -1):
            wp = repath[j]
            wp_tgts = compute_target_legs(legs, wp[0], wp[1], terrain)
            wp_order = find_safe_order(legs, wp_tgts, safe_map)
            if wp_order:
                for idx in wp_order:
                    lc = leg_center(legs[idx])
                    fd = env.step(GameAction.ACTION6, data={'x': lc[0], 'y': lc[1]})
                    actions += 1
                    fd = env.step(GameAction.ACTION6, data={'x': wp_tgts[idx][0], 'y': wp_tgts[idx][1]})
                    actions += 1
                jumped = True
                break

        if not jumped:
            print(f"    WARNING: stuck at {cur}, no safe jump found")
            break

    # Final correction
    final = compute_centroid(legs)
    if final != (target_cx, target_cy):
        others_x = sum(l.x + l.width // 2 for l in legs[:-1])
        others_y = sum(l.y + l.height // 2 for l in legs[:-1])
        fix_x = target_cx * n - others_x
        fix_y = target_cy * n - others_y
        fix_x = max(2, min(61, fix_x))
        fix_y = max(2, min(61, fix_y))

        test_cx = (others_x + fix_x) // n
        test_cy = (others_y + fix_y) // n
        terrain_ok = not check_leg_terrain(legs[-1], fix_x, fix_y, terrain)
        centroid_ok = not obstacles or (test_cx, test_cy) in safe_map
        if terrain_ok and centroid_ok:
            cur_lc = leg_center(legs[-1])
            fd = env.step(GameAction.ACTION6, data={'x': cur_lc[0], 'y': cur_lc[1]})
            actions += 1
            fd = env.step(GameAction.ACTION6, data={'x': fix_x, 'y': fix_y})
            actions += 1

    return actions, fd


# === SOLVE ===
env = arcade.make('r11l-aa269680')
fd = env.reset()
game = env._game

total_actions = 0

for lv in range(6):
    level = game.current_level
    print(f"\n=== Level {lv} ===")

    creatures = {}
    for color, data in game.brdck.items():
        body = data.get('kignw')
        legs = data.get('mdpcc', [])
        target = data.get('xwdrv')
        is_yukft = body and body.name.startswith('bdkaz-yukft') if body else False
        if body or legs or target:
            creatures[color] = {'body': body, 'legs': legs, 'target': target, 'is_yukft': is_yukft}

    obstacles = [s for s in level.get_sprites() if s.name.startswith("qtwnv")]
    if obstacles:
        print(f"  Obstacles: {len(obstacles)}")
    if not creatures:
        print(f"  No creatures found! brdck keys: {list(game.brdck.keys())}")
        for k, v in game.brdck.items():
            print(f"    {k}: body={v.get('kignw') is not None} legs={len(v.get('mdpcc',[]))} target={v.get('xwdrv') is not None}")

    lv_actions = 0

    # Separate normal creatures, yukft creatures, and orphaned targets
    normal_creatures = {}
    yukft_creatures = {}
    orphaned_targets = {}  # color -> target sprite (has target but no body)

    for color, cdata in creatures.items():
        if 'anqcf' in color:
            continue
        if cdata['is_yukft']:
            yukft_creatures[color] = cdata
        elif cdata['body'] and cdata['target']:
            normal_creatures[color] = cdata
        elif cdata['target'] and not cdata['body']:
            orphaned_targets[color] = cdata['target']

    print(f"  normal: {list(normal_creatures.keys())}")
    print(f"  yukft: {list(yukft_creatures.keys())}")
    print(f"  orphaned targets: {list(orphaned_targets.keys())}")

    # Handle normal creatures first
    for color, cdata in sorted(normal_creatures.items()):
        body = cdata['body']
        legs = cdata['legs']
        target = cdata['target']
        tc = sprite_center(target)
        bc = sprite_center(body)
        print(f"  {color}: body={bc} target={tc} legs={len(legs)}")

        creature_safe_map = None
        if obstacles:
            creature_safe_map = build_centroid_safe_map(body, obstacles)

        acts, fd2 = move_creature_to(env, game, cdata, tc[0], tc[1], obstacles, creature_safe_map)
        lv_actions += acts
        if fd2: fd = fd2
        bc = sprite_center(body)
        print(f"    result: body={bc} collides={body.collides_with(target)}")

    # Handle yukft creatures + orphaned targets
    if yukft_creatures and orphaned_targets:
        yukft_list = list(yukft_creatures.items())
        target_list = list(orphaned_targets.items())

        # Inspect colors
        for yk_color, yk_data in yukft_list:
            print(f"  yukft {yk_color}: body_colors={get_body_colors(yk_data['body'])}")
        for tgt_color, tgt_sprite in target_list:
            print(f"  target {tgt_color}: colors={get_target_colors(tgt_sprite)} pos={sprite_center(tgt_sprite)}")

        # Show available paints
        paints = game.nxahg[:]
        for p in paints:
            pcols = {int(c) for c in np.unique(p.pixels) if c > 0}
            print(f"  paint {p.name}: colors={pcols} pos={sprite_center(p)}")

        # Assign yukft bodies to targets greedily (minimize total distance)
        # Try all permutations if small enough
        from itertools import permutations as perms
        best_assignment = None
        best_cost = float('inf')
        n_assign = min(len(yukft_list), len(target_list))

        for yk_perm in perms(range(len(yukft_list)), n_assign):
            for tgt_perm in perms(range(len(target_list)), n_assign):
                cost = 0
                for i in range(n_assign):
                    yk_body = yukft_list[yk_perm[i]][1]['body']
                    tgt_spr = target_list[tgt_perm[i]][1]
                    bc = sprite_center(yk_body)
                    tc = sprite_center(tgt_spr)
                    cost += abs(bc[0]-tc[0]) + abs(bc[1]-tc[1])
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = list(zip(yk_perm, tgt_perm))

        print(f"  assignment: {[(yukft_list[a[0]][0], target_list[a[1]][0]) for a in best_assignment]}")

        for yk_idx, tgt_idx in best_assignment:
            yk_color, yk_data = yukft_list[yk_idx]
            tgt_color, tgt_sprite = target_list[tgt_idx]
            body = yk_data['body']
            legs = yk_data['legs']

            target_cols = get_target_colors(tgt_sprite)
            body_cols = get_body_colors(body)
            needed = target_cols - body_cols
            print(f"  {yk_color} -> {tgt_color}: need colors {needed} (body has {body_cols}, target needs {target_cols})")

            creature_safe_map = None
            if obstacles:
                creature_safe_map = build_centroid_safe_map(body, obstacles)

            # Plan paint absorption: find paints whose colors are subsets of target colors
            paint_plan = []
            remaining = set(needed)
            # First pass: only paints that contribute needed colors without harmful ones
            for paint in paints:
                if paint not in game.nxahg:
                    continue  # Already absorbed
                pcols = {int(c) for c in np.unique(paint.pixels) if c > 0}
                useful = pcols & remaining
                harmful = pcols - target_cols
                if useful and not harmful:
                    paint_plan.append(paint)
                    remaining -= pcols
                    if not remaining:
                        break

            # Second pass: if still missing, accept any paint with needed colors
            if remaining:
                for paint in paints:
                    if paint not in game.nxahg or paint in paint_plan:
                        continue
                    pcols = {int(c) for c in np.unique(paint.pixels) if c > 0}
                    if pcols & remaining:
                        paint_plan.append(paint)
                        remaining -= pcols
                        if not remaining:
                            break

            print(f"    paint_plan: {[(p.name, sprite_center(p)) for p in paint_plan]}")

            # Route through paints then to target
            for paint in paint_plan:
                pc = sprite_center(paint)
                print(f"    moving to paint {paint.name} at {pc}")
                acts, fd2 = move_creature_to(env, game, yk_data, pc[0], pc[1], obstacles, creature_safe_map)
                lv_actions += acts
                if fd2: fd = fd2
                print(f"    after paint: body_colors={get_body_colors(body)}")

            # Move to target
            tc = sprite_center(tgt_sprite)
            print(f"    moving to target {tgt_color} at {tc}")
            acts, fd2 = move_creature_to(env, game, yk_data, tc[0], tc[1], obstacles, creature_safe_map)
            lv_actions += acts
            if fd2: fd = fd2

            bc = sprite_center(body)
            final_cols = get_body_colors(body)
            print(f"    result: body={bc} colors={final_cols} target_colors={target_cols} match={final_cols==target_cols}")
            print(f"    collides={body.collides_with(tgt_sprite)}")

    total_actions += lv_actions
    print(f"  {lv_actions} actions, completed={fd.levels_completed}, state={fd.state.name}")

    if fd.state.name == 'WIN':
        print(f"\n=== ALL SOLVED! Total: {total_actions} ===")
        break

    if fd.levels_completed <= lv:
        for color, cdata in creatures.items():
            body, target = cdata['body'], cdata['target']
            if body and target and 'anqcf' not in color:
                print(f"  DBG {color}: body={sprite_center(body)} target={sprite_center(target)} collides={body.collides_with(target)}")
                if cdata['is_yukft']:
                    print(f"    colors: body={get_body_colors(body)} target={get_target_colors(target)}")
        break

print(f"\nFinal: completed={fd.levels_completed}, state={fd.state.name}, total={total_actions}")
