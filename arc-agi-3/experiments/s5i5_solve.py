#!/usr/bin/env python3
"""Solve s5i5: arrow resize/rotate puzzle.

Full analytical model with rotation support. State = (track_indices, arrow_orientations).
BFS explores grow/shrink/rotate actions. Collision via pixel occupancy sets.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
from collections import deque

CELL = 3
TIP_COLOR = 3

arcade = Arcade()
env = arcade.make('s5i5-a48e4b1d')
fd = env.reset()
game = env._game


def grid_to_display(gx, gy):
    cam_w = game.camera.width
    cam_h = game.camera.height
    scale = 64 // cam_w
    x_off = (64 - cam_w * scale) // 2
    y_off = (64 - cam_h * scale) // 2
    return gx * scale + x_off, gy * scale + y_off


def get_orientation(pixels):
    h, w = pixels.shape
    if h >= 3 and pixels[-1, 1] == TIP_COLOR: return 0
    elif w >= 3 and pixels[1, 0] == TIP_COLOR: return 90
    elif h >= 3 and pixels[0, 1] == TIP_COLOR: return 180
    elif w >= 3 and pixels[1, -1] == TIP_COLOR: return 270
    elif h > w: return 0
    else: return 270


def get_index(pixels):
    return max(pixels.shape) // CELL


def parse_level(level):
    tracks = level.get_sprites_by_tag("gdgcpukdrl")
    arrows_sprites = level.get_sprites_by_tag("agujdcrunq")
    targets = level.get_sprites_by_tag("zylvdxoiuq")
    goals = level.get_sprites_by_tag("cpdhnkdobh")
    rotate_btns = level.get_sprites_by_tag("myzmclysbl")

    controlled_colors = set()
    for t in tracks:
        for a in game.dfyrdkjdcj[t]:
            controlled_colors.add(int(a.pixels[1, 1]))

    arrow_list = []
    arrow_by_sprite = {}
    for i, a in enumerate(arrows_sprites):
        color = int(a.pixels[1, 1])
        h, w = a.pixels.shape
        non_trans = int(np.sum(a.pixels >= 0))
        has_parent = any(a in game.enplxxgoja.get(p, set())
                        for p in arrows_sprites if p is not a)
        if color in controlled_colors or has_parent:
            is_obstacle = False
        else:
            is_obstacle = (non_trans < h * w)

        info = {
            'id': i, 'sprite': a,
            'x0': a.x, 'y0': a.y,
            'index0': get_index(a.pixels),
            'orient0': get_orientation(a.pixels),
            'color': color,
            'children': [],
            'is_obstacle': is_obstacle,
            'track_idx': -1,
            'parent_idx': -1,
        }
        arrow_list.append(info)
        arrow_by_sprite[id(a)] = i

    # Initial obstacle pixel positions (will be extended after parent/track setup)
    obstacle_pixels = set()
    for i, ainfo in enumerate(arrow_list):
        if ainfo['is_obstacle']:
            a = ainfo['sprite']
            h, w = a.pixels.shape
            for r in range(h):
                for c in range(w):
                    if a.pixels[r, c] >= 0:
                        obstacle_pixels.add((a.x + c, a.y + r))

    target_list = []
    target_by_sprite = {}
    for i, t in enumerate(targets):
        target_list.append({'id': i, 'sprite': t, 'x0': t.x, 'y0': t.y})
        target_by_sprite[id(t)] = i

    goal_positions = set((g.x, g.y) for g in goals)

    for parent_sprite, children_set in game.enplxxgoja.items():
        if id(parent_sprite) not in arrow_by_sprite:
            continue
        pi = arrow_by_sprite[id(parent_sprite)]
        for child_sprite in children_set:
            if id(child_sprite) in arrow_by_sprite:
                ci = arrow_by_sprite[id(child_sprite)]
                arrow_list[pi]['children'].append(('arrow', ci))
                arrow_list[ci]['parent_idx'] = pi
            elif id(child_sprite) in target_by_sprite:
                arrow_list[pi]['children'].append(('target', target_by_sprite[id(child_sprite)]))

    track_info = []
    for t in tracks:
        is_h = t.width > t.height
        arrow_ids = []
        for a in game.dfyrdkjdcj[t]:
            if id(a) in arrow_by_sprite:
                ai = arrow_by_sprite[id(a)]
                arrow_ids.append(ai)
                arrow_list[ai]['track_idx'] = len(track_info)
        init_idx = arrow_list[arrow_ids[0]]['index0'] if arrow_ids else 1
        track_info.append({'sprite': t, 'is_h': is_h, 'arrow_ids': arrow_ids, 'index0': init_idx})

    # Detect truly static arrows (after parent/track setup):
    # no track, no parent, not controlled by any button
    btn_colors = set(int(b.pixels[b.height // 2, b.height // 2]) for b in rotate_btns)
    for i, ainfo in enumerate(arrow_list):
        if not ainfo['is_obstacle'] and ainfo['track_idx'] < 0 and ainfo['parent_idx'] < 0:
            if ainfo['color'] not in btn_colors:
                ainfo['is_obstacle'] = True
                a = ainfo['sprite']
                h, w = a.pixels.shape
                for r in range(h):
                    for c in range(w):
                        if a.pixels[r, c] >= 0:
                            obstacle_pixels.add((a.x + c, a.y + r))

    btn_info = []
    for b in rotate_btns:
        color = int(b.pixels[b.height // 2, b.height // 2])
        # Find which arrows this button controls
        matching_arrows = [i for i, a in enumerate(arrow_list)
                          if a['color'] == color and not a['is_obstacle']]
        btn_info.append({'sprite': b, 'color': color, 'arrow_ids': matching_arrows})

    return arrow_list, target_list, goal_positions, track_info, btn_info, obstacle_pixels


# ============ State representation ============
# State = (track_indices_tuple, arrow_orient_tuple)
# track_indices: one per track
# arrow_orient: one per non-obstacle arrow (0, 90, 180, 270)

def make_initial_state(track_info, arrow_list):
    track_idx = tuple(t['index0'] for t in track_info)
    arrow_orient = tuple(a['orient0'] for a in arrow_list if not a['is_obstacle'])
    return (track_idx, arrow_orient)


def arrow_dims(orient, idx):
    if orient in (0, 180):
        return CELL, idx * CELL  # w, h
    else:
        return idx * CELL, CELL


def arrow_self_disp(orient, delta):
    if orient == 0: return (0, -delta * CELL)
    elif orient == 270: return (-delta * CELL, 0)
    return (0, 0)


def growth_disp(orient, delta):
    if orient == 0: return (0, -delta * CELL)
    elif orient == 90: return (delta * CELL, 0)
    elif orient == 180: return (0, delta * CELL)
    elif orient == 270: return (-delta * CELL, 0)
    return (0, 0)


ORIENT_NEXT = {0: 270, 90: 0, 180: 90, 270: 180}  # CCW pixel rotation: tip direction changes


def compute_positions(state, arrow_list, target_list, track_info):
    """Compute all positions from state. Returns (ax, ay, aidx, aorient, tx, ty)."""
    track_idx_tuple, orient_tuple = state
    n_arrows = len(arrow_list)

    ax = [a['x0'] for a in arrow_list]
    ay = [a['y0'] for a in arrow_list]
    aidx = [a['index0'] for a in arrow_list]
    aorient = list(orient_tuple) if len(orient_tuple) == n_arrows else \
              [a['orient0'] for a in arrow_list]
    tx = [t['x0'] for t in target_list]
    ty = [t['y0'] for t in target_list]

    # Unpack orient for non-obstacle arrows
    oi = 0
    aorient_full = []
    for a in arrow_list:
        if a['is_obstacle']:
            aorient_full.append(a['orient0'])
        else:
            aorient_full.append(orient_tuple[oi])
            oi += 1
    aorient = aorient_full

    # Apply track index changes
    for ti, tinfo in enumerate(track_info):
        new_idx = track_idx_tuple[ti]
        for ai in tinfo['arrow_ids']:
            delta = new_idx - arrow_list[ai]['index0']
            if delta == 0:
                continue
            orient = aorient[ai]
            sdx, sdy = arrow_self_disp(orient, delta)
            ax[ai] += sdx
            ay[ai] += sdy
            aidx[ai] = new_idx
            cdx, cdy = growth_disp(orient, delta)
            _apply_child_disp(ax, ay, tx, ty, arrow_list, ai, cdx, cdy)

    return ax, ay, aidx, aorient, tx, ty


def _apply_child_disp(ax, ay, tx, ty, arrow_list, parent_idx, dx, dy):
    for ctype, cidx in arrow_list[parent_idx]['children']:
        if ctype == 'arrow':
            ax[cidx] += dx
            ay[cidx] += dy
            _apply_child_disp(ax, ay, tx, ty, arrow_list, cidx, dx, dy)
        elif ctype == 'target':
            tx[cidx] += dx
            ty[cidx] += dy


def arrow_pixels(x, y, idx, orient):
    w, h = arrow_dims(orient, idx)
    return set((x + dx, y + dy) for dy in range(h) for dx in range(w))


def check_collision_from_state(state, arrow_list, target_list, track_info, obstacle_pixels):
    ax, ay, aidx, aorient, tx, ty = compute_positions(state, arrow_list, target_list, track_info)
    n = len(ax)
    occ = []
    for i in range(n):
        if arrow_list[i]['is_obstacle']:
            occ.append(None)
        else:
            occ.append(arrow_pixels(ax[i], ay[i], aidx[i], aorient[i]))
    for i in range(n):
        if occ[i] is None: continue
        if occ[i] & obstacle_pixels: return True
        for j in range(i + 1, n):
            if occ[j] is None: continue
            if occ[i] & occ[j]: return True
    return False


def check_win(state, arrow_list, target_list, track_info, goal_positions):
    ax, ay, aidx, aorient, tx, ty = compute_positions(state, arrow_list, target_list, track_info)
    tpos = set((tx[i], ty[i]) for i in range(len(target_list)))
    return goal_positions <= tpos


def apply_grow_shrink(state, track_info, ti, grow):
    """Return new state after grow/shrink track ti, or None if invalid."""
    track_idx, orient = state
    new_idx = track_idx[ti] + (1 if grow else -1)
    if new_idx < 1:
        return None
    new_track = list(track_idx)
    new_track[ti] = new_idx
    return (tuple(new_track), orient)


def apply_rotate(state, arrow_list, target_list, track_info, btn_info, bi):
    """Apply rotation button bi. Returns new state or None.

    Rotation is complex: changes positions AND orientations of arrows and children.
    We need to compute the actual positions, apply rotation transforms, then
    encode back to state.

    But our state only stores track indices + orientations, not positions.
    Positions are derived. Rotation changes the mapping between indices and positions.

    This means rotation fundamentally changes the coordinate system. After rotation,
    the same track index produces different positions than before.

    The solution: track orientations AND base positions in the state.
    """
    # This requires a more complex state model. For now, use simulation.
    return None  # Placeholder


# ============ Analytical rotation BFS ============

def analytical_bfs(track_info, btn_info, arrow_list, target_list, goal_positions,
                   obstacle_pixels, step_limit, max_states=2_000_000):
    """BFS with analytical rotation. State = compact tuple, no pixel arrays.

    State layout: (ax0, ay0, ao0, ai0, ax1, ay1, ao1, ai1, ..., tx0, ty0, tx1, ty1, ...)
    where a=arrow (non-obstacle), t=target.
    """
    # Map non-obstacle arrows
    non_obs = [i for i, a in enumerate(arrow_list) if not a['is_obstacle']]
    non_obs_set = set(non_obs)
    noa = len(non_obs)  # number of non-obstacle arrows
    nt = len(target_list)
    # Index mapping: non_obs[k] = original arrow index
    # Reverse: orig_to_compact[orig_idx] = compact_idx (for non-obstacle only)
    orig_to_compact = {}
    for k, orig in enumerate(non_obs):
        orig_to_compact[orig] = k

    # Build children mapping in compact space
    # children[compact_idx] = list of ('arrow', compact_idx) or ('target', target_idx)
    compact_children = [[] for _ in range(noa)]
    for k, orig in enumerate(non_obs):
        for ctype, cidx in arrow_list[orig]['children']:
            if ctype == 'arrow' and cidx in orig_to_compact:
                compact_children[k].append(('arrow', orig_to_compact[cidx]))
            elif ctype == 'target':
                compact_children[k].append(('target', cidx))

    # Parent mapping in compact space
    compact_parent = [-1] * noa
    for k, orig in enumerate(non_obs):
        pi = arrow_list[orig]['parent_idx']
        if pi >= 0 and pi in orig_to_compact:
            compact_parent[k] = orig_to_compact[pi]

    # Track info in compact space
    compact_track_arrows = []  # list of list of compact indices per track
    for ti, tinfo in enumerate(track_info):
        ca = [orig_to_compact[ai] for ai in tinfo['arrow_ids'] if ai in orig_to_compact]
        compact_track_arrows.append(ca)

    # Button info: which compact arrows match each button's color
    compact_btn_arrows = []
    for bi, binfo in enumerate(btn_info):
        ca = [orig_to_compact[ai] for ai in binfo['arrow_ids'] if ai in orig_to_compact]
        compact_btn_arrows.append(ca)

    # Initial state
    def make_state():
        parts = []
        for k, orig in enumerate(non_obs):
            a = arrow_list[orig]
            parts.extend([a['x0'], a['y0'], a['orient0'], a['index0']])
        for t in target_list:
            parts.extend([t['x0'], t['y0']])
        return tuple(parts)

    def unpack(state):
        """Returns (ax, ay, ao, ai, tx, ty) as lists."""
        ax = [state[k*4] for k in range(noa)]
        ay = [state[k*4+1] for k in range(noa)]
        ao = [state[k*4+2] for k in range(noa)]
        ai = [state[k*4+3] for k in range(noa)]
        base = noa * 4
        tx = [state[base + j*2] for j in range(nt)]
        ty = [state[base + j*2+1] for j in range(nt)]
        return ax, ay, ao, ai, tx, ty

    def pack(ax, ay, ao, ai, tx, ty):
        parts = []
        for k in range(noa):
            parts.extend([ax[k], ay[k], ao[k], ai[k]])
        for j in range(nt):
            parts.extend([tx[j], ty[j]])
        return tuple(parts)

    # Precompute obstacle bounding boxes for fast checking
    obs_rects = []
    if obstacle_pixels:
        # Group obstacle pixels into bounding boxes (or just use the set)
        obs_min_x = min(p[0] for p in obstacle_pixels)
        obs_max_x = max(p[0] for p in obstacle_pixels) + 1
        obs_min_y = min(p[1] for p in obstacle_pixels)
        obs_max_y = max(p[1] for p in obstacle_pixels) + 1
        obs_bbox = (obs_min_x, obs_min_y, obs_max_x, obs_max_y)
    else:
        obs_bbox = None

    def check_collisions(ax, ay, ao, ai):
        """Check collisions using bounding box overlap (arrows are solid rects)."""
        rects = []
        for k in range(noa):
            w, h = arrow_dims(ao[k], ai[k])
            rects.append((ax[k], ay[k], ax[k]+w, ay[k]+h))
        # Arrow-arrow collision (bbox overlap = collision for solid rects)
        for k in range(noa):
            x1, y1, x2, y2 = rects[k]
            for j in range(k+1, noa):
                jx1, jy1, jx2, jy2 = rects[j]
                if x1 < jx2 and x2 > jx1 and y1 < jy2 and y2 > jy1:
                    return True
        # Arrow-obstacle collision
        if obs_bbox:
            ox1, oy1, ox2, oy2 = obs_bbox
            for k in range(noa):
                x1, y1, x2, y2 = rects[k]
                if x1 < ox2 and x2 > ox1 and y1 < oy2 and y2 > oy1:
                    # Bbox overlaps obstacle bbox — do precise pixel check
                    w, h = arrow_dims(ao[k], ai[k])
                    for dy in range(h):
                        for dx in range(w):
                            if (ax[k]+dx, ay[k]+dy) in obstacle_pixels:
                                return True
        return False

    def check_win(tx, ty):
        tpos = set((tx[j], ty[j]) for j in range(nt))
        return goal_positions <= tpos

    def _child_disp(ax, ay, tx, ty, k, dx, dy):
        """Recursively displace children of compact arrow k."""
        for ctype, cidx in compact_children[k]:
            if ctype == 'arrow':
                ax[cidx] += dx; ay[cidx] += dy
                _child_disp(ax, ay, tx, ty, cidx, dx, dy)
            elif ctype == 'target':
                tx[cidx] += dx; ty[cidx] += dy

    def _rotate_child(ax, ay, ao, ai, tx, ty, k, cx, cy):
        """Rotate compact arrow k around center (cx, cy). Recurse to children."""
        dx = cx - ax[k]
        dy = cy - ay[k]
        w, h = arrow_dims(ao[k], ai[k])
        ax[k] = cx - dy
        ay[k] = cy + dx - (w - CELL)  # w is the "width" before rotation
        ao[k] = ORIENT_NEXT[ao[k]]
        # Recurse
        for ctype, cidx in compact_children[k]:
            if ctype == 'arrow':
                _rotate_child(ax, ay, ao, ai, tx, ty, cidx, cx, cy)
            elif ctype == 'target':
                tdx = cx - tx[cidx]
                tdy = cy - ty[cidx]
                tx[cidx] = cx - tdy
                ty[cidx] = cy + tdx  # target width = CELL, so w-CELL = 0

    def _rotate_arrow_around_center(ax, ay, ao, ai, tx, ty, k, cx, cy):
        """Rotate compact arrow k (and children) around (cx, cy).
        Like ayrfrgnhfh but for the arrow being rotated (not the top-level)."""
        dx = cx - ax[k]
        dy = cy - ay[k]
        w, h = arrow_dims(ao[k], ai[k])
        ax[k] = cx - dy
        ay[k] = cy + dx - (w - CELL)
        ao[k] = ORIENT_NEXT[ao[k]]
        for ctype, cidx in compact_children[k]:
            if ctype == 'arrow':
                _rotate_child(ax, ay, ao, ai, tx, ty, cidx, cx, cy)
            elif ctype == 'target':
                tdx = cx - tx[cidx]
                tdy = cy - ty[cidx]
                tx[cidx] = cx - tdy
                ty[cidx] = cy + tdx

    def apply_grow_shrink(state, ti, grow):
        """Apply grow/shrink to track ti. Returns new state or None."""
        ax, ay, ao, ai, tx, ty = unpack(state)
        for k in compact_track_arrows[ti]:
            new_idx = ai[k] + (1 if grow else -1)
            if new_idx < 1:
                return None
            old_idx = ai[k]
            delta = new_idx - old_idx
            orient = ao[k]
            # Self displacement (arrow grows from base, tip extends)
            sdx, sdy = arrow_self_disp(orient, delta)
            ax[k] += sdx; ay[k] += sdy
            ai[k] = new_idx
            # Child displacement
            cdx, cdy = growth_disp(orient, delta)
            _child_disp(ax, ay, tx, ty, k, cdx, cdy)
        if check_collisions(ax, ay, ao, ai):
            return None
        return pack(ax, ay, ao, ai, tx, ty)

    def apply_rotate(state, bi):
        """Apply rotation button bi. Returns new state or None."""
        ax, ay, ao, ai, tx, ty = unpack(state)
        for k in compact_btn_arrows[bi]:
            orient = ao[k]
            w, h = arrow_dims(orient, ai[k])
            # Determine rotation center (from zszsrsbyzi)
            if orient == 0:
                cx, cy = ax[k], ay[k] + h - CELL
            elif orient == 90:
                cx, cy = ax[k], ay[k]
            elif orient == 180:
                cx, cy = ax[k], ay[k]
            elif orient == 270:
                cx, cy = ax[k] + w - CELL, ay[k]

            # Rotate children around center
            for ctype, cidx in compact_children[k]:
                if ctype == 'arrow':
                    _rotate_child(ax, ay, ao, ai, tx, ty, cidx, cx, cy)
                elif ctype == 'target':
                    tdx = cx - tx[cidx]
                    tdy = cy - ty[cidx]
                    tx[cidx] = cx - tdy
                    ty[cidx] = cy + tdx

            # Self displacement (from zszsrsbyzi)
            if orient == 0:
                ax[k] += -h + CELL; ay[k] += h - CELL
            elif orient == 90:
                ay[k] += -w + CELL
            elif orient == 180:
                pass  # no self-move
            elif orient == 270:
                ax[k] += w - CELL
            ao[k] = ORIENT_NEXT[ao[k]]

            # Check double rotation: if has parent, and abs(arrow_orient-90 - parent_orient)==180
            pk = compact_parent[k]
            if pk >= 0:
                # Need pre-rotation orient for the check
                # The game checks CURRENT orients before rotating
                # But we already rotated once. The double-rotation check happens
                # BEFORE the first rotation in the game. So we need to check with
                # the pre-rotation orientations.
                # Since we already applied one rotation, the pre-rotation orient was
                # the PREVIOUS orient (before this apply_rotate call).
                pass  # handled below

        if check_collisions(ax, ay, ao, ai):
            return None
        return pack(ax, ay, ao, ai, tx, ty)

    # Wait - the double rotation logic is tricky. In the game, for each matching arrow:
    #   1. Check if parent exists
    #   2. Get parent orient and arrow orient (BEFORE any rotation)
    #   3. If abs(arrow_orient - 90 - parent_orient) == 180: rotate once extra
    #   4. Then rotate once
    # So total rotations = 1 or 2 per arrow.
    # I need to handle this correctly. Let me redo apply_rotate.

    def apply_rotate_v2(state, bi):
        """Apply rotation button bi with double-rotation check."""
        ax, ay, ao, ai, tx, ty = unpack(state)

        for k in compact_btn_arrows[bi]:
            # Check double-rotation BEFORE any rotation
            pk = compact_parent[k]
            n_rotations = 1
            if pk >= 0:
                po = ao[pk]
                ko = ao[k]
                if abs(ko - 90 - po) == 180:
                    n_rotations = 2

            for _ in range(n_rotations):
                orient = ao[k]
                w, h = arrow_dims(orient, ai[k])
                if orient == 0:
                    cx, cy = ax[k], ay[k] + h - CELL
                elif orient == 90:
                    cx, cy = ax[k], ay[k]
                elif orient == 180:
                    cx, cy = ax[k], ay[k]
                elif orient == 270:
                    cx, cy = ax[k] + w - CELL, ay[k]

                # Rotate children
                for ctype, cidx in compact_children[k]:
                    if ctype == 'arrow':
                        _rotate_child(ax, ay, ao, ai, tx, ty, cidx, cx, cy)
                    elif ctype == 'target':
                        tdx = cx - tx[cidx]
                        tdy = cy - ty[cidx]
                        tx[cidx] = cx - tdy
                        ty[cidx] = cy + tdx

                # Self displacement
                if orient == 0:
                    ax[k] += -h + CELL; ay[k] += h - CELL
                elif orient == 90:
                    ay[k] += -w + CELL
                elif orient == 180:
                    pass
                elif orient == 270:
                    ax[k] += w - CELL
                ao[k] = ORIENT_NEXT[ao[k]]

        if check_collisions(ax, ay, ao, ai):
            return None
        return pack(ax, ay, ao, ai, tx, ty)

    # Build action list
    actions = []
    for ti in range(len(track_info)):
        actions.append(('grow', ti))
        actions.append(('shrink', ti))
    for bi in range(len(btn_info)):
        actions.append(('rotate', bi))

    init_state = make_state()
    ax, ay, ao, ai, tx, ty = unpack(init_state)
    if check_win(tx, ty):
        return []

    queue = deque([(init_state, [])])
    visited = {init_state}
    max_depth = min(step_limit - 2, 50)
    explored = 0

    while queue and len(visited) < max_states:
        state, moves = queue.popleft()
        if len(moves) >= max_depth:
            continue
        explored += 1
        if explored % 50000 == 0:
            print(f"    BFS: {explored} expanded, {len(visited)} visited, depth={len(moves)}")

        for atype, aidx in actions:
            if atype == 'grow':
                new_state = apply_grow_shrink(state, aidx, True)
            elif atype == 'shrink':
                new_state = apply_grow_shrink(state, aidx, False)
            else:
                new_state = apply_rotate_v2(state, aidx)

            if new_state is not None and new_state not in visited:
                visited.add(new_state)
                new_moves = moves + [(atype, aidx)]
                nx, ny, no, ni, ntx, nty = unpack(new_state)
                if check_win(ntx, nty):
                    print(f"    BFS: found {len(new_moves)}-click solution ({explored} expanded, {len(visited)} visited)")
                    print(f"    Solution: {new_moves}")
                    return new_moves
                queue.append((new_state, new_moves))

    print(f"    BFS exhausted: {explored} expanded, {len(visited)} visited")
    return None


# ============ Non-rotation solvers ============

def bfs_solve(arrow_list, target_list, goal_positions, track_info,
              obstacle_pixels, step_limit, max_states=500_000):
    """BFS over track-index state space (no rotations)."""
    n_tracks = len(track_info)
    n_arrows = len(arrow_list)
    initial = tuple(t['index0'] for t in track_info)
    orient = tuple(a['orient0'] for a in arrow_list if not a['is_obstacle'])

    # Precompute orient mapping
    orient_full = []
    oi = 0
    for a in arrow_list:
        if a['is_obstacle']:
            orient_full.append(a['orient0'])
        else:
            orient_full.append(orient[oi])
            oi += 1

    def eval_state(track_tuple):
        """Compute positions, check collision and win in one pass."""
        ax = [a['x0'] for a in arrow_list]
        ay = [a['y0'] for a in arrow_list]
        aidx = [a['index0'] for a in arrow_list]
        tx = [t['x0'] for t in target_list]
        ty = [t['y0'] for t in target_list]

        for ti, tinfo in enumerate(track_info):
            new_idx = track_tuple[ti]
            for ai in tinfo['arrow_ids']:
                delta = new_idx - arrow_list[ai]['index0']
                if delta == 0: continue
                o = orient_full[ai]
                if o == 0: ay[ai] -= delta * CELL
                elif o == 270: ax[ai] -= delta * CELL
                aidx[ai] = new_idx
                cdx, cdy = growth_disp(o, delta)
                _apply_child_disp(ax, ay, tx, ty, arrow_list, ai, cdx, cdy)

        # Check collision
        occ = []
        for i in range(n_arrows):
            if arrow_list[i]['is_obstacle']:
                occ.append(None); continue
            w, h = arrow_dims(orient_full[i], aidx[i])
            occ.append(set((ax[i]+dx, ay[i]+dy) for dy in range(h) for dx in range(w)))
        for i in range(n_arrows):
            if occ[i] is None: continue
            if occ[i] & obstacle_pixels: return True, False
            for j in range(i+1, n_arrows):
                if occ[j] is None: continue
                if occ[i] & occ[j]: return True, False

        # Check win
        tpos = set((tx[i], ty[i]) for i in range(len(target_list)))
        return False, goal_positions <= tpos

    _, win = eval_state(initial)
    if win: return []

    queue = deque([(initial, [])])
    visited = {initial}
    max_depth = min(step_limit - 2, 50)

    while queue and len(visited) < max_states:
        state, moves = queue.popleft()
        if len(moves) >= max_depth: continue
        for ti in range(n_tracks):
            for action, delta in [('grow', 1), ('shrink', -1)]:
                new_idx = state[ti] + delta
                if new_idx < 1: continue
                new_state = list(state)
                new_state[ti] = new_idx
                new_state = tuple(new_state)
                if new_state in visited: continue

                collision, win = eval_state(new_state)
                visited.add(new_state)
                if collision: continue
                new_moves = moves + [(action, ti)]
                if win: return new_moves
                queue.append((new_state, new_moves))
    return None


def greedy_solve(arrow_list, target_list, goal_positions, track_info,
                 obstacle_pixels, step_limit):
    """Analytical: compute target growth, shrink non-needed to min, interleave."""
    n_tracks = len(track_info)
    orient = tuple(a['orient0'] for a in arrow_list if not a['is_obstacle'])

    needed_tracks = set()
    target_deltas = {}
    for ti_tgt in range(len(target_list)):
        tgt = target_list[ti_tgt]
        goal = min(goal_positions,
                   key=lambda g: abs(g[0] - tgt['x0']) + abs(g[1] - tgt['y0']))
        dx = goal[0] - tgt['x0']
        dy = goal[1] - tgt['y0']
        for ai, a in enumerate(arrow_list):
            for ctype, cidx in a['children']:
                if ctype == 'target' and cidx == ti_tgt:
                    o = a['orient0']
                    delta = 0
                    if o == 90 and dx > 0: delta = dx // CELL
                    elif o == 270 and dx < 0: delta = (-dx) // CELL
                    elif o == 180 and dy > 0: delta = dy // CELL
                    elif o == 0 and dy < 0: delta = (-dy) // CELL
                    ti = a['track_idx']
                    if ti >= 0 and delta > 0:
                        needed_tracks.add(ti)
                        target_deltas[ti] = max(target_deltas.get(ti, 0), delta)
                    break

    track_state = {}
    for ti in range(n_tracks):
        if ti in needed_tracks:
            track_state[ti] = track_info[ti]['index0'] + target_deltas[ti]
        else:
            track_state[ti] = 1
    target_tuple = tuple(track_state.get(ti, track_info[ti]['index0']) for ti in range(n_tracks))

    full_state = (target_tuple, orient)
    ax, ay, aidx, aorient, tx, ty = compute_positions(full_state, arrow_list, target_list, track_info)
    tpos = set((tx[i], ty[i]) for i in range(len(target_list)))

    if not (goal_positions <= tpos):
        return None
    if check_collision_from_state(full_state, arrow_list, target_list, track_info, obstacle_pixels):
        return None

    # Interleave: shrinks first, then grows
    n_tracks_total = len(track_info)
    steps_needed = {}
    for ti in range(n_tracks_total):
        delta = target_tuple[ti] - track_info[ti]['index0']
        if delta != 0:
            steps_needed[ti] = delta

    current = list(track_info[ti]['index0'] for ti in range(n_tracks_total))
    remaining = dict(steps_needed)
    actions = []

    for _ in range(500):
        if not remaining: break
        shrinks = [ti for ti, d in remaining.items() if d < 0]
        grows = [ti for ti, d in remaining.items() if d > 0]
        made_progress = False
        for ti in shrinks + grows:
            d = remaining[ti]
            if d == 0:
                del remaining[ti]; continue
            step = 1 if d > 0 else -1
            test = list(current)
            test[ti] += step
            if test[ti] < 1: continue
            test_state = (tuple(test), orient)
            if not check_collision_from_state(test_state, arrow_list, target_list,
                                             track_info, obstacle_pixels):
                current[ti] += step
                remaining[ti] -= step
                if remaining[ti] == 0: del remaining[ti]
                actions.append(('grow' if step > 0 else 'shrink', ti))
                made_progress = True
                break
        if not made_progress:
            return None

    if remaining: return None
    final_state = (tuple(current), orient)
    if check_win(final_state, arrow_list, target_list, track_info, goal_positions):
        return actions
    return None


def verify_analytical_solution(solution, track_info, btn_info, arrow_list):
    """Verify analytical solution by replaying through game engine.
    Returns True if game engine also reports win after all actions."""
    level = game.current_level
    arrows = level.get_sprites_by_tag("agujdcrunq")
    targets = level.get_sprites_by_tag("zylvdxoiuq")
    save_data = [(s.x, s.y, s.pixels.copy()) for s in arrows + targets]

    for step_i, (atype, aidx) in enumerate(solution):
        if atype in ('grow', 'shrink'):
            track_sprite = track_info[aidx]['sprite']
            game.acgkuydgqx = dict()
            for arrow_s in game.dfyrdkjdcj[track_sprite]:
                game.yxdtnnaclf(arrow_s)
                cur = max(arrow_s.height, arrow_s.width) // CELL
                new_val = cur + (1 if atype == 'grow' else -1)
                if new_val < 1:
                    for sp in game.acgkuydgqx:
                        cl = game.acgkuydgqx[sp]; sp.set_position(cl.x, cl.y); sp.pixels = cl.pixels
                    game.acgkuydgqx = dict()
                    print(f"    VERIFY FAIL step {step_i}: shrink below 1")
                    # Restore
                    for i, s in enumerate(arrows + targets):
                        s.set_position(save_data[i][0], save_data[i][1]); s.pixels = save_data[i][2].copy()
                    return False
                game.ldmrjbrvwa(arrow_s, new_val)
            if game.ulzimrggno():
                for sp in game.acgkuydgqx:
                    cl = game.acgkuydgqx[sp]; sp.set_position(cl.x, cl.y); sp.pixels = cl.pixels
                game.acgkuydgqx = dict()
                print(f"    VERIFY FAIL step {step_i}: collision after {atype} T{aidx}")
                for i, s in enumerate(arrows + targets):
                    s.set_position(save_data[i][0], save_data[i][1]); s.pixels = save_data[i][2].copy()
                return False
            game.acgkuydgqx = dict()
        elif atype == 'rotate':
            btn_sprite = btn_info[aidx]['sprite']
            btn_color = int(btn_sprite.pixels[btn_sprite.height // 2, btn_sprite.height // 2])
            game.acgkuydgqx = dict()
            for arrow_s in arrows:
                if int(arrow_s.pixels[1, 1]) == btn_color:
                    game.yxdtnnaclf(arrow_s)
                    parents = [k for k in game.enplxxgoja if arrow_s in game.enplxxgoja[k]]
                    if parents:
                        p = parents[0]
                        po = game.fhkoulsvoi(p)
                        ao = game.fhkoulsvoi(arrow_s)
                        if abs(ao - 90 - po) == 180:
                            game.zszsrsbyzi(arrow_s)
                        game.zszsrsbyzi(arrow_s)
                    else:
                        game.zszsrsbyzi(arrow_s)
            if game.ulzimrggno():
                for sp in game.acgkuydgqx:
                    cl = game.acgkuydgqx[sp]; sp.set_position(cl.x, cl.y); sp.pixels = cl.pixels
                game.acgkuydgqx = dict()
                print(f"    VERIFY FAIL step {step_i}: collision after rotate B{aidx}")
                for i, s in enumerate(arrows + targets):
                    s.set_position(save_data[i][0], save_data[i][1]); s.pixels = save_data[i][2].copy()
                return False
            game.acgkuydgqx = dict()

    won = game.vodebmynqs()
    tpos = [(t.x, t.y) for t in targets]
    print(f"    VERIFY: win={won}, targets={tpos}")

    # Restore
    for i, s in enumerate(arrows + targets):
        s.set_position(save_data[i][0], save_data[i][1]); s.pixels = save_data[i][2].copy()
    return won


def debug_analytical_vs_engine(solution, track_info, btn_info, arrow_list, target_list,
                              obstacle_pixels):
    """Compare analytical vs game engine step by step."""
    level = game.current_level
    arrows = level.get_sprites_by_tag("agujdcrunq")
    targets = level.get_sprites_by_tag("zylvdxoiuq")
    save_data = [(s.x, s.y, s.pixels.copy()) for s in arrows + targets]

    # Build compact state (same as analytical_bfs)
    non_obs = [i for i, a in enumerate(arrow_list) if not a['is_obstacle']]
    orig_to_compact = {orig: k for k, orig in enumerate(non_obs)}
    noa = len(non_obs)
    nt = len(target_list)

    # Initial analytical state
    state_parts = []
    for orig in non_obs:
        a = arrow_list[orig]
        state_parts.extend([a['x0'], a['y0'], a['orient0'], a['index0']])
    for t in target_list:
        state_parts.extend([t['x0'], t['y0']])

    for step_i, (atype, aidx) in enumerate(solution[:5]):  # first 5 steps only
        print(f"\n  === Step {step_i}: {atype} {aidx} ===")

        # Apply to game engine
        if atype in ('grow', 'shrink'):
            track_sprite = track_info[aidx]['sprite']
            game.acgkuydgqx = dict()
            for arrow_s in game.dfyrdkjdcj[track_sprite]:
                game.yxdtnnaclf(arrow_s)
                cur = max(arrow_s.height, arrow_s.width) // CELL
                new_val = cur + (1 if atype == 'grow' else -1)
                game.ldmrjbrvwa(arrow_s, new_val)
            game.acgkuydgqx = dict()
        elif atype == 'rotate':
            btn_sprite = btn_info[aidx]['sprite']
            btn_color = int(btn_sprite.pixels[btn_sprite.height // 2, btn_sprite.height // 2])
            game.acgkuydgqx = dict()
            for arrow_s in arrows:
                if int(arrow_s.pixels[1, 1]) == btn_color:
                    game.yxdtnnaclf(arrow_s)
                    parents = [k for k in game.enplxxgoja if arrow_s in game.enplxxgoja[k]]
                    if parents:
                        p = parents[0]
                        po = game.fhkoulsvoi(p)
                        ao = game.fhkoulsvoi(arrow_s)
                        if abs(ao - 90 - po) == 180:
                            game.zszsrsbyzi(arrow_s)
                        game.zszsrsbyzi(arrow_s)
                    else:
                        game.zszsrsbyzi(arrow_s)
            game.acgkuydgqx = dict()

        # Print game state
        print(f"  ENGINE:")
        for i, a in enumerate(arrows):
            o = game.fhkoulsvoi(a)
            idx = max(a.width, a.height) // CELL
            print(f"    A{i}: ({a.x},{a.y}) o={o}° idx={idx}")
        for i, t in enumerate(targets):
            print(f"    T{i}: ({t.x},{t.y})")

    # Restore
    for i, s in enumerate(arrows + targets):
        s.set_position(save_data[i][0], save_data[i][1]); s.pixels = save_data[i][2].copy()


def solve_level(arrow_list, target_list, goal_positions, track_info, btn_info,
                obstacle_pixels, step_limit):
    n_tracks = len(track_info)
    has_rotations = len(btn_info) > 0

    if not has_rotations:
        if n_tracks <= 5:
            result = bfs_solve(arrow_list, target_list, goal_positions, track_info,
                              obstacle_pixels, step_limit)
            if result is not None: return result
        result = greedy_solve(arrow_list, target_list, goal_positions, track_info,
                             obstacle_pixels, step_limit)
        if result is not None: return result
        if n_tracks > 5:
            return bfs_solve(arrow_list, target_list, goal_positions, track_info,
                            obstacle_pixels, step_limit, max_states=3_000_000)
        return None

    # Levels with rotations: analytical BFS
    # First check which targets need solving and which are already at goals
    # If a target is already at goal and its chain is independent, exclude it
    solved_targets = set()
    for ti_idx, t in enumerate(target_list):
        if (t['x0'], t['y0']) in goal_positions:
            solved_targets.add(ti_idx)

    if solved_targets and len(solved_targets) < len(target_list):
        # Find which arrows/tracks are needed for unsolved targets
        needed_arrows = set()
        needed_tracks = set()
        for ti_idx, t in enumerate(target_list):
            if ti_idx in solved_targets:
                continue
            # Walk up the parent chain to find all involved arrows
            for ai, a in enumerate(arrow_list):
                for ctype, cidx in a['children']:
                    if ctype == 'target' and cidx == ti_idx:
                        # Trace up chain
                        cur = ai
                        while cur >= 0:
                            needed_arrows.add(cur)
                            if arrow_list[cur]['track_idx'] >= 0:
                                needed_tracks.add(arrow_list[cur]['track_idx'])
                            cur = arrow_list[cur]['parent_idx']

        # Check which buttons affect needed arrows
        btn_colors = set(arrow_list[ai]['color'] for ai in needed_arrows)
        needed_btns = [bi for bi, b in enumerate(btn_info) if b['color'] in btn_colors]

        # Check if solved targets are independent
        solved_independent = True
        for ti_idx in solved_targets:
            for ai, a in enumerate(arrow_list):
                for ctype, cidx in a['children']:
                    if ctype == 'target' and cidx == ti_idx:
                        cur = ai
                        while cur >= 0:
                            if cur in needed_arrows:
                                solved_independent = False
                                break
                            if arrow_list[cur]['track_idx'] in needed_tracks:
                                solved_independent = False
                                break
                            if arrow_list[cur]['color'] in btn_colors:
                                solved_independent = False
                                break
                            cur = arrow_list[cur]['parent_idx']
                        if not solved_independent:
                            break
                if not solved_independent:
                    break
            if not solved_independent:
                break

        if solved_independent:
            # Remove independent solved targets and their arrows from the problem
            # Build filtered lists
            filt_targets = [t for i, t in enumerate(target_list) if i not in solved_targets]
            filt_goals = set()
            for gx, gy in goal_positions:
                if not any(target_list[si]['x0'] == gx and target_list[si]['y0'] == gy
                          for si in solved_targets):
                    filt_goals.add((gx, gy))

            # Remap target indices in arrow children
            target_remap = {}
            ni = 0
            for i in range(len(target_list)):
                if i not in solved_targets:
                    target_remap[i] = ni
                    ni += 1

            filt_arrows = []
            for a in arrow_list:
                new_a = dict(a)
                new_children = []
                for ctype, cidx in a['children']:
                    if ctype == 'target' and cidx in solved_targets:
                        continue  # skip solved target children
                    elif ctype == 'target':
                        new_children.append(('target', target_remap[cidx]))
                    else:
                        new_children.append((ctype, cidx))
                new_a['children'] = new_children
                filt_arrows.append(new_a)

            # Filter tracks: exclude tracks only used by independent solved arrows
            skip_tracks = set()
            for ti, tinfo in enumerate(track_info):
                if ti not in needed_tracks:
                    skip_tracks.add(ti)

            filt_tracks = [t for i, t in enumerate(track_info) if i not in skip_tracks]
            filt_btns = [btn_info[bi] for bi in needed_btns]

            # Remap track indices in arrows
            track_remap = {}
            ni = 0
            for i in range(len(track_info)):
                if i not in skip_tracks:
                    track_remap[i] = ni
                    ni += 1
            for a in filt_arrows:
                if a['track_idx'] >= 0 and a['track_idx'] in track_remap:
                    a['track_idx'] = track_remap[a['track_idx']]
                elif a['track_idx'] >= 0:
                    a['track_idx'] = -1  # not in filtered tracks

            print(f"  Factored: {len(solved_targets)} targets already solved, "
                  f"{len(filt_arrows)}A {len(filt_tracks)}T {len(filt_btns)}R "
                  f"(was {len(arrow_list)}A {len(track_info)}T {len(btn_info)}R)")

            result = analytical_bfs(filt_tracks, filt_btns, filt_arrows, filt_targets,
                                   filt_goals, obstacle_pixels, step_limit)
            if result is not None:
                # Remap track/button indices back
                inv_track_remap = {v: k for k, v in track_remap.items()}
                inv_btn_map = {i: needed_btns[i] for i in range(len(needed_btns))}
                remapped = []
                for atype, aidx in result:
                    if atype in ('grow', 'shrink'):
                        remapped.append((atype, inv_track_remap[aidx]))
                    elif atype == 'rotate':
                        remapped.append((atype, inv_btn_map[aidx]))
                result = remapped
                if verify_analytical_solution(result, track_info, btn_info, arrow_list):
                    return result
                print(f"  Factored solution failed verification!")
            else:
                print(f"  Factored BFS failed, trying full...")

    print(f"  Using analytical BFS ({len(btn_info)} rotate buttons)")
    result = analytical_bfs(track_info, btn_info, arrow_list, target_list, goal_positions,
                           obstacle_pixels, step_limit)
    if result is not None:
        if verify_analytical_solution(result, track_info, btn_info, arrow_list):
            return result
        print(f"  Analytical solution failed verification!")
        debug_analytical_vs_engine(result, track_info, btn_info, arrow_list, target_list,
                                  obstacle_pixels)
        return None
    return None


def execute_solution(solution, track_info, btn_info):
    global fd
    actions = 0
    for action_type, idx in solution:
        if action_type in ('grow', 'shrink'):
            track = track_info[idx]
            t = track['sprite']
            is_h = track['is_h']
            half = (t.width if is_h else t.height) // 2
            if action_type == 'grow':
                if is_h:
                    gx, gy = t.x + half + 2, t.y + t.height // 2
                else:
                    gx, gy = t.x + t.width // 2, t.y + half + 2
            else:
                if is_h:
                    gx, gy = t.x + half - 2, t.y + t.height // 2
                else:
                    gx, gy = t.x + t.width // 2, t.y + half - 2
            dx, dy = grid_to_display(gx, gy)
            fd = env.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
            actions += 1
        elif action_type == 'rotate':
            btn = btn_info[idx]
            b = btn['sprite']
            bx = b.x + b.width // 2
            by = b.y + b.height // 2
            dx, dy = grid_to_display(bx, by)
            fd = env.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
            actions += 1
    return actions


total_actions = 0

for lv in range(len(game._levels)):
    level = game.current_level
    step_limit = level.get_data("StepCounter")

    arrow_list, target_list, goal_positions, track_info, btn_info, \
        obstacle_pixels = parse_level(level)

    n_obs = sum(1 for a in arrow_list if a['is_obstacle'])
    print(f"\n=== Level {lv} (steps={step_limit}) ===")
    print(f"  {len(track_info)}T {len(arrow_list)}A({n_obs}obs) {len(target_list)}tgt "
          f"{len(goal_positions)}G {len(btn_info)}R")

    for i, a in enumerate(arrow_list):
        tag = "OBS" if a['is_obstacle'] else f"T{a['track_idx']}" if a['track_idx'] >= 0 else "CHD"
        print(f"  A{i}[{tag}]: ({a['x0']},{a['y0']}) idx={a['index0']} o={a['orient0']}° "
              f"c={a['color']} ch={a['children']} par={a['parent_idx']}")
    for i, t in enumerate(target_list):
        print(f"  Tgt{i}: ({t['x0']},{t['y0']})")
    print(f"  Goals: {goal_positions}")

    solution = solve_level(arrow_list, target_list, goal_positions, track_info, btn_info,
                          obstacle_pixels, step_limit)

    if solution is None:
        print(f"  ERROR: No solution found!")
        break

    has_rot = any(a[0] == 'rotate' for a in solution)
    print(f"  Solution: {len(solution)} clicks{' (rotations)' if has_rot else ''}")

    lv_actions = execute_solution(solution, track_info, btn_info)
    total_actions += lv_actions

    actual_targets = level.get_sprites_by_tag("zylvdxoiuq")
    actual_tpos = [(t.x, t.y) for t in actual_targets]
    print(f"  {lv_actions} actions, completed={fd.levels_completed}, state={fd.state.name}")
    print(f"  Actual targets={actual_tpos}")

    if fd.state.name == 'WIN':
        print(f"\n=== ALL SOLVED! Total: {total_actions} ===")
        break

    if fd.levels_completed <= lv:
        print(f"  Level not completed!")
        break

print(f"\nFinal: completed={fd.levels_completed}, state={fd.state.name}, total={total_actions}")
