#!/usr/bin/env python3
"""L7 solver: saboteur-aware planner.

Strategy:
- Primary policy: destroy whites ASAP, then rescue wrong-slot pieces.
- Phase A: chase & destroy a white (facing-adjacent ACT5). Use walking to predict white's next move.
- Phase B (whites gone or unreachable): grab any loose/wrong-slot piece, carry to nearest aligned correct slot.
- Phase C: wait (idle safely) if blues are carrying all remaining loose pieces.
"""
import sys, os, time
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")

src = open('arc-agi-3/experiments/wa30_solve_final.py').read()
cut = src.index('total_actions = 0')
exec(src[:cut], globals())

def safe_idle_action(px, py, collidable, blocked):
    """Pick a move that does nothing useful but keeps player safe (doesn't enter wrong area).
    Bounce against wall: try UP, if blocked try DOWN."""
    for a, (dx,dy) in [(UP,(0,-4)),(DOWN,(0,4)),(LEFT,(-4,0)),(RIGHT,(4,0))]:
        nx, ny = px+dx, py+dy
        if (nx,ny) in collidable or (nx,ny) in blocked: continue
        if not (0<=nx<64 and 0<=ny<64): continue
        return a
    return UP

def l7_plan(player, goals, whites, blues, collidable, blocked, slots, slots2, slot_aligned):
    px, py, prot = player.x, player.y, player.rotation
    fx, fy = facing_pos(px, py, prot)
    pc = game.nsevyuople.get(player, None)

    # If carrying, deliver or drop
    if pc is not None:
        cpx, cpy = pc.x, pc.y
        cdx, cdy = cpx - px, cpy - py
        # Already at correct slot? drop.
        if (cpx, cpy) in slots:
            return ACT5
        # Find delivery target (any unfilled aligned slot)
        placed_pos = set((g.x,g.y) for g in goals
                         if (g.x,g.y) in slots and g not in game.zmqreragji and g is not pc)
        deliver_targets = set()
        for sx, sy in slot_aligned:
            if (sx,sy) in placed_pos: continue
            tpx, tpy = sx - cdx, sy - cdy
            if 0 <= tpx < 64 and 0 <= tpy < 64:
                deliver_targets.add((tpx, tpy))
        obs = collidable.copy()
        obs.discard((px,py))
        obs.discard((cpx,cpy))
        # Remove other pieces on wrong-slots? No — they're still collidable
        path = bfs_carry((px,py), deliver_targets, obs, cdx, cdy, blocked)
        if path and len(path) > 1:
            return pos_to_action(px, py, path[1][0], path[1][1])
        if path and len(path) == 1:
            return ACT5
        # Can't deliver this carry offset. Drop, reapproach.
        return ACT5

    # NOT CARRYING

    # Priority 1: destroy whites if facing or adjacent
    for w in whites:
        if (w.x, w.y) == (fx, fy):
            return ACT5
    for w in whites:
        if adjacent((px,py), (w.x,w.y)):
            # Face the white
            dx, dy = w.x - px, w.y - py
            if abs(dy) >= abs(dx):
                return UP if dy < 0 else DOWN
            return RIGHT if dx > 0 else LEFT

    # Priority 2: chase whites (BFS to adjacency)
    obs_plain = collidable.copy()
    obs_plain.discard((px,py))
    best_white_path = None
    for w in whites:
        adj_cells = set()
        for ddx, ddy in DIR_LIST:
            ax, ay = w.x+ddx, w.y+ddy
            if 0<=ax<64 and 0<=ay<64:
                adj_cells.add((ax,ay))
        path = bfs_to_targets((px,py), adj_cells, obs_plain)
        if path is None: continue
        if best_white_path is None or len(path) < len(best_white_path):
            best_white_path = path

    # Priority 3: find a piece to rescue/deliver
    # All pieces not on correct slot, not being carried by blue (player can take)
    all_pieces_info = []
    blue_sprites = set(b for b in blues)
    for g in goals:
        if (g.x,g.y) in slots and g not in game.zmqreragji: continue  # already placed
        claimed = game.zmqreragji.get(g)
        if claimed and claimed in blue_sprites: continue  # blue delivering
        if claimed and claimed is player: continue  # we are carrying (handled above)
        # Include pieces claimed by whites — player can destroy white to get them
        all_pieces_info.append((g, claimed))

    best_piece_plan = None  # (total_cost, approach_path, piece, offset_adj)
    for g, claimed in all_pieces_info:
        gx, gy = g.x, g.y
        if claimed is not None:
            # carried by white — skip for direct pickup (handled via white destroy)
            continue
        for ddx, ddy in DIR_LIST:
            ax, ay = gx+ddx, gy+ddy
            if not (0<=ax<64 and 0<=ay<64): continue
            if (ax,ay) in obs_plain and (ax,ay) != (px,py): continue
            off_x, off_y = gx - ax, gy - ay
            # Find delivery target
            placed_pos = set((gg.x,gg.y) for gg in goals
                             if (gg.x,gg.y) in slots and gg not in game.zmqreragji and gg is not g)
            deliver_targets = set()
            for sx, sy in slot_aligned:
                if (sx,sy) in placed_pos: continue
                tpx, tpy = sx - off_x, sy - off_y
                if 0 <= tpx < 64 and 0 <= tpy < 64:
                    deliver_targets.add((tpx, tpy))
            if not deliver_targets: continue
            carry_obs = obs_plain.copy()
            carry_obs.discard((gx,gy))
            cpath = bfs_carry((ax,ay), deliver_targets, carry_obs, off_x, off_y, blocked)
            if cpath is None: continue
            apath = bfs_path((px,py), (ax,ay), obs_plain)
            if apath is None: continue
            cost = len(apath) + len(cpath)
            if best_piece_plan is None or cost < best_piece_plan[0]:
                best_piece_plan = (cost, apath, g, (ax,ay))

    # Decision: ALWAYS chase whites if any exist and are reachable (top priority)
    if whites and best_white_path is not None:
        if len(best_white_path) > 1:
            return pos_to_action(px, py, best_white_path[1][0], best_white_path[1][1])
        return ACT5

    if best_piece_plan:
        _, apath, piece, (ax,ay) = best_piece_plan
        if (ax,ay) == (px,py):
            # face the piece
            gx, gy = piece.x, piece.y
            if (fx,fy) == (gx,gy):
                return ACT5
            dx, dy = gx-px, gy-py
            if abs(dy) >= abs(dx):
                return UP if dy < 0 else DOWN
            return RIGHT if dx > 0 else LEFT
        if len(apath) > 1:
            return pos_to_action(px, py, apath[1][0], apath[1][1])

    # Fallback: chase whites even if far
    if whites and best_white_path is not None and len(best_white_path) > 1:
        return pos_to_action(px, py, best_white_path[1][0], best_white_path[1][1])

    # Nothing useful — idle
    return safe_idle_action(px, py, collidable, blocked)


def run_l7(all_solutions, verbose=False):
    fd = replay_from_solutions(all_solutions)
    slots = game.wyzquhjerd
    slots2 = game.lqctaojiby
    slot_aligned = set((sx,sy) for sx,sy in slots if sx%4==0 and sy%4==0)
    solution = []
    best_placed_seen = 0
    for step in range(150):
        player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
        goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
        whites = game.current_level.get_sprites_by_tag("ysysltqlke")
        blues = game.current_level.get_sprites_by_tag("kdweefinfi")
        collidable = set(game.pkbufziase)
        blocked = set(game.qthdiggudy)
        placed = sum(1 for g in goals if (g.x,g.y) in slots and g not in game.zmqreragji)
        best_placed_seen = max(best_placed_seen, placed)
        action = l7_plan(player, goals, whites, blues, collidable, blocked, slots, slots2, slot_aligned)
        fd = env.step(action)
        solution.append(action)
        if verbose and step % 10 == 0:
            ws = [(w.x,w.y) for w in whites]
            print(f"s{step:3d} a={ACTION_NAMES[action]} p=({player.x},{player.y}) placed={placed} W={ws} |W|={len(whites)}")
        if fd.levels_completed > 7 or fd.state.name == 'WIN':
            print(f"L7 SOLVED! {len(solution)} moves")
            return solution
        if fd.state.name in ('LOSE','GAME_OVER'):
            print(f"L7 lost at step {step}, best_placed={best_placed_seen}")
            return None
    print(f"L7 out of steps, best_placed={best_placed_seen}")
    return None


KNOWN = KNOWN_SOLUTIONS
all_solutions = []
for lv in range(7):
    fd = replay_from_solutions(all_solutions)
    blues = game.current_level.get_sprites_by_tag("kdweefinfi")
    whites = game.current_level.get_sprites_by_tag("ysysltqlke")
    has_ai = len(blues)>0 or len(whites)>0
    if lv in KNOWN:
        sol = string_to_actions(KNOWN[lv])
    elif not has_ai:
        sol = solve_no_ai()
    else:
        sol = solve_with_ai(all_solutions)
    fd = replay_from_solutions(all_solutions)
    before = fd.levels_completed
    actual = []
    for m in sol:
        fd = env.step(m); actual.append(m)
        if fd.levels_completed > before or fd.state.name == 'WIN': break
    all_solutions.append(actual)

print("=== L7 attempt ===")
sol = run_l7(all_solutions, verbose=True)
if sol:
    print("Solution:", ''.join(ACTION_NAMES[a] for a in sol))
