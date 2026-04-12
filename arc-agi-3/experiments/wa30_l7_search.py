#!/usr/bin/env python3
"""L7: aggressive randomized search with saboteur-aware policy."""
import sys, os, time, random
sys.stdout.reconfigure(line_buffering=True)
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")

src = open('arc-agi-3/experiments/wa30_solve_final.py').read()
cut = src.index('total_actions = 0')
exec(src[:cut], globals())

def smart_action(player, goals, whites, blues, collidable, blocked, slots, slots2, slot_aligned):
    """Improved greedy with white-aware rescue + direct pickup of wrong-slot pieces."""
    px, py, prot = player.x, player.y, player.rotation
    fx, fy = facing_pos(px, py, prot)
    pc = game.nsevyuople.get(player, None)

    if pc is not None:
        cpx, cpy = pc.x, pc.y
        cdx, cdy = cpx-px, cpy-py
        if (cpx,cpy) in slots:
            return ACT5
        placed_pos = set((g.x,g.y) for g in goals
                         if (g.x,g.y) in slots and g not in game.zmqreragji and g is not pc)
        targets = set()
        for sx,sy in slot_aligned:
            if (sx,sy) in placed_pos: continue
            tpx, tpy = sx-cdx, sy-cdy
            if 0<=tpx<64 and 0<=tpy<64:
                targets.add((tpx,tpy))
        obs = collidable.copy()
        obs.discard((px,py))
        obs.discard((cpx,cpy))
        path = bfs_carry((px,py), targets, obs, cdx, cdy, blocked)
        if path and len(path) > 1:
            return pos_to_action(px,py,path[1][0],path[1][1])
        if path and len(path) == 1:
            return ACT5
        return ACT5  # stuck, drop

    # Not carrying.
    # P1: kill white if facing
    for w in whites:
        if (w.x,w.y) == (fx,fy):
            return ACT5
    # P2: adjacent to white -> face it
    for w in whites:
        if adjacent((px,py),(w.x,w.y)):
            dx,dy = w.x-px, w.y-py
            if abs(dy) >= abs(dx):
                return UP if dy<0 else DOWN
            return RIGHT if dx>0 else LEFT

    # P3: find pieces — BOTH loose AND wrong-slot are valid targets
    obs = collidable.copy()
    obs.discard((px,py))
    blue_set = set(blues)
    # Pieces player can claim: not at correct slot, not carried by a blue
    targets_pieces = []
    for g in goals:
        if (g.x,g.y) in slots and g not in game.zmqreragji: continue
        claimed = game.zmqreragji.get(g)
        if claimed in blue_set: continue  # leave to blue
        if claimed is player: continue
        if claimed is not None and 'ysysltqlke' in claimed.tags:
            # carried by white — need to kill white first (covered above)
            continue
        targets_pieces.append(g)

    # Score each piece by delivery cost
    best = None
    for g in targets_pieces:
        wrong_bonus = 0  # disabled
        gx, gy = g.x, g.y
        for ddx, ddy in DIR_LIST:
            ax, ay = gx+ddx, gy+ddy
            if not (0<=ax<64 and 0<=ay<64): continue
            if (ax,ay) in obs and (ax,ay) != (px,py): continue
            off_x, off_y = gx-ax, gy-ay
            placed_pos = set((gg.x,gg.y) for gg in goals
                             if (gg.x,gg.y) in slots and gg not in game.zmqreragji and gg is not g)
            dt = set()
            for sx,sy in slot_aligned:
                if (sx,sy) in placed_pos: continue
                tpx, tpy = sx-off_x, sy-off_y
                if 0<=tpx<64 and 0<=tpy<64:
                    dt.add((tpx,tpy))
            if not dt: continue
            carry_obs = obs.copy()
            carry_obs.discard((gx,gy))
            cpath = bfs_carry((ax,ay), dt, carry_obs, off_x, off_y, blocked)
            if cpath is None: continue
            apath = bfs_path((px,py),(ax,ay),obs)
            if apath is None: continue
            cost = len(apath) + len(cpath) + wrong_bonus
            if best is None or cost < best[0]:
                best = (cost, apath, g, (ax,ay))

    if best:
        cost, apath, piece, (ax,ay) = best
        if (ax,ay) == (px,py):
            gx, gy = piece.x, piece.y
            if (fx,fy)==(gx,gy): return ACT5
            dx, dy = gx-px, gy-py
            if abs(dy)>=abs(dx):
                return UP if dy<0 else DOWN
            return RIGHT if dx>0 else LEFT
        if len(apath)>1:
            return pos_to_action(px,py,apath[1][0],apath[1][1])

    # P4: chase white (long path)
    best_wp = None
    for w in whites:
        adj_cells = set()
        for ddx,ddy in DIR_LIST:
            ax,ay = w.x+ddx, w.y+ddy
            if 0<=ax<64 and 0<=ay<64 and (ax,ay) not in collidable:
                adj_cells.add((ax,ay))
        if not adj_cells: continue
        p = bfs_to_targets((px,py), adj_cells, obs)
        if p and (best_wp is None or len(p) < len(best_wp)):
            best_wp = p
    if best_wp and len(best_wp) > 1:
        return pos_to_action(px,py,best_wp[1][0],best_wp[1][1])

    # P5: wait
    for a, (dx,dy) in [(UP,(0,-4)),(DOWN,(0,4)),(LEFT,(-4,0)),(RIGHT,(4,0))]:
        nx, ny = px+dx, py+dy
        if (nx,ny) in collidable or (nx,ny) in blocked: continue
        return a
    return UP


def run(all_solutions, seed=0, randomness=0.0):
    random.seed(seed)
    fd = replay_from_solutions(all_solutions)
    slots = game.wyzquhjerd
    slots2 = game.lqctaojiby
    slot_aligned = set((sx,sy) for sx,sy in slots if sx%4==0 and sy%4==0)
    solution = []
    best_placed = 0
    for step in range(150):
        player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
        goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
        whites = game.current_level.get_sprites_by_tag("ysysltqlke")
        blues = game.current_level.get_sprites_by_tag("kdweefinfi")
        collidable = set(game.pkbufziase)
        blocked = set(game.qthdiggudy)
        placed = sum(1 for g in goals if (g.x,g.y) in slots and g not in game.zmqreragji)
        best_placed = max(best_placed, placed)
        if random.random() < randomness:
            action = random.choice([UP,DOWN,LEFT,RIGHT,ACT5])
        else:
            action = smart_action(player,goals,whites,blues,collidable,blocked,slots,slots2,slot_aligned)
        fd = env.step(action)
        solution.append(action)
        if fd.levels_completed > 7 or fd.state.name == 'WIN':
            return solution, 13, step
        if fd.state.name in ('LOSE','GAME_OVER'):
            return None, best_placed, step
    return None, best_placed, 150


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

t0 = time.time()
# Deterministic smart_action with verbose trace
fd = replay_from_solutions(all_solutions)
slots = game.wyzquhjerd
slots2 = game.lqctaojiby
slot_aligned = set((sx,sy) for sx,sy in slots if sx%4==0 and sy%4==0)
for step in range(150):
    player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    whites = game.current_level.get_sprites_by_tag("ysysltqlke")
    blues = game.current_level.get_sprites_by_tag("kdweefinfi")
    collidable = set(game.pkbufziase)
    blocked = set(game.qthdiggudy)
    pc = game.nsevyuople.get(player, None)
    placed = sum(1 for g in goals if (g.x,g.y) in slots and g not in game.zmqreragji)
    a = smart_action(player,goals,whites,blues,collidable,blocked,slots,slots2,slot_aligned)
    fd = env.step(a)
    if step % 10 == 0:
        bs = [(b.x,b.y, '+' if game.nsevyuople.get(b) else '') for b in blues]
        ws = [(w.x,w.y, '+' if game.nsevyuople.get(w) else '') for w in whites]
        print(f"s{step:3d} a={ACTION_NAMES[a]} p=({player.x},{player.y}){'+' if pc else ''} placed={placed} B={bs} W={ws}")
    if fd.levels_completed > 7 or fd.state.name == 'WIN':
        print(f"WIN step={step}")
        break
    if fd.state.name in ('LOSE','GAME_OVER'):
        print(f"End state={fd.state.name} step={step} placed={placed}")
        goals2 = game.current_level.get_sprites_by_tag("geezpjgiyd")
        for g in goals2:
            st_g = 'C' if (g.x,g.y) in slots else ('X' if (g.x,g.y) in slots2 else 'L')
            claimed = game.zmqreragji.get(g)
            ct = ''
            if claimed:
                if 'ysysltqlke' in claimed.tags: ct='W'
                elif 'kdweefinfi' in claimed.tags: ct='B'
                elif 'wbmdvjhthc' in claimed.tags: ct='P'
            print(f"  ({g.x},{g.y}) {st_g}{ct}")
        break

print("--- searching random ---")
best_overall = 0
best_sol = None
for seed in range(2000):
    rnd = 0.01 + 0.04 * ((seed % 20)/20)  # 0.01..0.05
    sol, bp, st = run(all_solutions, seed=seed, randomness=rnd)
    if sol:
        print(f"WIN seed={seed} rnd={rnd:.2f} len={len(sol)}")
        best_sol = sol
        break
    if bp > best_overall:
        best_overall = bp
        print(f"  seed={seed} rnd={rnd:.2f} placed={bp}")

print(f"Best: {best_overall}")
if best_sol:
    print("Solution:", ''.join(ACTION_NAMES[a] for a in best_sol))
import sys; sys.exit(0)

best_overall = 0
best_sol = None
for seed in range(300):
    rnd = 0.02 + 0.35 * (seed/300)
    t1 = time.time()
    sol, bp, st = run(all_solutions, seed=seed, randomness=rnd)
    if sol:
        print(f"WIN seed={seed} rnd={rnd:.2f} len={len(sol)}")
        best_sol = sol
        break
    if bp > best_overall:
        best_overall = bp
        print(f"  seed={seed} rnd={rnd:.2f} placed={bp}  ({time.time()-t1:.1f}s)")

print(f"Best: {best_overall}")
if best_sol:
    print("Solution:", ''.join(ACTION_NAMES[a] for a in best_sol))
