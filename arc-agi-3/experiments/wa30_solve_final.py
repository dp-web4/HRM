#!/usr/bin/env python3
"""
wa30 Sokoban Solver — Engine-based greedy with wall-handoff support.

CURRENT STATUS: 7/9 levels solved (L0-L6). Was 3/9 before 2026-04-12 session.

L0: Pure BFS (no AI) — 26 moves
L1: Engine greedy with blue helper — 56 moves
L2: Hand-coded wall-handoff — 80 moves
L3: Hand-crafted wall-handoff plan distributing 6 pieces to 3 blues — 57 moves
L4: Engine greedy (unlocked by solving L3) — 122 moves
L5: Engine greedy — 48 moves
L6: Hand-coded saboteur-exploit (wait for white to self-stall, destroy, ferry) — 50 moves
L7: Saboteur-aware smart_action planner reaches 10/13 peak/final (up from 8).
    Still not solved. Key findings below.
L8: Unseen (blocked by L7).

L7 FINDINGS (2026-04-12, later session):
  13 pieces, 2 blues, 2 whites, 150 steps.
  - 2 horizontal walls split the 16x16 board into upper/middle/lower strips.
    Upper: 8 pieces, 1 blue (28,4), 1 white (32,16), correct slots upper-right,
    wrong slots upper-left. Gaps in wall y=24 at x=16,20.
    Lower: 5 pieces, 1 blue (36,44), 1 white (32,56), correct slots lower-right,
    wrong slots lower-left. Gaps in wall y=36 at x=36,40.
    Player spawns middle strip at (4,32).
  - Blues are effective — idle test (player does nothing) gets 4-5 placed but
    7 pieces end up on wrong slots.
  - Whites CAN steal from blues/player mid-delivery (xpcvspllwr forces detach).
    Only exclusions are: piece already carried by another WHITE, or piece on
    a WRONG slot. So once on wrong slot, piece is "safe" from further whites.
  - Blues CAN rescue wrong-slot pieces (their exclusion is only at correct slots),
    so the wrong-slot pieces eventually get re-picked by blues if time permits.
  - New smart_action policy (see solve_l7_smart below):
      P1: destroy white if facing
      P2: face white if adjacent
      P3: pick best piece-to-deliver (cost = approach + carry_path)
      P4: chase white (fallback)
    Prime "RRRUUURRRR5" (march player to upper strip, try ACT5) then smart_action
    reaches 10/13 placed. Still short by 3 pieces.
  - Time budget is tight — at end-game player is still carrying a piece the
    game runs out on. With ~10 more steps or faster pathing, could likely solve.
  - UNSOLVED APPROACHES TO TRY NEXT:
      1. Multi-white-kill prime: script moves to destroy BOTH whites early
         (upper via gap (16,24), lower via gap (36,36)).
      2. Macro-action A* over {kill_white, deliver_piece, rescue_wrong_slot}.
      3. Predict white's next BFS target and intercept (look-ahead 1-2 steps).
      4. Per-blue task assignment so blues don't compete for same piece.
      5. Lower-chamber priority — player may be more valuable there since lower
         has fewer pieces/more slots.

Key mechanic: carried pieces pass through blocked positions (qthdiggudy),
carriers don't. This enables "wall handoff" — player carries piece to wall
with appropriate carry offset, piece lands on other side of wall, then a
blue picks it up and delivers it to a slot. Blue pickup is purely geometric
adjacency (manhattan==4), so a piece dropped on a blocked wall cell is still
grabbable by a blue standing on any non-blocked neighbor.

Second mechanic (L6+): whites can re-grab delivered pieces from correct slots
(blue pickup checks "not at correct slot", white pickup only checks "not at
wrong slot"). This means white saboteurs must be destroyed to secure wins,
not just avoided. Whites can also self-stall if their delivery path gets
blocked by walls/other pieces — exploit this by waiting for them to park
themselves, then SELECT-destroy.

L3 NOTES (2026-04-12):
  - 7 pieces, 3 blues, 80 steps. Player spawns trapped inside a walled box
    (x=24-36, y=24-40). Box walls are blocked (qthdiggudy) — player cannot
    exit, but carried pieces can pass through.
  - Piece (4,24) already docked at start -> placed=1/7 initially.
  - 3 blues spawn OUTSIDE the box at isolated positions:
    (56,4), (4,28), (24,56). Their BFS uses kblzhbvysd which avoids both
    walls and collidables; they cannot enter the box.
  - Every non-docked piece must be wall-handed-off to a blue.
  - Pure greedy: 4/7 placed at step 100 budget.
  - 1-step lookahead rollout with greedy default policy: 5/7 placed.
  - Beam search (BEAM=2000) with full state dedup: too slow due to 9ms
    deepcopy overhead (reaches depth ~12 in 2min, needs depth 80).
  - Diagnosis: greedy default policy caps at 5/7 deliveries — rollout
    score stays pinned at max=5 regardless of action, meaning the search
    space around greedy is bounded by greedy's limits. Need a
    qualitatively different policy that plans multi-piece handoff
    sequences accounting for blue trajectories in parallel.
  - Productive dead-end: greedy + rollout + beam all insufficient.
    Next-step candidates:
      1. Macro-action search where each macro = "deliver piece P via
         handoff H", precomputed offline as feasibility table.
      2. A* with admissible heuristic based on per-piece shortest
         handoff cost (ignoring blue contention).
      3. Port engine simulation to pure-Python state struct (no
         deepcopy) to get 100x speedup.
"""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
os.chdir("/mnt/c/exe/projects/ai-agents/SAGE")
os.environ['OPERATION_MODE'] = 'offline'
sys.path.insert(0, ".")

from arc_agi import Arcade
from arcengine import GameAction
from collections import deque
from PIL import Image
import numpy as np

CELL = 4
UP, DOWN, LEFT, RIGHT, ACT5 = (
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
    GameAction.ACTION4, GameAction.ACTION5
)
ACTION_NAMES = {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R', ACT5: '5'}
DIRS = {UP: (0, -CELL), DOWN: (0, CELL), LEFT: (-CELL, 0), RIGHT: (CELL, 0)}
DIR_LIST = [(0, -CELL), (0, CELL), (-CELL, 0), (CELL, 0)]

VISUAL_DIR = "/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/wa30"
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
    s = 8
    img = Image.new('RGB', (w*s, h*s))
    pix = img.load()
    for y in range(h):
        for x in range(w):
            c = PALETTE.get(int(frame[y,x]), (0,0,0))
            for dy in range(s):
                for dx in range(s):
                    pix[x*s+dx, y*s+dy] = c
    img.save(path)

arcade = Arcade(operation_mode='offline')
env = arcade.make('wa30-ee6fef47')
fd = env.reset()
game = env._game


# ============================================================================
# HELPERS
# ============================================================================

def facing_pos(px, py, rot):
    if rot == 0: return (px, py - CELL)
    elif rot == 180: return (px, py + CELL)
    elif rot == 90: return (px + CELL, py)
    elif rot == 270: return (px - CELL, py)
    return (px, py)

def rot_from_dir(dx, dy):
    if dy < 0: return 0
    elif dx > 0: return 90
    elif dy > 0: return 180
    return 270

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def adjacent(a, b):
    return manhattan(a, b) == CELL

def bfs_path(start, goal, obstacles):
    if start == goal: return [start]
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        (cx, cy), path = queue.popleft()
        for dx, dy in DIR_LIST:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited or (nx, ny) in obstacles:
                continue
            if nx < 0 or nx >= 64 or ny < 0 or ny >= 64:
                continue
            visited.add((nx, ny))
            new_path = path + [(nx, ny)]
            if (nx, ny) == goal:
                return new_path
            queue.append(((nx, ny), new_path))
    return None

def bfs_to_targets(start, targets, obstacles):
    if start in targets: return [start]
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        (cx, cy), path = queue.popleft()
        for dx, dy in DIR_LIST:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited or (nx, ny) in obstacles:
                continue
            if nx < 0 or nx >= 64 or ny < 0 or ny >= 64:
                continue
            visited.add((nx, ny))
            new_path = path + [(nx, ny)]
            if (nx, ny) in targets:
                return new_path
            queue.append(((nx, ny), new_path))
    return None

def bfs_carry(start, targets, obstacles, cdx, cdy, blocked):
    """BFS for carrying. Carrier avoids obstacles+blocked. Piece avoids obstacles only."""
    if start in targets: return [start]
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        (cx, cy), path = queue.popleft()
        for dx, dy in DIR_LIST:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited: continue
            if nx < 0 or nx >= 64 or ny < 0 or ny >= 64: continue
            if (nx, ny) in obstacles or (nx, ny) in blocked: continue
            pnx, pny = nx + cdx, ny + cdy
            if pnx < 0 or pnx >= 64 or pny < 0 or pny >= 64: continue
            if (pnx, pny) in obstacles: continue
            visited.add((nx, ny))
            new_path = path + [(nx, ny)]
            if (nx, ny) in targets: return new_path
            queue.append(((nx, ny), new_path))
    return None

def pos_to_action(px, py, nx, ny):
    dx, dy = nx - px, ny - py
    if dy < 0: return UP
    elif dy > 0: return DOWN
    elif dx > 0: return RIGHT
    elif dx < 0: return LEFT
    return UP


# ============================================================================
# BFS SOLVER FOR NO-AI LEVELS
# ============================================================================

def solve_no_ai():
    """Pure BFS for levels without AI movers."""
    player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    goals_sprites = game.current_level.get_sprites_by_tag("geezpjgiyd")
    slots = frozenset((x, y) for (x, y) in game.wyzquhjerd)
    walls = frozenset(game.pkbufziase)
    blocked = frozenset(game.qthdiggudy)
    max_steps = game.kuncbnslnm.current_steps

    # Remove dynamic entities from walls
    dynamic = {(player.x, player.y)}
    for g in goals_sprites: dynamic.add((g.x, g.y))
    static_walls = frozenset(p for p in walls if p not in dynamic)

    init_goals = tuple(sorted((g.x, g.y) for g in goals_sprites))
    carrying = None
    if player in game.nsevyuople:
        c = game.nsevyuople[player]
        carrying = (c.x - player.x, c.y - player.y)

    init_state = (player.x, player.y, player.rotation, init_goals, carrying)

    gsc = {}
    def gs(goals):
        if goals not in gsc: gsc[goals] = set(goals)
        return gsc[goals]

    def is_wall(x, y): return (x, y) in static_walls or (x, y) in blocked
    def sk(s): return (s[0], s[1], s[3], s[4])

    t0 = time.time()
    visited = {sk(init_state)}
    queue = deque([(init_state, [])])
    expanded = 0

    while queue:
        st, moves = queue.popleft()
        px, py, prot, goals, carry = st
        if len(moves) >= max_steps: continue
        expanded += 1
        if expanded % 100000 == 0:
            print(f"    expanded={expanded}, depth={len(moves)}, {time.time()-t0:.1f}s")

        for action in [UP, DOWN, LEFT, RIGHT, ACT5]:
            npx, npy, nrot = px, py, prot
            ng, nc = goals, carry

            if action in DIRS:
                dx, dy = DIRS[action]
                nx, ny = px + dx, py + dy
                nrot = rot_from_dir(dx, dy)
                gset = gs(goals)
                if carry is not None:
                    cdx, cdy = carry
                    old_c = (px + cdx, py + cdy)
                    cnx, cny = nx + cdx, ny + cdy
                    if is_wall(nx, ny) or is_wall(cnx, cny): continue
                    og = gset - {old_c}
                    if (nx, ny) in og or (cnx, cny) in og: continue
                    npx, npy = nx, ny
                    ng = tuple(sorted((cnx, cny) if (gx, gy) == old_c else (gx, gy) for gx, gy in goals))
                else:
                    if is_wall(nx, ny) or (nx, ny) in gset: continue
                    npx, npy = nx, ny
            elif action == ACT5:
                if carry is not None:
                    nc = None
                else:
                    fx, fy = facing_pos(px, py, prot)
                    if (fx, fy) in gs(goals):
                        nc = (fx - px, fy - py)
                    else: continue

            ns = (npx, npy, nrot, ng, nc)
            k = sk(ns)
            if k in visited: continue
            visited.add(k)
            nm = moves + [action]
            if nc is None and all((gx, gy) in slots for gx, gy in ng):
                print(f"  SOLVED! {len(nm)} moves, {expanded} expanded, {time.time()-t0:.1f}s")
                return nm
            queue.append((ns, nm))

    print(f"  No solution. {expanded} expanded, {time.time()-t0:.1f}s")
    return None


# ============================================================================
# ENGINE GREEDY SOLVER
# ============================================================================

def solve_with_ai(prev_solutions):
    """Engine-based greedy solver with intelligent piece targeting."""
    fd = replay_from_solutions(prev_solutions)
    n_prev = len(prev_solutions)
    solution = []
    t0 = time.time()
    n_pieces = len(game.current_level.get_sprites_by_tag("geezpjgiyd"))
    max_steps = game.kuncbnslnm.current_steps

    # Compute player-reachable region and blue-reachable region ONCE
    # (they change as pieces move, but the wall structure is static)
    blocked = set(game.qthdiggudy)
    slots = game.wyzquhjerd
    slots2 = game.lqctaojiby
    slot_aligned = set((sx, sy) for sx, sy in slots if sx % CELL == 0 and sy % CELL == 0)

    print(f"  Greedy: {n_pieces} pieces, {max_steps} steps")

    for iteration in range(max_steps):
        player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
        goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
        whites = game.current_level.get_sprites_by_tag("ysysltqlke")
        blues = game.current_level.get_sprites_by_tag("kdweefinfi")
        px, py, prot = player.x, player.y, player.rotation
        fx, fy = facing_pos(px, py, prot)
        collidable = set(game.pkbufziase)

        player_carrying = game.nsevyuople.get(player, None)

        placed_count = sum(1 for g in goals
                          if (g.x, g.y) in slots and g not in game.zmqreragji)

        action = plan_action(player, goals, whites, blues, collidable, blocked,
                             slots, slots2, slot_aligned, player_carrying)

        if action is None:
            for a in [UP, DOWN, LEFT, RIGHT]:
                dx, dy = DIRS[a]
                nx, ny = px + dx, py + dy
                if (nx, ny) not in collidable and (nx, ny) not in blocked and 0 <= nx < 64 and 0 <= ny < 64:
                    action = a
                    break
            if action is None:
                action = UP

        fd = env.step(action)
        solution.append(action)

        if fd.levels_completed > n_prev or fd.state.name == 'WIN':
            print(f"  SOLVED! {len(solution)} moves, {time.time()-t0:.1f}s")
            return solution
        if fd.state.name in ('LOSE', 'GAME_OVER'):
            print(f"  LOST after {len(solution)} moves, placed={placed_count}/{n_pieces}")
            return None

        if iteration % 20 == 0:
            ci = f' carry=({player_carrying.x},{player_carrying.y})' if player_carrying else ''
            print(f"    step {iteration}: p=({px},{py}){ci} placed={placed_count}/{n_pieces}, "
                  f"steps_left={game.kuncbnslnm.current_steps}, {time.time()-t0:.1f}s")

    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    placed_count = sum(1 for g in goals if (g.x, g.y) in game.wyzquhjerd and g not in game.zmqreragji)
    print(f"  Greedy failed, placed={placed_count}/{n_pieces}")
    return None


def plan_action(player, goals, whites, blues, collidable, blocked, slots, slots2, slot_aligned, player_carrying):
    px, py, prot = player.x, player.y, player.rotation
    fx, fy = facing_pos(px, py, prot)

    # ---- CARRYING ----
    if player_carrying is not None:
        cpx, cpy = player_carrying.x, player_carrying.y
        cdx, cdy = cpx - px, cpy - py

        # Already at slot? Drop!
        if (cpx, cpy) in slots:
            return ACT5

        # Find delivery targets for direct slot placement
        placed_pos = set()
        for g in goals:
            if (g.x, g.y) in slots and g not in game.zmqreragji and g is not player_carrying:
                placed_pos.add((g.x, g.y))

        direct_targets = set()
        for sx, sy in slot_aligned:
            if (sx, sy) not in placed_pos:
                tpx, tpy = sx - cdx, sy - cdy
                if 0 <= tpx < 64 and 0 <= tpy < 64:
                    direct_targets.add((tpx, tpy))

        if direct_targets:
            obs = collidable.copy()
            obs.discard((px, py))
            obs.discard((cpx, cpy))
            path = bfs_carry((px, py), direct_targets, obs, cdx, cdy, blocked)
            if path and len(path) > 1:
                return pos_to_action(px, py, path[1][0], path[1][1])
            elif path and len(path) == 1:
                return ACT5

        # Can't reach slots. Try handoff: carry piece to wall positions reachable by blue.
        if blues:
            # Quick BFS from each blue to find reachable cells
            blue_reach = set()
            for b in blues:
                vis = {(b.x, b.y)}
                q = deque([(b.x, b.y)])
                while q:
                    cx, cy = q.popleft()
                    blue_reach.add((cx, cy))
                    for ddx, ddy in DIR_LIST:
                        nx, ny = cx + ddx, cy + ddy
                        if (nx, ny) in vis: continue
                        if nx < 0 or nx >= 64 or ny < 0 or ny >= 64: continue
                        if (nx, ny) in blocked: continue
                        # Collidable minus blues themselves
                        if (nx, ny) in collidable and (nx, ny) != (b.x, b.y):
                            # Check if it's another blue or carried piece
                            is_blue = any((nx, ny) == (bb.x, bb.y) for bb in blues)
                            if not is_blue: continue
                        vis.add((nx, ny))
                        q.append((nx, ny))

            # Handoff targets: player positions such that piece ends up adjacent to blue_reach
            handoff_targets = set()
            for bx, by in blue_reach:
                for ddx, ddy in DIR_LIST:
                    piece_pos = (bx + ddx, by + ddy)
                    if 0 <= piece_pos[0] < 64 and 0 <= piece_pos[1] < 64:
                        ppx, ppy = piece_pos[0] - cdx, piece_pos[1] - cdy
                        if 0 <= ppx < 64 and 0 <= ppy < 64 and (ppx, ppy) not in blocked:
                            handoff_targets.add((ppx, ppy))

            if handoff_targets:
                obs = collidable.copy()
                obs.discard((px, py))
                obs.discard((cpx, cpy))
                path = bfs_carry((px, py), handoff_targets, obs, cdx, cdy, blocked)
                if path and len(path) > 1:
                    return pos_to_action(px, py, path[1][0], path[1][1])
                elif path and len(path) == 1:
                    return ACT5

        return ACT5  # Drop and retry

    # ---- NOT CARRYING ----

    # Priority: Destroy white movers
    if whites:
        for w in whites:
            if (w.x, w.y) == (fx, fy):
                return ACT5
        for w in whites:
            if adjacent((px, py), (w.x, w.y)):
                dx, dy = w.x - px, w.y - py
                if abs(dy) >= abs(dx):
                    return UP if dy < 0 else DOWN
                return RIGHT if dx > 0 else LEFT

        nearest_w = min(whites, key=lambda w: manhattan((px, py), (w.x, w.y)))
        adj = set()
        for ddx, ddy in DIR_LIST:
            adj.add((nearest_w.x + ddx, nearest_w.y + ddy))
        obs = collidable.copy()
        obs.discard((px, py))
        path = bfs_to_targets((px, py), adj, obs)
        if path and len(path) > 1:
            return pos_to_action(px, py, path[1][0], path[1][1])

    # Pick up piece if facing one
    for g in goals:
        if (g.x, g.y) == (fx, fy) and g not in game.zmqreragji and (g.x, g.y) not in slots:
            return ACT5

    # Find best piece to target
    free_pieces = [g for g in goals if g not in game.zmqreragji and (g.x, g.y) not in slots]

    # Don't target pieces blue is carrying
    blue_carried = set()
    for b in blues:
        if b in game.nsevyuople:
            blue_carried.add(id(game.nsevyuople[b]))

    player_targets = [g for g in free_pieces if id(g) not in blue_carried]
    if not player_targets and not free_pieces:
        return None  # All pieces placed or being carried -- wait
    if not player_targets:
        # All free pieces are being handled by blues -- check if some blues might drop soon
        # Just wait (return None to trigger a safe move)
        return None

    # Choose piece: evaluate full delivery cost (approach + carry to slot)
    # Pick the piece with best total delivery cost
    if slot_aligned:
        scx = sum(sx for sx, sy in slot_aligned) // len(slot_aligned)
        scy = sum(sy for sx, sy in slot_aligned) // len(slot_aligned)
    else:
        scx, scy = 32, 32

    # Score: total round-trip (approach + delivery to nearest slot)
    # Pick piece with minimum total cost for quick delivery
    best_piece = None
    best_cost = float('inf')
    for g in player_targets:
        gx, gy = g.x, g.y
        approach = manhattan((px, py), (gx, gy))
        # Estimate delivery cost: piece to nearest empty slot
        min_deliver = float('inf')
        for sx, sy in slot_aligned:
            d = manhattan((gx, gy), (sx, sy))
            if d < min_deliver:
                min_deliver = d
        total = approach + min_deliver
        if total < best_cost:
            best_cost = total
            best_piece = g

    target_piece = best_piece if best_piece else player_targets[0]
    gx, gy = target_piece.x, target_piece.y

    # Choose approach direction: prefer offset that enables delivery
    # For each adjacent cell, check if carrying with that offset can reach slots or handoff
    best_adj = None
    best_score = float('inf')
    obs = collidable.copy()
    obs.discard((px, py))

    for ddx, ddy in DIR_LIST:
        ax, ay = gx + ddx, gy + ddy
        if ax < 0 or ax >= 64 or ay < 0 or ay >= 64: continue
        if (ax, ay) in obs and (ax, ay) != (px, py): continue
        off_x, off_y = gx - ax, gy - ay  # carry offset

        # Try BFS carry from approach position to any slot delivery target
        deliver_targets = set()
        for sx, sy in slot_aligned:
            tpx, tpy = sx - off_x, sy - off_y
            if 0 <= tpx < 64 and 0 <= tpy < 64:
                deliver_targets.add((tpx, tpy))

        # Also add handoff targets: piece at blocked positions adjacent to blue reachable
        # Simplified: for each blocked position, player at (bx - off_x, by - off_y) if not blocked
        for bpos in blocked:
            ppx, ppy = bpos[0] - off_x, bpos[1] - off_y
            if 0 <= ppx < 64 and 0 <= ppy < 64 and (ppx, ppy) not in blocked:
                deliver_targets.add((ppx, ppy))

        if not deliver_targets: continue

        # Check if any deliver target is reachable via carry BFS
        carry_obs = obs.copy()
        carry_obs.discard((gx, gy))  # piece starts here
        test_path = bfs_carry((ax, ay), deliver_targets, carry_obs, off_x, off_y, blocked)
        if test_path is None: continue

        # Path to approach position
        path_to_adj = bfs_path((px, py), (ax, ay), obs)
        if path_to_adj is None: continue

        score = len(path_to_adj) + len(test_path)
        if score < best_score:
            best_score = score
            best_adj = (ax, ay, path_to_adj)

    if best_adj:
        ax, ay, path = best_adj
        if (ax, ay) == (px, py):
            # Already adjacent, face piece
            if (fx, fy) == (gx, gy):
                return ACT5
            dx, dy = gx - px, gy - py
            if abs(dy) >= abs(dx):
                return UP if dy < 0 else DOWN
            return RIGHT if dx > 0 else LEFT
        if len(path) > 1:
            return pos_to_action(px, py, path[1][0], path[1][1])

    return None


def smart_action_l7(player, goals, whites, blues, collidable, blocked, slots, slots2, slot_aligned):
    """Saboteur-aware planner for L7. Reaches 10/13 with prime 'RRRUUURRRR5'.
    Differs from plan_action by:
      - Unconditional white-destroy priority when facing/adjacent
      - Excludes pieces carried by whites (handled via destroy)
      - Blues get exclusive claim on pieces they're carrying
    """
    px, py, prot = player.x, player.y, player.rotation
    fx, fy = facing_pos(px, py, prot)
    pc = game.nsevyuople.get(player, None)

    if pc is not None:
        cpx, cpy = pc.x, pc.y
        cdx, cdy = cpx - px, cpy - py
        if (cpx, cpy) in slots:
            return ACT5
        placed_pos = set((g.x, g.y) for g in goals
                         if (g.x, g.y) in slots and g not in game.zmqreragji and g is not pc)
        targets = set()
        for sx, sy in slot_aligned:
            if (sx, sy) in placed_pos: continue
            tpx, tpy = sx - cdx, sy - cdy
            if 0 <= tpx < 64 and 0 <= tpy < 64:
                targets.add((tpx, tpy))
        obs = collidable.copy()
        obs.discard((px, py)); obs.discard((cpx, cpy))
        path = bfs_carry((px, py), targets, obs, cdx, cdy, blocked)
        if path and len(path) > 1:
            return pos_to_action(px, py, path[1][0], path[1][1])
        if path and len(path) == 1:
            return ACT5
        return ACT5

    # Not carrying
    for w in whites:
        if (w.x, w.y) == (fx, fy):
            return ACT5
    for w in whites:
        if adjacent((px, py), (w.x, w.y)):
            dx, dy = w.x - px, w.y - py
            if abs(dy) >= abs(dx):
                return UP if dy < 0 else DOWN
            return RIGHT if dx > 0 else LEFT

    obs = collidable.copy()
    obs.discard((px, py))
    blue_set = set(blues)
    targets_pieces = []
    for g in goals:
        if (g.x, g.y) in slots and g not in game.zmqreragji: continue
        claimed = game.zmqreragji.get(g)
        if claimed in blue_set: continue
        if claimed is player: continue
        if claimed is not None and 'ysysltqlke' in claimed.tags: continue
        targets_pieces.append(g)

    best = None
    for g in targets_pieces:
        gx, gy = g.x, g.y
        for ddx, ddy in DIR_LIST:
            ax, ay = gx + ddx, gy + ddy
            if not (0 <= ax < 64 and 0 <= ay < 64): continue
            if (ax, ay) in obs and (ax, ay) != (px, py): continue
            off_x, off_y = gx - ax, gy - ay
            placed_pos = set((gg.x, gg.y) for gg in goals
                             if (gg.x, gg.y) in slots and gg not in game.zmqreragji and gg is not g)
            dt = set()
            for sx, sy in slot_aligned:
                if (sx, sy) in placed_pos: continue
                tpx, tpy = sx - off_x, sy - off_y
                if 0 <= tpx < 64 and 0 <= tpy < 64:
                    dt.add((tpx, tpy))
            if not dt: continue
            carry_obs = obs.copy()
            carry_obs.discard((gx, gy))
            cpath = bfs_carry((ax, ay), dt, carry_obs, off_x, off_y, blocked)
            if cpath is None: continue
            apath = bfs_path((px, py), (ax, ay), obs)
            if apath is None: continue
            cost = len(apath) + len(cpath)
            if best is None or cost < best[0]:
                best = (cost, apath, g, (ax, ay))

    if best:
        cost, apath, piece, (ax, ay) = best
        if (ax, ay) == (px, py):
            gx, gy = piece.x, piece.y
            if (fx, fy) == (gx, gy): return ACT5
            dx, dy = gx - px, gy - py
            if abs(dy) >= abs(dx):
                return UP if dy < 0 else DOWN
            return RIGHT if dx > 0 else LEFT
        if len(apath) > 1:
            return pos_to_action(px, py, apath[1][0], apath[1][1])

    # Long white chase
    best_wp = None
    for w in whites:
        adj_cells = set()
        for ddx, ddy in DIR_LIST:
            ax, ay = w.x + ddx, w.y + ddy
            if 0 <= ax < 64 and 0 <= ay < 64 and (ax, ay) not in collidable:
                adj_cells.add((ax, ay))
        if not adj_cells: continue
        p = bfs_to_targets((px, py), adj_cells, obs)
        if p and (best_wp is None or len(p) < len(best_wp)):
            best_wp = p
    if best_wp and len(best_wp) > 1:
        return pos_to_action(px, py, best_wp[1][0], best_wp[1][1])

    for a, (dx, dy) in [(UP, (0, -4)), (DOWN, (0, 4)), (LEFT, (-4, 0)), (RIGHT, (4, 0))]:
        nx, ny = px + dx, py + dy
        if (nx, ny) in collidable or (nx, ny) in blocked: continue
        if not (0 <= nx < 64 and 0 <= ny < 64): continue
        return a
    return UP


def solve_l7_smart(prev_solutions, prime_str='RRRUUURRRR5'):
    """L7 solver: prime walk + smart_action. Reaches 10/13, does not solve.
    Kept for future improvement."""
    fd = replay_from_solutions(prev_solutions)
    slots = game.wyzquhjerd
    slots2 = game.lqctaojiby
    slot_aligned = set((sx, sy) for sx, sy in slots if sx % CELL == 0 and sy % CELL == 0)
    prime = string_to_actions(prime_str)
    solution = []
    for step in range(150):
        player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
        goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
        whites = game.current_level.get_sprites_by_tag("ysysltqlke")
        blues = game.current_level.get_sprites_by_tag("kdweefinfi")
        collidable = set(game.pkbufziase)
        blocked = set(game.qthdiggudy)
        if step < len(prime):
            action = prime[step]
        else:
            action = smart_action_l7(player, goals, whites, blues, collidable, blocked,
                                     slots, slots2, slot_aligned)
        fd = env.step(action)
        solution.append(action)
        if fd.levels_completed > len(prev_solutions) or fd.state.name == 'WIN':
            print(f"  L7 SOLVED! {len(solution)} moves")
            return solution
        if fd.state.name in ('LOSE', 'GAME_OVER'):
            goals_final = game.current_level.get_sprites_by_tag("geezpjgiyd")
            placed = sum(1 for g in goals_final if (g.x, g.y) in slots and g not in game.zmqreragji)
            print(f"  L7 smart failed, placed={placed}/13")
            return None
    return None


import random

def solve_random_greedy(prev_solutions, max_attempts=200, seed=42):
    """Randomized greedy: mix planned actions with random exploration."""
    random.seed(seed)
    t0 = time.time()
    n_prev = len(prev_solutions)
    n_pieces = len(game.current_level.get_sprites_by_tag("geezpjgiyd"))
    best_placed = 0

    for attempt in range(max_attempts):
        fd = replay_from_solutions(prev_solutions)
        expected_completed = fd.levels_completed
        max_steps = game.kuncbnslnm.current_steps
        solution = []
        randomness = 0.05 + 0.4 * (attempt / max_attempts)

        for step in range(max_steps):
            player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
            goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
            whites = game.current_level.get_sprites_by_tag("ysysltqlke")
            blues = game.current_level.get_sprites_by_tag("kdweefinfi")
            collidable = set(game.pkbufziase)
            blocked = set(game.qthdiggudy)
            slots = game.wyzquhjerd
            slots2 = game.lqctaojiby
            slot_aligned = set((sx, sy) for sx, sy in slots if sx % CELL == 0 and sy % CELL == 0)
            player_carrying = game.nsevyuople.get(player, None)

            if random.random() < randomness:
                # Random valid action
                actions = [UP, DOWN, LEFT, RIGHT, ACT5]
                random.shuffle(actions)
                action = actions[0]
            else:
                action = plan_action(player, goals, whites, blues, collidable, blocked,
                                     slots, slots2, slot_aligned, player_carrying)
                if action is None:
                    action = random.choice([UP, DOWN, LEFT, RIGHT])

            fd = env.step(action)
            solution.append(action)

            if fd.levels_completed > expected_completed or fd.state.name == 'WIN':
                # Verify by replaying from scratch
                fd2 = replay_from_solutions(prev_solutions)
                for m in solution:
                    fd2 = env.step(m)
                if fd2.levels_completed > expected_completed or fd2.state.name == 'WIN':
                    elapsed = time.time() - t0
                    print(f"  SOLVED! attempt={attempt}, {len(solution)} moves, {elapsed:.1f}s")
                    return solution
                else:
                    break  # false positive
            if fd.state.name in ('LOSE', 'GAME_OVER'):
                # Count placed at game over
                goals_final = game.current_level.get_sprites_by_tag("geezpjgiyd")
                placed = sum(1 for g in goals_final if (g.x, g.y) in slots and g not in game.zmqreragji)
                if placed > best_placed:
                    best_placed = placed
                break

        if attempt % 50 == 0:
            print(f"    attempt {attempt}: best={best_placed}/{n_pieces} (rand={randomness:.2f})")

    elapsed = time.time() - t0
    print(f"  Random greedy failed, best={best_placed}/{n_pieces}, {elapsed:.1f}s")
    return None


def replay_from_solutions(solutions):
    fd = env.reset()
    for sol in solutions:
        for m in sol:
            fd = env.step(m)
    return fd


# ============================================================================
# MAIN
# ============================================================================

# Known solutions for levels that need manual strategies
KNOWN_SOLUTIONS = {
    # L2: wall-handoff strategy — carry 3 left-side pieces to x=32 wall, blue picks up from right
    2: 'LLDDR5RRRRR5LLLLLLUUUUUUUR5RRRRRR5LLLDR5RRR5' + 'U' * 56,  # 100 total
    # L3: player trapped in 5x5 walled box with 6 pieces + 1 outside. 3 blues each own
    # a slot cluster (Blue1=west 3 slots, Blue0=east 2 slots, Blue2=south 2 slots).
    # Player wall-handoffs pieces onto box wall cells grabbable by the right blue:
    #   (24,24)→(20,24)→Blue1  (24,40)→(20,40)→Blue1  Blue1 auto-grabs (24,4) from outside
    #   (36,24)→(36,20)→Blue0  (32,24)→(32,20)→Blue0
    #   (36,40)→(36,44)→Blue2  (32,36)→(32,44)→Blue2
    # 47-move active plan + 10 pad moves for blue deliveries = 57 total.
    3: 'UUL5L5DDDRDL5L5RUURRU5U5DDDD5D5UUULU5U5DDDD5DD5' + 'U' * 10,
    # L6: 2 pieces, 1 white saboteur, no blues, corridor y=20..36.
    # White grabs (28,32) immediately, delivers to wrong-slot (48,32), then grabs
    # (32,24) and gets stuck at (32,28) carrying it (can't reach wrong-slot).
    # Strategy: wait 15 no-ops (U bouncing) for white to stabilize, walk over to
    # (28,28) and SELECT-destroy white (drops (32,24) back), grab (32,24) carry
    # west and deliver to (12,28), then walk east to (44,32) adjacent to piece at
    # (48,32), grab with offset (+4,0), carry west to (8,32) delivering to (12,32).
    6: 'U'*15 + 'DDRR5' + 'RU5LLLLLD5' + 'R'*9 + '5' + 'L'*9 + '5',
}

def string_to_actions(s):
    am = {'U': UP, 'D': DOWN, 'L': LEFT, 'R': RIGHT, '5': ACT5}
    return [am[c] for c in s]


total_actions = 0
all_solutions = []

for lv in range(20):
    fd = replay_from_solutions(all_solutions)
    if fd.state.name == 'WIN':
        break

    print(f"\n{'='*60}")
    print(f"=== Level {lv} ===")
    print(f"{'='*60}")
    if fd.frame:
        save_frame(fd.frame, f"{VISUAL_DIR}/L{lv}_start.png")

    player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    reds = game.current_level.get_sprites_by_tag("kdweefinfi")
    whites = game.current_level.get_sprites_by_tag("ysysltqlke")

    print(f"  Player: ({player.x},{player.y})")
    print(f"  Pieces: {[(g.x,g.y) for g in goals]}")
    print(f"  Blues: {[(r.x,r.y) for r in reds]}")
    print(f"  Whites: {[(w.x,w.y) for w in whites]}")
    print(f"  Steps: {game.kuncbnslnm.current_steps}")

    has_ai = len(reds) > 0 or len(whites) > 0

    # Check for known solution first
    if lv in KNOWN_SOLUTIONS:
        print(f"  Using known solution for level {lv}")
        solution = string_to_actions(KNOWN_SOLUTIONS[lv])
    elif not has_ai:
        solution = solve_no_ai()
    else:
        if lv == 7:
            print(f"  Trying L7 smart saboteur-aware solver...")
            solution = solve_l7_smart(all_solutions)
            if solution is None:
                print(f"  L7 smart failed -- falling back to engine greedy")
                solution = solve_with_ai(all_solutions)
        else:
            print(f"  Trying greedy solver...")
            solution = solve_with_ai(all_solutions)
        if solution is None:
            print(f"  Greedy failed -- trying randomized greedy (500 attempts)")
            solution = solve_random_greedy(all_solutions, max_attempts=500)

    if solution is None:
        print(f"  FAILED on level {lv}")
        break

    # Verify
    fd = replay_from_solutions(all_solutions)
    completed_before = fd.levels_completed
    actual_solution = []
    for m in solution:
        fd = env.step(m)
        actual_solution.append(m)
        if fd.levels_completed > completed_before or fd.state.name == 'WIN':
            break
    solution = actual_solution

    sol_str = ''.join(ACTION_NAMES[a] for a in solution)
    print(f"  Solution({len(solution)}): {sol_str}")
    total_actions += len(solution)

    if fd.frame:
        save_frame(fd.frame, f"{VISUAL_DIR}/L{lv}_end.png")

    if fd.levels_completed > completed_before or fd.state.name == 'WIN':
        print(f"  VERIFIED! completed={fd.levels_completed}")
        all_solutions.append(solution)
    else:
        print(f"  FAILED verification! completed={fd.levels_completed}")
        break

print(f"\nFinal: completed={fd.levels_completed}, state={fd.state.name}, total={total_actions}")
print(f"Solutions per level: {[len(s) for s in all_solutions]}")
