#!/usr/bin/env python3
"""
wa30 World-Model Solver v2 — Hybrid BFS + greedy planning.

Levels without AI: pure BFS simulation (fast, optimal)
Levels with AI: greedy planning using engine state (pick up piece -> deliver to slot)

Key improvement: For AI levels, instead of expensive beam search that replays
from scratch, we use a greedy forward planner that works with the engine in
real-time — no replay needed.
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

VISUAL_DIR = "shared-context/arc-agi-3/visual-memory/wa30"
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


def extract_level_data():
    """Extract state from current game level."""
    player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    reds = game.current_level.get_sprites_by_tag("kdweefinfi")
    whites = game.current_level.get_sprites_by_tag("ysysltqlke")

    px, py, prot = player.x, player.y, player.rotation
    goal_positions = tuple(sorted((g.x, g.y) for g in goals))
    red_positions = tuple(sorted((r.x, r.y) for r in reds))
    white_positions = tuple(sorted((w.x, w.y) for w in whites))

    dynamic_pos = {(px, py)}
    for g in goals: dynamic_pos.add((g.x, g.y))
    for r in reds: dynamic_pos.add((r.x, r.y))
    for w in whites: dynamic_pos.add((w.x, w.y))
    walls = frozenset(p for p in game.pkbufziase if p not in dynamic_pos)
    blocked = frozenset(game.qthdiggudy)
    slots = frozenset((x, y) for (x, y) in game.wyzquhjerd)
    slots2 = frozenset((x, y) for (x, y) in game.lqctaojiby)

    carrying = None
    if player in game.nsevyuople:
        carried = game.nsevyuople[player]
        carrying = (carried.x - player.x, carried.y - player.y)

    steps = game.kuncbnslnm.current_steps
    return {
        'walls': walls, 'blocked': blocked, 'slots': slots, 'slots2': slots2,
        'px': px, 'py': py, 'prot': prot,
        'goals': goal_positions, 'reds': red_positions, 'whites': white_positions,
        'carrying': carrying, 'steps': steps,
    }


def simulate_fast(level_data):
    """Pure BFS for levels without AI entities."""
    walls = level_data['walls']
    blocked = level_data['blocked']
    slots = level_data['slots']

    init_goals = level_data['goals']
    init_state = (level_data['px'], level_data['py'], level_data['prot'],
                  init_goals, level_data['carrying'])

    goal_set_cache = {}
    def goals_as_set(goals):
        if goals not in goal_set_cache:
            goal_set_cache[goals] = set(goals)
        return goal_set_cache[goals]

    def is_wall(x, y):
        return (x, y) in walls or (x, y) in blocked

    def at_slot(x, y):
        return (x, y) in slots

    def state_key(s):
        return (s[0], s[1], s[3], s[4])

    def is_won(goals):
        return all(at_slot(gx, gy) for gx, gy in goals)

    def facing(px, py, rot):
        if rot == 0: return (px, py - CELL)
        elif rot == 180: return (px, py + CELL)
        elif rot == 90: return (px + CELL, py)
        elif rot == 270: return (px - CELL, py)
        return (px, py)

    def rot_from_dir(dx, dy):
        if dy < 0: return 0
        if dy > 0: return 180
        if dx > 0: return 90
        if dx < 0: return 270
        return 0

    t0 = time.time()
    visited = {state_key(init_state)}
    queue = deque([(init_state, [])])
    expanded = 0

    while queue:
        state, moves = queue.popleft()
        px, py, prot, goals, carrying = state
        depth = len(moves)

        if depth >= level_data['steps']:
            continue

        expanded += 1
        if expanded % 100000 == 0:
            elapsed = time.time() - t0
            print(f"    expanded={expanded}, visited={len(visited)}, depth={depth}, {elapsed:.1f}s")

        for action in [UP, DOWN, LEFT, RIGHT, ACT5]:
            new_px, new_py, new_rot = px, py, prot
            new_goals = goals
            new_carrying = carrying

            if action in DIRS:
                dx, dy = DIRS[action]
                nx, ny = px + dx, py + dy
                new_rot = rot_from_dir(dx, dy)
                gset = goals_as_set(goals)

                if carrying is not None:
                    cdx, cdy = carrying
                    old_carried = (px + cdx, py + cdy)
                    cnx, cny = nx + cdx, ny + cdy
                    if is_wall(nx, ny) or is_wall(cnx, cny):
                        continue
                    other_goals = gset - {old_carried}
                    if (nx, ny) in other_goals or (cnx, cny) in other_goals:
                        continue
                    new_px, new_py = nx, ny
                    new_goals = tuple(sorted(
                        (cnx, cny) if (gx, gy) == old_carried else (gx, gy)
                        for gx, gy in goals
                    ))
                else:
                    if is_wall(nx, ny) or (nx, ny) in gset:
                        continue
                    new_px, new_py = nx, ny

            elif action == ACT5:
                if carrying is not None:
                    new_carrying = None
                else:
                    fx, fy = facing(px, py, prot)
                    if (fx, fy) in goals:
                        new_carrying = (fx - px, fy - py)
                    else:
                        continue

            new_state = (new_px, new_py, new_rot, new_goals, new_carrying)
            sk = state_key(new_state)
            if sk in visited:
                continue
            visited.add(sk)

            new_moves = moves + [action]

            if new_carrying is None and is_won(new_goals):
                elapsed = time.time() - t0
                print(f"  SOLVED! {len(new_moves)} moves, {expanded} expanded, {elapsed:.1f}s")
                return new_moves

            queue.append((new_state, new_moves))

    elapsed = time.time() - t0
    print(f"  No solution. {expanded} expanded, {len(visited)} visited, {elapsed:.1f}s")
    return None


def bfs_path(start, goal, obstacles):
    """Simple BFS from start to goal avoiding obstacles. Returns path."""
    if start == goal:
        return [start]
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


def path_to_actions(path):
    """Convert path to action sequence."""
    actions = []
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        if dy < 0: actions.append(UP)
        elif dy > 0: actions.append(DOWN)
        elif dx < 0: actions.append(LEFT)
        elif dx > 0: actions.append(RIGHT)
    return actions


def rot_from_dir(dx, dy):
    if dy < 0: return 0
    if dy > 0: return 180
    if dx > 0: return 90
    if dx < 0: return 270
    return 0


def facing_pos(px, py, rot):
    if rot == 0: return (px, py - CELL)
    elif rot == 180: return (px, py + CELL)
    elif rot == 90: return (px + CELL, py)
    elif rot == 270: return (px - CELL, py)
    return (px, py)


def action_to_face(px, py, tx, ty):
    """Get action to face from (px,py) toward (tx,ty)."""
    dx, dy = tx - px, ty - py
    if dy < 0: return UP
    elif dy > 0: return DOWN
    elif dx > 0: return RIGHT
    elif dx < 0: return LEFT
    return UP


def greedy_solve_with_ai(prev_solutions):
    """Greedy forward planner for levels with AI entities.

    Strategy:
    1. If carrying a piece, find nearest slot and deliver
    2. If not carrying, find nearest free piece and pick it up
    3. If white movers threaten, destroy them first
    4. Execute one step at a time, re-plan after each step
    """
    # Replay to current level start
    fd = replay_from_solutions(prev_solutions)
    n_prev = len(prev_solutions)

    ld = extract_level_data()
    max_steps = ld['steps']
    solution = []

    for iteration in range(max_steps):
        ld = extract_level_data()
        px, py = ld['px'], ld['py']
        prot = ld['prot']
        carrying = ld['carrying']

        # Check win
        goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
        slots = game.wyzquhjerd
        all_placed = all((g.x, g.y) in slots and g not in game.zmqreragji for g in goals)
        if all_placed:
            print(f"  All goals placed after {len(solution)} actions")
            break

        # Build obstacle set
        obstacles = set(ld['walls']) | set(ld['blocked'])
        # Add all collidable positions except player
        for p in game.pkbufziase:
            obstacles.add(p)
        obstacles.discard((px, py))
        if carrying:
            obstacles.discard((px + carrying[0], py + carrying[1]))

        # Get current unplaced pieces (not in slot OR linked to carrier)
        unplaced = []
        for g in goals:
            in_slot = (g.x, g.y) in slots
            linked = g in game.zmqreragji
            if not in_slot or linked:
                unplaced.append(g)

        if not unplaced:
            break

        # Plan next action
        action = plan_next_action(px, py, prot, carrying, unplaced, obstacles, ld)

        if action is None:
            # Fallback: try each direction
            for a in [UP, DOWN, LEFT, RIGHT]:
                dx, dy = DIRS[a]
                nx, ny = px + dx, py + dy
                if (nx, ny) not in obstacles and 0 <= nx < 64 and 0 <= ny < 64:
                    action = a
                    break
            if action is None:
                action = ACT5  # last resort

        fd = env.step(action)
        solution.append(action)

        if fd.levels_completed > n_prev or fd.state.name == 'WIN':
            print(f"  SOLVED! {len(solution)} moves")
            return solution

        if fd.state.name == 'LOSE':
            print(f"  LOST after {len(solution)} moves")
            return None

    # May not be won yet — check
    if fd.levels_completed > n_prev or fd.state.name == 'WIN':
        print(f"  SOLVED! {len(solution)} moves")
        return solution

    print(f"  Greedy failed after {len(solution)} moves")
    return None


def plan_next_action(px, py, prot, carrying, unplaced, obstacles, ld):
    """Plan the next single action."""

    if carrying:
        # Delivering a piece to slot
        # Find target slot positions
        slot_cells = set(s for s in ld['slots'] if s[0] % CELL == 0 and s[1] % CELL == 0)

        # Remove slots already occupied by placed pieces
        goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
        for g in goals:
            if (g.x, g.y) in slot_cells and (g.x, g.y) in ld['slots'] and g not in game.zmqreragji:
                carried_sprite = game.nsevyuople.get(
                    game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
                )
                if g is not carried_sprite:
                    slot_cells.discard((g.x, g.y))

        cdx, cdy = carrying
        carried_pos = (px + cdx, py + cdy)

        # Find closest reachable slot
        best_path = None
        best_len = float('inf')
        for sx, sy in slot_cells:
            target_px = sx - cdx
            target_py = sy - cdy
            if (target_px, target_py) in obstacles and (target_px, target_py) != (px, py):
                continue
            obs_copy = obstacles.copy()
            obs_copy.discard(carried_pos)
            obs_copy.discard((target_px, target_py))
            path = bfs_path((px, py), (target_px, target_py), obs_copy)
            if path and len(path) < best_len:
                best_path = path
                best_len = len(path)

        if best_path and len(best_path) > 1:
            return path_to_actions(best_path[:2])[0]
        elif best_path and len(best_path) == 1:
            return ACT5  # Drop
        else:
            return ACT5  # Can't reach slot, drop and retry

    # Not carrying — pick up a piece
    # First priority: destroy white movers that might steal pieces
    whites = game.current_level.get_sprites_by_tag("ysysltqlke")
    for w in whites:
        # Check if adjacent and facing
        for dx, dy in DIR_LIST:
            if px + dx == w.x and py + dy == w.y:
                # Adjacent! Face and interact
                face_action = action_to_face(px, py, w.x, w.y)
                # Check if we're already facing
                cur_facing = facing_pos(px, py, prot)
                if cur_facing == (w.x, w.y):
                    return ACT5
                else:
                    return face_action

        # Try to path to adjacent cell of white mover
        for dx, dy in DIR_LIST:
            adj = (w.x - dx, w.y - dy)
            if adj in obstacles:
                continue
            if adj == (px, py):
                return action_to_face(px, py, w.x, w.y)

    # Pick up closest free piece (not linked to AI carrier)
    free_pieces = [p for p in unplaced if p not in game.zmqreragji]

    if not free_pieces:
        # All unplaced pieces are carried by AI. Wait or try to intercept.
        return None

    best_plan = None
    best_cost = float('inf')

    for piece in free_pieces:
        # Find approach positions
        for dx, dy in DIR_LIST:
            adj = (piece.x - dx, piece.y - dy)
            if adj in obstacles:
                continue
            obs_copy = obstacles.copy()
            path = bfs_path((px, py), adj, obs_copy)
            if path and len(path) < best_cost:
                if len(path) == 1:
                    # Already adjacent, face the piece
                    best_plan = action_to_face(px, py, piece.x, piece.y)
                    best_cost = 0
                else:
                    # Move toward piece
                    best_plan = path_to_actions(path[:2])[0]
                    best_cost = len(path)

    if best_plan is not None:
        # Check if we're adjacent and facing a piece — interact!
        cur_facing = facing_pos(px, py, prot)
        for piece in free_pieces:
            if cur_facing == (piece.x, piece.y):
                return ACT5

        return best_plan

    return None


def replay_from_solutions(solutions):
    fd = env.reset()
    for sol in solutions:
        for m in sol:
            fd = env.step(m)
    return fd


# Main solver
total_actions = 0
all_solutions = []

for lv in range(20):
    fd = replay_from_solutions(all_solutions)
    if fd.state.name == 'WIN':
        break

    print(f"\n=== Level {lv} ===")
    ld = extract_level_data()
    print(f"  Player: ({ld['px']},{ld['py']}) rot={ld['prot']}")
    print(f"  Goals: {ld['goals']}")
    print(f"  Reds: {ld['reds']}")
    print(f"  Whites: {ld['whites']}")
    print(f"  Steps: {ld['steps']}")

    save_frame(fd.frame, f"{VISUAL_DIR}/L{lv}_start.png")

    if ld['reds'] or ld['whites']:
        print(f"  Has AI entities — using greedy planner")
        solution = greedy_solve_with_ai(all_solutions)
    else:
        print(f"  No AI entities — using fast BFS simulation")
        solution = simulate_fast(ld)

    if solution is None:
        print(f"  FAILED on level {lv}")
        break

    # Verify
    fd = replay_from_solutions(all_solutions)
    completed_before = fd.levels_completed
    for m in solution:
        fd = env.step(m)

    sol_names = ''.join(ACTION_NAMES[a] for a in solution)
    if len(sol_names) > 80:
        sol_names = sol_names[:80] + f"...({len(solution)} total)"
    print(f"  Solution: {sol_names}")
    total_actions += len(solution)

    save_frame(fd.frame, f"{VISUAL_DIR}/L{lv}_end.png")

    if fd.levels_completed > completed_before or fd.state.name == 'WIN':
        print(f"  Verified! completed={fd.levels_completed}, state={fd.state.name}")
        all_solutions.append(solution)
    else:
        print(f"  WARNING: didn't advance! completed={fd.levels_completed}, state={fd.state.name}")
        break

print(f"\nFinal: completed={fd.levels_completed}, state={fd.state.name}, total={total_actions}")
