#!/usr/bin/env python3
"""Solve wa30 using fast pure-Python simulation + BFS."""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
from collections import deque

CELL = 4
UP, DOWN, LEFT, RIGHT, ACT5 = (
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
    GameAction.ACTION4, GameAction.ACTION5
)
ACTION_NAMES = {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R', ACT5: '5'}
DIRS = {UP: (0, -CELL), DOWN: (0, CELL), LEFT: (-CELL, 0), RIGHT: (CELL, 0)}

arcade = Arcade()
env = arcade.make('wa30-ee6fef47')
fd = env.reset()
game = env._game


def extract_level_data():
    """Extract static and dynamic state from current game level."""
    player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    goals_sprites = game.current_level.get_sprites_by_tag("geezpjgiyd")
    reds_sprites = game.current_level.get_sprites_by_tag("kdweefinfi")
    whites_sprites = game.current_level.get_sprites_by_tag("ysysltqlke")

    # Dynamic positions (collidable entities)
    px, py, prot = player.x, player.y, player.rotation
    goal_positions = tuple(sorted((g.x, g.y) for g in goals_sprites))
    red_positions = tuple(sorted((r.x, r.y) for r in reds_sprites))
    white_positions = tuple(sorted((w.x, w.y) for w in whites_sprites))

    # Static walls: pkbufziase minus dynamic entity positions
    dynamic_pos = {(px, py)}
    for g in goals_sprites:
        dynamic_pos.add((g.x, g.y))
    for r in reds_sprites:
        dynamic_pos.add((r.x, r.y))
    for w in whites_sprites:
        dynamic_pos.add((w.x, w.y))
    walls = frozenset(p for p in game.pkbufziase if p not in dynamic_pos)

    blocked = frozenset(game.qthdiggudy)
    slots = frozenset((x, y) for (x, y) in game.wyzquhjerd)
    slots2 = frozenset((x, y) for (x, y) in game.lqctaojiby)

    # Carrying state
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
    """Fast BFS simulation for levels without AI entities (reds/whites)."""
    walls = level_data['walls']
    blocked = level_data['blocked']
    slots = level_data['slots']

    # State: (px, py, prot, goal_positions, carrying_offset_or_none)
    init_goals = level_data['goals']
    init_state = (level_data['px'], level_data['py'], level_data['prot'],
                  init_goals, level_data['carrying'])

    goal_set_cache = {}

    def goals_as_set(goals):
        if goals not in goal_set_cache:
            goal_set_cache[goals] = set(goals)
        return goal_set_cache[goals]

    def is_static_wall(x, y):
        return (x, y) in walls or (x, y) in blocked

    def at_slot(x, y):
        return (x, y) in slots

    def state_key(s):
        return (s[0], s[1], s[3], s[4])  # skip rotation for state dedup

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

        # Try each action
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
                    # Carrying: both player and carried object move
                    cdx, cdy = carrying
                    old_carried = (px + cdx, py + cdy)
                    cnx, cny = nx + cdx, ny + cdy
                    # Check: new player pos not wall, not goal (except carried)
                    if is_static_wall(nx, ny) or is_static_wall(cnx, cny):
                        continue
                    # Check goals excluding carried object
                    other_goals = gset - {old_carried}
                    if (nx, ny) in other_goals or (cnx, cny) in other_goals:
                        continue
                    new_px, new_py = nx, ny
                    new_goals = tuple(sorted(
                        (cnx, cny) if (gx, gy) == old_carried else (gx, gy)
                        for gx, gy in goals
                    ))
                else:
                    # Not carrying: check static walls and goal positions
                    if is_static_wall(nx, ny) or (nx, ny) in gset:
                        continue
                    new_px, new_py = nx, ny

            elif action == ACT5:
                if carrying is not None:
                    # Drop
                    new_carrying = None
                else:
                    # Pick up: check facing direction
                    fx, fy = facing(px, py, prot)
                    if (fx, fy) in goals:
                        # Pick up this goal
                        new_carrying = (fx - px, fy - py)
                    else:
                        continue  # nothing to pick up

            new_state = (new_px, new_py, new_rot, new_goals, new_carrying)
            sk = state_key(new_state)
            if sk in visited:
                continue
            visited.add(sk)

            new_moves = moves + [action]

            # Check win (all goals at slots AND not carrying)
            if new_carrying is None and is_won(new_goals):
                elapsed = time.time() - t0
                print(f"  SOLVED! {len(new_moves)} moves, {expanded} expanded, {elapsed:.1f}s")
                return new_moves

            queue.append((new_state, new_moves))

    elapsed = time.time() - t0
    print(f"  No solution. {expanded} expanded, {len(visited)} visited, {elapsed:.1f}s")
    return None


def replay_from_solutions(solutions):
    """Reset and replay all solved levels, then current level moves."""
    fd = env.reset()
    for sol in solutions:
        for m in sol:
            fd = env.step(m)
    return fd


def engine_state_key():
    """Get hashable state from engine (includes AI entity positions)."""
    player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    reds = game.current_level.get_sprites_by_tag("kdweefinfi")
    whites = game.current_level.get_sprites_by_tag("ysysltqlke")
    carrying = tuple(sorted((id(k), id(v)) for k, v in game.nsevyuople.items()))
    return (player.x, player.y,
            tuple(sorted((g.x, g.y) for g in goals)),
            tuple(sorted((r.x, r.y) for r in reds)),
            tuple(sorted((w.x, w.y) for w in whites)),
            carrying)


def count_goals_at_slots():
    """Count how many goals are currently at slot positions."""
    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    slots = set((x, y) for (x, y) in game.wyzquhjerd)
    return sum(1 for g in goals if (g.x, g.y) in slots)


def heuristic_score():
    """Score: higher is better. Goals-at-slots * 1000 - sum of min distances of unplaced goals to unoccupied slots."""
    goals = game.current_level.get_sprites_by_tag("geezpjgiyd")
    slots_all = [(x, y) for (x, y) in game.wyzquhjerd]

    goal_positions = [(g.x, g.y) for g in goals]
    goal_set = set(goal_positions)

    # Which slots are already occupied by a goal?
    occupied = set(s for s in slots_all if s in goal_set)
    at_slot = len(occupied)

    # Unplaced goals and empty slots
    unplaced = [g for g in goal_positions if g not in occupied]
    empty_slots = [s for s in slots_all if s not in occupied]

    remaining_dist = 0
    for gx, gy in unplaced:
        if empty_slots:
            best_d = min(abs(gx - sx) + abs(gy - sy) for sx, sy in empty_slots)
        else:
            best_d = 0
        remaining_dist += best_d

    return at_slot * 1000 - remaining_dist


def simulate_with_ai(level_data, prev_solutions, beam_width=200, max_depth=70):
    """Engine-based beam search for levels with AI entities.

    Beam search: at each depth, expand all states in beam, keep top-N by
    goals-at-slots score. Much more efficient than best-first for deep solutions.
    """
    n_goals = len(level_data['goals'])
    max_steps = level_data['steps']
    if max_depth > max_steps:
        max_depth = max_steps

    n_prev = len(prev_solutions)

    def replay_and_execute(moves):
        """Replay all previous solutions then execute moves for current level."""
        fd = replay_from_solutions(prev_solutions)
        for m in moves:
            fd = env.step(m)
        return fd

    t0 = time.time()

    # Get initial state
    replay_and_execute([])
    init_sk = engine_state_key()
    init_h = heuristic_score()
    init_goals_done = count_goals_at_slots()

    # Beam: list of (heuristic_score, moves)
    beam = [(init_h, [])]
    visited = {init_sk}
    total_expanded = 0
    best_goals = init_goals_done

    print(f"  Beam search: {n_goals} goals, {max_steps} steps, width={beam_width}, "
          f"init_h={init_h}, init_done={init_goals_done}")

    for depth in range(max_depth):
        if not beam:
            break

        candidates = []

        for _, moves in beam:
            for action in [UP, DOWN, LEFT, RIGHT, ACT5]:
                replay_and_execute(moves)
                fd = env.step(action)
                total_expanded += 1

                # Check win
                if fd.levels_completed > n_prev or fd.state.name == 'WIN':
                    new_moves = moves + [action]
                    elapsed = time.time() - t0
                    print(f"  SOLVED! {len(new_moves)} moves, {total_expanded} expanded, {elapsed:.1f}s")
                    return new_moves

                sk = engine_state_key()
                if sk in visited:
                    continue
                visited.add(sk)

                h = heuristic_score()
                goals_done = h // 1000 if h >= 0 else 0
                if goals_done > best_goals:
                    best_goals = goals_done
                    elapsed = time.time() - t0
                    print(f"    NEW BEST: {goals_done}/{n_goals} at depth {depth+1}, h={h}, {elapsed:.1f}s")

                candidates.append((h, moves + [action]))

        # Keep top beam_width candidates (highest heuristic first)
        candidates.sort(key=lambda x: -x[0])
        beam = candidates[:beam_width]

        elapsed = time.time() - t0
        if depth % 5 == 0 or best_goals > init_goals_done:
            cur_best = beam[0][0] if beam else -1
            print(f"    depth={depth+1}: {len(beam)} states, best_h={cur_best}, "
                  f"best_goals={best_goals}/{n_goals}, expanded={total_expanded}, {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  No solution. {total_expanded} expanded, {len(visited)} visited, {elapsed:.1f}s")
    return None


# Main solver
total_actions = 0
all_solutions = []

for lv in range(20):
    # Replay all previous solutions then extract current level state
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

    if ld['reds'] or ld['whites']:
        print(f"  Has AI entities — using engine search")
        solution = simulate_with_ai(ld, all_solutions)
    else:
        print(f"  No AI entities — using fast simulation")
        solution = simulate_fast(ld)

    if solution is None:
        print(f"  FAILED on level {lv}")
        break

    # Verify: replay everything and execute solution
    fd = replay_from_solutions(all_solutions)
    completed_before = fd.levels_completed
    for m in solution:
        fd = env.step(m)

    sol_names = ''.join(ACTION_NAMES[a] for a in solution)
    print(f"  Solution: {sol_names}")
    total_actions += len(solution)

    if fd.levels_completed > completed_before or fd.state.name == 'WIN':
        print(f"  Verified! completed={fd.levels_completed}, state={fd.state.name}")
        all_solutions.append(solution)
    else:
        print(f"  WARNING: didn't advance! completed={fd.levels_completed}, state={fd.state.name}")
        break

print(f"\nFinal: completed={fd.levels_completed}, state={fd.state.name}, total={total_actions}")
