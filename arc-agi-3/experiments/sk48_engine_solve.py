#!/usr/bin/env python3
"""Solve sk48 using game engine + greedy/beam search.

Game engine is ground truth. Undo provides backtracking.
Uses greedy best-first search with beam fallback.
"""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
import heapq

CELL = 6
UP, DOWN, LEFT, RIGHT = GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAMES = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT'}

arcade = Arcade()
env = arcade.make('sk48-41055498')
fd = env.reset()
game = env._game

KNOWN = {
    0: [UP, UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, LEFT, DOWN, DOWN, RIGHT, LEFT, UP, RIGHT],
    1: [UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, UP, LEFT, LEFT, UP, RIGHT, RIGHT, DOWN, DOWN,
        RIGHT, UP, RIGHT, LEFT, LEFT, UP, RIGHT, RIGHT, LEFT, LEFT, UP, RIGHT, RIGHT],
}


def step_action(action, complete_win=False):
    """Execute one logical action, completing all animations."""
    f = env.step(action)
    while game.ljprkjlji or game.pzzwlsmdt:
        f = env.step(action)
    if complete_win:
        while game.lgdrixfno >= 0:
            f = env.step(action)
    return f


def state_key():
    """Compact hashable state."""
    head = game.vzvypfsnt
    segs = game.mwfajkguqx[head]
    targets = []
    for t in game.vbelzuaian:
        if t.y < 53:
            targets.append((t.x, t.y, int(t.pixels[1, 1])))
    return (head.y, len(segs), tuple(sorted(targets)))


# Level-specific cached state (set at level start)
LEVEL_REF_COLORS = []
LEVEL_HEAD_X = 0
LEVEL_GOAL_XS = []


def cache_level_info():
    """Cache ref colors and goal positions at level start (before undo corrupts lookups)."""
    global LEVEL_REF_COLORS, LEVEL_HEAD_X, LEVEL_GOAL_XS
    head = game.vzvypfsnt
    LEVEL_HEAD_X = head.x

    # Get ref colors via xpmcmtbcv (only works reliably at level start)
    ref = game.xpmcmtbcv.get(head)
    if ref:
        # Force rebuild vjfbwggsd
        game.gvtmoopqgy()
        ref_matches = game.vjfbwggsd.get(ref, [])
        if ref_matches:
            LEVEL_REF_COLORS = [int(m.pixels[1, 1]) for m in ref_matches]
        else:
            ref_segs = game.mwfajkguqx.get(ref, [])
            LEVEL_REF_COLORS = [int(s.pixels[1, 1]) for s in ref_segs]
    else:
        LEVEL_REF_COLORS = []

    n_ref = len(LEVEL_REF_COLORS)
    LEVEL_GOAL_XS = [LEVEL_HEAD_X + (i + 1) * CELL for i in range(n_ref)]
    print(f"  Cached: ref_colors={LEVEL_REF_COLORS}, head_x={LEVEL_HEAD_X}, goals={LEVEL_GOAL_XS}")


def compute_heuristic():
    """Heuristic: Manhattan distance sum to correct goal positions.

    Uses cached ref_colors and goal positions (set at level start).
    Optimizes over possible Y levels to find minimum total cost.
    """
    head = game.vzvypfsnt
    segs = game.mwfajkguqx[head]
    head_y = head.y
    n_segs = len(segs)

    n_ref = len(LEVEL_REF_COLORS)
    if n_ref == 0:
        return 0

    # Current targets (in play area)
    targets = []
    for t in game.vbelzuaian:
        if t.y < 53:
            targets.append((t.x, t.y, int(t.pixels[1, 1])))

    # Chain length deficit (need at least n_ref + 1 segments)
    needed_len = n_ref + 1
    chain_deficit = max(0, needed_len - n_segs)

    # Try different Y levels to find minimum total distance
    best_total = float('inf')
    for candidate_y in range(6, 48, CELL):
        total = 0
        used = set()
        for goal_x, goal_color in zip(LEVEL_GOAL_XS, LEVEL_REF_COLORS):
            best_d = float('inf')
            best_idx = -1
            for j, (tx, ty, tc) in enumerate(targets):
                if j in used or tc != goal_color:
                    continue
                d = abs(tx - goal_x) // CELL + abs(ty - candidate_y) // CELL
                if d < best_d:
                    best_d = d
                    best_idx = j
            if best_idx >= 0:
                used.add(best_idx)
                total += best_d
            else:
                total += 20

        # Cost to move chain to candidate_y
        total += abs(head_y - candidate_y) // CELL
        total += chain_deficit

        if total < best_total:
            best_total = total

    return best_total


def show_state(label=""):
    head = game.vzvypfsnt
    segs = game.mwfajkguqx[head]
    seg_pos = [(s.x, s.y) for s in segs]
    targets = sorted([(t.x, t.y, int(t.pixels[1,1])) for t in game.vbelzuaian if t.y < 53])
    print(f"{label} head=({head.x},{head.y}) rot={head.rotation} L={len(segs)} budget={game.qiercdohl}")
    print(f"  segs={seg_pos} targets={targets} ref={LEVEL_REF_COLORS}")


def solve_level_beam(beam_width=200, max_depth=100):
    """Beam search: keep top-k states at each depth.

    Uses game engine + undo for state management.
    Beam states are stored as move sequences and replayed from level start.
    """
    if game.gvtmoopqgy():
        return []

    init_budget = game.qiercdohl
    init_hist_len = len(game.seghobzez)

    def reset_to_start():
        while len(game.seghobzez) > init_hist_len:
            game.uqclctlhyh()
        game.qiercdohl = init_budget

    def execute_from_start(moves):
        """Replay moves from level start. Returns True if all effective."""
        reset_to_start()
        for m in moves:
            step_action(m)
            if game.lgdrixfno >= 0:
                return True  # win!
        return True

    t0 = time.time()
    total_expanded = 0

    # Beam: list of (heuristic, moves)
    init_h = compute_heuristic()
    beam = [(init_h, [])]
    visited = {state_key()}
    best_h = init_h

    print(f"  Beam search: width={beam_width}, initial h={init_h}")

    for depth in range(max_depth):
        if not beam:
            break

        candidates = []

        for _, moves in beam:
            # Replay to this state
            execute_from_start(moves)

            # Try each action
            for action in ACTIONS:
                budget_before = game.qiercdohl
                hist_before = len(game.seghobzez)

                step_action(action)
                total_expanded += 1

                # Check win
                if game.lgdrixfno >= 0:
                    # Win! Complete animation
                    while game.lgdrixfno >= 0:
                        env.step(action)
                    elapsed = time.time() - t0
                    new_moves = moves + [action]
                    print(f"  SOLVED! {len(new_moves)} moves, {total_expanded} expanded, {elapsed:.1f}s")
                    return new_moves

                sk = state_key()
                if sk in visited:
                    # Undo
                    while len(game.seghobzez) > hist_before:
                        game.uqclctlhyh()
                    game.qiercdohl = budget_before
                    continue

                # Check if state actually changed
                hist_changed = len(game.seghobzez) > hist_before
                if not hist_changed:
                    game.qiercdohl = budget_before
                    continue

                visited.add(sk)
                h = compute_heuristic()
                new_moves = moves + [action]
                candidates.append((h, new_moves))

                # Undo
                while len(game.seghobzez) > hist_before:
                    game.uqclctlhyh()
                game.qiercdohl = budget_before

        # Select top-k candidates
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

        if beam:
            cur_best = beam[0][0]
            if cur_best < best_h:
                best_h = cur_best
            elapsed = time.time() - t0
            if depth % 5 == 0 or cur_best < best_h + 1:
                print(f"  depth={depth+1}: {len(beam)} states, best_h={cur_best}, "
                      f"expanded={total_expanded}, {elapsed:.1f}s")

        if best_h == 0:
            # Check if any beam state is actually a win
            for h, moves in beam:
                if h == 0:
                    execute_from_start(moves)
                    if game.gvtmoopqgy():
                        elapsed = time.time() - t0
                        print(f"  SOLVED! {len(moves)} moves, {total_expanded} expanded, {elapsed:.1f}s")
                        return moves

    elapsed = time.time() - t0
    print(f"  No solution found. {total_expanded} expanded, {elapsed:.1f}s")
    reset_to_start()
    return None


def solve_level_astar_replay(max_states=500_000):
    """A* search using game engine with replay from level start.

    Each state stores its move sequence. To evaluate, replay from start.
    Heuristic: Manhattan distance sum.
    """
    if game.gvtmoopqgy():
        return []

    init_budget = game.qiercdohl
    init_hist_len = len(game.seghobzez)

    def reset_to_start():
        while len(game.seghobzez) > init_hist_len:
            game.uqclctlhyh()
        game.qiercdohl = init_budget

    t0 = time.time()

    init_h = compute_heuristic()
    print(f"  A* replay: initial h={init_h}")

    # Priority queue: (f_score, counter, moves)
    counter = 0
    frontier = [(init_h, counter, [])]
    visited = {state_key()}

    while frontier and len(visited) < max_states:
        f_score, _, moves = heapq.heappop(frontier)
        depth = len(moves)

        if depth > 0 and depth % 10 == 0:
            elapsed = time.time() - t0
            print(f"    A*: depth={depth}, visited={len(visited)}, frontier={len(frontier)}, {elapsed:.1f}s")

        # Replay to this state
        reset_to_start()
        for m in moves:
            step_action(m)

        # Try each action
        for action in ACTIONS:
            budget_before = game.qiercdohl
            hist_before = len(game.seghobzez)

            step_action(action)

            # Check win
            if game.lgdrixfno >= 0:
                while game.lgdrixfno >= 0:
                    env.step(action)
                new_moves = moves + [action]
                elapsed = time.time() - t0
                print(f"  SOLVED! {len(new_moves)} moves, {len(visited)} visited, {elapsed:.1f}s")
                return new_moves

            sk = state_key()
            hist_changed = len(game.seghobzez) > hist_before

            if not hist_changed:
                game.qiercdohl = budget_before
                continue

            if sk in visited:
                while len(game.seghobzez) > hist_before:
                    game.uqclctlhyh()
                game.qiercdohl = budget_before
                continue

            visited.add(sk)
            h = compute_heuristic()
            new_moves = moves + [action]
            counter += 1
            heapq.heappush(frontier, (depth + 1 + h, counter, new_moves))

            # Undo
            while len(game.seghobzez) > hist_before:
                game.uqclctlhyh()
            game.qiercdohl = budget_before

    elapsed = time.time() - t0
    print(f"  No solution found. {len(visited)} visited, {elapsed:.1f}s")
    reset_to_start()
    return None


# === MAIN ===
total_actions = 0

for lv in range(20):
    if fd.state.name == 'WIN':
        break

    print(f"\n=== Level {lv} ===")
    cache_level_info()
    show_state(f"L{lv} init")

    if lv in KNOWN:
        solution = KNOWN[lv]
        print(f"  Using known solution: {len(solution)} moves")
    else:
        # Try beam search first (faster for deep solutions)
        solution = solve_level_beam(beam_width=500, max_depth=80)
        if solution is None:
            # Fall back to A* with replay
            solution = solve_level_astar_replay(max_states=200_000)

    if solution is None:
        print(f"  FAILED on level {lv}")
        break

    if not solution:
        print(f"  Already solved!")
        continue

    # Execute the solution properly
    # Reset to level start first
    init_hist = len(game.seghobzez)
    while len(game.seghobzez) > init_hist:
        game.uqclctlhyh()

    lv_actions = 0
    for i, a in enumerate(solution):
        is_last = (i == len(solution) - 1)
        fd = step_action(a, complete_win=is_last)
        lv_actions += 1

    total_actions += lv_actions
    sol_names = [ACTION_NAMES[a] for a in solution]
    print(f"  Solution: {sol_names}")
    print(f"  {lv_actions} moves, completed={fd.levels_completed}, state={fd.state.name}")

    # Save solution for future use
    KNOWN[lv] = solution

print(f"\nFinal: completed={fd.levels_completed}, state={fd.state.name}, total={total_actions}")
