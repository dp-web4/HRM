#!/usr/bin/env python3
"""Diagnose why h=0 doesn't match gvtmoopqgy() win check."""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction

CELL = 6
UP, DOWN, LEFT, RIGHT = GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAMES = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT'}

arcade = Arcade()
env = arcade.make('sk48-41055498')
fd = env.reset()
game = env._game

L0_SOL = [UP, UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, LEFT, DOWN, DOWN, RIGHT, LEFT, UP, RIGHT]
L1_SOL = [UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, UP, LEFT, LEFT, UP, RIGHT, RIGHT, DOWN, DOWN,
          RIGHT, UP, RIGHT, LEFT, LEFT, UP, RIGHT, RIGHT, LEFT, LEFT, UP, RIGHT, RIGHT]


def step_action(action):
    f = env.step(action)
    while game.ljprkjlji or game.pzzwlsmdt:
        f = env.step(action)
    while game.lgdrixfno >= 0:
        f = env.step(action)
    return f


def get_ref_colors():
    head = game.vzvypfsnt
    ref = game.xpmcmtbcv.get(head)
    if not ref:
        return []
    ref_matches = game.vjfbwggsd.get(ref, [])
    if ref_matches:
        return [int(m.pixels[1, 1]) for m in ref_matches]
    ref_segs = game.mwfajkguqx.get(ref, [])
    return [int(s.pixels[1, 1]) for s in ref_segs]


def compute_heuristic():
    head = game.vzvypfsnt
    segs = game.mwfajkguqx[head]
    head_x, head_y = head.x, head.y
    n_segs = len(segs)
    ref_colors = get_ref_colors()
    n_ref = len(ref_colors)
    if n_ref == 0:
        return 0, {}

    targets = []
    for t in game.vbelzuaian:
        if t.y < 53:
            targets.append((t.x, t.y, int(t.pixels[1, 1])))

    goal_xs = [head_x + (i + 1) * CELL for i in range(n_ref)]
    needed_len = n_ref + 1
    chain_deficit = max(0, needed_len - n_segs)

    best_total = float('inf')
    best_info = {}
    for candidate_y in range(6, 48, CELL):
        total = 0
        used = set()
        assignments = []
        for gi, (goal_x, goal_color) in enumerate(zip(goal_xs, ref_colors)):
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
                assignments.append((goal_x, candidate_y, goal_color, targets[best_idx], best_d))
            else:
                total += 20
                assignments.append((goal_x, candidate_y, goal_color, None, 20))

        total += abs(head_y - candidate_y) // CELL
        total += chain_deficit

        if total < best_total:
            best_total = total
            best_info = {
                'y': candidate_y, 'chain_cost': abs(head_y - candidate_y) // CELL,
                'chain_deficit': chain_deficit, 'assignments': assignments,
            }

    return best_total, best_info


# Complete L0, L1
for a in L0_SOL:
    fd = step_action(a)
for a in L1_SOL:
    fd = step_action(a)

init_budget = game.qiercdohl
init_hist_len = len(game.seghobzez)

# Run beam search with diagnostics on h=0
t0 = time.time()
beam = [(17, [])]  # approximate initial h
visited = set()

# Use same beam search but print diagnostic at h=0
def state_key():
    head = game.vzvypfsnt
    segs = game.mwfajkguqx[head]
    targets = []
    for t in game.vbelzuaian:
        if t.y < 53:
            targets.append((t.x, t.y, int(t.pixels[1, 1])))
    return (head.y, len(segs), tuple(sorted(targets)))


def reset_to_start():
    while len(game.seghobzez) > init_hist_len:
        game.uqclctlhyh()
    game.qiercdohl = init_budget


def execute_from_start(moves):
    reset_to_start()
    for m in moves:
        step_action(m)


# Recompute initial h properly
h0, info0 = compute_heuristic()
beam = [(h0, [])]
visited = {state_key()}
print(f"Initial h={h0}, info={info0}")

found_h0 = False
for depth in range(80):
    if not beam:
        break
    candidates = []
    for _, moves in beam:
        execute_from_start(moves)
        for action in ACTIONS:
            budget_before = game.qiercdohl
            hist_before = len(game.seghobzez)
            step_action(action)

            if game.lgdrixfno >= 0:
                while game.lgdrixfno >= 0:
                    env.step(action)
                new_moves = moves + [action]
                print(f"SOLVED! {len(new_moves)} moves at depth {depth+1}")
                print([ACTION_NAMES[a] for a in new_moves])
                sys.exit(0)

            sk = state_key()
            hist_changed = len(game.seghobzez) > hist_before
            if not hist_changed or sk in visited:
                while len(game.seghobzez) > hist_before:
                    game.uqclctlhyh()
                game.qiercdohl = budget_before
                continue

            visited.add(sk)
            h, info = compute_heuristic()
            new_moves = moves + [action]
            candidates.append((h, new_moves))

            # Diagnostic for h=0
            if h == 0 and not found_h0:
                found_h0 = True
                head = game.vzvypfsnt
                segs = game.mwfajkguqx[head]
                print(f"\n=== FIRST h=0 STATE (depth {depth+1}) ===")
                print(f"Head: ({head.x},{head.y}) L={len(segs)}")
                print(f"Segs: {[(s.x, s.y) for s in segs]}")
                for t in sorted(game.vbelzuaian, key=lambda t: (t.y, t.x)):
                    if t.y < 53:
                        print(f"  Target: ({t.x},{t.y}) color={int(t.pixels[1,1])}")
                print(f"Info: {info}")

                # Check actual vjfbwggsd
                won = game.gvtmoopqgy()
                print(f"gvtmoopqgy() = {won}")
                ref = game.xpmcmtbcv.get(head)
                print(f"vjfbwggsd[head]: {[(t.x, t.y, int(t.pixels[1,1])) for t in game.vjfbwggsd[head]]}")
                print(f"vjfbwggsd[ref]:  {[(t.x, t.y, int(t.pixels[1,1])) for t in game.vjfbwggsd[ref]]}")
                print(f"Moves: {[ACTION_NAMES[a] for a in new_moves]}")

            while len(game.seghobzez) > hist_before:
                game.uqclctlhyh()
            game.qiercdohl = budget_before

    candidates.sort(key=lambda x: x[0])
    beam = candidates[:500]

    if beam:
        cur_best = beam[0][0]
        elapsed = time.time() - t0
        if depth % 5 == 0 or cur_best == 0:
            print(f"depth={depth+1}: {len(beam)} states, best_h={cur_best}, {elapsed:.1f}s")

print("No solution found.")
