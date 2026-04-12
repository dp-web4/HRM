#!/usr/bin/env python3
"""sk48 Rail Weaver solver — engine-driven beam + greedy.

Core mechanic: extend rail through targets to visit them. Win = all visited.
For horizontal rail (rot=0): slide UP/DOWN, extend RIGHT, retract LEFT.
"""
import sys, os, json, time
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np
import heapq

CELL = 6
UP = GameAction.ACTION1
DOWN = GameAction.ACTION2
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
CLICK = GameAction.ACTION6
UNDO = GameAction.ACTION7

ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAMES = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT'}

VISUAL_DIR = '/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/sk48'

KNOWN = {
    0: [UP, UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, LEFT, DOWN, DOWN, RIGHT, LEFT, UP, RIGHT],
    1: [UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, UP, LEFT, LEFT, UP, RIGHT, RIGHT, DOWN, DOWN,
        RIGHT, UP, RIGHT, LEFT, LEFT, UP, RIGHT, RIGHT, LEFT, LEFT, UP, RIGHT, RIGHT],
}


def step_action(env, game, action, complete_win=False):
    """Execute one action. env.step handles animations internally."""
    return env.step(action)


def state_key(game):
    head = game.vzvypfsnt
    segs = game.mwfajkguqx.get(head, [])
    targets = tuple(sorted((t.x, t.y, int(t.pixels[1, 1]))
                           for t in game.vbelzuaian if t.y < 53))
    return (head.y, len(segs), targets)


def solve_beam(env, game, beam_width=500, max_depth=80):
    """Beam search using engine undo for state management."""

    # Cache level info
    head = game.vzvypfsnt
    head_x = head.x
    ref = game.xpmcmtbcv.get(head)
    ref_colors = []
    if ref:
        game.gvtmoopqgy()
        ref_matches = game.vjfbwggsd.get(ref, [])
        if ref_matches:
            ref_colors = [int(m.pixels[1, 1]) for m in ref_matches]
        else:
            ref_segs = game.mwfajkguqx.get(ref, [])
            ref_colors = [int(s.pixels[1, 1]) for s in ref_segs]

    n_ref = len(ref_colors)
    goal_xs = [head_x + (i + 1) * CELL for i in range(n_ref)]

    print(f"  ref_colors={ref_colors}, goal_xs={goal_xs}")

    def compute_h():
        hd = game.vzvypfsnt
        segs = game.mwfajkguqx.get(hd, [])
        targets = [(t.x, t.y, int(t.pixels[1, 1])) for t in game.vbelzuaian if t.y < 53]
        if not ref_colors:
            return len(targets)
        n_segs = len(segs)
        deficit = max(0, n_ref + 1 - n_segs)
        best = float('inf')
        for cy in range(6, 48, CELL):
            total = deficit + abs(hd.y - cy) // CELL
            used = set()
            for gx, gc in zip(goal_xs, ref_colors):
                bd, bi = float('inf'), -1
                for j, (tx, ty, tc) in enumerate(targets):
                    if j in used or tc != gc:
                        continue
                    d = abs(tx - gx) // CELL + abs(ty - cy) // CELL
                    if d < bd:
                        bd, bi = d, j
                if bi >= 0:
                    used.add(bi)
                    total += bd
                else:
                    total += 20
            best = min(best, total)
        return best

    if game.gvtmoopqgy():
        return []

    init_budget = game.qiercdohl
    init_hist = len(game.seghobzez)

    def reset():
        while len(game.seghobzez) > init_hist:
            game.uqclctlhyh()
        game.qiercdohl = init_budget

    t0 = time.time()
    h0 = compute_h()
    beam = [(h0, [])]
    visited = {state_key(game)}
    best_h = h0
    expanded = 0

    print(f"  Beam: w={beam_width}, h0={h0}")

    for depth in range(max_depth):
        if not beam:
            break
        cands = []
        for _, moves in beam:
            reset()
            for m in moves:
                step_action(env, game, m)

            for act in ACTIONS:
                bud = game.qiercdohl
                hist = len(game.seghobzez)
                step_action(env, game, act)
                expanded += 1

                if game.lgdrixfno >= 0:
                    while game.lgdrixfno >= 0:
                        env.step(act)
                    sol = moves + [act]
                    print(f"  SOLVED! {len(sol)} moves, {expanded} exp, {time.time()-t0:.1f}s")
                    return sol

                sk = state_key(game)
                changed = len(game.seghobzez) > hist
                if not changed or sk in visited:
                    while len(game.seghobzez) > hist:
                        game.uqclctlhyh()
                    game.qiercdohl = bud
                    continue

                visited.add(sk)
                h = compute_h()
                cands.append((h, moves + [act]))
                while len(game.seghobzez) > hist:
                    game.uqclctlhyh()
                game.qiercdohl = bud

        cands.sort(key=lambda x: x[0])
        beam = cands[:beam_width]

        if beam:
            cur = beam[0][0]
            if cur < best_h:
                best_h = cur
            if depth % 5 == 0:
                print(f"    d={depth}: {len(beam)} states, h={cur}, exp={expanded}, {time.time()-t0:.1f}s")

    print(f"  No solution. {expanded} expanded, {time.time()-t0:.1f}s")
    reset()
    return None


def main():
    import sys
    print("=" * 60, flush=True)
    print("sk48 Solver — Known + Beam", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()

    arcade = Arcade()
    env = arcade.make('sk48-41055498')
    fd = env.reset()
    game = env._game

    total = 0
    results = {}

    for lv in range(8):
        print(f"\n{'='*50}")
        print(f"Level {lv} (engine={game.level_index})")
        print(f"{'='*50}")

        if fd.state.name in ('WON', 'GAME_OVER'):
            print(f"  Game ended: {fd.state.name}")
            break

        if lv in KNOWN:
            sol = KNOWN[lv]
            print(f"  Known: {len(sol)} moves")
            for a in sol:
                step_action(env, game, a, complete_win=(a == sol[-1]))
            if game.level_index > lv:
                print(f"  L{lv} SOLVED!")
                results[lv] = len(sol)
                total += len(sol)
            else:
                print(f"  FAILED!")
                break
        else:
            t0 = time.time()
            sol = solve_beam(env, game, beam_width=500, max_depth=100)
            dt = time.time() - t0
            print(f"  Time: {dt:.1f}s")

            if sol is None:
                print(f"  FAILED L{lv}")
                break

            if game.level_index > lv:
                print(f"  L{lv} SOLVED! ({len(sol)} moves)")
                results[lv] = len(sol)
                total += len(sol)
            else:
                # Need to execute the solution
                for a in sol:
                    step_action(env, game, a, complete_win=(a == sol[-1]))
                if game.level_index > lv:
                    print(f"  L{lv} SOLVED! ({len(sol)} moves)")
                    results[lv] = len(sol)
                    total += len(sol)
                else:
                    print(f"  FAILED L{lv}")
                    break

    print(f"\n{'='*60}")
    print(f"FINAL: {len(results)}/8 solved, {total} actions")

    with open(f'{VISUAL_DIR}/sk48_results.json', 'w') as f:
        json.dump({'game_id': 'sk48-41055498', 'solved': len(results), 'total': total, 'per_level': results}, f, indent=2)


if __name__ == '__main__':
    main()
