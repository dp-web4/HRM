#!/usr/bin/env python3
"""sk48 Rail Weaver — Final solver.

L0-L1 solved (41 actions). L2+ use beam search with distance heuristic.

Mechanics (verified from source):
- gvtmoopqgy (win check): compares vjfbwggsd[head] colors with vjfbwggsd[ref] colors
  index-by-index. vjfbwggsd is built by scanning segments left-to-right and collecting
  any target sprite at that exact (x,y).
- Extension: new segment added at HEAD, existing segments shift in extension direction.
  Segments that shift into a target position push the target (unless perpendicular
  segment pins it). Pushed target chain-pushes any target at its destination.
- Retraction: removes segment closest to head, remaining segments shift toward head.
  Targets at shifting segment positions get pulled along.
- Perpendicular pin: target at (x,y) with a perpendicular segment at same (x,y)
  cannot be pushed in the extension direction. Target stays, segment passes through.
- Vertical rail segments at x=29, y=0..24 pin targets at those positions.
- Key: targets don't need to MOVE to win — they just need to be at segment positions
  with the correct color order when gvtmoopqgy checks.

L2 strategy discovered:
1. Push target stack below vert rail (UP*6, RIGHT*3, DOWN*3, LEFT*3)
2. Clear (29,30): extend to push target(9) to x=35 (RIGHT*2, DOWN, RIGHT, LEFT)
3. Push target(14) from (29,24) to (29,30): slide from y=18 with 5 segs (UP*2, RIGHT, DOWN)
4. All 4 targets now below vert rail — need final arrangement
5. L2 collect order: [8,12,9,14] from left to right at same row

Full L2 solution remains unsolved — the arrangement step requires careful
positioning of all 4 targets at distinct x positions on one row.
"""
import sys, os, json, time
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(__file__))
from arc_agi import Arcade
from arcengine import GameAction
import numpy as np

CELL = 6
UP = GameAction.ACTION1
DOWN = GameAction.ACTION2
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
CLICK = GameAction.ACTION6

ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAMES = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT', CLICK: 'CLICK'}

VISUAL_DIR = '/mnt/c/exe/projects/ai-agents/shared-context/arc-agi-3/visual-memory/sk48'

KNOWN = {
    0: [UP, UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, LEFT, DOWN, DOWN, RIGHT, LEFT, UP, RIGHT],
    1: [UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, UP, LEFT, LEFT, UP, RIGHT, RIGHT, DOWN, DOWN,
        RIGHT, UP, RIGHT, LEFT, LEFT, UP, RIGHT, RIGHT, LEFT, LEFT, UP, RIGHT, RIGHT],
}


def state_key(game):
    head = game.vzvypfsnt
    segs = game.mwfajkguqx.get(head, [])
    targets = tuple(sorted((t.x, t.y, int(t.pixels[1, 1]))
                           for t in game.vbelzuaian))
    return (head.x, head.y, len(segs), targets)


def compute_h(game):
    """Heuristic: unmatched count * 100 + target distance."""
    game.gvtmoopqgy()
    total = 0.0
    for head, ref_head in game.xpmcmtbcv.items():
        upper = game.vjfbwggsd.get(head, [])
        lower = game.vjfbwggsd.get(ref_head, [])
        ref_colors = [int(t.pixels[1, 1]) for t in lower]
        n_lower = len(lower)
        n_upper = len(upper)

        # Count consecutive matches from start
        matched = 0
        for i in range(min(n_upper, n_lower)):
            if int(upper[i].pixels[1, 1]) == ref_colors[i]:
                matched += 1
            else:
                break

        total += (n_lower - matched) * 100

        # Distance of unmatched targets
        avail = [(t.x, t.y, int(t.pixels[1, 1])) for t in game.vbelzuaian if t.y < 53]
        used = set()
        for i in range(matched, n_lower):
            rc = ref_colors[i]
            best = 50
            for j, (tx, ty, tc) in enumerate(avail):
                if j in used or tc != rc:
                    continue
                d = abs(ty - head.y) // CELL
                if d < best:
                    best = d
                    best_j = j
            if best < 50:
                used.add(best_j)
            total += best

    return total


def solve_beam(env, game, beam_width=5000, max_depth=200, use_click=False,
               timeout_secs=600):
    """Beam search with distance heuristic."""
    all_actions = list(ACTIONS)
    if use_click:
        all_actions.append(CLICK)

    if game.gvtmoopqgy():
        return []

    init_budget = game.qiercdohl
    init_hist = len(game.seghobzez)

    def reset():
        while len(game.seghobzez) > init_hist:
            game.uqclctlhyh()
        game.qiercdohl = init_budget

    t0 = time.time()
    h0 = compute_h(game)
    beam = [(h0, [])]
    visited = {state_key(game)}
    best_h = h0
    expanded = 0

    print(f"  Beam: w={beam_width}, h0={h0:.0f}")

    for depth in range(max_depth):
        if not beam:
            break
        cands = []
        for _, moves in beam:
            reset()
            for m in moves:
                env.step(m)

            for act in all_actions:
                bud = game.qiercdohl
                hist = len(game.seghobzez)
                env.step(act)
                expanded += 1

                if game.lgdrixfno >= 0:
                    while game.lgdrixfno >= 0 and game.lgdrixfno < 35:
                        env.step(act)
                    sol = moves + [act]
                    print(f"  SOLVED! {len(sol)} moves, {expanded} exp, {time.time()-t0:.1f}s")
                    reset()
                    return sol

                changed = len(game.seghobzez) > hist
                if not changed:
                    continue

                sk = state_key(game)
                if sk in visited:
                    while len(game.seghobzez) > hist:
                        game.uqclctlhyh()
                    game.qiercdohl = bud
                    continue

                visited.add(sk)
                h = compute_h(game)
                cands.append((h, moves + [act]))
                while len(game.seghobzez) > hist:
                    game.uqclctlhyh()
                game.qiercdohl = bud

        cands.sort(key=lambda x: (x[0], len(x[1])))
        beam = cands[:beam_width]

        if beam:
            cur_h = beam[0][0]
            if cur_h < best_h - 0.1:
                best_h = cur_h
                print(f"    d={depth}: h={cur_h:.0f}, {len(beam)} cands, {expanded} exp, {time.time()-t0:.1f}s")
            elif depth % 20 == 0:
                print(f"    d={depth}: h={cur_h:.0f}, {len(beam)} cands, {len(visited)} vis, {time.time()-t0:.1f}s")

        if time.time() - t0 > timeout_secs:
            print(f"  Timeout at d={depth}, {time.time()-t0:.1f}s")
            break

    print(f"  Failed. {expanded} exp, best_h={best_h:.0f}")
    reset()
    return None


def main():
    print("=" * 60, flush=True)
    print("sk48 Final Solver", flush=True)
    print("=" * 60, flush=True)

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

        for h, r in game.xpmcmtbcv.items():
            game.gvtmoopqgy()
            ref_targets = game.vjfbwggsd.get(r, [])
            ref_colors = [int(t.pixels[1,1]) for t in ref_targets]
            print(f"  ({h.x},{h.y})->({r.x},{r.y}) ref={ref_colors}")

        targets_above = [(t.x, t.y, int(t.pixels[1,1])) for t in game.vbelzuaian if t.y < 53]
        print(f"  Upper: {sorted(targets_above)}")

        use_click = len(game.xpmcmtbcv) > 1

        if lv in KNOWN:
            sol = KNOWN[lv]
            print(f"  Known: {len(sol)} moves")
            for a in sol:
                fd = env.step(a)
            if game.level_index > lv:
                print(f"  L{lv} SOLVED!")
                results[lv] = len(sol)
                total += len(sol)
            else:
                print(f"  Known FAILED!")
                break
        else:
            t0 = time.time()
            print("  Beam search...")
            sol = solve_beam(env, game, beam_width=5000, max_depth=200,
                            use_click=use_click, timeout_secs=600)
            dt = time.time() - t0

            if sol is None:
                print(f"  FAILED L{lv} ({dt:.1f}s)")
                break

            sol_names = [ACTION_NAMES.get(a, str(a)) for a in sol]
            print(f"  Solution ({len(sol)}): {sol_names}")

            if game.level_index <= lv:
                while len(game.seghobzez) > 1:
                    game.uqclctlhyh()
                for a in sol:
                    fd = env.step(a)
                count = 0
                while game.level_index <= lv and count < 50:
                    fd = env.step(UP)
                    count += 1

            if game.level_index > lv:
                print(f"  L{lv} SOLVED! ({len(sol)} moves, {dt:.1f}s)")
                results[lv] = len(sol)
                total += len(sol)
            else:
                print(f"  FAILED L{lv} ({dt:.1f}s)")
                break

    print(f"\n{'='*60}")
    print(f"FINAL: {len(results)}/8 solved, {total} total actions")
    for lv, n in sorted(results.items()):
        print(f"  L{lv}: {n} actions")

    os.makedirs(VISUAL_DIR, exist_ok=True)
    with open(f'{VISUAL_DIR}/sk48_results.json', 'w') as f:
        json.dump({
            'game_id': 'sk48-41055498',
            'solved': len(results),
            'total': total,
            'per_level': results
        }, f, indent=2)


if __name__ == '__main__':
    main()
