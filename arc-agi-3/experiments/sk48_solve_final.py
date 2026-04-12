#!/usr/bin/env python3
"""sk48 Rail Weaver — Final solver.

L0-L1 solved (41 actions). L2+ unsolved — see world model below.

=============================================================================
WORLD MODEL (verified by source + empirical trace, refined 2026-04-12)
=============================================================================

RAILS: game.mwfajkguqx maps each HEAD sprite to its list of segments.
  game.xpmcmtbcv maps upper HEAD -> lower REF (only paired rails).
  game.vzvypfsnt is the currently active head.
  NOT every rail in mwfajkguqx is paired! L2 has THREE rails:
    1. Upper horizontal player rail at head (5,42), paired with
    2. Lower horizontal ref rail at (17,56).
    3. A VERTICAL rail at head (29,0) with 5 segs to (29,24), UNPAIRED.
  The vertical rail is in mwfajkguqx (and has sys_click tag), but because it
  has no pair in xpmcmtbcv, CLICK cannot activate it. It is effectively
  STATIC SCENERY that pins targets perpendicular-wise.

HEAD / EXTENSION DIRECTION: rotation maps to hhvuoijeua:
    {0: (1,0), 90: (0,1), 180: (-1,0), 270: (0,-1)}
  Rot 0 = head is on the LEFT, extends RIGHT (+x). Rot 90 = head on TOP,
  extends DOWN (+y). Etc.

ACTIONS (when a horizontal-oriented rail is active, rot=0):
  ACTION4 (RIGHT, = extension direction): inserts new seg at head,
    existing segs shift +6 x. Segs hitting targets chain-push (may fail —
    target stays, segment passes through -> "embed").
  ACTION3 (LEFT, = retract): pops head seg, remaining segs shift -6 x.
    Segs at target positions DRAG the target along via recursive bnrdrdiakd.
  ACTION1/2 (UP/DOWN = perpendicular): only valid if an 'irkeobngyh' TRACK
    sprite is at (head.x+2, head.y+2 + dy/2). Moves all segs by (0,dy).
    Any seg at a target position chain-pushes the target by (0,dy); chain
    continues through more targets; fails if end of chain can't move.

WIN CHECK (gvtmoopqgy):
  Each pair (head, ref) in xpmcmtbcv has indicator list jdojcthkf[ref]
  built ONCE at init from ref's segments[1:] (so len = initial_ref_segs - 1).
  For each i in 0..len(jdojcthkf[ref])-1:
    upper_visit = vjfbwggsd[head] (= targets at current head seg positions,
                  in segment-list order which is head->tail).
    lower_visit = vjfbwggsd[ref] (same, for ref).
    Need upper_visit[i] to exist and upper_visit[i].color == lower_visit[i].color.
  Lower ref rail can be RETRACTED by player (if active) — changes lower_visit
  but NOT jdojcthkf (which is frozen at init). Reducing lower_visit below the
  required count would IndexError (so the game is designed to avoid this).

PERPENDICULAR PINNING (bnrdrdiakd target branch):
  When a segment tries to move a target, the target scans pptqisyill
  (excluding the calling segment as hstakrujhu) at its current+dest for
  PERPENDICULAR segments. If one exists AND (segment_dir_x==0)!=(move_x==0)
  (i.e. the found perpendicular seg is NOT parallel to the attempted move),
  the target returns False.
  Caller segment branch: on False with parallel-to-segment move direction,
  the segment passes through (target added to pzzwlsmdt / embed glow) and
  still moves itself. On perpendicular-to-segment move direction, the
  segment's move fails.

KEY CONSEQUENCE for L2:
  Target at (29,y) for y in [0..24] has vertical pin seg at (29,y). A
  HORIZONTAL extension at row y tries to push the target +x. Target scans
  for perpendicular (vertical) seg at (29,y) -> finds the vertical rail ->
  (seg_dir_x==0)=True, (move_x==0)=False -> True != False -> returns False.
  So horizontal extension CANNOT move target at (29,y) for y<=24. The
  horizontal segment still passes through (parallel move to its own dir).

  SLIDES (vertical move) do NOT hit the pin check: move is (0,±6),
  perpendicular seg has seg_dir_x=0, move_x=0: (True)!=(True)=False.
  So slides can chain-push the column of targets vertically.

  Below y=24, there is no pin, so horizontal extensions push targets freely.

L2 INITIAL STATE:
  Upper rail: head=(5,42), 2 segs (5,42),(11,42), rot=0 horizontal.
  Targets above y=53: (29,6)=14, (29,12)=9, (29,18)=8, (29,24)=12.
  Lower ref: 5 segs (17..41, 56). jdojcthkf[ref] has 4 entries.
  Required upper_visit colors = [8, 12, 9, 14].
  Tracks for UP/DOWN slides at (7, {8,14,20,26,32,38}). Head can reach
  y in {6, 12, 18, 24, 30, 36, 42}. Play area x in [11,53), y in [6,48).

PROBLEM: all 4 upper targets are at x=29, pinned for y<=24. Visit list
is built by scanning segments left-to-right — since all segments are on
one row, the visit list contains targets on that row in x order. To have
a 4-entry visit list we need 4 targets on ONE row at distinct x. We must
push targets below y=24 first (via slide-chain DOWN) then move them
horizontally (via extension below the pin).

PUSHING DOWN: from y=6 with rail crossing x=29, slide DOWN chain-pushes
all targets in the x=29 column by +6. Repeated slides push the column down
but keep all targets at x=29. After dn3 (starting from stack [6,12,18,24]):
targets at (29,24)=14, (29,30)=9, (29,36)=8, (29,42)=12 (colors shift).
Target at (29,24) still pinned. dn4 fails because bottom target (29,42)
can't move to (29,48) — out of bounds (play area y_max=48).

FREE RANGE: targets at y=30,36,42 are now below the pin rail and can be
pushed horizontally via extensions or dragged by retractions at that row.

FINAL ARRANGEMENT ORDER: ref=[8,12,9,14]. Must get 4 targets on one row
with colors left-to-right = [8,12,9,14]. Candidates: row y=24 (where target
14 lives pinned) or row y=30/36/42 for all four (requires dragging 14 off
x=29 which requires un-pinning — impossible unless target 14 is pushed
down to y=30 first, freeing it from the pin).

SUB-PROBLEM: get target 14 to y>=30. At init target 14 is at (29,6).
Slide-chain-down moves the whole column together. After dn3 target 14
(originally (29,6)) sits at (29,24) — still pinned! The chain can't go
further because y=48 is the bottom. So target 14 is STUCK at (29,24).

ALTERNATE APPROACH: don't slide the full column at once. Use shorter rail
or partial slides so target 14 ends up FURTHER DOWN than the others.
  - At y=6 with short rail NOT crossing x=29: slide down to y=12, rail
    arrives at y=12 without touching targets. Target 9 still at (29,12),
    target 14 still at (29,6).
  - From y=12 with short rail, extend across x=29: fails to move target 9
    (pinned), passes through. Not useful for positioning.
  - The only way to DIFFERENTIALLY move targets is for some to be at
    segment positions during a slide while others are not. Since all 4
    targets are at x=29 and segments are on one row, only ONE target is
    at a segment position per slide (the one at head.y). So sliding
    moves ONE target per slide.
  - Slide from y=6 (target 14 alone moved to y=12), but target 9 is
    already at y=12 -> chain-push 9 to y=18 -> chain 8 to y=24 -> chain
    12 to y=30. So in a single slide, we move 14 and displace the
    others downward by exactly the spacing.

KEY INSIGHT (unverified): the chain-push length depends on which
targets are CONSECUTIVE in the direction. If there's a gap in the
target column, the chain breaks. But here they start at 6,12,18,24 —
no gap — so all 4 always chain together.

To CREATE a gap: push 14 down while leaving 12 alone. This requires
the bottom target (12 at 24) to be blocked (pinned). And it IS pinned
at y=24! Its dest (29,30) has no pin (below y=24), so it can move.
Hmm — y=24 is the LAST row with pin rail segments. pptqisyill(29,24,
False, hstak=seg) finds the vertical seg at (29,24). But then when
moving (0,+6): (segment_dir_x==0) vs (move_x==0): True vs True =
False -> not return False. Move proceeds. So target 12 DOES move.

THE VERTICAL RAIL DOES NOT BLOCK VERTICAL PUSHES (it's parallel).
So slide-chain moves everything together. Can't create gap this way.

=============================================================================
L2 STATUS: blocked by inability to separate the 4 x=29 targets by row.
  Current best hypothesis: use RETRACTION-DRAG after slides to peel one
  target off. Need seg at target's position in retraction direction (-x)
  and target NOT pinned. After sliding down so target is at (29,y>24),
  extend rail at y to place a seg at (29,y), then retract — target drags
  left to (23,y), escaping x=29. Now we have one target at x<29 and
  others at x=29. Repeat for each and align final row.

  CHOREOGRAPHY (~30+ moves per target, 4 targets): budget 196 may be tight.
  Not yet attempted. Next step: hand-construct and verify in engine.
=============================================================================
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
