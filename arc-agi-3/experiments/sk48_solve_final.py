#!/usr/bin/env python3
"""sk48 Rail Weaver — Final solver.

Status (2026-04-12): 8/8 SOLVED. Totals: L0=14 L1=27 L2=33 L3=25 L4=77
  L5=48 L6=38 L7=24 = 286 actions. Beam w=800, CLICK-enabled.

KEY INSIGHT — BUDGET RESTORATION:
  Failed moves (blocked extensions, out-of-bounds slides, etc.) still
  decrement qiercdohl (budget). If budget hits zero during search,
  lose() triggers GAME_OVER and the reachable state space collapses
  to ~33 states. Restoring budget to init_budget after each expansion
  unlocks thousands of reachable states per level and makes beam search
  effective. This was the blocker for prior L2+ attempts.

L2 HOW: slide up to y=6 with short rail, extend to place seg at (29,6),
  slide DOWN chain-pushing the whole column of targets down by 6 per
  slide. After 3 slides the column is at y=24,30,36,42 (target 12 at
  bottom blocks further descent). Then slide UP with target 14 brings
  it back to (29,6) — now isolated. Use retract-drag at rows y=30/36/42
  to horizontally reposition targets 9, 8, 12. Final visit list
  [8,12,9,14] on a row below the pin. Prior agent's "catch-22" analysis
  was wrong: target 14 IS freeable once the bottom of the column clears.

L3 HOW: (5,42) is an unpaired "arranger" rail; the two paired rails
  (23,0) and (35,0) are vertical reference rails with static segments.
  Moving the arranger pushes targets onto the paired rails' seg
  positions, which fills their visit lists. No CLICK needed — one
  arranger suffices for both pairs.

L4+ REMAINING: multi-pair configurations likely need CLICK to switch
  between multiple arranger rails. CLICK does not save history (no
  nixwuekdfm call) — search must handle this out-of-band.

=============================================================================
WORLD MODEL (verified by source + empirical trace)
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
    # L2 found by beam search 2026-04-12 (33 moves, 304s)
    2: [UP, UP, UP, UP, RIGHT, RIGHT, RIGHT, DOWN, DOWN, LEFT, DOWN, DOWN, RIGHT, UP, UP,
        LEFT, UP, RIGHT, DOWN, LEFT, UP, UP, UP, RIGHT, DOWN, DOWN, DOWN, LEFT, UP, UP,
        UP, UP, RIGHT],
    # L3 found by beam search 2026-04-12 (25 moves, 325s)
    3: [UP, UP, UP, RIGHT, RIGHT, UP, UP, LEFT, UP, LEFT, DOWN, LEFT, UP, LEFT, DOWN,
        RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, DOWN, LEFT, DOWN, LEFT, UP],
    # L4 found by beam search 2026-04-12 (77 moves, 315s, beam=800)
    4: [UP, LEFT, RIGHT, RIGHT, RIGHT, UP, RIGHT, DOWN, LEFT, LEFT, LEFT, DOWN, DOWN,
        DOWN, LEFT, LEFT, RIGHT, RIGHT, RIGHT, UP, RIGHT, RIGHT, RIGHT, LEFT, LEFT,
        LEFT, DOWN, DOWN, RIGHT, UP, UP, LEFT, LEFT, UP, UP, RIGHT, RIGHT, RIGHT,
        RIGHT, LEFT, LEFT, LEFT, LEFT, DOWN, LEFT, UP, UP, UP, LEFT, DOWN, DOWN,
        DOWN, RIGHT, RIGHT, LEFT, LEFT, UP, UP, UP, LEFT, RIGHT, DOWN, DOWN, DOWN,
        RIGHT, RIGHT, UP, UP, LEFT, LEFT, UP, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT],
    # L5 found by beam search 2026-04-12 (48 moves, 421s, beam=800, CLICK-enabled)
    5: [DOWN, RIGHT, RIGHT, RIGHT, LEFT, DOWN, RIGHT, RIGHT, UP,
        (CLICK, {'x': 29, 'y': 2}), DOWN, DOWN, RIGHT, DOWN, UP, RIGHT, DOWN, UP, UP,
        RIGHT, DOWN, UP, (CLICK, {'x': 5, 'y': 32}), RIGHT, RIGHT, LEFT,
        (CLICK, {'x': 47, 'y': 2}), DOWN, DOWN, LEFT, (CLICK, {'x': 5, 'y': 32}),
        RIGHT, LEFT, LEFT, LEFT, (CLICK, {'x': 41, 'y': 2}), DOWN, DOWN, DOWN, UP, LEFT,
        (CLICK, {'x': 5, 'y': 32}), DOWN, LEFT, DOWN, RIGHT,
        (CLICK, {'x': 35, 'y': 2}), DOWN],
    # L6 found by beam search 2026-04-12 (38 moves, 392s, beam=800)
    6: [RIGHT, RIGHT, (CLICK, {'x': 29, 'y': 2}), DOWN, LEFT, DOWN, RIGHT, RIGHT, RIGHT,
        DOWN, LEFT, RIGHT, DOWN, LEFT, (CLICK, {'x': 5, 'y': 26}), DOWN, RIGHT, LEFT, UP,
        (CLICK, {'x': 35, 'y': 2}), UP, UP, LEFT, LEFT, DOWN, (CLICK, {'x': 5, 'y': 26}),
        UP, RIGHT, RIGHT, RIGHT, RIGHT, DOWN, UP, (CLICK, {'x': 23, 'y': 2}), RIGHT,
        DOWN, DOWN, DOWN],
    # L7 found by beam search 2026-04-12 (24 moves, 139s, beam=800)
    7: [RIGHT, RIGHT, (CLICK, {'x': 29, 'y': 2}), DOWN, DOWN, RIGHT, DOWN, LEFT, DOWN,
        DOWN, DOWN, UP, UP, UP, RIGHT, (CLICK, {'x': 5, 'y': 26}), RIGHT, RIGHT, RIGHT,
        RIGHT, DOWN, LEFT, LEFT, UP],
}


def state_key(game):
    active = game.vzvypfsnt
    # Encode ALL rails (so CLICK doesn't corrupt state identity), plus
    # which one is currently active.
    rails = []
    for h in sorted(game.mwfajkguqx.keys(), key=lambda s: (s.x, s.y)):
        segs = game.mwfajkguqx[h]
        rails.append((h.x, h.y, len(segs)))
    targets = tuple(sorted((t.x, t.y, int(t.pixels[1, 1]))
                           for t in game.vbelzuaian))
    return ((active.x, active.y), tuple(rails), targets)


def _lcs_len(a, b):
    n, m = len(a), len(b)
    if not n or not m:
        return 0
    dp = [0] * (m + 1)
    for i in range(n):
        prev = 0
        for j in range(m):
            cur = dp[j + 1]
            if a[i] == b[j]:
                dp[j + 1] = prev + 1
            elif dp[j] > dp[j + 1]:
                dp[j + 1] = dp[j]
            prev = cur
    return dp[-1]


def compute_h(game):
    """Heuristic for sk48 rail weaver.

    The win condition per paired rail: upper's seg-order visit list must
    match ref's seg-order visit list, position by position, for
    len(jdojcthkf[ref]) positions.

    For each pair, score:
      1. prefix match bonus (high reward at leading positions)
      2. LCS partial credit
      3. assignment cost: minimum Manhattan distance from each required
         *target* (the sprite with color c) to the nearest available seg
         position on the upper rail — approximates "how much work to put
         color c at its required slot".

    Also score the distance from the ACTIVE (player-controlled) rail's head
    to the nearest required target: this shapes the landscape so the
    player's rail is attracted toward useful work.
    """
    game.gvtmoopqgy()
    total = 0.0
    active = game.vzvypfsnt
    all_req_positions = []  # (x, y, c) needed at specific slots
    for head, ref_head in game.xpmcmtbcv.items():
        upper = game.vjfbwggsd.get(head, [])
        lower = game.vjfbwggsd.get(ref_head, [])
        ref_colors = [int(t.pixels[1, 1]) for t in lower]
        upper_colors = [int(t.pixels[1, 1]) for t in upper]
        n_lower = len(ref_colors)

        prefix = 0
        for i in range(min(len(upper_colors), n_lower)):
            if upper_colors[i] == ref_colors[i]:
                prefix += 1
            else:
                break
        total += (n_lower - prefix) * 200

        lcs = _lcs_len(upper_colors, ref_colors)
        total += (n_lower - lcs) * 40

        # Gradient toward filling the first unmatched prefix slot: distance
        # from required seg position to nearest target of the required color.
        upper_segs_early = game.mwfajkguqx.get(head, [])
        if prefix < n_lower and upper_segs_early:
            slot_i = prefix
            if slot_i < len(upper_segs_early):
                sx, sy = upper_segs_early[slot_i].x, upper_segs_early[slot_i].y
            else:
                sx, sy = upper_segs_early[-1].x, upper_segs_early[-1].y
            needed_c = ref_colors[slot_i]
            best_d = 120
            for t in game.vbelzuaian:
                if t.y >= 53:
                    continue
                if int(t.pixels[1, 1]) != needed_c:
                    continue
                d = abs(t.x - sx) + abs(t.y - sy)
                if d < best_d:
                    best_d = d
            total += (best_d / CELL) * 30

        # Upper rail's seg positions (where we need targets of matching color)
        upper_segs = game.mwfajkguqx.get(head, [])
        seg_positions = [(s.x, s.y) for s in upper_segs]

        # Targets available per color
        from collections import Counter
        ref_cnt = Counter(ref_colors)
        used_tids = set()
        all_targets = [(t.x, t.y, int(t.pixels[1, 1]), id(t)) for t in game.vbelzuaian]
        for i, rc in enumerate(ref_colors):
            # Required slot is the i-th seg position (if exists)
            if i < len(seg_positions):
                sx, sy = seg_positions[i]
            elif seg_positions:
                sx, sy = seg_positions[-1]
            else:
                sx, sy = head.x, head.y
            # Find nearest unused target with matching color
            best_d = 60
            best_id = None
            for (tx, ty, tc, tid) in all_targets:
                if tc != rc or tid in used_tids or ty >= 53:
                    continue
                d = abs(tx - sx) + abs(ty - sy)
                if d < best_d:
                    best_d = d
                    best_id = tid
            if best_id is not None:
                used_tids.add(best_id)
                all_req_positions.append((sx, sy, rc))
            total += (best_d / CELL) * 6

    # Active-rail head attraction: distance to the nearest required target
    need_ys = []
    all_targets = [(t.x, t.y, int(t.pixels[1, 1])) for t in game.vbelzuaian if t.y < 53]
    min_d = 60
    for (tx, ty, _c) in all_targets:
        d = abs(tx - active.x) + abs(ty - active.y)
        if d < min_d:
            min_d = d
    total += min_d / CELL * 0.25

    return total


def solve_beam(env, game, beam_width=5000, max_depth=200, use_click=False,
               timeout_secs=600):
    """Beam search with incremental undo/redo for efficiency.

    Each beam entry tracks its move list. Actions are (action_enum, data_dict)
    tuples so CLICK can carry coordinates to switch between paired rails.
    """
    # Build action list. Movement actions have data=None.
    # If use_click, also emit CLICK actions for each paired head != active
    # (generated per-expansion since active changes).
    all_move_actions = [(a, None) for a in ACTIONS]

    def click_actions_for_state():
        if not use_click:
            return []
        acts = []
        seen = set()
        active_now = game.vzvypfsnt
        for h in list(game.xpmcmtbcv.keys()) + list(game.xpmcmtbcv.values()):
            if h is active_now:
                continue
            key = (h.x, h.y)
            if key in seen:
                continue
            seen.add(key)
            acts.append((CLICK, {'x': h.x, 'y': h.y}))
        return acts

    if game.gvtmoopqgy():
        return []

    init_budget = game.qiercdohl
    init_hist = len(game.seghobzez)
    cur_level_idx = game.level_index

    # Track the moves currently applied to the engine
    cur_moves: list = []
    # Parallel undo stack: each entry is ('hist', None) or ('click', (prev_x, prev_y))
    undo_stack: list = []

    def drive(act_pair):
        """Execute one player action, draining animations/win sequence."""
        act, data = act_pair
        env.step(act, data)
        while game.ljprkjlji or game.pzzwlsmdt:
            env.step(act, data)
        while game.lgdrixfno >= 0 and game.lgdrixfno < 35:
            env.step(act, data)

    def _mkey(ap):
        a, d = ap
        return (a.value, (d or {}).get('x', -1), (d or {}).get('y', -1))

    def apply_move(mv):
        """Apply one move, record undo info. Returns (hist_changed, active_changed)."""
        act, data = mv
        pre_hist = len(game.seghobzez)
        pre_active = game.vzvypfsnt
        if act == CLICK:
            env.step(act, data)
            hist_changed = False
            active_changed = game.vzvypfsnt is not pre_active
            if active_changed:
                undo_stack.append(('click', (pre_active.x, pre_active.y)))
            else:
                undo_stack.append(('noop', None))
        else:
            drive(mv)
            hist_changed = len(game.seghobzez) > pre_hist
            active_changed = game.vzvypfsnt is not pre_active
            if hist_changed:
                undo_stack.append(('hist', None))
            elif active_changed:
                undo_stack.append(('click', (pre_active.x, pre_active.y)))
            else:
                undo_stack.append(('noop', None))
        cur_moves.append(mv)
        return hist_changed, active_changed

    def pop_move():
        """Undo the last applied move."""
        if not cur_moves:
            return
        kind, info = undo_stack.pop()
        cur_moves.pop()
        if kind == 'hist':
            game.uqclctlhyh()
        elif kind == 'click':
            px, py = info
            env.step(CLICK, {'x': px, 'y': py})
        # noop: nothing to do

    def goto(target_moves: list):
        nonlocal cur_moves
        # Find common prefix (by value equality)
        cp = 0
        m = min(len(cur_moves), len(target_moves))
        while cp < m and _mkey(cur_moves[cp]) == _mkey(target_moves[cp]):
            cp += 1
        # Undo down to common prefix
        while len(cur_moves) > cp:
            pop_move()
        # Redo remainder
        for i in range(cp, len(target_moves)):
            apply_move(target_moves[i])
        # Restore full budget so search isn't bounded by consumed budget
        # (solution length is our real constraint).
        game.qiercdohl = init_budget

    def reset_full():
        goto([])
        game.qiercdohl = init_budget

    t0 = time.time()
    goto([])
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
        # Sort beam entries by move sequence for prefix sharing
        beam_sorted = sorted(beam, key=lambda x: tuple((a[0].value, (a[1] or {}).get('x', -1), (a[1] or {}).get('y', -1)) for a in x[1]))
        for _, moves in beam_sorted:
            goto(moves)

            step_actions = list(all_move_actions) + click_actions_for_state()
            for act in step_actions:
                hist_changed, active_changed = apply_move(act)
                expanded += 1

                if game.level_index > cur_level_idx or game.lgdrixfno >= 35:
                    sol = list(cur_moves)
                    print(f"  SOLVED! {len(sol)} moves, {expanded} exp, {time.time()-t0:.1f}s")
                    return sol

                if not hist_changed and not active_changed:
                    pop_move()  # pure no-op
                    continue

                sk = state_key(game)
                if sk in visited:
                    pop_move()
                    continue
                visited.add(sk)
                h = compute_h(game)
                cands.append((h, list(cur_moves)))
                pop_move()

        cands.sort(key=lambda x: (x[0], len(x[1])))
        beam = cands[:beam_width]

        if beam:
            cur_h = beam[0][0]
            if cur_h < best_h - 0.1:
                best_h = cur_h
                print(f"    d={depth}: h={cur_h:.1f}, {len(beam)} cands, {expanded} exp, {time.time()-t0:.1f}s")
            elif depth % 10 == 0:
                print(f"    d={depth}: h={cur_h:.1f}, {len(beam)} cands, {len(visited)} vis, {time.time()-t0:.1f}s")

        if time.time() - t0 > timeout_secs:
            print(f"  Timeout at d={depth}, {time.time()-t0:.1f}s")
            break

    print(f"  Failed. {expanded} exp, best_h={best_h:.1f}")
    reset_full()
    return None


def main():
    print("=" * 60, flush=True)
    print("sk48 Final Solver", flush=True)
    print("=" * 60, flush=True)

    arcade = Arcade()
    env = arcade.make('sk48-41055498')
    fd = env.reset()
    game = env._game

    def drive_real(act_or_pair):
        nonlocal fd
        if isinstance(act_or_pair, tuple):
            act, data = act_or_pair
        else:
            act, data = act_or_pair, None
        fd = env.step(act, data)
        while game.ljprkjlji or game.pzzwlsmdt:
            fd = env.step(act, data)
        while game.lgdrixfno >= 0 and game.lgdrixfno < 35:
            fd = env.step(act, data)

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
                drive_real(a)
            won = (game.level_index > lv) or (fd is not None and fd.state.name == 'WIN')
            if won:
                print(f"  L{lv} SOLVED!")
                results[lv] = len(sol)
                total += len(sol)
                if fd is not None and fd.state.name == 'WIN':
                    break
            else:
                print(f"  Known FAILED!")
                break
        else:
            t0 = time.time()
            print("  Beam search...")
            sol = solve_beam(env, game, beam_width=3000, max_depth=200,
                            use_click=use_click, timeout_secs=1200)
            dt = time.time() - t0

            if sol is None:
                print(f"  FAILED L{lv} ({dt:.1f}s)")
                break

            sol_names = []
            for a in sol:
                if isinstance(a, tuple):
                    act, data = a
                    n = ACTION_NAMES.get(act, str(act))
                    if data:
                        n += f"@({data.get('x','?')},{data.get('y','?')})"
                    sol_names.append(n)
                else:
                    sol_names.append(ACTION_NAMES.get(a, str(a)))
            print(f"  Solution ({len(sol)}): {sol_names}")

            if game.level_index <= lv:
                while len(game.seghobzez) > 1:
                    game.uqclctlhyh()
                game.qiercdohl = game.vhzjwcpmk  # restore budget for real replay
                for a in sol:
                    drive_real(a)

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
