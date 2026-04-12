#!/usr/bin/env python3
"""
su15 L9 smart solver — efficient beam search with game state caching.

Key insight: reuse game states between beam expansions instead of replaying
from scratch each time. Use deepcopy of game objects.
"""

import sys
import copy
import time

sys.path.insert(0, 'environment_files/su15/4c352900')
from su15 import Su15, GameAction, ActionInput


def click(game, x, y):
    action = ActionInput(id=GameAction.ACTION6.value, data={'x': x, 'y': y})
    game.perform_action(action)


def state_key(game):
    fruits = tuple(sorted((game.amnmgwpkeb.get(f, 0), f.x, f.y) for f in game.hmeulfxgy))
    enemies = tuple(sorted((game.hirdajbmj.get(e, '?'), e.x, e.y) for e in game.peiiyyzum))
    return (fruits, enemies)


def score_state(game):
    """
    Score based on how close we are to winning.

    Goals: tier-4 on goal, tier-2 on goal, enemy-3 on goal
    Goal bounds for centers:
      G1 (11,41): cx in [7,16), cy in [37,46)
      G2 (53,55): cx in [49,58), cy in [51,60)
      G3 (11,55): cx in [7,16), cy in [51,60)
    """
    goal_centers = [(11,41), (53,55), (11,55)]

    s = 0.0

    # Count what we have
    has_t4 = False
    has_t2 = False
    has_e3 = False
    t4_on_goal = False
    t2_on_goal = False
    e3_on_goal = False

    for f in game.hmeulfxgy:
        t = game.amnmgwpkeb.get(f, 0)
        cx, cy = game.qmecbepbyz(f)
        on_goal = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)

        if t == 4:
            has_t4 = True
            if on_goal:
                t4_on_goal = True
                s += 100
            else:
                # Distance to nearest goal
                min_d = min(abs(cx - gx) + abs(cy - gy) for gx, gy in goal_centers)
                s += max(0, 30 - min_d)
        elif t == 2:
            has_t2 = True
            if on_goal:
                t2_on_goal = True
                s += 100
            else:
                min_d = min(abs(cx - gx) + abs(cy - gy) for gx, gy in goal_centers)
                s += max(0, 30 - min_d)
        elif t == 5:
            s += 10  # tier-5 exists but needs knockdown
        elif t >= 3:
            s += 5  # higher tier exists but wrong value
        # tier-0 and tier-1 less useful

    for e in game.peiiyyzum:
        et = game.hirdajbmj.get(e, '?')
        lvl = game.cqjufwqvag(et)
        cx, cy = game.qmecbepbyz(e)
        on_goal = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)

        if lvl == 3:
            has_e3 = True
            if on_goal:
                e3_on_goal = True
                s += 100
            else:
                min_d = min(abs(cx - gx) + abs(cy - gy) for gx, gy in goal_centers)
                s += max(0, 50 - min_d)
        elif lvl == 2:
            s += 15
        # type-1 is base, less useful

    # Big bonus for having all three items
    if has_t4: s += 20
    if has_t2: s += 20
    if has_e3: s += 30

    # Penalty for too many enemies (means merges haven't happened)
    n_enemies = len(game.peiiyyzum)
    if n_enemies > 2:
        s -= (n_enemies - 2) * 5

    # Penalize low remaining steps
    steps = game.step_counter_ui.current_steps
    if steps < 5:
        s -= 50

    return s, (t4_on_goal, t2_on_goal, e3_on_goal)


def make_candidates(game):
    """Generate click candidates based on current state."""
    cands = set()

    # Grid positions (coarse)
    for x in range(4, 61, 4):
        for y in range(12, 60, 4):
            cands.add((x, y))

    # Near fruits
    for f in game.hmeulfxgy:
        cx, cy = game.qmecbepbyz(f)
        for dx in range(-6, 7, 2):
            for dy in range(-6, 7, 2):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < 64 and 10 <= ny <= 62:
                    cands.add((nx, ny))

    # Near enemies
    for e in game.peiiyyzum:
        cx, cy = game.qmecbepbyz(e)
        for dx in range(-6, 7, 2):
            for dy in range(-6, 7, 2):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < 64 and 10 <= ny <= 62:
                    cands.add((nx, ny))

    # Goal centers
    for gx, gy in [(11,41), (53,55), (11,55)]:
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                nx, ny = gx + dx, gy + dy
                if 10 <= ny <= 62:
                    cands.add((nx, ny))

    return sorted(cands)


def beam_search():
    print("=== Smart beam search ===")

    BEAM_WIDTH = 300
    MAX_DEPTH = 30

    # Initial game state
    game0 = Su15()
    game0.set_level(8)

    # Beam entries: (score, game_copy, click_sequence)
    s0, _ = score_state(game0)
    beam = [(s0, game0, [])]

    best_ever_score = s0
    best_ever_seq = []

    t0 = time.time()

    for depth in range(MAX_DEPTH):
        if time.time() - t0 > 600:
            print("Timeout!")
            break

        next_beam = []
        seen_states = set()

        # Use adaptive candidates: for depth 0 start with merge positions
        if depth == 0:
            # First click: merge tier-1s or pull tier-5 or push enemies
            candidates = []
            # Tier-1 merge positions (sweet spots)
            for x in range(12, 28):
                for y in range(44, 58):
                    candidates.append((x, y))
            # Also try some enemy-push positions
            for x in range(4, 61, 4):
                for y in range(12, 60, 4):
                    candidates.append((x, y))
            candidates = list(set(candidates))
        else:
            candidates = None  # will use make_candidates per state

        for score, game, clicks in beam:
            if game.level_index != 8:
                print(f"\n*** WON! clicks={clicks}")
                return clicks
            if game.step_counter_ui.current_steps <= 0:
                continue

            cands = candidates if candidates else make_candidates(game)

            for cx, cy in cands:
                game2 = copy.deepcopy(game)
                click(game2, cx, cy)

                if game2.level_index != 8:
                    result = clicks + [(cx, cy)]
                    print(f"\n*** SOLUTION at depth {depth+1}: {result}")
                    return result

                if game2.step_counter_ui.current_steps <= 0:
                    continue

                sk = state_key(game2)
                if sk in seen_states:
                    continue
                seen_states.add(sk)

                s, flags = score_state(game2)
                next_beam.append((s, game2, clicks + [(cx, cy)]))

        if not next_beam:
            print(f"  Depth {depth+1}: dead end")
            break

        next_beam.sort(key=lambda x: -x[0])
        beam = next_beam[:BEAM_WIDTH]

        # Report
        best_s, best_g, best_c = beam[0]
        if best_s > best_ever_score:
            best_ever_score = best_s
            best_ever_seq = best_c

        items = []
        for f in best_g.hmeulfxgy:
            t = best_g.amnmgwpkeb.get(f, 0)
            cx, cy = best_g.qmecbepbyz(f)
            og = any(best_g.epvtlqtczz(cx, cy, g) for g in best_g.rqdsgrklq)
            items.append(f't{t}({cx},{cy}){"*" if og else ""}')
        for e in best_g.peiiyyzum:
            et = best_g.hirdajbmj.get(e, '?')
            lvl = best_g.cqjufwqvag(et)
            cx, cy = best_g.qmecbepbyz(e)
            og = any(best_g.epvtlqtczz(cx, cy, g) for g in best_g.rqdsgrklq)
            items.append(f'e{lvl}({cx},{cy}){"*" if og else ""}')

        elapsed = time.time() - t0
        print(f"  D{depth+1}: best={best_s:.0f}, beam={len(beam)}, "
              f"states={len(seen_states)}, "
              f"{' '.join(items)}, "
              f"steps={best_g.step_counter_ui.current_steps}, "
              f"{elapsed:.1f}s")

    print(f"\nBest ever: score={best_ever_score}, seq={best_ever_seq}")
    return None


if __name__ == "__main__":
    result = beam_search()
    if result:
        # Verify
        print(f"\nVerifying solution: {result}")
        game = Su15()
        game.set_level(8)
        for cx, cy in result:
            click(game, cx, cy)
            if game.level_index != 8:
                print(f"  WIN after click ({cx},{cy})!")
                break
