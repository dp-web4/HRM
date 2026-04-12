#!/usr/bin/env python3
"""
su15 L9 solver v2 — guided search with small candidate sets per phase.

Phase 1: Merge tier-1s -> tier-2 on goal 3 (single click)
Phase 2: Pull tier-5 toward goal 1 (2-3 clicks)
Phase 3: Get enemy to knock tier-5 -> tier-4 on goal 1
Phase 4: Merge enemies to type-2 pairs (2 clicks)
Phase 5: Position type-2 enemies near goal 2
Phase 6: Merge type-2 -> type-3 on goal 2 (final click)

Within each phase, search small candidate sets.
"""

import sys
import time
from itertools import product

sys.path.insert(0, 'environment_files/su15/4c352900')
from su15 import Su15, GameAction, ActionInput


def click(game, x, y):
    action = ActionInput(id=GameAction.ACTION6.value, data={'x': x, 'y': y})
    game.perform_action(action)


def replay(clicks):
    game = Su15()
    game.set_level(8)
    for cx, cy in clicks:
        if game.level_index != 8:
            return game
        click(game, cx, cy)
    return game


def pstate(game, label=''):
    items = []
    for f in game.hmeulfxgy:
        t = game.amnmgwpkeb.get(f, 0)
        cx, cy = game.qmecbepbyz(f)
        og = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)
        items.append(f't{t}({cx},{cy}){"*" if og else ""}')
    for e in game.peiiyyzum:
        et = game.hirdajbmj.get(e, '?')
        lvl = game.cqjufwqvag(et)
        cx, cy = game.qmecbepbyz(e)
        og = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)
        items.append(f'e{lvl}({cx},{cy}){"*" if og else ""}')
    print(f'{label} [{game.step_counter_ui.current_steps}s] {" ".join(items)}')


def check_items(game):
    """Check what items we have and their positions."""
    result = {}
    for f in game.hmeulfxgy:
        t = game.amnmgwpkeb.get(f, 0)
        cx, cy = game.qmecbepbyz(f)
        on_goal = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)
        result[f't{t}'] = (cx, cy, on_goal)
    for e in game.peiiyyzum:
        et = game.hirdajbmj.get(e, '?')
        lvl = game.cqjufwqvag(et)
        cx, cy = game.qmecbepbyz(e)
        on_goal = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)
        key = f'e{lvl}'
        idx = 1
        while f'{key}_{idx}' in result:
            idx += 1
        result[f'{key}_{idx}'] = (cx, cy, on_goal)
    return result


def exhaustive_search():
    """
    Try all possible click sequences up to N clicks.
    With very small candidate sets tailored to each step.
    """
    print("=== Exhaustive phase-based search ===")
    t0 = time.time()

    # Phase 1 candidates: tier-1 merge positions that put tier-2 on a goal
    # From earlier exploration: (15,52) and (15,53) put tier-2 on goal 3
    c1_candidates = [(15,52), (15,53)]

    # Phase 2 candidates: pull tier-5 toward goal 1
    # tier-5 starts at (35,48) center (38,51)
    # Goal 1 center (11,41)
    # Need 2-4 pull clicks to move tier-5 ~30px
    # Each pull within vacuum radius 8 of tier-5
    c2_candidates = []
    for x in range(20, 42, 2):
        for y in range(34, 52, 2):
            c2_candidates.append((x, y))

    # After some pulls, we need enemy collision -> knockdown
    # c3: positions where enemies converge and hit tier-5/tier-4
    c3_candidates = []
    for x in range(4, 61, 4):
        for y in range(12, 60, 4):
            c3_candidates.append((x, y))

    found = []
    tested = 0

    # Strategy: fixed C1, then search variable number of pulls + chases
    for c1 in c1_candidates:
        print(f"\nC1={c1}")
        game1 = replay([c1])
        if game1.level_index != 8:
            found.append([c1])
            continue

        # Try 1-3 pull clicks followed by 1-6 chase/position clicks
        for n_pulls in range(1, 4):
            for pulls in product(c2_candidates, repeat=n_pulls):
                seq = [c1] + list(pulls)
                game_p = replay(seq)
                if game_p.level_index != 8:
                    found.append(seq)
                    print(f"  WON with {seq}")
                    continue
                if game_p.step_counter_ui.current_steps <= 0:
                    continue
                tested += 1

                # Check: do we have tier-4 yet?
                items = check_items(game_p)
                has_t4 = any(k.startswith('t4') for k in items)
                has_t2 = any(k.startswith('t2') for k in items)

                if has_t4 and has_t2:
                    # Both fruits exist! Now search for positioning + enemy merge
                    t4_pos = [v for k,v in items.items() if k.startswith('t4')][0]
                    t2_pos = [v for k,v in items.items() if k.startswith('t2')][0]

                    if t2_pos[2]:  # t2 still on goal
                        # Search for remaining clicks to position t4 and merge enemies
                        for n_rest in range(1, min(8, game_p.step_counter_ui.current_steps + 1)):
                            for rest in product(c3_candidates, repeat=n_rest):
                                full = seq + list(rest)
                                if len(full) > 48:
                                    continue
                                game_r = replay(full)
                                if game_r.level_index != 8:
                                    found.append(full)
                                    print(f"  *** SOLUTION: {full}")
                                    return full
                                tested += 1
                                if tested % 10000 == 0:
                                    elapsed = time.time() - t0
                                    print(f"  {tested} tested, {elapsed:.0f}s")
                                if time.time() - t0 > 600:
                                    break
                            if found or time.time() - t0 > 600:
                                break

                if time.time() - t0 > 600:
                    break
            if time.time() - t0 > 600:
                break

    print(f"\nTotal tested: {tested}, time: {time.time()-t0:.0f}s")
    if found:
        print(f"Solutions: {found}")
    return found[0] if found else None


def guided_search():
    """
    More guided: explore specific promising sequences.
    """
    print("=== Guided exploration ===")

    # From my earlier experiments:
    # Sequence (15,52), (30,44), (22,38), (16,36), (12,38), (10,40) gives:
    #   t4(10,40)* t2(15,52)* e2(32,42) e2(12,42)*
    #   But enemy overlaps tier-4!

    # What if I change the last couple clicks?
    # After (15,52),(30,44),(22,38),(16,36):
    #   t4(29,53) t2(15,52)* e1(37,31) e1(7,36) e1(30,41) e1(40,47)
    # Wait, actually C5 at (12,38) already did the knockdown in the earlier test.
    # Let me re-examine.

    # Let me try: after getting tier-4, pull it toward goal WITHOUT catching enemies

    # First, explore what happens with different C5 after (15,52),(30,44),(22,38),(16,36)
    prefix = [(15,52), (30,44), (22,38), (16,36)]
    game_prefix = replay(prefix)
    print("After prefix", prefix)
    pstate(game_prefix, '  State')

    # The tier-5 is at sprite position. Let me check:
    items = check_items(game_prefix)
    print(f"  Items: {items}")

    # Now try many C5 options
    print("\nSearching C5 options...")
    good_c5 = []
    for x in range(0, 64, 2):
        for y in range(10, 63, 2):
            game5 = replay(prefix + [(x, y)])
            if game5.level_index != 8:
                print(f"  WON with C5=({x},{y})")
                return prefix + [(x, y)]
            items5 = check_items(game5)
            has_t4 = any(k.startswith('t4') for k in items5)
            has_t2 = any(k.startswith('t2') and v[2] for k,v in items5.items())
            has_e2 = sum(1 for k in items5 if k.startswith('e2'))
            n_e1 = sum(1 for k in items5 if k.startswith('e1'))
            n_enemies = sum(1 for k in items5 if k.startswith('e'))

            if has_t4 and has_t2:
                t4_info = [v for k,v in items5.items() if k.startswith('t4')][0]
                # Is tier-4 safe (no enemy overlapping)?
                t4_safe = True
                for e in game5.peiiyyzum:
                    for f in game5.hmeulfxgy:
                        if game5.amnmgwpkeb.get(f, 0) == 4:
                            if game5.rukauvoumh(e, f):
                                t4_safe = False
                if t4_safe:
                    good_c5.append((x, y, items5, has_e2, n_e1, t4_info[2], game5.step_counter_ui.current_steps))

    print(f"\nGood C5 options (t4+t2 exist, t4 safe): {len(good_c5)}")
    # Sort by: t4 on goal, more e2 merges done, more steps
    good_c5.sort(key=lambda x: (x[5], x[3], x[6]), reverse=True)
    for x, y, items, ne2, ne1, t4og, steps in good_c5[:20]:
        print(f"  C5=({x},{y}): t4_on_goal={t4og}, e2={ne2}, e1={ne1}, steps={steps}")
        # Show full state
        game5 = replay(prefix + [(x, y)])
        pstate(game5, f'    ')

    if not good_c5:
        print("No good C5 options found with this prefix.")
        # Try different prefix
        return None

    # For each good C5, search for C6+ to complete
    print("\nSearching C6+ for best C5 options...")
    for x5, y5, items5, ne2, ne1, t4og, steps5 in good_c5[:10]:
        prefix5 = prefix + [(x5, y5)]
        game5 = replay(prefix5)

        # Targeted C6 search: near entities in current state
        c6_cands = set()
        for f in game5.hmeulfxgy:
            cx, cy = game5.qmecbepbyz(f)
            for dx in range(-8, 9, 2):
                for dy in range(-8, 9, 2):
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < 64 and 10 <= ny <= 62:
                        c6_cands.add((nx, ny))
        for e in game5.peiiyyzum:
            cx, cy = game5.qmecbepbyz(e)
            for dx in range(-8, 9, 2):
                for dy in range(-8, 9, 2):
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < 64 and 10 <= ny <= 62:
                        c6_cands.add((nx, ny))
        # Goals
        for gx, gy in [(11,41), (53,55), (11,55)]:
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    c6_cands.add((gx+dx, gy+dy))
        # Corners
        c6_cands.update([(4,60), (60,60), (4,12), (60,12)])

        c6_cands = sorted(c6_cands)

        for x6, y6 in c6_cands:
            game6 = replay(prefix5 + [(x6, y6)])
            if game6.level_index != 8:
                return prefix5 + [(x6, y6)]

            # Quick check: any improvement?
            items6 = check_items(game6)
            has_t4 = any(k.startswith('t4') for k in items6)
            has_t2 = any(k.startswith('t2') and v[2] for k,v in items6.items())
            has_e3 = any(k.startswith('e3') for k in items6)

            if has_t4 and has_t2 and has_e3:
                # All three items exist! Check if e3 on goal
                e3_info = [v for k,v in items6.items() if k.startswith('e3')][0]
                t4_info = [v for k,v in items6.items() if k.startswith('t4')][0]
                if e3_info[2] and t4_info[2]:
                    print(f"  CLOSE! {prefix5 + [(x6,y6)]}")
                    pstate(game6, f'    ')

                # Try a few more clicks to position
                c7_cands = set()
                for ent in list(game6.hmeulfxgy) + list(game6.peiiyyzum):
                    cx, cy = game6.qmecbepbyz(ent)
                    for dx in range(-6, 7, 2):
                        for dy in range(-6, 7, 2):
                            nx, ny = cx+dx, cy+dy
                            if 0 <= nx < 64 and 10 <= ny <= 62:
                                c7_cands.add((nx, ny))
                for gx, gy in [(11,41), (53,55), (11,55)]:
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            c7_cands.add((gx+dx, gy+dy))

                for x7, y7 in sorted(c7_cands):
                    game7 = replay(prefix5 + [(x6,y6), (x7,y7)])
                    if game7.level_index != 8:
                        return prefix5 + [(x6,y6), (x7,y7)]

    return None


def wide_search():
    """
    Wider search: try many different C1-C4 prefixes, not just (15,52),(30,44),(22,38),(16,36).
    """
    print("=== Wide prefix search ===")
    t0 = time.time()

    # C1: tier-1 merge positions
    c1_opts = [(15,52), (15,53), (16,52), (16,53), (17,50), (18,50), (19,50), (20,50)]

    # C2-C3-C4: pull positions for tier-5, spaced 2px
    pull_opts = []
    for x in range(12, 44, 3):
        for y in range(30, 56, 3):
            pull_opts.append((x, y))

    print(f"C1 opts: {len(c1_opts)}, pull opts: {len(pull_opts)}")

    best = None
    best_score = 0
    tested = 0

    for c1 in c1_opts:
        for c2 in pull_opts:
            for c3 in pull_opts:
                prefix = [c1, c2, c3]
                game = replay(prefix)
                tested += 1
                if game.level_index != 8:
                    print(f"  WON with {prefix}")
                    return prefix
                if game.step_counter_ui.current_steps <= 0:
                    continue

                items = check_items(game)
                has_t4 = any(k.startswith('t4') for k in items)
                has_t2 = any(k.startswith('t2') and v[2] for k,v in items.items())
                ne2 = sum(1 for k in items if k.startswith('e2'))

                if has_t4 and has_t2:
                    t4_info = [v for k,v in items.items() if k.startswith('t4')][0]
                    # t4 on goal?
                    if t4_info[2]:
                        # Check t4 safe from enemies
                        t4_safe = True
                        for e in game.peiiyyzum:
                            for f in game.hmeulfxgy:
                                if game.amnmgwpkeb.get(f, 0) == 4:
                                    if game.rukauvoumh(e, f):
                                        t4_safe = False
                        if t4_safe:
                            score = 100 + ne2 * 20 + game.step_counter_ui.current_steps
                            if score > best_score:
                                best_score = score
                                best = prefix
                                pstate(game, f'  NEW BEST (score={score})')

                if tested % 50000 == 0:
                    elapsed = time.time() - t0
                    print(f"  {tested} tested, {elapsed:.0f}s, best_score={best_score}")

                if time.time() - t0 > 120:
                    break
            if time.time() - t0 > 120:
                break
        if time.time() - t0 > 120:
            break

    print(f"\nBest prefix: {best} (score={best_score})")
    print(f"Total tested: {tested}")

    if best:
        # Now search from best prefix
        print(f"\nSearching from best prefix {best}...")
        game_best = replay(best)
        pstate(game_best, '  State')

    return best


if __name__ == "__main__":
    # First try guided search
    result = guided_search()
    if result:
        print(f"\n*** FINAL SOLUTION: {result}")
        game = replay(result)
        if game.level_index != 8:
            print("CONFIRMED WIN!")
    else:
        print("\nGuided search didn't find solution. Trying wide search...")
        result = wide_search()
