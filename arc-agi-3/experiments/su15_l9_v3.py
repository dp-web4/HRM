#!/usr/bin/env python3
"""
su15 L9 solver v3 — target assignment: t4->G2, t2->G3, e3->G1

Goals:
  G1: center (11,41), bounds [7,16) x [37,46)
  G2: center (53,55), bounds [49,58) x [51,60)
  G3: center (11,55), bounds [7,16) x [51,60)

Strategy:
  1. Merge tier-1s -> tier-2 on G3 (click ~(15,52))
  2. Pull tier-5 toward G2 (2 clicks)
  3. Get enemy to knock tier-5 -> tier-4 near G2
  4. Position tier-4 on G2
  5. Merge all enemies to type-3 on G1 (left side)
  6. Win check: all three items on goals simultaneously

The key insight: enemies naturally converge toward the left side (chasing tier-2 on G3).
We can use vacuum to accelerate enemy convergence and trigger merges near G1.
"""

import sys
import time

sys.path.insert(0, 'environment_files/su15/4c352900')
from su15 import Su15, GameAction, ActionInput


def click(game, x, y):
    action = ActionInput(id=GameAction.ACTION6.value, data={'x': x, 'y': y})
    game.perform_action(action)


def replay(clicks):
    game = Su15()
    game.set_level(8)
    for cx, cy in clicks:
        if game.level_index != 8: return game
        click(game, cx, cy)
    return game


def pstate(game, label=''):
    items = []
    for f in game.hmeulfxgy:
        t = game.amnmgwpkeb.get(f, 0)
        cx, cy = game.qmecbepbyz(f)
        og = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)
        items.append(f't{t}({f.x},{f.y})c({cx},{cy}){"*" if og else ""}')
    for e in game.peiiyyzum:
        et = game.hirdajbmj.get(e, '?')
        lvl = game.cqjufwqvag(et)
        cx, cy = game.qmecbepbyz(e)
        og = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)
        items.append(f'e{lvl}({e.x},{e.y})c({cx},{cy}){"*" if og else ""}')
    print(f'{label} [{game.step_counter_ui.current_steps}s] {" ".join(items)}')


def check_state(game):
    """Return structured state info."""
    fruits = {}
    for f in game.hmeulfxgy:
        t = game.amnmgwpkeb.get(f, 0)
        cx, cy = game.qmecbepbyz(f)
        og = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)
        safe = True
        for e in game.peiiyyzum:
            if game.rukauvoumh(e, f):
                safe = False
        fruits[f't{t}'] = {'pos': (f.x, f.y), 'center': (cx, cy), 'on_goal': og, 'safe': safe}
    enemies = {}
    idx = {}
    for e in game.peiiyyzum:
        et = game.hirdajbmj.get(e, '?')
        lvl = game.cqjufwqvag(et)
        cx, cy = game.qmecbepbyz(e)
        og = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)
        key = f'e{lvl}'
        n = idx.get(key, 0) + 1
        idx[key] = n
        enemies[f'{key}_{n}'] = {'pos': (e.x, e.y), 'center': (cx, cy), 'on_goal': og}
    return fruits, enemies


def search_final_merge(prefix, max_additional=15):
    """
    From a state where t4 and t2 are on goals (or close),
    search for clicks that merge enemies to e3 on G1 while keeping fruits safe.
    """
    game0 = replay(prefix)
    fruits0, enemies0 = check_state(game0)

    # Check preconditions
    t4_info = fruits0.get('t4')
    t2_info = fruits0.get('t2')
    if not t4_info or not t2_info:
        return None

    print(f"  Starting final merge search from {prefix[-3:]}, "
          f"steps={game0.step_counter_ui.current_steps}")

    # BFS: try each additional click
    best = None
    best_score = 0

    for depth in range(1, max_additional + 1):
        if depth == 1:
            # Try all positions around entities
            cands = set()
            for e in game0.peiiyyzum:
                cx, cy = game0.qmecbepbyz(e)
                for dx in range(-8, 9, 2):
                    for dy in range(-8, 9, 2):
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < 64 and 10 <= ny <= 62:
                            cands.add((nx, ny))
            for gx, gy in [(11,41), (53,55), (11,55)]:
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        cands.add((gx+dx, gy+dy))
            # Empty corners for chase
            cands.update([(4,60), (60,60), (4,12), (60,12)])
            cands = sorted(cands)

            for cx, cy in cands:
                game = replay(prefix + [(cx, cy)])
                if game.level_index != 8:
                    print(f"  *** WON with +[({cx},{cy})]")
                    return prefix + [(cx, cy)]

                fruits, enemies = check_state(game)
                t4 = fruits.get('t4')
                t2 = fruits.get('t2')
                if not t4 or not t2:
                    continue
                if not t2.get('on_goal'):
                    continue

                # Score: e3 on goal + t4 on goal + t4 safe
                e3s = {k:v for k,v in enemies.items() if k.startswith('e3')}
                score = 0
                if t4['on_goal'] and t4['safe']:
                    score += 100
                if e3s:
                    for e3k, e3v in e3s.items():
                        if e3v['on_goal']:
                            score += 200
                        else:
                            cx3, cy3 = e3v['center']
                            d = abs(cx3-11) + abs(cy3-41)
                            score += max(0, 50-d)
                elif len(enemies) <= 3:
                    score += 20  # fewer enemies = progress toward merges

                if score > best_score:
                    best_score = score
                    best = prefix + [(cx, cy)]
                    pstate(game, f'    D{depth} best={score}')

    return best


def main():
    print("=== su15 L9 Solver v3 ===")
    print()
    t0 = time.time()

    # PHASE 1: Merge + position tier-5
    print("Phase 1: C1 merge + C2-C3 pull tier-5 to goal 2")

    # After C1=(15,52), C2=(46,53), C3=(52,55):
    #   t5(49,52)c(52,55)* t2(14,51)c(15,52)*
    #   e1(48,25)c(50,27) e1(13,24)c(15,26) e1(13,34)c(15,36) e1(50,45)c(52,47)

    base = [(15,52), (46,53), (52,55)]
    game = replay(base)
    pstate(game, 'After C1-C3')

    # PHASE 2: Knockdown tier-5 near goal 2
    # Need enemy to hit tier-5 at (49,52).
    # e1 at (50,45) center (52,47) is 8px away. It will chase tier-5.
    # But we need the knockback to keep the tier-4 ON goal 2.
    # The enemy approaches from ABOVE (y=45 → y=52), so knockback is DOWNWARD.
    # tier-5 at (49,52)-(56,59). Knockback 10px down → (49,62) center (52,65) → clamped to max.
    # That's off goal.

    # Alternative: use vacuum to pull the enemy THROUGH tier-5 from the LEFT.
    # If enemy approaches from left, knockback pushes right (stays on goal).
    # But the enemy at (50,45) is already RIGHT of tier-5.

    # What if I use the left-side enemies instead?
    # e1 at (13,24) center (15,26) — far left
    # e1 at (13,34) center (15,36) — far left
    # These chase the nearest fruit, which is t2 at (14,51).

    # Let me try: pull e1 at (50,45) upward (away from tier-5) while letting
    # left-side enemies chase tier-2. Then separately handle knockdown.

    # Actually, what about a different approach: let enemies chase naturally,
    # but use vacuum strategically to prevent knockdown of tier-2 while
    # allowing knockdown of tier-5.

    # Or: what if I pull tier-5 only to (46,53) with C2, NOT all the way to goal 2?
    # Then let enemy knock it, and the knockback sends it onto goal 2.

    # Let me try: C1=(15,52), C2=(46,53) — partial pull
    # After C2: t5(43,50)c(46,53), enemies chasing
    # Then let enemies converge. e1 at (44,21) chases t5.

    print("\nPhase 2: Knockdown with controlled direction")

    # Try C1=(15,52), C2=(46,53), then chase clicks
    base2 = [(15,52), (46,53)]
    game2 = replay(base2)
    pstate(game2, 'After C1-C2')

    for i in range(15):
        click(game2, 4, 12)  # click far top-left, won't catch anything important
        pstate(game2, f'C{i+3} (4,12)')
        fruits, enemies = check_state(game2)
        t4 = fruits.get('t4')
        t2 = fruits.get('t2')
        if t4:
            print(f'    t4 found! safe={t4["safe"]}, on_goal={t4["on_goal"]}')
            break
        if not fruits.get('t5') and not t4:
            print('    tier-5 destroyed!')
            break

    print("\n--- Alternative: systematically search C4 knockdown positions ---")

    # After C1=(15,52), C2=(46,53), try different C3 positions
    # to get the knockdown with the right direction
    base3 = [(15,52), (46,53)]
    best_kd_results = []

    for c3x in range(0, 64, 2):
        for c3y in range(10, 63, 2):
            game3 = replay(base3 + [(c3x, c3y)])
            if game3.level_index != 8:
                print(f"  WON with C3=({c3x},{c3y})")
                return base3 + [(c3x, c3y)]

            fruits, enemies = check_state(game3)
            t4 = fruits.get('t4')
            t2 = fruits.get('t2')

            if t4 and t2 and t2.get('on_goal'):
                t4og = t4['on_goal']
                t4safe = t4['safe']
                n_e = len(enemies)
                n_e2 = sum(1 for k in enemies if k.startswith('e2'))

                if t4og and t4safe:
                    steps = game3.step_counter_ui.current_steps
                    best_kd_results.append((c3x, c3y, n_e2, n_e, steps))

    best_kd_results.sort(key=lambda x: (x[2], x[4]), reverse=True)
    print(f"\nKnockdown options with t4 on goal & safe, t2 on goal: {len(best_kd_results)}")
    for x, y, ne2, ne, steps in best_kd_results[:20]:
        game3 = replay(base3 + [(x, y)])
        pstate(game3, f'  C3=({x},{y})')

    # If we found good knockdowns, search from there
    if best_kd_results:
        print(f"\nSearching from top knockdown results...")
        for x, y, ne2, ne, steps in best_kd_results[:5]:
            prefix = base3 + [(x, y)]
            result = search_final_merge(prefix, max_additional=5)
            if result:
                game_r = replay(result)
                if game_r.level_index != 8:
                    print(f"\n*** SOLUTION: {result}")
                    return result

    # Try also with different C2 positions
    print("\n--- Searching with different C2 positions ---")
    for c2x in range(38, 58, 2):
        for c2y in range(44, 58, 2):
            for c3x in range(0, 64, 4):
                for c3y in range(10, 63, 4):
                    seq = [(15,52), (c2x, c2y), (c3x, c3y)]
                    game = replay(seq)
                    if game.level_index != 8:
                        print(f"  WON: {seq}")
                        return seq
                    fruits, enemies = check_state(game)
                    t4 = fruits.get('t4')
                    t2 = fruits.get('t2')
                    if t4 and t2 and t2.get('on_goal') and t4.get('on_goal') and t4.get('safe'):
                        n_e2 = sum(1 for k in enemies if k.startswith('e2'))
                        if n_e2 >= 1:
                            pstate(game, f'  ({c2x},{c2y}),({c3x},{c3y})')

            if time.time() - t0 > 300:
                print("Time limit reached")
                break
        if time.time() - t0 > 300:
            break

    return None


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n*** FINAL SOLUTION: {result}")
        game = replay(result)
        if game.level_index != 8:
            print("CONFIRMED WIN!")
        else:
            pstate(game, "Final state:")
