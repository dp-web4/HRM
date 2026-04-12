#!/usr/bin/env python3
"""
su15 L9 final solver — search from established position.

After 4 clicks [(15,52),(16,19),(48,35),(43,29)]:
  t5(35,48) center (38,51)
  t2(14,51) ON goal 3
  e2(13,28) center (15,30) — left
  e2(42,27) center (44,29) — right
  44 steps left

Need: t4 on a goal, e3 on a goal, t2 on a goal (already done).

Strategy: Beam search with focused scoring.
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


PREFIX = [(15,52),(16,19),(48,35),(43,29)]


def state_key(game):
    fruits = tuple(sorted((game.amnmgwpkeb.get(f, 0), f.x, f.y) for f in game.hmeulfxgy))
    enemies = tuple(sorted((game.hirdajbmj.get(e, '?'), e.x, e.y) for e in game.peiiyyzum))
    return (fruits, enemies)


def score_state(game):
    """Score focusing on: t4 on goal, e3 on goal, t2 on goal, with safety."""
    goal_centers = [(11,41), (53,55), (11,55)]
    s = 0.0

    t4_on_goal = False
    t2_on_goal = False
    e3_on_goal = False
    t4_exists = False
    e3_exists = False
    t2_exists = False

    for f in game.hmeulfxgy:
        t = game.amnmgwpkeb.get(f, 0)
        cx, cy = game.qmecbepbyz(f)
        on_goal = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)
        safe = True
        for e in game.peiiyyzum:
            if game.rukauvoumh(e, f):
                safe = False

        if t == 4:
            t4_exists = True
            if on_goal and safe:
                t4_on_goal = True
                s += 200
            elif on_goal:
                s += 50  # on goal but overlapping enemy
            else:
                min_d = min(abs(cx-gx)+abs(cy-gy) for gx,gy in goal_centers)
                s += max(0, 60-min_d)
        elif t == 2:
            t2_exists = True
            if on_goal:
                t2_on_goal = True
                s += 200
            else:
                s += 20
        elif t == 5:
            s += 30  # needs knockdown

    for e in game.peiiyyzum:
        et = game.hirdajbmj.get(e, '?')
        lvl = game.cqjufwqvag(et)
        cx, cy = game.qmecbepbyz(e)
        on_goal = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)

        if lvl == 3:
            e3_exists = True
            if on_goal:
                e3_on_goal = True
                s += 200
            else:
                min_d = min(abs(cx-gx)+abs(cy-gy) for gx,gy in goal_centers)
                s += max(0, 70-min_d)
        elif lvl == 2:
            s += 20

    # Big bonus for having all three items existing
    if t4_exists: s += 50
    if e3_exists: s += 50
    if t2_exists: s += 30

    # Perfect score = all three on goals
    if t4_on_goal and t2_on_goal and e3_on_goal:
        s += 1000

    return s, (t4_on_goal, t2_on_goal, e3_on_goal)


def make_candidates(game):
    cands = set()
    # Near entities
    for f in game.hmeulfxgy:
        cx, cy = game.qmecbepbyz(f)
        for dx in range(-8, 9, 2):
            for dy in range(-8, 9, 2):
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < 64 and 10 <= ny <= 62:
                    cands.add((nx, ny))
    for e in game.peiiyyzum:
        cx, cy = game.qmecbepbyz(e)
        for dx in range(-8, 9, 2):
            for dy in range(-8, 9, 2):
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < 64 and 10 <= ny <= 62:
                    cands.add((nx, ny))
    # Goals
    for gx, gy in [(11,41), (53,55), (11,55)]:
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if 10 <= gy+dy <= 62:
                    cands.add((gx+dx, gy+dy))
    # Midpoints between enemy pairs
    enemies = list(game.peiiyyzum)
    for i in range(len(enemies)):
        for j in range(i+1, len(enemies)):
            cx1, cy1 = game.qmecbepbyz(enemies[i])
            cx2, cy2 = game.qmecbepbyz(enemies[j])
            mx, my = (cx1+cx2)//2, (cy1+cy2)//2
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if 0 <= mx+dx < 64 and 10 <= my+dy <= 62:
                        cands.add((mx+dx, my+dy))
    # Corners for chase
    cands.update([(4,12), (60,12), (4,60), (60,60)])
    return sorted(cands)


def beam_search():
    print("=== Beam search from established position ===")
    BEAM_WIDTH = 300
    MAX_DEPTH = 40

    beam = [(0, [])]  # (score, additional_clicks)
    best_ever = (0, [])
    t0 = time.time()
    total_replays = 0

    for depth in range(MAX_DEPTH):
        if time.time() - t0 > 600:
            print("Timeout!")
            break

        next_beam = []
        seen = set()

        for _, prev in beam:
            game = replay(PREFIX + prev)
            if game.level_index != 8:
                print(f"\n*** WON: {PREFIX + prev}")
                return PREFIX + prev
            if game.step_counter_ui.current_steps <= 0:
                continue

            cands = make_candidates(game)
            for cx, cy in cands:
                new = prev + [(cx, cy)]
                game2 = replay(PREFIX + new)
                total_replays += 1

                if game2.level_index != 8:
                    print(f"\n*** SOLUTION: {PREFIX + new}")
                    return PREFIX + new

                if game2.step_counter_ui.current_steps <= 0:
                    continue

                sk = state_key(game2)
                if sk in seen:
                    continue
                seen.add(sk)

                s, flags = score_state(game2)
                next_beam.append((s, new))

        if not next_beam:
            print(f"  D{depth+1}: dead end")
            break

        next_beam.sort(key=lambda x: -x[0])
        beam = next_beam[:BEAM_WIDTH]

        bs, bc = beam[0]
        if bs > best_ever[0]:
            best_ever = (bs, bc)

        bg = replay(PREFIX + bc)
        items = []
        for f in bg.hmeulfxgy:
            t = bg.amnmgwpkeb.get(f, 0)
            cx, cy = bg.qmecbepbyz(f)
            og = any(bg.epvtlqtczz(cx, cy, g) for g in bg.rqdsgrklq)
            items.append(f't{t}({cx},{cy}){"*" if og else ""}')
        for e in bg.peiiyyzum:
            et = bg.hirdajbmj.get(e, '?')
            lvl = bg.cqjufwqvag(et)
            cx, cy = bg.qmecbepbyz(e)
            og = any(bg.epvtlqtczz(cx, cy, g) for g in bg.rqdsgrklq)
            items.append(f'e{lvl}({cx},{cy}){"*" if og else ""}')

        elapsed = time.time() - t0
        print(f"  D{depth+1}: s={bs:.0f} beam={len(beam)} seen={len(seen)} "
              f"replays={total_replays} {' '.join(items)} "
              f"steps={bg.step_counter_ui.current_steps} {elapsed:.1f}s")

    print(f"\nBest: score={best_ever[0]}, full seq={PREFIX + best_ever[1]}")
    return None


if __name__ == "__main__":
    # Verify prefix
    game = replay(PREFIX)
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
    print(f"Prefix state: {' '.join(items)}, steps={game.step_counter_ui.current_steps}")
    print()

    result = beam_search()
    if result:
        print(f"\n*** FINAL: {result}")
        g = replay(result)
        print(f"Level after replay: {g.level_index}")
