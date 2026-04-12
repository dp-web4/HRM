#!/usr/bin/env python3
"""
su15 L9 fast solver — replay-based beam search with small candidate sets.

Key optimization: replay is fast because each click is just one perform_action call.
Keep beam width small and candidate set focused.
"""

import sys
import time

sys.path.insert(0, 'environment_files/su15/4c352900')
from su15 import Su15, GameAction, ActionInput


def click(game, x, y):
    action = ActionInput(id=GameAction.ACTION6.value, data={'x': x, 'y': y})
    game.perform_action(action)


def replay(clicks):
    """Replay a click sequence from fresh L9 start. Returns game."""
    game = Su15()
    game.set_level(8)
    for cx, cy in clicks:
        if game.level_index != 8:
            return game
        click(game, cx, cy)
    return game


def state_key(game):
    fruits = tuple(sorted((game.amnmgwpkeb.get(f, 0), f.x, f.y) for f in game.hmeulfxgy))
    enemies = tuple(sorted((game.hirdajbmj.get(e, '?'), e.x, e.y) for e in game.peiiyyzum))
    return (fruits, enemies)


def score_state(game):
    """Score how close to winning."""
    goal_centers = [(11,41), (53,55), (11,55)]
    s = 0.0
    flags = [False, False, False]  # t4_on_goal, t2_on_goal, e3_on_goal

    for f in game.hmeulfxgy:
        t = game.amnmgwpkeb.get(f, 0)
        cx, cy = game.qmecbepbyz(f)
        on_goal = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)
        if t == 4:
            s += 20
            if on_goal:
                flags[0] = True
                s += 100
            else:
                min_d = min(abs(cx-gx)+abs(cy-gy) for gx,gy in goal_centers)
                s += max(0, 40-min_d)
        elif t == 2:
            s += 20
            if on_goal:
                flags[1] = True
                s += 100
            else:
                min_d = min(abs(cx-gx)+abs(cy-gy) for gx,gy in goal_centers)
                s += max(0, 40-min_d)
        elif t == 5:
            s += 10
        elif t >= 3:
            s += 5

    for e in game.peiiyyzum:
        et = game.hirdajbmj.get(e, '?')
        lvl = game.cqjufwqvag(et)
        cx, cy = game.qmecbepbyz(e)
        on_goal = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)
        if lvl == 3:
            s += 30
            if on_goal:
                flags[2] = True
                s += 100
            else:
                min_d = min(abs(cx-gx)+abs(cy-gy) for gx,gy in goal_centers)
                s += max(0, 50-min_d)
        elif lvl == 2:
            s += 15

    return s, flags


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


def make_candidates_for_state(game):
    """Small focused candidate set based on current state."""
    cands = set()

    # Near all entities (fruits + enemies)
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

    # Goal centers and nearby
    for gx, gy in [(11,41), (53,55), (11,55)]:
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx, ny = gx+dx, gy+dy
                if 10 <= ny <= 62:
                    cands.add((nx, ny))

    # Midpoints between enemy pairs (for merges)
    enemies = list(game.peiiyyzum)
    for i in range(len(enemies)):
        for j in range(i+1, len(enemies)):
            cx1, cy1 = game.qmecbepbyz(enemies[i])
            cx2, cy2 = game.qmecbepbyz(enemies[j])
            mx, my = (cx1+cx2)//2, (cy1+cy2)//2
            for dx in range(-4, 5, 2):
                for dy in range(-4, 5, 2):
                    nx, ny = mx+dx, my+dy
                    if 0 <= nx < 64 and 10 <= ny <= 62:
                        cands.add((nx, ny))

    # Empty corners (for chase without vacuum)
    for x, y in [(4,12), (60,12), (4,58), (60,58), (4,60), (60,60)]:
        cands.add((x, y))

    return sorted(cands)


def beam_search():
    print("=== Fast beam search ===")

    BEAM_WIDTH = 200
    MAX_DEPTH = 25

    # Initial
    beam = [(0, [])]  # (score, clicks)

    best_ever = (0, [])
    t0 = time.time()
    total_replays = 0

    for depth in range(MAX_DEPTH):
        if time.time() - t0 > 600:
            print("Timeout!")
            break

        next_beam = []
        seen_states = set()

        for score, prev_clicks in beam:
            # Replay to get current state
            game = replay(prev_clicks)
            if game.level_index != 8:
                print(f"\n*** WON! {prev_clicks}")
                return prev_clicks
            if game.step_counter_ui.current_steps <= 0:
                continue

            cands = make_candidates_for_state(game)

            for cx, cy in cands:
                new_clicks = prev_clicks + [(cx, cy)]
                game2 = replay(new_clicks)
                total_replays += 1

                if game2.level_index != 8:
                    print(f"\n*** SOLUTION at depth {depth+1}: {new_clicks}")
                    return new_clicks

                if game2.step_counter_ui.current_steps <= 0:
                    continue

                sk = state_key(game2)
                if sk in seen_states:
                    continue
                seen_states.add(sk)

                s, flags = score_state(game2)
                next_beam.append((s, new_clicks))

        if not next_beam:
            print(f"  Depth {depth+1}: dead end")
            break

        next_beam.sort(key=lambda x: -x[0])
        beam = next_beam[:BEAM_WIDTH]

        bs, bc = beam[0]
        if bs > best_ever[0]:
            best_ever = (bs, bc)

        bg = replay(bc)
        elapsed = time.time() - t0
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

        print(f"  D{depth+1}: score={bs:.0f} beam={len(beam)} "
              f"states={len(seen_states)} replays={total_replays} "
              f"{' '.join(items)} steps={bg.step_counter_ui.current_steps} "
              f"{elapsed:.1f}s")

    print(f"\nBest: score={best_ever[0]}, clicks={best_ever[1]}")
    return None


if __name__ == "__main__":
    # First, test replay speed
    t0 = time.time()
    for _ in range(100):
        g = replay([(15,52), (30,44), (22,38), (16,36), (12,38)])
    t1 = time.time()
    print(f"Replay speed: {100/(t1-t0):.0f} replays/sec for 5-click sequence")
    print()

    result = beam_search()
    if result:
        print(f"\nFinal solution: {result}")
        print("Verifying...")
        game = replay(result)
        if game.level_index != 8:
            print("  CONFIRMED WIN!")
        else:
            pstate(game, "  Final state:")
