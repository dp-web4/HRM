#!/usr/bin/env python3
"""
su15 L9 BFS solver — systematic search over click sequences.

Strategy: Search for sequences where:
1. Tier-2 is on a goal (from tier-1 merge)
2. Tier-4 is on a goal (from knockdown)
3. Enemy-3 is on a goal (from enemy merges)
All simultaneously when win check fires.
"""

import sys
import copy
import time
import pickle
from collections import deque

sys.path.insert(0, 'environment_files/su15/4c352900')
from su15 import Su15, GameAction, ActionInput


def click(game, x, y):
    action = ActionInput(id=GameAction.ACTION6.value, data={'x': x, 'y': y})
    game.perform_action(action)


def get_compact_state(game):
    """Get compact state for comparison."""
    fruits = tuple(sorted((game.amnmgwpkeb.get(f, 0), f.x, f.y) for f in game.hmeulfxgy))
    enemies = []
    for e in game.peiiyyzum:
        et = game.hirdajbmj.get(e, '?')
        enemies.append((et, e.x, e.y))
    enemies = tuple(sorted(enemies))
    return (fruits, enemies)


def check_progress(game):
    """Check how close we are to winning."""
    # Goals: tier-4 on goal, tier-2 on goal, enemy-3 on goal
    score = 0
    t4_on_goal = False
    t2_on_goal = False
    e3_on_goal = False

    for f in game.hmeulfxgy:
        t = game.amnmgwpkeb.get(f, 0)
        cx, cy = game.qmecbepbyz(f)
        on_goal = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)
        if t == 4 and on_goal:
            t4_on_goal = True
            score += 10
        elif t == 2 and on_goal:
            t2_on_goal = True
            score += 10
        elif t == 4:
            score += 3
        elif t == 2:
            score += 3

    for e in game.peiiyyzum:
        et = game.hirdajbmj.get(e, '?')
        lvl = game.cqjufwqvag(et)
        cx, cy = game.qmecbepbyz(e)
        on_goal = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)
        if lvl == 3 and on_goal:
            e3_on_goal = True
            score += 10
        elif lvl == 3:
            score += 5
        elif lvl == 2:
            score += 2

    return score, t4_on_goal, t2_on_goal, e3_on_goal


def save_game(game):
    """Save game state for restoration via undo snapshots."""
    return game.dqxbwefew[-1] if game.dqxbwefew else None


def search_from_c1():
    """
    BFS from after C1 (tier-1 merge at (15,52)).

    After C1:
      t5(35,48)c(38,51), t2(14,51)c(15,52)*
      4 enemies at various positions

    Search over click positions for C2-C7+ to find winning sequence.
    """
    print("=== BFS from C1 (15,52) ===")
    print()

    # Generate candidate click positions
    # Grid of 4x4 resolution + key positions
    candidates = set()

    # Coarse grid
    for x in range(4, 61, 4):
        for y in range(10, 61, 4):
            candidates.add((x, y))

    # Goal centers
    for gx, gy in [(11,41), (53,55), (11,55)]:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                candidates.add((gx+dx, gy+dy))

    # Near initial fruit/enemy positions
    for x, y in [(38,51), (19,47), (24,53), (53,15), (16,14), (17,24), (56,35)]:
        for dx in range(-4, 5, 2):
            for dy in range(-4, 5, 2):
                nx, ny = x+dx, y+dy
                if 0 <= nx < 64 and 10 <= ny < 63:
                    candidates.add((nx, ny))

    candidates = sorted(candidates)
    print(f"Candidate positions: {len(candidates)}")

    # BFS with state dedup
    # State: after each click, record compact state
    # Prune states we've seen before

    # Actually, full BFS is too expensive. Let me do beam search instead.
    # Or better: iterative deepening with pruning.

    best_score = 0
    best_seq = []
    solutions = []

    t0 = time.time()

    # Try depth-limited DFS with pruning
    def dfs(clicks_so_far, depth_remaining, seen_states):
        nonlocal best_score, best_seq, solutions

        if time.time() - t0 > 300:  # 5 min timeout
            return

        # Create game and replay
        game = Su15()
        game.set_level(8)
        click(game, 15, 52)  # C1
        for cx, cy in clicks_so_far:
            if game.level_index != 8:
                # WON!
                solutions.append([(15,52)] + list(clicks_so_far))
                print(f"*** SOLUTION FOUND: {[(15,52)] + list(clicks_so_far)} ***")
                return
            click(game, cx, cy)

        if game.level_index != 8:
            solutions.append([(15,52)] + list(clicks_so_far))
            print(f"*** SOLUTION FOUND: {[(15,52)] + list(clicks_so_far)} ***")
            return

        if game.step_counter_ui.current_steps <= 0:
            return

        score, t4g, t2g, e3g = check_progress(game)

        if score > best_score:
            best_score = score
            best_seq = [(15,52)] + list(clicks_so_far)
            state_str = []
            for f in game.hmeulfxgy:
                t = game.amnmgwpkeb.get(f, 0)
                cx, cy = game.qmecbepbyz(f)
                og = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)
                state_str.append(f't{t}({cx},{cy}){"*" if og else ""}')
            for e in game.peiiyyzum:
                et = game.hirdajbmj.get(e, '?')
                lvl = game.cqjufwqvag(et)
                cx, cy = game.qmecbepbyz(e)
                og = any(game.epvtlqtczz(cx, cy, g) for g in game.rqdsgrklq)
                state_str.append(f'e{lvl}({cx},{cy}){"*" if og else ""}')
            print(f"  Best score={score} at depth {len(clicks_so_far)}: "
                  f"seq={best_seq}, state={' '.join(state_str)}, "
                  f"steps={game.step_counter_ui.current_steps}")

        if depth_remaining <= 0:
            return

        state = get_compact_state(game)
        if state in seen_states:
            return
        seen_states.add(state)

        # Prioritize candidates based on current state
        # For now, just try all
        for cx, cy in candidates:
            dfs(clicks_so_far + [(cx, cy)], depth_remaining - 1, seen_states)
            if solutions:
                return

    # Iterative deepening
    for depth in range(1, 48):
        print(f"\n--- Depth {depth} ---")
        seen = set()
        dfs([], depth, seen)
        print(f"  States explored: {len(seen)}")
        if solutions:
            break
        if time.time() - t0 > 300:
            print("Timeout!")
            break

    if solutions:
        print(f"\n*** SOLUTIONS: {solutions}")
    else:
        print(f"\nNo solution found. Best: score={best_score}, seq={best_seq}")


def beam_search():
    """
    Beam search: keep top-K states at each depth level.
    """
    print("=== Beam search ===")

    # Generate candidate click positions (coarser for speed)
    candidates = []
    for x in range(4, 61, 4):
        for y in range(10, 61, 4):
            candidates.append((x, y))
    # Add goal centers
    for gx, gy in [(11,41), (53,55), (11,55)]:
        candidates.append((gx, gy))

    print(f"Candidates: {len(candidates)}")

    BEAM_WIDTH = 200
    MAX_DEPTH = 20

    # State: list of clicks after C1
    # Each beam entry: (score, clicks, compact_state)

    beam = [(0, [])]  # initial: just C1 done

    t0 = time.time()

    for depth in range(MAX_DEPTH):
        if time.time() - t0 > 300:
            print("Timeout!")
            break

        next_beam = []
        seen_states = set()

        for _, prev_clicks in beam:
            for cx, cy in candidates:
                new_clicks = prev_clicks + [(cx, cy)]

                # Replay
                game = Su15()
                game.set_level(8)
                click(game, 15, 52)  # C1
                for pcx, pcy in new_clicks:
                    if game.level_index != 8:
                        print(f"\n*** SOLUTION at depth {depth+1}: {[(15,52)] + new_clicks} ***")
                        return [(15,52)] + new_clicks
                    click(game, pcx, pcy)

                if game.level_index != 8:
                    print(f"\n*** SOLUTION at depth {depth+1}: {[(15,52)] + new_clicks} ***")
                    return [(15,52)] + new_clicks

                if game.step_counter_ui.current_steps <= 0:
                    continue

                state = get_compact_state(game)
                if state in seen_states:
                    continue
                seen_states.add(state)

                score, t4g, t2g, e3g = check_progress(game)
                next_beam.append((score, new_clicks))

        if not next_beam:
            print(f"  Depth {depth+1}: no valid states")
            break

        # Keep top-K
        next_beam.sort(key=lambda x: -x[0])
        beam = next_beam[:BEAM_WIDTH]

        best = beam[0]
        # Print best state details
        game = Su15()
        game.set_level(8)
        click(game, 15, 52)
        for pcx, pcy in best[1]:
            click(game, pcx, pcy)

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

        elapsed = time.time() - t0
        print(f"  Depth {depth+1}: best={best[0]}, beam={len(beam)}, "
              f"state={' '.join(items)}, steps={game.step_counter_ui.current_steps}, "
              f"time={elapsed:.1f}s")

    print(f"\nBest: score={beam[0][0]}, clicks={[(15,52)] + beam[0][1]}")
    return None


def fast_beam():
    """
    Faster beam search — replay incrementally, not from scratch.
    Store game states directly (via undo mechanism or deepcopy).
    """
    print("=== Fast beam search ===")

    # Candidates: coarse grid + goals
    candidates = []
    for x in range(4, 61, 8):
        for y in range(12, 60, 8):
            candidates.append((x, y))
    for gx, gy in [(11,41), (53,55), (11,55)]:
        for dx in range(-2, 3, 2):
            for dy in range(-2, 3, 2):
                candidates.append((gx+dx, gy+dy))
    candidates = list(set(candidates))

    print(f"Candidates: {len(candidates)}")

    BEAM_WIDTH = 100
    MAX_DEPTH = 30

    def snapshot(game):
        """Save game state compactly."""
        return game.dqxbwefew[-1] if game.dqxbwefew else []

    # Use undo stack: each click pushes state, undo pops
    # But that only stores 1 level. Instead, save the compact click sequence.

    # Beam entries: (score, click_sequence)
    beam = [(0, [])]

    t0 = time.time()

    for depth in range(MAX_DEPTH):
        if time.time() - t0 > 600:
            print("Timeout!")
            break

        next_beam = []
        seen_states = set()

        for _, prev_clicks in beam:
            # Replay to get game state
            game = Su15()
            game.set_level(8)
            click(game, 15, 52)  # C1
            for pcx, pcy in prev_clicks:
                click(game, pcx, pcy)

            if game.level_index != 8 or game.step_counter_ui.current_steps <= 0:
                continue

            # Try each candidate
            for cx, cy in candidates:
                # Quick replay: undo + redo
                game2 = Su15()
                game2.set_level(8)
                click(game2, 15, 52)
                for pcx, pcy in prev_clicks:
                    click(game2, pcx, pcy)
                click(game2, cx, cy)

                if game2.level_index != 8:
                    seq = [(15,52)] + prev_clicks + [(cx,cy)]
                    print(f"\n*** SOLUTION at depth {depth+1}: {seq} ***")
                    return seq

                if game2.step_counter_ui.current_steps <= 0:
                    continue

                state = get_compact_state(game2)
                if state in seen_states:
                    continue
                seen_states.add(state)

                score, t4g, t2g, e3g = check_progress(game2)
                next_beam.append((score, prev_clicks + [(cx,cy)]))

        if not next_beam:
            print(f"  Depth {depth+1}: no valid states")
            break

        next_beam.sort(key=lambda x: -x[0])
        beam = next_beam[:BEAM_WIDTH]

        # Report best
        best = beam[0]
        game = Su15()
        game.set_level(8)
        click(game, 15, 52)
        for pcx, pcy in best[1]:
            click(game, pcx, pcy)

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

        elapsed = time.time() - t0
        print(f"  D{depth+1}: best={best[0]}, beam={len(beam)}, "
              f"{' '.join(items)}, steps={game.step_counter_ui.current_steps}, "
              f"{elapsed:.1f}s, clicks={(15,52)}+{best[1][-3:]}")

    print(f"\nBest: score={beam[0][0]}, clicks={[(15,52)] + beam[0][1]}")
    return None


if __name__ == "__main__":
    result = fast_beam()
    if result:
        print(f"\nFinal solution: {result}")
