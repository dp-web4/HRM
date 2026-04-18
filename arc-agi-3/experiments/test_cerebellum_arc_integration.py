#!/usr/bin/env python3
"""
End-to-end integration test: harness → cerebellum → habit replay.

Proves the full pipeline:
1. Structured probing discovers cd82 L1 winning sequence
2. ArcCerebellum records the (state, actions, outcome) triple
3. After 3 observations, habit compiles to mature
4. Habit replay wins cd82 L1 without deliberation

This is the cerebellum's proof-of-concept: a game-agnostic discovery
mechanism feeds a habit compiler that can replay wins automatically.
"""

import sys
import numpy as np
from pathlib import Path

SAGE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SAGE_DIR))

from sage.cognition.cerebellum.core import Cerebellum, StateSignature
from sage.cognition.cerebellum.arc_game_adapter import (
    ArcCerebellum,
    arc_state_signature,
    extract_state_from_env,
    ACTION_NAMES,
)


def int_to_ga(i):
    from arcengine.enums import GameAction
    return getattr(GameAction, f'ACTION{i}')


def discover_cd82_winning_sequence():
    """Run structured probing on cd82 L1 to find a winning action sequence.

    Returns (action_list, game_id) where action_list is a list of dicts
    with 'action' (int) and optional 'data' (dict for CLICK coords).
    """
    from arc_agi import Arcade

    arcade = Arcade()
    game_id = None
    for e in arcade.get_environments():
        if 'cd82' in e.game_id:
            game_id = e.game_id
            break

    if not game_id:
        print("cd82 not found!")
        return None, None

    env = arcade.make(game_id)
    fd = env.reset()
    game = env._game
    cam = game.camera

    available = fd.available_actions
    print(f"Game: {game_id}, actions: {[ACTION_NAMES.get(a, f'A{a}') for a in available]}")

    # Build game→display coordinate lookup
    g2d = {}
    for dx in range(64):
        for dy in range(64):
            gpos = cam.display_to_grid(dx, dy)
            if gpos:
                gx, gy = gpos
                if (gx, gy) not in g2d:
                    g2d[(gx, gy)] = (dx, dy)

    # Extract click targets from sprites
    click_targets = []
    for s in game.current_level.get_sprites():
        if s.is_visible and s.width > 0 and s.width <= 10 and s.height <= 10:
            c = int(s.pixels[min(1, s.height-1), min(1, s.width-1)])
            if c >= 0:
                gcx, gcy = s.x + s.width//2, s.y + s.height//2
                dpos = g2d.get((gcx, gcy))
                if dpos:
                    click_targets.append({'x': dpos[0], 'y': dpos[1], 'color': c})

    print(f"  {len(click_targets)} click targets detected")

    # Probe: try nav sequences + consequential actions
    # cd82 uses UP/DOWN/LEFT/RIGHT (move cursor) + SELECT (stamp) + CLICK (color)
    nav_ids = [a for a in available if ACTION_NAMES.get(a) in ('UP', 'DOWN', 'LEFT', 'RIGHT')]

    # Try depth-2 nav + SELECT (the winning pattern for cd82)
    from itertools import product

    best_win = None
    best_len = 999

    for depth in range(0, 5):
        if best_win:
            break
        nav_combos = list(product(nav_ids, repeat=depth)) if depth > 0 else [()]
        for nav_seq in nav_combos:
            if best_win:
                break
            # Try SELECT after each nav sequence
            select_id = next((a for a in available if ACTION_NAMES.get(a) == 'SELECT'), None)
            if select_id is None:
                continue

            probe_env = arcade.make(game_id)
            probe_fd = probe_env.reset()

            actions = []
            for nav in nav_seq:
                probe_fd = probe_env.step(int_to_ga(nav))
                actions.append({'action': nav})

            probe_fd = probe_env.step(int_to_ga(select_id))
            actions.append({'action': select_id})

            if probe_fd.levels_completed > 0:
                if len(actions) < best_len:
                    best_win = actions
                    best_len = len(actions)
                    print(f"  WIN at depth {depth}: {[ACTION_NAMES.get(a['action']) for a in actions]}")

    if not best_win:
        # Try CLICK targets too
        for ct in click_targets[:5]:
            click_id = next((a for a in available if ACTION_NAMES.get(a) == 'CLICK'), None)
            if not click_id:
                break
            for depth in range(0, 4):
                nav_combos = list(product(nav_ids, repeat=depth)) if depth > 0 else [()]
                for nav_seq in nav_combos:
                    probe_env = arcade.make(game_id)
                    probe_fd = probe_env.reset()

                    actions = []
                    # Click color first
                    probe_fd = probe_env.step(int_to_ga(click_id), data={'x': ct['x'], 'y': ct['y']})
                    actions.append({'action': click_id, 'data': {'x': ct['x'], 'y': ct['y']}})

                    for nav in nav_seq:
                        probe_fd = probe_env.step(int_to_ga(nav))
                        actions.append({'action': nav})

                    select_id = next((a for a in available if ACTION_NAMES.get(a) == 'SELECT'), None)
                    if select_id:
                        probe_fd = probe_env.step(int_to_ga(select_id))
                        actions.append({'action': select_id})

                    if probe_fd.levels_completed > 0:
                        if len(actions) < best_len:
                            best_win = actions
                            best_len = len(actions)
                            print(f"  WIN with click+nav: {[ACTION_NAMES.get(a['action']) for a in actions]}")

    if best_win:
        print(f"  Best winning sequence: {best_len} actions")
    else:
        print("  No winning sequence found in probe space")

    return best_win, game_id


def test_habit_compilation(winning_actions, game_id):
    """Train a habit from the winning sequence, verify it compiles and replays."""
    from arc_agi import Arcade

    arcade = Arcade()
    cb = ArcCerebellum(maturity_threshold=3)

    # Build state signature
    env = arcade.make(game_id)
    fd = env.reset()
    state = extract_state_from_env(env)

    print(f"\n--- Habit Training ---")
    print(f"State: domain={state.domain}, features={state.features}")
    print(f"Action sequence: {len(winning_actions)} steps")

    # Observe the winning sequence 3 times
    for i in range(3):
        success, result = cb.execute(
            type('H', (), {
                'action_sequence': winning_actions,
                'last_fired': 0.0,
                'training_count': 0,
                'success_count': 0,
            })(),
            arcade.make(game_id)
        )

        # Instead, just observe manually
        env_i = arcade.make(game_id)
        fd_i = env_i.reset()
        state_i = extract_state_from_env(env_i)

        # Replay and check
        for step in winning_actions:
            action_int = step['action']
            data = step.get('data', {})
            ga = int_to_ga(action_int)
            if data:
                fd_i = env_i.step(ga, data=data)
            else:
                fd_i = env_i.step(ga)

        won = fd_i.levels_completed > 0
        cb.observe(state_i, winning_actions, {"success": won, "summary": f"cd82 L1 attempt {i+1}"})
        print(f"  Observation {i+1}: {'WON' if won else 'lost'}")

    # Check habit compiled
    matches = cb.lookup(state)
    print(f"\n--- Habit Lookup ---")
    print(f"Matches found: {len(matches)}")

    if matches:
        h = matches[0]
        print(f"  Habit: {h.habit.habit_id}")
        print(f"  Mature: {h.habit.is_mature}")
        print(f"  Reliability: {h.habit.reliability:.2f}")
        print(f"  Confidence: {h.confidence:.2f}")
        print(f"  Actions: {len(h.habit.action_sequence)}")

    # Now test habit replay on a FRESH game
    print(f"\n--- Habit Replay (fresh game) ---")
    fresh_env = arcade.make(game_id)
    fresh_fd = fresh_env.reset()

    if matches:
        success, result = cb.execute(matches[0].habit, fresh_env)
        print(f"  Replay result: {'WON!' if result.get('success') else 'did not win'}")
        print(f"  Actions replayed: {result.get('actions_replayed', '?')}")
        print(f"  Levels completed: {result.get('levels_completed', '?')}")
        return result.get('success', False)

    return False


def test_persistence_roundtrip(cb):
    """Verify habit survives save/load."""
    import tempfile

    data = cb.export()
    cb2 = ArcCerebellum.load(data)
    print(f"\n--- Persistence ---")
    print(f"  Original: {cb.habit_count} habits, {cb.mature_count} mature")
    print(f"  Loaded:   {cb2.habit_count} habits, {cb2.mature_count} mature")
    return cb2.habit_count == cb.habit_count


def main():
    print("=" * 60)
    print("CEREBELLUM ↔ ARC-AGI-3 INTEGRATION TEST")
    print("=" * 60)

    # Step 1: Discover winning sequence
    print("\n[1] Discovering cd82 L1 winning sequence...")
    winning_actions, game_id = discover_cd82_winning_sequence()

    if not winning_actions:
        print("FAIL: Could not discover winning sequence")
        return False

    # Step 2: Train and verify habit
    print("\n[2] Training habit from winning sequence...")
    replay_won = test_habit_compilation(winning_actions, game_id)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  Discovery:    {'PASS' if winning_actions else 'FAIL'}")
    print(f"  Habit replay: {'PASS' if replay_won else 'FAIL'}")
    print("=" * 60)

    return winning_actions is not None and replay_won


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
