#!/usr/bin/env python3
"""
SAGE Spatial Scorer — Uses spatial reasoning for movement+click games.

Three-phase strategy:
1. SURVEY: Click each object type once to discover interactive elements
2. NAVIGATE: Use movement to reposition, then click interactive objects
3. COMBINE: Try movement→click sequences targeting spatial relationships

Designed for the 11 movement+click games that score 0 with pure clicking.

Usage:
    cd /Users/dennispalatov/repos/SAGE
    .venv/bin/python3 arc-agi-3/experiments/sage_spatial_scorer.py --all --runs 5
"""

import sys
import time
import json
import argparse
import random
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_frame, background_color
from arc_spatial import SpatialTracker

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}


def play_spatial(env, frame_data, max_steps: int = 300) -> dict:
    """Play with spatial awareness — survey, navigate, combine."""
    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (frame_data.available_actions or [])]
    has_click = 6 in available
    has_move = any(a in available for a in [1, 2, 3, 4])
    has_select = 5 in available

    if not has_click or not has_move:
        return {"steps": 0, "levels_completed": 0, "win_levels": 0,
                "state": "SKIP", "reason": "not a movement+click game"}

    grid = get_frame(frame_data)
    tracker = SpatialTracker(min_region_size=4)
    tracker.update(grid)

    moves = [a for a in available if a in [1, 2, 3, 4]]
    start_levels = frame_data.levels_completed
    max_levels = 0
    level_events = []

    # Phase budgets
    survey_budget = min(max_steps // 4, 50)
    navigate_budget = max_steps - survey_budget

    for step in range(max_steps):
        prev_grid = grid.copy()
        prev_levels = frame_data.levels_completed
        action_data = None

        if step < survey_budget:
            # SURVEY: systematically click each object type
            targets = tracker.suggest_click_targets(grid, n=1)
            untested = tracker.get_untested_objects()

            if untested and has_click:
                # Click the smallest untested object (likely a button)
                obj = sorted(untested, key=lambda o: o.size)[0]
                action_int = 6
                action_data = {'x': obj.cx, 'y': obj.cy}
            elif targets and has_click:
                x, y, reason = targets[0]
                action_int = 6
                action_data = {'x': x, 'y': y}
            else:
                action_int = random.choice(moves)
        else:
            # NAVIGATE + COMBINE: move then click interactive objects
            interactive = tracker.get_interactive_objects()

            if interactive and has_click:
                # 60% click interactive, 40% move then click
                if random.random() < 0.6:
                    obj = random.choice(interactive)
                    action_int = 6
                    action_data = {'x': obj.cx, 'y': obj.cy}
                else:
                    # Move first, then we'll click next step
                    action_int = random.choice(moves)
            elif has_click:
                # No known interactive objects — try spatial inference
                # Click objects near the center (game pieces often there)
                bg = background_color(grid)
                non_bg = np.argwhere(grid != bg)
                if len(non_bg) > 0:
                    # Prefer cells near grid center
                    center_r, center_c = grid.shape[0] // 2, grid.shape[1] // 2
                    dists = np.abs(non_bg[:, 0] - center_r) + np.abs(non_bg[:, 1] - center_c)
                    near_center = non_bg[np.argsort(dists)]
                    r, c = near_center[random.randint(0, min(20, len(near_center)-1))].tolist()
                    action_int = 6
                    action_data = {'x': int(c), 'y': int(r)}
                else:
                    action_int = random.choice(moves)
            else:
                action_int = random.choice(moves)

            # Every 10 steps in navigate phase, try a select/submit
            if has_select and step % 10 == 0 and step > survey_budget:
                action_int = 5
                action_data = None

        # Execute
        try:
            if action_data and action_int == 6:
                frame_data = env.step(GameAction.ACTION6, data=action_data)
            else:
                frame_data = env.step(INT_TO_GAME_ACTION[action_int])
        except Exception:
            continue

        if frame_data is None:
            break

        grid = get_frame(frame_data)
        changed = not np.array_equal(prev_grid, grid)
        n_changed = int(np.sum(prev_grid != grid))

        # Update spatial tracker
        diff = tracker.update(grid)

        # Record click outcome
        if action_data and action_int == 6:
            tracker.record_click(action_data['x'], action_data['y'], changed, n_changed)

        tracker.record_action_outcome(action_int, action_data, diff)

        # Update available actions
        new_avail = [a.value if hasattr(a, "value") else int(a)
                     for a in (frame_data.available_actions or [])]
        if new_avail:
            available = new_avail
            has_click = 6 in available
            has_move = any(a in available for a in [1, 2, 3, 4])
            has_select = 5 in available
            moves = [a for a in available if a in [1, 2, 3, 4]]

        # Level completion
        if frame_data.levels_completed > prev_levels:
            max_levels = frame_data.levels_completed
            level_events.append({"step": step, "levels": max_levels})

        if frame_data.state.name in ("WON", "LOST", "GAME_OVER"):
            break

    return {
        "steps": step + 1,
        "levels_completed": max_levels,
        "win_levels": getattr(frame_data, "win_levels", 0),
        "state": frame_data.state.name if frame_data else "UNKNOWN",
        "level_events": level_events,
        "interactive_objects": len(tracker.get_interactive_objects()),
        "movement_pattern": tracker.get_movement_pattern(),
        "spatial_summary": tracker.describe(),
    }


def main():
    parser = argparse.ArgumentParser(description="SAGE Spatial Scorer")
    parser.add_argument("--game", default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("SAGE Spatial Scorer — Movement+Click Games")
    print("=" * 70)

    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
    elif args.all:
        # Focus on movement+click games
        targets = []
        for e in envs:
            env = arcade.make(e.game_id)
            f = env.reset()
            avail = [a.value if hasattr(a, "value") else int(a) for a in (f.available_actions or [])]
            if 6 in avail and any(a in avail for a in [1, 2, 3, 4]):
                targets.append(e)
        print(f"\nFound {len(targets)} movement+click games")
    else:
        targets = envs[:5]

    print(f"{len(targets)} games, {args.runs} runs, {args.steps} steps/run\n")

    total_levels = 0
    scored = {}

    for env_info in targets:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        prefix = game_id.split("-")[0]
        best = 0
        best_result = None

        for run in range(args.runs):
            try:
                env = arcade.make(game_id)
                frame_data = env.reset()
                result = play_spatial(env, frame_data, max_steps=args.steps)
            except Exception:
                continue

            if result.get("state") == "SKIP":
                break

            levels = result["levels_completed"]
            total_levels += levels
            if levels > best:
                best = levels
                best_result = result

        if best > 0:
            scored[prefix] = best
            events = " ".join(f"L{e['levels']}@{e['step']}" for e in best_result["level_events"])
            print(f"  ★ {prefix:6s}  {best}/{best_result['win_levels']}  {events}")
            if args.verbose and best_result:
                print(f"    Interactive: {best_result['interactive_objects']} objects")
                if best_result['movement_pattern']:
                    print(f"    Movement: {best_result['movement_pattern']}")
        else:
            wl = best_result["win_levels"] if best_result else "?"
            if args.verbose and best_result:
                print(f"    {prefix:6s}  0/{wl}  interactive={best_result['interactive_objects']}, "
                      f"pattern={best_result.get('movement_pattern', 'none')}")
            else:
                print(f"    {prefix:6s}  0/{wl}")

    print(f"\n{'='*70}")
    print(f"TOTAL: {total_levels} levels across {len(scored)} games")

    if scored:
        print("\nScoring games:")
        for g, l in sorted(scored.items(), key=lambda x: -x[1]):
            print(f"  {g}: {l}")

    # Save
    log_path = f"arc-agi-3/experiments/logs/spatial_{int(time.time())}.json"
    try:
        import os
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({"scored": scored, "total_levels": total_levels}, f, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
