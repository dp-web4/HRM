#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Clicker — Coordinate-based ACTION6 explorer.

KEY DISCOVERY: ACTION6 is a ComplexAction that takes x,y coordinates.
Without coordinates, it does nothing. With coordinates, it clicks on
a specific grid cell. This is why 5M random steps scored 0 levels.

Strategy:
1. Identify non-background colored cells
2. Systematically click on them
3. Track which clicks cause the most changes
4. Detect level completions
5. Persist learned click patterns

Usage:
    cd /Users/dennispalatov/repos/SAGE
    .venv/bin/python3 arc-agi-3/experiments/sage_clicker.py --all --steps 5000
"""

import sys
import time
import json
import hashlib
import argparse
import random
from collections import defaultdict
import numpy as np

sys.path.insert(0, ".")

from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}
ACTION_LABELS = {
    1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
    5: "SELECT", 6: "CLICK", 7: "ACTION7",
}

EXPERIENCE_DIR = "arc-agi-3/experiments/experience"


def find_clickable_targets(grid: np.ndarray) -> list:
    """Find interesting positions to click on.

    Returns list of (row, col, color) tuples, prioritizing:
    1. Non-background colors
    2. Color boundaries (where colors meet)
    3. Unique/rare colors
    """
    bg = int(np.bincount(grid.flatten()).argmax())
    targets = []

    # All non-background positions
    non_bg = np.argwhere(grid != bg)
    if len(non_bg) == 0:
        # Grid is uniform — try clicking center and corners
        h, w = grid.shape
        return [(h//2, w//2, int(grid[h//2, w//2])),
                (h//4, w//4, int(grid[h//4, w//4])),
                (3*h//4, 3*w//4, int(grid[3*h//4, 3*w//4]))]

    # Group by color for systematic exploration
    colors_at = defaultdict(list)
    for r, c in non_bg:
        colors_at[int(grid[r, c])].append((int(r), int(c)))

    # Take samples from each color group (diverse exploration)
    for color, positions in colors_at.items():
        # Sample up to 5 per color
        sample = random.sample(positions, min(5, len(positions)))
        for r, c in sample:
            targets.append((r, c, color))

    random.shuffle(targets)
    return targets


def play_with_clicks(env, frame_data, max_steps: int, game_id: str = "",
                     verbose: bool = False) -> dict:
    """Play game using coordinate-aware ACTION6 clicks."""
    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (frame_data.available_actions or [])]
    has_click = 6 in available
    has_movement = any(a in available for a in [1, 2, 3, 4])

    grid = np.array(frame_data.frame)
    if grid.ndim == 3:
        grid = grid[-1]

    start_levels = frame_data.levels_completed
    prev_levels = start_levels
    level_events = []
    total_changes = 0
    effective_actions = 0
    step = 0

    # Click effectiveness tracking
    color_click_stats = defaultdict(lambda: {"tries": 0, "changes": 0})

    recent_actions = []  # For sequence tracking on level-up

    for step in range(max_steps):
        prev_grid = grid.copy()
        action_desc = ""

        if has_click:
            # Find targets to click
            targets = find_clickable_targets(grid)
            if targets:
                r, c, color = targets[step % len(targets)]
                try:
                    frame_data = env.step(GameAction.ACTION6, data={'x': c, 'y': r})
                    action_desc = f"CLICK({r},{c}) color={color}"
                except Exception:
                    # Try a movement action as fallback
                    if has_movement:
                        move = random.choice([a for a in available if a != 6])
                        try:
                            frame_data = env.step(INT_TO_GAME_ACTION[move])
                            action_desc = f"{ACTION_LABELS.get(move, f'A{move}')}"
                        except Exception:
                            continue
                    else:
                        continue
            else:
                # No targets — try random click on grid
                r, c = random.randint(0, 63), random.randint(0, 63)
                try:
                    frame_data = env.step(GameAction.ACTION6, data={'x': c, 'y': r})
                    action_desc = f"CLICK({r},{c}) random"
                except Exception:
                    continue
        elif has_movement:
            # Movement-only game — mix movement with any other available actions
            action = random.choice(available)
            try:
                frame_data = env.step(INT_TO_GAME_ACTION[action])
                action_desc = ACTION_LABELS.get(action, f"A{action}")
            except Exception:
                continue
        else:
            break

        if frame_data is None:
            break

        # Process new frame
        new_grid = np.array(frame_data.frame)
        if new_grid.ndim == 3:
            new_grid = new_grid[-1]
        if new_grid.size == 0 or new_grid.ndim != 2:
            continue

        changed = int(np.sum(prev_grid != new_grid))
        grid = new_grid
        total_changes += changed
        if changed > 0:
            effective_actions += 1

        recent_actions.append(action_desc)
        if len(recent_actions) > 20:
            recent_actions = recent_actions[-20:]

        # Track click effectiveness by color
        if "CLICK" in action_desc and "color=" in action_desc:
            color = int(action_desc.split("color=")[1].split(")")[0])
            color_click_stats[color]["tries"] += 1
            if changed > 0:
                color_click_stats[color]["changes"] += 1

        # Update available actions
        new_avail = [a.value if hasattr(a, "value") else int(a)
                     for a in (frame_data.available_actions or [])]
        if new_avail:
            available = new_avail
            has_click = 6 in available
            has_movement = any(a in available for a in [1, 2, 3, 4])

        # Level completion!
        if frame_data.levels_completed > prev_levels:
            window = recent_actions[-15:]
            level_events.append({
                "step": step,
                "levels_before": prev_levels,
                "levels_after": frame_data.levels_completed,
                "action_window": window,
            })
            if verbose:
                print(f"    ★ LEVEL {frame_data.levels_completed}/{frame_data.win_levels} at step {step}!")
                print(f"      Recent: {' | '.join(window[-5:])}")
            prev_levels = frame_data.levels_completed

        if frame_data.state.name in ("WON", "LOST"):
            break

        if verbose and step > 0 and step % 1000 == 0:
            rate = effective_actions / max(step, 1)
            print(f"    Step {step}: levels={frame_data.levels_completed}/{frame_data.win_levels}, "
                  f"effective={rate:.1%}, changes={total_changes}")

    # Color click effectiveness
    click_report = {}
    for color, stats in sorted(color_click_stats.items()):
        rate = stats["changes"] / max(stats["tries"], 1)
        click_report[color] = {"tries": stats["tries"], "changes": stats["changes"], "rate": round(rate, 3)}

    return {
        "steps": step + 1,
        "levels_completed": getattr(frame_data, 'levels_completed', 0),
        "win_levels": getattr(frame_data, 'win_levels', 0),
        "state": frame_data.state.name if hasattr(frame_data, 'state') else "UNKNOWN",
        "level_events": level_events,
        "total_levels_gained": getattr(frame_data, 'levels_completed', 0) - start_levels,
        "total_changes": total_changes,
        "effective_rate": effective_actions / max(step + 1, 1),
        "click_by_color": click_report,
    }


def main():
    parser = argparse.ArgumentParser(description="SAGE ARC-AGI-3 Clicker")
    parser.add_argument("--game", default=None, help="Game ID prefix")
    parser.add_argument("--steps", type=int, default=5000, help="Steps per run")
    parser.add_argument("--runs", type=int, default=3, help="Runs per game")
    parser.add_argument("--all", action="store_true", help="All games")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("SAGE ARC-AGI-3 Clicker — Coordinate-Based ACTION6 Explorer")
    print("=" * 70)

    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
    elif args.all:
        targets = envs
    else:
        # Focus on games with ACTION6
        targets = envs

    print(f"\nTargeting {len(targets)} game(s), {args.runs} runs, {args.steps} steps/run\n")

    all_results = []
    total_levels = 0
    start_time = time.time()

    for env_info in targets:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        print(f"{'─'*70}")
        print(f"Game: {game_id}")

        best_run = None
        for run in range(args.runs):
            try:
                env = arcade.make(game_id)
                frame_data = env.reset()
            except Exception as e:
                print(f"  Run {run+1}: FAILED ({e})")
                continue

            t0 = time.time()
            try:
                result = play_with_clicks(env, frame_data, args.steps,
                                          game_id=game_id, verbose=args.verbose)
            except Exception as e:
                print(f"  Run {run+1}: CRASHED ({e})")
                continue
            elapsed = time.time() - t0

            levels = result["total_levels_gained"]
            total_levels += levels
            rate = result["steps"] / max(elapsed, 0.001)

            status = f" ★ {levels} level(s)!" if levels > 0 else ""
            if result["state"] == "WON":
                status += " 🏆"

            print(f"  Run {run+1}: {result['steps']} steps ({rate:.0f}/s), "
                  f"{result['levels_completed']}/{result['win_levels']} levels, "
                  f"effective={result['effective_rate']:.0%}{status}")

            if result["click_by_color"]:
                for color, stats in sorted(result["click_by_color"].items()):
                    print(f"    color {color}: {stats['changes']}/{stats['tries']} = {stats['rate']:.0%}")

            if best_run is None or result["levels_completed"] > best_run["levels_completed"]:
                best_run = result

        if best_run:
            all_results.append({"game_id": game_id, **best_run})
        else:
            all_results.append({"game_id": game_id, "levels_completed": 0, "win_levels": 0,
                                "total_levels_gained": 0})

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"SUMMARY: {total_levels} total levels in {elapsed:.0f}s")
    print(f"{'='*70}")
    for r in all_results:
        lvl = f"{r.get('levels_completed',0)}/{r.get('win_levels',0)}"
        status = " ★" if r.get('total_levels_gained', 0) > 0 else ""
        print(f"  {r['game_id']:20s}  {lvl}{status}")

    # Save
    log_path = f"arc-agi-3/experiments/logs/clicker_{int(time.time())}.json"
    try:
        import os
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({"results": all_results, "total_levels": total_levels}, f, indent=2, default=str)
        print(f"\n  Saved: {log_path}")
    except Exception as e:
        print(f"  (save failed: {e})")


if __name__ == "__main__":
    main()
