#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Smart Clicker — Sprout Edition

Two-phase approach:
1. EXPLORE: Click random non-bg cells, learn which colors work per game
2. EXPLOIT: Focus exclusively on effective colors, maximize level completions

Key insight from McNugget: ACTION6 needs {x, y} coordinates.
Click patterns are game-specific (lp85: only color 8, su15: colors 3+9, etc.)

Usage:
    cd ~/ai-workspace/SAGE
    python3 arc-agi-3/experiments/sprout_smart_clicker.py --all
    python3 arc-agi-3/experiments/sprout_smart_clicker.py --game lp85 --steps 10000
"""

import sys
import os
import time
import json
import random
import argparse
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from arcengine import GameAction

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}
ACTION_LABELS = {1: "UP", 2: "DN", 3: "LF", 4: "RT", 5: "SEL", 6: "CLK", 7: "A7"}

# Known effective colors from McNugget's exploration
KNOWN_PATTERNS = {
    "lp85": {"effective": [8], "ineffective": [1, 2, 3, 5, 9, 10, 11, 15]},
    "su15": {"effective": [3, 9], "ineffective": [0, 4]},
    "sc25": {"effective": [3], "ineffective": [9, 10, 15]},
    "cn04": {"effective": [8, 12, 14], "ineffective": []},
}


def get_frame(fd):
    """Extract 2D grid from frame_data."""
    grid = np.array(fd.frame)
    if grid.ndim == 3:
        grid = grid[-1]
    return grid


def find_color_cells(grid, colors):
    """Find all cells of specified colors."""
    bg = int(np.bincount(grid.flatten()).argmax())
    targets = []
    for color in colors:
        if color == bg:
            continue
        positions = np.argwhere(grid == color)
        for r, c in positions:
            targets.append((int(r), int(c), color))
    return targets


def find_non_bg_cells(grid, exclude_colors=None):
    """Find all non-background cells, optionally excluding certain colors."""
    bg = int(np.bincount(grid.flatten()).argmax())
    exclude = set(exclude_colors or [])
    exclude.add(bg)
    non_bg = np.argwhere(~np.isin(grid, list(exclude)))
    targets = [(int(r), int(c), int(grid[r, c])) for r, c in non_bg]
    random.shuffle(targets)
    return targets


def play_game(env, frame_data, max_steps, game_prefix="", verbose=False):
    """Play a game with adaptive clicking strategy."""
    grid = get_frame(frame_data)
    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (frame_data.available_actions or [])]
    has_click = 6 in available
    has_move = any(a in available for a in [1, 2, 3, 4])

    # Track state
    levels_completed = frame_data.levels_completed
    win_levels = frame_data.win_levels
    max_levels = levels_completed
    level_events = []
    color_stats = defaultdict(lambda: {"tries": 0, "changes": 0, "level_ups": 0})
    effective_colors = set()
    ineffective_colors = set()

    # Load known patterns if available
    for prefix, pattern in KNOWN_PATTERNS.items():
        if prefix in game_prefix:
            effective_colors = set(pattern["effective"])
            ineffective_colors = set(pattern["ineffective"])
            if verbose:
                print(f"  Loaded known pattern: effective={effective_colors}")
            break

    # Phase tracking
    explore_steps = min(200, max_steps // 5) if not effective_colors else 0
    phase = "EXPLORE" if explore_steps > 0 else "EXPLOIT"
    step = 0

    for step in range(max_steps):
        prev_grid = grid.copy()
        prev_levels = levels_completed

        # Switch to exploit after explore phase
        if step == explore_steps and phase == "EXPLORE":
            phase = "EXPLOIT"
            # Determine effective colors from exploration
            for color, stats in color_stats.items():
                if stats["tries"] >= 3:
                    rate = stats["changes"] / stats["tries"]
                    if rate >= 0.5:
                        effective_colors.add(color)
                    elif rate < 0.1:
                        ineffective_colors.add(color)
            if verbose:
                print(f"  Phase switch: effective={effective_colors}, ineffective={ineffective_colors}")

        # Choose target
        if has_click:
            if phase == "EXPLOIT" and effective_colors:
                targets = find_color_cells(grid, effective_colors)
            else:
                targets = find_non_bg_cells(grid, exclude_colors=list(ineffective_colors))

            if not targets:
                targets = find_non_bg_cells(grid)

            if targets:
                r, c, color = targets[step % len(targets)]
                try:
                    frame_data = env.step(GameAction.ACTION6, data={'x': int(c), 'y': int(r)})
                except Exception:
                    if has_move:
                        move = random.choice([a for a in available if a <= 5])
                        try:
                            frame_data = env.step(INT_TO_GAME_ACTION[move])
                        except Exception:
                            continue
                    continue
            else:
                # Random click
                r, c = random.randint(0, 63), random.randint(0, 63)
                color = int(grid[r, c])
                try:
                    frame_data = env.step(GameAction.ACTION6, data={'x': int(c), 'y': int(r)})
                except Exception:
                    continue
        elif has_move:
            action = random.choice(available)
            color = -1
            try:
                frame_data = env.step(INT_TO_GAME_ACTION[action])
            except Exception:
                continue
        else:
            break

        # Process result
        grid = get_frame(frame_data)
        changed = int(np.sum(prev_grid != grid))

        # Update available actions
        new_avail = [a.value if hasattr(a, "value") else int(a)
                     for a in (frame_data.available_actions or [])]
        if new_avail:
            available = new_avail
            has_click = 6 in available
            has_move = any(a in available for a in [1, 2, 3, 4])

        # Track per-color stats
        if color >= 0 and has_click:
            color_stats[color]["tries"] += 1
            if changed > 0:
                color_stats[color]["changes"] += 1

        # Track levels — use max to handle resets
        levels_completed = frame_data.levels_completed
        win_levels = frame_data.win_levels or win_levels
        if levels_completed > max_levels:
            max_levels = levels_completed

        if levels_completed > prev_levels:
            if color >= 0:
                color_stats[color]["level_ups"] += 1
            level_events.append({
                "step": step,
                "level": levels_completed,
                "win_levels": win_levels,
                "color": color,
            })
            if verbose:
                print(f"  ★ LEVEL {levels_completed}/{win_levels} at step {step} (color={color})!")

            # Re-learn effective colors after level up
            for col, stats in color_stats.items():
                if stats["level_ups"] > 0:
                    effective_colors.add(col)

        if frame_data.state.name == "WON":
            if verbose:
                print(f"  🏆 WON at step {step}!")
            break
        elif frame_data.state.name == "GAME_OVER":
            if verbose:
                print(f"  GAME OVER at step {step}")
            break

    # Build report
    click_report = {}
    for color, stats in sorted(color_stats.items()):
        rate = stats["changes"] / max(stats["tries"], 1)
        click_report[color] = {
            "tries": stats["tries"],
            "changes": stats["changes"],
            "rate": round(rate, 3),
            "level_ups": stats["level_ups"],
        }

    return {
        "steps": step + 1,
        "levels_completed": max_levels,
        "win_levels": win_levels,
        "state": frame_data.state.name if hasattr(frame_data, "state") else "?",
        "level_events": level_events,
        "click_by_color": click_report,
        "effective_colors": sorted(effective_colors),
        "ineffective_colors": sorted(ineffective_colors),
    }


def main():
    parser = argparse.ArgumentParser(description="SAGE ARC-AGI-3 Smart Clicker — Sprout")
    parser.add_argument("--game", default=None, help="Game ID prefix")
    parser.add_argument("--steps", type=int, default=5000, help="Steps per game")
    parser.add_argument("--runs", type=int, default=3, help="Runs per game")
    parser.add_argument("--all", action="store_true", help="All games")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE ARC-AGI-3 Smart Clicker — Sprout")
    print("=" * 60)

    from arc_agi import Arcade
    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
    elif args.all:
        targets = envs
    else:
        targets = envs

    print(f"\n{len(targets)} game(s), {args.runs} runs, {args.steps} steps/run\n")

    all_results = []
    total_levels = 0
    total_games_scored = 0
    start = time.time()

    for env_info in targets:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        prefix = game_id.split("-")[0]
        print(f"{'─'*60}")
        print(f"Game: {game_id}")

        best_result = None
        for run in range(args.runs):
            try:
                env = arcade.make(game_id)
                fd = env.reset()
            except Exception as e:
                print(f"  Run {run+1}: FAILED to load ({e})")
                continue

            t0 = time.time()
            try:
                result = play_game(env, fd, args.steps, game_prefix=prefix,
                                   verbose=args.verbose)
            except Exception as e:
                print(f"  Run {run+1}: CRASHED ({e})")
                continue
            elapsed = time.time() - t0

            lvl = result["levels_completed"]
            win = result["win_levels"]
            rate = result["steps"] / max(elapsed, 0.001)
            status = f" ★ {lvl} level(s)!" if lvl > 0 else ""
            if result["state"] == "WON":
                status += " 🏆"

            print(f"  Run {run+1}: {result['steps']} steps ({rate:.0f}/s), "
                  f"{lvl}/{win} levels{status}")

            if args.verbose and result["click_by_color"]:
                for color, stats in sorted(result["click_by_color"].items()):
                    lvl_mark = f" ★{stats['level_ups']}" if stats["level_ups"] > 0 else ""
                    print(f"    color {color}: {stats['changes']}/{stats['tries']} = {stats['rate']:.0%}{lvl_mark}")

            if best_result is None or result["levels_completed"] > best_result["levels_completed"]:
                best_result = result

        if best_result:
            lvl = best_result["levels_completed"]
            if lvl > 0:
                total_levels += lvl
                total_games_scored += 1
            all_results.append({"game_id": game_id, **best_result})
        else:
            all_results.append({"game_id": game_id, "levels_completed": 0, "win_levels": 0})

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"SUMMARY — Sprout (Qwen 3.5 0.8B)")
    print(f"{'='*60}")
    for r in all_results:
        lvl = r.get("levels_completed", 0)
        win = r.get("win_levels", 0)
        status = f" ★ {lvl} level(s)" if lvl > 0 else ""
        eff = sorted(r.get("effective_colors", []))
        eff_str = f"  eff=[{','.join(str(c) for c in eff)}]" if eff else ""
        print(f"  {r['game_id']:20s}  {lvl}/{win}{status}{eff_str}")

    print(f"\n  Total: {total_levels} levels across {total_games_scored} games in {elapsed:.0f}s")

    # Save
    log_path = f"arc-agi-3/experiments/logs/smart_clicker_sprout_{int(time.time())}.json"
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({
                "machine": "sprout",
                "model": "qwen3.5:0.8b",
                "total_levels": total_levels,
                "games_scored": total_games_scored,
                "elapsed_s": round(elapsed, 1),
                "results": all_results,
            }, f, indent=2, default=str)
        print(f"  Log: {log_path}")
    except Exception as e:
        print(f"  (Log failed: {e})")


if __name__ == "__main__":
    main()
