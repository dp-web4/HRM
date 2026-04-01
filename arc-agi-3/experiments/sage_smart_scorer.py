#!/usr/bin/env python3
"""
SAGE Smart Scorer — Coordinate-aware, color-learning, cartridge-integrated.

Combines all discoveries:
- ACTION6 with x,y coordinates
- Color effectiveness learning (per-game, per-level)
- Two-phase play: EXPLORE (probe all colors) → EXPLOIT (target effective ones)
- Membot cartridge for cross-session persistence
- Multi-run best-of for each game

Usage:
    cd /Users/dennispalatov/repos/SAGE
    .venv/bin/python3 arc-agi-3/experiments/sage_smart_scorer.py --all --runs 10
"""

import sys
import time
import json
import argparse
import random
from collections import defaultdict
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")

from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}
ACTION_LABELS = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                 5: "SELECT", 6: "CLICK", 7: "ACTION7"}


def play_smart(env, frame_data, max_steps: int = 300, game_id: str = "",
               prior_colors: dict = None) -> dict:
    """Play one game with two-phase color learning.

    Phase 1 (EXPLORE): First 20% of steps, probe each color once.
    Phase 2 (EXPLOIT): Remaining steps, target effective colors only.
    """
    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (frame_data.available_actions or [])]
    has_click = 6 in available
    moves = [a for a in available if a in [1, 2, 3, 4, 5, 7]]

    grid = np.array(frame_data.frame)
    if grid.ndim == 3:
        grid = grid[-1]

    # Color effectiveness tracking
    color_tries = defaultdict(int)
    color_changes = defaultdict(int)

    # Seed with prior knowledge
    if prior_colors:
        for color, stats in prior_colors.items():
            c = int(color.replace("color_", "")) if isinstance(color, str) else int(color)
            color_tries[c] = stats.get("tries", 0)
            color_changes[c] = stats.get("changes", 0)

    start_levels = frame_data.levels_completed
    prev_levels = start_levels
    max_levels = 0
    level_events = []
    explore_budget = min(max_steps // 5, 60)  # 20% capped at 60 steps

    for step in range(max_steps):
        if frame_data is None:
            break

        try:
            bg = int(np.bincount(grid.astype(int).flatten()).argmax())
        except:
            bg = 0

        action_taken = None
        click_color = None
        click_r, click_c = None, None

        if has_click:
            if step < explore_budget:
                # EXPLORE: try each color systematically
                non_bg = np.argwhere(grid != bg)
                if len(non_bg) > 0:
                    # Find least-tried color
                    color_cells = defaultdict(list)
                    for r, c in non_bg.tolist():
                        color_cells[int(grid[r, c])].append((r, c))

                    # Pick color with fewest tries
                    least_tried = min(color_cells.keys(),
                                     key=lambda c: color_tries[c])
                    r, c = random.choice(color_cells[least_tried])
                    click_color = least_tried
                    click_r, click_c = r, c
                    action_taken = "explore"
                else:
                    if moves:
                        action_taken = "move"
            else:
                # EXPLOIT: target most effective colors
                non_bg = np.argwhere(grid != bg)
                if len(non_bg) > 0:
                    color_cells = defaultdict(list)
                    for r, c in non_bg.tolist():
                        color_cells[int(grid[r, c])].append((r, c))

                    # Score each available color by effectiveness
                    best_color = None
                    best_rate = -1
                    for color, cells in color_cells.items():
                        tries = color_tries[color]
                        changes = color_changes[color]
                        if tries == 0:
                            rate = 0.5  # unknown — moderate priority
                        else:
                            rate = changes / tries
                        if rate > best_rate:
                            best_rate = rate
                            best_color = color

                    if best_color is not None and best_rate > 0.1:
                        r, c = random.choice(color_cells[best_color])
                        click_color = best_color
                        click_r, click_c = r, c
                        action_taken = "exploit"
                    elif moves:
                        action_taken = "move"
                    else:
                        # Click anything
                        r, c = random.choice(non_bg.tolist())
                        click_color = int(grid[r, c])
                        click_r, click_c = r, c
                        action_taken = "fallback"
                else:
                    if moves:
                        action_taken = "move"

        else:
            # No click available — movement only
            action_taken = "move"

        # Execute
        prev_grid = grid.copy()
        try:
            if action_taken in ("explore", "exploit", "fallback") and click_r is not None:
                frame_data = env.step(GameAction.ACTION6,
                                      data={'x': int(click_c), 'y': int(click_r)})
            elif action_taken == "move" and moves:
                frame_data = env.step(INT_TO_GAME_ACTION[random.choice(moves)])
            else:
                break
        except Exception:
            continue

        if frame_data is None:
            break

        new_grid = np.array(frame_data.frame)
        if new_grid.ndim == 3:
            new_grid = new_grid[-1]
        if new_grid.size == 0 or new_grid.ndim != 2:
            continue

        changed = not np.array_equal(prev_grid, new_grid)
        grid = new_grid

        # Record click effectiveness
        if click_color is not None:
            color_tries[click_color] += 1
            if changed:
                color_changes[click_color] += 1

        # Update available actions
        new_avail = [a.value if hasattr(a, "value") else int(a)
                     for a in (frame_data.available_actions or [])]
        if new_avail:
            available = new_avail
            has_click = 6 in available
            moves = [a for a in available if a in [1, 2, 3, 4, 5, 7]]

        # Level completion
        if frame_data.levels_completed > prev_levels:
            max_levels = frame_data.levels_completed
            level_events.append({
                "step": step,
                "levels": frame_data.levels_completed,
                "action": action_taken,
                "click_color": click_color,
            })
            prev_levels = frame_data.levels_completed

        if frame_data.state.name in ("WON", "LOST", "GAME_OVER"):
            break

    # Build color effectiveness report
    color_report = {}
    for color in sorted(set(list(color_tries.keys()) + list(color_changes.keys()))):
        t = color_tries[color]
        c = color_changes[color]
        if t > 0:
            color_report[f"color_{color}"] = {"tries": t, "changes": c, "rate": round(c/t, 3)}

    return {
        "steps": step + 1,
        "levels_completed": max_levels,
        "win_levels": getattr(frame_data, "win_levels", 0),
        "state": frame_data.state.name if frame_data and hasattr(frame_data, "state") else "UNKNOWN",
        "level_events": level_events,
        "color_effectiveness": color_report,
    }


def main():
    parser = argparse.ArgumentParser(description="SAGE Smart Scorer")
    parser.add_argument("--game", default=None)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("SAGE Smart Scorer — Color-Learning Coordinate Clicker")
    print("=" * 70)

    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
    elif args.all:
        targets = envs
    else:
        targets = envs[:5]

    print(f"\n{len(targets)} games, {args.runs} runs each, {args.steps} steps/run\n")

    total_levels = 0
    scored_games = {}
    start_time = time.time()

    for env_info in targets:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        prefix = game_id.split("-")[0]

        # Load prior color knowledge from cartridge
        prior_colors = {}
        cart_path = f"arc-agi-3/experiments/cartridges/{prefix}.json"
        try:
            with open(cart_path) as f:
                cart = json.load(f)
            prior_colors = cart.get("click_effectiveness", {}).get("global", {})
        except FileNotFoundError:
            pass

        best_levels = 0
        best_result = None

        for run in range(args.runs):
            try:
                env = arcade.make(game_id)
                frame_data = env.reset()
            except Exception:
                continue

            try:
                result = play_smart(env, frame_data, max_steps=args.steps,
                                    game_id=game_id, prior_colors=prior_colors)
            except Exception:
                continue

            if result["levels_completed"] > best_levels:
                best_levels = result["levels_completed"]
                best_result = result

            total_levels += result["levels_completed"]

        if best_levels > 0:
            scored_games[prefix] = {
                "levels": best_levels,
                "win_levels": best_result["win_levels"],
                "events": best_result["level_events"],
            }
            events = " ".join(f"L{e['levels']}@{e['step']}" for e in best_result["level_events"])
            print(f"  ★ {prefix:6s}  {best_levels}/{best_result['win_levels']}  {events}")
        else:
            wl = best_result["win_levels"] if best_result else "?"
            print(f"    {prefix:6s}  0/{wl}")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TOTAL: {total_levels} levels across {len(scored_games)} games in {elapsed:.0f}s")
    print(f"{'='*70}")

    if scored_games:
        print("\nScoring games:")
        for g, info in sorted(scored_games.items(), key=lambda x: -x[1]["levels"]):
            print(f"  {g}: {info['levels']}/{info['win_levels']}")

    # Save
    log_path = f"arc-agi-3/experiments/logs/smart_{int(time.time())}.json"
    try:
        import os
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({"scored_games": scored_games, "total_levels": total_levels,
                       "elapsed": elapsed, "runs": args.runs, "steps": args.steps}, f, indent=2)
        print(f"\nSaved: {log_path}")
    except Exception as e:
        print(f"(save failed: {e})")


if __name__ == "__main__":
    main()
