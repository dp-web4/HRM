#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Multi-Game Scorer — Sprout Edition

Aggressive scoring strategy with membot cartridge persistence:
1. PROBE: 1 click per non-bg color to find effective colors (minimal budget)
2. SWEEP: Click ALL cells of effective colors systematically
3. ADAPT: After level-up, re-probe the new grid (colors may change)
4. PERSIST: Store effective colors in cartridge for cross-run learning

Key insight from survey: 15/19 click games have effective colors.
Challenge: "effective" (causes change) != "scoring" (causes level-up).
Need to find the RIGHT effective colors and click pattern.
"""
import sys
import os
import time
import json
import random
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from arcengine import GameAction
from arc_agi import Arcade
from membot_cartridge import MembotCartridge

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}


def get_frame(fd):
    grid = np.array(fd.frame)
    if grid.ndim == 3:
        grid = grid[-1]
    return grid


def probe_colors(env, grid, max_probes=15):
    """Probe each non-bg color with 1 click. Returns {color: cells_changed}."""
    bg = int(np.bincount(grid.flatten()).argmax())
    unique = sorted(set(int(c) for c in np.unique(grid)) - {bg})
    results = {}

    for color in unique[:max_probes]:
        positions = np.argwhere(grid == color)
        if len(positions) == 0:
            continue
        # Click center-most cell of this color
        r, c = int(positions[len(positions)//2][0]), int(positions[len(positions)//2][1])
        prev = grid.copy()
        try:
            fd = env.step(GameAction.ACTION6, data={'x': int(c), 'y': int(r)})
        except Exception:
            results[color] = -1
            continue
        new_grid = get_frame(fd)
        changed = int(np.sum(prev != new_grid))
        results[color] = changed
        grid = new_grid

        # Check for level-up during probing!
        if fd.levels_completed > 0:
            results[color] = 1000 + changed  # Mark as level-scorer
            return results, fd, grid

    return results, None, grid


def sweep_color(env, grid, color, fd_current, max_clicks=50):
    """Click ALL cells of a given color. Track changes and level-ups."""
    positions = np.argwhere(grid == color)
    if len(positions) == 0:
        return grid, fd_current, 0, 0, 0

    clicks = 0
    total_changes = 0
    level_ups = 0
    prev_levels = fd_current.levels_completed

    for pos in positions[:max_clicks]:
        r, c = int(pos[0]), int(pos[1])
        prev = grid.copy()
        try:
            fd_current = env.step(GameAction.ACTION6, data={'x': int(c), 'y': int(r)})
        except Exception:
            continue
        grid = get_frame(fd_current)
        changed = int(np.sum(prev != grid))
        total_changes += changed
        clicks += 1

        if fd_current.levels_completed > prev_levels:
            level_ups += fd_current.levels_completed - prev_levels
            prev_levels = fd_current.levels_completed

        if fd_current.state.name in ("WON", "GAME_OVER"):
            break

    return grid, fd_current, clicks, total_changes, level_ups


def sweep_all_non_bg(env, grid, fd_current, exclude=None, max_clicks=50):
    """Click all non-bg cells (no color preference). For desperate exploration."""
    bg = int(np.bincount(grid.flatten()).argmax())
    exclude_set = set(exclude or []) | {bg}
    positions = np.argwhere(~np.isin(grid, list(exclude_set)))
    if len(positions) == 0:
        positions = np.argwhere(grid != bg)
    if len(positions) == 0:
        return grid, fd_current, 0, 0, 0

    np.random.shuffle(positions)
    clicks = 0
    total_changes = 0
    level_ups = 0
    prev_levels = fd_current.levels_completed

    for pos in positions[:max_clicks]:
        r, c = int(pos[0]), int(pos[1])
        prev = grid.copy()
        try:
            fd_current = env.step(GameAction.ACTION6, data={'x': int(c), 'y': int(r)})
        except Exception:
            continue
        grid = get_frame(fd_current)
        changed = int(np.sum(prev != grid))
        total_changes += changed
        clicks += 1

        if fd_current.levels_completed > prev_levels:
            level_ups += fd_current.levels_completed - prev_levels
            prev_levels = fd_current.levels_completed

        if fd_current.state.name in ("WON", "GAME_OVER"):
            break

    return grid, fd_current, clicks, total_changes, level_ups


def play_movement_game(env, fd, max_steps=100):
    """For movement-only games: try systematic movement patterns."""
    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (fd.available_actions or [])]
    grid = get_frame(fd)
    max_levels = fd.levels_completed
    prev_levels = fd.levels_completed

    # Try movement patterns: repeated single direction, then combos
    patterns = [
        [1]*10, [2]*10, [3]*10, [4]*10,  # Single direction
        [1,1,1,5]*3, [2,2,2,5]*3,        # Direction + select
        [1,2]*5, [3,4]*5,                  # Alternating
        [5]*5,                              # Pure select
    ]

    steps = 0
    for pattern in patterns:
        for action in pattern:
            if action not in available:
                continue
            try:
                fd = env.step(INT_TO_GAME_ACTION[action])
            except Exception:
                continue
            grid = get_frame(fd)
            steps += 1

            if fd.levels_completed > max_levels:
                max_levels = fd.levels_completed

            if fd.state.name in ("WON", "GAME_OVER"):
                return max_levels, fd.win_levels, steps, fd.state.name

            if steps >= max_steps:
                return max_levels, fd.win_levels, steps, fd.state.name

    # Random fill remaining budget
    while steps < max_steps:
        action = random.choice(available)
        try:
            fd = env.step(INT_TO_GAME_ACTION[action])
        except Exception:
            continue
        steps += 1
        if fd.levels_completed > max_levels:
            max_levels = fd.levels_completed
        if fd.state.name in ("WON", "GAME_OVER"):
            break

    return max_levels, fd.win_levels, steps, fd.state.name


def play_click_game(env, fd, game_id, cartridge_data=None, verbose=False):
    """Play a click game with probe→sweep strategy."""
    grid = get_frame(fd)
    prefix = game_id.split("-")[0]
    max_levels = fd.levels_completed
    win_levels = fd.win_levels
    total_steps = 0
    scoring_colors = []

    # Load known effective colors from cartridge
    known_effective = set()
    if cartridge_data and "color_effectiveness" in cartridge_data:
        for color_str, stats in cartridge_data.get("color_effectiveness", {}).items():
            if stats.get("level_ups", 0) > 0:
                known_effective.add(int(color_str))

    round_num = 0
    while fd.state.name not in ("WON", "GAME_OVER"):
        round_num += 1

        if known_effective and round_num == 1:
            # Use known scoring colors first
            if verbose:
                print(f"    Round {round_num}: Using known colors {known_effective}")
            for color in known_effective:
                grid, fd, clicks, changes, lvls = sweep_color(
                    env, grid, color, fd, max_clicks=100)
                total_steps += clicks
                if lvls > 0:
                    scoring_colors.append(color)
                if fd.levels_completed > max_levels:
                    max_levels = fd.levels_completed
                if fd.state.name in ("WON", "GAME_OVER"):
                    break
            if fd.state.name in ("WON", "GAME_OVER"):
                break

        # Probe phase
        probe_result, probe_fd, grid = probe_colors(env, grid)
        total_steps += len(probe_result)
        if probe_fd:
            fd = probe_fd
            if fd.levels_completed > max_levels:
                max_levels = fd.levels_completed
            # Color that scored during probing
            for c, v in probe_result.items():
                if v >= 1000:
                    scoring_colors.append(c)

        effective = sorted([c for c, v in probe_result.items() if isinstance(v, int) and v > 0])
        if verbose:
            print(f"    Round {round_num}: Probe results: {probe_result}")
            print(f"    Effective colors: {effective}")

        if fd.state.name in ("WON", "GAME_OVER"):
            break

        # Sweep effective colors
        if effective:
            for color in effective:
                grid, fd, clicks, changes, lvls = sweep_color(
                    env, grid, color, fd, max_clicks=80)
                total_steps += clicks
                if lvls > 0:
                    scoring_colors.append(color)
                    if verbose:
                        print(f"    ★ Color {color} scored! Level {fd.levels_completed}/{fd.win_levels}")
                if fd.levels_completed > max_levels:
                    max_levels = fd.levels_completed
                if fd.state.name in ("WON", "GAME_OVER"):
                    break
        else:
            # No effective colors found — sweep all non-bg
            grid, fd, clicks, changes, lvls = sweep_all_non_bg(
                env, grid, fd, max_clicks=30)
            total_steps += clicks
            if lvls > 0 and verbose:
                print(f"    ★ Level from non-bg sweep! {fd.levels_completed}/{fd.win_levels}")

        if fd.levels_completed > max_levels:
            max_levels = fd.levels_completed

        if fd.state.name in ("WON", "GAME_OVER"):
            break

        # Safety: don't loop forever
        if total_steps > 2000:
            break

    return {
        "levels": max_levels,
        "win_levels": win_levels or fd.win_levels,
        "steps": total_steps,
        "state": fd.state.name,
        "scoring_colors": list(set(scoring_colors)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAGE ARC-AGI-3 Multi-Game Scorer — Sprout")
    parser.add_argument("--game", default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE ARC-AGI-3 Multi-Game Scorer — Sprout")
    print("=" * 60)

    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
    else:
        targets = envs

    print(f"\n{len(targets)} game(s), {args.runs} runs each\n")

    all_results = []
    total_levels = 0
    total_scored_games = 0
    start = time.time()

    for env_info in targets:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        prefix = game_id.split("-")[0]

        # Load cartridge
        cart = MembotCartridge(game_id)
        cart_data = cart.read()

        print(f"{'─'*60}")
        print(f"Game: {game_id}")
        if cart_data.get("color_effectiveness"):
            scoring = {k: v for k, v in cart_data.get("color_effectiveness", {}).items()
                       if v.get("level_ups", 0) > 0}
            if scoring:
                print(f"  Cartridge: known scoring colors = {list(scoring.keys())}")

        best = None
        for run in range(args.runs):
            try:
                env = arcade.make(game_id)
                fd = env.reset()
            except Exception as e:
                print(f"  Run {run+1}: FAILED ({e})")
                continue

            available = [a.value if hasattr(a, "value") else int(a)
                         for a in (fd.available_actions or [])]
            has_click = 6 in available

            t0 = time.time()
            try:
                if has_click:
                    result = play_click_game(env, fd, game_id,
                                             cartridge_data=cart_data,
                                             verbose=args.verbose)
                else:
                    lvl, win, steps, state = play_movement_game(env, fd)
                    result = {"levels": lvl, "win_levels": win, "steps": steps,
                              "state": state, "scoring_colors": []}
            except Exception as e:
                print(f"  Run {run+1}: CRASHED ({e})")
                continue

            elapsed = time.time() - t0
            lvl = result["levels"]
            win = result["win_levels"]
            status = f" ★ {lvl} level(s)!" if lvl > 0 else ""
            if result["state"] == "WON":
                status += " 🏆"

            print(f"  Run {run+1}: {result['steps']} steps, "
                  f"{lvl}/{win} levels{status} ({elapsed:.1f}s)")

            if best is None or result["levels"] > best["levels"]:
                best = result

        # Update cartridge with best result
        if best and best["levels"] > 0:
            total_levels += best["levels"]
            total_scored_games += 1

            # Store scoring colors in cartridge
            if best.get("scoring_colors"):
                if "color_effectiveness" not in cart.data:
                    cart.data["color_effectiveness"] = {}
                for color in best["scoring_colors"]:
                    key = str(color)
                    if key not in cart.data["color_effectiveness"]:
                        cart.data["color_effectiveness"][key] = {"level_ups": 0, "runs": 0}
                    cart.data["color_effectiveness"][key]["level_ups"] += 1
                    cart.data["color_effectiveness"][key]["runs"] += 1
                cart.write()

        all_results.append({"game_id": game_id, **(best or {"levels": 0, "win_levels": 0})})

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"SUMMARY — Sprout Multi-Scorer")
    print(f"{'='*60}")
    for r in sorted(all_results, key=lambda x: x.get("levels", 0), reverse=True):
        lvl = r.get("levels", 0)
        win = r.get("win_levels", 0)
        star = f" ★ {lvl}" if lvl > 0 else ""
        sc = r.get("scoring_colors", [])
        sc_str = f" scoring={sc}" if sc else ""
        print(f"  {r['game_id'].split('-')[0]:6s}  {lvl}/{win}{star}{sc_str}")

    print(f"\n  Total: {total_levels} levels across {total_scored_games} games in {elapsed:.0f}s")

    # Save
    log_path = f"arc-agi-3/experiments/logs/multi_scorer_{int(time.time())}.json"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "machine": "sprout",
            "total_levels": total_levels,
            "games_scored": total_scored_games,
            "elapsed_s": round(elapsed, 1),
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
