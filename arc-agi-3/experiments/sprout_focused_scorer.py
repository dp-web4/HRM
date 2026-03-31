#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Focused Scorer — Sprout Edition

Key insight: games with tiny budgets (30-60 steps) can't afford to probe
multiple colors. Instead: dedicate ALL clicks to ONE color per run.
Run multiple times with different target colors.

For each game:
1. Get non-bg colors
2. For each color, do a dedicated run: click EVERY cell of that color
3. Track which single-color-focus produces level-ups
4. Store winners in cartridge

Also tries: click-all-same-color, click-in-order (top-left to bottom-right),
and "clear the board" patterns.
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


def click_all_of_color(env, fd, color, max_clicks=500):
    """Click every cell of a given color, top-to-bottom, left-to-right."""
    grid = get_frame(fd)
    max_levels = fd.levels_completed
    steps = 0

    while fd.state.name not in ("WON", "GAME_OVER") and steps < max_clicks:
        positions = np.argwhere(grid == color)
        if len(positions) == 0:
            break
        # Sort top-left to bottom-right
        positions = positions[np.lexsort((positions[:, 1], positions[:, 0]))]

        for pos in positions:
            r, c = int(pos[0]), int(pos[1])
            try:
                fd = env.step(GameAction.ACTION6, data={'x': c, 'y': r})
            except Exception:
                continue
            grid = get_frame(fd)
            steps += 1

            if fd.levels_completed > max_levels:
                max_levels = fd.levels_completed

            if fd.state.name in ("WON", "GAME_OVER"):
                break
            if steps >= max_clicks:
                break

        # After sweeping, check if color is gone or grid changed
        remaining = np.sum(grid == color)
        if remaining == 0 or steps >= max_clicks:
            break

    return max_levels, fd.win_levels, steps, fd.state.name, fd


def click_all_non_bg_sorted(env, fd, max_clicks=500):
    """Click all non-bg cells in order (top-left to bottom-right)."""
    grid = get_frame(fd)
    bg = int(np.bincount(grid.flatten()).argmax())
    max_levels = fd.levels_completed
    steps = 0

    while fd.state.name not in ("WON", "GAME_OVER") and steps < max_clicks:
        non_bg = np.argwhere(grid != bg)
        if len(non_bg) == 0:
            break
        non_bg = non_bg[np.lexsort((non_bg[:, 1], non_bg[:, 0]))]

        for pos in non_bg:
            r, c = int(pos[0]), int(pos[1])
            try:
                fd = env.step(GameAction.ACTION6, data={'x': c, 'y': r})
            except Exception:
                continue
            grid = get_frame(fd)
            steps += 1

            if fd.levels_completed > max_levels:
                max_levels = fd.levels_completed

            if fd.state.name in ("WON", "GAME_OVER"):
                break
            if steps >= max_clicks:
                break

        break  # Only one pass

    return max_levels, fd.win_levels, steps, fd.state.name, fd


def click_rarest_color(env, fd, max_clicks=500):
    """Click cells of the rarest non-bg color first."""
    grid = get_frame(fd)
    bg = int(np.bincount(grid.flatten()).argmax())
    colors, counts = np.unique(grid, return_counts=True)
    color_counts = {int(c): int(n) for c, n in zip(colors, counts) if int(c) != bg}

    if not color_counts:
        return fd.levels_completed, fd.win_levels, 0, fd.state.name, fd

    # Sort by rarity (fewest cells first)
    sorted_colors = sorted(color_counts.keys(), key=lambda c: color_counts[c])

    max_levels = fd.levels_completed
    steps = 0

    for color in sorted_colors:
        if fd.state.name in ("WON", "GAME_OVER") or steps >= max_clicks:
            break
        lvl, win, s, state, fd = click_all_of_color(env, fd, color, max_clicks - steps)
        steps += s
        if lvl > max_levels:
            max_levels = lvl

    return max_levels, fd.win_levels, steps, fd.state.name, fd


def try_movement_patterns(env, fd, max_steps=100):
    """Try systematic movement patterns for movement-only games."""
    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (fd.available_actions or [])]
    max_levels = fd.levels_completed
    steps = 0

    # Try: repeated single directions, direction combos, select combos
    patterns = []

    # Pure direction runs
    for d in [1, 2, 3, 4]:
        if d in available:
            patterns.append([d] * 20)

    # Direction + select combos
    if 5 in available:
        for d in [1, 2, 3, 4]:
            if d in available:
                patterns.append([d, d, d, 5] * 5)
                patterns.append([d, 5] * 10)

    # All directions cycling
    dirs = [d for d in [1, 2, 3, 4] if d in available]
    if dirs:
        patterns.append(dirs * 10)

    for pattern in patterns:
        if fd.state.name in ("WON", "GAME_OVER") or steps >= max_steps:
            break
        for action in pattern:
            if action not in available or steps >= max_steps:
                break
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE ARC-AGI-3 Focused Scorer — Sprout")
    print("=" * 60)

    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
    else:
        targets = envs

    print(f"\n{len(targets)} game(s)\n")

    all_results = []
    total_levels = 0
    total_scored = 0
    start = time.time()

    for env_info in targets:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        prefix = game_id.split("-")[0]

        print(f"{'─'*60}")
        print(f"Game: {game_id}")

        # Check available actions
        env = arcade.make(game_id)
        fd = env.reset()
        available = [a.value if hasattr(a, "value") else int(a)
                     for a in (fd.available_actions or [])]
        has_click = 6 in available

        if not has_click:
            # Movement-only game
            best_lvl = 0
            best_result = None
            for run in range(3):
                env2 = arcade.make(game_id)
                fd2 = env2.reset()
                lvl, win, steps, state = try_movement_patterns(env2, fd2)
                star = f" ★ {lvl}" if lvl > 0 else ""
                print(f"  Movement run {run+1}: {steps} steps, {lvl}/{win}{star}")
                if lvl > best_lvl:
                    best_lvl = lvl
                    best_result = {"levels": lvl, "win_levels": win, "steps": steps,
                                   "state": state, "strategy": "movement"}

            if best_lvl > 0:
                total_levels += best_lvl
                total_scored += 1
            all_results.append({"game_id": game_id, **(best_result or {"levels": 0, "win_levels": fd.win_levels})})
            continue

        # Click game — get color info
        grid = get_frame(fd)
        bg = int(np.bincount(grid.flatten()).argmax())
        colors = sorted(set(int(c) for c in np.unique(grid)) - {bg})
        win_levels = fd.win_levels

        print(f"  Colors: {colors}, bg={bg}, win={win_levels}")

        best_lvl = 0
        best_result = None
        best_color = None

        # Strategy 1: Click ALL cells of each individual color
        for color in colors:
            env2 = arcade.make(game_id)
            fd2 = env2.reset()
            lvl, win, steps, state, _ = click_all_of_color(env2, fd2, color)
            star = f" ★ {lvl}" if lvl > 0 else ""
            if args.verbose or lvl > 0:
                print(f"  Color {color:2d}: {steps:3d} steps, {lvl}/{win}{star}")
            if lvl > best_lvl:
                best_lvl = lvl
                best_color = color
                best_result = {"levels": lvl, "win_levels": win, "steps": steps,
                               "state": state, "strategy": f"color_{color}"}

        # Strategy 2: Click all non-bg in order
        env2 = arcade.make(game_id)
        fd2 = env2.reset()
        lvl, win, steps, state, _ = click_all_non_bg_sorted(env2, fd2)
        if args.verbose or lvl > 0:
            print(f"  All-sorted: {steps:3d} steps, {lvl}/{win}")
        if lvl > best_lvl:
            best_lvl = lvl
            best_result = {"levels": lvl, "win_levels": win, "steps": steps,
                           "state": state, "strategy": "all_sorted"}

        # Strategy 3: Rarest color first
        env2 = arcade.make(game_id)
        fd2 = env2.reset()
        lvl, win, steps, state, _ = click_rarest_color(env2, fd2)
        if args.verbose or lvl > 0:
            print(f"  Rarest-first: {steps:3d} steps, {lvl}/{win}")
        if lvl > best_lvl:
            best_lvl = lvl
            best_result = {"levels": lvl, "win_levels": win, "steps": steps,
                           "state": state, "strategy": "rarest_first"}

        # Strategy 4: If clicking single colors failed, try clicking the
        # MOST changed color repeatedly (it might need multiple sweeps)
        if best_lvl == 0:
            # Quick probe to find most-changed color
            env2 = arcade.make(game_id)
            fd2 = env2.reset()
            g = get_frame(fd2)
            changes = {}
            for color in colors[:8]:
                positions = np.argwhere(g == color)
                if len(positions) == 0:
                    continue
                r, c = int(positions[0][0]), int(positions[0][1])
                prev = g.copy()
                try:
                    fd2 = env2.step(GameAction.ACTION6, data={'x': int(c), 'y': int(r)})
                except Exception:
                    continue
                g = get_frame(fd2)
                changes[color] = int(np.sum(prev != g))

            if changes:
                best_probe_color = max(changes, key=changes.get)
                # Dedicate a full run to this color with many sweeps
                env3 = arcade.make(game_id)
                fd3 = env3.reset()
                lvl, win, steps, state, _ = click_all_of_color(
                    env3, fd3, best_probe_color, max_clicks=1000)
                if args.verbose or lvl > 0:
                    print(f"  Max-change ({best_probe_color}): {steps} steps, {lvl}/{win}")
                if lvl > best_lvl:
                    best_lvl = lvl
                    best_result = {"levels": lvl, "win_levels": win, "steps": steps,
                                   "state": state, "strategy": f"max_change_{best_probe_color}"}

        if best_lvl > 0:
            total_levels += best_lvl
            total_scored += 1
            # Save to cartridge
            cart = MembotCartridge(game_id)
            cart_data = cart.read()
            if "color_effectiveness" not in cart_data:
                cart_data["color_effectiveness"] = {}
            strat = best_result.get("strategy", "")
            cart_data["best_strategy"] = strat
            cart_data["best_score"] = {"levels": best_lvl, "steps": best_result["steps"]}
            cart.data = cart_data
            cart.write()

        star = f" ★ {best_lvl} level(s)!" if best_lvl > 0 else ""
        strat = best_result.get("strategy", "") if best_result else ""
        print(f"  BEST: {best_lvl}/{win_levels}{star} [{strat}]")

        all_results.append({"game_id": game_id, **(best_result or {"levels": 0, "win_levels": win_levels})})

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"SUMMARY — Sprout Focused Scorer")
    print(f"{'='*60}")
    for r in sorted(all_results, key=lambda x: x.get("levels", 0), reverse=True):
        lvl = r.get("levels", 0)
        win = r.get("win_levels", 0)
        star = f" ★ {lvl}" if lvl > 0 else ""
        strat = r.get("strategy", "")
        print(f"  {r['game_id'].split('-')[0]:6s}  {lvl}/{win}{star}  [{strat}]")

    print(f"\n  Total: {total_levels} levels across {total_scored} games in {elapsed:.0f}s")

    # Save
    log_path = f"arc-agi-3/experiments/logs/focused_scorer_{int(time.time())}.json"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "machine": "sprout",
            "total_levels": total_levels,
            "games_scored": total_scored,
            "elapsed_s": round(elapsed, 1),
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
