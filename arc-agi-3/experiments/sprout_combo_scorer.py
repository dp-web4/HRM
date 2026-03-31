#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Combo Scorer — Sprout

Games that have both movement AND click may need movement to "prime"
the grid before clicking works. McNugget found sc25 needs UP+UP+UP+CLICK.

Strategy:
1. For click-only games: try focused single-color clicking (already works for lp85, vc33)
2. For move+click games: try movement patterns THEN clicking
3. For move-only games: systematic movement patterns

Also tries: ACTION7 (unknown action available in some games), SELECT before click, etc.
"""
import sys
import os
import time
import json
import random

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


def get_non_bg_colors(grid):
    bg = int(np.bincount(grid.flatten()).argmax())
    return sorted(set(int(c) for c in np.unique(grid)) - {bg}), bg


def click_all_of_color(env, fd, color, max_clicks=500):
    """Click every cell of color, return (levels, win, steps, state, fd)."""
    grid = get_frame(fd)
    max_levels = fd.levels_completed
    steps = 0

    while fd.state.name not in ("WON", "GAME_OVER") and steps < max_clicks:
        positions = np.argwhere(grid == color)
        if len(positions) == 0:
            break
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
            if fd.state.name in ("WON", "GAME_OVER") or steps >= max_clicks:
                break
        remaining = np.sum(grid == color)
        if remaining == 0:
            break
    return max_levels, fd.win_levels, steps, fd.state.name, fd


def do_movement(env, fd, moves):
    """Execute a sequence of movement actions. Returns (fd, steps_used)."""
    steps = 0
    for m in moves:
        try:
            fd = env.step(INT_TO_GAME_ACTION[m])
            steps += 1
        except Exception:
            continue
        if fd.state.name in ("WON", "GAME_OVER"):
            break
    return fd, steps


def try_combo_patterns(env, fd, game_id, verbose=False):
    """Try movement→click combo patterns."""
    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (fd.available_actions or [])]
    has_click = 6 in available
    has_select = 5 in available
    has_a7 = 7 in available
    dirs = [d for d in [1, 2, 3, 4] if d in available]

    grid = get_frame(fd)
    colors, bg = get_non_bg_colors(grid)
    max_levels = fd.levels_completed
    total_steps = 0

    if not has_click:
        return max_levels, fd.win_levels, 0, fd.state.name, "none"

    best_levels = max_levels
    best_strategy = "none"

    # Pattern 1: SELECT then click each color
    if has_select:
        for color in colors[:5]:
            env2 = Arcade().make(game_id)
            fd2 = env2.reset()
            # SELECT first
            try:
                fd2 = env2.step(GameAction.ACTION5)
            except Exception:
                pass
            lvl, win, s, state, _ = click_all_of_color(env2, fd2, color, max_clicks=200)
            if lvl > best_levels:
                best_levels = lvl
                best_strategy = f"select+color_{color}"
                if verbose:
                    print(f"    ★ SELECT+color {color}: {lvl}/{win} in {s+1} steps")

    # Pattern 2: Direction(s) then click each color (sc25 pattern)
    for d in dirs:
        for n_moves in [1, 3, 5]:
            for color in colors[:5]:
                env2 = Arcade().make(game_id)
                fd2 = env2.reset()
                fd2, ms = do_movement(env2, fd2, [d] * n_moves)
                if fd2.state.name in ("WON", "GAME_OVER"):
                    continue
                lvl, win, s, state, _ = click_all_of_color(env2, fd2, color, max_clicks=200)
                if lvl > best_levels:
                    best_levels = lvl
                    dir_name = {1:"UP",2:"DN",3:"LF",4:"RT"}.get(d, str(d))
                    best_strategy = f"{dir_name}x{n_moves}+color_{color}"
                    if verbose:
                        print(f"    ★ {dir_name}x{n_moves}+color {color}: {lvl}/{win} in {ms+s} steps")

    # Pattern 3: ACTION7 then click (for games with A7)
    if has_a7:
        for color in colors[:5]:
            env2 = Arcade().make(game_id)
            fd2 = env2.reset()
            try:
                fd2 = env2.step(GameAction.ACTION7)
            except Exception:
                continue
            lvl, win, s, state, _ = click_all_of_color(env2, fd2, color, max_clicks=200)
            if lvl > best_levels:
                best_levels = lvl
                best_strategy = f"a7+color_{color}"
                if verbose:
                    print(f"    ★ A7+color {color}: {lvl}/{win} in {s+1} steps")

    # Pattern 4: Repeated direction-click cycles
    for d in dirs[:2]:
        for color in colors[:3]:
            env2 = Arcade().make(game_id)
            fd2 = env2.reset()
            g = get_frame(fd2)
            lvls = fd2.levels_completed
            steps = 0
            for cycle in range(10):
                # Move
                fd2, ms = do_movement(env2, fd2, [d])
                steps += ms
                if fd2.state.name in ("WON", "GAME_OVER"):
                    break
                # Click one cell of color
                g = get_frame(fd2)
                positions = np.argwhere(g == color)
                if len(positions) == 0:
                    continue
                r, c = int(positions[0][0]), int(positions[0][1])
                try:
                    fd2 = env2.step(GameAction.ACTION6, data={'x': c, 'y': r})
                except Exception:
                    continue
                g = get_frame(fd2)
                steps += 1
                if fd2.levels_completed > lvls:
                    lvls = fd2.levels_completed
                if fd2.state.name in ("WON", "GAME_OVER"):
                    break

            if lvls > best_levels:
                best_levels = lvls
                dir_name = {1:"UP",2:"DN",3:"LF",4:"RT"}.get(d, str(d))
                best_strategy = f"cycle_{dir_name}+color_{color}"
                if verbose:
                    print(f"    ★ Cycle {dir_name}+color {color}: {lvls}/{fd2.win_levels} in {steps} steps")

    return best_levels, fd.win_levels, 0, "TESTED", best_strategy


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE ARC-AGI-3 Combo Scorer — Sprout")
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

        env = arcade.make(game_id)
        fd = env.reset()
        available = [a.value if hasattr(a, "value") else int(a)
                     for a in (fd.available_actions or [])]
        has_click = 6 in available
        grid = get_frame(fd)
        colors, bg = get_non_bg_colors(grid)

        action_str = ",".join(str(a) for a in sorted(available))
        print(f"  Actions=[{action_str}] Colors={colors} bg={bg}")

        best_lvl = 0
        best_strategy = ""
        win_levels = fd.win_levels

        if has_click:
            # Strategy A: Pure single-color clicking
            for color in colors:
                env2 = arcade.make(game_id)
                fd2 = env2.reset()
                lvl, win, steps, state, _ = click_all_of_color(env2, fd2, color)
                if lvl > best_lvl:
                    best_lvl = lvl
                    best_strategy = f"click_color_{color}"
                    win_levels = win
                    if args.verbose:
                        print(f"  ★ color {color}: {lvl}/{win}")

            # Strategy B: Combo patterns (move+click)
            if len(available) > 1:  # Has more than just click
                lvl, win, _, _, strat = try_combo_patterns(
                    env, fd, game_id, verbose=args.verbose)
                if lvl > best_lvl:
                    best_lvl = lvl
                    best_strategy = strat
                    win_levels = win
        else:
            # Movement-only: try patterns
            patterns = []
            dirs = [d for d in [1,2,3,4] if d in available]
            for d in dirs:
                patterns.append(([d]*30, f"pure_{d}"))
            if 5 in available:
                for d in dirs:
                    patterns.append(([d,d,d,5]*8, f"dir{d}_sel"))
                    patterns.append(([d,5]*15, f"alt{d}_sel"))

            for pattern, name in patterns:
                env2 = arcade.make(game_id)
                fd2 = env2.reset()
                fd2, steps = do_movement(env2, fd2, pattern)
                lvl = fd2.levels_completed
                if lvl > best_lvl:
                    best_lvl = lvl
                    best_strategy = name
                    win_levels = fd2.win_levels
                    if args.verbose:
                        print(f"  ★ {name}: {lvl}/{fd2.win_levels}")

        star = f" ★ {best_lvl}" if best_lvl > 0 else ""
        print(f"  BEST: {best_lvl}/{win_levels}{star} [{best_strategy}]")

        if best_lvl > 0:
            total_levels += best_lvl
            total_scored += 1
            # Save to cartridge
            cart = MembotCartridge(game_id)
            cart_data = cart.read()
            cart_data["best_strategy"] = best_strategy
            cart_data["best_score"] = {"levels": best_lvl}
            cart.data = cart_data
            cart.write()

        all_results.append({"game_id": game_id, "levels": best_lvl,
                           "win_levels": win_levels, "strategy": best_strategy})

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"SUMMARY — Sprout Combo Scorer")
    print(f"{'='*60}")
    for r in sorted(all_results, key=lambda x: x.get("levels", 0), reverse=True):
        lvl = r.get("levels", 0)
        win = r.get("win_levels", 0)
        star = f" ★ {lvl}" if lvl > 0 else ""
        print(f"  {r['game_id'].split('-')[0]:6s}  {lvl}/{win}{star}  [{r.get('strategy','')}]")

    print(f"\n  Total: {total_levels} levels across {total_scored} games in {elapsed:.0f}s")

    # Save
    log_path = f"arc-agi-3/experiments/logs/combo_scorer_{int(time.time())}.json"
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
