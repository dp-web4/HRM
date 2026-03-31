#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Observation Learner — Sprout

No code reading. Pure observation-based learning with membot persistence.

Each game run:
1. RECALL: Query membot for prior knowledge about this game
2. OBSERVE: Take actions, observe grid changes
3. HYPOTHESIZE: What actions cause what effects?
4. TEST: Try hypothesized strategies
5. STORE: Save discoveries to membot for future runs

The learner builds up knowledge over multiple runs:
- Which colors respond to clicks
- Which action sequences cause level-ups
- Grid structure patterns (borders, regions, buttons)
- Per-game effective strategies

This is the competition-viable approach. Game code analysis informs
the CATEGORIES we look for, but the agent learns from observation.
"""
import sys, os, time, json, random
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from arcengine import GameAction
from arc_agi import Arcade

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}
MEMBOT_URL = "http://localhost:8000"


def get_frame(fd):
    grid = np.array(fd.frame)
    return grid[-1] if grid.ndim == 3 else grid


def membot_recall(query, n=3):
    """Retrieve relevant memories from membot."""
    try:
        resp = requests.post(f"{MEMBOT_URL}/api/search",
            json={"query": query, "n": n}, timeout=3)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            return [r["text"] for r in results if r.get("score", 0) > 0.5]
    except Exception:
        pass
    return []


def membot_store(text):
    """Store a new memory in membot."""
    try:
        requests.post(f"{MEMBOT_URL}/api/store",
            json={"text": text}, timeout=3)
    except Exception:
        pass


def analyze_grid(grid):
    """Extract features from grid for pattern recognition."""
    bg = int(np.bincount(grid.flatten()).argmax())
    unique = sorted(set(int(c) for c in np.unique(grid)) - {bg})
    non_bg_count = int(np.sum(grid.flatten() != bg))

    # Color distribution
    color_counts = {}
    for c in unique:
        color_counts[c] = int(np.sum(grid == c))

    # Spatial features: are colors clustered or scattered?
    features = {
        "bg": bg,
        "colors": unique,
        "color_counts": color_counts,
        "non_bg_cells": non_bg_count,
        "grid_shape": grid.shape,
    }
    return features


def probe_actions(env, fd, budget=20):
    """Probe available actions to see what they do.
    Returns dict of action → observed_effect."""
    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (fd.available_actions or [])]
    grid = get_frame(fd)
    results = {}
    steps_used = 0

    for action in sorted(available):
        if steps_used >= budget:
            break

        if action == 6:
            # Click: try a few non-bg cells
            bg = int(np.bincount(grid.flatten()).argmax())
            non_bg = np.argwhere(grid != bg)
            if len(non_bg) > 0:
                # Try 3 different cells
                samples = non_bg[np.random.choice(len(non_bg), min(3, len(non_bg)), replace=False)]
                click_effects = []
                for pos in samples:
                    r, c = int(pos[0]), int(pos[1])
                    color = int(grid[r, c])
                    prev = grid.copy()
                    try:
                        fd = env.step(GameAction.ACTION6, data={'x': c, 'y': r})
                    except Exception:
                        continue
                    grid = get_frame(fd)
                    changed = int(np.sum(prev != grid))
                    click_effects.append({"color": color, "changes": changed,
                                          "level_up": fd.levels_completed > 0})
                    steps_used += 1
                results[6] = click_effects
        elif action == 7:
            prev = grid.copy()
            try:
                fd = env.step(GameAction.ACTION7)
            except Exception:
                continue
            grid = get_frame(fd)
            changed = int(np.sum(prev != grid))
            results[7] = {"changes": changed}
            steps_used += 1
        else:
            # Movement/select actions
            prev = grid.copy()
            try:
                fd = env.step(INT_TO_GAME_ACTION[action])
            except Exception:
                continue
            grid = get_frame(fd)
            changed = int(np.sum(prev != grid))
            results[action] = {"changes": changed}
            steps_used += 1

    return results, fd, grid, steps_used


def exploit_strategy(env, fd, strategy, max_steps=200):
    """Execute a learned strategy."""
    grid = get_frame(fd)
    max_levels = fd.levels_completed
    steps = 0

    if strategy["type"] == "click_color":
        target_color = strategy["color"]
        while steps < max_steps:
            positions = np.argwhere(grid == target_color)
            if len(positions) == 0:
                break
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
                if fd.state.name in ("WON", "GAME_OVER") or steps >= max_steps:
                    break
            if fd.state.name in ("WON", "GAME_OVER"):
                break

    elif strategy["type"] == "click_all_non_bg":
        bg = int(np.bincount(grid.flatten()).argmax())
        while steps < max_steps:
            non_bg = np.argwhere(grid != bg)
            if len(non_bg) == 0:
                break
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
                if fd.state.name in ("WON", "GAME_OVER") or steps >= max_steps:
                    break
            break

    elif strategy["type"] == "movement_pattern":
        pattern = strategy["pattern"]
        for action in pattern * (max_steps // len(pattern)):
            try:
                fd = env.step(INT_TO_GAME_ACTION[action])
            except Exception:
                continue
            grid = get_frame(fd)
            steps += 1
            if fd.levels_completed > max_levels:
                max_levels = fd.levels_completed
            if fd.state.name in ("WON", "GAME_OVER") or steps >= max_steps:
                break

    return max_levels, fd.win_levels, steps, fd.state.name


def learn_game(arcade, game_id, verbose=False):
    """Run one learning episode on a game."""
    prefix = game_id.split("-")[0]

    # Phase 1: RECALL — check membot for prior knowledge
    memories = membot_recall(f"ARC-AGI-3 {prefix} game strategy effective")
    prior_knowledge = "\n".join(memories) if memories else "No prior knowledge"
    if verbose and memories:
        print(f"  Recalled {len(memories)} memories")

    # Phase 2: OBSERVE — probe actions
    env = arcade.make(game_id)
    fd = env.reset()
    grid = get_frame(fd)
    features = analyze_grid(grid)
    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (fd.available_actions or [])]

    if verbose:
        print(f"  Grid: {features['grid_shape']}, Colors: {features['colors']}, "
              f"Actions: {available}")

    probe_results, fd, grid, probe_steps = probe_actions(env, fd, budget=15)

    # Phase 3: HYPOTHESIZE — what works?
    strategies = []

    if 6 in probe_results:
        click_effects = probe_results[6]
        # Find colors that cause most changes
        effective_colors = {}
        for effect in click_effects:
            c = effect["color"]
            if c not in effective_colors:
                effective_colors[c] = {"changes": 0, "level_ups": 0, "tries": 0}
            effective_colors[c]["changes"] += effect["changes"]
            effective_colors[c]["tries"] += 1
            if effect["level_up"]:
                effective_colors[c]["level_ups"] += 1

        # Strategy: click most effective color
        for color, stats in sorted(effective_colors.items(),
                                     key=lambda x: x[1]["changes"], reverse=True):
            if stats["changes"] > 0:
                strategies.append({
                    "type": "click_color",
                    "color": color,
                    "expected_changes": stats["changes"] / max(stats["tries"], 1),
                    "priority": 2 if stats["level_ups"] > 0 else 1,
                })

        # Also try all non-bg
        strategies.append({"type": "click_all_non_bg", "priority": 0})

    # Movement strategies
    movement_actions = [a for a in available if a <= 4]
    if movement_actions:
        for d in movement_actions:
            strategies.append({
                "type": "movement_pattern",
                "pattern": [d] * 10,
                "priority": 0,
            })
        if 5 in available:
            for d in movement_actions:
                strategies.append({
                    "type": "movement_pattern",
                    "pattern": [d, d, d, 5] * 5,
                    "priority": 0,
                })

    # Sort by priority (highest first)
    strategies.sort(key=lambda s: s.get("priority", 0), reverse=True)

    # Phase 4: TEST — try strategies
    # First, try focused single-color clicking for each non-bg color (proven approach)
    best_levels = 0
    best_strategy = None
    best_win = fd.win_levels

    if 6 in available:
        bg = int(np.bincount(grid.flatten()).argmax())
        all_colors = sorted(set(int(c) for c in np.unique(grid)) - {bg})
        for color in all_colors:
            env2 = arcade.make(game_id)
            fd2 = env2.reset()
            lvl, win, steps, state = exploit_strategy(env2, fd2,
                {"type": "click_color", "color": color})
            if lvl > best_levels:
                best_levels = lvl
                best_win = win
                best_strategy = {"type": "click_color", "color": color}
                if verbose:
                    print(f"  ★ click_color({color}): {lvl}/{win}")

    # Then try generated strategies
    for strategy in strategies[:5]:
        env2 = arcade.make(game_id)
        fd2 = env2.reset()
        lvl, win, steps, state = exploit_strategy(env2, fd2, strategy)
        if lvl > best_levels:
            best_levels = lvl
            best_strategy = strategy
            best_win = win
            if verbose:
                print(f"  ★ Strategy {strategy['type']}"
                      f"{'(color='+str(strategy.get('color',''))+')' if 'color' in strategy else ''}"
                      f": {lvl}/{win}")

    # Phase 5: STORE — save discoveries
    if best_levels > 0:
        strat_desc = best_strategy["type"]
        if "color" in best_strategy:
            strat_desc += f" color={best_strategy['color']}"
        membot_store(
            f"ARC-AGI-3 {prefix}: scored {best_levels}/{best_win} levels. "
            f"Strategy: {strat_desc}. "
            f"Colors: {features['colors']}, Actions: {available}."
        )
    else:
        # Store what we learned even if 0 levels
        effective_desc = ""
        if 6 in probe_results:
            for effect in probe_results[6]:
                if effect["changes"] > 0:
                    effective_desc += f"color {effect['color']}: {effect['changes']} changes. "
        if effective_desc:
            membot_store(
                f"ARC-AGI-3 {prefix}: 0 levels but observed: {effective_desc}"
                f"Actions: {available}."
            )

    return {
        "game_id": game_id,
        "levels": best_levels,
        "win_levels": best_win,
        "strategy": best_strategy,
        "features": {
            "colors": features["colors"],
            "actions": available,
            "non_bg_cells": features["non_bg_cells"],
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=None)
    parser.add_argument("--rounds", type=int, default=3, help="Learning rounds")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE ARC-AGI-3 Observation Learner — Sprout")
    print("=" * 60)

    # Ensure membot is mounted
    try:
        requests.post(f"{MEMBOT_URL}/api/mount",
            json={"name": "sage-sprout"}, timeout=3)
    except Exception:
        print("  Warning: membot not available")

    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
    else:
        targets = envs

    print(f"\n{len(targets)} game(s), {args.rounds} round(s)\n")

    cumulative_levels = 0
    cumulative_scored = 0

    for round_num in range(args.rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_num + 1}/{args.rounds}")
        print(f"{'='*60}")

        round_levels = 0
        round_scored = 0

        for env_info in targets:
            game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
            prefix = game_id.split("-")[0]

            result = learn_game(arcade, game_id, verbose=args.verbose)
            lvl = result["levels"]
            win = result["win_levels"]
            star = f" ★ {lvl}" if lvl > 0 else ""
            strat = result["strategy"]["type"] if result["strategy"] else ""
            print(f"  {prefix:6s}  {lvl}/{win}{star}  [{strat}]")

            if lvl > 0:
                round_levels += lvl
                round_scored += 1

        print(f"\n  Round {round_num+1}: {round_levels} levels across {round_scored} games")
        cumulative_levels = max(cumulative_levels, round_levels)

    print(f"\n{'='*60}")
    print(f"BEST: {cumulative_levels} levels")

    # Save membot cartridge
    try:
        requests.post(f"{MEMBOT_URL}/api/save", json={"name": "sage-sprout"}, timeout=5)
    except Exception:
        pass

    # Save log
    log_path = f"arc-agi-3/experiments/logs/learner_{int(time.time())}.json"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({"machine": "sprout", "rounds": args.rounds,
                   "best_levels": cumulative_levels}, f, indent=2)
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
