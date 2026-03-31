#!/usr/bin/env python3
"""
Quick survey of all 25 ARC-AGI-3 games: actions, grid sizes, color counts,
step budgets, and how many steps until GAME_OVER with random play.

Goal: identify which games are tractable for systematic clicking.
"""
import sys, os, time, json, random
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from arcengine import GameAction
from arc_agi import Arcade

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}


def survey_game(arcade, game_id):
    """Survey a single game: actions, grid, step budget."""
    env = arcade.make(game_id)
    fd = env.reset()

    grid = np.array(fd.frame)
    if grid.ndim == 3:
        grid = grid[-1]

    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (fd.available_actions or [])]
    has_click = 6 in available

    # Color analysis
    flat = grid.flatten()
    bg = int(np.bincount(flat).argmax())
    unique_colors = sorted(set(int(c) for c in np.unique(flat)))
    non_bg_colors = [c for c in unique_colors if c != bg]
    non_bg_count = int(np.sum(flat != bg))

    # Find step budget by playing random actions until GAME_OVER
    budget_steps = 0
    levels_hit = 0
    fd2 = env.reset()
    for i in range(10000):
        try:
            if has_click:
                r, c = random.randint(0, grid.shape[0]-1), random.randint(0, grid.shape[1]-1)
                fd2 = env.step(GameAction.ACTION6, data={'x': int(c), 'y': int(r)})
            else:
                a = random.choice(available)
                fd2 = env.step(INT_TO_GAME_ACTION[a])
        except Exception:
            continue

        budget_steps = i + 1
        if fd2.levels_completed > levels_hit:
            levels_hit = fd2.levels_completed

        if fd2.state.name in ("WON", "GAME_OVER"):
            break

    # Try targeted clicking: click each non-bg color once, see what changes
    color_effects = {}
    if has_click:
        env2 = arcade.make(game_id)
        fd3 = env2.reset()
        g = np.array(fd3.frame)
        if g.ndim == 3:
            g = g[-1]

        for color in non_bg_colors[:10]:  # Test up to 10 colors
            positions = np.argwhere(g == color)
            if len(positions) == 0:
                continue
            r, c = int(positions[0][0]), int(positions[0][1])
            prev = g.copy()
            try:
                fd3 = env2.step(GameAction.ACTION6, data={'x': int(c), 'y': int(r)})
            except Exception:
                color_effects[color] = "error"
                continue
            new_g = np.array(fd3.frame)
            if new_g.ndim == 3:
                new_g = new_g[-1]
            changed = int(np.sum(prev != new_g))
            color_effects[color] = changed
            g = new_g

    return {
        "game_id": game_id,
        "grid_shape": list(grid.shape),
        "actions": sorted(available),
        "has_click": has_click,
        "bg_color": bg,
        "unique_colors": unique_colors,
        "non_bg_colors": non_bg_colors,
        "non_bg_cells": non_bg_count,
        "win_levels": fd.win_levels,
        "budget_steps": budget_steps,
        "budget_state": fd2.state.name,
        "random_levels": levels_hit,
        "color_effects": color_effects,
    }


def main():
    print("=" * 60)
    print("ARC-AGI-3 Game Survey — Sprout")
    print("=" * 60)

    arcade = Arcade()
    envs = arcade.get_environments()

    results = []
    for env_info in envs:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        prefix = game_id.split("-")[0]

        t0 = time.time()
        try:
            info = survey_game(arcade, game_id)
        except Exception as e:
            print(f"  {prefix:6s} FAILED: {e}")
            results.append({"game_id": game_id, "error": str(e)})
            continue
        elapsed = time.time() - t0

        # Highlight promising games
        effective = [c for c, v in info["color_effects"].items() if isinstance(v, int) and v > 0]
        star = " ★" if effective else ""
        rlvl = f" (random:{info['random_levels']}lvl)" if info['random_levels'] > 0 else ""

        print(f"  {prefix:6s} {info['grid_shape'][0]}x{info['grid_shape'][1]} "
              f"actions={info['actions']} "
              f"colors={len(info['non_bg_colors'])} "
              f"budget={info['budget_steps']:>5d} "
              f"win={info['win_levels']} "
              f"eff={effective}{star}{rlvl} "
              f"({elapsed:.1f}s)")

        results.append(info)

    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    click_games = [r for r in results if r.get("has_click")]
    move_games = [r for r in results if not r.get("has_click") and not r.get("error")]

    print(f"\nClick games (ACTION6): {len(click_games)}")
    for r in sorted(click_games, key=lambda x: len([c for c,v in x.get('color_effects',{}).items() if isinstance(v,int) and v>0]), reverse=True):
        eff = [c for c,v in r.get("color_effects",{}).items() if isinstance(v,int) and v>0]
        prefix = r["game_id"].split("-")[0]
        print(f"  {prefix}: budget={r['budget_steps']}, effective_colors={eff}, win={r['win_levels']}")

    print(f"\nMovement-only games: {len(move_games)}")
    for r in move_games:
        prefix = r["game_id"].split("-")[0]
        print(f"  {prefix}: budget={r['budget_steps']}, actions={r['actions']}, win={r['win_levels']}")

    # Save
    log_path = f"arc-agi-3/experiments/logs/game_survey_{int(time.time())}.json"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {log_path}")


if __name__ == "__main__":
    main()
