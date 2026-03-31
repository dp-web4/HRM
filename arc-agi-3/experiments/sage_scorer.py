#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Scorer — Hybrid random/LLM approach to get first score.

Phase 1: Random exploration (fast, 5000 steps/sec) to discover level completion
Phase 2: When a level is completed, record the sequence that caused it
Phase 3: LLM analyzes winning sequences and plans strategy for remaining levels

Key insight: random agents complete levels by accident. We just need to
detect it and learn from it.

Usage:
    cd /Users/dennispalatov/repos/SAGE
    .venv/bin/python3 arc-agi-3/experiments/sage_scorer.py
    .venv/bin/python3 arc-agi-3/experiments/sage_scorer.py --game ls20 --max-steps 5000
"""

import sys
import time
import json
import argparse
import random
import numpy as np

sys.path.insert(0, ".")

from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}


def play_random_phase(env, frame_data, max_steps: int = 3000) -> dict:
    """Phase 1: Random exploration. Fast. Finds level completions by accident."""
    available = [a.value if hasattr(a, "value") else int(a) for a in (frame_data.available_actions or [])]
    if not available:
        return {"steps": 0, "levels": 0, "sequences": []}

    history = []      # all actions taken
    level_events = [] # {step, action_window, levels_before, levels_after}
    grid = np.array(frame_data.frame)
    if grid.ndim == 3:
        grid = grid[-1]

    start_levels = frame_data.levels_completed
    prev_levels = start_levels

    for step in range(max_steps):
        action = random.choice(available)
        history.append(action)

        try:
            frame_data = env.step(INT_TO_GAME_ACTION[action])
        except Exception:
            # Some games crash on certain actions — skip and continue
            continue

        if frame_data is None:
            break

        # Update available actions (may change)
        new_available = [a.value if hasattr(a, "value") else int(a) for a in (frame_data.available_actions or [])]
        if new_available:
            available = new_available
        if not available:
            break

        # Detect level completion
        if frame_data.levels_completed > prev_levels:
            # Record the window of actions that preceded this level-up
            window_size = min(20, len(history))
            window = history[-window_size:]
            level_events.append({
                "step": step,
                "levels_before": prev_levels,
                "levels_after": frame_data.levels_completed,
                "action_window": window,
                "action_names": [
                    {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                     5: "SELECT", 6: "ACTION6", 7: "ACTION7"}.get(a, f"A{a}")
                    for a in window
                ],
            })
            prev_levels = frame_data.levels_completed

        # Check game over
        if frame_data.state.name in ("WON", "LOST"):
            break

    return {
        "steps": step + 1,
        "levels_completed": frame_data.levels_completed,
        "win_levels": frame_data.win_levels,
        "state": frame_data.state.name,
        "level_events": level_events,
        "total_levels_gained": frame_data.levels_completed - start_levels,
    }


def main():
    parser = argparse.ArgumentParser(description="SAGE ARC-AGI-3 Scorer")
    parser.add_argument("--game", default=None, help="Game ID prefix")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max steps per random phase")
    parser.add_argument("--runs", type=int, default=5, help="Number of random runs per game")
    parser.add_argument("--all-games", action="store_true", help="Try all available games")
    args = parser.parse_args()

    print("=" * 70)
    print("SAGE ARC-AGI-3 Scorer — Hybrid Random/LLM")
    print("=" * 70)

    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
    elif args.all_games:
        targets = envs
    else:
        targets = envs[:5]  # first 5 games

    print(f"\nTargeting {len(targets)} game(s), {args.runs} runs each, {args.max_steps} steps/run")
    print(f"Total budget: up to {len(targets) * args.runs * args.max_steps:,} steps\n")

    all_results = []
    total_levels = 0
    total_steps = 0
    games_with_levels = 0
    start_time = time.time()

    for env_info in targets:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        print(f"{'─'*70}")
        print(f"Game: {game_id}")

        best_run = None
        game_total_levels = 0

        for run in range(args.runs):
            try:
                env = arcade.make(game_id)
                frame_data = env.reset()
            except Exception as e:
                print(f"  Run {run+1}: FAILED to init ({e})")
                continue
            available = [a.value if hasattr(a, "value") else int(a) for a in (frame_data.available_actions or [])]

            t0 = time.time()
            try:
                result = play_random_phase(env, frame_data, max_steps=args.max_steps)
            except Exception as e:
                print(f"  Run {run+1}: CRASHED ({e})")
                continue
            elapsed = time.time() - t0

            total_steps += result["steps"]
            levels = result["total_levels_gained"]
            game_total_levels += levels
            rate = result["steps"] / max(elapsed, 0.001)

            status = ""
            if levels > 0:
                status = f" ★ {levels} level(s)!"
                for ev in result["level_events"]:
                    actions = " → ".join(ev["action_names"][-10:])
                    print(f"  LEVEL UP at step {ev['step']}: [{actions}]")
            if result["state"] == "WON":
                status += " 🏆 WON!"

            print(f"  Run {run+1}: {result['steps']} steps, {result['levels_completed']}/{result['win_levels']} levels, {rate:.0f} steps/s{status}")

            if best_run is None or result["levels_completed"] > best_run["levels_completed"]:
                best_run = result

        if game_total_levels > 0:
            games_with_levels += 1
        total_levels += game_total_levels

        all_results.append({
            "game_id": game_id,
            "best_levels": best_run["levels_completed"],
            "win_levels": best_run["win_levels"],
            "best_state": best_run["state"],
            "total_levels_across_runs": game_total_levels,
            "winning_events": best_run["level_events"],
        })

    # ─── Summary ───
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Games tested: {len(targets)}")
    print(f"  Games with level completions: {games_with_levels}")
    print(f"  Total levels completed: {total_levels}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Time: {elapsed:.0f}s ({total_steps/max(elapsed,1):.0f} steps/s)")

    print(f"\n  Per-game results:")
    for r in all_results:
        status = ""
        if r["total_levels_across_runs"] > 0:
            status = f" ★ {r['total_levels_across_runs']} level(s) across runs"
        if r["best_state"] == "WON":
            status += " 🏆"
        print(f"    {r['game_id']:20s}  best: {r['best_levels']}/{r['win_levels']}{status}")

    # Save results
    log_path = f"arc-agi-3/experiments/logs/scorer_{int(time.time())}.json"
    try:
        import os
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({
                "timestamp": time.time(),
                "total_steps": total_steps,
                "total_levels": total_levels,
                "games_with_levels": games_with_levels,
                "elapsed_s": elapsed,
                "results": all_results,
            }, f, indent=2)
        print(f"\n  Results saved: {log_path}")
    except Exception as e:
        print(f"\n  (Could not save: {e})")

    # Print winning sequences for analysis
    if total_levels > 0:
        print(f"\n{'='*70}")
        print(f"WINNING SEQUENCES (for LLM analysis)")
        print(f"{'='*70}")
        for r in all_results:
            for ev in r.get("winning_events", []):
                actions = " → ".join(ev["action_names"])
                print(f"  {r['game_id']}: [{actions}]")


if __name__ == "__main__":
    main()
