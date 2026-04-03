#!/usr/bin/env python3
"""
Broad sweep: 100 iterations × 25 games, budget = human baseline (100% efficiency).

Each game gets sum(baseline_actions) as its action budget per attempt.
KB accumulates across attempts — learning is the objective.
Results logged per-game and summarized at the end.

Usage:
    cd ~/ai-workspace/SAGE  (or /mnt/c/exe/projects/ai-agents/SAGE)
    PYTHONPATH=".:arc-agi-3/experiments" python3 arc-agi-3/experiments/sweep_all_games.py

    # Resume from a specific game (skip ones already done):
    PYTHONPATH=".:arc-agi-3/experiments" python3 arc-agi-3/experiments/sweep_all_games.py --resume-from lp85

    # Run a subset:
    PYTHONPATH=".:arc-agi-3/experiments" python3 arc-agi-3/experiments/sweep_all_games.py --games lp85,ft09,sb26
"""

import sys
import os
import time
import json
import argparse

sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")

import numpy as np
from arc_agi import Arcade
from arcengine import GameAction

from sage.irp.plugins.grid_vision_irp import GridVisionIRP, StepRecord
from sage.irp.plugins.game_action_effector import GameActionEffector
from sage.irp.plugins.grid_cartridge_irp import GridCartridgeIRP
from sage.irp.plugins.game_knowledge_base import GameKnowledgeBase, LevelSolution
from arc_perception import (
    full_perception, color_effectiveness_summary,
    color_name as arc_color_name, find_color_regions,
)

# Import prompts and helpers from v4 runner
import sage_game_runner_v4 as v4

RESULTS_DIR = "arc-agi-3/experiments/sweep_results"
OLLAMA_URL = "http://localhost:11434/api/generate"


def get_model():
    """Detect best available model."""
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        models = []

    # Prefer larger models for better reasoning
    for preferred in ["gemma3:12b", "gemma3:4b", "gemma3:1b", "qwen3.5:0.8b", "tinyllama:latest"]:
        if preferred in models:
            return preferred
    return models[0] if models else "gemma3:4b"


def sweep_one_game(arcade, env_info, model, max_attempts=100):
    """Run up to max_attempts on a single game. Returns summary dict."""
    game_id = env_info.game_id
    game_family = game_id.split("-")[0]
    baseline = env_info.baseline_actions
    budget = sum(baseline)

    print(f"\n{'='*70}")
    print(f"GAME: {game_id}")
    print(f"Levels: {len(baseline)} | Baseline budget: {budget} actions")
    print(f"Per-level baseline: {baseline}")
    print(f"Attempts: {max_attempts}")
    print(f"{'='*70}")

    # Set model on the v4 module
    v4.MODEL = model

    env = arcade.make(game_id)
    frame_data = env.reset()
    grid = v4.get_grid(frame_data)
    available = [a.value if hasattr(a, "value") else int(a) for a in (frame_data.available_actions or [])]

    print(f"Grid: {grid.shape}, Actions: {available}, Win: {frame_data.win_levels}")

    # Load or create knowledge base
    kb = GameKnowledgeBase(game_family)
    prior = kb.load()

    # Load strategic insights from cartridge if available
    cartridge_json = f"arc-agi-3/experiments/cartridges/{game_family}.json"
    if os.path.exists(cartridge_json):
        try:
            with open(cartridge_json) as f:
                cdata = json.load(f)
            kb.strategic_insights = cdata.get("strategic_insights", [])
        except Exception:
            pass

    print(f"KB: {'LOADED' if prior else 'NEW'} | Objects: {len(kb.objects)} | "
          f"Solutions: {list(kb.level_solutions.keys())} | Best: {kb.best_level}")

    # SAGE components
    gv = GridVisionIRP({"entity_id": f"sweep_{game_family}", "buffer_size": 100})
    gc = GridCartridgeIRP({"game_id": game_id, "top_k": 3, "min_score": 0.2})
    gv.push_raw_frame(grid, step_number=0, level_id=game_id)

    game_start = time.time()
    best_level = kb.best_level
    results = []

    for attempt in range(1, max_attempts + 1):
        kb.session_count += 1

        print(f"\n{'─'*50}")
        print(f"  Attempt {attempt}/{max_attempts} | KB: {len(kb.objects)} obj, "
              f"{len(kb.level_solutions)} sol | Best: {best_level}")

        lvl, state, steps, clicks, elapsed = v4.play_one_session(
            env, frame_data, grid, kb,
            argparse.Namespace(budget=budget, think_time=0),
            game_id, gc, gv, attempt,
        )

        if lvl > best_level:
            best_level = lvl

        results.append({
            "attempt": attempt,
            "levels": lvl,
            "win_levels": frame_data.win_levels,
            "steps": steps,
            "state": state,
            "elapsed": round(elapsed, 1),
            "kb_objects": len(kb.objects),
            "kb_solutions": len(kb.level_solutions),
        })

        # Compact progress line
        marker = "★ WON" if state == "WON" else f"L{lvl}/{frame_data.win_levels}"
        print(f"  → {marker} in {steps} steps, {elapsed:.0f}s | "
              f"KB: {len(kb.objects)} obj, {len(kb.level_solutions)} sol")

        if state == "WON":
            print(f"\n  ★★★ GAME COMPLETE on attempt {attempt} ★★★")
            break

        # Reset for next attempt
        frame_data = env.reset()
        grid = v4.get_grid(frame_data)

    game_elapsed = time.time() - game_start

    # Save KB
    kb.save()
    gc.flush()

    # Build summary
    levels_reached = [r["levels"] for r in results]
    summary = {
        "game_id": game_id,
        "game_family": game_family,
        "win_levels": frame_data.win_levels,
        "baseline_budget": budget,
        "baseline_per_level": baseline,
        "model": model,
        "attempts_run": len(results),
        "best_level": best_level,
        "won": any(r["state"] == "WON" for r in results),
        "won_on_attempt": next((r["attempt"] for r in results if r["state"] == "WON"), None),
        "avg_level": round(sum(levels_reached) / len(levels_reached), 2),
        "max_level": max(levels_reached),
        "total_elapsed_s": round(game_elapsed, 1),
        "kb_final": {
            "objects": len(kb.objects),
            "active": sum(1 for o in kb.objects.values() if o.effect_count > 0),
            "solutions": len(kb.level_solutions),
            "mechanics": len(kb.mechanics),
            "failures": len(kb.failed_approaches),
        },
        "attempts": results,
    }

    # Save per-game results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = f"{RESULTS_DIR}/{game_family}_sweep_{int(time.time())}.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved: {result_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Broad Sweep — 100 iterations × 25 games")
    parser.add_argument("--attempts", type=int, default=100, help="Attempts per game")
    parser.add_argument("--resume-from", type=str, default=None, help="Skip games before this prefix")
    parser.add_argument("--games", type=str, default=None, help="Comma-separated game prefixes to run")
    args = parser.parse_args()

    model = get_model()
    print(f"{'='*70}")
    print(f"ARC-AGI-3 BROAD SWEEP")
    print(f"Model: {model}")
    print(f"Attempts per game: {args.attempts}")
    print(f"Budget: human baseline (100% efficiency) per game")
    print(f"Objective: LEARNING across attempts")
    print(f"{'='*70}")

    # Warm up Ollama
    print("\nWarming up Ollama...", end=" ", flush=True)
    t0 = time.time()
    v4.MODEL = model  # ensure v4 uses our model BEFORE warmup
    test = v4.ask_ollama("Reply with: ready", timeout=90)
    print(f"OK ({time.time()-t0:.1f}s) — model: {model}")
    if "error" in test.lower():
        print(f"FAILED: {test}")
        return

    arcade = Arcade()
    envs = arcade.get_environments()

    # Sort by baseline budget (easiest first)
    envs_sorted = sorted(envs, key=lambda e: sum(e.baseline_actions))

    # Filter if --games specified
    if args.games:
        prefixes = [g.strip() for g in args.games.split(",")]
        envs_sorted = [e for e in envs_sorted if any(
            p in (e.game_id if hasattr(e, "game_id") else str(e)) for p in prefixes
        )]
        print(f"Filtered to {len(envs_sorted)} games: {[e.game_id.split('-')[0] for e in envs_sorted]}")

    # Resume support
    skip = args.resume_from is not None
    if skip:
        print(f"Resuming from game matching '{args.resume_from}'")

    sweep_start = time.time()
    all_summaries = []

    for i, env_info in enumerate(envs_sorted):
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        game_family = game_id.split("-")[0]

        if skip:
            if args.resume_from in game_id:
                skip = False
            else:
                print(f"  Skipping {game_family} (before resume point)")
                continue

        print(f"\n\n{'█'*70}")
        print(f"  GAME {i+1}/{len(envs_sorted)}: {game_family}")
        print(f"{'█'*70}")

        try:
            summary = sweep_one_game(arcade, env_info, model, max_attempts=args.attempts)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n  ERROR on {game_family}: {e}")
            import traceback
            traceback.print_exc()
            all_summaries.append({"game_id": game_id, "error": str(e)})
            continue

    # Final summary
    sweep_elapsed = time.time() - sweep_start
    print(f"\n\n{'='*70}")
    print(f"SWEEP COMPLETE — {len(all_summaries)} games in {sweep_elapsed:.0f}s ({sweep_elapsed/3600:.1f}h)")
    print(f"{'='*70}")

    won = [s for s in all_summaries if s.get("won")]
    scored = [s for s in all_summaries if s.get("best_level", 0) > 0 and not s.get("error")]
    failed = [s for s in all_summaries if s.get("error")]

    print(f"\n  Games won:    {len(won)}/{len(all_summaries)}")
    print(f"  Games scored: {len(scored)}/{len(all_summaries)}")
    print(f"  Games failed: {len(failed)}/{len(all_summaries)}")

    print(f"\n  Per-game results:")
    for s in all_summaries:
        if s.get("error"):
            print(f"    {s['game_id'].split('-')[0]:6s}: ERROR — {s['error'][:60]}")
        else:
            won_str = f" ★ WON attempt {s['won_on_attempt']}" if s.get("won") else ""
            print(f"    {s.get('game_family','?'):6s}: best {s['best_level']}/{s['win_levels']} "
                  f"| avg {s['avg_level']:.1f} | {s['attempts_run']} attempts "
                  f"| {s['total_elapsed_s']:.0f}s{won_str}")

    # Save master summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    master_path = f"{RESULTS_DIR}/sweep_master_{int(time.time())}.json"
    with open(master_path, "w") as f:
        json.dump({
            "model": model,
            "attempts_per_game": args.attempts,
            "total_elapsed_s": round(sweep_elapsed, 1),
            "games_won": len(won),
            "games_scored": len(scored),
            "summaries": all_summaries,
        }, f, indent=2)
    print(f"\n  Master results: {master_path}")


if __name__ == "__main__":
    main()
