#!/usr/bin/env python3
"""
SAGE Multi-Step Action Planner

Takes spatial state + game observations and produces multi-step action plans.
Uses Gemma 12B for reasoning about spatial relationships and action sequences.

Strategy: observe → describe state compactly → LLM plans 3-5 actions →
execute → observe result → feed back to LLM.

Focus: movement+click games that need spatial reasoning.

Usage:
    cd /Users/dennispalatov/repos/SAGE
    .venv/bin/python3 arc-agi-3/experiments/sage_planner.py --game tr87 --budget 200
"""

import sys
import os
import time
import json
import argparse
import random
import re
import numpy as np
import requests

sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_frame, background_color, color_name, find_color_regions, grid_diff
from arc_spatial import SpatialTracker

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:12b")

ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                5: "SELECT/SUBMIT", 6: "CLICK(x,y)", 7: "UNDO"}


def ask_llm(prompt: str, max_tokens: int = 150) -> str:
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.3, "num_predict": max_tokens},
        }, timeout=60)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception:
        pass
    return ""


def compact_grid_description(grid: np.ndarray, tracker: SpatialTracker) -> str:
    """Produce a compact spatial description for the LLM.

    Focus on: what objects exist, which are interactive, spatial relationships.
    Keep under ~200 tokens.
    """
    bg = background_color(grid)
    parts = [f"Grid 64x64, bg={color_name(bg)}"]

    # Interactive vs static objects
    interactive = tracker.get_interactive_objects()
    static = tracker.get_static_objects()
    untested = tracker.get_untested_objects()

    if interactive:
        items = [f"{color_name(o.color)}({o.w}x{o.h})@({o.cx},{o.cy})" for o in interactive[:5]]
        parts.append(f"Interactive: {', '.join(items)}")

    if untested:
        items = [f"{color_name(o.color)}({o.w}x{o.h})@({o.cx},{o.cy})" for o in
                 sorted(untested, key=lambda o: o.size)[:5]]
        parts.append(f"Untested: {', '.join(items)}")

    # Alignments
    h_groups = tracker.get_aligned_objects("horizontal")
    if h_groups:
        for group in h_groups[:2]:
            names = [color_name(o.color) for o in group]
            parts.append(f"Row y={group[0].cy}: {', '.join(names)}")

    # Movement pattern
    pattern = tracker.get_movement_pattern()
    if pattern:
        parts.append(f"Movement: {pattern}")

    return "\n".join(parts)


def parse_plan(response: str, available: list) -> list:
    """Parse LLM plan response into action list.

    Expects responses like:
    1. UP
    2. RIGHT
    3. CLICK(30, 18)
    4. SELECT

    Returns list of (action_int, data_or_None) tuples.
    """
    actions = []
    for line in response.split("\n"):
        line = line.strip().upper()

        # Match directional
        for name, num in [("UP", 1), ("DOWN", 2), ("LEFT", 3), ("RIGHT", 4)]:
            if name in line and num in available:
                actions.append((num, None))
                break

        # Match SELECT/SUBMIT/ROTATE
        if any(w in line for w in ["SELECT", "SUBMIT", "ROTATE"]) and 5 in available:
            actions.append((5, None))

        # Match UNDO
        if "UNDO" in line and 7 in available:
            actions.append((7, None))

        # Match CLICK(x,y)
        m = re.search(r'CLICK\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', line)
        if m and 6 in available:
            x, y = int(m.group(1)), int(m.group(2))
            actions.append((6, {'x': x, 'y': y}))

    return actions


def build_plan_prompt(grid, available, tracker, action_log, levels, win_levels, step) -> str:
    """Build a planning prompt for the LLM."""
    avail_names = [f"{a}={ACTION_NAMES.get(a, f'A{a}')}" for a in available]
    spatial = compact_grid_description(grid, tracker)

    # Recent action outcomes
    recent = ""
    if action_log:
        recent_entries = action_log[-5:]
        recent = "\nRecent actions:\n" + "\n".join(
            f"  {e['action_name']}: {e['outcome']}" for e in recent_entries)

    return f"""You are playing a puzzle game. Plan your next 3-5 actions.

GAME STATE (step {step}, levels {levels}/{win_levels}):
Actions: {', '.join(avail_names)}

SPATIAL STATE:
{spatial}
{recent}

RULES:
- Each action costs 1 step. Budget is limited.
- Try different actions to discover what they do.
- If movement changes the grid, explore in that direction.
- CLICK needs coordinates: CLICK(x, y).
- SELECT/SUBMIT often confirms or tests your current arrangement.

Plan 3-5 actions. Number each step. Be specific with coordinates.
Example:
1. RIGHT
2. RIGHT
3. CLICK(30, 18)
4. SELECT"""


def play_with_planning(arcade, game_id, budget=200, verbose=False):
    """Play a game using LLM-planned multi-step action sequences."""
    env = arcade.make(game_id)
    fd = env.reset()
    grid = get_frame(fd)
    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (fd.available_actions or [])]
    prefix = game_id.split("-")[0]

    if verbose:
        print(f"\n  Game: {prefix}, Actions: {available}, Levels: 0/{fd.win_levels}")

    tracker = SpatialTracker()
    tracker.update(grid)

    if verbose:
        print(f"  Objects: {len(tracker.objects)}")

    action_log = []
    best_levels = 0
    step = 0
    plan_interval = 6  # Re-plan every N steps

    while step < budget:
        # Plan a sequence
        prompt = build_plan_prompt(grid, available, tracker, action_log,
                                   fd.levels_completed, fd.win_levels, step)
        t0 = time.time()
        response = ask_llm(prompt, max_tokens=200)
        plan_time = time.time() - t0

        plan = parse_plan(response, available)
        if not plan:
            # Fallback: random action
            a = random.choice(available)
            if a == 6:
                targets = tracker.suggest_click_targets(grid, n=1)
                if targets:
                    x, y, _ = targets[0]
                    plan = [(6, {'x': x, 'y': y})]
                else:
                    plan = [(a, None)]
            else:
                plan = [(a, None)]

        if verbose:
            plan_names = []
            for a, d in plan:
                if d:
                    plan_names.append(f"CLICK({d['x']},{d['y']})")
                else:
                    plan_names.append(ACTION_NAMES.get(a, f"A{a}"))
            print(f"  [{step}] Plan ({plan_time:.1f}s): {' → '.join(plan_names)}")

        # Execute plan
        for action_int, data in plan:
            if step >= budget:
                break

            prev_grid = grid.copy()
            prev_levels = fd.levels_completed

            try:
                if data:
                    fd = env.step(INT_TO_GAME_ACTION[action_int], data=data)
                else:
                    fd = env.step(INT_TO_GAME_ACTION[action_int])
            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")
                break

            if fd is None:
                break

            grid = get_frame(fd)
            step += 1
            changed = not np.array_equal(prev_grid, grid)
            n_changed = int(np.sum(prev_grid != grid))

            # Update spatial tracker
            diff = tracker.update(grid)
            if action_int == 6 and data:
                tracker.record_click(data['x'], data['y'], changed, n_changed)

            # Update available actions
            new_avail = [a.value if hasattr(a, "value") else int(a)
                         for a in (fd.available_actions or [])]
            if new_avail:
                available = new_avail

            # Log outcome
            action_name = ACTION_NAMES.get(action_int, f"A{action_int}")
            if data:
                action_name = f"CLICK({data['x']},{data['y']})"
            outcome = f"{n_changed} cells changed" if changed else "no change"
            action_log.append({"action_name": action_name, "outcome": outcome,
                               "changed": changed})

            if verbose and changed:
                spatial_desc = tracker.describe_diff(diff)
                print(f"    {action_name}: {n_changed}px, {spatial_desc}")

            # Level completion
            if fd.levels_completed > prev_levels:
                best_levels = fd.levels_completed
                if verbose:
                    print(f"    ★ LEVEL {best_levels}/{fd.win_levels}!")

            if fd.state.name in ("WON", "LOST", "GAME_OVER"):
                if verbose:
                    print(f"    {fd.state.name} at step {step}")
                break

        if fd is None or fd.state.name in ("WON", "LOST", "GAME_OVER"):
            break

    return {
        "game_id": game_id,
        "levels": best_levels,
        "win_levels": fd.win_levels if fd else 0,
        "steps": step,
        "state": fd.state.name if fd else "UNKNOWN",
    }


def main():
    parser = argparse.ArgumentParser(description="SAGE Multi-Step Planner")
    parser.add_argument("--game", default=None)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--all-combo", action="store_true",
                        help="Run all movement+click games")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE Multi-Step Planner — LLM-Guided Action Sequences")
    print(f"Model: {OLLAMA_MODEL}")
    print("=" * 60)

    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
    elif args.all_combo:
        targets = []
        for e in envs:
            env = arcade.make(e.game_id)
            f = env.reset()
            avail = [a.value if hasattr(a, "value") else int(a) for a in (f.available_actions or [])]
            if any(a in avail for a in [1, 2, 3, 4]) and (6 in avail or 5 in avail):
                targets.append(e)
    else:
        targets = envs[:3]

    print(f"\n{len(targets)} games, {args.runs} runs, budget={args.budget}\n")

    total_levels = 0
    scored = {}

    for env_info in targets:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        prefix = game_id.split("-")[0]
        best = 0
        best_result = None

        for run in range(args.runs):
            try:
                result = play_with_planning(arcade, game_id, budget=args.budget,
                                            verbose=args.verbose)
            except Exception as e:
                if args.verbose:
                    print(f"  Run {run+1}: CRASHED ({e})")
                continue

            levels = result["levels"]
            total_levels += levels
            if levels > best:
                best = levels
                best_result = result

        if best > 0:
            scored[prefix] = best
            print(f"  ★ {prefix:6s}  {best}/{best_result['win_levels']}  ({best_result['steps']} steps)")
        else:
            wl = best_result["win_levels"] if best_result else "?"
            print(f"    {prefix:6s}  0/{wl}")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_levels} levels across {len(scored)} games")
    if scored:
        for g, l in sorted(scored.items(), key=lambda x: -x[1]):
            print(f"  {g}: {l}")


if __name__ == "__main__":
    main()
