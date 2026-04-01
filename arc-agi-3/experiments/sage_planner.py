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
from arc_action_model import ActionEffectModel

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

    Includes: downsampled grid map + object summary + spatial relationships.
    """
    bg = background_color(grid)
    parts = [f"Grid 64x64, bg={color_name(bg)}"]

    # Downsampled map (8x8 = 8 rows of 8 hex chars = 80 tokens)
    ds = grid[::8, ::8].astype(int)
    color_legend = {}
    for c in np.unique(ds):
        color_legend[format(int(c), 'x')] = color_name(int(c))
    legend = " ".join(f"{k}={v}" for k, v in sorted(color_legend.items()))
    parts.append(f"Colors: {legend}")
    parts.append("Map (8x downsample, hex):")
    for r in range(ds.shape[0]):
        parts.append("  " + "".join(format(int(c), "x") for c in ds[r]))

    # Interactive vs untested objects
    interactive = tracker.get_interactive_objects()
    untested = tracker.get_untested_objects()

    if interactive:
        items = [f"{color_name(o.color)}@({o.cx},{o.cy})" for o in interactive[:5]]
        parts.append(f"Interactive: {', '.join(items)}")

    if untested and len(untested) <= 10:
        items = [f"{color_name(o.color)}@({o.cx},{o.cy})" for o in
                 sorted(untested, key=lambda o: o.size)[:5]]
        parts.append(f"Notable: {', '.join(items)}")

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


def build_plan_prompt(grid, available, tracker, action_log, levels, win_levels,
                      step, action_model=None) -> str:
    """Build a planning prompt with action model."""
    avail_names = [f"{a}={ACTION_NAMES.get(a, f'A{a}')}" for a in available]
    spatial = compact_grid_description(grid, tracker)

    # Action model (learned from probing)
    model_desc = ""
    if action_model and action_model.total_observations > 0:
        model_desc = f"\n{action_model.describe()}\n"
        game_type = action_model.infer_game_type()
        if game_type == "cursor_cycle":
            model_desc += "\nThis looks like a cursor+cycling puzzle: use movement to position, then cycle actions to change the value at the cursor."
        elif game_type == "navigation":
            model_desc += "\nThis looks like a navigation puzzle: find a path to the goal."

    # Recent action outcomes
    recent = ""
    if action_log:
        recent_entries = action_log[-5:]
        recent = "\nRecent actions:\n" + "\n".join(
            f"  {e['action_name']}: {e['outcome']}" for e in recent_entries)

    return f"""You are playing a puzzle game. Plan your next 3-5 actions.

GAME STATE (step {step}, levels {levels}/{win_levels}):
Actions: {', '.join(avail_names)}
{model_desc}
GRID:
{spatial}
{recent}

STRATEGY:
- Use the ACTION MODEL above to understand what each action does.
- Movement actions position a cursor or player. Cycle/toggle actions change state.
- The goal is to complete levels. Think about what the win condition might be.
- CLICK needs coordinates: CLICK(x, y).

Plan 3-5 actions. Be specific. Number each step.
Example:
1. RIGHT
2. DOWN
3. SELECT"""


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
    action_model = ActionEffectModel()

    if verbose:
        print(f"  Objects: {len(tracker.objects)}")

    action_log = []
    best_levels = 0
    step = 0

    # PROBE PHASE: try each action 2-3 times to build action model
    probe_budget = min(budget // 4, len(available) * 3)
    if verbose:
        print(f"  Probing {len(available)} actions ({probe_budget} steps)...")

    for probe_step in range(probe_budget):
        action_to_probe = available[probe_step % len(available)]
        prev_grid = grid.copy()

        try:
            if action_to_probe == 6:
                # Click with coordinates
                targets = tracker.suggest_click_targets(grid, n=1)
                if targets:
                    x, y, _ = targets[0]
                    fd = env.step(INT_TO_GAME_ACTION[6], data={'x': x, 'y': y})
                else:
                    fd = env.step(INT_TO_GAME_ACTION[action_to_probe])
            else:
                fd = env.step(INT_TO_GAME_ACTION[action_to_probe])
        except Exception:
            continue

        if fd is None:
            break

        grid = get_frame(fd)
        step += 1
        spatial_diff = tracker.update(grid)
        action_model.observe(action_to_probe, prev_grid, grid, spatial_diff)

        if fd.levels_completed > best_levels:
            best_levels = fd.levels_completed
            if verbose:
                print(f"    ★ LEVEL {best_levels} during probe!")

        if fd.state.name in ("WON", "LOST", "GAME_OVER"):
            break

        new_avail = [a.value if hasattr(a, "value") else int(a)
                     for a in (fd.available_actions or [])]
        if new_avail:
            available = new_avail

    if verbose:
        print(f"  {action_model.describe()}")
        print(f"  Game type: {action_model.infer_game_type()}")

    while step < budget:
        # Plan a sequence
        prompt = build_plan_prompt(grid, available, tracker, action_log,
                                   fd.levels_completed, fd.win_levels, step,
                                   action_model=action_model)
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

            # Update spatial tracker + action model
            diff = tracker.update(grid)
            action_model.observe(action_int, prev_grid, grid, diff)
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
