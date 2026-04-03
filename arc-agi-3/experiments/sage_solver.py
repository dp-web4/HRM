#!/usr/bin/env python3
"""
SAGE Puzzle Solver — Strategic LLM reasoning + fast execution.

Architecture: 3-5 LLM calls per attempt, not 100+.

1. PROBE (fast, no LLM): try each action 3x, build action-effect model
2. STRATEGIZE (LLM once): "here's what each action does. What's the puzzle? What's the goal?"
3. PLAN (LLM once): "given the puzzle type and goal, plan the full solution"
4. EXECUTE (fast, no LLM): carry out the plan
5. CHECK: did it work? If level advanced, store solution and continue.
6. REFLECT (LLM once): "the plan failed. Here's what happened. Revise."
7. RE-PLAN (LLM once): new plan based on reflection
8. EXECUTE again

The 12B model reasons about the PUZZLE, not individual actions.

Usage:
    cd /Users/dennispalatov/repos/SAGE
    .venv/bin/python3 arc-agi-3/experiments/sage_solver.py --game lp85
    .venv/bin/python3 arc-agi-3/experiments/sage_solver.py --all --attempts 5
"""

import sys
import os
import time
import json
import random
import re
import numpy as np
import requests

sys.path.insert(0, ".")
sys.path.insert(0, "arc-agi-3/experiments")

from arc_agi import Arcade
from arcengine import GameAction
from arc_perception import get_frame, background_color, color_name, find_color_regions
from arc_spatial import SpatialTracker
from arc_action_model import ActionEffectModel

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:12b")

ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                5: "SELECT", 6: "CLICK(x,y)", 7: "UNDO"}


def ask_llm(prompt: str, max_tokens: int = 500) -> str:
    """Single LLM call — used sparingly for strategic reasoning."""
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.3, "num_predict": max_tokens},
        }, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[error: {e}]"
    return ""


def probe_actions(env, fd, grid, available, budget=15):
    """Fast probing — no LLM. Tries each action 3x, builds action model.

    Returns (action_model, tracker, fd, grid, steps_used, level_ups)
    """
    model = ActionEffectModel()
    tracker = SpatialTracker()
    tracker.update(grid)
    steps = 0
    level_ups = 0
    start_levels = fd.levels_completed

    # Try each non-click action 2x, then probe each color region for clicks
    non_click = [a for a in available if a != 6]
    for action in non_click:
        for _ in range(2):
            if steps >= budget:
                break
            prev_grid = grid.copy()
            try:
                fd = env.step(INT_TO_GAME_ACTION[action])
            except Exception:
                continue
            if fd is None:
                break
            grid = get_frame(fd)
            steps += 1
            diff = tracker.update(grid)
            model.observe(action, prev_grid, grid, diff)
            if fd.levels_completed > start_levels + level_ups:
                level_ups = fd.levels_completed - start_levels
            new_avail = [a.value if hasattr(a, "value") else int(a)
                         for a in (fd.available_actions or [])]
            if new_avail:
                available = new_avail
            if fd.state.name in ("WON", "LOST", "GAME_OVER"):
                return model, tracker, fd, grid, steps, level_ups

    # Probe clicks: try one click per distinct color region
    if 6 in available:
        regions = find_color_regions(grid, min_size=4)
        seen_colors = set()
        for region in regions:
            if steps >= budget:
                break
            color = region["color"]
            if color in seen_colors:
                continue
            seen_colors.add(color)
            prev_grid = grid.copy()
            data = {'x': region["cx"], 'y': region["cy"]}
            try:
                fd = env.step(GameAction.ACTION6, data=data)
            except Exception:
                continue

            if fd is None:
                break
            grid = get_frame(fd)
            steps += 1
            diff = tracker.update(grid)
            model.observe(6, prev_grid, grid, diff)
            changed = not np.array_equal(prev_grid, grid)
            n_px = int(np.sum(prev_grid != grid))
            tracker.record_click(data['x'], data['y'], changed, n_px)
            if changed:
                # Found an interactive region — try 2 more clicks on same color
                for _ in range(2):
                    if steps >= budget:
                        break
                    same_color = [r for r in regions if r["color"] == color and r is not region]
                    if same_color:
                        target = random.choice(same_color)
                        data2 = {'x': target["cx"], 'y': target["cy"]}
                    else:
                        data2 = data  # re-click same spot
                    prev_grid = grid.copy()
                    try:
                        fd = env.step(GameAction.ACTION6, data=data2)
                    except Exception:
                        break
                    if fd is None:
                        break
                    grid = get_frame(fd)
                    steps += 1
                    diff = tracker.update(grid)
                    model.observe(6, prev_grid, grid, diff)
                    ch = not np.array_equal(prev_grid, grid)
                    tracker.record_click(data2['x'], data2['y'], ch,
                                         int(np.sum(prev_grid != grid)))
            if fd.levels_completed > start_levels + level_ups:
                level_ups = fd.levels_completed - start_levels
            new_avail = [a.value if hasattr(a, "value") else int(a)
                         for a in (fd.available_actions or [])]
            if new_avail:
                available = new_avail
            if fd.state.name in ("WON", "LOST", "GAME_OVER"):
                break

    return model, tracker, fd, grid, steps, level_ups


def grid_snapshot(grid, downsample=4):
    """Compact grid description for LLM — 16x16 hex map."""
    ds = grid[::downsample, ::downsample].astype(int)
    lines = []
    for r in range(ds.shape[0]):
        lines.append("".join(format(int(c), "x") for c in ds[r]))
    return "\n".join(lines)


def strategize(action_model, tracker, grid, available, levels, win_levels):
    """LLM call #1: What's the puzzle? What's the goal?"""
    model_desc = action_model.describe()
    game_type = action_model.infer_game_type()

    interactive = tracker.get_interactive_objects()
    interactive_desc = ", ".join(o.describe() for o in interactive[:5]) if interactive else "none found"

    grid_map = grid_snapshot(grid)
    bg = background_color(grid)
    colors = {color_name(int(c)): int(n) for c, n in
              zip(*np.unique(grid.astype(int), return_counts=True)) if c != bg}

    avail = [f"{a}={ACTION_NAMES.get(a, f'A{a}')}" for a in available]

    prompt = f"""You are analyzing a puzzle game to figure out how to solve it.

ACTIONS AVAILABLE: {', '.join(avail)}

WHAT EACH ACTION DOES (learned from testing):
{model_desc}
Game type: {game_type}

INTERACTIVE OBJECTS: {interactive_desc}

GRID (4x downsampled, hex colors):
{grid_map}

NON-BACKGROUND COLORS: {colors}

LEVEL: {levels}/{win_levels}

ANALYZE:
1. What TYPE of puzzle is this? (rotation, maze, matching, sorting, placement, etc.)
2. What is the likely WIN CONDITION? (align colors, reach exit, match pattern, etc.)
3. What spatial structure do you see in the grid?

Reply as JSON:
{{"puzzle_type": "...", "win_condition": "...", "key_observations": "...", "strategy": "..."}}"""

    return ask_llm(prompt, max_tokens=300)


def plan_solution(strategy_response, action_model, grid, available, levels, win_levels, tracker=None):
    """LLM call #2: Plan the step-by-step solution."""
    avail = [f"{a}={ACTION_NAMES.get(a, f'A{a}')}" for a in available]
    grid_map = grid_snapshot(grid, downsample=8)

    # Add interactive object locations for click targeting
    interactive_info = ""
    if tracker:
        interactive = tracker.get_interactive_objects()
        if interactive:
            interactive_info = "\nINTERACTIVE OBJECTS (confirmed clickable, 64x64 coords):\n"
            for o in interactive:
                interactive_info += f"  {color_name(o.color)} at CLICK({o.cx}, {o.cy}) — {o.click_effectiveness:.0%} effective\n"

    prompt = f"""You analyzed this puzzle and determined:
{strategy_response}

AVAILABLE ACTIONS: {', '.join(avail)}
{interactive_info}
CURRENT GRID (8x downsample):
{grid_map}
LEVEL: {levels}/{win_levels}

Now plan the COMPLETE SOLUTION for this level. Write a numbered list of actions.
For CLICK, specify coordinates in the FULL 64x64 grid: CLICK(x, y) where x=column(0-63), y=row(0-63)
For movement: UP, DOWN, LEFT, RIGHT
For confirmation: SELECT

Be SPECIFIC. Plan enough actions to complete the level, not just one step.
Use the interactive objects found during probing — click at THEIR coordinates.
If this is a rotation puzzle, plan the full rotation sequence.
If this is a maze, plan the full path.
If this is pattern matching, plan which items to match and how.

SOLUTION PLAN:"""

    return ask_llm(prompt, max_tokens=500)


def reflect_and_replan(strategy, plan, what_happened, grid, available, levels, win_levels):
    """LLM call #3: The plan failed. What went wrong? New plan."""
    avail = [f"{a}={ACTION_NAMES.get(a, f'A{a}')}" for a in available]
    grid_map = grid_snapshot(grid, downsample=8)

    prompt = f"""Your puzzle strategy was:
{strategy[:200]}

Your plan was:
{plan[:300]}

WHAT HAPPENED:
{what_happened}

CURRENT GRID (8x downsample):
{grid_map}
LEVEL: {levels}/{win_levels}
AVAILABLE ACTIONS: {', '.join(avail)}

The plan didn't complete the level. REFLECT:
1. What went wrong?
2. What did you learn from the result?
3. What should the REVISED plan be?

Reply with a new numbered action list. Be more specific this time.

REVISED PLAN:"""

    return ask_llm(prompt, max_tokens=500)


def parse_plan_to_actions(plan_text, available, grid):
    """Parse LLM plan text into executable action list."""
    actions = []
    for line in plan_text.split("\n"):
        line = line.strip().upper()

        # CLICK(x, y)
        m = re.search(r'CLICK\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', line)
        if m and 6 in available:
            actions.append((6, {'x': int(m.group(1)), 'y': int(m.group(2))}))
            continue

        # Directional
        for name, num in [("UP", 1), ("DOWN", 2), ("LEFT", 3), ("RIGHT", 4)]:
            if name in line and num in available:
                actions.append((num, None))
                break
        else:
            if any(w in line for w in ["SELECT", "SUBMIT", "CONFIRM", "ROTATE"]) and 5 in available:
                actions.append((5, None))
            elif "UNDO" in line and 7 in available:
                actions.append((7, None))

    return actions


def execute_plan(env, fd, grid, plan_actions, max_steps=200):
    """Fast execution — no LLM. Returns (fd, grid, steps, outcomes)."""
    outcomes = []
    steps = 0

    for action_int, data in plan_actions:
        if steps >= max_steps:
            break

        prev_grid = grid.copy()
        prev_levels = fd.levels_completed

        try:
            if data:
                fd = env.step(INT_TO_GAME_ACTION[action_int], data=data)
            else:
                fd = env.step(INT_TO_GAME_ACTION[action_int])
        except Exception as e:
            outcomes.append(f"Action {action_int}: ERROR {e}")
            continue

        if fd is None:
            break

        grid = get_frame(fd)
        steps += 1
        changed = not np.array_equal(prev_grid, grid)
        n_changed = int(np.sum(prev_grid != grid))

        action_name = ACTION_NAMES.get(action_int, f"A{action_int}")
        if data:
            action_name = f"CLICK({data['x']},{data['y']})"

        if fd.levels_completed > prev_levels:
            outcomes.append(f"{action_name}: LEVEL UP! ({fd.levels_completed}/{fd.win_levels})")
        elif changed:
            outcomes.append(f"{action_name}: {n_changed}px changed")
        else:
            outcomes.append(f"{action_name}: no change")

        if fd.state.name in ("WON", "LOST", "GAME_OVER"):
            break

    return fd, grid, steps, outcomes


def solve_game(arcade, game_id, max_attempts=5, budget=500, verbose=False):
    """Solve a game using strategic LLM reasoning + fast execution."""
    prefix = game_id.split("-")[0]
    best_levels = 0
    best_attempt = None

    for attempt in range(max_attempts):
        env = arcade.make(game_id)
        fd = env.reset()
        grid = get_frame(fd)
        available = [a.value if hasattr(a, "value") else int(a)
                     for a in (fd.available_actions or [])]
        total_steps = 0

        if verbose:
            print(f"\n  Attempt {attempt+1}/{max_attempts} | Actions: {available} | Levels: 0/{fd.win_levels}")

        # ═══ PHASE 1: PROBE (fast, no LLM) ═══
        probe_budget = min(budget // 5, 20)
        action_model, tracker, fd, grid, probe_steps, probe_levels = \
            probe_actions(env, fd, grid, available, budget=probe_budget)
        total_steps += probe_steps

        if verbose:
            print(f"  Probe: {probe_steps} steps, {action_model.describe()}")
            if probe_levels > 0:
                print(f"    Probe scored {probe_levels} level(s)!")

        if fd.state.name in ("WON", "LOST", "GAME_OVER"):
            if fd.levels_completed > best_levels:
                best_levels = fd.levels_completed
            continue

        remaining = budget - total_steps
        plan_cycles = 0

        while remaining > 0 and plan_cycles < 3 and fd.state.name not in ("WON", "LOST", "GAME_OVER"):
            plan_cycles += 1

            # ═══ PHASE 2: STRATEGIZE (LLM call #1) ═══
            if plan_cycles == 1:
                t0 = time.time()
                strategy_text = strategize(action_model, tracker, grid, available,
                                           fd.levels_completed, fd.win_levels)
                if verbose:
                    print(f"  Strategy ({time.time()-t0:.0f}s): {strategy_text[:120]}...")

            # ═══ PHASE 3: PLAN (LLM call #2) ═══
            t0 = time.time()
            if plan_cycles == 1:
                plan_text = plan_solution(strategy_text, action_model, grid, available,
                                          fd.levels_completed, fd.win_levels, tracker=tracker)
            else:
                plan_text = reflect_and_replan(strategy_text, plan_text,
                                               "\n".join(outcomes[-10:]),
                                               grid, available,
                                               fd.levels_completed, fd.win_levels)
            plan_actions = parse_plan_to_actions(plan_text, available, grid)
            if verbose:
                print(f"  Plan ({time.time()-t0:.0f}s, {len(plan_actions)} actions): {plan_text[:100]}...")

            if not plan_actions:
                if verbose:
                    print(f"  No parseable actions from plan, falling back to exploration")
                # Fallback: try each effective action
                for a in action_model.get_effective_actions()[:5]:
                    plan_actions.append((a, None) if a != 6 else
                                       (6, {'x': grid.shape[1]//2, 'y': grid.shape[0]//2}))
                if not plan_actions:
                    break

            # ═══ PHASE 4: EXECUTE (fast, no LLM) ═══
            prev_levels = fd.levels_completed
            fd, grid, exec_steps, outcomes = execute_plan(
                env, fd, grid, plan_actions, max_steps=min(remaining, len(plan_actions) + 10))
            total_steps += exec_steps
            remaining = budget - total_steps

            if verbose:
                for o in outcomes[-5:]:
                    print(f"    {o}")

            # ═══ PHASE 5: CHECK ═══
            if fd.levels_completed > prev_levels:
                if verbose:
                    print(f"  ★ Level {fd.levels_completed}/{fd.win_levels}!")
                # Success — continue to next level with fresh strategy
                strategy_text = None  # Force re-strategize for new level
                plan_cycles = 0
                # Update available actions for new level
                new_avail = [a.value if hasattr(a, "value") else int(a)
                             for a in (fd.available_actions or [])]
                if new_avail:
                    available = new_avail
                # Re-probe for new level
                if remaining > 10:
                    action_model, tracker, fd, grid, ps, pl = \
                        probe_actions(env, fd, grid, available, budget=min(10, remaining))
                    total_steps += ps
                    remaining -= ps
                continue

        if fd.levels_completed > best_levels:
            best_levels = fd.levels_completed
            best_attempt = {
                "attempt": attempt + 1,
                "levels": fd.levels_completed,
                "win_levels": fd.win_levels,
                "steps": total_steps,
                "state": fd.state.name,
            }

        if verbose:
            print(f"  Result: {fd.levels_completed}/{fd.win_levels} in {total_steps} steps ({fd.state.name})")

    return {
        "game_id": game_id,
        "best_levels": best_levels,
        "win_levels": best_attempt["win_levels"] if best_attempt else 0,
        "best_attempt": best_attempt,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAGE Puzzle Solver")
    parser.add_argument("--game", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"SAGE Puzzle Solver — Strategic LLM ({MODEL})")
    print("3-5 LLM calls per attempt, not 100+")
    print("=" * 60)

    # Warmup
    print("\nWarming up...", end=" ", flush=True)
    t = ask_llm("ready?", max_tokens=5)
    print(f"OK" if "error" not in t.lower() else f"WARN: {t[:40]}")

    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in e.game_id]
    elif args.all:
        targets = envs
    else:
        targets = envs[:5]

    print(f"\n{len(targets)} games, {args.attempts} attempts, budget={args.budget}\n")

    total_levels = 0
    scored = {}

    for env_info in targets:
        game_id = env_info.game_id
        prefix = game_id.split("-")[0]

        try:
            result = solve_game(arcade, game_id, max_attempts=args.attempts,
                               budget=args.budget, verbose=args.verbose)
        except Exception as e:
            if args.verbose:
                print(f"  {prefix}: CRASHED ({e})")
            continue

        levels = result["best_levels"]
        total_levels += levels
        wl = result["win_levels"]

        if levels > 0:
            scored[prefix] = levels
            print(f"  ★ {prefix:6s}  {levels}/{wl}")
        else:
            print(f"    {prefix:6s}  0/{wl}")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_levels} levels across {len(scored)} games")
    if scored:
        for g, l in sorted(scored.items(), key=lambda x: -x[1]):
            print(f"  {g}: {l}")


if __name__ == "__main__":
    main()
