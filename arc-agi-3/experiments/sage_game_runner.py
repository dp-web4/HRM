#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Game Runner — First playable test.

Minimal perception → reasoning → action loop using:
- GridVisionIRP for perception (frame diff, change detection)
- GameActionEffector for action dispatch (efficiency tracking)
- Ollama (Gemma 3 12B) for reasoning about game state

Not the full consciousness loop — this is the adapter test.

Usage:
    cd /Users/dennispalatov/repos/SAGE
    .venv/bin/python3 arc-agi-3/experiments/sage_game_runner.py
    .venv/bin/python3 arc-agi-3/experiments/sage_game_runner.py --game tu93 --steps 50
"""

import sys
import time
import json
import argparse
import numpy as np
import requests

sys.path.insert(0, ".")

from arc_agi import Arcade
from arcengine import GameAction
from sage.irp.plugins.grid_vision_irp import GridVisionIRP
from sage.irp.plugins.game_action_effector import GameActionEffector, ACTION_NAMES
from sage.interfaces.base_effector import EffectorCommand

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:12b"

# GameAction enum uses compound keys — build int→enum lookup
INT_TO_GAME_ACTION = {a.value: a for a in GameAction}


def grid_to_text(grid: np.ndarray, max_rows: int = 64) -> str:
    """Convert grid to compact summary for LLM.

    Full 64x64 hex dump wastes tokens. Instead, provide:
    - Color distribution
    - Bounding box of non-background content
    - Small region around the most active area
    """
    bg_color = int(np.bincount(grid.flatten()).argmax())  # most common = background
    non_bg = np.argwhere(grid != bg_color)

    if len(non_bg) == 0:
        return f"Grid is uniform color {bg_color}."

    r_min, c_min = non_bg.min(axis=0)
    r_max, c_max = non_bg.max(axis=0)

    # Color counts (excluding background)
    colors, counts = np.unique(grid[grid != bg_color], return_counts=True)
    color_info = ", ".join(f"color {int(c)}:{int(n)} cells" for c, n in zip(colors, counts))

    # Extract the active region (capped at 16x16 for token budget)
    crop_r1 = max(0, r_min - 1)
    crop_r2 = min(grid.shape[0], r_max + 2)
    crop_c1 = max(0, c_min - 1)
    crop_c2 = min(grid.shape[1], c_max + 2)

    # If active region is too large, show center 16x16
    if (crop_r2 - crop_r1) > 20 or (crop_c2 - crop_c1) > 20:
        cr = (r_min + r_max) // 2
        cc = (c_min + c_max) // 2
        crop_r1, crop_r2 = max(0, cr - 8), min(grid.shape[0], cr + 8)
        crop_c1, crop_c2 = max(0, cc - 8), min(grid.shape[1], cc + 8)

    crop = grid[crop_r1:crop_r2, crop_c1:crop_c2]
    lines = []
    for r in range(crop.shape[0]):
        hex_row = "".join(format(int(c), "x") for c in crop[r])
        lines.append(hex_row)
    crop_text = "\n".join(lines)

    return (
        f"Background: color {bg_color}. Active region: rows {r_min}-{r_max}, cols {c_min}-{c_max}.\n"
        f"Colors: {color_info}\n"
        f"Active region ({crop_r2-crop_r1}x{crop_c2-crop_c1} crop at r{crop_r1},c{crop_c1}):\n{crop_text}"
    )


ACTION_LABELS = {
    1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
    5: "SELECT", 6: "ACTION6", 7: "ACTION7",
}


def describe_changes(obs, prev_grid=None, curr_grid=None) -> str:
    """Describe what changed since last frame with spatial detail."""
    if not obs.changes:
        return "No visible changes."

    n = len(obs.changes)
    if n == 0:
        return "No visible changes."

    # Spatial summary: where did changes happen?
    rows = [c["cell"][0] for c in obs.changes]
    cols = [c["cell"][1] for c in obs.changes]
    r_min, r_max = min(rows), max(rows)
    c_min, c_max = min(cols), max(cols)

    # Color flow: what colors appeared/disappeared?
    old_colors = set(c["was"] for c in obs.changes)
    new_colors = set(c["now"] for c in obs.changes)

    return (
        f"{n} cells changed in region r{r_min}-{r_max}, c{c_min}-{c_max}. "
        f"Colors {sorted(old_colors)} → {sorted(new_colors)}."
    )


class ActionMemory:
    """Rolling memory of action-outcome pairs for in-context learning."""

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.entries: list = []
        self.hypothesis: str = ""

    def record(self, action: int, n_changes: int, change_desc: str,
               level_before: int, level_after: int):
        entry = {
            "action": ACTION_LABELS.get(action, f"A{action}"),
            "action_int": action,
            "changes": n_changes,
            "effect": change_desc,
            "leveled_up": level_after > level_before,
        }
        self.entries.append(entry)
        if len(self.entries) > self.capacity:
            self.entries = self.entries[-self.capacity:]

    def to_text(self) -> str:
        if not self.entries:
            return "No actions taken yet."
        lines = []
        for i, e in enumerate(self.entries):
            lvl = " ** LEVEL UP! **" if e["leveled_up"] else ""
            lines.append(
                f"  {i+1}. {e['action']}: {e['changes']} cells changed. {e['effect']}{lvl}"
            )
        return "\n".join(lines)

    def dead_actions(self) -> list:
        """Actions that have been tried 3+ times with 0 changes."""
        from collections import Counter
        zero_counts = Counter()
        total_counts = Counter()
        for e in self.entries:
            total_counts[e["action_int"]] += 1
            if e["changes"] == 0:
                zero_counts[e["action_int"]] += 1
        return [a for a in zero_counts if zero_counts[a] >= 3 and zero_counts[a] == total_counts[a]]

    def effective_actions(self) -> list:
        """Actions that have caused level-ups or large changes."""
        from collections import defaultdict
        stats = defaultdict(lambda: {"total_changes": 0, "count": 0, "level_ups": 0})
        for e in self.entries:
            s = stats[e["action_int"]]
            s["total_changes"] += e["changes"]
            s["count"] += 1
            if e["leveled_up"]:
                s["level_ups"] += 1
        return sorted(stats.keys(), key=lambda a: stats[a]["total_changes"], reverse=True)


def ask_ollama(prompt: str, timeout: float = 60.0) -> str:
    """Send prompt to Ollama, return response text."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 200,
                },
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return f"[Ollama error: {resp.status_code}]"
    except Exception as e:
        return f"[Ollama error: {e}]"


def build_prompt(
    grid: np.ndarray,
    obs,
    available_actions: list,
    step: int,
    levels_completed: int,
    win_levels: int,
    memory: ActionMemory,
) -> str:
    """Build reasoning prompt with action-outcome memory."""
    avail = [f"{a}={ACTION_LABELS.get(a, f'A{a}')}" for a in available_actions]
    grid_text = grid_to_text(grid)
    changes = describe_changes(obs)
    memory_text = memory.to_text()

    hypothesis_section = ""
    if memory.hypothesis:
        hypothesis_section = f"\nCURRENT HYPOTHESIS: {memory.hypothesis}\n"

    dead = memory.dead_actions()
    dead_warning = ""
    if dead:
        dead_names = [f"{a}={ACTION_LABELS.get(a, f'A{a}')}" for a in dead]
        dead_warning = f"\nDEAD ACTIONS (tried 3+ times, never changed anything — DO NOT use): {', '.join(dead_names)}\n"

    return f"""You are exploring an interactive grid game with NO instructions. Discover the rules by experimenting.

GAME STATE (step {step}):
- Levels completed: {levels_completed}/{win_levels}
- Available actions: {', '.join(avail)}
- Last frame changes: {changes}
{dead_warning}
GRID:
{grid_text}

ACTION-OUTCOME MEMORY (what your recent actions did):
{memory_text}
{hypothesis_section}
STRATEGY:
1. Study action-outcome memory. Which actions cause changes? Which don't?
2. Try different ACTION SEQUENCES — maybe UP then DOWN does something different than just UP.
3. Explore systematically: try each direction, observe, then try combinations.
4. If an action changes nothing after 3 tries, STOP using it.
5. The goal is to COMPLETE LEVELS. Look for what triggers level completion.

Respond with ONLY a JSON object:
{{"action": <number>, "hypothesis": "<what you think the game rules are>", "reason": "<why this specific action next>"}}"""


def main():
    parser = argparse.ArgumentParser(description="SAGE ARC-AGI-3 Game Runner")
    parser.add_argument("--game", default=None, help="Game ID prefix (e.g. tu93)")
    parser.add_argument("--steps", type=int, default=30, help="Max steps to play")
    parser.add_argument("--random-fallback", action="store_true",
                        help="Fall back to random actions if LLM fails")
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE ARC-AGI-3 Game Runner")
    print(f"Model: {MODEL} via Ollama")
    print("=" * 60)

    # --- Verify & warm up Ollama ---
    print("\nWarming up Ollama...", end=" ", flush=True)
    t0 = time.time()
    test = ask_ollama("Reply with just: ok", timeout=60)
    warmup_s = time.time() - t0
    if "error" in test.lower():
        print(f"FAILED: {test}")
        return
    print(f"OK ({MODEL}, {warmup_s:.1f}s warmup)")

    # --- Initialize game ---
    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        matches = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
        if not matches:
            print(f"No game matching '{args.game}'. Available: {[e.game_id for e in envs[:10]]}")
            return
        env_info = matches[0]
    else:
        # Pick a movement game (4 directional actions)
        import random
        env_info = random.choice(envs)

    game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
    print(f"\nGame: {game_id}")

    env = arcade.make(game_id)
    frame_data = env.reset()
    grid = np.array(frame_data.frame)

    print(f"Grid: {grid.shape}, Actions: {frame_data.available_actions}")
    print(f"Levels: {frame_data.levels_completed}/{frame_data.win_levels}")

    # --- Initialize SAGE components ---
    gv = GridVisionIRP({"entity_id": "grid_vision_arc3", "buffer_size": 20})

    def action_callback(action_int):
        """Bridge between GameActionEffector and SDK."""
        nonlocal frame_data, grid
        ga = INT_TO_GAME_ACTION[action_int]
        frame_data = env.step(ga)
        new_grid = np.array(frame_data.frame)
        if new_grid.ndim == 3:
            new_grid = new_grid[-1]
        old_grid = grid
        grid = new_grid
        changed = int(np.sum(old_grid != new_grid))
        return {"frame_changed": changed > 0, "cells_changed": changed}

    gae = GameActionEffector({
        "effector_id": "game_action_arc3",
        "action_callback": action_callback,
    })

    # Push initial frame
    obs = gv.push_raw_frame(grid, step_number=0, level_id=game_id)
    # Squeeze grid for display after push (IRP handles internally)
    if grid.ndim == 3:
        grid = grid[-1]

    memory = ActionMemory(capacity=10)
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"{'Step':>4} | {'Action':>8} | {'Chg':>5} | {'Levels':>6} | {'Time':>5} | Reason")
    print(f"{'-'*4:>4}-+-{'-'*8:>8}-+-{'-'*5:>5}-+-{'-'*6:>6}-+-{'-'*5:>5}-+-{'-'*30}")

    for step in range(1, args.steps + 1):
        available = [a.value if hasattr(a, "value") else int(a)
                     for a in (frame_data.available_actions or [])]
        if not available:
            print(f"  No available actions — game may be over.")
            break

        levels_before = frame_data.levels_completed

        # --- Reasoning ---
        prompt = build_prompt(
            grid, obs, available, step,
            frame_data.levels_completed, frame_data.win_levels,
            memory,
        )

        t0 = time.time()
        response = ask_ollama(prompt)
        reasoning_s = time.time() - t0

        # Parse action + hypothesis from response
        chosen_action = None
        reason = ""
        hypothesis = ""
        try:
            json_str = response
            if "{" in response:
                json_str = response[response.index("{"):response.rindex("}") + 1]
            parsed = json.loads(json_str)
            chosen_action = int(parsed.get("action", 0))
            reason = parsed.get("reason", "")[:50]
            hypothesis = parsed.get("hypothesis", "")
        except (json.JSONDecodeError, ValueError):
            reason = f"parse fail: {response[:40]}"

        # Validate action
        if chosen_action not in available:
            if args.random_fallback:
                import random
                chosen_action = random.choice(available)
                reason = f"(random) {reason}"
            else:
                chosen_action = available[0]
                reason = f"(default) {reason}"

        # Update hypothesis if LLM provided one
        if hypothesis:
            memory.hypothesis = hypothesis

        # --- Act ---
        cmd = EffectorCommand(
            effector_id="game_action_arc3",
            effector_type="game",
            action="game_action",
            parameters={"action": chosen_action, "rationale": reason},
        )
        result = gae.execute(cmd)

        # --- Observe new state ---
        new_grid = np.array(frame_data.frame)
        obs = gv.push_raw_frame(
            new_grid,
            step_number=step,
            action_taken=chosen_action,
            level_id=game_id,
        )
        if new_grid.ndim == 3:
            grid = new_grid[-1]
        else:
            grid = new_grid

        n_changes = len(obs.changes)
        change_desc = describe_changes(obs)

        # Record in memory
        memory.record(
            action=chosen_action,
            n_changes=n_changes,
            change_desc=change_desc,
            level_before=levels_before,
            level_after=frame_data.levels_completed,
        )

        action_name = ACTION_LABELS.get(chosen_action, f"A{chosen_action}")
        lvl = f"{frame_data.levels_completed}/{frame_data.win_levels}"

        print(f"{step:4d} | {action_name:>8} | {n_changes:5d} | {lvl:>6} | {reasoning_s:4.1f}s | {reason}")

        # Print hypothesis updates
        if hypothesis and step <= 3 or (step % 5 == 0):
            print(f"     | {'':>8} | {'':>5} | {'':>6} | {'':>5} | Hypothesis: {hypothesis[:60]}")

        # Check win/loss
        if frame_data.state.name == "WON":
            print(f"\n  WON after {step} steps!")
            break
        elif frame_data.state.name == "LOST":
            print(f"\n  LOST after {step} steps.")
            break

    # --- Final report ---
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"FINAL REPORT")
    print(f"{'='*60}")
    print(f"  Game: {game_id}")
    print(f"  Steps: {step}")
    print(f"  Levels: {frame_data.levels_completed}/{frame_data.win_levels}")
    print(f"  State: {frame_data.state.name}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/step:.1f}s/step)")
    print(f"  Effector efficiency: {gae.efficiency_ratio:.1%}")
    print(f"  RHAE estimate: {gae.rhae_estimate:.3f}")
    print(f"  Vision frames: {gv.stats['frames_received']}")
    if memory.hypothesis:
        print(f"  Final hypothesis: {memory.hypothesis}")
    print(f"\n  Action-outcome memory:")
    print(memory.to_text())


if __name__ == "__main__":
    main()
