#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Game Runner — Sprout Edition (Qwen 3.5 0.8B)

Optimized for speed on 8GB Jetson Orin Nano:
- Ultra-compact prompts (~6s/action vs 14s+ with verbose)
- Pre-seeded JSON format to minimize output tokens
- Sequence planning (3 actions per LLM call)
- Action-outcome memory for learning within a game
- Reflection every N sequences to revise strategy

Key difference from McNugget's v2: we can't afford 12B reasoning.
0.8B needs tight prompts and <20 output tokens per plan.

Usage:
    cd ~/ai-workspace/SAGE
    python3 arc-agi-3/experiments/sprout_game_runner.py
    python3 arc-agi-3/experiments/sprout_game_runner.py --game tu93 --sequences 20
    python3 arc-agi-3/experiments/sprout_game_runner.py --list  # list available games
"""

import sys
import os
import time
import json
import random
import argparse
import logging

import numpy as np
import requests

# Suppress matplotlib/numpy warnings from arc_agi import
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from arcengine import GameAction

logging.basicConfig(level=logging.WARNING)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = os.environ.get("SAGE_ARC_MODEL", "qwen3.5:0.8b")

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}
ACTION_LABELS = {1: "UP", 2: "DN", 3: "LF", 4: "RT", 5: "SEL", 6: "A6", 7: "A7"}


# ── Grid analysis ──────────────────────────────────────────

def grid_diff(prev: np.ndarray, curr: np.ndarray) -> dict:
    """Fast diff between two grids."""
    if prev is None:
        return {"n": 0, "desc": "first"}
    mask = prev != curr
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "desc": "no change"}
    coords = np.argwhere(mask)
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    old_c = set(int(prev[r, c]) for r, c in coords)
    new_c = set(int(curr[r, c]) for r, c in coords)
    return {
        "n": n,
        "region": f"r{r_min}-{r_max},c{c_min}-{c_max}",
        "colors": f"{sorted(old_c)}->{sorted(new_c)}",
        "desc": f"{n}chg r{r_min}-{r_max} c{c_min}-{c_max} {sorted(old_c)}->{sorted(new_c)}",
    }


def grid_compact(grid: np.ndarray) -> str:
    """Compact grid summary for prompt (minimal tokens)."""
    bg = int(np.bincount(grid.flatten()).argmax())
    non_bg = np.argwhere(grid != bg)
    if len(non_bg) == 0:
        return f"empty(bg={bg})"
    colors, counts = np.unique(grid[grid != bg], return_counts=True)
    color_str = " ".join(f"{int(c)}:{int(n)}" for c, n in zip(colors, counts))
    r_min, c_min = non_bg.min(axis=0)
    r_max, c_max = non_bg.max(axis=0)
    return f"bg={bg} active=r{r_min}-{r_max},c{c_min}-{c_max} colors=[{color_str}]"


# ── Game memory ────────────────────────────────────────────

class Memory:
    """Compact action-outcome memory."""

    def __init__(self):
        self.log = []        # [{actions, changes, levelup}]
        self.hypothesis = ""
        self.total_steps = 0
        self.level_ups = 0
        self.winners = []    # sequences that caused level-ups

    def record(self, actions, changes, diffs, lvl_before, lvl_after):
        names = [ACTION_LABELS.get(a, f"A{a}") for a in actions]
        lvl_up = lvl_after - lvl_before
        self.total_steps += len(actions)
        entry = {"a": names, "chg": changes, "lvl": lvl_up}
        if diffs:
            best = max(diffs, key=lambda d: d["n"])
            if best["n"] > 0:
                entry["d"] = best["desc"][:60]
        self.log.append(entry)
        if len(self.log) > 6:
            self.log = self.log[-6:]
        if lvl_up > 0:
            self.level_ups += lvl_up
            self.winners.append(names)

    def compact(self) -> str:
        """Ultra-compact memory for prompt."""
        lines = []
        for e in self.log[-4:]:
            arrow = "->".join(e["a"])
            lvl = " *LVL*" if e["lvl"] > 0 else ""
            lines.append(f"[{arrow}]={e['chg']}chg{lvl}")
        return "; ".join(lines) if lines else "none"

    def winners_text(self) -> str:
        if not self.winners:
            return ""
        return "Winners: " + ", ".join("->".join(w) for w in self.winners[-3:])


# ── Ollama interface ───────────────────────────────────────

def ask_llm(prompt: str, max_tokens: int = 25, timeout: float = 30.0) -> str:
    """Send prompt to Ollama, return response. Uses /api/chat with think:false."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.3, "num_predict": max_tokens},
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        return ""
    except Exception as e:
        return ""


def parse_sequence(response: str, available: list) -> list:
    """Extract action sequence from LLM response."""
    # Try JSON parse
    try:
        if "{" in response:
            js = response[response.index("{"):response.rindex("}") + 1]
            parsed = json.loads(js)
            seq = parsed.get("s") or parsed.get("sequence", [])
            return [int(a) for a in seq if int(a) in available]
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: extract any numbers that are valid actions
    nums = [int(c) for c in response if c.isdigit() and int(c) in available]
    return nums[:5] if nums else []


# ── Prompts ────────────────────────────────────────────────

def plan_prompt(available: list, lvl_completed: int, win_levels: int,
                memory: Memory, step: int, grid_info: str) -> str:
    """Ultra-compact planning prompt optimized for 0.8B."""
    acts = ",".join(f"{a}={ACTION_LABELS.get(a, f'A{a}')}" for a in available)
    mem = memory.compact()
    hyp = memory.hypothesis[:60] if memory.hypothesis else "unknown"
    win = memory.winners_text()

    return (
        f"Game: {acts}. Lvl {lvl_completed}/{win_levels}. Step {step}. "
        f"Grid: {grid_info[:80]}. "
        f"History: {mem}. "
        f"{win} "
        f"Hyp: {hyp}. "
        f'Plan 3 actions. JSON: {{"s":[nums],"h":"theory"}}'
    )


def reflect_prompt(memory: Memory, lvl_completed: int, win_levels: int) -> str:
    """Compact reflection prompt."""
    mem = memory.compact()
    win = memory.winners_text()
    return (
        f"Game analysis. Lvl {lvl_completed}/{win_levels}. Steps: {memory.total_steps}. "
        f"History: {mem}. {win} "
        f"Which actions cause changes? What pattern? "
        f'JSON: {{"h":"hypothesis","next":"strategy"}}'
    )


# ── Main game loop ─────────────────────────────────────────

def run_game(game_id: str, env, frame_data, args):
    """Play one game."""
    grid = np.array(frame_data.frame)
    if grid.ndim == 3:
        grid = grid[-1]

    available = [a.value if hasattr(a, "value") else int(a)
                 for a in (frame_data.available_actions or [])]

    print(f"  Grid: {grid.shape}, Actions: {available}")
    print(f"  Levels: {frame_data.levels_completed}/{frame_data.win_levels}")
    print()

    memory = Memory()
    prev_grid = grid.copy()
    total_steps = 0
    start_time = time.time()

    for seq_num in range(1, args.sequences + 1):
        levels_before = frame_data.levels_completed

        # Reflection every N sequences
        if seq_num > 1 and (seq_num - 1) % args.reflect_every == 0:
            rprompt = reflect_prompt(memory, frame_data.levels_completed, frame_data.win_levels)
            t0 = time.time()
            rresp = ask_llm(rprompt, max_tokens=40)
            rs = time.time() - t0
            try:
                if "{" in rresp:
                    rj = json.loads(rresp[rresp.index("{"):rresp.rindex("}") + 1])
                    memory.hypothesis = rj.get("h", rj.get("hypothesis", memory.hypothesis))
                    print(f"  >> Reflect ({rs:.1f}s): {memory.hypothesis[:70]}")
            except json.JSONDecodeError:
                print(f"  >> Reflect ({rs:.1f}s): parse failed")
            print()

        # Plan
        grid_info = grid_compact(grid)
        pprompt = plan_prompt(available, frame_data.levels_completed,
                              frame_data.win_levels, memory, total_steps, grid_info)
        t0 = time.time()
        presp = ask_llm(pprompt, max_tokens=25)
        plan_s = time.time() - t0

        sequence = parse_sequence(presp, available)
        if not sequence:
            sequence = [random.choice(available) for _ in range(3)]

        seq_names = [ACTION_LABELS.get(a, f"A{a}") for a in sequence]
        print(f"  Seq {seq_num:2d}: [{' '.join(seq_names)}] ({plan_s:.1f}s)", end="", flush=True)

        # Execute
        seq_changes = 0
        seq_diffs = []
        for action in sequence:
            prev_grid = grid.copy()
            ga = INT_TO_GAME_ACTION[action]
            frame_data = env.step(ga)
            new_grid = np.array(frame_data.frame)
            if new_grid.ndim == 3:
                new_grid = new_grid[-1]
            grid = new_grid
            total_steps += 1

            d = grid_diff(prev_grid, grid)
            seq_diffs.append(d)
            seq_changes += d["n"]

            available = [a.value if hasattr(a, "value") else int(a)
                         for a in (frame_data.available_actions or [])]

            if frame_data.levels_completed > levels_before:
                print(f" *LVL {frame_data.levels_completed}/{frame_data.win_levels}*", end="")

            if frame_data.state.name in ("WON", "GAME_OVER"):
                break

        memory.record(sequence, seq_changes, seq_diffs,
                      levels_before, frame_data.levels_completed)

        # Status
        if seq_changes == 0:
            print(f" -> 0chg")
        else:
            best = max(seq_diffs, key=lambda d: d["n"])
            print(f" -> {seq_changes}chg ({best['desc'][:50]})")

        if frame_data.state.name == "WON":
            print(f"\n  ** WON after {total_steps} steps! **")
            break
        elif frame_data.state.name == "GAME_OVER":
            print(f"\n  ** GAME OVER after {total_steps} steps **")
            break

    # Report
    elapsed = time.time() - start_time
    result = {
        "game_id": game_id,
        "model": MODEL,
        "machine": "sprout",
        "total_steps": total_steps,
        "sequences": seq_num,
        "levels_completed": frame_data.levels_completed,
        "win_levels": frame_data.win_levels,
        "state": frame_data.state.name,
        "elapsed_s": round(elapsed, 1),
        "s_per_step": round(elapsed / max(total_steps, 1), 1),
        "s_per_seq": round(elapsed / max(seq_num, 1), 1),
        "hypothesis": memory.hypothesis,
        "winning_sequences": [{"a": w} for w in memory.winners],
        "sequence_log": memory.log,
    }

    print(f"\n{'='*60}")
    print(f"  Game: {game_id}")
    print(f"  Steps: {total_steps} in {seq_num} sequences")
    print(f"  Levels: {frame_data.levels_completed}/{frame_data.win_levels}")
    print(f"  State: {frame_data.state.name}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/max(total_steps,1):.1f}s/step, {elapsed/max(seq_num,1):.1f}s/seq)")
    if memory.hypothesis:
        print(f"  Hypothesis: {memory.hypothesis}")
    if memory.winners:
        print(f"  Winners: {memory.winners}")

    # Save log
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{game_id}_sprout_{int(time.time())}.json")
    try:
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Log: {log_path}")
    except Exception as e:
        print(f"  (Log save failed: {e})")

    return result


def main():
    parser = argparse.ArgumentParser(description="SAGE ARC-AGI-3 — Sprout (0.8B)")
    parser.add_argument("--game", default=None, help="Game ID prefix (e.g. tu93)")
    parser.add_argument("--sequences", type=int, default=20, help="Max sequences per game")
    parser.add_argument("--reflect-every", type=int, default=5, help="Reflect every N sequences")
    parser.add_argument("--list", action="store_true", help="List available games and exit")
    parser.add_argument("--all", action="store_true", help="Play all available games")
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE ARC-AGI-3 — Sprout (Qwen 3.5 0.8B)")
    print("=" * 60)

    # Warm up Ollama
    print("\nWarming up...", end=" ", flush=True)
    t0 = time.time()
    test = ask_llm("Reply: ok", max_tokens=5)
    warmup = time.time() - t0
    if not test:
        print(f"FAILED ({warmup:.1f}s)")
        print("Is Ollama running with qwen3.5:0.8b?")
        return
    print(f"OK ({warmup:.1f}s)")

    # Load games
    from arc_agi import Arcade
    arcade = Arcade()
    envs = arcade.get_environments()
    print(f"Games available: {len(envs)}")

    if args.list:
        for e in envs:
            gid = e.game_id if hasattr(e, "game_id") else str(e)
            print(f"  {gid}")
        return

    if args.all:
        games_to_play = envs
    elif args.game:
        matches = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
        if not matches:
            print(f"No game matching '{args.game}'")
            return
        games_to_play = matches
    else:
        games_to_play = [random.choice(envs)]

    # Play
    results = []
    for env_info in games_to_play:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        print(f"\n{'='*60}")
        print(f"GAME: {game_id}")
        print(f"{'='*60}")

        try:
            env = arcade.make(game_id)
            frame_data = env.reset()
            result = run_game(game_id, env, frame_data, args)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"FLEET SUMMARY ({len(results)} games)")
        print(f"{'='*60}")
        total_levels = sum(r["levels_completed"] for r in results)
        total_wins = sum(r["win_levels"] for r in results)
        total_time = sum(r["elapsed_s"] for r in results)
        for r in results:
            status = "WON" if r["state"] == "WON" else f"{r['levels_completed']}/{r['win_levels']}"
            print(f"  {r['game_id']}: {status} ({r['total_steps']} steps, {r['elapsed_s']:.0f}s)")
        print(f"\n  Total: {total_levels}/{total_wins} levels in {total_time:.0f}s")


if __name__ == "__main__":
    main()
