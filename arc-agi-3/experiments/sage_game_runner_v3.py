#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Game Runner v3 — GridCartridgeIRP cross-session memory.

Key additions over v2:
- GridCartridgeIRP: persistent cross-session memory per game family
- Cross-level query at game start: "seen this type before?"
- Prior memory injected into plan prompt: LLM reasons over past reflections
- StepRecord written after every reflection + winning sequence
- Cognitive type tagging: reflections surface first in future searches
- Session log flushed to .grid.json cartridge on exit

Phase 1: text search (no embedding required). Embedding slot ready for
Andy's CLIP pipeline (Phase 2 — set andy_url in GridCartridgeIRP config).

Usage:
    cd ~/ai-workspace/SAGE
    .venv-arc/bin/python3 arc-agi-3/experiments/sage_game_runner_v3.py --game lp85
    .venv-arc/bin/python3 arc-agi-3/experiments/sage_game_runner_v3.py --game tu93 --sequences 15
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
from sage.irp.plugins.grid_vision_irp import GridVisionIRP, GridObservation, StepRecord
from sage.irp.plugins.game_action_effector import GameActionEffector, ACTION_NAMES
from sage.irp.plugins.grid_cartridge_irp import GridCartridgeIRP, QUERY_ACTION_OUTCOME, QUERY_CROSS_LEVEL
from sage.interfaces.base_effector import EffectorCommand

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:12b"
INT_TO_GAME_ACTION = {a.value: a for a in GameAction}

ACTION_LABELS = {
    1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
    5: "SELECT", 6: "ACTION6", 7: "ACTION7",
}


# ─────────────────────────────────────────────────────────────
# Grid analysis
# ─────────────────────────────────────────────────────────────

def grid_summary(grid: np.ndarray) -> dict:
    """Analyze grid and return structured summary."""
    bg = int(np.bincount(grid.flatten()).argmax())
    non_bg = np.argwhere(grid != bg)

    if len(non_bg) == 0:
        return {"bg": bg, "empty": True}

    r_min, c_min = non_bg.min(axis=0)
    r_max, c_max = non_bg.max(axis=0)
    colors, counts = np.unique(grid[grid != bg], return_counts=True)

    # Centroid of non-background content
    centroid_r = float(np.mean(non_bg[:, 0]))
    centroid_c = float(np.mean(non_bg[:, 1]))

    return {
        "bg": bg,
        "empty": False,
        "bbox": (int(r_min), int(c_min), int(r_max), int(c_max)),
        "centroid": (round(centroid_r, 1), round(centroid_c, 1)),
        "colors": {int(c): int(n) for c, n in zip(colors, counts)},
        "n_foreground": int(len(non_bg)),
    }


def grid_crop_text(grid: np.ndarray, center_r: int, center_c: int, size: int = 12) -> str:
    """Extract a small crop around a point, rendered as hex text."""
    r1 = max(0, center_r - size // 2)
    r2 = min(grid.shape[0], center_r + size // 2)
    c1 = max(0, center_c - size // 2)
    c2 = min(grid.shape[1], center_c + size // 2)
    crop = grid[r1:r2, c1:c2]
    lines = []
    for r in range(crop.shape[0]):
        lines.append("".join(format(int(c), "x") for c in crop[r]))
    return f"({r2-r1}x{c2-c1} crop at r{r1},c{c1}):\n" + "\n".join(lines)


def diff_summary(prev_grid: np.ndarray, curr_grid: np.ndarray) -> dict:
    """Detailed diff between two grids."""
    if prev_grid is None:
        return {"n_changes": 0, "desc": "First frame."}
    mask = prev_grid != curr_grid
    n = int(mask.sum())
    if n == 0:
        return {"n_changes": 0, "desc": "No changes."}

    coords = np.argwhere(mask)
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    center_r = int(np.mean(coords[:, 0]))
    center_c = int(np.mean(coords[:, 1]))

    old_colors = set(int(prev_grid[r, c]) for r, c in coords)
    new_colors = set(int(curr_grid[r, c]) for r, c in coords)

    # Direction of movement (where did the centroid shift?)
    prev_non_bg = np.argwhere(prev_grid != int(np.bincount(prev_grid.flatten()).argmax()))
    curr_non_bg = np.argwhere(curr_grid != int(np.bincount(curr_grid.flatten()).argmax()))
    shift_r, shift_c = 0, 0
    if len(prev_non_bg) > 0 and len(curr_non_bg) > 0:
        shift_r = float(np.mean(curr_non_bg[:, 0]) - np.mean(prev_non_bg[:, 0]))
        shift_c = float(np.mean(curr_non_bg[:, 1]) - np.mean(prev_non_bg[:, 1]))

    desc = (
        f"{n} cells changed in r{r_min}-{r_max}, c{c_min}-{c_max}. "
        f"Colors {sorted(old_colors)}→{sorted(new_colors)}. "
        f"Content shift: ({shift_r:+.1f} rows, {shift_c:+.1f} cols)."
    )

    return {
        "n_changes": n,
        "region": (int(r_min), int(c_min), int(r_max), int(c_max)),
        "center": (center_r, center_c),
        "shift": (round(shift_r, 1), round(shift_c, 1)),
        "desc": desc,
    }


# ─────────────────────────────────────────────────────────────
# Game memory
# ─────────────────────────────────────────────────────────────

class GameMemory:
    """Structured memory for game learning."""

    def __init__(self):
        self.sequence_log: list = []  # list of {actions, total_changes, level_ups, desc}
        self.hypothesis = ""
        self.strategy = ""
        self.total_steps = 0
        self.total_level_ups = 0
        self.winning_sequences: list = []  # sequences that caused level-ups

    def record_sequence(self, actions: list, total_changes: int, diffs: list,
                        level_before: int, level_after: int):
        seq_names = [ACTION_LABELS.get(a, f"A{a}") for a in actions]
        level_ups = level_after - level_before
        self.total_steps += len(actions)

        diff_descs = [d["desc"] for d in diffs if d["n_changes"] > 0]

        entry = {
            "actions": seq_names,
            "total_changes": total_changes,
            "level_ups": level_ups,
            "diffs": diff_descs[:3],  # keep top 3 most interesting
        }
        self.sequence_log.append(entry)
        if len(self.sequence_log) > 8:
            self.sequence_log = self.sequence_log[-8:]

        if level_ups > 0:
            self.total_level_ups += level_ups
            self.winning_sequences.append({"actions": seq_names, "level_ups": level_ups})

    def to_text(self) -> str:
        if not self.sequence_log:
            return "No sequences tried yet."
        lines = []
        for i, s in enumerate(self.sequence_log):
            arrow = " → ".join(s["actions"])
            lvl = f" ★ LEVEL UP x{s['level_ups']}! ★" if s["level_ups"] > 0 else ""
            lines.append(f"  Seq {i+1}: [{arrow}] → {s['total_changes']} cells changed{lvl}")
            for d in s["diffs"][:2]:
                lines.append(f"         {d}")
        return "\n".join(lines)

    def winning_text(self) -> str:
        if not self.winning_sequences:
            return ""
        lines = ["WINNING SEQUENCES (these caused level-ups — repeat/extend them):"]
        for w in self.winning_sequences:
            lines.append(f"  [{' → '.join(w['actions'])}] → {w['level_ups']} level(s)")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Ollama interface
# ─────────────────────────────────────────────────────────────

def ask_ollama(prompt: str, timeout: float = 90.0, max_tokens: int = 250) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.4, "num_predict": max_tokens},
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return f"[Ollama error: {resp.status_code}]"
    except Exception as e:
        return f"[Ollama error: {e}]"


# ─────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────

def plan_prompt(grid: np.ndarray, summary: dict, available: list,
                levels_completed: int, win_levels: int,
                memory: GameMemory, step_count: int,
                fast: bool = False,
                prior_memory: list = None) -> str:
    """Ask LLM to plan a sequence of 3-5 actions.

    prior_memory: list of SearchResult.to_dict() from GridCartridgeIRP.
    Injected as a CARTRIDGE MEMORY section when present.
    """
    avail = [f"{a}={ACTION_LABELS.get(a, f'A{a}')}" for a in available]

    # Build cartridge memory section (shared by fast and full prompts)
    cart_section = ""
    if prior_memory:
        lines = []
        for r in prior_memory[:3]:
            tag = f"[{r['cognitive_type']}]" if r.get('cognitive_type') else "[step]"
            text = r.get("prior_reasoning", "")[:100]
            score = r.get("score", 0)
            if text:
                lines.append(f"  {tag} ({score:.2f}) {text}")
        if lines:
            cart_section = "CARTRIDGE MEMORY (prior sessions):\n" + "\n".join(lines)

    if fast:
        # Compact prompt for speed (~70 tokens instead of ~300)
        mem_lines = []
        for s in memory.sequence_log[-4:]:
            arrow = "→".join(s["actions"])
            lvl = " ★LEVELUP★" if s["level_ups"] > 0 else ""
            mem_lines.append(f"[{arrow}]={s['total_changes']}chg{lvl}")
        mem_compact = "; ".join(mem_lines) if mem_lines else "none"
        hyp = memory.hypothesis[:80] if memory.hypothesis else "unknown"
        cart_line = f"\n{cart_section}" if cart_section else ""

        return f"""Grid game. Acts: {','.join(avail)}. Lvl {levels_completed}/{win_levels}. Step {step_count}.
History: {mem_compact}
Hypothesis: {hyp}{cart_line}
Plan 3-5 actions. JSON only: {{"sequence":[nums],"hypothesis":"brief","goal":"brief"}}"""

    # Full prompt for quality reasoning
    if not summary["empty"]:
        cr, cc = int(summary["centroid"][0]), int(summary["centroid"][1])
        crop = grid_crop_text(grid, cr, cc, size=16)
        grid_section = f"Colors: {summary['colors']}\nCentroid: r{cr},c{cc}\n{crop}"
    else:
        grid_section = "Grid empty."

    mem_text = memory.to_text()
    winning = memory.winning_text()
    hyp = f"\nHYPOTHESIS: {memory.hypothesis}" if memory.hypothesis else ""
    strat = f"\nSTRATEGY: {memory.strategy}" if memory.strategy else ""
    cart_block = f"\n{cart_section}\n" if cart_section else ""

    return f"""Exploring grid game. No rules given — discover by experimenting.

STATE: step {step_count}, levels {levels_completed}/{win_levels}
Actions: {', '.join(avail)}

GRID:
{grid_section}

MEMORY:
{mem_text}
{winning}{hyp}{strat}{cart_block}
Plan 3-5 actions. Extend patterns that caused changes. Repeat winning sequences.
JSON only: {{"sequence": [nums], "hypothesis": "game theory", "goal": "what to test"}}"""


def reflect_prompt(grid: np.ndarray, summary: dict,
                   levels_completed: int, win_levels: int,
                   memory: GameMemory, step_count: int) -> str:
    """Deep reflection every N sequences — revise strategy."""
    mem_text = memory.to_text()
    winning = memory.winning_text()

    return f"""DEEP REFLECTION — You've played {step_count} actions across {len(memory.sequence_log)} sequences.

RESULTS SO FAR:
- Levels completed: {levels_completed}/{win_levels}
- Total level-ups: {memory.total_level_ups}

SEQUENCE HISTORY:
{mem_text}
{winning}

CURRENT HYPOTHESIS: {memory.hypothesis or '(none yet)'}

REFLECT:
1. What patterns do you see across all your sequences?
2. Which actions consistently cause changes? Which don't?
3. What do the spatial patterns of changes suggest (movement? matching? building?)?
4. What is your REVISED hypothesis about the game rules?
5. What STRATEGY should guide your next sequences?

Respond with ONLY a JSON object:
{{"hypothesis": "<revised theory of game rules>", "strategy": "<what to try next and why>", "key_insight": "<the most important thing you've learned>"}}"""


# ─────────────────────────────────────────────────────────────
# Main game loop
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAGE ARC-AGI-3 Game Runner v2")
    parser.add_argument("--game", default=None, help="Game ID prefix")
    parser.add_argument("--sequences", type=int, default=15, help="Max sequences to play")
    parser.add_argument("--reflect-every", type=int, default=5, help="Reflect every N sequences")
    parser.add_argument("--fast", action="store_true", help="Use compact prompts (~3s/action instead of ~16s)")
    args = parser.parse_args()

    print("=" * 70)
    print("SAGE ARC-AGI-3 Game Runner v2 — Sequence Planning + Reflection")
    print(f"Model: {MODEL} via Ollama")
    print("=" * 70)

    # Warm up
    print("\nWarming up Ollama...", end=" ", flush=True)
    t0 = time.time()
    test = ask_ollama("Reply with: ready", timeout=90)
    print(f"OK ({time.time()-t0:.1f}s)")
    if "error" in test.lower():
        print(f"FAILED: {test}")
        return

    # Game setup
    arcade = Arcade()
    envs = arcade.get_environments()
    if args.game:
        matches = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
        if not matches:
            print(f"No game matching '{args.game}'.")
            return
        env_info = matches[0]
    else:
        import random
        env_info = random.choice(envs)

    game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
    print(f"\nGame: {game_id}")

    env = arcade.make(game_id)
    frame_data = env.reset()
    grid = np.array(frame_data.frame)
    if grid.ndim == 3:
        grid = grid[-1]

    available = [a.value if hasattr(a, "value") else int(a) for a in (frame_data.available_actions or [])]
    print(f"Grid: {grid.shape}, Actions: {available}, Levels: {frame_data.levels_completed}/{frame_data.win_levels}")

    # SAGE components
    gv = GridVisionIRP({"entity_id": "grid_vision_v3", "buffer_size": 50})
    gae = GameActionEffector({"effector_id": "game_action_v3"})
    gc = GridCartridgeIRP({"game_id": game_id, "top_k": 3, "min_score": 0.2})
    gv.push_raw_frame(grid, step_number=0, level_id=game_id)

    # Cross-level memory: does this game family remind us of anything?
    cross_level_context = []
    cl_state = gc.init_state(
        {"query_text": f"game {gc.game_family} level start winning sequence"},
        {"query_mode": QUERY_CROSS_LEVEL, "level_id": game_id},
    )
    cl_state = gc.step(cl_state)
    if cl_state.meta["n_results"] > 0:
        cross_level_context = cl_state.x["results"]
        print(f"  Cartridge: {len(cross_level_context)} cross-level memories loaded")
        for r in cross_level_context[:2]:
            tag = r.get("cognitive_type", "step")
            print(f"    [{tag}] {r.get('prior_reasoning','')[:80]}")
    else:
        print(f"  Cartridge: no prior memory for {gc.game_family} (first session)")

    memory = GameMemory()
    # Seed memory with cross-level insights if available
    for r in cross_level_context:
        if r.get("cognitive_type") == "reflection":
            sr = r.get("step_record", {})
            if sr.get("hypothesis") and not memory.hypothesis:
                memory.hypothesis = sr["hypothesis"]
            if sr.get("strategy") and not memory.strategy:
                memory.strategy = sr["strategy"]
    if memory.hypothesis:
        print(f"  Seeded hypothesis from cartridge: {memory.hypothesis[:80]}")
    prev_grid = grid.copy()
    total_steps = 0
    start_time = time.time()

    print(f"\n{'='*70}")

    for seq_num in range(1, args.sequences + 1):
        levels_before = frame_data.levels_completed
        summary = grid_summary(grid)

        # --- Reflection cycle ---
        if seq_num > 1 and (seq_num - 1) % args.reflect_every == 0:
            print(f"\n  {'─'*60}")
            print(f"  REFLECTING (after {total_steps} steps, {frame_data.levels_completed} levels)...")
            rprompt = reflect_prompt(grid, summary, frame_data.levels_completed,
                                    frame_data.win_levels, memory, total_steps)
            t0 = time.time()
            rresponse = ask_ollama(rprompt, max_tokens=300)
            reflect_s = time.time() - t0
            insight = ""
            try:
                if "{" in rresponse:
                    rj = json.loads(rresponse[rresponse.index("{"):rresponse.rindex("}") + 1])
                    memory.hypothesis = rj.get("hypothesis", memory.hypothesis)
                    memory.strategy = rj.get("strategy", memory.strategy)
                    insight = rj.get("key_insight", "")
                    print(f"  Hypothesis: {memory.hypothesis[:80]}")
                    print(f"  Strategy: {memory.strategy[:80]}")
                    if insight:
                        print(f"  Key insight: {insight[:80]}")
                    print(f"  ({reflect_s:.1f}s)")
            except json.JSONDecodeError:
                print(f"  (reflection parse failed, continuing)")

            # Write reflection to cartridge — highest priority for future recall
            if memory.hypothesis:
                obs_now = gv.get_latest()
                if obs_now is not None:
                    sr = StepRecord(
                        step=total_steps,
                        action_taken=0,
                        action_rationale="deep reflection",
                        salience={"surprise": 0.0, "novelty": 0.8,
                                  "arousal": 0.6, "reward": 0.5, "conflict": 0.3},
                        metabolic_state="REFLECT",
                        atp_spent=0.05,
                        trust_posture={"confidence": 0.7, "label": "reflective"},
                        policy_gate="approved",
                        timestamp=time.time(),
                        reasoning_text=rresponse[:500] if rresponse else None,
                        hypothesis=memory.hypothesis,
                        strategy=memory.strategy,
                        key_insight=insight or None,
                        level_id=game_id,
                    )
                    sr.auto_tag_cognitive_type()
                    gc.write_step_record(obs_now, sr)
                    print(f"  Cartridge: reflection written (cognitive_type={sr.cognitive_type})")

            print(f"  {'─'*60}\n")

        # --- Cartridge query: prior memory for this plan ---
        prior_memory = []
        query_text = " ".join([
            gc.game_family,
            memory.hypothesis[:60] if memory.hypothesis else "",
            " ".join(ACTION_LABELS.get(a, "") for a in (memory.sequence_log[-1]["actions"] if memory.sequence_log else [])),
        ]).strip()
        if query_text:
            cart_state = gc.init_state(
                {"query_text": query_text, "level_id": game_id},
                {"query_mode": QUERY_ACTION_OUTCOME, "level_id": game_id},
            )
            cart_state = gc.step(cart_state)
            prior_memory = cart_state.x.get("results", [])

        # --- Plan sequence ---
        pprompt = plan_prompt(grid, summary, available,
                             frame_data.levels_completed, frame_data.win_levels,
                             memory, total_steps, fast=args.fast,
                             prior_memory=prior_memory)
        t0 = time.time()
        presponse = ask_ollama(pprompt, max_tokens=80 if args.fast else 200)
        plan_s = time.time() - t0

        sequence = []
        goal = ""
        try:
            if "{" in presponse:
                pj = json.loads(presponse[presponse.index("{"):presponse.rindex("}") + 1])
                sequence = [int(a) for a in pj.get("sequence", [])]
                memory.hypothesis = pj.get("hypothesis", memory.hypothesis)
                goal = pj.get("goal", "")[:60]
        except (json.JSONDecodeError, ValueError):
            pass

        # Validate sequence
        sequence = [a for a in sequence if a in available]
        if not sequence:
            # Fallback: random exploration
            import random
            sequence = [random.choice(available) for _ in range(3)]
            goal = "(random exploration)"

        seq_names = [ACTION_LABELS.get(a, f"A{a}") for a in sequence]
        print(f"Seq {seq_num:2d}: [{' → '.join(seq_names)}]  plan:{plan_s:.1f}s  goal: {goal}")

        # --- Execute sequence ---
        seq_changes = 0
        seq_diffs = []
        level_up_during = False

        for i, action in enumerate(sequence):
            prev_grid = grid.copy()

            # Execute action
            ga = INT_TO_GAME_ACTION[action]
            frame_data = env.step(ga)
            new_grid = np.array(frame_data.frame)
            if new_grid.ndim == 3:
                new_grid = new_grid[-1]
            grid = new_grid
            total_steps += 1

            # Track changes
            obs = gv.push_raw_frame(grid, step_number=total_steps, action_taken=action, level_id=game_id)
            d = diff_summary(prev_grid, grid)
            seq_diffs.append(d)
            seq_changes += d["n_changes"]

            # Update available actions (may change after action)
            available = [a.value if hasattr(a, "value") else int(a) for a in (frame_data.available_actions or [])]

            # Check level up mid-sequence
            if frame_data.levels_completed > levels_before:
                level_up_during = True
                print(f"        ★ LEVEL UP at step {i+1}! ({frame_data.levels_completed}/{frame_data.win_levels}) ★")
                # Write winning action to cartridge — high signal for future recall
                sr_win = StepRecord(
                    step=total_steps,
                    action_taken=action,
                    action_rationale=f"LEVEL UP — action {ACTION_LABELS.get(action,'?')} "
                                     f"completed level {frame_data.levels_completed - 1}",
                    salience={"surprise": 0.9, "novelty": 0.7,
                              "arousal": 1.0, "reward": 1.0, "conflict": 0.0},
                    metabolic_state="FOCUS",
                    atp_spent=0.01,
                    trust_posture={"confidence": 0.95, "label": "confident"},
                    policy_gate="approved",
                    timestamp=time.time(),
                    hypothesis=memory.hypothesis,
                    strategy=memory.strategy,
                    key_insight=f"Winning sequence ended with {ACTION_LABELS.get(action,'?')}",
                    cognitive_type="discovery",
                    level_id=game_id,
                )
                gc.write_step_record(obs, sr_win)

            # Check game over
            if frame_data.state.name in ("WON", "LOST"):
                break

        # Record sequence outcome
        memory.record_sequence(
            sequence, seq_changes, seq_diffs,
            levels_before, frame_data.levels_completed,
        )

        # Print outcome
        lvl = f"{frame_data.levels_completed}/{frame_data.win_levels}"
        status = ""
        if level_up_during:
            status = " ★★★"
        elif seq_changes == 0:
            status = " (no effect)"

        # Show the most informative diff
        best_diff = max(seq_diffs, key=lambda d: d["n_changes"]) if seq_diffs else {"desc": "none"}
        print(f"        → {seq_changes} cells changed, levels: {lvl}{status}")
        if seq_changes > 0:
            print(f"        → {best_diff['desc'][:80]}")

        # Check game over
        if frame_data.state.name == "WON":
            print(f"\n  ★ WON after {total_steps} steps! ★")
            break
        elif frame_data.state.name == "LOST":
            print(f"\n  LOST after {total_steps} steps.")
            break

    # ─── Final Report ───
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"FINAL REPORT")
    print(f"{'='*70}")
    print(f"  Game: {game_id}")
    print(f"  Total steps: {total_steps}")
    print(f"  Sequences: {seq_num}")
    print(f"  Levels: {frame_data.levels_completed}/{frame_data.win_levels}")
    print(f"  State: {frame_data.state.name}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/max(total_steps,1):.1f}s/step, {elapsed/max(seq_num,1):.1f}s/seq)")
    print(f"  Efficiency: {gae.efficiency_ratio if gae.total_actions > 0 else 0:.1%}")
    if memory.hypothesis:
        print(f"\n  Final hypothesis: {memory.hypothesis}")
    if memory.strategy:
        print(f"  Final strategy: {memory.strategy}")
    if memory.winning_sequences:
        print(f"\n  Winning sequences:")
        for w in memory.winning_sequences:
            print(f"    [{' → '.join(w['actions'])}] → {w['level_ups']} level(s)")

    # Flush cartridge to disk
    gc.flush()
    print(f"  Cartridge: {gc.stats}")

    # Save session log
    log_path = f"arc-agi-3/experiments/logs/{game_id}_{int(time.time())}.json"
    try:
        import os
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({
                "game_id": game_id,
                "total_steps": total_steps,
                "levels_completed": frame_data.levels_completed,
                "win_levels": frame_data.win_levels,
                "state": frame_data.state.name,
                "elapsed_s": elapsed,
                "hypothesis": memory.hypothesis,
                "strategy": memory.strategy,
                "winning_sequences": memory.winning_sequences,
                "sequence_log": memory.sequence_log,
            }, f, indent=2)
        print(f"\n  Session log: {log_path}")
    except Exception as e:
        print(f"\n  (Could not save log: {e})")


if __name__ == "__main__":
    main()
