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
sys.path.insert(0, "arc-agi-3/experiments")

from arc_agi import Arcade
from arcengine import GameAction
from sage.irp.plugins.grid_vision_irp import GridVisionIRP, GridObservation, StepRecord
from sage.irp.plugins.game_action_effector import GameActionEffector, ACTION_NAMES
from sage.irp.plugins.grid_cartridge_irp import GridCartridgeIRP, QUERY_ACTION_OUTCOME, QUERY_CROSS_LEVEL
from sage.interfaces.base_effector import EffectorCommand
from arc_perception import (
    full_perception, color_effectiveness_summary, grid_diff as arc_grid_diff,
    color_name as arc_color_name, find_color_regions,
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"
INT_TO_GAME_ACTION = {a.value: a for a in GameAction}

ACTION_LABELS = {
    # Note: ACTION6 is a ComplexAction requiring x,y coordinates.
    # env.step(GameAction.ACTION6, data={'x': col, 'y': row})
    # Without coordinates it does nothing — confirmed discovery 2026-03-29
    # See: arc-agi-3/ENVIRONMENT.md line 83, sage_clicker.py docstring
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

def interactive_targets_text(grid: np.ndarray, click_history: list = None,
                            max_targets: int = 18) -> str:
    """Extract small (likely interactive) regions as click targets for LLM.

    Marks positions that have been clicked many times as [tried N×].
    Highlights untried positions so LLM varies exploration.
    """
    regions = find_color_regions(grid, min_size=4)  # include small regions

    # Build click count map from history: (r,c) → count
    tried_counts: dict = {}
    if click_history:
        for h in click_history:
            pos = (h["r"], h["c"])
            tried_counts[pos] = tried_counts.get(pos, 0) + 1

    # Separate small interactive targets from large structural regions
    buttons = []
    for r in regions:
        # Button-sized: small square-ish objects (likely UI elements / rotation buttons)
        if 4 <= r["size"] <= 64 and r["w"] <= 12 and r["h"] <= 12:
            buttons.append(r)

    if not buttons:
        buttons = sorted(regions, key=lambda r: r["size"])[:max_targets]

    # Sort: untried first, then least tried
    def sort_key(r):
        n = tried_counts.get((r["cy"], r["cx"]), 0)
        return n

    buttons = sorted(buttons, key=sort_key)

    lines = []
    for r in buttons[:max_targets]:
        n = tried_counts.get((r["cy"], r["cx"]), 0)
        tried_marker = f" [tried {n}×]" if n > 0 else " ← NEW"
        lines.append(
            f"  {r['color_name']}({r['w']}x{r['h']}) @({r['cx']},{r['cy']}){tried_marker}"
        )
    return "\n".join(lines) if lines else "  (no small regions detected)"


def find_click_target(grid: np.ndarray, color_stats: dict = None,
                      clicked: set = None) -> tuple:
    """Pick the best (row, col) coordinate for ACTION6.

    Priority:
    1. Unclicked cells of highest-effectiveness color (from color_stats)
    2. Any unclicked non-background cell
    3. Random non-background cell (fallback)

    color_stats: {color_int: {"tries": int, "changes": int}} — learned effectiveness
    clicked: set of (r, c) already tried this level
    Returns: (row, col, color)
    """
    if clicked is None:
        clicked = set()

    bg = int(np.bincount(grid.flatten()).argmax())
    non_bg = np.argwhere(grid != bg)

    if len(non_bg) == 0:
        h, w = grid.shape
        return (h // 2, w // 2, int(grid[h // 2, w // 2]))

    # Score each non-background position
    scored = []
    for r, c in non_bg:
        color = int(grid[r, c])
        if (r, c) in clicked:
            priority = 0.1  # already tried, low priority
        elif color_stats and color in color_stats:
            tries = color_stats[color]["tries"]
            changes = color_stats[color]["changes"]
            priority = (changes / tries) if tries > 0 else 0.5
        else:
            priority = 0.5
        scored.append((priority, int(r), int(c), color))

    scored.sort(reverse=True)
    _, r, c, color = scored[0]
    return (r, c, color)


class GameMemory:
    """Structured memory for game learning."""

    def __init__(self):
        self.sequence_log: list = []  # list of {actions, total_changes, level_ups, desc}
        self.hypothesis = ""
        self.strategy = ""
        self.total_steps = 0
        self.total_level_ups = 0
        self.winning_sequences: list = []  # sequences that caused level-ups
        self.color_stats: dict = {}        # {color: {"tries": int, "changes": int}}
        self.clicked_this_level: set = set()
        self.click_history: list = []     # recent click causal feedback for LLM

    def record_click(self, r: int, c: int, color: int, changed: bool):
        """Track coordinate click effectiveness by color."""
        self.clicked_this_level.add((r, c))
        if color not in self.color_stats:
            self.color_stats[color] = {"tries": 0, "changes": 0}
        self.color_stats[color]["tries"] += 1
        if changed:
            self.color_stats[color]["changes"] += 1

    def record_click_detail(self, step: int, r: int, c: int, color: int,
                            n_changed: int, region_desc: str = ""):
        """Track detailed click causal feedback for LLM reasoning."""
        self.click_history.append({
            "step": step, "r": r, "c": c,
            "color_name": arc_color_name(color), "color": color,
            "changes": n_changed, "region": region_desc,
        })
        if len(self.click_history) > 10:
            self.click_history = self.click_history[-10:]

    def click_history_text(self, n: int = 6) -> str:
        """Format recent clicks as causal feedback for LLM."""
        if not self.click_history:
            return "No clicks yet."
        lines = []
        for h in self.click_history[-n:]:
            effect = f"→ {h['changes']} cells changed" if h['changes'] > 0 else "→ NO CHANGE"
            loc = h['region'] if h['region'] else f"r={h['r']},c={h['c']}"
            lines.append(f"  Step {h['step']}: clicked {h['color_name']} @({loc}) {effect}")
        return "\n".join(lines)

    def color_tries_changes(self) -> tuple:
        """Return (color_tries, color_changes) dicts for color_effectiveness_summary."""
        tries = {c: s["tries"] for c, s in self.color_stats.items()}
        changes = {c: s["changes"] for c, s in self.color_stats.items()}
        return tries, changes

    def reset_level_clicks(self):
        self.clicked_this_level = set()
        self.click_history = []  # fresh slate for new level — avoid anchoring on old positions

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
                prior_memory: list = None,
                strategic_insights: list = None) -> str:
    """Ask LLM to plan a sequence of 3-5 actions.

    prior_memory: list of SearchResult.to_dict() from GridCartridgeIRP.
    strategic_insights: list of insight strings from cartridge (e.g. rotation puzzle facts).
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

    # Known game knowledge from strategic_insights
    knowledge_section = ""
    if strategic_insights:
        lines = [f"  - {s}" for s in strategic_insights[:5]]
        knowledge_section = "KNOWN GAME KNOWLEDGE:\n" + "\n".join(lines)

    if fast:
        # Compact prompt for speed
        mem_lines = []
        for s in memory.sequence_log[-4:]:
            arrow = "→".join(s["actions"])
            lvl = " ★LEVELUP★" if s["level_ups"] > 0 else ""
            mem_lines.append(f"[{arrow}]={s['total_changes']}chg{lvl}")
        mem_compact = "; ".join(mem_lines) if mem_lines else "none"
        hyp = memory.hypothesis[:80] if memory.hypothesis else "unknown"
        knowledge_line = f"\n{knowledge_section}" if knowledge_section else ""
        cart_line = f"\n{cart_section}" if cart_section else ""

        return f"""Grid game. Acts: {','.join(avail)}. Lvl {levels_completed}/{win_levels}. Step {step_count}.
History: {mem_compact}
Hypothesis: {hyp}{knowledge_line}{cart_line}
Plan 3-5 actions. JSON only: {{"sequence":[nums],"hypothesis":"brief","goal":"brief"}}"""

    # Full prompt for quality reasoning — use full_perception for grid description
    perception = full_perception(grid)
    targets = interactive_targets_text(grid, click_history=memory.click_history)
    color_tries, color_changes = memory.color_tries_changes()
    effectiveness = color_effectiveness_summary(color_tries, color_changes)
    click_hist = memory.click_history_text()

    mem_text = memory.to_text()
    winning = memory.winning_text()
    hyp = f"\nHYPOTHESIS: {memory.hypothesis}" if memory.hypothesis else ""
    strat = f"\nSTRATEGY: {memory.strategy}" if memory.strategy else ""
    knowledge_block = f"\n{knowledge_section}\n" if knowledge_section else ""
    cart_block = f"\n{cart_section}\n" if cart_section else ""

    return f"""Exploring grid game. No rules given — learn from observation.

STATE: step {step_count}, levels {levels_completed}/{win_levels}
Available actions: {', '.join(avail)}
{knowledge_block}
GRID LAYOUT (color regions as name(WxH)@(x,y) where x=col, y=row):
{perception}

INTERACTIVE TARGETS (small objects — likely buttons/pieces to click):
{targets}

CLICK HISTORY (causal feedback):
{click_hist}

COLOR EFFECTIVENESS (what causes changes):
{effectiveness}

SEQUENCE HISTORY:
{mem_text}
{winning}{hyp}{strat}{cart_block}
Plan 3-5 actions. IMPORTANT: Prioritize targets marked ← NEW from INTERACTIVE TARGETS — avoid repeating [tried N×] positions that haven't caused level-ups.
Coordinates: @(x,y) means x=col, y=row. Use exactly those values.
JSON only: {{"sequence": [nums], "clicks": [{{"x": col, "y": row}}, ...], "hypothesis": "game theory", "goal": "what to test next"}}
(one {{x,y}} per ACTION6 — try DIFFERENT targets each sequence)"""


def reflect_prompt(grid: np.ndarray, summary: dict,
                   levels_completed: int, win_levels: int,
                   memory: GameMemory, step_count: int,
                   strategic_insights: list = None) -> str:
    """Deep reflection every N sequences — revise strategy."""
    mem_text = memory.to_text()
    winning = memory.winning_text()
    perception = full_perception(grid)
    color_tries, color_changes = memory.color_tries_changes()
    effectiveness = color_effectiveness_summary(color_tries, color_changes)
    click_hist = memory.click_history_text(n=8)

    knowledge_block = ""
    if strategic_insights:
        lines = [f"  - {s}" for s in strategic_insights]
        knowledge_block = f"\nKNOWN GAME KNOWLEDGE (verified facts):\n" + "\n".join(lines) + "\n"

    return f"""DEEP REFLECTION — {step_count} actions, {len(memory.sequence_log)} sequences played.

RESULTS SO FAR:
- Levels completed: {levels_completed}/{win_levels}
- Total level-ups: {memory.total_level_ups}
{knowledge_block}
CURRENT GRID PERCEPTION:
{perception}

RECENT CLICK HISTORY (causal):
{click_hist}

COLOR EFFECTIVENESS:
{effectiveness}

SEQUENCE HISTORY:
{mem_text}
{winning}

CURRENT HYPOTHESIS: {memory.hypothesis or '(none yet)'}

REFLECT — cross-reference what you know with what you observed:
1. How do the KNOWN GAME KNOWLEDGE facts explain the click outcomes you've seen?
2. Which specific actions/colors caused changes? Do they match what the knowledge predicts?
3. Given the rotation puzzle mechanics, what is your STRATEGIC PLAN for the next sequences?
4. What specific positions or colors should you target next?

Respond with ONLY a JSON object:
{{"hypothesis": "<theory consistent with known facts + observations>", "strategy": "<specific plan: what to click and why>", "key_insight": "<most important realization>"}}"""


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
    print("SAGE ARC-AGI-3 Game Runner v3 — GridCartridgeIRP Cross-Session Memory")
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

    # Cross-session memory: load prior reflections and discoveries directly
    prior_reflections = gc.get_reflections()
    prior_discoveries = gc.get_discoveries()
    n_prior = len(gc._all_entries)

    # Load strategic insights from game cartridge JSON (older membot format)
    # These are verified facts about the game (e.g. "lp85 is a rotation puzzle")
    strategic_insights = []
    cartridge_json_path = f"arc-agi-3/experiments/cartridges/{gc.game_family}.json"
    try:
        import os
        if os.path.exists(cartridge_json_path):
            with open(cartridge_json_path) as _f:
                _cdata = json.load(_f)
            strategic_insights = _cdata.get("strategic_insights", [])
    except Exception as _e:
        print(f"  (Could not load strategic insights from {cartridge_json_path}: {_e})")
    if strategic_insights:
        print(f"\n  STRATEGIC INSIGHTS ({len(strategic_insights)}):")
        for s in strategic_insights[:4]:
            print(f"    • {s[:90]}")

    memory = GameMemory()
    if strategic_insights:
        # Strategic insights take priority — seed hypothesis from verified facts,
        # not from old reflections that may predate the rotation puzzle discovery
        memory.hypothesis = strategic_insights[0][:120]
        memory.strategy = (
            "Identify and click rotation buttons (small 4x4 squares). "
            "Try different button positions each sequence. "
            "Track which buttons cause changes and which alignment direction helps."
        )
        print(f"  Memory seeded from strategic_insights ({len(strategic_insights)} facts).")
    elif prior_reflections:
        # Fall back to prior reflections only when no verified knowledge exists
        last_r = prior_reflections[-1].step_record
        memory.hypothesis = last_r.get("hypothesis", "")
        memory.strategy = last_r.get("strategy", "")
        print(f"  Cartridge: {n_prior} entries loaded — seeding from {len(prior_reflections)} reflection(s)")
        print(f"  Hypothesis: {memory.hypothesis[:90]}")
        print(f"  Strategy:   {memory.strategy[:90]}")
    elif prior_discoveries:
        last_d = prior_discoveries[-1].step_record
        print(f"  Cartridge: {n_prior} entries — {len(prior_discoveries)} discovery(s), no reflections yet")
        print(f"  Last win: {last_d.get('action_rationale','')[:80]}")
    else:
        print(f"  Cartridge: no prior memory for {gc.game_family} (first session)")
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
                                    frame_data.win_levels, memory, total_steps,
                                    strategic_insights=strategic_insights)
            t0 = time.time()
            rresponse = ask_ollama(rprompt, max_tokens=300)
            reflect_s = time.time() - t0
            insight = ""
            try:
                if "{" in rresponse and "}" in rresponse:
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
                             prior_memory=prior_memory,
                             strategic_insights=strategic_insights)
        t0 = time.time()
        presponse = ask_ollama(pprompt, max_tokens=80 if args.fast else 300)
        plan_s = time.time() - t0

        sequence = []
        llm_clicks = []  # LLM-specified (r,c) targets for each ACTION6
        goal = ""
        try:
            if "{" in presponse and "}" in presponse:
                pj = json.loads(presponse[presponse.index("{"):presponse.rindex("}") + 1])
                sequence = [int(a) for a in pj.get("sequence", [])]
                memory.hypothesis = pj.get("hypothesis", memory.hypothesis)
                goal = pj.get("goal", "")[:60]
                # Parse LLM-directed click coordinates (x=col, y=row to match perception @(x,y))
                for click in pj.get("clicks", []):
                    if isinstance(click, dict):
                        # Support both {x,y} (preferred) and legacy {r,c}
                        if "x" in click and "y" in click:
                            cc = int(click["x"])   # x = column
                            cr = int(click["y"])   # y = row
                        elif "r" in click and "c" in click:
                            cr = int(click["r"])
                            cc = int(click["c"])
                        else:
                            continue
                        # Clamp to grid bounds
                        cr = max(0, min(grid.shape[0] - 1, cr))
                        cc = max(0, min(grid.shape[1] - 1, cc))
                        llm_clicks.append((cr, cc))
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
        click_coords = ",".join(f"r{r}c{c}" for r, c in llm_clicks[:3])
        click_info = f"  clicks:[{click_coords}{'...' if len(llm_clicks)>3 else ''}]" if llm_clicks else ""
        print(f"Seq {seq_num:2d}: [{' → '.join(seq_names)}]  plan:{plan_s:.1f}s{click_info}  goal: {goal}")

        # --- Execute sequence ---
        seq_changes = 0
        seq_diffs = []
        level_up_during = False
        click_idx = 0  # index into llm_clicks for ACTION6s

        for i, action in enumerate(sequence):
            prev_grid = grid.copy()

            # Execute action — ACTION6 requires coordinates
            ga = INT_TO_GAME_ACTION[action]
            if action == 6:
                if click_idx < len(llm_clicks):
                    # Use LLM-directed coordinate
                    r, c = llm_clicks[click_idx]
                    color = int(grid[r, c])
                    click_idx += 1
                else:
                    # Fallback: heuristic target selection
                    r, c, color = find_click_target(
                        grid, memory.color_stats, memory.clicked_this_level
                    )
                frame_data = env.step(ga, data={'x': c, 'y': r})
            else:
                frame_data = env.step(ga)
            new_grid = np.array(frame_data.frame) if frame_data.frame is not None else grid
            if new_grid.ndim == 3:
                new_grid = new_grid[-1]
            if new_grid.shape != grid.shape:
                new_grid = grid  # safety: ignore malformed frame
            n_changed = int(np.sum(grid != new_grid))
            changed = n_changed > 0
            if action == 6:
                memory.record_click(r, c, color, changed)
                # Record causal feedback for LLM reasoning
                d_preview = diff_summary(prev_grid, new_grid)
                region_desc = d_preview.get("desc", "")[:60] if n_changed > 0 else ""
                memory.record_click_detail(total_steps + 1, r, c, color, n_changed, region_desc)
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
                memory.reset_level_clicks()
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
