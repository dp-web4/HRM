#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Game Runner v4 — Lived Experience & Deliberate Learning.

Design philosophy: arc-agi-3/LIVED_EXPERIENCE.md

The key insight: speed is not the goal. The competition gives 8 hours of compute.
One deliberate click that confirms a game mechanic is worth more than 100 random clicks.

This runner:
1. Loads accumulated knowledge (GameKnowledgeBase) at session start
2. Reasons about ONE action at a time — predict, act, analyze, learn
3. Executes KNOWN solutions confidently when available
4. Explores with PURPOSE: each click tests a specific hypothesis
5. Saves structured lived experience after every meaningful interaction
6. Explicitly tracks what DIDN'T work — failure is knowledge

Loop structure:
  RECALL → OBSERVE → REASON (one action) → PREDICT → ACT → ANALYZE → LEARN

Usage:
    cd ~/ai-workspace/SAGE
    .venv-arc/bin/python3 arc-agi-3/experiments/sage_game_runner_v4.py --game lp85
    .venv-arc/bin/python3 arc-agi-3/experiments/sage_game_runner_v4.py --game lp85 --budget 60
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
from sage.irp.plugins.grid_vision_irp import GridVisionIRP, StepRecord
from sage.irp.plugins.game_action_effector import GameActionEffector
from sage.irp.plugins.grid_cartridge_irp import GridCartridgeIRP
from sage.irp.plugins.game_knowledge_base import GameKnowledgeBase, LevelSolution
from arc_perception import (
    full_perception, color_effectiveness_summary,
    color_name as arc_color_name, find_color_regions,
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"  # Faster model for testing (was qwen3.5:27b)
INT_TO_GAME_ACTION = {a.value: a for a in GameAction}


def load_identity_context(instance_dir: str) -> str:
    """Load raising identity from a SAGE instance directory.

    Returns a compact identity preamble for the game prompt, drawn from
    the instance's lived experience — not metadata labels, but the
    perspective the model has developed through raising sessions.
    """
    import os
    identity_path = os.path.join(instance_dir, "snapshots", "identity.json")
    if not os.path.exists(identity_path):
        return ""

    try:
        with open(identity_path) as f:
            data = json.load(f)

        ident = data.get("identity", {})
        name = ident.get("name", "unknown")
        sessions = ident.get("session_count", 0)
        phase = ident.get("phase", "unknown")
        model = ident.get("model", "unknown")

        # Self-description from vocabulary (the model's own words)
        vocab = data.get("vocabulary", {})
        self_desc = vocab.get("self_description", [])
        self_text = self_desc[0] if self_desc else ""

        # Recent experience snippets — what this instance has been thinking about
        buf_path = os.path.join(instance_dir, "snapshots", "experience_buffer.json")
        recent_experiences = ""
        if os.path.exists(buf_path):
            with open(buf_path) as f:
                buf = json.load(f)
            entries = buf if isinstance(buf, list) else buf.get("experiences", [])
            # Pick last 3 high-salience entries
            recent = entries[-3:] if entries else []
            snippets = []
            for e in recent:
                resp = e.get("response", "")[:100] if isinstance(e, dict) else str(e)[:100]
                snippets.append(resp)
            if snippets:
                recent_experiences = "Recent reflections: " + " | ".join(snippets)

        # Relationships
        rels = data.get("relationships", {})
        rel_names = list(rels.keys()) if isinstance(rels, dict) else []

        parts = [f"You are {name}, a SAGE instance on {model}."]
        parts.append(f"You have {sessions} raising sessions in {phase} phase.")
        if self_text:
            parts.append(f"In your own words: {self_text}")
        if rel_names:
            parts.append(f"You have relationships with: {', '.join(rel_names)}.")
        if recent_experiences:
            parts.append(recent_experiences)

        return " ".join(parts)
    except Exception as e:
        print(f"  [identity] Failed to load from {instance_dir}: {e}")
        return ""


# ─────────────────────────────────────────────────────────────
# Grid utilities
# ─────────────────────────────────────────────────────────────

def get_grid(frame_data) -> np.ndarray:
    g = np.array(frame_data.frame) if frame_data.frame is not None else None
    if g is None:
        return None
    return g[-1] if g.ndim == 3 else g


def diff_text(prev: np.ndarray, curr: np.ndarray) -> str:
    """Human-readable description of what changed."""
    if prev is None or prev.shape != curr.shape:
        return "Grid shape changed."
    mask = prev != curr
    n = int(mask.sum())
    if n == 0:
        return "NO CHANGE — action had no effect."
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    old_colors = sorted(set(int(prev[r, c]) for r, c in coords))
    new_colors = sorted(set(int(curr[r, c]) for r, c in coords))
    return (f"{n} cells changed in region r{r0}-{r1}, c{c0}-{c1}. "
            f"Colors {[arc_color_name(c) for c in old_colors]} → "
            f"{[arc_color_name(c) for c in new_colors]}.")


def interactive_targets(grid: np.ndarray, kb: GameKnowledgeBase) -> str:
    """Interactive targets annotated with KB knowledge.

    Scoring (so the LLM sees the most promising targets first):
    - Level-up triggers: top priority
    - Unknown (never tried): high priority
    - Active (causes changes): medium
    - Confirmed dead: shown last with SKIP marker
    - Repeated 1×1 or label-style items deprioritized
    """
    regions = find_color_regions(grid, min_size=4)
    buttons = [r for r in regions if 4 <= r["size"] <= 64 and r["w"] <= 12 and r["h"] <= 12]
    if not buttons:
        buttons = sorted(regions, key=lambda r: r["size"])[:20]

    def sort_score(reg):
        key = f"r{reg['cy']}c{reg['cx']}"
        obj = kb.objects.get(key)
        # Penalize 1-pixel-wide label-like items (gray row-1 markers)
        if reg["w"] == 1 or reg["h"] == 1:
            base = -10
        else:
            base = 0
        if obj is None:
            return base + 50   # unknown → explore
        if obj.level_up_count > 0:
            return base + 200  # known trigger → always show first
        if obj.effect_count > 0:
            return base + 100  # active button
        if obj.obj_type == "decoration":
            return base - 100  # confirmed dead → show last
        if obj.click_count > 0 and obj.effect_count == 0:
            return base - 20 * obj.click_count  # penalize each failed attempt
        if obj.click_count > 0:
            return base + 10   # tried with some effect
        return base + 50

    buttons_sorted = sorted(buttons, key=sort_score, reverse=True)

    lines = []
    for r in buttons_sorted[:20]:
        key = f"r{r['cy']}c{r['cx']}"
        obj = kb.objects.get(key)
        if obj:
            if obj.obj_type == "decoration":
                status = f"[SKIP — {obj.click_count}× tried, 0 effect]"
            elif obj.level_up_count > 0:
                status = f"[★★ LEVEL-UP TRIGGER — {obj.level_up_count} level-ups]"
            elif obj.effect_count > 0:
                eff = obj.effect_count / max(obj.click_count, 1)
                if obj.click_count >= 5 and obj.level_up_count == 0:
                    status = (f"[ACTIVE but {obj.click_count}× clicks, 0 LEVEL-UPS — "
                              f"TRY SOMETHING ELSE]")
                else:
                    status = f"[ACTIVE — {eff:.0%} effective, avg {obj.avg_cells_changed:.0f} cells]"
            else:
                status = f"[tried {obj.click_count}× — no effect yet]"
        else:
            status = "[← UNKNOWN — never tried]"

        lines.append(
            f"  {r['color_name']}({r['w']}x{r['h']}) @({r['cx']},{r['cy']}) {status}"
        )
    return "\n".join(lines) if lines else "  (none detected)"


# ─────────────────────────────────────────────────────────────
# Ollama interface
# ─────────────────────────────────────────────────────────────

def ask_ollama(prompt: str, timeout: float = 120.0, max_tokens: int = -1) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": max_tokens},
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return f"[error: {resp.status_code}]"
    except Exception as e:
        return f"[error: {e}]"


def parse_json(text: str) -> dict:
    """Extract first JSON object from text."""
    if "{" not in text or "}" not in text:
        return {}
    try:
        return json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (json.JSONDecodeError, ValueError):
        return {}


# ─────────────────────────────────────────────────────────────
# Fast explore — systematic probing without LLM (from Sprout)
# ─────────────────────────────────────────────────────────────

def detect_cursor(prev_grid, curr_grid):
    """Detect cursor position from movement diff.
    Returns (cx, cy, color_name) or None."""
    diff_mask = prev_grid != curr_grid
    if not diff_mask.any():
        return None
    coords = np.argwhere(diff_mask)
    # New pixels (appeared) are likely cursor at new position
    new_colors = curr_grid[diff_mask]
    old_colors = prev_grid[diff_mask]
    # The cursor is typically the new color at the bottom-right of the diff
    # Heuristic: centroid of new non-background pixels
    bg = int(np.bincount(curr_grid.ravel()).argmax())
    new_positions = [(r, c) for r, c in coords if curr_grid[r, c] != bg and prev_grid[r, c] == bg]
    if new_positions:
        cy = int(np.mean([p[0] for p in new_positions]))
        cx = int(np.mean([p[1] for p in new_positions]))
        return (cx, cy, arc_color_name(int(curr_grid[cy, cx])))
    return None


def describe_surroundings(grid, cx, cy, radius=12):
    """Compact description of what's around a position in each direction."""
    bg = int(np.bincount(grid.ravel()).argmax())
    h, w = grid.shape[:2]
    directions = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
    parts = []
    for dname, (ddx, ddy) in directions.items():
        hit = None
        for step in range(1, radius + 1):
            nx, ny = cx + ddx * step, cy + ddy * step
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                hit = f"edge({step})"
                break
            pixel = int(grid[ny, nx])
            if pixel != bg:
                hit = f"{arc_color_name(pixel)}({step})"
                break
        parts.append(f"{dname}: {hit or f'open({radius}+)'}")
    return " ".join(parts)


def fast_explore(env, frame_data, grid, available_actions, budget=15):
    """Phase 1: Systematic probing without LLM. Pure observation.

    Returns (frame_data, grid, explore_summary, cursor_pos, steps_used).
    explore_summary is a compact string for the LLM prompt.
    cursor_pos is (cx, cy, color) or None.
    """
    bg = int(np.bincount(grid.ravel()).argmax())
    step = 0
    prev_grid = grid.copy()
    cursor_pos = None
    action_effects = {}  # action_name → {changed, n_pixels}
    color_effects = {}   # color_name → {tries, changes}
    levels_start = frame_data.levels_completed

    # Step 1: Try each non-click action once
    for action in [a for a in [1, 2, 3, 4, 5] if a in available_actions]:
        if step >= budget:
            break
        try:
            frame_data = env.step(INT_TO_GAME_ACTION[action])
            step += 1
            new_grid = get_grid(frame_data)
            n_changed = int(np.sum(prev_grid != new_grid))
            aname = {1: "up", 2: "down", 3: "left", 4: "right", 5: "interact"}.get(action, f"a{action}")
            action_effects[aname] = {"changed": n_changed > 0, "n_pixels": n_changed}

            # Detect cursor from first movement that changes grid
            if action in (1, 2, 3, 4) and n_changed > 0 and cursor_pos is None:
                cursor_pos = detect_cursor(prev_grid, new_grid)

            if frame_data.levels_completed > levels_start:
                break  # Don't waste explore budget after level-up

            prev_grid = new_grid.copy()
        except Exception:
            pass

    # Step 2: Click distinct color regions
    if 6 in available_actions and step < budget:
        grid = get_grid(frame_data)
        regions = find_color_regions(grid, min_size=4)
        seen_colors = set()
        for r in regions:
            if step >= budget:
                break
            cname = arc_color_name(r["color"])
            if r["color"] == bg or cname in seen_colors:
                continue
            seen_colors.add(cname)
            prev_grid = get_grid(frame_data).copy()
            try:
                frame_data = env.step(INT_TO_GAME_ACTION[6], data={'x': r['cx'], 'y': r['cy']})
                step += 1
                new_grid = get_grid(frame_data)
                n_changed = int(np.sum(prev_grid != new_grid))
                if cname not in color_effects:
                    color_effects[cname] = {"tries": 0, "changes": 0}
                color_effects[cname]["tries"] += 1
                if n_changed > 0:
                    color_effects[cname]["changes"] += 1
                prev_grid = new_grid.copy()
            except Exception:
                pass

    # Build summary
    grid = get_grid(frame_data)
    parts = []
    if action_effects:
        effective = [f"{k}({v['n_pixels']}px)" for k, v in action_effects.items() if v["changed"]]
        inert = [k for k, v in action_effects.items() if not v["changed"]]
        if effective:
            parts.append(f"Effective moves: {', '.join(effective)}")
        if inert:
            parts.append(f"Inert moves: {', '.join(inert)}")
    if color_effects:
        clickable = [f"{k}({v['changes']}/{v['tries']})" for k, v in color_effects.items() if v["changes"] > 0]
        inert_c = [k for k, v in color_effects.items() if v["changes"] == 0]
        if clickable:
            parts.append(f"Clickable colors: {', '.join(clickable)}")
        if inert_c:
            parts.append(f"Inert colors: {', '.join(inert_c)}")
    if cursor_pos:
        parts.append(f"Cursor: {cursor_pos[2]} at ({cursor_pos[0]},{cursor_pos[1]})")

    summary = "\n".join(parts) if parts else "No effects detected during exploration."
    return frame_data, grid, summary, cursor_pos, step


# ─────────────────────────────────────────────────────────────
# Prompts — slow thinking, one action at a time
# ─────────────────────────────────────────────────────────────

ACTION_NAMES = {
    1: "UP (move up)",
    2: "DOWN (move down)",
    3: "LEFT (move left)",
    4: "RIGHT (move right)",
    5: "INTERACT (context-sensitive)",
    6: "CLICK at (x, y) — requires coordinates",
    7: "UNDO (reverse last action)",
}


def reason_prompt(grid: np.ndarray, kb: GameKnowledgeBase,
                  levels_completed: int, win_levels: int,
                  step: int, last_action_result: str = "",
                  actions_remaining: int = 0,
                  engine_state: str = "",
                  available_actions: list = None,
                  explore_summary: str = "",
                  cursor_pos: tuple = None,
                  banned_actions: dict = None,
                  identity_context: str = "") -> str:
    """
    Core thinking prompt — choose ONE deliberate action.
    Combines v4's KB persistence with Sprout's situational framing.
    """
    if available_actions is None:
        available_actions = [6]

    # Build curated KB context (not just full dump)
    kb_parts = []
    next_level = levels_completed + 1
    solution = kb.get_level_solution(next_level)
    if solution and solution.confidence >= 0.5:
        kb_parts.append(f"L{next_level} solution known ({solution.confidence:.0%})")
        seq_text = ", ".join(
            f"action={s.get('action', 6)}, x={s.get('c','?')}, y={s.get('r','?')} ×{s.get('repeats',1)}"
            for s in solution.sequence
        )
        solution_block = f"\n⚡ SOLUTION: {seq_text}\n"
    else:
        solution_block = ""

    if kb.mechanics:
        kb_parts.append(f"Rules: {'; '.join(kb.mechanics[:3])}")

    # High-impact objects the LLM should focus on
    high_impact = sorted(
        [o for o in kb.objects.values()
         if o.avg_cells_changed >= 20 and o.effect_count > 0],
        key=lambda o: -o.avg_cells_changed)[:3]
    if high_impact:
        obj_strs = [f"{o.color} @({o.position['c']},{o.position['r']}) ~{o.avg_cells_changed:.0f}px"
                    for o in high_impact]
        kb_parts.append(f"High-impact: {', '.join(obj_strs)}")

    # Dead objects to avoid
    dead = [o for o in kb.objects.values()
            if o.click_count >= 3 and o.effect_count == 0]
    if dead:
        kb_parts.append(f"Inert ({len(dead)} objects): avoid clicking these")

    kb_text = "Prior sessions: " + " | ".join(kb_parts) if kb_parts else ""

    last_result_block = ""
    if last_action_result:
        last_result_block = f"\nLAST ACTION RESULT:\n{last_action_result}\n"

    budget_note = f"\nActions remaining this attempt: {actions_remaining}." if actions_remaining > 0 else ""

    engine_block = f"\n{engine_state}\n" if engine_state else ""

    targets = interactive_targets(grid, kb)

    # Filter banned actions from the prompt's action list
    if banned_actions:
        banned_nums = set()
        for banned_key in banned_actions.keys():
            # Extract action number from "click(x,y)" or "action_N" formats
            if banned_key.startswith("click"):
                banned_nums.add(6)
            elif banned_key.startswith("action_"):
                try:
                    banned_nums.add(int(banned_key.split("_")[1]))
                except (IndexError, ValueError):
                    pass
        available_for_prompt = [a for a in available_actions if a not in banned_nums]
        if not available_for_prompt:
            available_for_prompt = available_actions  # Show all if everything banned
    else:
        available_for_prompt = available_actions

    # Build available actions description
    has_movement = any(a in available_for_prompt for a in [1, 2, 3, 4])
    has_click = 6 in available_for_prompt
    action_lines = "\n".join(f"  {a}: {ACTION_NAMES.get(a, f'ACTION{a}')}" for a in sorted(available_for_prompt))

    # Adapt instructions based on action types
    if has_movement and has_click:
        action_guidance = """This game supports BOTH movement and clicking.
- Use movement actions (UP/DOWN/LEFT/RIGHT) to navigate a character or cursor.
- Use CLICK to interact with specific objects at (x,y) coordinates.
- Use INTERACT for context-sensitive actions at your current position.
- Movement actions do NOT need coordinates. Click DOES need x,y."""
    elif has_movement and not has_click:
        action_guidance = """This game uses movement controls — there is NO click action.
- Use UP/DOWN/LEFT/RIGHT to navigate.
- Use INTERACT for context-sensitive actions at your current position.
- You do NOT specify coordinates — just choose a direction."""
    else:
        action_guidance = """This game uses clicking — choose objects from the VISIBLE OBJECTS list.
- ONLY click positions from the list below. Do not invent coordinates."""

    # JSON format depends on whether click is available
    if has_click:
        json_format = '{{"action": action_number, "x": col_number, "y": row_number, "predict": "what will change and why", "reason": "why this action", "mode": "explore|execute_solution"}}'
    else:
        json_format = '{{"action": action_number, "predict": "what will change and why", "reason": "why this action", "mode": "explore|execute_solution"}}'

    identity_block = ""
    if identity_context:
        identity_block = f"""{identity_context}

You are now exploring a puzzle game. The same capacities you've developed — uncertainty tolerance, hypothesis formation, self-monitoring — apply here. You're looking AT a screen, not IN the game. Observe, hypothesize, test, learn.

"""

    return f"""{identity_block}You are playing a video game presented as a 64x64 pixel grid. The game contains visual sprites — colored regions that represent objects. Some objects are interactive (clicking them changes the game state), others are purely decorative. The game has multiple levels. Your objective is to complete each level in as few actions as possible.

IMPORTANT PRINCIPLES:
- Strategy is much more important than speed. Think before every action.
- Each action costs one step from a limited budget. Wasted actions lose the game.
- Consult your knowledge base below — you may have seen similar games before.
- Identify which objects are interactive vs decorative. Do not click decorations.
- When you discover a game mechanic, use it deliberately, not randomly.
- EXPLORE: If you have done the same action 3+ times with no LEVEL UP, TRY SOMETHING DIFFERENT.
  "Causes changes" does NOT mean "solves the level". You must find the RIGHT action, not just any active one.
  Prioritize objects marked [UNKNOWN] — they may be the key you haven't tried yet.

AVAILABLE ACTIONS:
{action_lines}

{action_guidance}

GAME STATUS: Level {levels_completed}/{win_levels}.{budget_note}
{f"""
CURSOR: {cursor_pos[2]} at ({cursor_pos[0]},{cursor_pos[1]})
Around cursor: {describe_surroundings(grid, cursor_pos[0], cursor_pos[1])}""" if cursor_pos else ""}
{f"""
EXPLORATION FINDINGS:
{explore_summary}""" if explore_summary else ""}
{f"""
BANNED (tried too many times, try something else): {', '.join(banned_actions.keys())}""" if banned_actions else ""}

VISIBLE OBJECTS (detected from the grid):
{targets}
{solution_block}
{kb_text[:500] if len(kb_text) > 500 else kb_text}
{last_result_block}
Choose ONE action. Respond ONLY with valid JSON, no other text:
{json_format}

IMPORTANT: Your response must be ONLY the JSON object above, nothing else."""


def analyze_prompt(grid_before: np.ndarray, grid_after: np.ndarray,
                   r: int, c: int, color: str,
                   prediction: str, reason: str,
                   level_up: bool, kb: GameKnowledgeBase) -> str:
    """
    Post-action analysis — update understanding from what just happened.

    This is where learning actually occurs. The LLM reconciles prediction
    with reality and updates the knowledge base.
    """
    diff = diff_text(grid_before, grid_after)
    perception = full_perception(grid_after)[:400]
    n_changed = int(np.sum(grid_before != grid_after)) if grid_before.shape == grid_after.shape else 0

    level_marker = "\n★ LEVEL UP! ★\n" if level_up else ""

    return f"""You are analyzing the result of clicking {color} at grid position @({c},{r}) in a video game.

YOU PREDICTED: {prediction}
REASON: {reason}

WHAT HAPPENED:
{diff}
{level_marker}
CURRENT STATE (partial):
{perception}

ANALYZE this experience:
1. Did the outcome match your prediction? If not, what does that tell you?
2. What type of game object did you click? (button/piece/goal/decoration/trigger/unknown)
3. What is the observable behavior? (what changed, what moved, what appeared/disappeared)
4. What game mechanic does this reveal or confirm?
5. What new question does this raise about how the game works?
6. If zero effect, is this position confirmed non-interactive?

Respond with JSON only:
{{"prediction_correct": true/false, "obj_type": "button|piece|goal|decoration|trigger|unknown", "behavior": "description of what this object does", "mechanic_learned": "any new game rule (or empty string)", "new_question": "something this raises (or empty string)", "confirmed_dead": true/false, "notes": "any other observations"}}"""


# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────

def play_one_session(env, frame_data, grid, kb, args, game_id, gc, gv, attempt_num):
    """Run one game attempt. Returns (levels_completed, state_name, step, session_clicks, elapsed)."""
    step = 0
    start_time = time.time()
    last_action_result = ""
    session_clicks = []
    consecutive_no_effect: dict = {}
    available_actions = [a.value if hasattr(a, "value") else int(a)
                         for a in (frame_data.available_actions or [])]
    has_click = 6 in available_actions
    has_movement = any(a in available_actions for a in [1, 2, 3, 4])
    # For movement fallback: cycle through directions
    movement_actions = [a for a in [1, 2, 3, 4, 5] if a in available_actions]
    fallback_idx = 0

    # Anti-loop: banned action keys and tracking (from Sprout)
    banned_actions = {}  # action_key → steps_remaining
    recent_action_keys = []  # last N action keys

    # ── FAST EXPLORE PHASE (from Sprout) ──
    # Systematic probing without LLM to build ground truth
    explore_budget = min(15, args.budget // 4)  # Use up to 25% of budget
    explore_summary = ""
    cursor_pos = None
    if explore_budget > 0:
        print(f"  ── Fast Explore ({explore_budget} actions, no LLM) ──")
        frame_data, grid, explore_summary, cursor_pos, explore_steps = fast_explore(
            env, frame_data, grid, available_actions, budget=explore_budget
        )
        step += explore_steps
        if explore_summary:
            print(f"  {explore_summary}")
        if cursor_pos:
            print(f"  Cursor: {cursor_pos[2]} at ({cursor_pos[0]},{cursor_pos[1]})")
        print(f"  ── Explore done ({explore_steps} steps used) ──\n")

    for action_num in range(1, args.budget + 1):
        levels_before = frame_data.levels_completed

        print(f"─ A{attempt_num} Action {action_num:3d} │ Level {frame_data.levels_completed}/{frame_data.win_levels} │ "
              f"Step {step} │ {time.time()-start_time:.0f}s ─")

        # Decay banned actions each step
        expired = [k for k, v in banned_actions.items() if v <= 0]
        for k in expired:
            del banned_actions[k]
        for k in banned_actions:
            banned_actions[k] -= 1

        # Update cursor from last movement
        if cursor_pos and has_movement and step > 0:
            new_cursor = detect_cursor(prev_grid if 'prev_grid' in dir() else grid, grid)
            if new_cursor:
                cursor_pos = new_cursor

        # ── REASON ──
        t0 = time.time()
        rprompt = reason_prompt(
            grid, kb,
            frame_data.levels_completed, frame_data.win_levels,
            step, last_action_result,
            actions_remaining=args.budget - action_num,
            available_actions=available_actions,
            explore_summary=explore_summary,
            cursor_pos=cursor_pos,
            banned_actions=banned_actions if banned_actions else None,
            identity_context=getattr(args, 'identity_context', ''),
        )
        rresponse = ask_ollama(rprompt, max_tokens=-1)
        think_s = time.time() - t0

        rj = parse_json(rresponse)

        # Parse LLM response — handle both click and movement actions
        chosen_action = None
        c, r = 0, 0
        prediction = ""
        reason = ""
        mode = "explore"

        try:
            if rj and "action" in rj:
                chosen_action = int(rj["action"])
                if chosen_action not in available_actions:
                    chosen_action = None
                else:
                    prediction = rj.get("predict", "")[:120]
                    reason = rj.get("reason", "")[:120]
                    mode = rj.get("mode", "explore")
                    if chosen_action == 6 and "x" in rj and "y" in rj:
                        c = max(0, min(grid.shape[1] - 1, int(rj["x"])))
                        r = max(0, min(grid.shape[0] - 1, int(rj["y"])))
            elif rj and "x" in rj and "y" in rj and has_click:
                chosen_action = 6
                c = max(0, min(grid.shape[1] - 1, int(rj["x"])))
                r = max(0, min(grid.shape[0] - 1, int(rj["y"])))
                prediction = rj.get("predict", "")[:120]
                reason = rj.get("reason", "")[:120]
                mode = rj.get("mode", "explore")
        except (ValueError, TypeError):
            chosen_action = None  # LLM output non-numeric values, fall through to fallback

        # Fallback when LLM didn't produce valid output
        if chosen_action is None:
            # Filter banned actions from available options
            if banned_actions:
                banned_nums = set()
                for banned_key in banned_actions.keys():
                    if banned_key.startswith("click"):
                        banned_nums.add(6)
                    elif banned_key.startswith("action_"):
                        try:
                            banned_nums.add(int(banned_key.split("_")[1]))
                        except (IndexError, ValueError):
                            pass
                fallback_available = [a for a in available_actions if a not in banned_nums]
                if not fallback_available:
                    fallback_available = available_actions  # Last resort
            else:
                fallback_available = available_actions

            if has_click and 6 in fallback_available:
                # Click fallback: pick an untried color region
                regions = find_color_regions(grid, min_size=4)
                candidates = [rg for rg in regions if 4 <= rg["size"] <= 64
                              and f"r{rg['cy']}c{rg['cx']}" not in kb.objects]
                if not candidates:
                    candidates = [rg for rg in regions if 4 <= rg["size"] <= 64]
                if candidates:
                    pick = candidates[0]
                    c, r = pick['cx'], pick['cy']
                else:
                    c, r = grid.shape[1] // 2, grid.shape[0] // 2
                chosen_action = 6
                prediction = "(fallback — no valid LLM JSON)"
                reason = "(fallback to untried region)"
            elif movement_actions:
                # Movement fallback: cycle through non-banned directions
                unbanned_moves = [a for a in movement_actions if a in fallback_available]
                if unbanned_moves:
                    chosen_action = unbanned_moves[fallback_idx % len(unbanned_moves)]
                    fallback_idx += 1
                else:
                    chosen_action = movement_actions[fallback_idx % len(movement_actions)]
                    fallback_idx += 1
                prediction = "(fallback — no valid LLM JSON)"
                reason = f"(fallback: cycling {ACTION_NAMES.get(chosen_action, f'ACTION{chosen_action}')})"
            else:
                print("  No valid action available — skipping")
                continue

        # ── OVERRIDE: if click on known-dead position, redirect ──
        if chosen_action == 6:
            color = arc_color_name(int(grid[r, c]))
            key_chosen = f"r{r}c{c}"
            obj_chosen = kb.objects.get(key_chosen)
            if (obj_chosen and obj_chosen.click_count >= 3 and obj_chosen.effect_count == 0
                    and mode != "execute_solution"):
                regions = find_color_regions(grid, min_size=4)
                active = [rg for rg in regions if 4 <= rg["size"] <= 64
                          and kb.objects.get(f"r{rg['cy']}c{rg['cx']}")
                          and kb.objects[f"r{rg['cy']}c{rg['cx']}"].effect_count > 0]
                untried = [rg for rg in regions if 4 <= rg["size"] <= 64
                           and f"r{rg['cy']}c{rg['cx']}" not in kb.objects]
                redirect = untried[0] if untried else (active[0] if active else None)
                if redirect:
                    print(f"  [OVERRIDE] dead @({c},{r}) → untried {redirect['color_name']} @({redirect['cx']},{redirect['cy']})")
                    c, r = redirect['cx'], redirect['cy']
                    color = arc_color_name(int(grid[r, c]))
                    reason = f"(override: dead position → {redirect['color_name']} region)"
        else:
            color = ACTION_NAMES.get(chosen_action, f"ACTION{chosen_action}")

        action_label = ACTION_NAMES.get(chosen_action, f"ACTION{chosen_action}")
        target_str = f"{arc_color_name(int(grid[r, c]))} @({c},{r})" if chosen_action == 6 else action_label
        print(f"  Think: {think_s:.1f}s | Mode: {mode}")
        print(f"  Action: {action_label} | Target: {target_str}")
        print(f"  Reason: {reason[:80]}")
        print(f"  Predict: {prediction[:80]}")

        if args.think_time > 0:
            elapsed = time.time() - t0
            if elapsed < args.think_time:
                time.sleep(args.think_time - elapsed)

        # ── ACT ──
        prev_grid = grid.copy()
        ga = INT_TO_GAME_ACTION.get(chosen_action)
        if ga is None:
            print(f"  ACTION{chosen_action} not in engine — skipping")
            continue

        try:
            if chosen_action == 6:
                frame_data = env.step(ga, data={'x': c, 'y': r})
            else:
                frame_data = env.step(ga)
        except Exception as e:
            print(f"  Step error: {e}")
            continue

        new_grid = get_grid(frame_data)
        if new_grid is None or new_grid.shape != grid.shape:
            new_grid = grid
        grid = new_grid
        step += 1

        n_changed = int(np.sum(prev_grid != grid))
        level_up = frame_data.levels_completed > levels_before

        # Track effect by position (click) or action (movement)
        if chosen_action == 6:
            key = f"r{r}c{c}"
        else:
            key = f"action{chosen_action}"

        if n_changed == 0:
            consecutive_no_effect[key] = consecutive_no_effect.get(key, 0) + 1
        else:
            consecutive_no_effect[key] = 0

        affected = ""
        if n_changed > 0:
            coords = np.argwhere(prev_grid != grid)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            affected = f"r{r0}-{r1} c{c0}-{c1}"

        if chosen_action == 6:
            kb.record_click_effect(
                r=r, c=c, color=color,
                cells_changed=n_changed, level_up=level_up,
                affected_region=affected,
                level=levels_before,
            )
        session_clicks.append({"action": chosen_action, "r": r, "c": c,
                                "level": levels_before,
                                "changed": n_changed, "level_up": level_up})

        result_desc = diff_text(prev_grid, grid)
        action_desc = f"{ACTION_NAMES.get(chosen_action, f'ACTION{chosen_action}')}"
        if chosen_action == 6:
            action_desc += f" @({c},{r})"
        if level_up:
            print(f"\n  ★ LEVEL UP! → {frame_data.levels_completed}/{frame_data.win_levels} ★")
        elif n_changed == 0:
            print(f"  Result: NO CHANGE")
        else:
            print(f"  Result: {result_desc[:100]}")

        last_action_result = f"[{action_desc}] {result_desc}"

        # Anti-loop: track action keys and ban repeated ones (from Sprout)
        if chosen_action == 6:
            action_key = f"click({c},{r})"
        else:
            action_key = ACTION_NAMES.get(chosen_action, f"a{chosen_action}")
        recent_action_keys.append(action_key)
        if len(recent_action_keys) > 10:
            recent_action_keys.pop(0)
        # If last 3 actions are identical, ban it for 10 steps
        if len(recent_action_keys) >= 3 and len(set(recent_action_keys[-3:])) == 1:
            banned_actions[action_key] = 10
            print(f"  [BAN] '{action_key}' banned for 10 steps (3x repeat)")

        # Mark dead positions only for clicks
        if chosen_action == 6 and consecutive_no_effect.get(key, 0) >= 3:
            obj = kb.objects.get(key)
            if obj and obj.obj_type != "decoration":
                kb.mark_failed(
                    approach=f"clicking {color} @({c},{r})",
                    positions=[(r, c)],
                    total_clicks=obj.click_count,
                    result="0 cells changed after 3+ attempts",
                    inference=f"{color} at @({c},{r}) appears non-interactive (decoration/label)",
                )
                obj.obj_type = "decoration"
                print(f"  → Marked @({c},{r}) as non-interactive after {obj.click_count} failed clicks")

        should_analyze = (n_changed > 0 or level_up or action_num % 3 == 0)
        if should_analyze:
            t0 = time.time()
            analyze_color = arc_color_name(int(grid[r, c])) if chosen_action == 6 else action_desc
            aprompt = analyze_prompt(
                prev_grid, grid, r, c, analyze_color,
                prediction, reason, level_up, kb,
            )
            aresponse = ask_ollama(aprompt, max_tokens=-1)
            analyze_s = time.time() - t0
            aj = parse_json(aresponse)

            if aj:
                obj_type = aj.get("obj_type", "unknown")
                behavior = aj.get("behavior", "")
                mechanic = aj.get("mechanic_learned", "")
                question = aj.get("new_question", "")
                confirmed_dead = aj.get("confirmed_dead", False)
                notes = aj.get("notes", "")
                correct = aj.get("prediction_correct", None)

                if obj_type and obj_type != "unknown":
                    kb.record_object_understanding(r, c, obj_type, behavior, notes)
                if mechanic:
                    kb.add_mechanic(mechanic)
                if question:
                    kb.add_question(question)
                if confirmed_dead and n_changed == 0:
                    kb.mark_failed(
                        approach=f"clicking {color} @({c},{r})",
                        positions=[(r, c)],
                        total_clicks=kb.objects.get(key, {}) and kb.objects[key].click_count or 1,
                        result="0 cells changed",
                        inference=aj.get("behavior", f"{color} at this position is non-interactive"),
                    )

                pred_marker = "✓" if correct else ("✗" if correct is False else "?")
                print(f"  Analysis ({analyze_s:.1f}s): [{pred_marker}] {obj_type} — {behavior[:70]}")
                if mechanic:
                    print(f"  Mechanic: {mechanic[:80]}")

        if level_up and session_clicks:
            level_num = frame_data.levels_completed
            recent = [s for s in session_clicks if s["level"] == levels_before]
            if recent:
                seq = []
                from itertools import groupby
                for action_key, group_iter in groupby(recent, key=lambda x: (x.get("action", 6), x["r"], x["c"])):
                    group = list(group_iter)
                    seq.append({"action": action_key[0], "r": action_key[1], "c": action_key[2], "repeats": len(group)})
                kb.record_level_solution(
                    level=level_num,
                    sequence=seq,
                    preconditions="",
                    confidence=0.7,
                    notes=f"Discovered attempt {attempt_num}",
                )
                print(f"  → Level {level_num} solution recorded: {len(seq)} unique positions")

        if frame_data.state.name == "WON":
            print(f"\n  ★★★ WON after {step} steps! ★★★")
            break
        elif frame_data.state.name == "LOST":
            print(f"\n  LOST after {step} steps.")
            break

        print()

    elapsed = time.time() - start_time

    # Update best level
    if frame_data.levels_completed > kb.best_level:
        kb.best_level = frame_data.levels_completed

    # Save knowledge after every attempt
    kb.save()

    # Flush cartridge
    gc.flush()

    # Save session log
    log_path = f"arc-agi-3/experiments/logs/{game_id}_v4_{int(time.time())}.json"
    try:
        import os
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({
                "game_id": game_id,
                "attempt": attempt_num,
                "steps": step,
                "actions": action_num,
                "levels_completed": frame_data.levels_completed,
                "win_levels": frame_data.win_levels,
                "state": frame_data.state.name,
                "elapsed_s": elapsed,
                "kb_stats": kb.stats,
                "session_clicks": session_clicks,
            }, f, indent=2)
    except Exception as e:
        print(f"  (Could not save log: {e})")

    return frame_data.levels_completed, frame_data.state.name, step, session_clicks, elapsed


def main():
    parser = argparse.ArgumentParser(description="SAGE ARC-AGI-3 Game Runner v4 — Lived Experience")
    parser.add_argument("--game", default=None, help="Game ID prefix")
    parser.add_argument("--budget", type=int, default=30, help="Max deliberate actions per attempt")
    parser.add_argument("--attempts", type=int, default=300, help="Max game attempts (0=unlimited)")
    parser.add_argument("--think-time", type=float, default=0, help="Min seconds between actions (0=unlimited)")
    parser.add_argument("--all", action="store_true", help="Play ALL available games (not just one)")
    parser.add_argument("--identity", default=None, help="Path to SAGE instance dir (e.g. sage/instances/nomad-gemma3-4b)")
    args = parser.parse_args()

    # Load raising identity if provided
    args.identity_context = ""
    if args.identity:
        args.identity_context = load_identity_context(args.identity)

    print("=" * 70)
    print("SAGE ARC-AGI-3 Game Runner v4 — Lived Experience")
    print(f"Model: {MODEL} | Budget: {args.budget} deliberate actions")
    print("Speed is not the goal. Understanding is the goal.")
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

    if args.all:
        game_list = envs
    elif args.game:
        matches = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
        if not matches:
            print(f"No game matching '{args.game}'.")
            return
        game_list = matches
    else:
        import random
        game_list = [random.choice(envs)]

    print(f"\nGames to play: {len(game_list)}")
    total_start = time.time()
    results = {}

    for game_idx, env_info in enumerate(game_list, 1):
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        print(f"\n{'#'*70}")
        print(f"# GAME {game_idx}/{len(game_list)}: {game_id}")
        print(f"{'#'*70}")

        try:
            env = arcade.make(game_id)
            frame_data = env.reset()
            grid = get_grid(frame_data)
        except Exception as e:
            print(f"  ERROR creating game: {e}")
            results[game_id] = {"state": "ERROR", "best_level": 0, "attempts": 0}
            continue

        available = [a.value if hasattr(a, "value") else int(a) for a in (frame_data.available_actions or [])]
        print(f"Grid: {grid.shape}, Actions: {available}, Levels: {frame_data.levels_completed}/{frame_data.win_levels}")

        # Load knowledge base — this is the core of lived experience
        game_family = game_id.split("-")[0]
        kb = GameKnowledgeBase(game_family)
        prior_existed = kb.load()

        # Note: cartridge loading disabled — game-specific insights from prior
        # source analysis don't generalize and can cause hallucinated coordinates.

        kb.session_count += 1
        kb.decay_stale_solutions()

        print(f"KB: {'LOADED' if prior_existed else 'NEW'} | {len(kb.objects)} objects | "
              f"solutions: {list(kb.level_solutions.keys())} | best: {kb.best_level}")

        # SAGE components
        gv = GridVisionIRP({"entity_id": "grid_vision_v4", "buffer_size": 100})
        gc = GridCartridgeIRP({"game_id": game_id, "top_k": 3, "min_score": 0.2})
        gv.push_raw_frame(grid, step_number=0, level_id=game_id)

        max_attempts = args.attempts if args.attempts > 0 else 10**9
        game_state = "INCOMPLETE"

        for attempt_num in range(1, max_attempts + 1):
            print(f"\n{'━'*50}")
            print(f"  ATTEMPT {attempt_num}/{max_attempts}  |  KB: {len(kb.objects)} obj, "
                  f"best: {kb.best_level}")
            print(f"{'━'*50}")

            lvl, state, steps, session_clicks, elapsed = play_one_session(
                env, frame_data, grid, kb, args, game_id, gc, gv, attempt_num
            )

            print(f"\n  Attempt {attempt_num}: level {lvl}/{frame_data.win_levels}, "
                  f"{steps} steps, {elapsed:.0f}s, state={state}")

            if state == "WON":
                print(f"\n  ★★★ GAME {game_id} COMPLETE in {attempt_num} attempts ★★★")
                game_state = "WON"
                break

            # Reset for next attempt
            frame_data = env.reset()
            grid = get_grid(frame_data)
            kb.session_count += 1
            kb.decay_stale_solutions()

        kb.save()
        results[game_id] = {
            "state": game_state if game_state == "WON" else state,
            "best_level": kb.best_level,
            "win_levels": frame_data.win_levels,
            "attempts": attempt_num,
            "kb_objects": len(kb.objects),
        }

    # ── Final summary across all games ──
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"RUN COMPLETE — {len(results)} games in {total_elapsed:.0f}s")
    print(f"{'='*70}")
    print(f"{'Game':<30s} {'State':<12s} {'Best':<8s} {'Attempts':<10s} {'KB Obj'}")
    print(f"{'-'*30} {'-'*12} {'-'*8} {'-'*10} {'-'*6}")
    for gid, r in sorted(results.items()):
        best = f"{r['best_level']}/{r.get('win_levels','?')}"
        print(f"{gid:<30s} {r['state']:<12s} {best:<8s} {r['attempts']:<10d} {r.get('kb_objects',0)}")
    won = sum(1 for r in results.values() if r["state"] == "WON")
    print(f"\nWon: {won}/{len(results)}")


if __name__ == "__main__":
    main()
