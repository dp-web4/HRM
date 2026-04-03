#!/usr/bin/env python3
"""
ARC-AGI-3 Autonomous SAGE Agent

The competition-ready agent loop. SAGE solves games using:
1. arc_perception.py — grid analysis (what do I see?)
2. membot — persistent memory (what do I know about this type of game?)
3. LLM reasoning — the 0.8B model decides what to do
4. Action execution — click/move/submit based on LLM decision

No game-specific code. No brute force. SAGE reasons its way through.

Architecture:
  PERCEIVE → RECALL → REASON → ACT → OBSERVE → LEARN

The LLM receives a compact text prompt:
  "You see: [perception]. You know: [memories]. Available actions: [list].
   What should you do? Reply with action and coordinates."

Then we parse the LLM's response into an executable action.

Usage:
    python3 arc_sage_agent.py --game sb26 -v
    python3 arc_sage_agent.py --all --budget 500
"""

import sys, os, time, json, re, hashlib
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")
import numpy as np
import requests
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from arcengine import GameAction
from arc_agi import Arcade
from arc_perception import (
    get_frame, full_perception, grid_diff, grid_summary,
    find_color_regions, find_markers, background_color, color_name,
    detect_sections, find_row_patterns,
    visual_similarity, visual_memory_context,
)
from arc_spatial import SpatialTracker
from membot_cartridge import MembotCartridge
try:
    from arc_spatial import SpatialTracker
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False

# GameKnowledgeBase — persistent lived experience across runs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sage", "irp", "plugins"))
try:
    from game_knowledge_base import GameKnowledgeBase
    HAS_KB = True
except ImportError:
    HAS_KB = False

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}
MEMBOT_URL = "http://localhost:8000"
SAGE_URL = "http://localhost:8750"  # SAGE daemon for LLM reasoning


# ─── Membot integration ───

def membot_recall(query, n=5):
    """Retrieve relevant memories."""
    try:
        resp = requests.post(f"{MEMBOT_URL}/api/search",
            json={"query": query, "top_k": n}, timeout=15)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            return [r["text"] for r in results if r.get("score", 0) > 0.4]
    except Exception:
        pass
    return []


def membot_store(text):
    """Store a memory. Timeout is generous because embedding generation
    via Ollama can take 10-15s when the GPU is busy with the game model."""
    try:
        requests.post(f"{MEMBOT_URL}/api/store",
            json={"text": text}, timeout=20)
    except Exception:
        pass


# ─── LLM reasoning ───

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:0.8b")


def sage_reason(prompt, system_prompt=None, max_tokens=150):
    """Ask LLM to reason about the game state.

    Uses Ollama chat API directly with system+user message separation.
    This bypasses SAGE daemon identity prompts that confuse game reasoning.
    Returns the LLM's text response, or None if unavailable.
    """
    if system_prompt is None:
        system_prompt = "You are playing a puzzle game. You control a cursor on a grid. Reply with ONLY your next action. No explanation."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "think": False,  # Required for Qwen 3.5
                "options": {"temperature": 0.3, "num_predict": max_tokens},
            },
            timeout=60)
        if resp.status_code == 200:
            return resp.json().get("message", {}).get("content", "").strip()
    except Exception:
        pass

    return None


def parse_action(llm_response, available_actions, grid):
    """Parse LLM response into executable action(s).

    Looks for patterns like:
    - "click (X, Y)" or "click X Y"
    - "click red" or "click the red item"
    - "move up" / "press up"
    - "submit" / "press select"
    - "undo"

    Returns list of (action_int, data_dict_or_None) tuples.
    """
    if not llm_response:
        return []

    text = llm_response.lower().strip()
    actions = []

    # Reject instruction echoes — if response contains "reply:" or "or move"
    # followed by listing all actions, it's echoing the prompt, not deciding
    if ("reply:" in text and "click" in text and "move" in text) or \
       ("or move" in text and "or submit" in text):
        return []

    # Parse click with coordinates (highest priority — most specific)
    coord_patterns = [
        r'click\s*\(?(\d+)\s*,\s*(\d+)\)?',
        r'click\s+at\s+\(?(\d+)\s*,\s*(\d+)\)?',
        r'click\s+position\s+\(?(\d+)\s*,\s*(\d+)\)?',
    ]
    for pattern in coord_patterns:
        for match in re.finditer(pattern, text):
            x, y = int(match.group(1)), int(match.group(2))
            if 6 in available_actions:
                actions.append((6, {'x': x, 'y': y}))
            if len(actions) >= 2:  # Max 2 clicks per response
                break
        if actions:
            break  # Use first matching pattern only

    # Parse click with color name
    if not actions and 6 in available_actions:
        color_map = {name: idx for idx, name in enumerate(
            ["black", "blue", "red", "green", "yellow", "gray",
             "magenta", "orange", "cyan", "brown", "pink", "maroon",
             "olive", "navy", "teal", "white"])}
        for cname, cidx in color_map.items():
            if f"click {cname}" in text or f"click the {cname}" in text:
                positions = np.argwhere(grid.astype(int) == cidx)
                if len(positions) > 0:
                    # Click center of color region
                    cy = int(positions[:, 0].mean())
                    cx = int(positions[:, 1].mean())
                    actions.append((6, {'x': cx, 'y': cy}))
                    break

    # Parse "click item at bottom then slot at middle" pattern
    if not actions and 6 in available_actions:
        # Look for "then" indicating a two-step click sequence
        then_match = re.search(
            r'click\s+.*?(\d+)\s*,\s*(\d+).*?then.*?click\s+.*?(\d+)\s*,\s*(\d+)',
            text)
        if then_match:
            x1, y1 = int(then_match.group(1)), int(then_match.group(2))
            x2, y2 = int(then_match.group(3)), int(then_match.group(4))
            actions.append((6, {'x': x1, 'y': y1}))
            actions.append((6, {'x': x2, 'y': y2}))

    # If we already have click actions, return them (don't also parse directions)
    if actions:
        return actions[:2]

    # Parse directional actions — return only the FIRST one found
    direction_map = {"up": 1, "down": 2, "left": 3, "right": 4}
    for dname, daction in direction_map.items():
        if re.search(rf'\b{dname}\b', text) and daction in available_actions:
            return [(daction, None)]

    # Parse submit/select
    if any(w in text for w in ["submit", "select", "confirm", "enter"]):
        if 5 in available_actions:
            return [(5, None)]

    # Parse undo
    if any(w in text for w in ["undo", "reset", "back"]):
        if 7 in available_actions:
            return [(7, None)]

    return actions


# ─── Perception-based heuristic strategies ───

def hash_grid_state(grid):
    """Fast hash of grid state for cycle detection.

    Uses blake2b (fast, collision-resistant) on grid bytes.
    Returns hex string suitable for set membership checks.
    """
    return hashlib.blake2b(grid.tobytes(), digest_size=16).hexdigest()


def detect_cursor(grid, prev_grid=None):
    """Find the cursor/player position on the grid.

    Strategy: the cursor is the thing that moved when a direction was pressed.
    If no prev_grid, use heuristics: smallest unique-colored region.

    Returns (x, y, color_name) or None.
    """
    bg = background_color(grid)
    regions = find_color_regions(grid, min_size=1)
    non_bg = [r for r in regions if r["color"] != bg]
    if not non_bg:
        return None

    if prev_grid is not None:
        # Find what moved: compare regions between frames
        old_regions = find_color_regions(prev_grid, min_size=1)
        old_non_bg = [r for r in old_regions if r["color"] != bg]

        # Look for a region that exists in new but not old position
        for nr in non_bg:
            # Is there a same-color region in old frame at a DIFFERENT position?
            for old_r in old_non_bg:
                if (old_r["color"] == nr["color"]
                        and old_r["size"] == nr["size"]
                        and (abs(old_r["cx"] - nr["cx"]) > 0
                             or abs(old_r["cy"] - nr["cy"]) > 0)
                        and abs(old_r["cx"] - nr["cx"]) < 20
                        and abs(old_r["cy"] - nr["cy"]) < 20):
                    return (nr["cx"], nr["cy"], color_name(nr["color"]))

    # Heuristic: cursor is usually small and unique-colored
    color_counts = defaultdict(int)
    for r in non_bg:
        color_counts[r["color"]] += 1
    unique = [r for r in non_bg if color_counts[r["color"]] == 1]
    if unique:
        cursor = min(unique, key=lambda r: r["size"])
    else:
        cursor = min(non_bg, key=lambda r: r["size"])
    return (cursor["cx"], cursor["cy"], color_name(cursor["color"]))


def describe_surroundings(grid, cx, cy, radius=12, cursor_color=None):
    """Describe what's around a position in each direction.

    Skips the cursor's own color to find the NEXT thing in each direction.
    Returns a compact string like:
    "up: open(8) down: wall(3) left: teal(4) right: edge(1)"
    """
    bg = background_color(grid)
    h, w = grid.shape[:2]

    # Colors to skip (background + cursor itself)
    skip_colors = {bg}
    if cursor_color:
        # Map cursor color name back to index
        color_names = ["black", "blue", "red", "green", "yellow", "gray",
                       "magenta", "orange", "cyan", "brown", "pink", "maroon",
                       "olive", "navy", "teal", "white"]
        for i, cn in enumerate(color_names):
            if cn == cursor_color:
                skip_colors.add(i)
                break

    directions = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }
    parts = []
    for dname, (ddx, ddy) in directions.items():
        hit = None
        dist = 0
        for step in range(1, radius + 1):
            nx, ny = cx + ddx * step, cy + ddy * step
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                hit = "edge"
                dist = step
                break
            pixel = int(grid[ny, nx])
            if pixel not in skip_colors:
                hit = color_name(pixel)
                dist = step
                break
        if hit is None:
            parts.append(f"{dname}: open({radius}+)")
        else:
            parts.append(f"{dname}: {hit}({dist})")
    return " ".join(parts)


def detect_game_category(grid, available_actions):
    """Analyze grid to guess game category.

    Returns a category string and structured observations.
    """
    bg = background_color(grid)
    regions = find_color_regions(grid, min_size=6)
    sections = detect_sections(grid)
    has_click = 6 in available_actions
    has_move = any(a in available_actions for a in [1, 2, 3, 4])
    has_submit = 5 in available_actions

    obs = {
        "n_regions": len(regions),
        "n_sections": len(sections),
        "has_click": has_click,
        "has_move": has_move,
        "has_submit": has_submit,
        "section_heights": [s["y_end"] - s["y_start"] for s in sections],
    }

    # Heuristics for category detection
    if has_click and has_submit and not has_move:
        # Click + submit = arrangement/selection puzzle
        if len(sections) >= 3:
            # Multiple sections: likely sequence matching or placement
            # Check if top section has colored items (destination indicators)
            top_sec = sections[0]
            top_regions = [r for r in regions
                          if top_sec["y_start"] <= r["cy"] <= top_sec["y_end"]]
            if top_regions:
                obs["category"] = "placement_puzzle"
                obs["hint"] = "Top section has colored indicators. Items may need placement to match."
            else:
                obs["category"] = "selection_puzzle"
                obs["hint"] = "Multiple sections with click+submit. Try selecting items then submitting."
        else:
            obs["category"] = "click_puzzle"
            obs["hint"] = "Click-based puzzle. Try clicking colored items."

    elif has_click and not has_move and not has_submit:
        # Click-only games: need to classify further
        has_undo = 7 in available_actions
        if has_undo and len(sections) >= 3:
            obs["category"] = "click_arrange"
            obs["hint"] = "Click-only with undo and multiple sections. Likely arrange/place items."
        elif len(regions) > 20:
            obs["category"] = "click_grid"
            obs["hint"] = "Dense grid with click-only. May be tile rotation, painting, or toggle puzzle."
        elif len(sections) >= 3:
            obs["category"] = "click_multi_section"
            obs["hint"] = "Multiple sections, click-only. Interact with items across sections."
        else:
            obs["category"] = "click_puzzle"
            obs["hint"] = "Click-based puzzle. Try clicking colored items to discover mechanics."

    elif has_move and has_click:
        obs["category"] = "navigation_click"
        obs["hint"] = "Can both move and click. Navigate to targets, then click them."

    elif has_move and has_submit:
        obs["category"] = "navigation_submit"
        obs["hint"] = "Navigate puzzle. Move to arrange, then submit."

    elif has_move and not has_click:
        obs["category"] = "pure_navigation"
        obs["hint"] = "Movement only. Navigate character through environment."

    else:
        obs["category"] = "unknown"
        obs["hint"] = "Unknown game type. Explore available actions."

    return obs


def build_placement_prompt(grid, regions, sections):
    """Build a prompt for placement-type puzzles (sb26-like).

    Uses few-shot completion style that 0.8B models handle well.
    Returns (system_prompt, user_prompt) tuple.
    """
    bg = background_color(grid)

    # Identify top indicators, bottom items, middle slots
    if len(sections) < 2:
        return None, None

    top_sec = sections[0]
    bot_sec = sections[-1]

    # Filter top regions: exclude bg, gray (border), and very large regions (frames)
    # Real indicators are small (4-6px wide) colored blocks
    BORDER_COLORS = {bg, 5}  # yellow(bg), gray — common frame/border colors
    top_regions = sorted(
        [r for r in regions if top_sec["y_start"] <= r["cy"] <= top_sec["y_end"]
         and r["color"] not in BORDER_COLORS
         and r["size"] < 50],  # Real indicators are small, not frame-sized
        key=lambda r: r["x"])
    bot_regions = sorted(
        [r for r in regions if bot_sec["y_start"] <= r["cy"] <= bot_sec["y_end"]
         and r["color"] not in BORDER_COLORS
         and r["size"] < 50],
        key=lambda r: r["x"])

    # Find slots: color 2 markers in MIDDLE sections only (not top/bottom/lines)
    all_markers = find_markers(grid, 2, min_cluster=2)
    # Filter: exclude markers on thin horizontal lines (1-2 row sections)
    # and only keep markers in middle area between top indicators and bottom palette
    thin_line_rows = set()
    for s in sections:
        if s["y_end"] - s["y_start"] <= 1:  # Single-row or 2-row section = line
            for r in range(s["y_start"], s["y_end"] + 1):
                thin_line_rows.add(r)
    mid_y_min = top_sec["y_end"] + 1
    mid_y_max = bot_sec["y_start"] - 1
    slots = [s for s in all_markers
             if mid_y_min <= s["y"] <= mid_y_max
             and s["y"] not in thin_line_rows]

    if not top_regions or not bot_regions:
        return None, None

    # Deduplicate top colors (borders create duplicate regions at same position)
    seen_positions = set()
    deduped_top = []
    for r in top_regions:
        # Group by approximate x position (within 3px)
        key = r["cx"] // 4
        if key not in seen_positions:
            seen_positions.add(key)
            deduped_top.append(r)

    bot_items = [(r["color_name"], r["cx"], r["cy"]) for r in bot_regions]
    slot_positions = [(s["x"], s["y"]) for s in slots]
    top_colors = [r["color_name"] for r in deduped_top]

    system = "You solve puzzles. Reply with ONLY click actions. No explanation."

    parts = [
        f"Puzzle: match the top order by placing bottom items into slots.",
        f"\nTOP order: {', '.join(top_colors)}",
        f"BOTTOM items: {' '.join(f'{name}({x},{y})' for name, x, y in bot_items)}",
    ]
    if slot_positions:
        parts.append(f"SLOTS: {' '.join(f's{i+1}({x},{y})' for i, (x, y) in enumerate(slot_positions))}")

    # Build the plan
    plan_parts = []
    for i, target_color in enumerate(top_colors):
        # Find matching bottom item
        for name, bx, by in bot_items:
            if name == target_color:
                if i < len(slot_positions):
                    sx, sy = slot_positions[i]
                    plan_parts.append(f"{target_color}→s{i+1}")
                break
    if plan_parts:
        parts.append(f"\nPlan: {', '.join(plan_parts)}")

    parts.append("\nActions:")

    return system, "\n".join(parts)


def compute_placement_actions(grid):
    """Pure perception-based placement: match top indicator colors to bottom items.

    Returns list of (action_int, data) tuples for click-place-submit, or empty list.
    This does NOT need the LLM — it's pure spatial/color reasoning in code.
    """
    bg = background_color(grid)
    regions = find_color_regions(grid, min_size=6)
    sections = detect_sections(grid)

    if len(sections) < 2:
        return []

    top_sec = sections[0]
    bot_sec = sections[-1]

    BORDER_COLORS = {bg, 5}  # yellow(bg), gray — frame/border colors
    top_regions = sorted(
        [r for r in regions if top_sec["y_start"] <= r["cy"] <= top_sec["y_end"]
         and r["color"] not in BORDER_COLORS and r["size"] < 50],
        key=lambda r: r["x"])
    bot_regions = sorted(
        [r for r in regions if bot_sec["y_start"] <= r["cy"] <= bot_sec["y_end"]
         and r["color"] not in BORDER_COLORS and r["size"] < 50],
        key=lambda r: r["x"])

    # Deduplicate top by position
    seen_pos = set()
    deduped_top = []
    for r in top_regions:
        key = r["cx"] // 4
        if key not in seen_pos:
            seen_pos.add(key)
            deduped_top.append(r)

    # Find middle slots (color 2 markers, not on thin lines)
    all_markers = find_markers(grid, 2, min_cluster=2)
    thin_line_rows = set()
    for s in sections:
        if s["y_end"] - s["y_start"] <= 1:
            for r in range(s["y_start"], s["y_end"] + 1):
                thin_line_rows.add(r)
    mid_y_min = top_sec["y_end"] + 1
    mid_y_max = bot_sec["y_start"] - 1
    slots = [s for s in all_markers
             if mid_y_min <= s["y"] <= mid_y_max
             and s["y"] not in thin_line_rows]

    if not deduped_top or not bot_regions or not slots:
        return []

    # Build color→position lookup for bottom items
    bot_by_color = {}
    for r in bot_regions:
        if r["color_name"] not in bot_by_color:
            bot_by_color[r["color_name"]] = (r["cx"], r["cy"])

    # Match: for each top indicator color, find matching bottom item → place in slot
    actions = []
    for i, top_r in enumerate(deduped_top):
        target_color = top_r["color_name"]
        if target_color in bot_by_color and i < len(slots):
            px, py = bot_by_color[target_color]
            sx, sy = slots[i]["x"], slots[i]["y"]
            actions.append((6, {'x': px, 'y': py}))  # Click palette item
            actions.append((6, {'x': sx, 'y': sy}))  # Click slot

    # Submit after all placements
    if actions:
        actions.append((5, None))

    return actions


def build_strategy_prompt(grid, available_actions, memories, level_info,
                          prev_action_result=None, cartridge=None,
                          initial_grid=None, exploration_summary=None,
                          cycle_info=None, spatial_tracker=None,
                          goal_similarity=None, cursor_pos=None,
                          prev_grid=None, last_action_desc=None,
                          action_history=None, kb_context=None):
    """Build a situational briefing for the LLM.

    The model is a player looking at a game screen. It controls a cursor.
    The prompt tells it: what's on screen, where the cursor is, what just
    happened, and what it's learned so far. Then asks: what do you do?

    Returns (prompt, category) tuple.
    """
    category = detect_game_category(grid, available_actions)
    cat_name = category.get("category", "")
    bg = background_color(grid)
    h, w = grid.shape[:2]

    action_names = {1: "up", 2: "down", 3: "left", 4: "right",
                    5: "submit", 6: "click(x,y)", 7: "undo"}
    avail_str = ", ".join(action_names.get(a, f"action{a}") for a in sorted(available_actions))

    prompt_parts = []

    # ── Situation: what are you looking at? ──
    prompt_parts.append(
        f"Puzzle game. Level {level_info.get('current', '?')}/{level_info.get('total', '?')}. "
        f"Grid {h}x{w}."
    )

    # Cursor position — the model's primary tool
    if cursor_pos:
        cx, cy, ccolor = cursor_pos
        prompt_parts.append(f"Cursor: {ccolor} at ({cx},{cy})")
        # What's nearby in each direction (skip cursor's own color)
        surroundings = describe_surroundings(grid, cx, cy, radius=12, cursor_color=ccolor)
        prompt_parts.append(f"Around cursor: {surroundings}")

    # ── What just happened? (causal feedback — multi-turn) ──
    if action_history and len(action_history) > 0:
        history_str = " | ".join(action_history)
        prompt_parts.append(f"Recent: {history_str}")
    elif last_action_desc:
        prompt_parts.append(f"Last action: {last_action_desc}")
    elif prev_action_result:
        prompt_parts.append(f"Last result: {prev_action_result[:100]}")

    if cycle_info:
        prompt_parts.append(f"WARNING: {cycle_info}. Try something DIFFERENT.")

    # ── What you've learned (from exploration) ──
    if exploration_summary:
        prompt_parts.append(exploration_summary)

    # ── What you know from prior sessions (KB) ──
    if kb_context:
        # Curated KB — only the most actionable lines, not the full dump
        # Keep it under 200 chars to not overwhelm 0.8B context
        prompt_parts.append(kb_context[:200])

    # ── What you see on screen ──
    regions = find_color_regions(grid, min_size=4)
    non_bg_regions = [r for r in regions if r["color"] != bg]

    if non_bg_regions:
        # Show objects, prioritizing effective/clickable ones
        shown = non_bg_regions
        if exploration_summary and "Clickable colors" in exploration_summary:
            eff_names = set()
            eff_line = exploration_summary.split("Clickable colors")[1].split("\n")[0]
            if ":" in eff_line:
                eff_line = eff_line.split(":")[-1]
            for part in eff_line.split(","):
                name = part.strip().split("(")[0].strip()
                if name and name[0].isalpha():
                    eff_names.add(name)
            if eff_names:
                effective = [r for r in non_bg_regions if r["color_name"] in eff_names]
                if effective:
                    shown = effective

        # Compact object list — nearest first if cursor known
        if cursor_pos:
            cx, cy, _ = cursor_pos
            shown = sorted(shown, key=lambda r: abs(r["cx"] - cx) + abs(r["cy"] - cy))
        targets = [f"{r['color_name']}({r['cx']},{r['cy']})" for r in shown[:6]]
        label = "Clickable" if exploration_summary and "Clickable" in exploration_summary else "Objects"
        prompt_parts.append(f"{label}: {', '.join(targets)}")

    # ── Goal progress ──
    if goal_similarity is not None and goal_similarity < 0.95:
        prompt_parts.append(f"Progress: {100 - goal_similarity*100:.0f}% changed from start")

    # ── What to do ──
    prompt_parts.append(f"Actions: {avail_str}")
    prompt_parts.append("What do you do? Reply with ONE action.")

    return "\n".join(prompt_parts), category


# ─── Fallback heuristic (no LLM needed) ───

def heuristic_action(grid, available_actions, category, step_count,
                     effective_actions=None, color_effectiveness=None):
    """Fallback action when LLM is unavailable or response unparseable.

    Uses perception-based heuristics informed by exploration results.
    """
    bg = background_color(grid)

    if category.get("category") == "placement_puzzle" and 6 in available_actions:
        # For placement puzzles: try compute_placement_actions first
        placement = compute_placement_actions(grid)
        if placement:
            return placement

        # Fallback: identify items and slots manually
        sections = detect_sections(grid)
        if len(sections) >= 3:
            bot = sections[-1]
            bot_regions = find_color_regions(grid, min_size=4)
            bot_items = [r for r in bot_regions
                        if bot["y_start"] <= r["cy"] <= bot["y_end"]
                        and r["color"] != bg]
            mid_markers = find_markers(grid, 2, min_cluster=2)

            if bot_items and mid_markers:
                item = bot_items[step_count % len(bot_items)]
                slot = mid_markers[step_count % len(mid_markers)]
                return [
                    (6, {'x': item["cx"], 'y': item["cy"]}),
                    (6, {'x': slot["x"], 'y': slot["y"]}),
                ]

    if 6 in available_actions:
        # Prefer clicking effective colors (from exploration)
        if color_effectiveness:
            effective_colors = [c for c, s in color_effectiveness.items()
                                if s["changes"] > 0]
            if effective_colors:
                target_color_name = effective_colors[step_count % len(effective_colors)]
                # Find that color's position
                regions = find_color_regions(grid, min_size=4)
                matching = [r for r in regions if r["color_name"] == target_color_name]
                if matching:
                    r = matching[step_count % len(matching)]
                    return [(6, {'x': r["cx"], 'y': r["cy"]})]

        # Default: click a non-background pixel
        non_bg = np.argwhere(grid.astype(int) != bg)
        if len(non_bg) > 0:
            idx = step_count % len(non_bg)
            r, c = int(non_bg[idx, 0]), int(non_bg[idx, 1])
            return [(6, {'x': c, 'y': r})]

    # Movement: prefer directions that caused changes during exploration
    moves = [a for a in available_actions if a in [1, 2, 3, 4]]
    if moves:
        if effective_actions:
            # Prefer directions that were effective
            effective_dirs = [ea["action"] for ea in effective_actions
                              if ea["action"] in moves]
            if effective_dirs:
                return [(effective_dirs[step_count % len(effective_dirs)], None)]
        # Cycle through directions — but with a bias toward forward exploration
        # Pattern: go same direction 3x, then try next (covers more ground)
        dir_idx = (step_count // 3) % len(moves)
        return [(moves[dir_idx], None)]

    return []


def navigate_toward_interest(grid, available_actions, visited_positions=None,
                              prev_grids=None):
    """For navigation games: choose the most promising direction.

    Strategy: try to move in a direction that will produce a NEW grid state.
    If we have previous grids from each direction, prefer directions that
    haven't been tried recently. Otherwise, prefer directions away from
    where we've already been.

    Returns (action_int, None) or None if no useful direction found.
    """
    import random
    moves = [a for a in [1, 2, 3, 4] if a in available_actions]
    if not moves:
        return None

    # If we have direction history, prefer unexplored directions
    if visited_positions:
        # Score each direction by how many unvisited cells lie that way
        bg = background_color(grid)
        h, w = grid.shape[:2]

        # Try to find what might be a "player" marker — look for a unique
        # small region that appears in only one place
        regions = find_color_regions(grid, min_size=1)
        non_bg_regions = [r for r in regions if r["color"] != bg]

        if non_bg_regions:
            # Player heuristic: smallest or rarest-color region
            color_counts = defaultdict(int)
            for r in non_bg_regions:
                color_counts[r["color"]] += 1
            # Regions with unique colors (appear only once) are likely the player
            unique_regions = [r for r in non_bg_regions if color_counts[r["color"]] == 1]
            if unique_regions:
                player = min(unique_regions, key=lambda r: r["size"])
            else:
                player = min(non_bg_regions, key=lambda r: r["size"])
            px, py = player["cx"], player["cy"]

            # Score directions by unvisited area
            dir_scores = {}
            for action in moves:
                name = {1: "up", 2: "down", 3: "left", 4: "right"}[action]
                # Look at the coarse grid cells in each direction
                unvisited_count = 0
                total_check = 0
                for dy in range(-8, 9, 2):
                    for dx in range(-8, 9, 2):
                        if name == "up" and dy >= 0:
                            continue
                        if name == "down" and dy <= 0:
                            continue
                        if name == "left" and dx >= 0:
                            continue
                        if name == "right" and dx <= 0:
                            continue
                        cy, cx = (py + dy) // 4, (px + dx) // 4
                        if 0 <= cy < h // 4 and 0 <= cx < w // 4:
                            total_check += 1
                            if (cy, cx) not in visited_positions:
                                unvisited_count += 1
                dir_scores[action] = unvisited_count

            if dir_scores:
                best_score = max(dir_scores.values())
                if best_score > 0:
                    best_dirs = [a for a, s in dir_scores.items() if s == best_score]
                    return (random.choice(best_dirs), None)

    # Fallback: random direction, prefer ones we haven't done recently
    return (random.choice(moves), None)


# ─── Main agent loop ───

class SageAgent:
    """Autonomous ARC-AGI-3 game-playing agent.

    Uses perception + memory + LLM reasoning to solve games.
    No game-specific code — learns and reasons its way through.
    """

    def __init__(self, use_llm=True, verbose=False, explore_budget=30):
        self.use_llm = use_llm
        self.verbose = verbose
        self.llm_available = None  # Will check on first use
        self.action_history = []
        self.level_observations = []
        self.explore_budget = explore_budget  # Actions for fast exploration phase

        # Action-outcome tracking (built during exploration, consumed by LLM)
        self.outcome_map = {}  # (action, color_at_pos) → {changes, level_ups, no_effect}
        self.color_effectiveness = {}  # color → {tries, changes}
        self.effective_actions = []  # Actions that caused grid changes
        self.level_up_sequences = []  # Action sequences that preceded level-ups

        # Spatial reasoning — initialized per-game in play_game()
        self.spatial_tracker = None

        # Anti-repetition: track recent actions and ban repeated ones
        self.banned_actions = {}  # action_key → steps_remaining
        self.recent_action_keys = deque(maxlen=10)  # Last 10 action keys for pattern detection
        self.total_cycles = 0  # Never resets — escalates over game lifetime

        # Click data tracking (for KB solution recording)
        self.click_data_history = []  # [(action_int, data_dict_or_None), ...]

    @staticmethod
    def _action_key(actions):
        """Create a hashable key from an action list for repetition tracking."""
        parts = []
        for a, d in actions:
            if d and isinstance(d, dict):
                parts.append(f"{a}({d.get('x','')},{d.get('y','')})")
            else:
                parts.append(str(a))
        return "|".join(parts)

    def check_llm(self):
        """Check if Ollama is available for reasoning."""
        if self.llm_available is not None:
            return self.llm_available
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            self.llm_available = resp.status_code == 200
        except Exception:
            self.llm_available = False
        if self.verbose:
            print(f"  LLM available: {self.llm_available}")
        return self.llm_available

    def fast_explore(self, env, fd, available_actions, budget):
        """Phase 1: Systematic rapid probing. No LLM. Pure observation.

        Probes the game systematically to build an action-outcome map:
        1. Try each non-click action once (submit, undo, directions)
        2. Click distinct color regions (one click per unique color)
        3. If a click causes change, try more clicks on that color
        4. Track what changes the grid vs what doesn't

        Returns (fd, steps_used) — updated frame data and action count.
        """
        grid = get_frame(fd)
        bg = background_color(grid)
        step = 0
        prev_grid = grid.copy()
        levels_at_start = fd.levels_completed

        if self.verbose:
            print(f"  ── Fast Explore Phase ({budget} actions) ──")

        # Step 1: Try non-click actions to see what they do
        non_click = [a for a in available_actions if a != 6]
        for action in non_click:
            if step >= budget:
                break
            try:
                fd = env.step(INT_TO_GAME_ACTION[action])
                step += 1
                self.action_history.append(action)
                new_grid = get_frame(fd)
                diff = grid_diff(prev_grid, new_grid)
                changed = diff != "No change"

                action_name = {1: "up", 2: "down", 3: "left", 4: "right",
                               5: "submit", 7: "undo"}.get(action, f"a{action}")
                # Count actual pixels changed for magnitude comparison
                n_pixels = int(np.sum(prev_grid != new_grid)) if changed else 0
                self.outcome_map[f"action_{action_name}"] = {
                    "changed": changed, "diff": diff[:100] if changed else None,
                    "n_pixels": n_pixels,
                    "level_up": fd.levels_completed > levels_at_start,
                }

                if changed:
                    self.effective_actions.append({"action": action, "step": step})
                    # Detect cursor from movement (first direction that changes grid)
                    if action in (1, 2, 3, 4) and not hasattr(self, '_cursor_color'):
                        cursor = detect_cursor(new_grid, prev_grid)
                        if cursor:
                            self._cursor_color = cursor[2]  # Remember cursor color
                            self._cursor_pos = (cursor[0], cursor[1])
                            if self.verbose:
                                print(f"    Cursor detected: {cursor[2]} at ({cursor[0]},{cursor[1]})")
                    if self.verbose:
                        print(f"    [{step}] {action_name} → CHANGED: {diff[:60]}")
                elif self.verbose:
                    print(f"    [{step}] {action_name} → no change")

                if fd.levels_completed > levels_at_start:
                    self.level_up_sequences.append(list(self.action_history[-5:]))
                    if self.verbose:
                        print(f"    ★ Level up from {action_name}!")

                prev_grid = new_grid.copy()
                # Undo if submit changed things (don't burn a level accidentally)
                if action == 5 and changed and 7 in available_actions:
                    fd = env.step(INT_TO_GAME_ACTION[7])
                    step += 1
                    self.action_history.append(7)
                    prev_grid = get_frame(fd).copy()

            except Exception as e:
                if self.verbose:
                    print(f"    [{step}] action {action} error: {e}")

        # Step 2: Click distinct color regions
        if 6 in available_actions and step < budget:
            grid = get_frame(fd)
            regions = find_color_regions(grid, min_size=4)

            # Get one representative position per unique color
            color_targets = {}
            for r in regions:
                c = r["color"]
                if c != bg and c not in color_targets:
                    color_targets[c] = (r["cx"], r["cy"])

            if self.verbose:
                print(f"    Probing {len(color_targets)} distinct colors...")

            for color, (cx, cy) in color_targets.items():
                if step >= budget:
                    break
                prev_grid = get_frame(fd).copy()
                prev_levels = fd.levels_completed

                try:
                    fd = env.step(INT_TO_GAME_ACTION[6], data={'x': cx, 'y': cy})
                    step += 1
                    self.action_history.append(6)
                    new_grid = get_frame(fd)
                    diff = grid_diff(prev_grid, new_grid)
                    changed = diff != "No change"
                    cname = color_name(color)

                    # Track color effectiveness
                    if cname not in self.color_effectiveness:
                        self.color_effectiveness[cname] = {"tries": 0, "changes": 0}
                    self.color_effectiveness[cname]["tries"] += 1
                    if changed:
                        self.color_effectiveness[cname]["changes"] += 1
                        self.effective_actions.append({
                            "action": 6, "color": cname, "pos": (cx, cy), "step": step
                        })

                    if self.verbose:
                        status = "CHANGED" if changed else "no change"
                        print(f"    [{step}] click {cname}@({cx},{cy}) → {status}")

                    if fd.levels_completed > prev_levels:
                        self.level_up_sequences.append(list(self.action_history[-5:]))
                        if self.verbose:
                            print(f"    ★ Level up from clicking {cname}!")

                    # If clicking caused change, probe MORE of this color
                    if changed and step < budget:
                        same_color = [r for r in regions
                                      if r["color"] == color and (r["cx"], r["cy"]) != (cx, cy)]
                        for extra in same_color[:3]:  # Up to 3 more clicks
                            if step >= budget:
                                break
                            prev_grid = get_frame(fd).copy()
                            fd = env.step(INT_TO_GAME_ACTION[6],
                                          data={'x': extra["cx"], 'y': extra["cy"]})
                            step += 1
                            self.action_history.append(6)
                            self.color_effectiveness[cname]["tries"] += 1
                            nd = grid_diff(prev_grid, get_frame(fd))
                            if nd != "No change":
                                self.color_effectiveness[cname]["changes"] += 1
                                self.effective_actions.append({
                                    "action": 6, "color": cname,
                                    "pos": (extra["cx"], extra["cy"]), "step": step
                                })

                    # Undo if we want to keep exploring from clean state
                    if changed and 7 in available_actions and step < budget:
                        fd = env.step(INT_TO_GAME_ACTION[7])
                        step += 1
                        self.action_history.append(7)

                except Exception:
                    pass

        # Step 3: Try clicking in different sections (spatial probing)
        if 6 in available_actions and step < budget:
            grid = get_frame(fd)
            sections = detect_sections(grid)
            for sec in sections[:5]:
                if step >= budget:
                    break
                # Click center of section
                sy = (sec["y_start"] + sec["y_end"]) // 2
                sx = grid.shape[1] // 2
                prev_grid = get_frame(fd).copy()
                try:
                    fd = env.step(INT_TO_GAME_ACTION[6], data={'x': sx, 'y': sy})
                    step += 1
                    self.action_history.append(6)
                    diff = grid_diff(prev_grid, get_frame(fd))
                    if diff != "No change":
                        self.effective_actions.append({
                            "action": 6, "section": f"rows_{sec['y_start']}-{sec['y_end']}",
                            "pos": (sx, sy), "step": step
                        })
                        if self.verbose:
                            print(f"    [{step}] section click ({sx},{sy}) → CHANGED")
                except Exception:
                    pass

        if self.verbose:
            n_eff = len(self.effective_actions)
            n_colors = sum(1 for v in self.color_effectiveness.values() if v["changes"] > 0)
            print(f"  ── Explore done: {step} actions, {n_eff} effective, {n_colors} reactive colors ──")

        return fd, step

    def exploration_summary(self):
        """Compile exploration results into text for LLM consumption."""
        lines = []

        # Color effectiveness
        if self.color_effectiveness:
            effective = []
            inert = []
            for cname, stats in sorted(self.color_effectiveness.items(),
                                        key=lambda x: -x[1]["changes"]):
                rate = stats["changes"] / max(stats["tries"], 1)
                if stats["changes"] > 0:
                    effective.append(f"{cname}({stats['changes']}/{stats['tries']})")
                else:
                    inert.append(cname)
            if effective:
                lines.append(f"Clickable colors (caused change): {', '.join(effective)}")
            if inert:
                lines.append(f"Inert colors (no effect): {', '.join(inert)}")

        # Non-click action results — include pixel magnitudes for directions
        action_results = {k: v for k, v in self.outcome_map.items()
                          if k.startswith("action_")}
        if action_results:
            # Separate directions from other actions
            directions = {}
            other_active = []
            passive = []
            for k, v in action_results.items():
                name = k.replace("action_", "")
                if name in ("up", "down", "left", "right"):
                    if v["changed"]:
                        directions[name] = v.get("n_pixels", 0)
                    else:
                        passive.append(name)
                elif v["changed"]:
                    other_active.append(name)
                else:
                    passive.append(name)

            if directions:
                # Show directions with relative magnitude
                dir_strs = [f"{d}({n}px)" for d, n in
                            sorted(directions.items(), key=lambda x: -x[1])]
                lines.append(f"Movement: {', '.join(dir_strs)}")
            if other_active:
                lines.append(f"Active: {', '.join(other_active)}")
            if passive:
                lines.append(f"Inactive: {', '.join(passive)}")

        # Level-up patterns
        if self.level_up_sequences:
            lines.append(f"Level-up sequences found: {self.level_up_sequences}")

        return "\n".join(lines) if lines else "Exploration: no reactive elements found."

    def play_game(self, arcade, game_id, budget=500):
        """Play one game, return results.

        Args:
            arcade: Arcade instance
            game_id: Game to play
            budget: Max actions to take

        Returns:
            dict with levels_completed, win_levels, actions_used, learnings
        """
        prefix = game_id.split("-")[0]
        env = arcade.make(game_id)
        fd = env.reset()
        grid = get_frame(fd)
        available = [a.value if hasattr(a, 'value') else int(a)
                     for a in (fd.available_actions or [])]

        if self.verbose:
            print(f"\n  Game: {prefix}")
            print(f"  Actions: {available}, Win: {fd.win_levels} levels")

        # VISUAL MEMORY: Create cartridge and store initial state
        cartridge = MembotCartridge(f"sage_visual_{prefix}")
        cartridge.read()
        initial_grid = grid.copy()
        cartridge.store_frame_snapshot(
            "initial",
            initial_grid,
            metadata={"description": "Initial game state", "level": 0}
        )
        if self.verbose:
            print(f"  Visual memory: initialized for {prefix}")

        # GOAL DETECTION: Track similarity to initial state
        goal_similarity = 1.0  # Start at initial state
        goal_history = [1.0]  # Track similarity over time

        # SPATIAL TRACKING: Initialize object tracker
        if self.spatial_tracker:
            self.spatial_tracker = SpatialTracker()  # Fresh tracker per game
            self.spatial_tracker.update(grid)
            if self.verbose:
                n_obj = len(self.spatial_tracker.objects)
                print(f"  Spatial: {n_obj} objects detected")

        # RECALL: what do we know about this game type?
        memories = membot_recall(f"ARC-AGI-3 {prefix} game strategy")
        category_memories = membot_recall(f"ARC-AGI-3 placement puzzle sequence matching")
        all_memories = memories + category_memories

        if self.verbose and all_memories:
            print(f"  Recalled {len(all_memories)} memories")

        # Detect game category
        category = detect_game_category(grid, available)
        if self.verbose:
            print(f"  Category: {category.get('category', '?')} — {category.get('hint', '')}")

        # KNOWLEDGE BASE: Load persistent game knowledge (fleet-contributed)
        kb = None
        if HAS_KB:
            kb = GameKnowledgeBase(prefix)
            kb.load()
            kb.session_count += 1
            kb.decay_stale_solutions()
            if self.verbose:
                s = kb.stats
                print(f"  KB: session {s['session_count']}, {s['n_objects']} objects, "
                      f"{s['n_solutions']} solutions, best L{s['best_level']}")

        # SPATIAL TRACKING: Initialize object tracker
        if HAS_SPATIAL:
            self.spatial_tracker = SpatialTracker(min_region_size=4)
            spatial_diff = self.spatial_tracker.update(grid)
            if self.verbose:
                print(f"  Spatial: {spatial_diff['n_objects']} objects detected")

        # ═══ PHASE 1: Fast Exploration ═══
        # Systematic probing — no LLM, pure observation
        # Builds action-outcome map for the LLM to consume later
        explore_actions = min(self.explore_budget, budget // 3)  # Max 1/3 of budget
        if explore_actions > 0:
            fd, steps_used = self.fast_explore(env, fd, available, explore_actions)
            grid = get_frame(fd)
        else:
            steps_used = 0

        # Feed exploration results to KB
        if kb:
            for cname, stats in self.color_effectiveness.items():
                # Record each probed color as an effect
                for ea in self.effective_actions:
                    if ea.get("color") == cname and "pos" in ea:
                        px, py = ea["pos"]
                        kb.record_click_effect(
                            r=py, c=px, color=cname,
                            cells_changed=stats["changes"],
                            level_up=False, level=0)

        # Check if exploration already solved levels
        best_levels = fd.levels_completed
        if best_levels > 0 and self.verbose:
            print(f"  Exploration solved {best_levels} level(s)!")

        # ═══ PHASE 1.5: KB Solution Replay ═══
        # If KB has a known solution for the current level, try it before LLM
        if kb and best_levels < fd.win_levels:
            sol = kb.get_level_solution(best_levels + 1)
            if sol and sol.confidence >= 0.5:
                if self.verbose:
                    print(f"  KB: replaying solution for L{best_levels+1} (confidence {sol.confidence:.0%})")
                prev_grid_sol = get_frame(fd).copy()
                for s in sol.sequence:
                    if steps_used >= budget:
                        break
                    r, c = s.get("r", 0), s.get("c", 0)
                    repeats = s.get("repeats", 1)
                    for _ in range(repeats):
                        if steps_used >= budget:
                            break
                        try:
                            fd = env.step(INT_TO_GAME_ACTION[6], data={'x': c, 'y': r})
                            steps_used += 1
                            self.action_history.append(6)
                        except Exception:
                            break
                if fd.levels_completed > best_levels:
                    best_levels = fd.levels_completed
                    sol.attempts += 1
                    sol.successes += 1
                    sol.last_verified_session = kb.session_count
                    if self.verbose:
                        print(f"  ★ KB solution worked! L{best_levels}")
                else:
                    sol.attempts += 1
                    sol.confidence = max(0.1, sol.confidence - 0.1)
                    if self.verbose:
                        print(f"  KB solution failed (confidence → {sol.confidence:.0%})")
                grid = get_frame(fd)

        # ═══ PHASE 2: Perception-driven placement (if applicable) ═══
        # For placement puzzles, try pure perception before LLM
        if (category.get("category") in ("placement_puzzle", "click_arrange")
                and best_levels < fd.win_levels):
            placement_actions = compute_placement_actions(grid)
            if placement_actions:
                if self.verbose:
                    print(f"  Trying perception-driven placement ({len(placement_actions)} actions)...")
                prev_grid = grid.copy()
                for action_int, data in placement_actions:
                    if steps_used >= budget:
                        break
                    try:
                        if data:
                            fd = env.step(INT_TO_GAME_ACTION[action_int], data=data)
                        else:
                            fd = env.step(INT_TO_GAME_ACTION[action_int])
                        steps_used += 1
                        self.action_history.append(action_int)
                    except Exception:
                        break
                new_grid = get_frame(fd)
                if fd.levels_completed > best_levels:
                    best_levels = fd.levels_completed
                    if self.verbose:
                        print(f"  ★ Placement solved level {best_levels}!")
                    # KB: Record placement solution
                    if kb:
                        seq = [{"r": d['y'], "c": d['x'], "repeats": 1}
                               for a, d in placement_actions if a == 6 and d]
                        kb.record_level_solution(best_levels, seq,
                                                 preconditions="placement_puzzle",
                                                 confidence=0.9,
                                                 notes="perception-driven placement")
                grid = new_grid

        # ═══ PHASE 3: LLM-guided play ═══
        step = steps_used
        prev_result = None
        prev_grid = grid.copy()
        llm_cooldown = 0
        llm_interval = 3  # Call LLM every N steps — it needs to reason
        steps_since_level_up = 0  # Track steps without level progress
        last_level = best_levels
        recent_llm_actions = []  # Track LLM action repetition
        last_action_desc = None  # Human-readable description of what last action did
        action_history_descs = deque(maxlen=3)  # Last 3 action descriptions for multi-turn

        # CURSOR TRACKING: use exploration-detected cursor, or heuristic
        if hasattr(self, '_cursor_color') and hasattr(self, '_cursor_pos'):
            # We know the cursor color from exploration — find it now
            cursor_pos = (self._cursor_pos[0], self._cursor_pos[1], self._cursor_color)
        else:
            cursor_pos = detect_cursor(grid, prev_grid)
        if cursor_pos and self.verbose:
            cx, cy, cc = cursor_pos
            print(f"  Cursor: {cc} at ({cx},{cy})")

        # STATE CYCLE DETECTION: Track seen states to break loops
        seen_states = set()  # All states visited (for exact match detection)
        recent_states = deque(maxlen=10)  # Last 10 states (for loop pattern detection)
        cycle_detected = False
        cycle_info = None
        consecutive_cycles = 0  # Count how many cycles in a row (for loop-breaking)

        # Navigation tracking
        is_nav_game = (category.get("category", "").startswith("navigation")
                       or category.get("category") == "pure_navigation")
        visited_nav_positions = set()

        while step < budget and fd.state.name not in ("WON", "GAME_OVER"):
            # PERCEIVE
            grid = get_frame(fd)

            # Decay banned actions
            expired = [k for k, v in self.banned_actions.items() if v <= 0]
            for k in expired:
                del self.banned_actions[k]
            for k in self.banned_actions:
                self.banned_actions[k] -= 1

            # Track level changes
            if fd.levels_completed > best_levels:
                best_levels = fd.levels_completed
                steps_since_level_up = 0
                diff_desc = grid_diff(prev_grid, grid)
                self.level_observations.append({
                    "level": best_levels,
                    "actions_before": list(self.action_history[-10:]),
                    "grid_change": diff_desc,
                })
                if self.verbose:
                    print(f"  ★ Level {best_levels}/{fd.win_levels}!")

                # Store level-up snapshot in visual memory
                cartridge.store_frame_snapshot(
                    f"level_{best_levels}",
                    grid.copy(),
                    metadata={
                        "description": f"Solved level {best_levels}",
                        "level": best_levels,
                        "actions": list(self.action_history[-10:]),
                    }
                )

                # KB: Record level solution from recent click data
                if kb:
                    recent_clicks = [(a, d) for a, d in self.click_data_history[-20:]
                                     if a == 6 and d]
                    if recent_clicks:
                        seq = [{"r": d['y'], "c": d['x'], "repeats": 1}
                               for a, d in recent_clicks]
                        kb.record_level_solution(best_levels, seq,
                                                 confidence=0.7,
                                                 notes=f"auto-captured session {kb.session_count}")
                    kb.best_level = max(kb.best_level, best_levels)

                # Store level-up knowledge
                recent_actions = self.action_history[-15:]
                membot_store(
                    f"ARC-AGI-3 {prefix}: level {best_levels} reached. "
                    f"Category: {category.get('category')}. "
                    f"Recent actions: {recent_actions}. "
                    f"Grid change: {diff_desc[:200]}"
                )

                # Level-up resets cycle detection (new game state space)
                seen_states.clear()
                recent_states.clear()
                cycle_detected = False
                cycle_info = None
                consecutive_cycles = 0
                self.total_cycles = 0
                self.banned_actions.clear()
                visited_nav_positions.clear()

                # Re-try perception-driven placement on new level
                if category.get("category") in ("placement_puzzle", "click_arrange"):
                    new_grid = get_frame(fd)
                    placement_actions = compute_placement_actions(new_grid)
                    if placement_actions:
                        if self.verbose:
                            print(f"  Re-running placement for level {best_levels+1}...")
                        for pa, pd in placement_actions:
                            if step >= budget:
                                break
                            try:
                                if pd:
                                    fd = env.step(INT_TO_GAME_ACTION[pa], data=pd)
                                else:
                                    fd = env.step(INT_TO_GAME_ACTION[pa])
                                step += 1
                                self.action_history.append(pa)
                            except Exception:
                                break
                        continue  # Re-check level after placement
            else:
                steps_since_level_up += 1

            # CYCLE DETECTION: Check if we've seen this state before
            state_hash = hash_grid_state(grid)

            if state_hash in seen_states:
                cycle_detected = True
                consecutive_cycles += 1
                self.total_cycles += 1  # Never resets — escalates
                if state_hash in recent_states:
                    loop_length = len(recent_states) - list(recent_states).index(state_hash)
                    cycle_info = f"Loop detected (length {loop_length}, #{consecutive_cycles})"
                else:
                    cycle_info = f"Visited state (#{consecutive_cycles})"
                if self.verbose and consecutive_cycles % 5 == 0:
                    print(f"  [{step}] {cycle_info} (total: {self.total_cycles})")
            else:
                consecutive_cycles = 0
                cycle_detected = False
                cycle_info = None

            seen_states.add(state_hash)
            recent_states.append(state_hash)

            # Navigation: track coarse position
            if is_nav_game:
                bg = background_color(grid)
                nav_regions = find_color_regions(grid, min_size=2)
                smallest_r = min(nav_regions, key=lambda r: r["size"]) if nav_regions else None
                if smallest_r:
                    visited_nav_positions.add((smallest_r["cy"] // 4, smallest_r["cx"] // 4))

            # REASON (LLM or heuristic)
            actions_to_take = []

            # ESCALATING LOOP-BREAKING: severity increases with total_cycles
            if consecutive_cycles >= 3 or self.total_cycles >= 10:
                import random

                # Escalation levels based on total game-lifetime cycles
                if self.total_cycles >= 30:
                    # Severe: try submit or undo to escape entirely
                    if 5 in available and random.random() < 0.3:
                        actions_to_take = [(5, None)]
                        if self.verbose:
                            print(f"  [{step}] SEVERE loop ({self.total_cycles}) — submit escape")
                    elif 7 in available:
                        actions_to_take = [(7, None)]
                        if self.verbose:
                            print(f"  [{step}] SEVERE loop ({self.total_cycles}) — undo escape")

                if not actions_to_take and self.total_cycles >= 15:
                    # High: navigation-aware — move toward unexplored areas
                    nav_action = navigate_toward_interest(grid, available, visited_nav_positions)
                    if nav_action:
                        actions_to_take = [nav_action]
                        if self.verbose:
                            d = {1: "up", 2: "down", 3: "left", 4: "right"}.get(nav_action[0], "?")
                            print(f"  [{step}] Loop break — navigate {d}")

                if not actions_to_take:
                    # Normal: spatial targets or random
                    if self.spatial_tracker and 6 in available:
                        targets = self.spatial_tracker.suggest_click_targets(grid, n=3)
                        untested = [(x, y, r) for x, y, r in targets if "untested" in r]
                        if untested:
                            x, y, reason = random.choice(untested)
                            actions_to_take = [(6, {'x': x, 'y': y})]

                    if not actions_to_take:
                        # Pick a RANDOM action, excluding recently banned ones
                        unbanned = [a for a in available
                                    if str(a) not in self.banned_actions]
                        if not unbanned:
                            unbanned = available
                        a = random.choice(unbanned)
                        if a == 6:
                            non_bg_px = np.argwhere(grid.astype(int) != background_color(grid))
                            if len(non_bg_px) > 0:
                                idx = random.randint(0, len(non_bg_px) - 1)
                                r, c = int(non_bg_px[idx, 0]), int(non_bg_px[idx, 1])
                                actions_to_take = [(6, {'x': c, 'y': r})]
                        if not actions_to_take:
                            actions_to_take = [(a, None)]

                # Don't reset consecutive_cycles to 0 — only new states reset it

            elif steps_since_level_up >= 50 and is_nav_game:
                # Navigation games: if no level after 50 steps, use nav heuristic
                nav_action = navigate_toward_interest(grid, available, visited_nav_positions)
                if nav_action:
                    actions_to_take = [nav_action]

            elif self.use_llm and self.check_llm() and llm_cooldown <= 0:
                # Build situational briefing for the LLM
                explore_ctx = self.exploration_summary()

                # Build last-action description (concrete causal feedback)
                action_desc = last_action_desc
                if not action_desc and steps_since_level_up >= 20:
                    action_desc = f"{steps_since_level_up} actions with no level progress. Try something DIFFERENT."
                if self.banned_actions:
                    ban_str = ", ".join(list(self.banned_actions.keys())[:3])
                    action_desc = (action_desc or "") + f" DO NOT repeat: {ban_str}"

                # Build curated KB context for prompt (not full dump)
                kb_ctx = None
                if kb:
                    # Extract just the most actionable KB info
                    cur_lvl = fd.levels_completed
                    sol = kb.get_level_solution(cur_lvl + 1)
                    if sol and sol.confidence >= 0.5:
                        kb_ctx = f"Prior sessions: L{cur_lvl+1} solution known ({sol.confidence:.0%} confidence)"
                    elif kb.failed_approaches:
                        fails = [fa.approach for fa in kb.failed_approaches[:2]]
                        kb_ctx = f"Prior sessions: tried and failed: {'; '.join(fails)}"
                    elif kb.mechanics:
                        kb_ctx = f"Known: {'; '.join(kb.mechanics[:2])}"

                prompt, category = build_strategy_prompt(
                    grid, available, all_memories,
                    {"current": fd.levels_completed, "total": fd.win_levels},
                    cartridge=cartridge,
                    initial_grid=initial_grid,
                    exploration_summary=explore_ctx,
                    cycle_info=cycle_info,
                    spatial_tracker=self.spatial_tracker,
                    goal_similarity=goal_similarity,
                    cursor_pos=cursor_pos,
                    prev_grid=prev_grid,
                    last_action_desc=action_desc,
                    action_history=list(action_history_descs),
                    kb_context=kb_ctx,
                )
                llm_response = sage_reason(prompt)

                if llm_response:
                    actions_to_take = parse_action(llm_response, available, grid)

                    # Check if action is banned
                    if actions_to_take:
                        action_key = self._action_key(actions_to_take)
                        if action_key in self.banned_actions:
                            if self.verbose:
                                print(f"  [{step}] LLM chose banned action {action_key} — overriding")
                            actions_to_take = []  # Fall to heuristic

                    if self.verbose and actions_to_take:
                        print(f"  [{step}] LLM: {llm_response[:80]}...")
                        print(f"        → {actions_to_take}")

                    # Detect LLM repetition — if same action 3+ times, BAN it
                    if actions_to_take:
                        action_key = self._action_key(actions_to_take)
                        self.recent_action_keys.append(action_key)
                        if (len(self.recent_action_keys) >= 3
                                and len(set(list(self.recent_action_keys)[-3:])) == 1):
                            if self.verbose:
                                print(f"  [{step}] LLM repeating {action_key} — BANNING for 10 steps")
                            self.banned_actions[action_key] = 10
                            actions_to_take = []  # Fall to heuristic/nav

                    llm_cooldown = llm_interval

            # Fallback: navigation heuristic for nav games, else regular heuristic
            if not actions_to_take:
                if is_nav_game:
                    nav_action = navigate_toward_interest(grid, available, visited_nav_positions)
                    if nav_action:
                        actions_to_take = [nav_action]
                if not actions_to_take:
                    actions_to_take = heuristic_action(
                        grid, available, category, step,
                        effective_actions=self.effective_actions,
                        color_effectiveness=self.color_effectiveness)
                llm_cooldown = max(0, llm_cooldown - 1)

            if not actions_to_take:
                # Absolute fallback: random available action
                import random
                a = random.choice(available)
                if a == 6:
                    r, c = grid.shape[0] // 2, grid.shape[1] // 2
                    actions_to_take = [(6, {'x': c, 'y': r})]
                else:
                    actions_to_take = [(a, None)]

            # ACT
            prev_grid = grid.copy()
            for action_int, data in actions_to_take:
                if step >= budget:
                    break
                try:
                    if data:
                        fd = env.step(INT_TO_GAME_ACTION[action_int], data=data)
                    else:
                        fd = env.step(INT_TO_GAME_ACTION[action_int])
                except Exception as e:
                    if self.verbose:
                        print(f"  [{step}] Error: {e}")
                    break

                step += 1
                self.action_history.append(action_int)
                self.click_data_history.append((action_int, data))

            # OBSERVE what changed
            new_grid = get_frame(fd)
            diff = grid_diff(prev_grid, new_grid)
            prev_result = diff if diff != "No change" else None
            n_changed = int(np.sum(prev_grid != new_grid))

            # KB: Record click effects
            if kb:
                for a_done, d_done in actions_to_take:
                    if a_done == 6 and d_done:
                        click_x, click_y = d_done['x'], d_done['y']
                        click_color = color_name(int(prev_grid[click_y, click_x])) if 0 <= click_y < prev_grid.shape[0] and 0 <= click_x < prev_grid.shape[1] else "unknown"
                        level_up = fd.levels_completed > best_levels
                        kb.record_click_effect(
                            r=click_y, c=click_x, color=click_color,
                            cells_changed=n_changed, level_up=level_up,
                            level=fd.levels_completed)

            # Update cursor position and build action description
            old_cursor = cursor_pos
            cursor_pos = detect_cursor(new_grid, prev_grid)

            # Build human-readable description of what just happened
            action_name_map = {1: "up", 2: "down", 3: "left", 4: "right",
                               5: "submit", 6: "click", 7: "undo"}
            if actions_to_take:
                a_int, a_data = actions_to_take[0]
                a_name = action_name_map.get(a_int, f"action{a_int}")
                if a_int == 6 and a_data:
                    a_name = f"click({a_data['x']},{a_data['y']})"
                if diff == "No change":
                    last_action_desc = f"{a_name} → no change"
                else:
                    n_px = int(np.sum(prev_grid != new_grid))
                    if cursor_pos and old_cursor:
                        dx = cursor_pos[0] - old_cursor[0]
                        dy = cursor_pos[1] - old_cursor[1]
                        if dx != 0 or dy != 0:
                            last_action_desc = f"{a_name} → cursor moved ({dx:+d},{dy:+d}), {n_px}px changed"
                        else:
                            last_action_desc = f"{a_name} → {n_px}px changed, cursor stayed"
                    else:
                        last_action_desc = f"{a_name} → {n_px}px changed"
            if last_action_desc:
                action_history_descs.append(last_action_desc)

            # SPATIAL: update object tracker with new frame
            if self.spatial_tracker:
                spatial_diff = self.spatial_tracker.update(new_grid)
                for action_int_done, data_done in actions_to_take:
                    if action_int_done == 6 and data_done:
                        changed = diff != "No change"
                        n_px = int(np.sum(prev_grid != new_grid))
                        self.spatial_tracker.record_click(
                            data_done['x'], data_done['y'], changed, n_px)
                    self.spatial_tracker.record_action_outcome(
                        action_int_done, data_done, spatial_diff)

            # GOAL DETECTION: Track similarity to initial state
            goal_similarity = visual_similarity(new_grid, initial_grid)
            goal_history.append(goal_similarity)
            if len(goal_history) >= 3:
                trend = goal_history[-1] - goal_history[-3]
                if self.verbose and abs(trend) > 0.05:
                    direction = "toward" if trend > 0 else "away from"
                    print(f"  [{step}] Goal: {goal_similarity:.1%} ({direction} initial)")

        # LEARN: store final results
        result = {
            "game_id": game_id,
            "prefix": prefix,
            "levels": best_levels,
            "win_levels": fd.win_levels,
            "steps": step,
            "category": category.get("category", "unknown"),
            "state": fd.state.name,
        }

        # LEARN: Store exploration findings + game results to membot
        # This is the key learning channel — even 0 scores teach SAGE about the game
        explore_ctx = self.exploration_summary()
        learning = (
            f"ARC-AGI-3 {prefix}: {best_levels}/{fd.win_levels} levels in {step} steps. "
            f"Category: {category.get('category')}. Actions: {available}. "
            f"Exploration: {explore_ctx[:200]}"
        )
        # Add spatial learnings if available
        if self.spatial_tracker and hasattr(self.spatial_tracker, 'click_history') and self.spatial_tracker.click_history:
            n_effective = sum(1 for c in self.spatial_tracker.click_history if c.get("changed"))
            n_total = len(self.spatial_tracker.click_history)
            learning += f" Clicks: {n_effective}/{n_total} effective."
        membot_store(learning)

        if best_levels > 0:
            # Store winning strategies separately for stronger recall
            membot_store(
                f"ARC-AGI-3 {prefix} WINNING: scored {best_levels}/{fd.win_levels}. "
                f"Level-up actions: {self.level_up_sequences}. "
                f"Category: {category.get('category')}."
            )

        # LLM REFLECTION: Raising-style metacognitive reflection
        # Not just "what pattern" but "what did I try, what worked, what would I change"
        if self.use_llm and self.check_llm() and step > 10:
            reflection_prompt = (
                f"Game {prefix}: {best_levels}/{fd.win_levels} levels in {step} steps. "
                f"Category: {category.get('category')}. "
                f"{explore_ctx[:100]} "
                f"What did you try? What worked or didn't? What would you do differently? One sentence each."
            )
            reflection = sage_reason(reflection_prompt, max_tokens=100)
            if reflection:
                membot_store(f"ARC-AGI-3 {prefix} reflection: {reflection[:200]}")
                if self.verbose:
                    print(f"  Reflection: {reflection[:100]}")

                # KB: Extract mechanics or questions from reflection
                if kb:
                    if best_levels > 0:
                        kb.add_mechanic(f"Scored {best_levels}L: {reflection[:100]}")
                    elif step > 100:
                        kb.add_question(f"0L after {step} steps: {reflection[:80]}")

        # Save knowledge base (persistent across runs)
        if kb:
            kb.save()
            if self.verbose:
                s = kb.stats
                print(f"  KB saved: {s['n_objects']} objects, {s['n_effects']} effects, "
                      f"{s['n_solutions']} solutions")

        # Save visual memory cartridge
        cartridge.write()
        if self.verbose:
            print(f"  Visual memory: saved {len(cartridge.data.get('visual_memory', {}).get('snapshots', {}))} snapshots")

        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAGE Autonomous ARC-AGI-3 Agent")
    parser.add_argument("--game", default=None, help="Game ID substring filter")
    parser.add_argument("--all", action="store_true", help="Run all games")
    parser.add_argument("--budget", type=int, default=500, help="Max actions per game")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM reasoning")
    parser.add_argument("--runs", type=int, default=1, help="Runs per game")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE Autonomous Agent — ARC-AGI-3")
    print("=" * 60)

    # Mount membot cartridge
    try:
        requests.post(f"{MEMBOT_URL}/api/mount",
            json={"name": "sage-sprout"}, timeout=3)
        print("Membot: mounted sage-sprout cartridge")
    except Exception:
        print("Membot: not available (running without memory)")

    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in
                   (e.game_id if hasattr(e, "game_id") else str(e))]
    elif args.all:
        targets = envs
    else:
        targets = envs[:5]  # Default: first 5

    print(f"\n{len(targets)} game(s), budget={args.budget}, runs={args.runs}")
    print(f"LLM: {'enabled' if not args.no_llm else 'disabled'}")

    total_levels = 0
    total_scored = 0
    results = []

    for env_info in targets:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        prefix = game_id.split("-")[0]

        best_result = None
        for run in range(args.runs):
            agent = SageAgent(use_llm=not args.no_llm, verbose=args.verbose)
            result = agent.play_game(arcade, game_id, budget=args.budget)

            if best_result is None or result["levels"] > best_result["levels"]:
                best_result = result

        lvl = best_result["levels"]
        win = best_result["win_levels"]
        cat = best_result["category"]
        star = f" ★{lvl}" if lvl > 0 else ""
        print(f"  {prefix:6s}  {lvl}/{win}{star}  [{cat}]")

        if lvl > 0:
            total_levels += lvl
            total_scored += 1
        results.append(best_result)

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_levels} levels across {total_scored}/{len(targets)} games")

    # Save membot
    try:
        requests.post(f"{MEMBOT_URL}/api/save",
            json={"name": "sage-sprout"}, timeout=5)
    except Exception:
        pass

    # Save log
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"sage_agent_{int(time.time())}.json")
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "total_levels": total_levels,
            "games_scored": total_scored,
            "results": results,
        }, f, indent=2)
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
