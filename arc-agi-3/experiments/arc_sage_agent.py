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

import sys, os, time, json, re
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from arcengine import GameAction
from arc_agi import Arcade
from arc_perception import (
    get_frame, full_perception, grid_diff, grid_summary,
    find_color_regions, find_markers, background_color, color_name,
    detect_sections, find_row_patterns,
)

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}
MEMBOT_URL = "http://localhost:8000"
SAGE_URL = "http://localhost:8750"  # SAGE daemon for LLM reasoning


# ─── Membot integration ───

def membot_recall(query, n=5):
    """Retrieve relevant memories."""
    try:
        resp = requests.post(f"{MEMBOT_URL}/api/search",
            json={"query": query, "n": n}, timeout=3)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            return [r["text"] for r in results if r.get("score", 0) > 0.4]
    except Exception:
        pass
    return []


def membot_store(text):
    """Store a memory."""
    try:
        requests.post(f"{MEMBOT_URL}/api/store",
            json={"text": text}, timeout=3)
    except Exception:
        pass


# ─── LLM reasoning ───

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:12b")


def sage_reason(prompt, max_tokens=150):
    """Ask LLM to reason about the game state.

    Tries SAGE daemon first, falls back to Ollama direct.
    Returns the LLM's text response, or None if unavailable.
    """
    # Try SAGE daemon
    try:
        resp = requests.post(f"{SAGE_URL}/chat",
            json={"message": prompt, "max_tokens": max_tokens},
            timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("response", data.get("text", ""))
            if text and "SAGE instance" not in text:  # Skip if daemon is in character
                return text
    except Exception:
        pass

    # Fallback: Ollama direct
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                   "options": {"temperature": 0.3, "num_predict": max_tokens}},
            timeout=60)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
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

    # Parse click with coordinates
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

    # Parse directional actions
    direction_map = {"up": 1, "down": 2, "left": 3, "right": 4}
    for dname, daction in direction_map.items():
        if dname in text and daction in available_actions:
            actions.append((daction, None))

    # Parse submit/select
    if any(w in text for w in ["submit", "select", "confirm", "enter"]):
        if 5 in available_actions:
            actions.append((5, None))

    # Parse undo
    if any(w in text for w in ["undo", "reset", "back"]):
        if 7 in available_actions:
            actions.append((7, None))

    return actions


# ─── Perception-based heuristic strategies ───

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


def build_strategy_prompt(grid, available_actions, memories, level_info, prev_action_result=None):
    """Build the prompt for LLM reasoning.

    Compact enough for 0.8B model to handle, detailed enough to be useful.
    """
    perception = full_perception(grid)
    category = detect_game_category(grid, available_actions)

    action_names = {1: "up", 2: "down", 3: "left", 4: "right",
                    5: "submit", 6: "click(x,y)", 7: "undo"}
    avail_str = ", ".join(action_names.get(a, f"action{a}") for a in sorted(available_actions))

    prompt_parts = [
        f"GAME PUZZLE - Level {level_info.get('current', '?')}/{level_info.get('total', '?')}",
        f"\nWhat I see:\n{perception}",
        f"\nGame type guess: {category.get('category', 'unknown')} — {category.get('hint', '')}",
        f"\nAvailable actions: {avail_str}",
    ]

    if memories:
        prompt_parts.append(f"\nWhat I remember:\n" + "\n".join(f"- {m}" for m in memories[:3]))

    if prev_action_result:
        prompt_parts.append(f"\nLast action result: {prev_action_result}")

    prompt_parts.append(
        "\nWhat should I do? Think step by step, then say your action. "
        "For clicks, say 'click(X, Y)'. For movement, say 'move up/down/left/right'. "
        "For submit, say 'submit'."
    )

    return "\n".join(prompt_parts), category


# ─── Fallback heuristic (no LLM needed) ───

def heuristic_action(grid, available_actions, category, step_count):
    """Fallback action when LLM is unavailable or response unparseable.

    Uses perception-based heuristics.
    """
    bg = background_color(grid)

    if category.get("category") == "placement_puzzle" and 6 in available_actions:
        # For placement puzzles: identify items and slots
        sections = detect_sections(grid)
        if len(sections) >= 3:
            # Try clicking bottom section items then middle section slots
            bot = sections[-1]
            mid = sections[len(sections) // 2]

            bot_regions = find_color_regions(grid, min_size=4)
            bot_items = [r for r in bot_regions
                        if bot["y_start"] <= r["cy"] <= bot["y_end"]
                        and r["color"] != bg]
            mid_markers = find_markers(grid, 2, min_cluster=2)  # color 2 = red markers often = empty slots

            if bot_items and mid_markers:
                # Click first bottom item, then first slot
                item = bot_items[0]
                slot = mid_markers[0]
                return [
                    (6, {'x': item["cx"], 'y': item["cy"]}),
                    (6, {'x': slot["x"], 'y': slot["y"]}),
                ]

    if 6 in available_actions:
        # Default: click a non-background pixel
        non_bg = np.argwhere(grid.astype(int) != bg)
        if len(non_bg) > 0:
            idx = step_count % len(non_bg)  # cycle through positions
            r, c = int(non_bg[idx, 0]), int(non_bg[idx, 1])
            return [(6, {'x': c, 'y': r})]

    # Movement fallback: cycle through directions
    moves = [a for a in available_actions if a in [1, 2, 3, 4]]
    if moves:
        return [(moves[step_count % len(moves)], None)]

    return []


# ─── Main agent loop ───

class SageAgent:
    """Autonomous ARC-AGI-3 game-playing agent.

    Uses perception + memory + LLM reasoning to solve games.
    No game-specific code — learns and reasons its way through.
    """

    def __init__(self, use_llm=True, verbose=False):
        self.use_llm = use_llm
        self.verbose = verbose
        self.llm_available = None  # Will check on first use
        self.action_history = []
        self.level_observations = []

    def check_llm(self):
        """Check if SAGE daemon is available for reasoning."""
        if self.llm_available is not None:
            return self.llm_available
        try:
            resp = requests.get(f"{SAGE_URL}/health", timeout=5)
            self.llm_available = resp.status_code == 200
        except Exception:
            self.llm_available = False
        if self.verbose:
            print(f"  LLM available: {self.llm_available}")
        return self.llm_available

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

        # Play loop
        best_levels = 0
        step = 0
        prev_result = None
        prev_grid = grid.copy()
        llm_cooldown = 0  # Steps until next LLM call (don't call every step)
        llm_interval = 5  # Call LLM every N steps

        while step < budget and fd.state.name not in ("WON", "GAME_OVER"):
            # PERCEIVE
            grid = get_frame(fd)

            # Track level changes
            if fd.levels_completed > best_levels:
                best_levels = fd.levels_completed
                diff_desc = grid_diff(prev_grid, grid)
                self.level_observations.append({
                    "level": best_levels,
                    "actions_before": list(self.action_history[-10:]),
                    "grid_change": diff_desc,
                })
                if self.verbose:
                    print(f"  ★ Level {best_levels}/{fd.win_levels}!")

                # Store level-up knowledge
                recent_actions = self.action_history[-15:]
                membot_store(
                    f"ARC-AGI-3 {prefix}: level {best_levels} reached. "
                    f"Category: {category.get('category')}. "
                    f"Recent actions: {recent_actions}. "
                    f"Grid change: {diff_desc[:200]}"
                )

            # REASON (LLM or heuristic)
            actions_to_take = []

            if self.use_llm and self.check_llm() and llm_cooldown <= 0:
                # Build prompt and ask LLM
                prompt, category = build_strategy_prompt(
                    grid, available, all_memories,
                    {"current": fd.levels_completed, "total": fd.win_levels},
                    prev_result,
                )
                llm_response = sage_reason(prompt)

                if llm_response:
                    actions_to_take = parse_action(llm_response, available, grid)
                    if self.verbose and actions_to_take:
                        print(f"  [{step}] LLM: {llm_response[:80]}...")
                        print(f"        → {actions_to_take}")
                    llm_cooldown = llm_interval

            # Fallback to heuristic if no LLM actions
            if not actions_to_take:
                actions_to_take = heuristic_action(grid, available, category, step)
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

            # OBSERVE what changed
            new_grid = get_frame(fd)
            diff = grid_diff(prev_grid, new_grid)
            prev_result = diff if diff != "No change" else None

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

        if best_levels > 0:
            membot_store(
                f"ARC-AGI-3 {prefix}: scored {best_levels}/{fd.win_levels}. "
                f"Category: {category.get('category')}. "
                f"Actions: {available}. Steps used: {step}/{budget}."
            )

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
