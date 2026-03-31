#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Hybrid: Membot Learning + LLM Navigator

Combines:
- Membot cartridge: Cross-session color effectiveness learning
- LLM Navigator (Qwen 3.5 27B): Strategic guidance every N actions
- Coordinate-aware clicking: ACTION6 with {x, y} targeting

Usage:
    cd ~/ai-workspace/SAGE
    .venv-arc/bin/python3 arc-agi-3/experiments/sage_hybrid_qwen.py --game lp85 --steps 5000
"""

import sys
import time
import json
import random
import requests
from collections import defaultdict
import numpy as np

sys.path.insert(0, ".")
from arc_agi import Arcade
from arcengine import GameAction

sys.path.insert(0, "arc-agi-3/experiments")
from membot_cartridge import MembotCartridge

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}


class HybridPlayer:
    """Membot learning + LLM strategic navigation."""

    def __init__(self, game_id: str, available_actions: list, grid_shape: tuple):
        self.game_id = game_id
        self.available = available_actions
        self.has_click = 6 in self.available
        self.grid_shape = grid_shape

        # LLM Navigator state
        self.llm_model = "qwen3.5:27b"
        self.actions_since_reflection = 0
        self.reflection_interval = 20
        self.current_strategy = {"focus_colors": [], "exploration_rate": 0.5}

        # Membot integration
        self.cartridge = MembotCartridge(game_id)
        self.cartridge_data = self.cartridge.read()

        # Color effectiveness (loaded from membot) - now with per-level tracking
        self.color_stats = defaultdict(lambda: {
            "tries": 0,
            "changes": 0,
            "level_ups": 0,
            "per_level": {}  # {level_num: {"tries": 0, "changes": 0}}
        })
        self._load_color_effectiveness()

        # Action history for Navigator
        self.recent_actions = []
        self.recent_changes = []

        # Stagnation detection
        self.actions_on_current_level = 0
        self.last_level = 0

        print(f"  Hybrid: Membot loaded, Navigator: {self.llm_model}")
        print(f"  Membot: {self.cartridge.summary()}")

    def _load_color_effectiveness(self):
        """Load prior color effectiveness from membot."""
        if not self.cartridge_data:
            return

        click_eff = self.cartridge_data.get("click_effectiveness", {}).get("global", {})
        for color_key, stats in click_eff.items():
            if color_key.startswith("color_"):
                color = int(color_key.split("_")[1])
                self.color_stats[color]["tries"] = stats.get("tries", 0)
                self.color_stats[color]["changes"] = stats.get("changes", 0)
                self.color_stats[color]["level_ups"] = stats.get("level_ups", 0)

                # Load per-level data if available
                per_level = stats.get("per_level", {})
                for level_str, level_stats in per_level.items():
                    level = int(level_str)
                    self.color_stats[color]["per_level"][level] = {
                        "tries": level_stats.get("tries", 0),
                        "changes": level_stats.get("changes", 0)
                    }

        if click_eff:
            top_colors = sorted(click_eff.items(),
                              key=lambda x: x[1].get("rate", 0), reverse=True)[:3]
            self.current_strategy["focus_colors"] = [
                int(k.split("_")[1]) for k, _ in top_colors
            ]
            print(f"    Prior knowledge: focus on colors {self.current_strategy['focus_colors']}")

    def find_targets(self, grid: np.ndarray, current_level: int = 1) -> list:
        """Find click targets prioritized by membot + LLM strategy with level-specific learning."""
        bg = int(np.bincount(grid.flatten()).argmax())
        candidates = []

        non_bg = np.argwhere(grid != bg)
        if len(non_bg) == 0:
            h, w = grid.shape
            return [(h//2, w//2, int(grid[h//2, w//2]), 0.5)]

        # Prioritize by membot learning (level-specific when available) + LLM focus
        for r, c in non_bg:
            color = int(grid[r, c])
            stats = self.color_stats[color]

            # Use level-specific stats if available, fallback to global
            level_stats = stats["per_level"].get(current_level)
            if level_stats and level_stats["tries"] > 0:
                # Level-specific effectiveness (prioritized)
                effectiveness = level_stats["changes"] / level_stats["tries"]
                level_up_bonus = 0  # Already level-specific
                priority = effectiveness
            elif stats["tries"] > 0:
                # Global effectiveness (fallback) - but penalize slightly
                effectiveness = stats["changes"] / stats["tries"]
                level_up_bonus = stats["level_ups"] * 0.5
                priority = (effectiveness + level_up_bonus) * 0.8  # 20% penalty for not level-specific
            else:
                # Unknown color - neutral priority
                priority = 0.5

            # Boost priority if LLM Navigator suggests this color
            if color in self.current_strategy["focus_colors"]:
                priority *= 1.5

            candidates.append((int(r), int(c), color, priority))

        candidates.sort(key=lambda x: x[3], reverse=True)
        return candidates[:50]  # Top 50 targets

    def select_target(self, targets: list, step: int, current_level: int) -> tuple:
        """Select target with exploration/exploitation balance + stagnation detection."""
        if not targets:
            return (random.randint(0, 63), random.randint(0, 63), 0)

        # Detect stagnation: if stuck on same level for >100 actions, force exploration
        if current_level == self.last_level:
            self.actions_on_current_level += 1
        else:
            self.actions_on_current_level = 0
            self.last_level = current_level

        # Force high exploration if stagnant
        if self.actions_on_current_level > 100:
            explore_rate = 0.9  # 90% exploration when stagnant
            if self.actions_on_current_level % 50 == 0:
                print(f"    Stagnation detected: {self.actions_on_current_level} actions on Level {current_level}, forcing 90% exploration")
        else:
            explore_rate = self.current_strategy["exploration_rate"]

        if random.random() < explore_rate:
            r, c, color, _ = random.choice(targets)
        else:
            r, c, color, _ = targets[0]

        return (r, c, color)

    def record_action(self, action: str, changed: bool):
        """Record action outcome."""
        self.recent_actions.append(action)
        self.recent_changes.append(changed)
        if len(self.recent_actions) > 30:
            self.recent_actions = self.recent_actions[-30:]
            self.recent_changes = self.recent_changes[-30:]
        self.actions_since_reflection += 1

    def record_click(self, color: int, changed: bool, current_level: int):
        """Record click outcome for membot learning (global + per-level)."""
        # Global stats
        self.color_stats[color]["tries"] += 1
        if changed:
            self.color_stats[color]["changes"] += 1

        # Per-level stats
        if current_level not in self.color_stats[color]["per_level"]:
            self.color_stats[color]["per_level"][current_level] = {"tries": 0, "changes": 0}

        self.color_stats[color]["per_level"][current_level]["tries"] += 1
        if changed:
            self.color_stats[color]["per_level"][current_level]["changes"] += 1

    def record_level_up(self, level: int):
        """Record level completion and update membot."""
        recent_colors = []
        for action in self.recent_actions[-15:]:
            if "color=" in action:
                color = int(action.split("color=")[1].split(")")[0])
                recent_colors.append(color)

        for color in set(recent_colors):
            self.color_stats[color]["level_ups"] += 1

        if self.cartridge_data is None:
            self.cartridge_data = self.cartridge._empty_cartridge()

        # Update membot with global + per-level stats
        click_eff = self.cartridge_data.setdefault("click_effectiveness", {"global": {}})
        for color, stats in self.color_stats.items():
            if stats["tries"] > 0:
                color_data = {
                    "tries": stats["tries"],
                    "changes": stats["changes"],
                    "level_ups": stats["level_ups"],
                    "rate": stats["changes"] / stats["tries"]
                }

                # Add per-level stats
                if stats["per_level"]:
                    color_data["per_level"] = {}
                    for lvl, lvl_stats in stats["per_level"].items():
                        color_data["per_level"][str(lvl)] = {
                            "tries": lvl_stats["tries"],
                            "changes": lvl_stats["changes"],
                            "rate": lvl_stats["changes"] / lvl_stats["tries"] if lvl_stats["tries"] > 0 else 0
                        }

                click_eff["global"][f"color_{color}"] = color_data

        if recent_colors:
            top_color = max(set(recent_colors), key=recent_colors.count)
            insight = f"Level {level}: clicking color-{top_color} completes level"
            self.cartridge.add_strategic_insight(insight, confidence=0.9)

        self.cartridge.add_winning_sequence(level, [], [])
        self.cartridge.write()
        print(f"    Membot: Stored level {level} pattern")

        # Reset strategy after level completion to explore new patterns
        self.current_strategy["focus_colors"] = []
        self.current_strategy["exploration_rate"] = 0.7  # High exploration for new level
        self.actions_since_reflection = self.reflection_interval  # Trigger immediate reflection
        print(f"    Strategy: Reset for Level {level+1} - exploring new patterns")

    def navigate(self, grid: np.ndarray, levels_completed: int, win_levels: int):
        """LLM Navigator provides strategic guidance."""
        if not self.has_click:
            return

        # Prepare context for LLM
        effective_rate = sum(self.recent_changes[-20:]) / max(len(self.recent_changes[-20:]), 1)

        color_summary = []
        for color, stats in sorted(self.color_stats.items(),
                                   key=lambda x: x[1]["changes"] / max(x[1]["tries"], 1),
                                   reverse=True)[:5]:
            if stats["tries"] > 0:
                rate = stats["changes"] / stats["tries"]
                color_summary.append(f"color-{color}: {rate:.0%} ({stats['changes']}/{stats['tries']})")

        prompt = f"""You are analyzing a grid puzzle game. Current state:

Progress: Level {levels_completed}/{win_levels}
Recent effectiveness: {effective_rate:.0%} of actions cause changes
Top colors by effectiveness: {', '.join(color_summary) if color_summary else 'none yet'}

Recent actions: {' '.join(self.recent_actions[-10:])}

Question: What strategy should we use next? Which colors should we focus on?

Respond with ONLY valid JSON:
{{
  "focus_colors": [list of 1-3 color numbers to prioritize],
  "exploration_rate": 0.1 to 0.9,
  "reasoning": "brief explanation"
}}"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.llm_model, "prompt": prompt, "stream": False},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "")

                # Extract JSON from response
                if "{" in text and "}" in text:
                    json_start = text.index("{")
                    json_end = text.rindex("}") + 1
                    strategy = json.loads(text[json_start:json_end])

                    self.current_strategy["focus_colors"] = strategy.get("focus_colors", [])
                    self.current_strategy["exploration_rate"] = strategy.get("exploration_rate", 0.5)

                    print(f"    Navigator: Focus {self.current_strategy['focus_colors']}, "
                          f"explore {self.current_strategy['exploration_rate']:.0%}")
                    print(f"    Reasoning: {strategy.get('reasoning', 'N/A')[:80]}")

        except Exception as e:
            print(f"    Navigator: Error ({e})")

        self.actions_since_reflection = 0

    def save_final_stats(self):
        """Save final stats to membot (global + per-level)."""
        if self.cartridge_data is None:
            self.cartridge_data = self.cartridge._empty_cartridge()

        click_eff = self.cartridge_data.setdefault("click_effectiveness", {"global": {}})
        for color, stats in self.color_stats.items():
            if stats["tries"] > 0:
                color_data = {
                    "tries": stats["tries"],
                    "changes": stats["changes"],
                    "level_ups": stats["level_ups"],
                    "rate": stats["changes"] / stats["tries"]
                }

                # Add per-level stats
                if stats["per_level"]:
                    color_data["per_level"] = {}
                    for lvl, lvl_stats in stats["per_level"].items():
                        color_data["per_level"][str(lvl)] = {
                            "tries": lvl_stats["tries"],
                            "changes": lvl_stats["changes"],
                            "rate": lvl_stats["changes"] / lvl_stats["tries"] if lvl_stats["tries"] > 0 else 0
                        }

                click_eff["global"][f"color_{color}"] = color_data

        self.cartridge.increment_attempts()
        self.cartridge.write()


def play_hybrid(env, frame_data, max_steps: int, game_id: str, verbose: bool = False):
    """Play game with hybrid approach."""
    available = [a.value if hasattr(a, "value") else int(a)
                for a in (frame_data.available_actions or [])]

    grid = np.array(frame_data.frame)
    if grid.ndim == 3:
        grid = grid[-1]

    player = HybridPlayer(game_id, available, grid.shape)

    if not player.has_click:
        print(f"  No ACTION6 available")
        return {"steps": 0, "levels_completed": 0}

    start_levels = frame_data.levels_completed
    prev_levels = start_levels
    level_events = []

    for step in range(max_steps):
        prev_grid = grid.copy()

        # Current level (levels_completed + 1 = current level being attempted)
        current_level = frame_data.levels_completed + 1

        # Find and select target with level-specific learning + stagnation detection
        targets = player.find_targets(grid, current_level)
        r, c, color = player.select_target(targets, step, current_level)

        # Execute click
        try:
            frame_data = env.step(GameAction.ACTION6, data={'x': c, 'y': r})
            action_desc = f"CLICK({r},{c}) color={color}"
        except Exception as e:
            if verbose and step % 1000 == 0:
                print(f"    Step {step}: Click failed ({e})")
            continue

        if frame_data is None:
            break

        new_grid = np.array(frame_data.frame)
        if new_grid.ndim == 3:
            new_grid = new_grid[-1]
        if new_grid.size == 0:
            continue

        changed = int(np.sum(prev_grid != new_grid)) > 0
        grid = new_grid

        player.record_action(action_desc, changed)
        player.record_click(color, changed, current_level)

        # Navigator reflection every N actions
        if player.actions_since_reflection >= player.reflection_interval:
            player.navigate(grid, frame_data.levels_completed, frame_data.win_levels)

        # Level completion!
        if frame_data.levels_completed > prev_levels:
            player.record_level_up(frame_data.levels_completed)
            level_events.append({
                "step": step,
                "level": frame_data.levels_completed,
                "action_window": player.recent_actions[-10:]
            })
            if verbose:
                print(f"    ★ LEVEL {frame_data.levels_completed}/{frame_data.win_levels} at step {step}!")
            prev_levels = frame_data.levels_completed

        if frame_data.state.name in ("WON", "LOST"):
            break

        if verbose and step > 0 and step % 1000 == 0:
            print(f"    Step {step}: {frame_data.levels_completed}/{frame_data.win_levels} levels")

    player.save_final_stats()

    return {
        "steps": step + 1 if 'step' in dir() else 0,
        "levels_completed": prev_levels,
        "win_levels": getattr(frame_data, 'win_levels', 0),
        "state": frame_data.state.name if hasattr(frame_data, 'state') else "UNKNOWN",
        "level_events": level_events,
        "total_levels_gained": prev_levels - start_levels
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAGE Hybrid: Membot + LLM Navigator")
    parser.add_argument("--game", default="lp85", help="Game ID prefix")
    parser.add_argument("--steps", type=int, default=5000, help="Steps per run")
    parser.add_argument("--runs", type=int, default=1, help="Runs per game")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("SAGE ARC-AGI-3 Hybrid: Membot + LLM Navigator (Qwen 3.5 27B)")
    print("=" * 70)

    arcade = Arcade()
    envs = arcade.get_environments()

    targets = [e for e in envs if args.game in str(e.game_id)]

    print(f"\nTargeting {len(targets)} game(s), {args.runs} runs, {args.steps} steps/run\n")

    all_results = []
    total_levels = 0

    for env_info in targets:
        game_id = env_info.game_id
        print(f"{'─'*70}")
        print(f"Game: {game_id}")

        for run in range(args.runs):
            try:
                env = arcade.make(game_id)
                frame_data = env.reset()
                result = play_hybrid(env, frame_data, args.steps, game_id, verbose=args.verbose)
            except Exception as e:
                print(f"  Run {run+1}: FAILED ({e})")
                continue

            levels = result["total_levels_gained"]
            total_levels += levels

            status = f" ★ {levels} level(s)!" if levels > 0 else ""
            print(f"  Run {run+1}: {result['steps']} steps, "
                  f"{result['levels_completed']}/{result['win_levels']} levels{status}")

            all_results.append({"game_id": game_id, **result})

    print(f"\n{'='*70}")
    print(f"SUMMARY: {total_levels} total levels")
    print(f"{'='*70}")
    for r in all_results:
        lvl = f"{r.get('levels_completed',0)}/{r.get('win_levels',0)}"
        status = " ★" if r.get('total_levels_gained', 0) > 0 else ""
        print(f"  {r['game_id']:20s}  {lvl}{status}")

    # Save log
    log_path = f"arc-agi-3/experiments/logs/hybrid_qwen_{int(time.time())}.json"
    try:
        import os
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({"results": all_results, "total_levels": total_levels}, f, indent=2, default=str)
        print(f"\n  Saved: {log_path}")
    except Exception as e:
        print(f"  (save failed: {e})")


if __name__ == "__main__":
    main()
