#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Membot-Integrated Clicker

Builds federated modular cartridge for competition:
- Loads prior color effectiveness from membot at game start
- Prioritizes clicking colors with known high effectiveness
- Stores winning click patterns in membot after level completions
- Accumulates cross-session and cross-machine learning

Usage:
    cd /path/to/SAGE
    .venv-arc/bin/python3 arc-agi-3/experiments/sage_membot_clicker.py --game lp85 --steps 5000
"""

import sys
import time
import json
import random
from collections import defaultdict
import numpy as np

sys.path.insert(0, ".")

from arc_agi import Arcade
from arcengine import GameAction

# Import membot_cartridge
sys.path.insert(0, "arc-agi-3/experiments")
from membot_cartridge import MembotCartridge

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}


class MembotClicker:
    """Coordinate-aware clicker with membot cross-session learning."""

    def __init__(self, game_id: str, available_actions: list):
        self.game_id = game_id
        self.game_family = game_id.split("-")[0]
        self.available = available_actions
        self.has_click = 6 in self.available

        # Membot integration
        self.cartridge = MembotCartridge(game_id)
        self.cartridge_data = self.cartridge.read()

        # Color effectiveness (loaded from membot + updated live)
        self.color_stats = defaultdict(lambda: {"tries": 0, "changes": 0, "level_ups": 0})
        self._load_color_effectiveness()

        # Recent actions for level-up tracking
        self.recent_actions = []
        self.recent_coords = []

        print(f"  Membot: {self.cartridge.summary()}")

    def _load_color_effectiveness(self):
        """Load prior color effectiveness from membot cartridge."""
        if not self.cartridge_data:
            return

        # Load from cartridge's click_effectiveness
        click_eff = self.cartridge_data.get("click_effectiveness", {})
        global_eff = click_eff.get("global", {})

        for color_key, stats in global_eff.items():
            if color_key.startswith("color_"):
                color = int(color_key.split("_")[1])
                self.color_stats[color]["tries"] = stats.get("tries", 0)
                self.color_stats[color]["changes"] = stats.get("changes", 0)
                self.color_stats[color]["level_ups"] = stats.get("level_ups", 0)

        if global_eff:
            print(f"    Loaded prior click data for {len(global_eff)} colors")

    def find_targets(self, grid: np.ndarray) -> list:
        """Find click targets prioritized by membot-learned effectiveness.

        Returns:
            List of (row, col, color, priority) tuples, sorted by priority
        """
        bg = int(np.bincount(grid.flatten()).argmax())
        candidates = []

        # Gather all non-background positions
        non_bg = np.argwhere(grid != bg)
        if len(non_bg) == 0:
            # Uniform grid - try center
            h, w = grid.shape
            return [(h//2, w//2, int(grid[h//2, w//2]), 0.5)]

        # Group by color with learned priority
        for r, c in non_bg:
            color = int(grid[r, c])
            stats = self.color_stats[color]

            # Priority = (changes / tries) + bonus for level_ups
            if stats["tries"] > 0:
                effectiveness = stats["changes"] / stats["tries"]
                level_up_bonus = stats["level_ups"] * 0.5
                priority = effectiveness + level_up_bonus
            else:
                # Unknown color - medium priority for exploration
                priority = 0.5

            candidates.append((int(r), int(c), color, priority))

        # Sort by priority descending
        candidates.sort(key=lambda x: x[3], reverse=True)

        # Take top 20 per color for exploration
        color_groups = defaultdict(list)
        for r, c, color, pri in candidates:
            color_groups[color].append((r, c, color, pri))

        targets = []
        for color, positions in color_groups.items():
            # Sample up to 5 per color, weighted by priority
            sample = positions[:5]
            targets.extend(sample)

        return targets

    def select_target(self, targets: list, step: int) -> tuple:
        """Select best target with exploration/exploitation balance.

        Args:
            targets: List of (row, col, color, priority) tuples
            step: Current step number

        Returns:
            (row, col, color) tuple
        """
        if not targets:
            # Fallback: random position
            return (random.randint(0, 63), random.randint(0, 63), 0)

        # Exploration rate decays over time
        explore_rate = max(0.1, 0.5 - (step / 10000) * 0.4)

        if random.random() < explore_rate:
            # Explore: sample from all targets
            r, c, color, _ = random.choice(targets)
        else:
            # Exploit: pick highest priority
            r, c, color, _ = targets[0]

        return (r, c, color)

    def record_click(self, color: int, changed: bool, action_desc: str):
        """Record click outcome for learning."""
        self.color_stats[color]["tries"] += 1
        if changed:
            self.color_stats[color]["changes"] += 1

        self.recent_actions.append(action_desc)
        if len(self.recent_actions) > 30:
            self.recent_actions = self.recent_actions[-30:]

    def record_level_up(self, level: int):
        """Record level completion and update membot."""
        # Analyze recent actions for winning colors
        recent_colors = []
        for action in self.recent_actions[-15:]:
            if "color=" in action:
                color = int(action.split("color=")[1].split(")")[0])
                recent_colors.append(color)

        # Increment level_ups for colors in winning sequence
        for color in set(recent_colors):
            self.color_stats[color]["level_ups"] += 1

        # Store in membot cartridge
        if self.cartridge_data is None:
            self.cartridge_data = self.cartridge._empty_cartridge()

        # Update click_effectiveness
        click_eff = self.cartridge_data.setdefault("click_effectiveness", {"global": {}})
        for color, stats in self.color_stats.items():
            if stats["tries"] > 0:
                click_eff["global"][f"color_{color}"] = {
                    "tries": stats["tries"],
                    "changes": stats["changes"],
                    "level_ups": stats["level_ups"],
                    "rate": stats["changes"] / stats["tries"]
                }

        # Add winning sequence
        self.cartridge.add_winning_sequence(
            level=level,
            actions=[],  # Action strings, not ints
            state_hashes=[]
        )

        # Add strategic insight about winning color
        if recent_colors:
            top_color = max(set(recent_colors), key=recent_colors.count)
            insight = f"Level {level}: clicking color-{top_color} completes level"
            self.cartridge.add_strategic_insight(insight, confidence=0.9)

        self.cartridge.write()
        print(f"    Membot: Stored level {level} pattern (color-{top_color})")

    def save_final_stats(self):
        """Save final click effectiveness to membot."""
        if self.cartridge_data is None:
            self.cartridge_data = self.cartridge._empty_cartridge()

        # Update all color stats
        click_eff = self.cartridge_data.setdefault("click_effectiveness", {"global": {}})
        for color, stats in self.color_stats.items():
            if stats["tries"] > 0:
                click_eff["global"][f"color_{color}"] = {
                    "tries": stats["tries"],
                    "changes": stats["changes"],
                    "level_ups": stats["level_ups"],
                    "rate": stats["changes"] / stats["tries"]
                }

        self.cartridge.increment_attempts()
        self.cartridge.write()

    def effectiveness_report(self) -> str:
        """Generate effectiveness report."""
        lines = []
        sorted_colors = sorted(self.color_stats.items(),
                             key=lambda x: x[1]["changes"] / max(x[1]["tries"], 1),
                             reverse=True)

        for color, stats in sorted_colors:
            if stats["tries"] > 0:
                rate = stats["changes"] / stats["tries"]
                level_up_str = f" ({stats['level_ups']} level-ups)" if stats["level_ups"] > 0 else ""
                lines.append(f"  color {color}: {stats['changes']}/{stats['tries']} = {rate:.0%}{level_up_str}")

        return "\n".join(lines)


def play_with_membot(env, frame_data, max_steps: int, game_id: str, verbose: bool = False):
    """Play game with membot-integrated clicker."""
    available = [a.value if hasattr(a, "value") else int(a) for a in (frame_data.available_actions or [])]

    grid = np.array(frame_data.frame)
    if grid.ndim == 3:
        grid = grid[-1]

    # Initialize membot clicker
    clicker = MembotClicker(game_id, available)

    if not clicker.has_click:
        print(f"  No ACTION6 available, skipping")
        return {"steps": 0, "levels_completed": 0, "win_levels": 0}

    start_levels = frame_data.levels_completed
    prev_levels = start_levels
    level_events = []

    for step in range(max_steps):
        prev_grid = grid.copy()

        # Find and select target
        targets = clicker.find_targets(grid)
        r, c, color = clicker.select_target(targets, step)

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

        # Process frame
        new_grid = np.array(frame_data.frame)
        if new_grid.ndim == 3:
            new_grid = new_grid[-1]
        if new_grid.size == 0:
            continue

        changed = int(np.sum(prev_grid != new_grid)) > 0
        grid = new_grid

        clicker.record_click(color, changed, action_desc)

        # Level completion!
        if frame_data.levels_completed > prev_levels:
            clicker.record_level_up(frame_data.levels_completed)
            level_events.append({
                "step": step,
                "level": frame_data.levels_completed,
                "action_window": clicker.recent_actions[-10:]
            })
            if verbose:
                print(f"    ★ LEVEL {frame_data.levels_completed}/{frame_data.win_levels} at step {step}!")
            prev_levels = frame_data.levels_completed

        if frame_data.state.name in ("WON", "LOST"):
            break

        if verbose and step > 0 and step % 1000 == 0:
            print(f"    Step {step}: {frame_data.levels_completed}/{frame_data.win_levels} levels")

    # Save final stats to membot
    clicker.save_final_stats()

    return {
        "steps": step + 1 if 'step' in dir() else 0,
        "levels_completed": prev_levels,
        "win_levels": getattr(frame_data, 'win_levels', 0),
        "state": frame_data.state.name if hasattr(frame_data, 'state') else "UNKNOWN",
        "level_events": level_events,
        "total_levels_gained": prev_levels - start_levels,
        "effectiveness": clicker.effectiveness_report()
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAGE Membot-Integrated Clicker")
    parser.add_argument("--game", default=None, help="Game ID prefix")
    parser.add_argument("--steps", type=int, default=5000, help="Steps per run")
    parser.add_argument("--runs", type=int, default=3, help="Runs per game")
    parser.add_argument("--all", action="store_true", help="All games")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("SAGE ARC-AGI-3 Membot-Integrated Clicker")
    print("=" * 70)

    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in str(e.game_id)]
    elif args.all:
        targets = envs
    else:
        targets = envs[:5]

    print(f"\nTargeting {len(targets)} game(s), {args.runs} runs, {args.steps} steps/run\n")

    all_results = []
    total_levels = 0

    for env_info in targets:
        game_id = env_info.game_id
        print(f"{'─'*70}")
        print(f"Game: {game_id}")

        best_run = None
        for run in range(args.runs):
            try:
                env = arcade.make(game_id)
                frame_data = env.reset()
                result = play_with_membot(env, frame_data, args.steps, game_id, verbose=args.verbose)
            except Exception as e:
                print(f"  Run {run+1}: FAILED ({e})")
                continue

            levels = result["total_levels_gained"]
            total_levels += levels

            status = f" ★ {levels} level(s)!" if levels > 0 else ""
            print(f"  Run {run+1}: {result['steps']} steps, {result['levels_completed']}/{result['win_levels']} levels{status}")

            if best_run is None or result["levels_completed"] > best_run["levels_completed"]:
                best_run = result

        if best_run:
            print(f"  Best run effectiveness:\n{best_run['effectiveness']}")
            all_results.append({"game_id": game_id, **best_run})

    print(f"\n{'='*70}")
    print(f"SUMMARY: {total_levels} total levels")
    print(f"{'='*70}")
    for r in all_results:
        lvl = f"{r.get('levels_completed',0)}/{r.get('win_levels',0)}"
        status = " ★" if r.get('total_levels_gained', 0) > 0 else ""
        print(f"  {r['game_id']:20s}  {lvl}{status}")

    # Save log
    log_path = f"arc-agi-3/experiments/logs/membot_clicker_{int(time.time())}.json"
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
