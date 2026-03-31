#!/usr/bin/env python3
"""
SAGE ARC-AGI-3 Learner — Learns which actions cause changes, biases toward them.

Inspired by StochasticGoose's approach but without a CNN:
- Hash game states (downsampled grid → MD5)
- Track which actions cause frame changes at similar states
- Bias exploration toward effective actions (softmax on change counts)
- Detect level completion and record winning sequences
- Reset action knowledge per level (new mechanics)

This is the "build the driver" approach: learn the action-effect mapping
through experience, not through reasoning.

Usage:
    cd /Users/dennispalatov/repos/SAGE
    .venv/bin/python3 arc-agi-3/experiments/sage_learner.py --game ls20 --steps 50000
    .venv/bin/python3 arc-agi-3/experiments/sage_learner.py --all --steps 100000
"""

import sys
import time
import json
import hashlib
import argparse
import random
from collections import defaultdict
import numpy as np

sys.path.insert(0, ".")

from arc_agi import Arcade
from arcengine import GameAction

INT_TO_GAME_ACTION = {a.value: a for a in GameAction}
ACTION_LABELS = {
    1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
    5: "SELECT", 6: "ACTION6", 7: "ACTION7",
}


def grid_hash(grid: np.ndarray, downsample: int = 8) -> str:
    """Hash a grid state, downsampled for generalization.

    Downsampling means similar (but not identical) states map to the
    same hash, giving the agent spatial generalization without a CNN.
    """
    # Downsample by taking every Nth pixel
    small = grid[::downsample, ::downsample]
    return hashlib.md5(small.tobytes()).hexdigest()[:12]


def grid_region_hash(grid: np.ndarray) -> str:
    """Hash just the non-background region (position-invariant)."""
    bg = int(np.bincount(grid.flatten()).argmax())
    non_bg = grid[grid != bg]
    if len(non_bg) == 0:
        return "empty"
    colors = tuple(sorted(np.unique(non_bg).tolist()))
    count = len(non_bg)
    return hashlib.md5(f"{colors}:{count}".encode()).hexdigest()[:8]


EXPERIENCE_DIR = "arc-agi-3/experiments/experience"


class ActionLearner:
    """Learns action effectiveness from experience.

    Tracks how often each action causes frame changes.
    Biases action selection toward effective actions using softmax.
    Persists experience to disk so learning accumulates across sessions.
    """

    def __init__(self, available_actions: list, game_id: str = "",
                 exploration_rate: float = 0.3):
        self.available = available_actions
        self.game_id = game_id
        self.exploration_rate = exploration_rate

        # Global action stats
        self.action_tries = defaultdict(int)     # action → total tries
        self.action_changes = defaultdict(int)    # action → tries that caused change

        # State-action stats (keyed by grid hash)
        self.state_action_tries = defaultdict(lambda: defaultdict(int))
        self.state_action_changes = defaultdict(lambda: defaultdict(int))

        # Sequence tracking
        self.recent_actions = []      # last N actions
        self.recent_changed = []      # did each cause change?
        self.effective_sequences = [] # sequences that preceded level-ups

        # Load prior experience
        self._load_experience()

    def select_action(self, state_hash: str) -> int:
        """Select action biased toward ones that cause changes."""
        if random.random() < self.exploration_rate:
            return random.choice(self.available)

        # Compute effectiveness score per action
        scores = {}
        for a in self.available:
            # Blend global and state-local stats
            global_tries = self.action_tries[a]
            global_changes = self.action_changes[a]
            local_tries = self.state_action_tries[state_hash][a]
            local_changes = self.state_action_changes[state_hash][a]

            # Bayesian-ish: prior from global, update from local
            total_tries = global_tries + local_tries + 1  # +1 smoothing
            total_changes = global_changes + local_changes + 0.5  # optimistic prior
            scores[a] = total_changes / total_tries

        # Softmax selection
        max_score = max(scores.values())
        exp_scores = {a: np.exp(3.0 * (s - max_score)) for a, s in scores.items()}
        total = sum(exp_scores.values())
        probs = {a: s / total for a, s in exp_scores.items()}

        r = random.random()
        cumulative = 0
        for a in self.available:
            cumulative += probs[a]
            if r <= cumulative:
                return a
        return self.available[-1]

    def record(self, action: int, state_hash: str, changed: bool):
        self.action_tries[action] += 1
        self.state_action_tries[state_hash][action] += 1
        if changed:
            self.action_changes[action] += 1
            self.state_action_changes[state_hash][action] += 1

        self.recent_actions.append(action)
        self.recent_changed.append(changed)
        if len(self.recent_actions) > 30:
            self.recent_actions = self.recent_actions[-30:]
            self.recent_changed = self.recent_changed[-30:]

    def record_level_up(self):
        """Record the recent action sequence as a winning sequence."""
        window = list(zip(self.recent_actions[-15:], self.recent_changed[-15:]))
        self.effective_sequences.append(window)

    def reset_for_new_level(self):
        """Reset state-local stats for new level (new mechanics)."""
        self.state_action_tries.clear()
        self.state_action_changes.clear()
        # Keep global stats — some actions are universally useful

    def _experience_path(self) -> str:
        import os
        os.makedirs(EXPERIENCE_DIR, exist_ok=True)
        # Strip version suffix from game_id for persistence (e.g. ls20-9607627b → ls20)
        base_game = self.game_id.split("-")[0] if self.game_id else "unknown"
        return os.path.join(EXPERIENCE_DIR, f"{base_game}.json")

    def _load_experience(self):
        """Load prior experience from disk."""
        path = self._experience_path()
        try:
            with open(path) as f:
                data = json.load(f)
            for a_str, count in data.get("action_tries", {}).items():
                self.action_tries[int(a_str)] += count
            for a_str, count in data.get("action_changes", {}).items():
                self.action_changes[int(a_str)] += count
            self.effective_sequences = data.get("effective_sequences", [])
            prior_steps = sum(self.action_tries.values())
            if prior_steps > 0:
                print(f"    Loaded {prior_steps} prior steps from {path}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"    Warning: failed to load experience: {e}")

    def save_experience(self):
        """Save accumulated experience to disk."""
        path = self._experience_path()
        data = {
            "game_id": self.game_id,
            "action_tries": {str(k): v for k, v in self.action_tries.items()},
            "action_changes": {str(k): v for k, v in self.action_changes.items()},
            "effective_sequences": self.effective_sequences[-20:],  # keep last 20
            "saved_at": time.time(),
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"    Warning: failed to save experience: {e}")

    def effectiveness_report(self) -> str:
        lines = []
        for a in sorted(self.available):
            tries = self.action_tries[a]
            changes = self.action_changes[a]
            rate = changes / max(tries, 1)
            name = ACTION_LABELS.get(a, f"A{a}")
            lines.append(f"  {name}: {changes}/{tries} = {rate:.1%} effective")
        return "\n".join(lines)


def play_game(env, frame_data, max_steps: int, game_id: str = "", verbose: bool = False) -> dict:
    """Play one game with learning-biased exploration."""
    available = [a.value if hasattr(a, "value") else int(a) for a in (frame_data.available_actions or [])]
    if not available:
        return {"steps": 0, "levels": 0, "win_levels": 0}

    grid = np.array(frame_data.frame)
    if grid.ndim == 3:
        grid = grid[-1]

    learner = ActionLearner(available, game_id=game_id, exploration_rate=0.3)
    start_levels = frame_data.levels_completed
    prev_levels = start_levels
    level_events = []
    states_seen = set()

    # Decay exploration over time
    for step in range(max_steps):
        # Adaptive exploration: decrease over time
        if step > 1000:
            learner.exploration_rate = max(0.05, 0.3 - (step / max_steps) * 0.25)

        state_h = grid_hash(grid)
        states_seen.add(state_h)

        action = learner.select_action(state_h)

        prev_grid = grid.copy()
        try:
            frame_data = env.step(INT_TO_GAME_ACTION[action])
        except Exception:
            continue

        if frame_data is None:
            break

        new_grid = np.array(frame_data.frame)
        if new_grid.ndim == 3:
            new_grid = new_grid[-1]
        if new_grid.size == 0 or new_grid.ndim != 2:
            continue

        changed = not np.array_equal(prev_grid, new_grid)
        grid = new_grid

        learner.record(action, state_h, changed)

        # Update available actions
        new_avail = [a.value if hasattr(a, "value") else int(a) for a in (frame_data.available_actions or [])]
        if new_avail and set(new_avail) != set(available):
            available = new_avail
            learner.available = available

        # Level completion
        if frame_data.levels_completed > prev_levels:
            learner.record_level_up()
            window_actions = learner.recent_actions[-15:]
            window_names = [ACTION_LABELS.get(a, f"A{a}") for a in window_actions]
            level_events.append({
                "step": step,
                "levels_before": prev_levels,
                "levels_after": frame_data.levels_completed,
                "action_window": window_names,
            })
            if verbose:
                print(f"    ★ LEVEL {frame_data.levels_completed}/{frame_data.win_levels} at step {step}! [{' → '.join(window_names[-10:])}]")
            prev_levels = frame_data.levels_completed
            learner.reset_for_new_level()

        if frame_data.state.name in ("WON", "LOST"):
            break

        # Periodic verbose output
        if verbose and step > 0 and step % 10000 == 0:
            print(f"    Step {step}: levels={frame_data.levels_completed}/{frame_data.win_levels}, "
                  f"states_seen={len(states_seen)}, explore={learner.exploration_rate:.2f}")

    # Persist learning
    learner.save_experience()

    return {
        "steps": step + 1 if 'step' in dir() else 0,
        "levels_completed": getattr(frame_data, 'levels_completed', 0),
        "win_levels": getattr(frame_data, 'win_levels', 0),
        "state": frame_data.state.name if hasattr(frame_data, 'state') else "UNKNOWN",
        "level_events": level_events,
        "total_levels_gained": getattr(frame_data, 'levels_completed', 0) - start_levels,
        "states_seen": len(states_seen),
        "effectiveness": learner.effectiveness_report(),
        "winning_sequences": learner.effective_sequences,
    }


def main():
    parser = argparse.ArgumentParser(description="SAGE ARC-AGI-3 Learner")
    parser.add_argument("--game", default=None, help="Game ID prefix")
    parser.add_argument("--steps", type=int, default=50000, help="Steps per run")
    parser.add_argument("--runs", type=int, default=3, help="Runs per game")
    parser.add_argument("--all", action="store_true", help="All games")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("SAGE ARC-AGI-3 Learner — Action-Effect Learning")
    print("=" * 70)

    arcade = Arcade()
    envs = arcade.get_environments()

    if args.game:
        targets = [e for e in envs if args.game in (e.game_id if hasattr(e, "game_id") else str(e))]
    elif args.all:
        targets = envs
    else:
        targets = envs[:5]

    print(f"\nTargeting {len(targets)} game(s), {args.runs} runs, {args.steps} steps/run\n")

    all_results = []
    total_levels = 0
    start_time = time.time()

    for env_info in targets:
        game_id = env_info.game_id if hasattr(env_info, "game_id") else str(env_info)
        print(f"{'─'*70}")
        print(f"Game: {game_id}")

        best_run = None
        for run in range(args.runs):
            try:
                env = arcade.make(game_id)
                frame_data = env.reset()
            except Exception as e:
                print(f"  Run {run+1}: FAILED ({e})")
                continue

            t0 = time.time()
            try:
                result = play_game(env, frame_data, args.steps, game_id=game_id, verbose=args.verbose)
            except Exception as e:
                print(f"  Run {run+1}: CRASHED ({e})")
                continue
            elapsed = time.time() - t0

            levels = result["total_levels_gained"]
            total_levels += levels
            rate = result["steps"] / max(elapsed, 0.001)

            status = f" ★ {levels} level(s)!" if levels > 0 else ""
            if result["state"] == "WON":
                status += " 🏆"
            print(f"  Run {run+1}: {result['steps']} steps ({rate:.0f}/s), "
                  f"{result['levels_completed']}/{result['win_levels']} levels, "
                  f"{result['states_seen']} states{status}")

            if best_run is None or result["levels_completed"] > best_run["levels_completed"]:
                best_run = result

        if best_run:
            print(f"  Action effectiveness:\n{best_run['effectiveness']}")
            all_results.append({"game_id": game_id, **best_run})
        else:
            all_results.append({"game_id": game_id, "levels_completed": 0, "win_levels": 0})

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"SUMMARY: {total_levels} total levels in {elapsed:.0f}s")
    print(f"{'='*70}")
    for r in all_results:
        lvl = f"{r.get('levels_completed',0)}/{r.get('win_levels',0)}"
        status = " ★" if r.get('total_levels_gained', 0) > 0 else ""
        print(f"  {r['game_id']:20s}  {lvl}{status}")

    # Save
    log_path = f"arc-agi-3/experiments/logs/learner_{int(time.time())}.json"
    try:
        import os
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump({"results": all_results, "total_levels": total_levels, "elapsed": elapsed}, f, indent=2, default=str)
        print(f"\n  Saved: {log_path}")
    except Exception as e:
        print(f"  (save failed: {e})")


if __name__ == "__main__":
    main()
