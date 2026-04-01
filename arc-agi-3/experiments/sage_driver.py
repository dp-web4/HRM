#!/usr/bin/env python3
"""
SAGE Driver — Fast action loop with dual biasing (learned + LLM).

Extends sage_learner.py with Navigator LLM guidance integration.
Maintains fast action selection (<1ms) while respecting strategic biases.
"""

import sys
import json
import hashlib
import random
from collections import defaultdict, deque
import numpy as np

sys.path.insert(0, ".")

ACTION_LABELS = {
    1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
    5: "SELECT", 6: "ACTION6", 7: "ACTION7",
}


def grid_hash(grid: np.ndarray, downsample: int = 8) -> str:
    """Hash a grid state, downsampled for generalization."""
    if grid.size == 0:
        return "empty"
    small = grid[::downsample, ::downsample]
    return hashlib.md5(small.tobytes()).hexdigest()[:12]


EXPERIENCE_DIR = "arc-agi-3/experiments/experience"


class SageDriver:
    """Fast action selection with dual biasing: learned effectiveness + LLM guidance.

    The Driver runs the inner loop:
    - Select actions in <1ms
    - Track state→action→outcome
    - Blend learned patterns with Navigator's strategic guidance
    - Persist experience to disk for cross-session learning

    The Navigator (LLM) provides strategic biases every N actions.
    """

    def __init__(self, available_actions: list, game_id: str = "",
                 exploration_rate: float = 0.3, blend_weight: float = 0.6):
        """
        Args:
            available_actions: List of action integers
            game_id: Game identifier for experience persistence
            exploration_rate: Probability of random action (decays over time)
            blend_weight: Weight for learned effectiveness (vs LLM bias)
                         0.6 = 60% learned, 40% LLM
        """
        self.available = available_actions
        self.game_id = game_id
        self.exploration_rate = exploration_rate
        self.blend_weight = blend_weight

        # Effectiveness tracking (same as sage_learner.py)
        self.action_tries = defaultdict(int)
        self.action_changes = defaultdict(int)
        self.state_action_tries = defaultdict(lambda: defaultdict(int))
        self.state_action_changes = defaultdict(lambda: defaultdict(int))

        # Recent history for Navigator reflection (deque for efficiency)
        self.recent_actions = deque(maxlen=30)
        self.recent_changed = deque(maxlen=30)
        self.recent_grids = deque(maxlen=30)  # Store grid snapshots

        # LLM bias from Navigator (updated periodically)
        self.llm_bias = None  # {action: score} dict from Navigator

        # Winning sequences
        self.effective_sequences = []

        # Load prior experience
        self._load_experience()

    def select_action(self, state_hash: str, grid: np.ndarray = None):
        """Select action blending learned effectiveness + LLM bias.

        Args:
            state_hash: Hash of current grid state
            grid: Current grid (for storage in recent history)

        Returns:
            int for simple actions, or (int, dict) for coordinate actions.
            e.g. 1 for UP, or (6, {'x': col, 'y': row}) for ACTION6 click.
        """
        # Store grid for Navigator reflection
        if grid is not None:
            self.recent_grids.append(grid.copy())

        # Exploration: random action
        if random.random() < self.exploration_rate:
            return self._with_coordinates(random.choice(self.available), grid)

        # Exploitation: blend learned + LLM signals
        scores = {}
        for a in self.available:
            # Learned effectiveness (Bayesian blend of global + local)
            global_tries = self.action_tries[a]
            global_changes = self.action_changes[a]
            local_tries = self.state_action_tries[state_hash][a]
            local_changes = self.state_action_changes[state_hash][a]

            total_tries = global_tries + local_tries + 1
            total_changes = global_changes + local_changes + 0.5
            learned_score = total_changes / total_tries

            # LLM bias (Navigator guidance)
            if self.llm_bias:
                llm_score = self.llm_bias.get(a, 0.5)
            else:
                llm_score = 0.5  # neutral if no Navigator guidance

            # Blend: 60% learned, 40% LLM (configurable)
            scores[a] = (self.blend_weight * learned_score +
                        (1 - self.blend_weight) * llm_score)

        # Softmax selection (temperature=3.0 for reasonable exploration)
        max_score = max(scores.values())
        exp_scores = {a: np.exp(3.0 * (s - max_score)) for a, s in scores.items()}
        total = sum(exp_scores.values())
        probs = {a: s / total for a, s in exp_scores.items()}

        r = random.random()
        cumulative = 0
        selected = self.available[-1]
        for a in self.available:
            cumulative += probs[a]
            if r <= cumulative:
                selected = a
                break
        return self._with_coordinates(selected, grid)

    def set_color_preferences(self, effective: list = None, ineffective: list = None):
        """Set color targeting preferences from cartridge/experience data.

        Args:
            effective: Colors known to cause changes (prioritized)
            ineffective: Colors known to NOT cause changes (avoided)
        """
        self.preferred_colors = set(effective or [])
        self.avoided_colors = set(ineffective or [])

    def _with_coordinates(self, action: int, grid: np.ndarray = None):
        """Wrap ACTION6 with x,y coordinates targeting a cell.

        Targeting priority:
        1. Preferred colors (from cartridge/experience)
        2. Any non-background color not in avoided set
        3. Random non-background cell
        4. Grid center (fallback)

        Returns:
            int for simple actions (1-5, 7)
            (6, {'x': col, 'y': row}) for ACTION6 coordinate click
        """
        if action != 6:
            return action

        # Validate grid
        if grid is None or grid.size == 0 or grid.ndim not in (2, 3):
            return action  # Return bare action if grid invalid

        # Handle 3D grids
        if grid.ndim == 3:
            grid = grid[:, :, -1]

        try:
            bg = int(np.bincount(grid.astype(int).flatten()).argmax())
        except (TypeError, ValueError):
            bg = 0

        # Priority 1: preferred colors
        if hasattr(self, 'preferred_colors') and self.preferred_colors:
            for color in self.preferred_colors:
                cells = np.argwhere(grid == color)
                if len(cells) > 0:
                    r, c = random.choice(cells.tolist())
                    return (6, {'x': int(c), 'y': int(r)})

        # Priority 2: non-background, non-avoided
        avoided = getattr(self, 'avoided_colors', set())
        non_bg = np.argwhere(grid != bg)
        if len(non_bg) > 0 and avoided:
            good_cells = [(r, c) for r, c in non_bg.tolist()
                          if int(grid[r, c]) not in avoided]
            if good_cells:
                r, c = random.choice(good_cells)
                return (6, {'x': int(c), 'y': int(r)})

        # Priority 3: any non-background
        if len(non_bg) > 0:
            r, c = random.choice(non_bg.tolist())
            return (6, {'x': int(c), 'y': int(r)})

        # Fallback: center
        r, c = grid.shape[0] // 2, grid.shape[1] // 2
        return (6, {'x': int(c), 'y': int(r)})

    def record(self, action: int, state_hash: str, changed: bool):
        """Record action outcome for learning."""
        self.action_tries[action] += 1
        self.state_action_tries[state_hash][action] += 1
        if changed:
            self.action_changes[action] += 1
            self.state_action_changes[state_hash][action] += 1

        self.recent_actions.append(action)
        self.recent_changed.append(changed)

    def update_llm_bias(self, bias: dict):
        """Update LLM bias from Navigator reflection.

        Args:
            bias: {action: score} dict, scores sum to ~1.0
        """
        self.llm_bias = bias

    def record_level_up(self):
        """Record recent actions as a winning sequence."""
        window = list(zip(
            list(self.recent_actions)[-15:],
            list(self.recent_changed)[-15:]
        ))
        self.effective_sequences.append(window)

    def reset_for_new_level(self):
        """Reset state-local stats for new level mechanics."""
        self.state_action_tries.clear()
        self.state_action_changes.clear()
        # Keep global stats and LLM bias

    def get_recent_history(self, n: int = 10) -> dict:
        """Get recent actions + grids for Navigator reflection.

        Args:
            n: Number of recent steps to return

        Returns:
            {
                "actions": [action_ints],
                "action_names": [action_name_strings],
                "changed": [bool],
                "grids": [numpy arrays]
            }
        """
        n = min(n, len(self.recent_actions))
        return {
            "actions": list(self.recent_actions)[-n:],
            "action_names": [ACTION_LABELS.get(a, f"A{a}")
                            for a in list(self.recent_actions)[-n:]],
            "changed": list(self.recent_changed)[-n:],
            "grids": list(self.recent_grids)[-n:] if self.recent_grids else []
        }

    def _experience_path(self) -> str:
        import os
        os.makedirs(EXPERIENCE_DIR, exist_ok=True)
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
                print(f"    Driver: Loaded {prior_steps} prior steps from {path}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"    Driver: Warning - failed to load experience: {e}")

    def save_experience(self):
        """Save accumulated experience to disk."""
        path = self._experience_path()
        import time
        data = {
            "game_id": self.game_id,
            "action_tries": {str(k): v for k, v in self.action_tries.items()},
            "action_changes": {str(k): v for k, v in self.action_changes.items()},
            "effective_sequences": self.effective_sequences[-20:],
            "saved_at": time.time(),
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"    Driver: Warning - failed to save experience: {e}")

    def load_from_cartridge(self, cartridge: dict):
        """Load prior knowledge from membot cartridge.

        Args:
            cartridge: Membot cartridge dict with action_effectiveness
        """
        if not cartridge:
            return

        effectiveness = cartridge.get("action_effectiveness", {})

        # Load global action stats
        global_stats = effectiveness.get("global", {})
        for action_name, score in global_stats.items():
            # Convert "UP" -> 1, etc.
            action_int = next((k for k, v in ACTION_LABELS.items() if v == action_name), None)
            if action_int:
                # Bootstrap tries/changes from effectiveness score
                tries = 10  # Bootstrap with 10 virtual tries
                changes = int(score * tries)
                self.action_tries[action_int] += tries
                self.action_changes[action_int] += changes

        # Load state-dependent stats
        state_stats = effectiveness.get("state_dependent", {})
        for state_hash, actions in state_stats.items():
            for action_name, score in actions.items():
                action_int = next((k for k, v in ACTION_LABELS.items() if v == action_name), None)
                if action_int:
                    tries = 5
                    changes = int(score * tries)
                    self.state_action_tries[state_hash][action_int] += tries
                    self.state_action_changes[state_hash][action_int] += changes

        print(f"    Driver: Loaded cartridge knowledge for {len(global_stats)} actions")

    def effectiveness_report(self) -> str:
        """Generate effectiveness report for logging."""
        lines = []
        for a in sorted(self.available):
            tries = self.action_tries[a]
            changes = self.action_changes[a]
            rate = changes / max(tries, 1)
            name = ACTION_LABELS.get(a, f"A{a}")
            bias = f" (LLM bias: {self.llm_bias.get(a, 0.5):.2f})" if self.llm_bias else ""
            lines.append(f"  {name}: {changes}/{tries} = {rate:.1%} effective{bias}")
        return "\n".join(lines)
