#!/usr/bin/env python3
"""
SAGE Navigator — LLM reflection for strategic guidance.

The Navigator runs periodically (every 10-20 actions) to:
- Assess progress toward goal
- Generate strategic hypothesis
- Bias action selection toward promising patterns
- Query membot for similar patterns (future)

Time budget: <5s per reflection to maintain throughput.
"""

import json
import requests
import time
import numpy as np
from typing import Dict, List, Optional

OLLAMA_URL = "http://localhost:11434/api/generate"

ACTION_LABELS = {
    1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
    5: "SELECT", 6: "ACTION6", 7: "ACTION7",
}
ACTION_NAMES_TO_INT = {v: k for k, v in ACTION_LABELS.items()}


class SageNavigator:
    """LLM-based strategic reflection for grid puzzle games.

    The Navigator provides high-level guidance to the Driver:
    - Analyzes recent actions and grid changes
    - Forms hypotheses about game mechanics
    - Biases action selection toward promising patterns
    - Learns from level completions
    """

    def __init__(self, model: str = "phi4:14b", timeout: float = 10.0):
        """
        Args:
            model: Ollama model name (phi4:14b or gemma3:12b)
            timeout: Max time for reflection (seconds)
        """
        self.model = model
        self.timeout = timeout
        self.total_reflections = 0
        self.total_time = 0.0
        self.last_hypothesis = ""
        self.last_confidence = 0.0

    def reflect(self, driver_history: dict, current_grid: np.ndarray,
                levels_completed: int, win_levels: int,
                cartridge_insights: List[str] = None) -> dict:
        """Perform strategic reflection on recent gameplay.

        Args:
            driver_history: Recent actions/grids from Driver.get_recent_history()
            current_grid: Current grid state
            levels_completed: Current level count
            win_levels: Total levels in game
            cartridge_insights: Prior strategic insights from membot

        Returns:
            {
                "hypothesis": "pattern description",
                "action_bias": {action_int: score},
                "confidence": 0.0-1.0,
                "strategy": "exploration" | "exploitation",
                "reflection_time": seconds
            }
        """
        t0 = time.time()

        # Build reflection prompt
        prompt = self._build_prompt(
            driver_history, current_grid,
            levels_completed, win_levels,
            cartridge_insights
        )

        # Call LLM
        try:
            response = self._call_ollama(prompt)
            parsed = self._parse_response(response, driver_history["actions"])
        except Exception as e:
            print(f"    Navigator: Reflection failed ({e}), using fallback")
            parsed = self._fallback_response(driver_history["actions"])

        elapsed = time.time() - t0
        self.total_reflections += 1
        self.total_time += elapsed

        parsed["reflection_time"] = elapsed
        self.last_hypothesis = parsed["hypothesis"]
        self.last_confidence = parsed["confidence"]

        return parsed

    def _build_prompt(self, history: dict, grid: np.ndarray,
                     levels: int, win_levels: int,
                     insights: List[str] = None) -> str:
        """Build concise reflection prompt."""
        # Grid representation (ASCII, compact)
        grid_str = self._grid_to_ascii(grid)

        # Recent actions summary
        actions = history["action_names"][-10:]
        changed = history["changed"][-10:]
        action_summary = ", ".join([
            f"{a}{'✓' if c else '✗'}"
            for a, c in zip(actions, changed)
        ])

        # Prior insights (if any)
        insights_str = ""
        if insights:
            insights_str = "\n\nPrior insights:\n" + "\n".join(f"- {i}" for i in insights[-3:])

        prompt = f"""You are analyzing a grid puzzle game.

Progress: Level {levels}/{win_levels}

Recent actions (✓=changed grid, ✗=no change):
{action_summary}

Current grid:
{grid_str}
{insights_str}

Goal: Complete the level (levels_completed will increase when successful).

Question: What pattern do you notice? Which actions should we try next?

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "hypothesis": "brief pattern description (1 sentence)",
  "action_bias": {{"UP": 0.3, "SELECT": 0.5, ...}},
  "confidence": 0.7,
  "strategy": "exploration"
}}

Notes:
- action_bias values should sum to ~1.0
- confidence: 0.0 (guessing) to 1.0 (certain)
- strategy: "exploration" (try new things) or "exploitation" (repeat what works)
"""

        return prompt

    def _grid_to_ascii(self, grid: np.ndarray, max_size: int = 20) -> str:
        """Convert grid to compact ASCII representation."""
        if grid.size == 0:
            return "[empty grid]"

        h, w = grid.shape
        if h > max_size or w > max_size:
            # Downsample large grids
            grid = grid[:max_size, :max_size]
            h, w = grid.shape

        # Map colors to characters
        chars = " .,:;+*#@"
        lines = []
        for row in grid:
            line = ""
            for val in row:
                idx = min(int(val), len(chars) - 1)
                line += chars[idx]
            lines.append(line)

        return "\n".join(lines)

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API with timeout."""
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more consistent JSON
                        "num_predict": 200,  # Limit response length
                    }
                },
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama call failed: {e}")

    def _parse_response(self, response: str, available_actions: List[int]) -> dict:
        """Parse LLM JSON response into action biases."""
        # Strip markdown code fences if present
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if len(lines) > 2 else lines[1:])
            response = response.replace("```json", "").replace("```", "").strip()

        # Parse JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON parse failed: {e}\nResponse: {response[:200]}")

        # Extract fields
        hypothesis = data.get("hypothesis", "No pattern identified")
        confidence = float(data.get("confidence", 0.5))
        strategy = data.get("strategy", "exploration")
        raw_bias = data.get("action_bias", {})

        # Convert action names to integers
        action_bias = {}
        for name, score in raw_bias.items():
            action_int = ACTION_NAMES_TO_INT.get(name.upper())
            if action_int in available_actions:
                action_bias[action_int] = float(score)

        # Normalize to sum to 1.0
        total = sum(action_bias.values())
        if total > 0:
            action_bias = {a: s / total for a, s in action_bias.items()}
        else:
            # Fallback: uniform distribution
            action_bias = {a: 1.0 / len(available_actions) for a in available_actions}

        return {
            "hypothesis": hypothesis,
            "action_bias": action_bias,
            "confidence": confidence,
            "strategy": strategy
        }

    def _fallback_response(self, available_actions: List[int]) -> dict:
        """Fallback response when LLM fails."""
        return {
            "hypothesis": "Exploration mode (reflection failed)",
            "action_bias": {a: 1.0 / len(available_actions) for a in available_actions},
            "confidence": 0.0,
            "strategy": "exploration"
        }

    def average_reflection_time(self) -> float:
        """Get average reflection time across all calls."""
        if self.total_reflections == 0:
            return 0.0
        return self.total_time / self.total_reflections

    def get_initial_bias(self, cartridge: dict, available_actions: List[int]) -> dict:
        """Get initial action bias from cartridge before first reflection.

        Args:
            cartridge: Membot cartridge with action_effectiveness
            available_actions: List of available action integers

        Returns:
            Action bias dict {action: score}
        """
        if not cartridge:
            return {a: 1.0 / len(available_actions) for a in available_actions}

        effectiveness = cartridge.get("action_effectiveness", {}).get("global", {})
        if not effectiveness:
            return {a: 1.0 / len(available_actions) for a in available_actions}

        # Convert action names to integers and normalize
        bias = {}
        for action_name, score in effectiveness.items():
            action_int = ACTION_NAMES_TO_INT.get(action_name)
            if action_int in available_actions:
                bias[action_int] = score

        # Normalize
        total = sum(bias.values())
        if total > 0:
            bias = {a: s / total for a, s in bias.items()}
        else:
            bias = {a: 1.0 / len(available_actions) for a in available_actions}

        return bias


def test_navigator():
    """Quick test of Navigator with a sample grid."""
    print("Testing Navigator...")

    nav = SageNavigator(model="gemma3:12b", timeout=15.0)

    # Mock driver history
    grid = np.array([
        [0, 0, 0, 14, 14],
        [0, 0, 0, 14, 14],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
    ])

    history = {
        "actions": [1, 1, 6, 6],
        "action_names": ["UP", "UP", "ACTION6", "ACTION6"],
        "changed": [True, True, True, False],
        "grids": []
    }

    result = nav.reflect(
        driver_history=history,
        current_grid=grid,
        levels_completed=0,
        win_levels=6,
        cartridge_insights=["Movement actions are very effective"]
    )

    print(f"\nReflection result:")
    print(f"  Hypothesis: {result['hypothesis']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Strategy: {result['strategy']}")
    print(f"  Action bias: {result['action_bias']}")
    print(f"  Time: {result['reflection_time']:.2f}s")


if __name__ == "__main__":
    test_navigator()
