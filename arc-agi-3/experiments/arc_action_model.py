#!/usr/bin/env python3
"""
ARC-AGI-3 Action-Effect Model Builder

Observes what each action does across multiple executions, builds a
compact model: "action N = moves object X by (dx, dy)" or
"action N = cycles color at cursor position."

This model is then fed to the LLM for structured planning instead
of raw frame diffs.

Usage:
    from arc_action_model import ActionEffectModel
    model = ActionEffectModel()
    model.observe(action, grid_before, grid_after, tracker_diff)
    print(model.describe())  # For LLM prompt
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from arc_perception import background_color, color_name


class ActionEffect:
    """Observed effect of a single action type."""

    def __init__(self, action_id: int):
        self.action_id = action_id
        self.observations = 0
        self.caused_change = 0

        # Movement effects: which objects move, by how much
        self.movements: List[dict] = []  # {color, dx, dy}

        # Pixel change stats
        self.pixel_changes: List[int] = []

        # Color transitions: what colors appear/disappear
        self.color_appeared: defaultdict = defaultdict(int)
        self.color_disappeared: defaultdict = defaultdict(int)

        # Position sensitivity: does effect depend on where we are?
        self.position_dependent = None  # None=unknown, True/False

    def add_observation(self, changed: bool, n_pixels: int,
                        movements: list, colors_before: dict, colors_after: dict):
        self.observations += 1
        if changed:
            self.caused_change += 1
            self.pixel_changes.append(n_pixels)

            for m in movements:
                self.movements.append({
                    "color": m.get("color", "?"),
                    "dx": m.get("dx", 0),
                    "dy": m.get("dy", 0),
                })

            # Track color transitions
            for c in colors_after:
                if c not in colors_before:
                    self.color_appeared[c] += 1
            for c in colors_before:
                if c not in colors_after:
                    self.color_disappeared[c] += 1

    @property
    def change_rate(self) -> float:
        return self.caused_change / max(self.observations, 1)

    @property
    def avg_pixels(self) -> float:
        return sum(self.pixel_changes) / max(len(self.pixel_changes), 1)

    def dominant_movement(self) -> Optional[dict]:
        """Most common movement pattern for this action."""
        if not self.movements:
            return None

        # Group by color and average dx/dy
        by_color = defaultdict(list)
        for m in self.movements:
            by_color[m["color"]].append((m["dx"], m["dy"]))

        best_color = max(by_color, key=lambda c: len(by_color[c]))
        moves = by_color[best_color]
        avg_dx = sum(m[0] for m in moves) / len(moves)
        avg_dy = sum(m[1] for m in moves) / len(moves)

        return {"color": best_color, "dx": round(avg_dx, 1), "dy": round(avg_dy, 1),
                "count": len(moves)}

    def describe(self) -> str:
        """Compact description for LLM."""
        name_map = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                    5: "SELECT", 6: "CLICK", 7: "UNDO"}
        name = name_map.get(self.action_id, f"ACTION{self.action_id}")

        if self.observations == 0:
            return f"{name}: untested"

        if self.change_rate < 0.1:
            return f"{name}: no effect ({self.observations} tries)"

        parts = [f"{name}: {self.change_rate:.0%} effective, ~{self.avg_pixels:.0f}px"]

        dom = self.dominant_movement()
        if dom:
            dx, dy = dom["dx"], dom["dy"]
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
                parts.append(f"moves {dom['color']} {direction} {abs(dx):.0f}px")
            elif abs(dy) > abs(dx):
                direction = "down" if dy > 0 else "up"
                parts.append(f"moves {dom['color']} {direction} {abs(dy):.0f}px")
            elif dx != 0 or dy != 0:
                parts.append(f"moves {dom['color']} ({dx:+.0f},{dy:+.0f})")

        return ", ".join(parts)


class ActionEffectModel:
    """Builds and maintains an action-effect model from observations.

    Observe actions → build model → describe for LLM.
    """

    def __init__(self):
        self.effects: Dict[int, ActionEffect] = {}
        self.total_observations = 0

    def observe(self, action: int, grid_before: np.ndarray, grid_after: np.ndarray,
                spatial_diff: Optional[dict] = None):
        """Record an action and its observed effect."""
        if action not in self.effects:
            self.effects[action] = ActionEffect(action)

        effect = self.effects[action]
        changed = not np.array_equal(grid_before, grid_after)
        n_pixels = int(np.sum(grid_before != grid_after))

        # Extract movements from spatial diff
        movements = spatial_diff.get("movements", []) if spatial_diff else []

        # Color census before/after
        bg = background_color(grid_before)
        from collections import Counter
        colors_before = set(Counter(grid_before.astype(int).flatten()).keys()) - {bg}
        colors_after = set(Counter(grid_after.astype(int).flatten()).keys()) - {bg}

        effect.add_observation(changed, n_pixels, movements,
                               colors_before, colors_after)
        self.total_observations += 1

    def describe(self) -> str:
        """Full model description for LLM prompt."""
        if not self.effects:
            return "No actions observed yet."

        lines = [f"ACTION MODEL ({self.total_observations} observations):"]
        for action_id in sorted(self.effects.keys()):
            lines.append(f"  {self.effects[action_id].describe()}")
        return "\n".join(lines)

    def get_effective_actions(self) -> List[int]:
        """Actions that consistently cause changes."""
        return [a for a, e in self.effects.items()
                if e.change_rate > 0.3 and e.observations >= 2]

    def get_movement_actions(self) -> Dict[int, dict]:
        """Actions that move objects — returns {action: dominant_movement}."""
        result = {}
        for action_id, effect in self.effects.items():
            dom = effect.dominant_movement()
            if dom and (abs(dom["dx"]) > 1 or abs(dom["dy"]) > 1):
                result[action_id] = dom
        return result

    def get_cycling_actions(self) -> List[int]:
        """Actions that change pixels but don't move objects (likely cycling/toggling)."""
        cycling = []
        for action_id, effect in self.effects.items():
            if effect.change_rate > 0.3 and not effect.dominant_movement():
                cycling.append(action_id)
            elif effect.change_rate > 0.3:
                dom = effect.dominant_movement()
                if dom and abs(dom["dx"]) <= 1 and abs(dom["dy"]) <= 1:
                    cycling.append(action_id)
        return cycling

    def infer_game_type(self) -> str:
        """Infer game type from action effects."""
        moves = self.get_movement_actions()
        cycles = self.get_cycling_actions()

        if len(moves) >= 2 and not cycles:
            return "navigation"  # Pure movement game (maze, sokoban)
        elif len(moves) >= 2 and cycles:
            return "cursor_cycle"  # Movement + cycling (pattern rules, conveyor)
        elif cycles and not moves:
            return "toggle"  # Pure toggle/cycle game
        elif 6 in self.effects and self.effects[6].change_rate > 0.3:
            return "click_interact"  # Click-based interaction
        return "unknown"
