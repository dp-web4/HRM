"""
navigate_to — move primary entity to target (x, y).

The most-used motor skill across ARC-AGI-3 games. Computes direction
toward target each step, emits the appropriate movement action, halts
when position matches target, detects stuck via repeated position.

Params:
    x: int — target x coordinate
    y: int — target y coordinate

Games: bp35, lf52, wa30, m0r0, ls20, ka59, sc25, g50t, tr87, tu93
"""

from typing import Any, Dict, List

from sage.cognition.motor_skills.types import Observation, Skill
from sage.cognition.motor_skills.registry import register_skill


class NavigateToSkill:
    """Navigate primary entity to target position."""

    goal_type = "navigate_to"

    # Action mapping: direction → action ID
    # Standard ARC-AGI-3: UP=1, DOWN=2, LEFT=3, RIGHT=4
    ACTIONS = {
        'UP': 1,
        'DOWN': 2,
        'LEFT': 3,
        'RIGHT': 4,
    }

    def step(self, obs: Observation, params: Dict[str, Any]) -> int:
        """Return movement action toward target."""
        if obs.position is None:
            return self.ACTIONS['UP']  # default: try moving

        cx, cy = obs.position
        tx, ty = params['x'], params['y']

        dx = tx - cx
        dy = ty - cy

        # Prefer larger delta axis first
        if abs(dx) >= abs(dy):
            if dx > 0:
                return self.ACTIONS['RIGHT']
            elif dx < 0:
                return self.ACTIONS['LEFT']
        if dy > 0:
            return self.ACTIONS['DOWN']
        elif dy < 0:
            return self.ACTIONS['UP']

        # Already at target (shouldn't reach here if halt_condition works)
        return self.ACTIONS['UP']

    def halt_condition(self, obs: Observation, params: Dict[str, Any]) -> bool:
        """True when entity is at target position."""
        if obs.position is None:
            return False
        return obs.position[0] == params['x'] and obs.position[1] == params['y']

    def stuck_condition(
        self, obs: Observation, params: Dict[str, Any], recent: List[Observation]
    ) -> bool:
        """True when position hasn't changed across recent observations."""
        if len(recent) < 2:
            return False
        positions = [o.position for o in recent if o.position is not None]
        if len(positions) < 2:
            return False
        # Stuck if all recent positions are identical
        return all(p == positions[0] for p in positions)

    def progress_metric(self, obs: Observation, params: Dict[str, Any]) -> float:
        """Manhattan distance to target, normalized to [0, 1].

        Returns 1.0 when at target, 0.0 when far away.
        """
        if obs.position is None:
            return 0.0
        cx, cy = obs.position
        tx, ty = params['x'], params['y']
        dist = abs(tx - cx) + abs(ty - cy)
        # Normalize: assume max board is 64x64, max dist ~128
        return max(0.0, 1.0 - dist / 128.0)


# Register on import
_instance = NavigateToSkill()
register_skill(_instance)
