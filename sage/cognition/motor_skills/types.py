"""
Motor skills data types.

Observation: what the skill sees each step (position, frame diff, game state).
Skill: protocol that every registered skill implements.
SkillInvocation: request to execute a skill with params.
SkillResult: outcome of a skill execution.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import time


@dataclass
class Observation:
    """What a skill sees each step.

    Intentionally minimal — skills consume pre-computed state,
    not raw frames. Domain-specific fields go in `extra`.
    """
    position: Optional[tuple] = None    # (x, y) of primary entity
    frame_hash: Optional[int] = None    # hash of current frame (for change detection)
    level: int = 0                      # current level
    levels_completed: int = 0           # for win detection
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillInvocation:
    """Request to execute a skill."""
    skill_id: str
    params: Dict[str, Any]
    origin: str = "router"              # "router" | "plugin" | "test"
    max_steps: int = 50
    max_stuck: int = 5


@dataclass
class SkillResult:
    """Outcome of a skill execution."""
    skill_id: str
    status: str                         # "halted" | "stuck" | "max_steps" | "error"
    steps_taken: int
    pre_state: Dict[str, Any]
    post_state: Dict[str, Any]
    final_obs: Optional[Observation] = None
    reward: Optional[float] = None
    error: Optional[str] = None
    elapsed: float = 0.0


@runtime_checkable
class Skill(Protocol):
    """Interface every registered skill implements."""

    goal_type: str

    def step(self, obs: Observation, params: Dict[str, Any]) -> Any:
        """Return the next effector action given current observation + goal params."""
        ...

    def halt_condition(self, obs: Observation, params: Dict[str, Any]) -> bool:
        """True when the goal has been achieved."""
        ...

    def stuck_condition(
        self, obs: Observation, params: Dict[str, Any], recent: List[Observation]
    ) -> bool:
        """True when the skill detects it's not making progress."""
        ...

    def progress_metric(self, obs: Observation, params: Dict[str, Any]) -> float:
        """Scalar in [0,1]: how close to the goal."""
        ...
