"""TaskBlock — what am I doing RIGHT NOW in this invocation.

Irreducible (priority 100). Task is why the invocation is happening
at all — if the model doesn't know the task, nothing else matters.

Sprint: Phase 5 B8
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from sage.context.mrh.base import MRHBlock


@dataclass
class TaskBlock(MRHBlock):
    """The current moment's task description.

    Attributes:
        goal: one-sentence goal for this invocation (e.g.,
            "Pick the best next action on toy_b L0").
        invoke_reasons: why the LLM is being consulted right now
            (e.g., ["stuck", "novelty"]).
        step_index: current step/turn number in the session.
        level: current level (if applicable).
        game_family: game slug for gameplay tasks.
    """
    priority: int = 100
    goal: str = ""
    # `description` is an alias for `goal` (Sprout's raising usage —
    # e.g. `description="Raising session N — phase: creating"`). Same field.
    description: str = ""
    invoke_reasons: List[str] = field(default_factory=list)
    step_index: Optional[int] = None
    level: Optional[int] = None
    game_family: str = ""
    # Level-specific hint (bbox, stride, step count for click games, etc).
    # Placed in task rather than effectors because it changes per-level;
    # the system prompt should stay stable across level transitions for
    # conversation-cache friendliness (McNugget Q7: effector-context in
    # spirit, user-turn in plumbing).
    level_hint: str = ""

    def __post_init__(self) -> None:
        if self.description and not self.goal:
            self.goal = self.description
        self.kind = "task"

    def render(self, budget_tokens: int) -> str:
        parts: List[str] = ["## Task"]
        if self.game_family:
            header = f"Game: {self.game_family}"
            if self.level is not None:
                header += f" | Level: {self.level}"
            if self.step_index is not None:
                header += f" | Step: {self.step_index}"
            parts.append(header)

        if self.goal:
            parts.append(self.goal)

        if self.invoke_reasons:
            parts.append(f"Invoked because: {', '.join(self.invoke_reasons)}.")

        if self.level_hint:
            parts.append(f"Level hint: {self.level_hint}")

        if len(parts) == 1:
            parts.append("(no task specified)")
        return "\n".join(parts)

    def summarize(self, budget_tokens: int) -> str:
        bits: List[str] = []
        if self.game_family:
            bits.append(self.game_family)
        if self.level is not None:
            bits.append(f"L{self.level}")
        if self.step_index is not None:
            bits.append(f"step {self.step_index}")
        if self.invoke_reasons:
            bits.append(f"triggers: {','.join(self.invoke_reasons)}")
        return "Task: " + " | ".join(bits) if bits else "Task: (unspecified)"

    def minimal(self) -> str:
        return self.summarize(budget_tokens=200)

    def estimate_cost(self) -> int:
        base = 24
        base += len(self.goal) // 4
        base += sum(len(r) // 4 + 1 for r in self.invoke_reasons)
        return base
