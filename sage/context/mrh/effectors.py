"""EffectorsBlock — what the model can do this invocation.

Irreducible (priority 90). Without effectors there's no action space
to reason within.

Per McNugget (Phase 5 consensus): habits and motor skills live here
as FIELDS, not as separate blocks. A habit is a pre-compiled effector
sequence; a skill is a named action affordance. Both are effector
capabilities, not epistemic categories.

Sprint: Phase 5 B4
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


from sage.context.mrh.base import MRHBlock


@dataclass
class HabitMatch:
    """A cached action sequence that matches the current state."""
    habit_id: str
    confidence: float                    # 0..1 match confidence
    reliability: float = 1.0             # prior success rate
    sequence: List[str] = field(default_factory=list)  # human-readable action names
    source_count: int = 0                # successful replays


@dataclass
class MotorSkill:
    """A named action affordance the model can invoke by name."""
    name: str
    signature: str          # how it's called, e.g., "navigate_to X=<int> Y=<int>"
    description: str = ""


_ACTION_SURFACES: Dict[str, str] = {
    "game_actions": (
        "Available actions: A0 / UP / DOWN / LEFT / RIGHT / SEL / CLICK(x,y).\n"
        "CLICK requires X and Y coords in [0, 63]. The other six act "
        "without coords."
    ),
    "text": (
        "Respond with a single text turn. Keep it relevant to the "
        "conversation; partnership matters."
    ),
    "text+tools": (
        "Respond with a single text turn. You may invoke tools using "
        "the grammar provided with the tool list (when tools are "
        "attached)."
    ),
    "approval_vote": (
        "Respond with one of: approve / reject / request-clarification. "
        "If reject, give a one-sentence reason."
    ),
}


@dataclass
class EffectorsBlock(MRHBlock):
    """Action surface for the invocation.

    Attributes:
        kind_profile: which action surface (game_actions / text / text+tools /
                       approval_vote). Determines the base rendering.
        habit_match: optional cached habit matching the current state
                     (McNugget's cerebellum output).
        motor_skills: list of named skill affordances (McNugget's motor tier).
        level_annotation: optional per-level effector context (e.g., bbox +
                          stride for click games — McNugget's addition).
        response_format: canonical response format string for the LLM
                         (ACTION=<n> [X=<x> Y=<y>]). Kept with effectors
                         since it describes the action-emission protocol.
    """
    priority: int = 90
    kind_profile: str = "game_actions"
    habit_match: Optional[HabitMatch] = None
    motor_skills: List[MotorSkill] = field(default_factory=list)
    level_annotation: Optional[str] = None
    response_format: str = (
        "ACTION=<0-6>[ X=<0-63> Y=<0-63>]\n"
        "<one-sentence rationale>"
    )

    def __post_init__(self) -> None:
        self.kind = "effectors"

    def render(self, budget_tokens: int) -> str:
        parts: List[str] = ["## Action surface"]
        parts.append(_ACTION_SURFACES.get(self.kind_profile, _ACTION_SURFACES["game_actions"]))

        if self.level_annotation:
            parts.append(f"Level hint: {self.level_annotation}")

        if self.habit_match is not None:
            h = self.habit_match
            seq = " → ".join(h.sequence) if h.sequence else "(empty)"
            parts.append(
                f"### Cached habit match\n"
                f"habit_id: {h.habit_id}\n"
                f"confidence: {h.confidence:.2f} | reliability: {h.reliability:.2f} | sources: {h.source_count}\n"
                f"sequence: {seq}\n"
                f"If you want the full sequence executed, respond with ACTION=HABIT.\n"
                f"Otherwise emit a single action as usual."
            )

        if self.motor_skills:
            skill_lines = ["### Available skills"]
            for s in self.motor_skills:
                line = f"- {s.name}: {s.signature}"
                if s.description:
                    line += f" — {s.description}"
                skill_lines.append(line)
            skill_lines.append("Invoke a skill with: SKILL=<name> [args]")
            parts.append("\n".join(skill_lines))

        parts.append(f"## Response format\n{self.response_format}")
        return "\n\n".join(parts)

    def summarize(self, budget_tokens: int) -> str:
        # ADP form: terse surface spec + habit flag
        surface = _ACTION_SURFACES.get(self.kind_profile, "").split(".")[0]
        out = f"Actions: {surface}."
        if self.habit_match:
            out += f" Cached habit available ({self.habit_match.habit_id})."
        if self.motor_skills:
            out += f" {len(self.motor_skills)} skill(s) available."
        return out

    def minimal(self) -> str:
        return f"Actions: {self.kind_profile} surface."

    def estimate_cost(self) -> int:
        surface = _ACTION_SURFACES.get(self.kind_profile, "")
        base = len(surface) // 4 + 16
        if self.habit_match and self.habit_match.sequence:
            base += len(" ".join(self.habit_match.sequence)) // 4 + 20
        if self.motor_skills:
            base += sum(
                (len(s.name) + len(s.signature) + len(s.description)) // 4 + 4
                for s in self.motor_skills
            )
        if self.level_annotation:
            base += len(self.level_annotation) // 4 + 4
        base += len(self.response_format) // 4 + 8
        return base
