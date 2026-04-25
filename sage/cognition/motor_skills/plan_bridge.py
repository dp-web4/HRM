"""
Bridge between Codification Project plan_executor and motor skills.

Converts plan steps into SkillInvocations and vice versa. The plan
executor runs structured plans (LLM-authored); motor skills execute
individual steps with internal halt/stuck loops.

Connection:
  Plan step {"do": "navigate_to", "x": 5, "y": 7}
  → SkillInvocation(skill_id="navigate_to", params={"x": 5, "y": 7})

  Plan step {"do": "UP", "repeat": 3}
  → 3× raw action (no skill — direct env.step)

  Plan step {"do": "click_object", "x": 38, "y": 46}
  → SkillInvocation(skill_id="click_object", params={"x": 38, "y": 46})

The bridge decides: is this step a skill invocation or a raw action?
Skills have halt/stuck detection; raw actions are fire-and-forget.
"""

from typing import Any, Dict, List, Optional, Tuple

from sage.cognition.motor_skills.types import SkillInvocation
from sage.cognition.motor_skills.registry import get_skill


# Actions that are always raw (never skills)
RAW_ACTIONS = {"UP", "DOWN", "LEFT", "RIGHT", "SEL", "CLICK", "A0",
               "1", "2", "3", "4", "5", "6"}


def step_to_invocation(step: Dict[str, Any]) -> Optional[SkillInvocation]:
    """Convert a plan step to a SkillInvocation if it names a registered skill.

    Returns None if the step is a raw action (should be executed directly).
    """
    do = str(step.get("do", "")).strip()

    # Check if it's a registered skill
    skill = get_skill(do)
    if skill is not None:
        # Extract params — everything except "do", "expect", "repeat", "_phase"
        params = {k: v for k, v in step.items()
                  if k not in ("do", "expect", "repeat", "_phase", "note")}
        return SkillInvocation(
            skill_id=do,
            params=params,
            origin="plan",
            max_steps=step.get("max_steps", 50),
            max_stuck=step.get("max_stuck", 5),
        )

    # Check if it's a skill by name even if uppercase (navigate_to vs NAVIGATE_TO)
    skill_lower = get_skill(do.lower())
    if skill_lower is not None:
        params = {k: v for k, v in step.items()
                  if k not in ("do", "expect", "repeat", "_phase", "note")}
        return SkillInvocation(
            skill_id=do.lower(),
            params=params,
            origin="plan",
        )

    # Raw action — return None
    return None


def is_skill_step(step: Dict[str, Any]) -> bool:
    """Check if a plan step should be executed as a skill."""
    return step_to_invocation(step) is not None


def plan_to_mixed_sequence(plan_steps: List[Dict]) -> List[Tuple[str, Any]]:
    """Convert a plan into a mixed sequence of skills and raw actions.

    Returns list of ("skill", SkillInvocation) or ("raw", step_dict) tuples.
    The caller executes skills via execute_skill() and raw steps via env.step().
    """
    sequence = []
    for step in plan_steps:
        inv = step_to_invocation(step)
        if inv is not None:
            sequence.append(("skill", inv))
        else:
            repeat = int(step.get("repeat", 1))
            for _ in range(repeat):
                sequence.append(("raw", step))
    return sequence


def invocation_to_plan_step(inv: SkillInvocation) -> Dict[str, Any]:
    """Convert a SkillInvocation back to a plan step dict.

    Useful for logging: after a skill executes, record what it did
    in the plan format.
    """
    step = {"do": inv.skill_id}
    step.update(inv.params)
    return step
