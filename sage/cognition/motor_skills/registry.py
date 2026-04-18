"""
Skill registry — maps skill_id to Skill implementations.

Skills register themselves at import time via @register_skill decorator
or explicit register_skill() call.
"""

from typing import Dict, Optional, Type

from sage.cognition.motor_skills.types import Skill


SKILL_REGISTRY: Dict[str, Skill] = {}


def register_skill(skill: Skill) -> Skill:
    """Register a skill instance. Can be used as decorator on class."""
    goal_type = skill.goal_type
    if goal_type in SKILL_REGISTRY:
        raise ValueError(f"Skill {goal_type!r} already registered")
    SKILL_REGISTRY[goal_type] = skill
    return skill


def get_skill(skill_id: str) -> Optional[Skill]:
    """Look up a registered skill by ID."""
    return SKILL_REGISTRY.get(skill_id)


def list_skills() -> list:
    """Return list of registered skill IDs."""
    return list(SKILL_REGISTRY.keys())
