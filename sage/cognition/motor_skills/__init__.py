"""
SAGE Motor Skills — Parameterized Action Primitives
====================================================

The reflex tier between router dispatch and raw effectors.
Skills are parameterized policies with internal step/observe/halt
loops that execute without LLM involvement.

Brain architecture component: McNugget (extends cerebellum)
Review pair: Sprout (router dispatches to skills)

Spec: phase2/brain-arch/motor-skills.md
"""

from sage.cognition.motor_skills.types import (
    Observation,
    Skill,
    SkillInvocation,
    SkillResult,
)
from sage.cognition.motor_skills.registry import (
    SKILL_REGISTRY,
    register_skill,
    get_skill,
)
from sage.cognition.motor_skills.executor import execute_skill

__all__ = [
    'Observation',
    'Skill',
    'SkillInvocation',
    'SkillResult',
    'SKILL_REGISTRY',
    'register_skill',
    'get_skill',
    'execute_skill',
]
