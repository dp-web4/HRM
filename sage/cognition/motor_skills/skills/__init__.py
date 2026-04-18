"""
Skill implementations.

Each skill is a separate file that registers itself on import.
Import this package to load all built-in skills.
"""

from sage.cognition.motor_skills.skills.navigate_to import NavigateToSkill

__all__ = ['NavigateToSkill']
