"""
SAGE Cerebellum / Habit Compiler
================================

Detects repeated successful (state → action_sequence → outcome) patterns
and compiles them into cached habits that the router can invoke directly,
skipping full deliberation.

Brain architecture component: McNugget
Review pair: Thor (episodic index)
"""

from sage.cognition.cerebellum.core import (
    StateSignature,
    Habit,
    HabitMatch,
    Cerebellum,
)
from sage.cognition.cerebellum.episodic_bridge import (
    episode_to_cerebellum_dict,
    compile_habits_from_episodes,
)

__all__ = [
    'StateSignature',
    'Habit',
    'HabitMatch',
    'Cerebellum',
    'episode_to_cerebellum_dict',
    'compile_habits_from_episodes',
]
