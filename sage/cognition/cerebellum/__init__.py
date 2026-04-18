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
    compile_habits_from_episodes,
    compile_habits_from_trajectories,
    episode_to_cerebellum_dict,
    group_episodes_into_trajectories,
    trajectory_to_cerebellum_dict,
)
from sage.cognition.cerebellum.loop_hook import CerebellumLoopHook

__all__ = [
    'StateSignature',
    'Habit',
    'HabitMatch',
    'Cerebellum',
    'CerebellumLoopHook',
    'episode_to_cerebellum_dict',
    'compile_habits_from_episodes',
    'group_episodes_into_trajectories',
    'trajectory_to_cerebellum_dict',
    'compile_habits_from_trajectories',
]
