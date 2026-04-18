"""
Skill executor — runs a skill's internal step/observe/halt loop.

One skill invocation may produce N effector actions, but the router
sees it as ONE dispatch event. The internal loop is invisible above.
"""

import time
from typing import Any, Callable, Dict, Optional

from sage.cognition.motor_skills.types import (
    Observation,
    Skill,
    SkillInvocation,
    SkillResult,
)
from sage.cognition.motor_skills.registry import get_skill


def execute_skill(
    invocation: SkillInvocation,
    observe_fn: Callable[[], Observation],
    act_fn: Callable[[Any], None],
    pre_state: Optional[Dict[str, Any]] = None,
) -> SkillResult:
    """Execute a skill's internal loop.

    Args:
        invocation: What skill to run with what params.
        observe_fn: Callable that returns current Observation (called each step).
        act_fn: Callable that applies an action to the environment.
        pre_state: WM snapshot at invocation time (for SkillResult).

    Returns:
        SkillResult with status, steps_taken, and state snapshots.
    """
    skill = get_skill(invocation.skill_id)
    if skill is None:
        return SkillResult(
            skill_id=invocation.skill_id,
            status="error",
            steps_taken=0,
            pre_state=pre_state or {},
            post_state=pre_state or {},
            error=f"Unknown skill: {invocation.skill_id!r}",
        )

    recent_obs: list[Observation] = []
    t0 = time.time()

    for step in range(invocation.max_steps):
        obs = observe_fn()
        recent_obs.append(obs)
        if len(recent_obs) > invocation.max_stuck:
            recent_obs = recent_obs[-invocation.max_stuck:]

        # Check halt (goal achieved)
        if skill.halt_condition(obs, invocation.params):
            return SkillResult(
                skill_id=invocation.skill_id,
                status="halted",
                steps_taken=step,
                pre_state=pre_state or {},
                post_state={},
                final_obs=obs,
                elapsed=time.time() - t0,
            )

        # Check stuck (no progress)
        if len(recent_obs) >= invocation.max_stuck:
            if skill.stuck_condition(obs, invocation.params, recent_obs):
                return SkillResult(
                    skill_id=invocation.skill_id,
                    status="stuck",
                    steps_taken=step,
                    pre_state=pre_state or {},
                    post_state={},
                    final_obs=obs,
                    elapsed=time.time() - t0,
                )

        # Execute one step
        action = skill.step(obs, invocation.params)
        act_fn(action)

    # Exhausted step budget
    final_obs = observe_fn()
    return SkillResult(
        skill_id=invocation.skill_id,
        status="max_steps",
        steps_taken=invocation.max_steps,
        pre_state=pre_state or {},
        post_state={},
        final_obs=final_obs,
        elapsed=time.time() - t0,
    )
