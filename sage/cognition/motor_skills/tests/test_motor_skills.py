"""Tests for motor skills: executor, registry, and navigate_to."""

from sage.cognition.motor_skills.types import Observation, SkillInvocation, SkillResult
from sage.cognition.motor_skills.registry import SKILL_REGISTRY, get_skill, list_skills
from sage.cognition.motor_skills.executor import execute_skill

# Import skills to trigger registration
import sage.cognition.motor_skills.skills  # noqa: F401


# ─── Registry tests ────────────────────────────────────────────────

def test_navigate_to_registered():
    assert "navigate_to" in SKILL_REGISTRY
    assert get_skill("navigate_to") is not None


def test_list_skills():
    skills = list_skills()
    assert "navigate_to" in skills


def test_unknown_skill_returns_none():
    assert get_skill("nonexistent_skill_xyz") is None


# ─── navigate_to skill tests ───────────────────────────────────────

def test_navigate_to_step_right():
    skill = get_skill("navigate_to")
    obs = Observation(position=(0, 5))
    action = skill.step(obs, {"x": 10, "y": 5})
    assert action == 4  # RIGHT


def test_navigate_to_step_left():
    skill = get_skill("navigate_to")
    obs = Observation(position=(10, 5))
    action = skill.step(obs, {"x": 0, "y": 5})
    assert action == 3  # LEFT


def test_navigate_to_step_down():
    skill = get_skill("navigate_to")
    obs = Observation(position=(5, 0))
    action = skill.step(obs, {"x": 5, "y": 10})
    assert action == 2  # DOWN


def test_navigate_to_step_up():
    skill = get_skill("navigate_to")
    obs = Observation(position=(5, 10))
    action = skill.step(obs, {"x": 5, "y": 0})
    assert action == 1  # UP


def test_navigate_to_prefers_larger_delta():
    skill = get_skill("navigate_to")
    # dx=5, dy=2 → horizontal first
    obs = Observation(position=(0, 0))
    action = skill.step(obs, {"x": 5, "y": 2})
    assert action == 4  # RIGHT (larger delta)


def test_navigate_to_halt_at_target():
    skill = get_skill("navigate_to")
    obs = Observation(position=(5, 5))
    assert skill.halt_condition(obs, {"x": 5, "y": 5}) is True


def test_navigate_to_no_halt_away():
    skill = get_skill("navigate_to")
    obs = Observation(position=(5, 5))
    assert skill.halt_condition(obs, {"x": 10, "y": 5}) is False


def test_navigate_to_stuck_detection():
    skill = get_skill("navigate_to")
    # Same position 5 times in a row
    recent = [Observation(position=(3, 3)) for _ in range(5)]
    obs = recent[-1]
    assert skill.stuck_condition(obs, {"x": 10, "y": 10}, recent) is True


def test_navigate_to_not_stuck_when_moving():
    skill = get_skill("navigate_to")
    recent = [
        Observation(position=(3, 3)),
        Observation(position=(4, 3)),
        Observation(position=(5, 3)),
    ]
    obs = recent[-1]
    assert skill.stuck_condition(obs, {"x": 10, "y": 3}, recent) is False


def test_navigate_to_progress_metric():
    skill = get_skill("navigate_to")
    # At target → 1.0
    assert skill.progress_metric(Observation(position=(5, 5)), {"x": 5, "y": 5}) == 1.0
    # Far away → close to 0
    far = skill.progress_metric(Observation(position=(0, 0)), {"x": 60, "y": 60})
    assert far < 0.1
    # Halfway
    mid = skill.progress_metric(Observation(position=(30, 30)), {"x": 60, "y": 60})
    assert 0.4 < mid < 0.7


# ─── Executor tests ───────────────────────────────────────────────

class MockEnv:
    """Simulated grid environment for testing."""

    def __init__(self, start, walls=None):
        self.pos = list(start)
        self.walls = walls or set()
        self.actions_applied = []

    def observe(self):
        return Observation(position=tuple(self.pos))

    def act(self, action):
        self.actions_applied.append(action)
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(action, (0, 0))
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        if (nx, ny) not in self.walls:
            self.pos = [nx, ny]


def test_executor_halts_at_target():
    env = MockEnv(start=(0, 0))
    inv = SkillInvocation(skill_id="navigate_to", params={"x": 3, "y": 0})
    result = execute_skill(inv, env.observe, env.act)
    assert result.status == "halted"
    assert result.steps_taken == 3
    assert env.pos == [3, 0]


def test_executor_halts_diagonal():
    env = MockEnv(start=(0, 0))
    inv = SkillInvocation(skill_id="navigate_to", params={"x": 2, "y": 3})
    result = execute_skill(inv, env.observe, env.act)
    assert result.status == "halted"
    assert env.pos == [2, 3]
    assert result.steps_taken == 5  # 2 right + 3 down


def test_executor_stuck_at_wall():
    # Wall blocks path to target
    env = MockEnv(start=(0, 0), walls={(1, 0), (0, 1)})
    inv = SkillInvocation(
        skill_id="navigate_to",
        params={"x": 5, "y": 5},
        max_stuck=3,
        max_steps=20,
    )
    result = execute_skill(inv, env.observe, env.act)
    assert result.status == "stuck"
    assert result.steps_taken < 20  # should detect stuck well before max_steps


def test_executor_max_steps():
    # Target unreachable but position keeps changing (oscillation)
    class OscillatingEnv:
        def __init__(self):
            self.step_count = 0
        def observe(self):
            # Position alternates so stuck detection doesn't fire
            return Observation(position=(self.step_count % 2, 0))
        def act(self, action):
            self.step_count += 1

    env = OscillatingEnv()
    inv = SkillInvocation(
        skill_id="navigate_to",
        params={"x": 99, "y": 99},
        max_steps=10,
        max_stuck=5,
    )
    result = execute_skill(inv, env.observe, env.act)
    assert result.status == "max_steps"
    assert result.steps_taken == 10


def test_executor_unknown_skill():
    inv = SkillInvocation(skill_id="does_not_exist", params={})
    result = execute_skill(inv, lambda: Observation(), lambda a: None)
    assert result.status == "error"
    assert "Unknown skill" in result.error


def test_executor_pre_state_passthrough():
    env = MockEnv(start=(0, 0))
    pre = {"goal": "test", "wm_slots": 3}
    inv = SkillInvocation(skill_id="navigate_to", params={"x": 1, "y": 0})
    result = execute_skill(inv, env.observe, env.act, pre_state=pre)
    assert result.pre_state == pre


def test_executor_immediate_halt():
    """Already at target → halts at step 0, no actions taken."""
    env = MockEnv(start=(5, 5))
    inv = SkillInvocation(skill_id="navigate_to", params={"x": 5, "y": 5})
    result = execute_skill(inv, env.observe, env.act)
    assert result.status == "halted"
    assert result.steps_taken == 0
    assert env.actions_applied == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"  {t.__name__}...", end=" ")
        t()
        print("OK")
    print(f"\nAll {len(tests)} tests passed.")
