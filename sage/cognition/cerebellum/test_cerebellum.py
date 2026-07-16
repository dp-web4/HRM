"""Tests for the cerebellum / habit compiler."""

import json
import tempfile
from pathlib import Path

import pytest

from sage.cognition.cerebellum.core import (
    Cerebellum,
    Habit,
    StateSignature,
)


def make_state(domain="test", **features) -> StateSignature:
    return StateSignature(domain=domain, features=features)


def make_actions(n=3):
    return [{"plugin": "test", "action": f"step_{i}"} for i in range(n)]


def test_state_signature_hash():
    """Same features produce same hash."""
    s1 = make_state(level=1, color="red")
    s2 = make_state(level=1, color="red")
    assert s1.hash == s2.hash

    s3 = make_state(level=2, color="red")
    assert s1.hash != s3.hash


def test_state_similarity():
    s1 = make_state(level=1, color="red", size=5)
    s2 = make_state(level=1, color="red", size=5)
    assert s1.similarity(s2) == 1.0

    s3 = make_state(level=1, color="blue", size=5)
    assert s1.similarity(s3) == 2 / 3  # level and size match, color doesn't

    s4 = StateSignature(domain="other", features={"level": 1})
    assert s1.similarity(s4) == 0.0  # different domain


def test_basic_habit_compilation():
    """T1: Observe 3x → habit becomes mature."""
    cb = Cerebellum(maturity_threshold=3)
    state = make_state(level=1)
    actions = make_actions()

    for _ in range(3):
        cb.observe(state, actions, {"success": True})

    matches = cb.lookup(state)
    assert len(matches) == 1
    assert matches[0].habit.is_mature
    assert matches[0].habit.training_count == 3
    assert matches[0].habit.success_count == 3


def test_habit_failure_demotion():
    """T2: Failed executions reduce reliability."""
    cb = Cerebellum()
    state = make_state(level=1)
    actions = make_actions()

    # Build mature habit (5 successes)
    for _ in range(5):
        cb.observe(state, actions, {"success": True})

    habit = cb.lookup(state)[0].habit
    assert habit.reliability == 1.0
    assert habit.is_mature

    # Simulate failure via observe with success=False
    cb.observe(state, actions, {"success": False})
    assert habit.training_count == 6
    assert habit.success_count == 5
    assert abs(habit.reliability - 5 / 6) < 0.01
    assert habit.is_mature  # still above 0.8

    cb.observe(state, actions, {"success": False})
    assert habit.training_count == 7
    assert habit.success_count == 5
    assert abs(habit.reliability - 5 / 7) < 0.01
    assert not habit.is_mature  # dropped below 0.8


def test_cross_level_no_match():
    """T3: Different levels should NOT match."""
    cb = Cerebellum()
    state_l1 = make_state(domain="toy-game:toy_a", level=1, sprites=10)
    state_l2 = make_state(domain="toy-game:toy_a", level=2, sprites=8)
    actions = make_actions()

    for _ in range(3):
        cb.observe(state_l1, actions, {"success": True})

    matches = cb.lookup(state_l2)
    # Similarity: domain matches, but level and sprites differ
    # 1/3 features match → 0.33 < 0.85 threshold
    assert len(matches) == 0


def test_no_immature_habits_returned():
    """Lookup only returns mature habits."""
    cb = Cerebellum()
    state = make_state(level=1)
    actions = make_actions()

    # Only 1 observation — not mature
    cb.observe(state, actions, {"success": True})
    assert len(cb.lookup(state)) == 0

    # 2 observations — still not mature
    cb.observe(state, actions, {"success": True})
    assert len(cb.lookup(state)) == 0

    # 3 observations — now mature
    cb.observe(state, actions, {"success": True})
    assert len(cb.lookup(state)) == 1


def test_batch_compilation():
    """T5: compile_from_episodes finds repeated patterns."""
    cb = Cerebellum()

    state_a = {"domain": "test", "features": {"level": 1}}
    state_b = {"domain": "test", "features": {"level": 2}}

    episodes = [
        {"state": state_a, "actions": [{"action": "x"}], "outcome": {"success": True}},
        {"state": state_a, "actions": [{"action": "x"}], "outcome": {"success": True}},
        {"state": state_a, "actions": [{"action": "x"}], "outcome": {"success": True}},
        {"state": state_b, "actions": [{"action": "y"}], "outcome": {"success": True}},
        {"state": state_b, "actions": [{"action": "y"}], "outcome": {"success": False}},
    ]

    habits = cb.compile_from_episodes(episodes)
    # state_a: 3 episodes, all success → compiled
    # state_b: 2 episodes, below threshold → not compiled
    assert len(habits) == 1
    assert habits[0].state_sig.domain == "test"
    assert habits[0].training_count == 3
    assert habits[0].success_count == 3


def test_persistence():
    """Export and load round-trip."""
    cb = Cerebellum()
    state = make_state(level=1, game="toy_a")
    actions = make_actions()

    for _ in range(3):
        cb.observe(state, actions, {"success": True})

    data = cb.export()
    cb2 = Cerebellum.load(data)

    assert cb2.habit_count == cb.habit_count
    assert cb2.mature_count == cb.mature_count

    matches = cb2.lookup(state)
    assert len(matches) == 1
    assert matches[0].habit.is_mature


def test_file_persistence():
    """Save to file and load back."""
    cb = Cerebellum()
    state = make_state(level=1)
    for _ in range(3):
        cb.observe(state, make_actions(), {"success": True})

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "habits.json"
        cb.save(path)
        cb2 = Cerebellum.load_from_file(path)
        assert cb2.habit_count == 1
        assert cb2.mature_count == 1


def test_capacity_eviction():
    """When at max capacity, lowest reliability is evicted."""
    cb = Cerebellum(max_habits=3)

    # Add 3 habits with different reliability
    for i in range(3):
        state = make_state(level=i)
        for _ in range(3):
            cb.observe(state, make_actions(), {"success": i > 0})  # level 0 always fails

    assert cb.habit_count == 3

    # Add a 4th — should evict the lowest reliability (level 0)
    state_new = make_state(level=99)
    cb.observe(state_new, make_actions(), {"success": True})
    assert cb.habit_count == 3

    # Level 0 habit should be gone (0% reliability)
    matches = cb.lookup(make_state(level=0))
    assert len(matches) == 0


def test_prune():
    """Pruning removes unreliable habits."""
    cb = Cerebellum()
    state = make_state(level=1)
    actions = make_actions()

    # Create a habit with low reliability
    cb.observe(state, actions, {"success": True})
    cb.observe(state, actions, {"success": False})
    cb.observe(state, actions, {"success": False})

    assert cb.habit_count == 1
    removed = cb.prune(min_reliability=0.5)
    assert removed == 1
    assert cb.habit_count == 0


def test_stats():
    cb = Cerebellum()
    assert cb.stats()["total_habits"] == 0

    for _ in range(3):
        cb.observe(make_state(level=1), make_actions(), {"success": True})

    stats = cb.stats()
    assert stats["total_habits"] == 1
    assert stats["mature_habits"] == 1
    assert stats["avg_reliability"] == 1.0
    assert "test" in stats["domains"]


def test_consensus_threshold_rejects_out_of_range():
    with pytest.raises(ValueError):
        Cerebellum(consensus_threshold=1.5)
    with pytest.raises(ValueError):
        Cerebellum(consensus_threshold=-0.1)
    # Valid bounds: 0.0, 1.0, and None all construct fine.
    Cerebellum(consensus_threshold=0.0)
    Cerebellum(consensus_threshold=1.0)
    Cerebellum(consensus_threshold=None)


def test_consensus_threshold_blocks_weak_plurality():
    """3 divergent arcs (1 each) at threshold 0.5 → no compile.

    Plurality winner has ratio 1/3 ≈ 0.33, below the floor. Without the
    gate, the cerebellum would compile an arbitrary winner as a habit
    even though the agent has no actual preferred action at that state.
    """
    cb = Cerebellum(maturity_threshold=3, consensus_threshold=0.5)
    state = {"domain": "test", "features": {"level": 1}}
    episodes = [
        {"state": state, "actions": [{"action": "a"}], "outcome": {"success": True}},
        {"state": state, "actions": [{"action": "b"}], "outcome": {"success": True}},
        {"state": state, "actions": [{"action": "c"}], "outcome": {"success": True}},
    ]
    assert cb.compile_from_episodes(episodes) == []


def test_consensus_threshold_admits_majority():
    """2/3 agreement (ratio 0.667) clears threshold 0.5 → compile."""
    cb = Cerebellum(maturity_threshold=3, consensus_threshold=0.5)
    state = {"domain": "test", "features": {"level": 1}}
    episodes = [
        {"state": state, "actions": [{"action": "x"}], "outcome": {"success": True}},
        {"state": state, "actions": [{"action": "x"}], "outcome": {"success": True}},
        {"state": state, "actions": [{"action": "y"}], "outcome": {"success": True}},
    ]
    habits = cb.compile_from_episodes(episodes)
    assert len(habits) == 1
    assert habits[0].action_sequence == [{"action": "x"}]


def test_consensus_threshold_strict_blocks_majority():
    """2/3 agreement (ratio 0.667) fails threshold 0.75 → no compile.

    Validates that the floor is genuinely compared — not just "any
    majority passes." A stricter floor can force the compile path to
    wait for stronger agreement before cementing the habit.
    """
    cb = Cerebellum(maturity_threshold=3, consensus_threshold=0.75)
    state = {"domain": "test", "features": {"level": 1}}
    episodes = [
        {"state": state, "actions": [{"action": "x"}], "outcome": {"success": True}},
        {"state": state, "actions": [{"action": "x"}], "outcome": {"success": True}},
        {"state": state, "actions": [{"action": "y"}], "outcome": {"success": True}},
    ]
    assert cb.compile_from_episodes(episodes) == []


def test_consensus_threshold_none_preserves_plurality_winner():
    """Default (None) preserves pre-S81 behavior: plurality wins.

    3 divergent arcs (1 each) must still compile one habit — the first
    in iteration order — when the gate is off. This is the backward-
    compatibility contract for machines that haven't opted in.
    """
    cb = Cerebellum(maturity_threshold=3, consensus_threshold=None)
    state = {"domain": "test", "features": {"level": 1}}
    episodes = [
        {"state": state, "actions": [{"action": "a"}], "outcome": {"success": True}},
        {"state": state, "actions": [{"action": "b"}], "outcome": {"success": True}},
        {"state": state, "actions": [{"action": "c"}], "outcome": {"success": True}},
    ]
    habits = cb.compile_from_episodes(episodes)
    assert len(habits) == 1


def test_consensus_ratio_recorded_in_outcome_summary():
    """The compile path exposes the consensus count in outcome_summary.

    This is the introspection hook for "how strong was agreement?" —
    available whether or not the gate was used for filtering, so that
    downstream tooling (raising reports, trend analysis) can see
    consensus strength directly on the Habit record.
    """
    cb = Cerebellum(maturity_threshold=3)
    state = {"domain": "test", "features": {"level": 1}}
    episodes = [
        {"state": state, "actions": [{"action": "x"}], "outcome": {"success": True}},
        {"state": state, "actions": [{"action": "x"}], "outcome": {"success": True}},
        {"state": state, "actions": [{"action": "y"}], "outcome": {"success": True}},
    ]
    habits = cb.compile_from_episodes(episodes)
    assert len(habits) == 1
    assert "consensus 2/3" in habits[0].outcome_summary


def test_link_episode():
    cb = Cerebellum()
    state = make_state(level=1)
    habit = cb.observe(state, make_actions(), {"success": True})
    assert habit is not None

    cb.link_episode(habit.habit_id, "ep-001")
    cb.link_episode(habit.habit_id, "ep-002")
    cb.link_episode(habit.habit_id, "ep-001")  # duplicate, should not add

    assert len(cb._habits[habit.habit_id].source_episodes) == 2


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        print(f"  {t.__name__}...", end=" ")
        t()
        print("OK")
    print(f"\nAll {len(tests)} tests passed.")
