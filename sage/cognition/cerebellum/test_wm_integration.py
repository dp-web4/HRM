"""Tests for cerebellum ↔ working memory ↔ loop hook integration."""

from sage.cognition.cerebellum.core import Cerebellum, StateSignature
from sage.cognition.cerebellum.loop_hook import CerebellumLoopHook
from sage.cognition.working_memory import WorkingMemory


def test_state_signature_from_wm():
    """StateSignature.from_wm() extracts stable features from WM."""
    wm = WorkingMemory(capacity=7)
    wm.add_item("goal", {"game": "toy_a", "level": 1}, priority=0.9)
    wm.add_item("binding", {"sprite": "cursor", "pos": (10, 20)}, priority=0.5)

    sig = StateSignature.from_wm(wm, domain="toy-game:toy_a")
    assert sig.domain == "toy-game:toy_a"
    assert sig.hash  # non-empty
    assert "wm_goal_count" in sig.features
    assert sig.features["wm_goal_count"] == 1
    assert "wm_binding_count" in sig.features


def test_stable_key_determinism():
    """Same WM contents → same state signature hash."""
    wm1 = WorkingMemory(capacity=7)
    wm1.add_item("goal", {"game": "toy_a"}, priority=0.9)

    wm2 = WorkingMemory(capacity=7)
    wm2.add_item("goal", {"game": "toy_a"}, priority=0.9)

    sig1 = StateSignature.from_wm(wm1, "test")
    sig2 = StateSignature.from_wm(wm2, "test")
    assert sig1.hash == sig2.hash


def test_goal_scoping():
    """from_wm with goal_id only considers slots for that goal."""
    wm = WorkingMemory(capacity=7)
    wm.add_item("goal", {"game": "toy_a"}, priority=0.9, goal_id="g1")
    wm.add_item("goal", {"game": "toy_b"}, priority=0.8, goal_id="g2")

    sig_g1 = StateSignature.from_wm(wm, "test", goal_id="g1")
    sig_g2 = StateSignature.from_wm(wm, "test", goal_id="g2")
    assert sig_g1.hash != sig_g2.hash


def test_empty_wm_produces_valid_signature():
    """Empty WM still produces a valid (but featureless) signature."""
    wm = WorkingMemory(capacity=7)
    sig = StateSignature.from_wm(wm, "test")
    assert sig.domain == "test"
    assert sig.hash  # non-empty (hashes domain + empty features)


def test_loop_hook_select():
    """CerebellumLoopHook.on_select() builds sig from WM and queries habits."""
    cb = Cerebellum()
    hook = CerebellumLoopHook(cb, domain="test")
    wm = WorkingMemory(capacity=7)
    wm.add_item("goal", {"task": "solve"}, priority=0.9)

    # No habits yet → empty matches
    matches = hook.on_select(wm)
    assert matches == []

    # Train a habit with the same WM state
    state = StateSignature.from_wm(wm, "test")
    actions = [{"action": "step1"}, {"action": "step2"}]
    for _ in range(3):
        cb.observe(state, actions, {"success": True})

    # Now should find it
    matches = hook.on_select(wm)
    assert len(matches) == 1
    assert matches[0].habit.is_mature


def test_loop_hook_learn():
    """CerebellumLoopHook.on_learn() records observations."""
    cb = Cerebellum()
    hook = CerebellumLoopHook(cb, domain="test")
    wm = WorkingMemory(capacity=7)
    wm.add_item("goal", {"task": "solve"}, priority=0.9)

    # Select first (captures state)
    hook.on_select(wm)

    # Then learn
    actions = [{"action": "click", "data": {"x": 10, "y": 20}}]
    habit = hook.on_learn(wm, actions, {"success": True})
    assert habit is not None
    assert cb.habit_count == 1


def test_loop_hook_full_cycle():
    """Full loop: select → execute → learn → select (habit fires)."""
    cb = Cerebellum(maturity_threshold=3)
    hook = CerebellumLoopHook(cb, domain="test")
    wm = WorkingMemory(capacity=7)
    wm.add_item("goal", {"task": "navigate"}, priority=0.9)

    actions = [{"plugin": "nav", "action": "go_north"}]

    # 3 cycles of select→learn
    for _ in range(3):
        hook.on_select(wm)
        hook.on_learn(wm, actions, {"success": True})

    # 4th select should find a mature habit
    matches = hook.on_select(wm)
    assert len(matches) == 1
    assert matches[0].habit.is_mature
    assert matches[0].confidence > 0.8


def test_loop_hook_learn_without_prior_select():
    """on_learn works even if on_select wasn't called first."""
    cb = Cerebellum()
    hook = CerebellumLoopHook(cb, domain="test")
    wm = WorkingMemory(capacity=7)
    wm.add_item("goal", {"task": "x"}, priority=0.9)

    habit = hook.on_learn(wm, [{"action": "a"}], {"success": True})
    assert habit is not None


def test_loop_hook_stats():
    cb = Cerebellum()
    hook = CerebellumLoopHook(cb, domain="test")
    assert hook.stats["total_habits"] == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"  {t.__name__}...", end=" ")
        t()
        print("OK")
    print(f"\nAll {len(tests)} tests passed.")
