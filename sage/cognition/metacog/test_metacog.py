"""Tests for the metacog interoceptive monitor — 10 behavioral cases from spec."""

import pytest
from .core import Metacog, MetacogConfig, MetacogSignal


class TestPerseveration:
    def test_fires_on_repeated_actions_no_delta(self):
        """Case 1: 3 identical actions with zero state delta → fires."""
        m = Metacog(config=MetacogConfig(perseveration_window=3))
        action = {"type": "click", "target": (5, 5)}
        for tick in range(3):
            signals = m.observe_tick(tick, action_taken=action, state_delta=None)
        assert any(s.signal == "perseveration" for s in signals)

    def test_no_fire_on_different_actions(self):
        """Case 2: 3 different actions that all fail → no perseveration."""
        m = Metacog(config=MetacogConfig(perseveration_window=3))
        actions = [
            {"type": "click", "target": (1, 1)},
            {"type": "move", "dir": "left"},
            {"type": "click", "target": (5, 5)},
        ]
        signals = []
        for tick, action in enumerate(actions):
            signals = m.observe_tick(tick, action_taken=action, state_delta=None)
        assert not any(s.signal == "perseveration" for s in signals)


class TestIntentionAmnesia:
    def test_fires_on_goal_action_mismatch(self):
        """Case 3: goal says blue, plan says green → fires."""
        m = Metacog()
        signals = m.observe_tick(
            0,
            goal_content={"objective": "deliver blue crystal to northern vault"},
            plan_step_content={"action": "rotate red lever in southern chamber"},
        )
        assert any(s.signal == "intention_amnesia" for s in signals)

    def test_no_fire_on_aligned_goal_action(self):
        m = Metacog()
        signals = m.observe_tick(
            0,
            goal_content={"objective": "deliver blue block to target"},
            plan_step_content={"action": "move blue block toward target"},
        )
        assert not any(s.signal == "intention_amnesia" for s in signals)


class TestStaleReference:
    def test_fires_on_old_binding(self):
        """Case 4: binding slot aged 6 ticks → fires."""

        class MockWM:
            class MockSlot:
                slot_id = "b1"
                last_access_tick = 0

            def get_by_type(self, t):
                if t == "binding":
                    return [self.MockSlot()]
                return []

        m = Metacog(wm=MockWM(), config=MetacogConfig(stale_reference_ticks=5))
        # Advance to tick 6
        signals = m.observe_tick(6)
        assert any(s.signal == "stale_reference" for s in signals)

    def test_no_fire_after_refresh(self):
        """Case 5: binding refreshed recently → no signal."""

        class MockWM:
            class MockSlot:
                slot_id = "b1"
                last_access_tick = 4

            def get_by_type(self, t):
                return [self.MockSlot()] if t == "binding" else []

        m = Metacog(wm=MockWM(), config=MetacogConfig(stale_reference_ticks=5))
        signals = m.observe_tick(5)
        assert not any(s.signal == "stale_reference" for s in signals)


class TestBudgetAnxiety:
    def test_fires_at_ratio_below_threshold(self):
        """Case 6: budget ratio below 1.5 → fires (first occurrence)."""
        m = Metacog(config=MetacogConfig(
            budget_anxiety_ratio=1.5,
            signal_cooldown_ticks=0,  # disable cooldown for this test
        ))
        # Single action establishing avg cost of 2.0
        signals = m.observe_tick(
            0,
            action_taken={"type": "x"},
            atp_cost=2.0,
            atp_balance=24.0,  # 24/2=12 remaining, est 12 → ratio 1.0
            estimated_actions_to_goal=12.0,
        )
        anxiety = [s for s in signals if s.signal in ("budget_anxiety", "budget_critical")]
        assert len(anxiety) == 1

    def test_fires_critical_at_ratio_below_one(self):
        """Case 7: 10 ATP, avg cost 2.0, est 8 → 5/8=0.625 < 1.0 → critical."""
        m = Metacog()
        for tick in range(5):
            m.observe_tick(tick, action_taken={"type": "x"}, atp_cost=2.0,
                           atp_balance=20.0, estimated_actions_to_goal=8.0)
        signals = m.observe_tick(
            5,
            action_taken={"type": "x"},
            atp_cost=2.0,
            atp_balance=8.0,
            estimated_actions_to_goal=8.0,
        )
        critical = [s for s in signals if s.signal == "budget_critical"]
        assert len(critical) == 1
        assert critical[0].severity >= 0.9


class TestExplorationStarvation:
    def test_fires_on_low_novelty(self):
        """Case 8: 10-tick avg novelty = 0.05, goal incomplete → fires."""
        m = Metacog(config=MetacogConfig(
            exploration_window=10, exploration_novelty_floor=0.1,
            signal_cooldown_ticks=0,  # disable cooldown for this test
        ))
        for tick in range(10):
            m.observe_tick(tick, snarc_novelty=0.05, goal_status="in_progress")
        signals = m.observe_tick(10, snarc_novelty=0.05, goal_status="in_progress")
        assert any(s.signal == "exploration_starvation" for s in signals)

    def test_no_fire_when_goal_completed(self):
        m = Metacog(config=MetacogConfig(exploration_window=5))
        for tick in range(10):
            m.observe_tick(tick, snarc_novelty=0.01, goal_status="completed")
        signals = m.observe_tick(10, snarc_novelty=0.01, goal_status="completed")
        assert not any(s.signal == "exploration_starvation" for s in signals)


class TestCooldown:
    def test_prevents_spam(self):
        """Case 9: same signal within cooldown window doesn't re-fire."""
        m = Metacog(config=MetacogConfig(
            perseveration_window=3, signal_cooldown_ticks=5
        ))
        action = {"type": "click", "target": (5, 5)}
        # First fire at tick 2 (3rd repeated action)
        for tick in range(3):
            signals = m.observe_tick(tick, action_taken=action, state_delta=None)
        first_fire = [s for s in signals if s.signal == "perseveration"]
        assert len(first_fire) == 1

        # Ticks 3 and 4: within cooldown (3-2=1, 4-2=2 < 5) → blocked
        s3 = m.observe_tick(3, action_taken=action, state_delta=None)
        s4 = m.observe_tick(4, action_taken=action, state_delta=None)
        assert not any(s.signal == "perseveration" for s in s3)
        assert not any(s.signal == "perseveration" for s in s4)

        # Tick 7: 7-2=5 >= 5 → fires again
        for tick in range(5, 7):
            m.observe_tick(tick, action_taken=action, state_delta=None)
        s7 = m.observe_tick(7, action_taken=action, state_delta=None)
        assert any(s.signal == "perseveration" for s in s7)

        # Tick 8: 8-7=1 < 5 → blocked again
        s8 = m.observe_tick(8, action_taken=action, state_delta=None)
        assert not any(s.signal == "perseveration" for s in s8)


class TestAcknowledge:
    def test_clears_signal(self):
        """Case 10: after acknowledge, signal no longer in active_signals."""
        m = Metacog(config=MetacogConfig(perseveration_window=3))
        action = {"type": "click", "target": (5, 5)}
        for tick in range(3):
            signals = m.observe_tick(tick, action_taken=action, state_delta=None)
        assert len(m.active_signals()) >= 1
        for sig in m.active_signals():
            m.acknowledge(sig)
        assert len(m.active_signals()) == 0


class TestCallbackAndStats:
    def test_on_signal_callback(self):
        received = []
        m = Metacog(
            config=MetacogConfig(perseveration_window=3),
            on_signal=lambda s: received.append(s),
        )
        action = {"type": "x"}
        for tick in range(3):
            m.observe_tick(tick, action_taken=action, state_delta=None)
        assert len(received) >= 1
        assert received[0].signal == "perseveration"

    def test_stats(self):
        m = Metacog()
        m.observe_tick(0, action_taken={"x": 1}, atp_cost=2.0, snarc_novelty=0.5)
        m.observe_tick(1, action_taken={"x": 2}, atp_cost=3.0, snarc_novelty=0.3)
        s = m.stats()
        assert s["actions_taken"] == 2
        assert s["total_atp_spent"] == pytest.approx(5.0)
        assert s["avg_novelty"] == pytest.approx(0.4)

    def test_reset_clears_all(self):
        m = Metacog()
        m.observe_tick(0, action_taken={"x": 1}, atp_cost=1.0, snarc_novelty=0.5)
        m.reset()
        s = m.stats()
        assert s["actions_taken"] == 0
        assert s["total_atp_spent"] == 0.0


class TestRouterIntegration:
    def test_get_block_list_empty_when_no_critical(self):
        m = Metacog()
        assert m.get_block_list() == []

    def test_get_block_list_includes_critical_signals(self):
        """budget_critical (severity 0.9) should appear in block list."""
        m = Metacog(config=MetacogConfig(signal_cooldown_ticks=0))
        # Trigger budget_critical
        m.observe_tick(
            0, action_taken={"x": 1}, atp_cost=5.0,
            atp_balance=3.0, estimated_actions_to_goal=10.0,
        )
        blocks = m.get_block_list()
        assert any("budget_critical" in b for b in blocks)

    def test_get_block_list_excludes_low_severity(self):
        """budget_anxiety (severity 0.5) should NOT appear in block list."""
        m = Metacog(config=MetacogConfig(
            signal_cooldown_ticks=0, budget_anxiety_ratio=1.5
        ))
        m.observe_tick(
            0, action_taken={"x": 1}, atp_cost=1.0,
            atp_balance=12.0, estimated_actions_to_goal=10.0,
        )
        blocks = m.get_block_list()
        assert not any("budget_anxiety" in b for b in blocks)
