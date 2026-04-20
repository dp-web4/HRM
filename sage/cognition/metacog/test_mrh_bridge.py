"""Tests for the metacog → MetabolicBlock bridge."""

from sage.cognition.metacog.core import Metacog, MetacogConfig
from sage.cognition.metacog.mrh_bridge import metacog_to_metabolic_block


class TestMetacogToMetabolicBlock:
    def test_clean_state_produces_nominal_block(self):
        m = Metacog()
        block = metacog_to_metabolic_block(m, atp_balance=50.0)
        assert block.metacog_signals == []
        assert block.stuck_duration == 0
        assert block.swap_recommendations == []
        assert block.atp_balance == 50.0
        assert block.metabolic_state == "active"

    def test_perseveration_populates_stuck_and_swap(self):
        m = Metacog(config=MetacogConfig(perseveration_window=3))
        action = {"type": "click", "target": (5, 5)}
        for tick in range(3):
            m.observe_tick(tick, action_taken=action, state_delta=None)

        block = metacog_to_metabolic_block(m, atp_balance=30.0)
        assert len(block.metacog_signals) >= 1
        assert block.metacog_signals[0].signal == "perseveration"
        assert block.stuck_duration >= 3
        assert "mechanics:stuck_escape" in block.swap_recommendations

    def test_budget_critical_triggers_swap(self):
        m = Metacog(config=MetacogConfig(signal_cooldown_ticks=0))
        m.observe_tick(
            0, action_taken={"x": 1}, atp_cost=5.0,
            atp_balance=3.0, estimated_actions_to_goal=10.0,
        )
        block = metacog_to_metabolic_block(m, atp_balance=3.0)
        assert any(s.signal == "budget_critical" for s in block.metacog_signals)
        assert "mechanics:budget_conserve" in block.swap_recommendations

    def test_exploration_starvation_triggers_swap(self):
        m = Metacog(config=MetacogConfig(
            exploration_window=5, exploration_novelty_floor=0.1,
            signal_cooldown_ticks=0,
        ))
        for tick in range(6):
            m.observe_tick(tick, snarc_novelty=0.02, goal_status="in_progress")
        block = metacog_to_metabolic_block(m)
        assert any(s.signal == "exploration_starvation" for s in block.metacog_signals)
        assert "mechanics:exploration_hints" in block.swap_recommendations

    def test_caller_fields_pass_through(self):
        m = Metacog()
        block = metacog_to_metabolic_block(
            m,
            metabolic_state="focus",
            atp_balance=99.0,
            atp_trend="rising",
            confidence=0.85,
            actions_since_last_invoke=12,
        )
        assert block.metabolic_state == "focus"
        assert block.atp_balance == 99.0
        assert block.atp_trend == "rising"
        assert block.confidence == 0.85
        assert block.actions_since_last_invoke == 12

    def test_block_renders_cleanly(self):
        m = Metacog(config=MetacogConfig(perseveration_window=3))
        for tick in range(3):
            m.observe_tick(tick, action_taken={"x": 1}, state_delta=None)
        block = metacog_to_metabolic_block(m, atp_balance=20.0, confidence=0.4)
        text = block.render(budget_tokens=500)
        assert "System state" in text
        assert "perseveration" in text
        assert "ATP=20" in text

    def test_block_summarizes_cleanly(self):
        m = Metacog(config=MetacogConfig(perseveration_window=3))
        for tick in range(3):
            m.observe_tick(tick, action_taken={"x": 1}, state_delta=None)
        block = metacog_to_metabolic_block(m)
        summary = block.summarize(budget_tokens=100)
        assert "perseveration" in summary
        assert len(summary) < 200
