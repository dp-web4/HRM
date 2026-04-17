"""Tests for RPE — behavioral contract from spec."""

import json
import pytest
from .core import (
    RewardPredictionError, PriorTable, Prediction,
    compute_outcome_value,
)


class TestComputeOutcomeValue:
    def test_death_returns_negative_one(self):
        assert compute_outcome_value(0, False, True) == -1.0

    def test_level_up_returns_one(self):
        assert compute_outcome_value(100, True, False) == 1.0

    def test_no_change_slightly_negative(self):
        assert compute_outcome_value(0, False, False) < 0

    def test_big_change_positive(self):
        assert compute_outcome_value(600, False, False) > 0

    def test_normal_movement_positive(self):
        val = compute_outcome_value(50, False, False)
        assert 0 < val < 0.5


class TestPriorTable:
    def test_default_prior(self):
        t = PriorTable(default_prior=0.5)
        assert t.predict("s1", "UP") == 0.5

    def test_update_changes_prior(self):
        t = PriorTable(default_prior=0.5)
        rpe, before, after, alpha = t.update("s1", "UP", 1.0)
        assert before == 0.5
        assert after > before  # moved toward 1.0
        assert rpe == 0.5  # 1.0 - 0.5

    def test_learning_rate_decays(self):
        t = PriorTable()
        # First update: α = 1/(1+0) = 1.0
        _, _, _, alpha1 = t.update("s1", "UP", 1.0)
        assert alpha1 == 1.0
        # Second update: α = 1/(1+1) = 0.5
        _, _, _, alpha2 = t.update("s1", "UP", 1.0)
        assert alpha2 == 0.5
        # 10th update: α = 1/(1+9) = 0.1
        for _ in range(8):
            t.update("s1", "UP", 1.0)
        _, _, _, alpha10 = t.update("s1", "UP", 1.0)
        assert abs(alpha10 - 0.1) < 0.01

    def test_prior_bounded(self):
        t = PriorTable()
        # Push toward extreme
        for _ in range(100):
            t.update("s1", "UP", 1.0)
        assert t.predict("s1", "UP") <= 1.0
        for _ in range(100):
            t.update("s2", "DOWN", -1.0)
        assert t.predict("s2", "DOWN") >= 0.0

    def test_different_states_independent(self):
        t = PriorTable()
        t.update("s1", "RIGHT", 0.9)
        t.update("s2", "RIGHT", 0.1)
        assert t.predict("s1", "RIGHT") > t.predict("s2", "RIGHT")

    def test_dump_roundtrip(self):
        t = PriorTable()
        t.update("s1", "UP", 0.8)
        t.update("s1", "DOWN", 0.2)
        dump = t.dump()
        serialized = json.dumps(dump)
        restored = PriorTable.from_dump(json.loads(serialized))
        assert abs(restored.predict("s1", "UP") - t.predict("s1", "UP")) < 0.001
        assert abs(restored.predict("s1", "DOWN") - t.predict("s1", "DOWN")) < 0.001

    def test_action_priors(self):
        t = PriorTable(default_prior=0.5)
        t.update("s1", "UP", 0.9)
        priors = t.get_action_priors("s1", ["UP", "DOWN", "LEFT"])
        assert priors["UP"] > 0.5  # updated
        assert priors["DOWN"] == 0.5  # default
        assert priors["LEFT"] == 0.5  # default


class TestRewardPredictionError:
    def test_first_action_alpha_one(self):
        """Test 1: First action produces α=1.0 update."""
        rpe = RewardPredictionError(default_prior=0.5)
        pred = rpe.predict("s1", "RIGHT")
        signal = rpe.observe(pred, actual_value=1.0)
        assert signal.learning_rate == 1.0
        assert signal.rpe == 0.5  # 1.0 - 0.5
        # New prior = 0.5 + 1.0 * 0.5 = 1.0
        assert signal.prior_after == 1.0

    def test_stable_prior_converges(self):
        """Test 2: Repeated same value → prior converges."""
        rpe = RewardPredictionError()
        for _ in range(20):
            pred = rpe.predict("s1", "UP")
            rpe.observe(pred, actual_value=0.8)
        final = rpe.priors.predict("s1", "UP")
        assert abs(final - 0.8) < 0.05

    def test_death_negative_rpe(self):
        """Test 3: Death produces large negative RPE."""
        rpe = RewardPredictionError()
        pred = rpe.predict("s1", "RIGHT")
        signal = rpe.observe(pred, died=True)
        assert signal.rpe < -0.5
        assert signal.sign == "negative"

    def test_level_up_positive_rpe(self):
        """Test 4: Level-up produces positive RPE."""
        rpe = RewardPredictionError(default_prior=0.3)
        pred = rpe.predict("s1", "LEFT")
        signal = rpe.observe(pred, level_up=True)
        assert signal.rpe > 0
        assert signal.sign == "positive"

    def test_noop_zero_rpe(self):
        """Test 5: Action with no change → near-zero RPE against calibrated prior."""
        rpe = RewardPredictionError()
        # First train the prior to expect ~0
        for _ in range(10):
            pred = rpe.predict("s1", "WAIT")
            rpe.observe(pred, pixel_delta=0)
        # Now the prior should be near the value for pixel_delta=0
        pred = rpe.predict("s1", "WAIT")
        signal = rpe.observe(pred, pixel_delta=0)
        assert signal.magnitude < 0.15  # near-zero; exact 0 not required

    def test_state_action_independence(self):
        """Test 6: Different states, same action → independent priors."""
        rpe = RewardPredictionError()
        # State A: RIGHT is good
        for _ in range(5):
            pred = rpe.predict("stateA", "RIGHT")
            rpe.observe(pred, actual_value=0.9)
        # State B: RIGHT is bad
        for _ in range(5):
            pred = rpe.predict("stateB", "RIGHT")
            rpe.observe(pred, actual_value=0.1)
        priors_a = rpe.get_action_priors("stateA", ["RIGHT"])
        priors_b = rpe.get_action_priors("stateB", ["RIGHT"])
        assert priors_a["RIGHT"] > 0.7
        assert priors_b["RIGHT"] < 0.3

    def test_dump_roundtrip(self):
        """Test 7: dump() → JSON → reload → priors preserved."""
        rpe = RewardPredictionError()
        pred = rpe.predict("s1", "UP")
        rpe.observe(pred, actual_value=0.7)
        dump = rpe.dump()
        serialized = json.dumps(dump)
        loaded = json.loads(serialized)
        restored = RewardPredictionError.from_dump(loaded)
        assert abs(restored.priors.predict("s1", "UP") -
                   rpe.priors.predict("s1", "UP")) < 0.001

    def test_signal_emission(self):
        """Test 8: Every observe() triggers on_signal callback."""
        signals = []
        rpe = RewardPredictionError(on_signal=lambda s: signals.append(s))
        for _ in range(5):
            pred = rpe.predict("s1", "UP")
            rpe.observe(pred, actual_value=0.5)
        assert len(signals) == 5

    def test_stats(self):
        """Stats aggregate correctly."""
        rpe = RewardPredictionError()
        for v in [0.8, 0.2, 0.8, -1.0, 1.0]:
            pred = rpe.predict("s1", "UP")
            rpe.observe(pred, actual_value=v)
        stats = rpe.get_stats()
        assert stats["total_signals"] == 5
        assert stats["prior_table_size"] == 1
        assert "calibration" in stats

    def test_get_recent_signals(self):
        rpe = RewardPredictionError()
        for i in range(15):
            pred = rpe.predict(f"s{i}", "UP")
            rpe.observe(pred, actual_value=float(i % 2))
        recent = rpe.get_recent_signals(5)
        assert len(recent) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
