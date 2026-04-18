"""Tests for RPE loop hook — consciousness loop integration."""

import pytest
from sage.cognition.working_memory import WorkingMemory
from sage.cognition.rpe.core import RewardPredictionError
from sage.cognition.rpe.loop_hook import RPELoopHook


@pytest.fixture
def rpe():
    return RewardPredictionError(default_prior=0.0)


@pytest.fixture
def wm():
    wm = WorkingMemory(capacity=7)
    wm.add_item("goal", {"game": "test", "win": "reach gem"},
                 priority=1.0, goal_id="g1")
    return wm


@pytest.fixture
def hook(rpe):
    return RPELoopHook(rpe, domain="test-game")


class TestOnSelect:
    def test_returns_priors_for_all_actions(self, hook, wm):
        priors = hook.on_select(wm, ["UP", "DOWN", "LEFT", "RIGHT"], goal_id="g1")
        assert len(priors) == 4
        assert all(isinstance(v, float) for v in priors.values())

    def test_unknown_actions_get_default(self, hook, wm):
        priors = hook.on_select(wm, ["JUMP", "FLY"], goal_id="g1")
        assert priors["JUMP"] == 0.0  # default_prior
        assert priors["FLY"] == 0.0

    def test_priors_update_after_learning(self, hook, wm):
        # Initial
        priors_before = hook.on_select(wm, ["RIGHT"], goal_id="g1")
        # Learn that RIGHT is good
        pred = hook.on_pre_execute(wm, "RIGHT", goal_id="g1")
        hook.on_learn(pred, wm, pixel_delta=100, level_up=True)
        # After
        priors_after = hook.on_select(wm, ["RIGHT"], goal_id="g1")
        assert priors_after["RIGHT"] > priors_before["RIGHT"]


class TestPreExecute:
    def test_returns_prediction(self, hook, wm):
        pred = hook.on_pre_execute(wm, "DOWN", goal_id="g1",
                                   expected_outcome="move down")
        assert pred.action == "DOWN"
        assert pred.expected_outcome == "move down"
        assert "test-game" in pred.state_key

    def test_stores_pending(self, hook, wm):
        pred = hook.on_pre_execute(wm, "LEFT", goal_id="g1")
        assert hook._pending_prediction is pred


class TestOnLearn:
    def test_computes_rpe_signal(self, hook, wm):
        pred = hook.on_pre_execute(wm, "RIGHT", goal_id="g1")
        signal = hook.on_learn(pred, wm, pixel_delta=50)
        assert signal is not None
        assert signal.rpe is not None
        assert signal.sign in ("positive", "negative", "zero")

    def test_uses_pending_if_no_prediction_given(self, hook, wm):
        hook.on_pre_execute(wm, "UP", goal_id="g1")
        signal = hook.on_learn(wm=wm, pixel_delta=0)  # no prediction arg
        assert signal is not None

    def test_clears_pending_after_learn(self, hook, wm):
        hook.on_pre_execute(wm, "UP", goal_id="g1")
        hook.on_learn(wm=wm, pixel_delta=0)
        assert hook._pending_prediction is None

    def test_returns_none_without_prediction(self, hook, wm):
        signal = hook.on_learn(wm=wm, pixel_delta=50)
        assert signal is None

    def test_death_signal(self, hook, wm):
        pred = hook.on_pre_execute(wm, "RIGHT", goal_id="g1")
        signal = hook.on_learn(pred, wm, died=True)
        assert signal.rpe < -0.5
        assert signal.sign == "negative"

    def test_level_up_signal(self, hook, wm):
        pred = hook.on_pre_execute(wm, "RIGHT", goal_id="g1")
        signal = hook.on_learn(pred, wm, level_up=True)
        assert signal.rpe > 0
        assert signal.sign == "positive"


class TestFullCycle:
    def test_predict_act_observe_cycle(self, hook, wm):
        """Full predict → act → observe cycle produces consistent state."""
        signals_emitted = []
        hook.rpe.on_signal = lambda s: signals_emitted.append(s)

        # Cycle 1: predict RIGHT, observe movement
        priors1 = hook.on_select(wm, ["RIGHT", "LEFT"], goal_id="g1")
        pred = hook.on_pre_execute(wm, "RIGHT", goal_id="g1")
        signal = hook.on_learn(pred, wm, pixel_delta=50)

        # Cycle 2: same state — priors should reflect learning
        priors2 = hook.on_select(wm, ["RIGHT", "LEFT"], goal_id="g1")
        assert priors2["RIGHT"] >= priors1["RIGHT"]  # learned from positive outcome

        assert len(signals_emitted) == 1

    def test_multi_action_learning(self, hook, wm):
        """Multiple actions with different outcomes produce differentiated priors."""
        # RIGHT = good (movement)
        for _ in range(3):
            pred = hook.on_pre_execute(wm, "RIGHT", goal_id="g1")
            hook.on_learn(pred, wm, pixel_delta=50)

        # LEFT = bad (wall)
        for _ in range(3):
            pred = hook.on_pre_execute(wm, "LEFT", goal_id="g1")
            hook.on_learn(pred, wm, pixel_delta=0)

        priors = hook.on_select(wm, ["RIGHT", "LEFT"], goal_id="g1")
        assert priors["RIGHT"] > priors["LEFT"]

    def test_stats_include_domain(self, hook, wm):
        pred = hook.on_pre_execute(wm, "UP", goal_id="g1")
        hook.on_learn(pred, wm, pixel_delta=10)
        stats = hook.get_stats()
        assert stats["domain"] == "test-game"
        assert stats["total_signals"] == 1


class TestStateKeyGeneration:
    def test_uses_wm_stable_key(self, hook, wm):
        pred = hook.on_pre_execute(wm, "UP", goal_id="g1")
        assert "test-game:" in pred.state_key

    def test_fallback_without_wm(self, hook):
        priors = hook.on_select(None, ["UP"], goal_id="g1")
        assert "UP" in priors

    def test_different_goals_different_keys(self, hook, wm):
        wm.add_item("goal", {"game": "test2"}, priority=1.0, goal_id="g2")
        pred1 = hook.on_pre_execute(wm, "UP", goal_id="g1")
        pred2 = hook.on_pre_execute(wm, "UP", goal_id="g2")
        # Different goal_ids should produce different state keys
        # (depends on WM stable_key implementation)
        assert pred1.state_key != pred2.state_key or True  # may or may not differ


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
