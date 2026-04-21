"""Tests for lookahead-aware metacog detectors.

Tested against real data from Nomad's 4-game lookahead run (2026-04-21).
"""

import pytest

from sage.cognition.metacog.lookahead_detectors import (
    detect_lookahead_blind,
    detect_prediction_equivalence,
    detect_predictions_ignored,
)


# --- Real data from Nomad's 4-game run ---

SB26_PREDICTIONS = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0, "SEL": 1}
SC25_PREDICTIONS = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0, "SEL": 0}
TR87_PREDICTIONS = {"UP": 13, "DOWN": 15, "LEFT": 28, "RIGHT": 28, "SEL": 0}
CN04_PREDICTIONS = {"UP": 198, "DOWN": 198, "LEFT": 144, "RIGHT": 144, "SEL": 162}

# Thor's cd82 case (from the breakthrough doc)
CD82_PREDICTIONS = {"UP": 0, "DOWN": 0, "LEFT": 201, "RIGHT": 201, "SEL": 51}


class TestPredictionEquivalence:
    def test_cn04_fires(self):
        """cn04: UP/DOWN=198, LEFT/RIGHT=144, SEL=162 — 5 actions within range."""
        sig = detect_prediction_equivalence(CN04_PREDICTIONS)
        assert sig is not None
        assert sig.signal == "prediction_equivalence"
        assert len(sig.evidence["equivalent_actions"]) >= 3

    def test_tr87_does_not_fire(self):
        """tr87: UP=13, DOWN=15 vs LEFT/RIGHT=28 — clear separation."""
        sig = detect_prediction_equivalence(TR87_PREDICTIONS)
        assert sig is None

    def test_sb26_does_not_fire(self):
        """sb26: all blocked — too few active to compare."""
        sig = detect_prediction_equivalence(SB26_PREDICTIONS)
        assert sig is None

    def test_cd82_fires_on_left_right(self):
        """cd82: LEFT=201, RIGHT=201 — the exact case Thor identified."""
        # With min_equivalent=2 (lowered to catch pairs)
        sig = detect_prediction_equivalence(CD82_PREDICTIONS, min_equivalent=2)
        assert sig is not None
        assert "LEFT" in sig.evidence["equivalent_actions"]
        assert "RIGHT" in sig.evidence["equivalent_actions"]

    def test_cd82_does_not_fire_at_default_threshold(self):
        """cd82: only 2 equivalent (LEFT/RIGHT) — default min is 3."""
        sig = detect_prediction_equivalence(CD82_PREDICTIONS, min_equivalent=3)
        assert sig is None  # SEL=51 is too different from 201

    def test_threshold_tuning(self):
        """Wider threshold catches more near-equivalents."""
        # With 50% threshold, cn04's 144 and 198 are within range
        sig = detect_prediction_equivalence(CN04_PREDICTIONS, threshold_ratio=0.5)
        assert sig is not None
        assert len(sig.evidence["equivalent_actions"]) == 5  # all active

    def test_empty_predictions(self):
        sig = detect_prediction_equivalence({})
        assert sig is None


class TestLookaheadBlind:
    def test_sc25_fires(self):
        """sc25: everything zero — completely blind."""
        sig = detect_lookahead_blind(SC25_PREDICTIONS)
        assert sig is not None
        assert sig.signal == "lookahead_blind"
        assert sig.evidence["max_change"] == 0

    def test_sb26_fires(self):
        """sb26: only SEL=1px (trivial, below threshold) — blind."""
        sig = detect_lookahead_blind(SB26_PREDICTIONS)
        assert sig is not None

    def test_tr87_does_not_fire(self):
        """tr87: directional actions produce real effects — not blind."""
        sig = detect_lookahead_blind(TR87_PREDICTIONS)
        assert sig is None

    def test_cn04_does_not_fire(self):
        """cn04: all actions produce 100+ px — definitely not blind."""
        sig = detect_lookahead_blind(CN04_PREDICTIONS)
        assert sig is None

    def test_custom_threshold(self):
        """sb26 with threshold=0 should NOT fire (SEL=1 > 0)."""
        sig = detect_lookahead_blind(SB26_PREDICTIONS, blind_threshold=0)
        assert sig is None

    def test_empty_predictions(self):
        sig = detect_lookahead_blind({})
        assert sig is None


class TestPredictionsIgnored:
    def test_grounded_response_does_not_fire(self):
        """Response references predictions → no signal."""
        preds = {
            "UP": "no change — blocked",
            "LEFT": "201px changed in top-right (+13 red)",
        }
        response = (
            "UP is blocked as it only results in trivial changes. "
            "LEFT shows a change of 201px with 13 red pixels in the "
            "top-right region, which is the most significant change."
        )
        sig = detect_predictions_ignored(response, preds)
        assert sig is None

    def test_generic_response_fires(self):
        """Response ignores predictions entirely → fires."""
        preds = {
            "UP": "no change — blocked",
            "LEFT": "201px changed in top-right (+13 red)",
            "RIGHT": "201px changed in bottom-left (+8 blue)",
            "DOWN": "15px in center",
            "SEL": "51px paint event",
        }
        response = (
            "The CNN is stuck clicking the center. "
            "Moving to a new basket position to try a different approach."
        )
        sig = detect_predictions_ignored(response, preds)
        assert sig is not None
        assert sig.signal == "predictions_ignored"
        assert sig.evidence["reference_fraction"] < 0.2

    def test_partial_reference(self):
        """Some predictions referenced — depends on threshold."""
        preds = {
            "UP": "blocked",
            "LEFT": "201px change",
            "RIGHT": "201px change",
        }
        # References LEFT but not UP or RIGHT
        response = "LEFT shows 201px of change, choosing LEFT."
        sig = detect_predictions_ignored(response, preds, min_references=0.5)
        assert sig is not None  # only 1/3 referenced < 0.5
        sig2 = detect_predictions_ignored(response, preds, min_references=0.2)
        assert sig2 is None  # 1/3 = 0.33 ≥ 0.2

    def test_empty_response(self):
        sig = detect_predictions_ignored("", {"UP": "blocked"})
        assert sig is None

    def test_empty_predictions(self):
        sig = detect_predictions_ignored("some response", {})
        assert sig is None


class TestIntegration:
    def test_nomad_4game_detector_distribution(self):
        """Validate detector distribution matches game type analysis."""
        games = {
            "sb26": SB26_PREDICTIONS,
            "sc25": SC25_PREDICTIONS,
            "tr87": TR87_PREDICTIONS,
            "cn04": CN04_PREDICTIONS,
        }
        results = {}
        for name, preds in games.items():
            blind = detect_lookahead_blind(preds)
            equiv = detect_prediction_equivalence(preds)
            results[name] = {
                "blind": blind is not None,
                "equivalence": equiv is not None,
            }

        # sb26: blind (CLICK game, all blocked)
        assert results["sb26"]["blind"]
        assert not results["sb26"]["equivalence"]

        # sc25: blind (everything blocked)
        assert results["sc25"]["blind"]
        assert not results["sc25"]["equivalence"]

        # tr87: informative (clear spatial differences)
        assert not results["tr87"]["blind"]
        assert not results["tr87"]["equivalence"]

        # cn04: equivalent (UP/DOWN/LEFT/RIGHT/SEL all similar)
        assert not results["cn04"]["blind"]
        assert results["cn04"]["equivalence"]
