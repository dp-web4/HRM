#!/usr/bin/env python3
"""
Unit tests for router Phase 0 Track 2 — feature extraction.

Coverage (sprint-doc Track 2 acceptance criteria):

  1. All-present — every component supplies data.
  2. Sparse-present — WM + SNARC + metabolic only; others None.
  3. All-stubbed — every component None.
  4. Edge cases — empty WM, zero salience, missing goal_id.
  5. Determinism — same inputs → same RouterInput.
  6. Field correctness — each sub-extractor maps to the right
     RouterInput fields per PRD §3.1.
  7. Failure isolation — a sub-extractor whose component raises
     must not propagate; it returns defaults.
  8. Vector length invariant — RouterInput shape is fixed regardless
     of which optional components are None.

Run:
    python3 -m pytest sage/cognition/router/tests/test_feature_extraction.py -v
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from sage.cognition.router import (
    CARTRIDGE_EMBEDDING_DIM,
    RouterInput,
    VALID_ATP_TRENDS,
    VALID_METABOLIC_STATES,
)
from sage.cognition.router.feature_extraction import (
    _extract_cartridge_features,
    _extract_cerebellum_features,
    _extract_episodic_features,
    _extract_metabolic_features,
    _extract_metacog_features,
    _extract_rpe_features,
    _extract_sensory_features,
    _extract_snarc_features,
    _extract_wm_features,
    extract_router_input,
)

# Import real WM to exercise the actual public API contract.
from sage.cognition.working_memory import WorkingMemory


# ──────────────────────────────────────────────────────────────────────
# Lightweight stubs — each mimics the contract documented in the PRD
# for its component, without pulling in torch / numpy heavy deps.
# ──────────────────────────────────────────────────────────────────────

class _SNARCStub(dict):
    """SNARC can be passed as a dict or an attr-bag.  Test both."""


@dataclass
class _MetabolicStub:
    current_state: Any = "focus"
    atp_current: float = 65.0
    atp_trend: str = "stable"


class _EpisodicStub:
    def __init__(self, results: Optional[list] = None):
        self._results = results or []
        self.calls = 0

    def recall(self, cue, k: int = 5):
        self.calls += 1
        return list(self._results)[:k]


class _RecallStub:
    def __init__(self, similarity: float, reward: Optional[float]):
        self.similarity = similarity
        # Mirror the real RecallResult shape: `.episode` with `.reward`
        self.episode = SimpleNamespace(reward=reward)


class _HabitMatchStub:
    def __init__(self, confidence: float):
        self.confidence = confidence


class _CerebellumStub:
    def __init__(self, matches: Optional[list] = None):
        self._matches = matches or []
        self.calls = 0

    def lookup(self, state):
        self.calls += 1
        return list(self._matches)


class _RPEStub:
    def __init__(self, priors: Dict[str, float]):
        self.priors = priors
        self.calls = 0
        self.last_state_key: Optional[str] = None
        self.last_actions: Optional[list] = None

    def get_action_priors(self, state_key: str, actions: List[str]) -> Dict[str, float]:
        self.calls += 1
        self.last_state_key = state_key
        self.last_actions = list(actions)
        return {a: self.priors.get(a, 0.0) for a in actions}


class _MetacogStub:
    def __init__(self, block_list: List[str]):
        self.block_list = block_list


class _MetacogCallableStub:
    def __init__(self, block_list: List[str]):
        self._block_list = block_list
        self.calls = 0

    def get_block_list(self) -> List[str]:
        self.calls += 1
        return list(self._block_list)


class _RaisingStub:
    """Any attr access raises.  Used for failure-isolation tests."""

    def __getattr__(self, item):
        raise RuntimeError(f"boom-{item}")


# ──────────────────────────────────────────────────────────────────────
# WM sub-extractor
# ──────────────────────────────────────────────────────────────────────

def _make_wm() -> WorkingMemory:
    """Real WM with a goal + plan step + hypothesis."""
    wm = WorkingMemory(capacity=7)
    wm.add_item("goal", {"game": "toy_a"}, priority=0.9, goal_id="g1")
    wm.add_item("plan_step", {"step": 1}, priority=0.7, goal_id="g1")
    wm.add_item("hypothesis", {"claim": "A"}, priority=0.5, goal_id="g1")
    return wm


def test_wm_none_returns_defaults():
    out = _extract_wm_features(None)
    assert out["wm_state_key"] == "0" * 16
    assert out["wm_slot_counts"] == {}
    assert out["wm_goal_active"] is False
    assert out["wm_age_ticks"] == 0
    assert out["wm_pressure"] == 0.0


def test_wm_populated_returns_live_features():
    wm = _make_wm()
    out = _extract_wm_features(wm, goal_id="g1")

    # state_key is 16-char hex from WM
    assert isinstance(out["wm_state_key"], str) and len(out["wm_state_key"]) == 16
    assert out["wm_state_key"] == wm.stable_key("g1")

    # slot counts filtered to goal_id
    assert out["wm_slot_counts"]["goal"] == 1
    assert out["wm_slot_counts"]["plan_step"] == 1
    assert out["wm_slot_counts"]["hypothesis"] == 1

    # goal active
    assert out["wm_goal_active"] is True

    # pressure = 3/7
    assert abs(out["wm_pressure"] - 3 / 7) < 1e-9


def test_wm_goal_filtering_ignores_other_goals():
    wm = WorkingMemory(capacity=7)
    wm.add_item("goal", {"g": "A"}, priority=0.9, goal_id="A")
    wm.add_item("goal", {"g": "B"}, priority=0.9, goal_id="B")
    out = _extract_wm_features(wm, goal_id="A")
    # Only the A-goal should show up
    assert out["wm_slot_counts"].get("goal", 0) == 1


def test_wm_empty_returns_zero_pressure():
    wm = WorkingMemory(capacity=7)
    out = _extract_wm_features(wm)
    assert out["wm_pressure"] == 0.0
    assert out["wm_slot_counts"] == {}
    assert out["wm_goal_active"] is False


def test_wm_failure_isolation_returns_defaults():
    out = _extract_wm_features(_RaisingStub())
    # Should not raise; should return defaults.
    assert out["wm_state_key"] == "0" * 16
    assert out["wm_slot_counts"] == {}


def test_wm_missing_goal_id_does_not_filter():
    wm = WorkingMemory(capacity=7)
    wm.add_item("goal", "A", priority=0.9, goal_id="A")
    wm.add_item("goal", "B", priority=0.9, goal_id="B")
    out = _extract_wm_features(wm, goal_id=None)
    # Without filter, both goals count
    assert out["wm_slot_counts"].get("goal", 0) == 2
    assert out["wm_goal_active"] is True


# ──────────────────────────────────────────────────────────────────────
# SNARC sub-extractor
# ──────────────────────────────────────────────────────────────────────

def test_snarc_none_returns_zeros():
    out = _extract_snarc_features(None)
    assert out["snarc_surprise"] == 0.0
    assert out["snarc_novelty"] == 0.0
    assert out["snarc_arousal"] == 0.0
    assert out["snarc_reward"] == 0.0
    assert out["snarc_conflict"] == 0.0


def test_snarc_dict_input():
    snarc = {"surprise": 0.3, "novelty": 0.4, "arousal": 0.5, "reward": -0.2, "conflict": 0.1}
    out = _extract_snarc_features(snarc)
    assert out["snarc_surprise"] == 0.3
    assert out["snarc_reward"] == -0.2
    assert out["snarc_conflict"] == 0.1


def test_snarc_object_input():
    snarc = SimpleNamespace(surprise=0.9, novelty=0.8, arousal=0.7, reward=0.6, conflict=0.5)
    out = _extract_snarc_features(snarc)
    assert out["snarc_surprise"] == 0.9
    assert out["snarc_reward"] == 0.6


def test_snarc_clamps_out_of_range():
    # Above-1 should clamp to 1, below-0 should clamp to 0.
    snarc = {"surprise": 1.5, "novelty": -0.3, "arousal": 2.0, "reward": -5.0, "conflict": 99.0}
    out = _extract_snarc_features(snarc)
    assert out["snarc_surprise"] == 1.0
    assert out["snarc_novelty"] == 0.0
    assert out["snarc_arousal"] == 1.0
    assert out["snarc_reward"] == -1.0
    assert out["snarc_conflict"] == 1.0


def test_snarc_missing_keys_default_to_zero():
    snarc = {"surprise": 0.5}  # only one key supplied
    out = _extract_snarc_features(snarc)
    assert out["snarc_surprise"] == 0.5
    assert out["snarc_novelty"] == 0.0
    assert out["snarc_arousal"] == 0.0
    assert out["snarc_reward"] == 0.0
    assert out["snarc_conflict"] == 0.0


# ──────────────────────────────────────────────────────────────────────
# Metabolic sub-extractor
# ──────────────────────────────────────────────────────────────────────

def test_metabolic_none_returns_defaults():
    out = _extract_metabolic_features(None)
    assert out["metabolic_state"] == "wake"
    assert out["atp_level"] == 0.0
    assert out["atp_trend"] == "stable"


def test_metabolic_object_input():
    stub = _MetabolicStub(current_state="focus", atp_current=65.0, atp_trend="rising")
    out = _extract_metabolic_features(stub)
    assert out["metabolic_state"] == "focus"
    assert out["atp_level"] == 65.0
    assert out["atp_trend"] == "rising"


def test_metabolic_enum_like_state():
    enum_like = SimpleNamespace(value="dream")
    stub = _MetabolicStub(current_state=enum_like, atp_current=20.0, atp_trend="falling")
    out = _extract_metabolic_features(stub)
    assert out["metabolic_state"] == "dream"
    assert out["atp_trend"] == "falling"


def test_metabolic_invalid_state_falls_back():
    stub = _MetabolicStub(current_state="not_a_real_state")
    out = _extract_metabolic_features(stub)
    assert out["metabolic_state"] == "wake"


def test_metabolic_invalid_atp_trend_falls_back():
    stub = _MetabolicStub(atp_trend="sideways")
    out = _extract_metabolic_features(stub)
    assert out["atp_trend"] == "stable"


def test_metabolic_atp_clamped():
    stub = _MetabolicStub(atp_current=150.0)
    out = _extract_metabolic_features(stub)
    assert out["atp_level"] == 100.0
    stub = _MetabolicStub(atp_current=-10.0)
    out = _extract_metabolic_features(stub)
    assert out["atp_level"] == 0.0


def test_metabolic_all_valid_states():
    # Smoke test: all canonical states serialize as-is.
    for s in VALID_METABOLIC_STATES:
        stub = _MetabolicStub(current_state=s)
        out = _extract_metabolic_features(stub)
        assert out["metabolic_state"] == s


def test_metabolic_all_valid_trends():
    for t in VALID_ATP_TRENDS:
        stub = _MetabolicStub(atp_trend=t)
        out = _extract_metabolic_features(stub)
        assert out["atp_trend"] == t


def test_metabolic_dict_input():
    out = _extract_metabolic_features(
        {"current_state": "rest", "atp_current": 30.0, "atp_trend": "falling"}
    )
    assert out["metabolic_state"] == "rest"
    assert out["atp_level"] == 30.0
    assert out["atp_trend"] == "falling"


# ──────────────────────────────────────────────────────────────────────
# Episodic sub-extractor
# ──────────────────────────────────────────────────────────────────────

def test_episodic_none_returns_defaults():
    out = _extract_episodic_features(None)
    assert out["recall_count"] == 0
    assert out["recall_best_similarity"] == 0.0
    assert out["recall_best_outcome"] is None


def test_episodic_missing_cue_returns_defaults():
    # Component present but no cue → treated as "cannot query"
    stub = _EpisodicStub(results=[_RecallStub(0.9, 0.5)])
    out = _extract_episodic_features(stub, cue=None)
    assert out["recall_count"] == 0
    assert stub.calls == 0


def test_episodic_with_results():
    results = [_RecallStub(0.9, 0.3), _RecallStub(0.8, 0.1), _RecallStub(0.7, None)]
    stub = _EpisodicStub(results=results)
    out = _extract_episodic_features(stub, cue=object(), k=3)
    assert out["recall_count"] == 3
    assert out["recall_best_similarity"] == 0.9
    assert out["recall_best_outcome"] == 0.3
    assert stub.calls == 1


def test_episodic_empty_results():
    stub = _EpisodicStub(results=[])
    out = _extract_episodic_features(stub, cue=object())
    assert out["recall_count"] == 0
    assert out["recall_best_similarity"] == 0.0
    assert out["recall_best_outcome"] is None


def test_episodic_best_has_no_reward():
    results = [_RecallStub(0.9, None)]
    stub = _EpisodicStub(results=results)
    out = _extract_episodic_features(stub, cue=object())
    assert out["recall_count"] == 1
    assert out["recall_best_similarity"] == 0.9
    assert out["recall_best_outcome"] is None


def test_episodic_failure_isolation():
    class _Boom:
        def recall(self, cue, k=5):
            raise RuntimeError("boom")
    out = _extract_episodic_features(_Boom(), cue=object())
    assert out["recall_count"] == 0
    assert out["recall_best_similarity"] == 0.0


# ──────────────────────────────────────────────────────────────────────
# Cerebellum sub-extractor
# ──────────────────────────────────────────────────────────────────────

def test_cerebellum_none_returns_defaults():
    out = _extract_cerebellum_features(None)
    assert out["habit_available"] is False
    assert out["habit_confidence"] == 0.0


def test_cerebellum_missing_state_returns_defaults():
    # Component present but no state → cannot query.
    stub = _CerebellumStub(matches=[_HabitMatchStub(0.9)])
    out = _extract_cerebellum_features(stub, state=None)
    assert out["habit_available"] is False
    assert stub.calls == 0


def test_cerebellum_match_returns_available():
    matches = [_HabitMatchStub(0.85), _HabitMatchStub(0.75)]
    stub = _CerebellumStub(matches=matches)
    out = _extract_cerebellum_features(stub, state=object())
    assert out["habit_available"] is True
    assert out["habit_confidence"] == 0.85


def test_cerebellum_empty_match_list():
    stub = _CerebellumStub(matches=[])
    out = _extract_cerebellum_features(stub, state=object())
    assert out["habit_available"] is False
    assert out["habit_confidence"] == 0.0


def test_cerebellum_clamps_confidence():
    matches = [_HabitMatchStub(1.5)]
    stub = _CerebellumStub(matches=matches)
    out = _extract_cerebellum_features(stub, state=object())
    assert out["habit_confidence"] == 1.0


def test_cerebellum_failure_isolation():
    class _Boom:
        def lookup(self, s):
            raise RuntimeError("boom")
    out = _extract_cerebellum_features(_Boom(), state=object())
    assert out["habit_available"] is False


# ──────────────────────────────────────────────────────────────────────
# RPE sub-extractor
# ──────────────────────────────────────────────────────────────────────

def test_rpe_none_returns_uniform_prior():
    out = _extract_rpe_features(None, state_key="state")
    # Uniform over 3 classes → 1/3 each
    for k in ("prior_invoke", "prior_habit", "prior_noop"):
        assert abs(out[k] - 1 / 3) < 1e-9


def test_rpe_missing_state_key_returns_uniform():
    stub = _RPEStub({"invoke": 0.6, "habit": 0.3, "noop": 0.1})
    out = _extract_rpe_features(stub, state_key=None)
    for k in ("prior_invoke", "prior_habit", "prior_noop"):
        assert abs(out[k] - 1 / 3) < 1e-9
    assert stub.calls == 0


def test_rpe_reads_priors():
    stub = _RPEStub({"invoke": 0.5, "habit": 0.3, "noop": 0.2})
    out = _extract_rpe_features(stub, state_key="sk-1")
    assert out["prior_invoke"] == 0.5
    assert out["prior_habit"] == 0.3
    assert out["prior_noop"] == 0.2
    assert stub.calls == 1
    assert stub.last_state_key == "sk-1"
    assert stub.last_actions == ["invoke", "habit", "noop"]


def test_rpe_priors_clamped_to_unit():
    stub = _RPEStub({"invoke": 1.5, "habit": -0.3, "noop": 0.4})
    out = _extract_rpe_features(stub, state_key="sk-1")
    assert out["prior_invoke"] == 1.0
    assert out["prior_habit"] == 0.0
    assert out["prior_noop"] == 0.4


def test_rpe_failure_isolation():
    class _Boom:
        def get_action_priors(self, sk, actions):
            raise RuntimeError("boom")
    out = _extract_rpe_features(_Boom(), state_key="sk")
    # Falls back to uniform
    for k in ("prior_invoke", "prior_habit", "prior_noop"):
        assert abs(out[k] - 1 / 3) < 1e-9


# ──────────────────────────────────────────────────────────────────────
# Metacog sub-extractor
# ──────────────────────────────────────────────────────────────────────

def test_metacog_none_returns_empty():
    out = _extract_metacog_features(None)
    assert out["metacog_block_list"] == []


def test_metacog_attr_list():
    stub = _MetacogStub(["vision_plugin", "audio_plugin"])
    out = _extract_metacog_features(stub)
    assert out["metacog_block_list"] == ["vision_plugin", "audio_plugin"]


def test_metacog_callable_get_block_list():
    stub = _MetacogCallableStub(["llm_plugin"])
    out = _extract_metacog_features(stub)
    assert out["metacog_block_list"] == ["llm_plugin"]
    assert stub.calls == 1


def test_metacog_dict_input():
    out = _extract_metacog_features({"block_list": ["a", "b"]})
    assert out["metacog_block_list"] == ["a", "b"]


def test_metacog_list_input():
    out = _extract_metacog_features(["a", "b", "c"])
    assert out["metacog_block_list"] == ["a", "b", "c"]


def test_metacog_filters_non_strings():
    # Non-strings are dropped
    out = _extract_metacog_features([1, "a", None, "b", 2.5])
    assert out["metacog_block_list"] == ["a", "b"]


def test_metacog_failure_isolation():
    class _Boom:
        @property
        def block_list(self):
            raise RuntimeError("boom")
    out = _extract_metacog_features(_Boom())
    assert out["metacog_block_list"] == []


# ──────────────────────────────────────────────────────────────────────
# Sensory sub-extractor
# ──────────────────────────────────────────────────────────────────────

def test_sensory_none_returns_defaults():
    out = _extract_sensory_features(None)
    assert out["sensory_modalities"] == []
    assert out["sensory_novelty"] == 0.0
    assert out["sensory_urgency"] == 0.0


def test_sensory_dict_list_modalities():
    sensory = {"modalities": ["vision", "time"], "novelty": 0.4, "urgency": 0.2}
    out = _extract_sensory_features(sensory)
    assert out["sensory_modalities"] == ["vision", "time"]
    assert out["sensory_novelty"] == 0.4
    assert out["sensory_urgency"] == 0.2


def test_sensory_dict_dict_modalities():
    # Sometimes modality dicts carry metadata; we only take the keys
    sensory = {"modalities": {"vision": {"foo": 1}, "time": {}}, "novelty": 0.5, "urgency": 0.5}
    out = _extract_sensory_features(sensory)
    assert set(out["sensory_modalities"]) == {"vision", "time"}


def test_sensory_object_input():
    sensory = SimpleNamespace(modalities=["message"], novelty=0.1, urgency=0.9)
    out = _extract_sensory_features(sensory)
    assert out["sensory_modalities"] == ["message"]
    assert out["sensory_novelty"] == 0.1
    assert out["sensory_urgency"] == 0.9


def test_sensory_clamps_and_filters():
    sensory = {
        "modalities": ["vision", 42, None, "time"],
        "novelty": 1.5,
        "urgency": -0.3,
    }
    out = _extract_sensory_features(sensory)
    assert out["sensory_modalities"] == ["vision", "time"]
    assert out["sensory_novelty"] == 1.0
    assert out["sensory_urgency"] == 0.0


# ──────────────────────────────────────────────────────────────────────
# Cartridge sub-extractor (Phase 0 stub)
# ──────────────────────────────────────────────────────────────────────

def test_cartridge_none_returns_stub():
    out = _extract_cartridge_features(None)
    assert out["cartridge_recall_count"] == 0
    assert out["cartridge_recall_best_similarity"] == 0.0
    assert out["cartridge_recall_embedding"] == [0.0] * CARTRIDGE_EMBEDDING_DIM
    assert len(out["cartridge_recall_embedding"]) == CARTRIDGE_EMBEDDING_DIM


def test_cartridge_present_still_returns_stub_at_phase_0():
    # Phase 0: we never consume a real cartridge.  Guaranteed stub.
    stub = SimpleNamespace(search=lambda *a, **k: [])
    out = _extract_cartridge_features(stub)
    assert out["cartridge_recall_count"] == 0
    assert out["cartridge_recall_best_similarity"] == 0.0
    assert out["cartridge_recall_embedding"] == [0.0] * CARTRIDGE_EMBEDDING_DIM


# ──────────────────────────────────────────────────────────────────────
# Top-level extract_router_input
# ──────────────────────────────────────────────────────────────────────

def _all_present_kit(goal_id: str = "g1"):
    """Build a full component kit for 'all-present' tests.

    Returns a dict: {wm, snarc, metabolic, episodic, cerebellum, rpe,
                     metacog, sensory, episodic_cue, cerebellum_state}.
    """
    wm = _make_wm()
    snarc = {"surprise": 0.3, "novelty": 0.4, "arousal": 0.5, "reward": -0.2, "conflict": 0.1}
    metabolic = _MetabolicStub(current_state="focus", atp_current=65.0, atp_trend="stable")
    episodic = _EpisodicStub(results=[_RecallStub(0.85, 0.2), _RecallStub(0.75, None)])
    cerebellum = _CerebellumStub(matches=[_HabitMatchStub(0.88)])
    rpe = _RPEStub({"invoke": 0.5, "habit": 0.3, "noop": 0.2})
    metacog = _MetacogStub(["perseverating_plugin"])
    sensory = {"modalities": ["vision", "time"], "novelty": 0.1, "urgency": 0.2}
    return dict(
        wm=wm,
        snarc=snarc,
        metabolic=metabolic,
        episodic=episodic,
        cerebellum=cerebellum,
        rpe=rpe,
        metacog=metacog,
        sensory=sensory,
        episodic_cue=object(),  # placeholder; episodic stub ignores content
        cerebellum_state=object(),
    )


def test_all_present_extraction_produces_valid_routerinput():
    kit = _all_present_kit()
    ri = extract_router_input(tick=42, goal_id="g1", timestamp=1700000000.0, **kit)
    assert isinstance(ri, RouterInput)
    assert ri.tick == 42
    assert ri.timestamp == 1700000000.0
    assert ri.goal_id == "g1"
    # Fields populated from sub-extractors
    assert ri.metabolic_state == "focus"
    assert ri.atp_level == 65.0
    assert ri.atp_trend == "stable"
    assert ri.snarc_surprise == 0.3
    assert ri.snarc_reward == -0.2
    assert ri.recall_count == 2
    assert ri.recall_best_similarity == 0.85
    assert ri.recall_best_outcome == 0.2
    assert ri.habit_available is True
    assert ri.habit_confidence == 0.88
    assert ri.prior_invoke == 0.5
    assert ri.prior_habit == 0.3
    assert ri.prior_noop == 0.2
    assert ri.metacog_block_list == ["perseverating_plugin"]
    assert ri.sensory_modalities == ["vision", "time"]
    assert ri.wm_goal_active is True
    # Cartridge is always stubbed at Phase 0
    assert ri.cartridge_recall_count == 0
    assert ri.cartridge_recall_best_similarity == 0.0
    assert len(ri.cartridge_recall_embedding) == CARTRIDGE_EMBEDDING_DIM


def test_sparse_present_wm_snarc_metabolic_only():
    """Sprint-doc Track 2 acceptance: sparse-present — WM + SNARC +
    metabolic; other components None."""
    ri = extract_router_input(
        wm=_make_wm(),
        snarc={"surprise": 0.5, "novelty": 0.5, "arousal": 0.5, "reward": 0.0, "conflict": 0.0},
        metabolic=_MetabolicStub(current_state="wake", atp_current=40.0),
        episodic=None,
        cerebellum=None,
        rpe=None,
        metacog=None,
        sensory=None,
        tick=5,
        goal_id="g1",
    )
    # WM populated
    assert ri.wm_goal_active is True
    # SNARC populated
    assert ri.snarc_surprise == 0.5
    # Metabolic populated
    assert ri.metabolic_state == "wake"
    # Missing components use defaults
    assert ri.recall_count == 0
    assert ri.recall_best_outcome is None
    assert ri.habit_available is False
    # Priors: uniform
    assert abs(ri.prior_invoke - 1 / 3) < 1e-9
    # Metacog and sensory empty
    assert ri.metacog_block_list == []
    assert ri.sensory_modalities == []
    # Cartridge stub
    assert len(ri.cartridge_recall_embedding) == CARTRIDGE_EMBEDDING_DIM


def test_all_stubbed_every_component_none():
    """Sprint-doc Track 2 acceptance: all-stubbed — every component
    None."""
    ri = extract_router_input(tick=1)
    assert isinstance(ri, RouterInput)
    assert ri.tick == 1
    assert ri.goal_id is None
    assert ri.wm_state_key == "0" * 16
    assert ri.wm_slot_counts == {}
    assert ri.wm_goal_active is False
    assert ri.wm_age_ticks == 0
    assert ri.wm_pressure == 0.0
    assert ri.sensory_modalities == []
    assert ri.sensory_novelty == 0.0
    assert ri.sensory_urgency == 0.0
    assert ri.snarc_surprise == 0.0
    assert ri.snarc_novelty == 0.0
    assert ri.snarc_arousal == 0.0
    assert ri.snarc_reward == 0.0
    assert ri.snarc_conflict == 0.0
    assert ri.metabolic_state == "wake"
    assert ri.atp_level == 0.0
    assert ri.atp_trend == "stable"
    assert ri.recall_count == 0
    assert ri.recall_best_similarity == 0.0
    assert ri.recall_best_outcome is None
    assert ri.habit_available is False
    assert ri.habit_confidence == 0.0
    assert abs(ri.prior_invoke - 1 / 3) < 1e-9
    assert abs(ri.prior_habit - 1 / 3) < 1e-9
    assert abs(ri.prior_noop - 1 / 3) < 1e-9
    assert ri.metacog_block_list == []
    assert ri.cartridge_recall_count == 0
    assert ri.cartridge_recall_best_similarity == 0.0
    assert len(ri.cartridge_recall_embedding) == CARTRIDGE_EMBEDDING_DIM
    # All zeros
    assert all(v == 0.0 for v in ri.cartridge_recall_embedding)


def test_edge_case_empty_wm_zero_salience_no_goal():
    """Edge case: empty WM, zero SNARC, no goal_id (sprint-doc Track 2)."""
    wm = WorkingMemory(capacity=7)  # empty
    snarc = {"surprise": 0.0, "novelty": 0.0, "arousal": 0.0, "reward": 0.0, "conflict": 0.0}
    ri = extract_router_input(
        wm=wm,
        snarc=snarc,
        metabolic=_MetabolicStub(current_state="rest", atp_current=10.0, atp_trend="falling"),
        tick=0,
        goal_id=None,
    )
    assert ri.wm_goal_active is False
    assert ri.wm_pressure == 0.0
    assert ri.wm_slot_counts == {}
    assert ri.snarc_arousal == 0.0
    # state_key is deterministic even for empty WM
    assert isinstance(ri.wm_state_key, str)
    assert len(ri.wm_state_key) == 16
    assert ri.goal_id is None


def test_determinism_same_inputs_same_output():
    """Sprint-doc Track 2 acceptance: determinism — same inputs →
    same RouterInput."""
    kit = _all_present_kit()
    ri1 = extract_router_input(tick=7, goal_id="g1", timestamp=1700000000.0, **kit)
    ri2 = extract_router_input(tick=7, goal_id="g1", timestamp=1700000000.0, **kit)
    # Same inputs → identical serializations.
    assert ri1.to_dict() == ri2.to_dict()


def test_feature_vector_length_fixed_regardless_of_none_components():
    """Sprint-doc Track 2 acceptance: feature vector length is fixed
    regardless of which optional components are None.

    We verify this by round-tripping to dict and confirming the key
    set is identical across every combination.
    """
    combos = [
        # All None
        dict(tick=1),
        # WM only
        dict(tick=1, wm=_make_wm()),
        # SNARC + metabolic
        dict(tick=1, snarc={"surprise": 0.5}, metabolic=_MetabolicStub()),
        # Everything
        dict(tick=1, **_all_present_kit()),
    ]
    key_sets = []
    embedding_lengths = []
    for c in combos:
        ri = extract_router_input(**c)
        d = ri.to_dict()
        key_sets.append(frozenset(d.keys()))
        embedding_lengths.append(len(ri.cartridge_recall_embedding))

    # All combos produce the same RouterInput key-set (fixed shape)
    assert len(set(key_sets)) == 1, f"key sets differ: {key_sets}"

    # Cartridge embedding always 768
    assert all(length == CARTRIDGE_EMBEDDING_DIM for length in embedding_lengths)


def test_timestamp_defaults_to_now():
    t0 = time.time()
    ri = extract_router_input(tick=1)
    t1 = time.time()
    assert t0 - 1.0 <= ri.timestamp <= t1 + 1.0


def test_goal_id_optional_and_forwarded():
    ri = extract_router_input(tick=1, goal_id=None)
    assert ri.goal_id is None
    ri = extract_router_input(tick=1, goal_id="g-xyz")
    assert ri.goal_id == "g-xyz"


def test_negative_tick_rejected_by_routerinput():
    # Boundary contract: the extractor passes its tick through to
    # RouterInput.__post_init__, which raises on negative values.
    with pytest.raises(ValueError):
        extract_router_input(tick=-1)


def test_rpe_uses_wm_state_key_as_state_key():
    """The RPE sub-extractor should be called with the state key
    derived from WM — not from the caller.  This is the binding that
    guarantees serving-time features match training-time features
    (PRD §4.7.G: no training-serving skew)."""
    wm = _make_wm()
    rpe = _RPEStub({"invoke": 0.5, "habit": 0.3, "noop": 0.2})
    ri = extract_router_input(
        wm=wm,
        rpe=rpe,
        tick=1,
        goal_id="g1",
    )
    assert rpe.last_state_key == wm.stable_key("g1")
    assert ri.wm_state_key == wm.stable_key("g1")


def test_cartridge_kwarg_passes_without_effect():
    # ``cartridge`` is plumbed through so a future track can wire it;
    # at Phase 0 it must not change output.
    fixed_ts = 1700000000.0
    ri_none = extract_router_input(tick=1, timestamp=fixed_ts)
    ri_passed = extract_router_input(
        tick=1, timestamp=fixed_ts, cartridge=SimpleNamespace(search=lambda *a, **k: [])
    )
    assert ri_none.to_dict() == ri_passed.to_dict()


def test_plugin_registry_kwarg_noop_at_phase_0():
    # ``plugin_registry`` is reserved; must not change output.
    fixed_ts = 1700000000.0
    ri_none = extract_router_input(tick=1, timestamp=fixed_ts)
    ri_passed = extract_router_input(
        tick=1, timestamp=fixed_ts, plugin_registry=SimpleNamespace(names=lambda: ["p1"])
    )
    assert ri_none.to_dict() == ri_passed.to_dict()


def test_full_json_roundtrip():
    """Extracted RouterInput round-trips through to_dict/from_dict."""
    kit = _all_present_kit()
    ri = extract_router_input(tick=99, goal_id="g1", timestamp=1234567.0, **kit)
    d = ri.to_dict()
    ri2 = RouterInput.from_dict(d)
    assert ri.to_dict() == ri2.to_dict()


def test_extractor_does_not_mutate_inputs():
    """Sub-extractors are pure — running extraction twice must not
    change the underlying components."""
    kit = _all_present_kit()
    wm = kit["wm"]
    # Snapshot key WM state
    slots_before = {sid: slot.content_type for sid, slot in wm.slots.items()}
    total_adds_before = wm.total_adds
    total_accesses_before = wm.total_accesses

    # Extract twice
    extract_router_input(tick=1, goal_id="g1", **kit)
    extract_router_input(tick=2, goal_id="g1", **kit)

    # WM slot map unchanged
    slots_after = {sid: slot.content_type for sid, slot in wm.slots.items()}
    assert slots_before == slots_after
    # No new adds
    assert wm.total_adds == total_adds_before
    # No read-side accesses (we don't go through wm.get_item)
    assert wm.total_accesses == total_accesses_before


def test_extractor_recovers_from_broken_component():
    """If a single component raises, the extractor must still return
    a valid RouterInput with defaults for that component."""
    # Broken episodic.
    class _BadEpisodic:
        def recall(self, cue, k=5):
            raise RuntimeError("episodic-boom")

    ri = extract_router_input(
        wm=_make_wm(),
        snarc={"surprise": 0.5},
        metabolic=_MetabolicStub(),
        episodic=_BadEpisodic(),
        episodic_cue=object(),
        cerebellum=_CerebellumStub(matches=[_HabitMatchStub(0.8)]),
        cerebellum_state=object(),
        tick=1,
        goal_id="g1",
    )
    # Episodic defaulted
    assert ri.recall_count == 0
    assert ri.recall_best_similarity == 0.0
    # Cerebellum still works
    assert ri.habit_available is True
    assert ri.habit_confidence == 0.8


def test_missing_cue_and_state_means_optional_components_inert():
    """Providing episodic + cerebellum but no cue/state means they
    are inert — the extractor reads defaults rather than calling them."""
    episodic = _EpisodicStub(results=[_RecallStub(0.9, 0.5)])
    cerebellum = _CerebellumStub(matches=[_HabitMatchStub(0.9)])
    ri = extract_router_input(
        wm=_make_wm(),
        episodic=episodic,
        cerebellum=cerebellum,
        tick=1,
        goal_id="g1",
        episodic_cue=None,
        cerebellum_state=None,
    )
    assert ri.recall_count == 0
    assert ri.habit_available is False
    assert episodic.calls == 0
    assert cerebellum.calls == 0
