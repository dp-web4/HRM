#!/usr/bin/env python3
"""
Unit tests for router Phase 0 Track 1 schemas.

Covers:
  - Construction with valid args (each dataclass)
  - PRD §3.3 validation rules (per rule)
  - JSON round-trip for every dataclass
  - Schema versioning (required + non-empty)
  - PluginTier enum coverage
  - Validator edge cases called out in the sprint doc

Run: ``python3 -m pytest sage/cognition/router/tests/test_schemas.py -v``
"""

import json
import time

import pytest

from sage.cognition.router import (
    CARTRIDGE_EMBEDDING_DIM,
    Event,
    PluginTier,
    ROUTER_SCHEMA_VERSION,
    RouterInput,
    RouterOutput,
    RouterRecord,
    VALID_ACTIONS,
    VALID_EVENT_KINDS,
    VALID_RATIONALE_CODES,
)
from sage.cognition.router.events import ROUTER_KINDS, WM_COMPATIBLE_KINDS


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

def _sample_router_input(**overrides) -> RouterInput:
    defaults = dict(
        tick=42,
        timestamp=1700000000.0,
        goal_id="goal-xyz",
        wm_state_key="0123456789abcdef",
        wm_slot_counts={"goal": 1, "plan_step": 2},
        wm_goal_active=True,
        wm_age_ticks=3,
        wm_pressure=0.5,
        sensory_modalities=["vision", "time"],
        sensory_novelty=0.1,
        sensory_urgency=0.2,
        snarc_surprise=0.3,
        snarc_novelty=0.4,
        snarc_arousal=0.5,
        snarc_reward=-0.2,
        snarc_conflict=0.1,
        metabolic_state="focus",
        atp_level=65.0,
        atp_trend="stable",
        recall_count=2,
        recall_best_similarity=0.7,
        recall_best_outcome=0.3,
        habit_available=True,
        habit_confidence=0.8,
        prior_invoke=0.4,
        prior_habit=0.4,
        prior_noop=0.2,
        metacog_block_list=["vision_plugin"],
    )
    defaults.update(overrides)
    return RouterInput(**defaults)


def _sample_invoke() -> RouterOutput:
    return RouterOutput(
        action="invoke",
        plugin="vision_plugin",
        plugin_tier=PluginTier.ROUTINE.value,
        payload_hint="frame_diff",
        habit_id=None,
        confidence=0.9,
        energy_estimate=8.0,
        rationale_code="high_novelty",
    )


def _sample_habit() -> RouterOutput:
    return RouterOutput(
        action="habit",
        plugin=None,
        plugin_tier=None,
        payload_hint=None,
        habit_id="habit-abc",
        confidence=0.85,
        energy_estimate=0.5,
        rationale_code="habit_match",
    )


def _sample_noop() -> RouterOutput:
    return RouterOutput.noop(rationale_code="low_atp_rest")


# ──────────────────────────────────────────────────────────────────────
# PluginTier enum
# ──────────────────────────────────────────────────────────────────────

def test_plugin_tier_values_cover_prd_spec():
    """PRD §1.2 lists five tiers."""
    assert set(PluginTier.values()) == {
        "reflex", "routine", "specialized", "frontal_lobe", "federate"
    }


def test_plugin_tier_is_valid_accepts_known_rejects_unknown():
    assert PluginTier.is_valid("reflex") is True
    assert PluginTier.is_valid("frontal_lobe") is True
    assert PluginTier.is_valid("hyperintelligence") is False
    assert PluginTier.is_valid("") is False


def test_plugin_tier_str_equality_and_json():
    """str-Enum means the value is its own string."""
    assert PluginTier.FRONTAL_LOBE == "frontal_lobe"
    assert json.dumps(PluginTier.FRONTAL_LOBE.value) == '"frontal_lobe"'


# ──────────────────────────────────────────────────────────────────────
# Event
# ──────────────────────────────────────────────────────────────────────

def test_event_accepts_wm_compatible_kinds():
    for kind in WM_COMPATIBLE_KINDS:
        ev = Event(kind=kind, slot_id=None, slot_type=None, timestamp=time.time())
        assert ev.kind == kind


def test_event_accepts_router_specific_kinds():
    for kind in ROUTER_KINDS:
        ev = Event(kind=kind, slot_id="rid", slot_type="route", timestamp=time.time())
        assert ev.kind == kind
        assert ev.kind in VALID_EVENT_KINDS


def test_event_rejects_unknown_kind():
    with pytest.raises(ValueError):
        Event(kind="gossip", slot_id=None, slot_type=None, timestamp=time.time())


def test_event_roundtrip():
    ev = Event(
        kind="route",
        slot_id="rec-1",
        slot_type="invoke",
        timestamp=1700000000.5,
        reason="routine tier",
    )
    d = ev.to_dict()
    s = json.dumps(d)
    back = Event.from_dict(json.loads(s))
    assert back == ev


# ──────────────────────────────────────────────────────────────────────
# RouterInput — construction + validation
# ──────────────────────────────────────────────────────────────────────

def test_router_input_construction_defaults_cartridge_stubs():
    """Phase 0 stub: cartridge_recall_* default to 'no recall'."""
    ri = _sample_router_input()
    assert ri.cartridge_recall_count == 0
    assert ri.cartridge_recall_best_similarity == 0.0
    assert len(ri.cartridge_recall_embedding) == CARTRIDGE_EMBEDDING_DIM
    assert all(v == 0.0 for v in ri.cartridge_recall_embedding)


def test_router_input_rejects_out_of_range_unit():
    with pytest.raises(ValueError):
        _sample_router_input(snarc_surprise=1.5)
    with pytest.raises(ValueError):
        _sample_router_input(wm_pressure=-0.1)


def test_router_input_accepts_reward_in_signed_range():
    # -1..1 is the PRD range for snarc_reward
    _sample_router_input(snarc_reward=-1.0)
    _sample_router_input(snarc_reward=1.0)
    with pytest.raises(ValueError):
        _sample_router_input(snarc_reward=-1.01)
    with pytest.raises(ValueError):
        _sample_router_input(snarc_reward=1.01)


def test_router_input_rejects_unknown_metabolic_state():
    with pytest.raises(ValueError):
        _sample_router_input(metabolic_state="euphoric")


def test_router_input_rejects_unknown_atp_trend():
    with pytest.raises(ValueError):
        _sample_router_input(atp_trend="oscillating")


def test_router_input_rejects_wrong_cartridge_embedding_dim():
    with pytest.raises(ValueError):
        _sample_router_input(cartridge_recall_embedding=[0.0] * 512)


def test_router_input_rejects_empty_wm_state_key():
    with pytest.raises(ValueError):
        _sample_router_input(wm_state_key="")


def test_router_input_rejects_negative_tick():
    with pytest.raises(ValueError):
        _sample_router_input(tick=-1)


def test_router_input_roundtrip_json():
    ri = _sample_router_input()
    s = json.dumps(ri.to_dict())
    back = RouterInput.from_dict(json.loads(s))
    assert back == ri


def test_router_input_from_dict_ignores_unknown_keys():
    """Forward compatibility: new fields in the payload don't break old readers."""
    ri = _sample_router_input()
    d = ri.to_dict()
    d["new_field_from_future_schema"] = 42
    back = RouterInput.from_dict(d)
    assert back == ri


def test_router_input_from_dict_fills_missing_cartridge_defaults():
    """Old records written before cartridge fields existed still load."""
    ri = _sample_router_input()
    d = ri.to_dict()
    d.pop("cartridge_recall_count")
    d.pop("cartridge_recall_best_similarity")
    d.pop("cartridge_recall_embedding")
    back = RouterInput.from_dict(d)
    assert back.cartridge_recall_count == 0
    assert back.cartridge_recall_best_similarity == 0.0
    assert len(back.cartridge_recall_embedding) == CARTRIDGE_EMBEDDING_DIM


# ──────────────────────────────────────────────────────────────────────
# RouterOutput — construction + validation (PRD §3.3)
# ──────────────────────────────────────────────────────────────────────

def test_router_output_invoke_valid():
    out = _sample_invoke()
    assert out.is_valid()


def test_router_output_habit_valid():
    out = _sample_habit()
    assert out.is_valid()


def test_router_output_noop_valid():
    out = _sample_noop()
    assert out.is_valid()


def test_router_output_rule_1_invoke_requires_plugin():
    """PRD §3.3 rule 1: invoke without plugin is invalid."""
    out = RouterOutput(
        action="invoke", plugin=None, plugin_tier=None, payload_hint=None,
        habit_id=None, confidence=0.5, energy_estimate=1.0,
        rationale_code="default",
    )
    ok, reason = out.validate()
    assert not ok
    assert reason == "invoke_missing_plugin"
    with pytest.raises(ValueError):
        out.validate_or_raise()


def test_router_output_rule_1_invoke_unknown_plugin():
    """PRD §3.3 rule 1: plugin must be in the registry when provided."""
    out = _sample_invoke()
    ok, reason = out.validate(known_plugins={"other_plugin"})
    assert not ok
    assert reason == "invoke_unknown_plugin"


def test_router_output_rule_2_habit_requires_habit_id():
    """PRD §3.3 rule 2: habit without habit_id is invalid."""
    out = RouterOutput(
        action="habit", plugin=None, plugin_tier=None, payload_hint=None,
        habit_id=None, confidence=0.9, energy_estimate=0.5,
        rationale_code="habit_match",
    )
    ok, reason = out.validate()
    assert not ok
    assert reason == "habit_missing_habit_id"


def test_router_output_rule_2_habit_unknown_id():
    out = _sample_habit()
    ok, reason = out.validate(known_habits={"other_habit"})
    assert not ok
    assert reason == "habit_unknown_habit_id"


def test_router_output_rule_3_noop_rejects_optional_fields():
    """PRD §3.3 rule 3: noop with optional fields set is invalid."""
    out = RouterOutput(
        action="noop", plugin="vision_plugin", plugin_tier="routine",
        payload_hint=None, habit_id=None, confidence=1.0,
        energy_estimate=0.0, rationale_code="default",
    )
    ok, reason = out.validate()
    assert not ok
    assert reason == "noop_with_optional_fields"


def test_router_output_rule_4_confidence_range():
    """PRD §3.3 rule 4: confidence ∈ [0, 1]."""
    out = RouterOutput(
        action="noop", plugin=None, plugin_tier=None, payload_hint=None,
        habit_id=None, confidence=1.5, energy_estimate=0.0,
        rationale_code="default",
    )
    ok, reason = out.validate()
    assert not ok
    assert reason == "confidence_out_of_range"


def test_router_output_rule_4_energy_nonneg():
    """PRD §3.3 rule 4: energy_estimate ≥ 0."""
    out = RouterOutput(
        action="noop", plugin=None, plugin_tier=None, payload_hint=None,
        habit_id=None, confidence=0.5, energy_estimate=-0.1,
        rationale_code="default",
    )
    ok, reason = out.validate()
    assert not ok
    assert reason == "energy_estimate_negative"


def test_router_output_rejects_unknown_action():
    out = RouterOutput(
        action="meditate", plugin=None, plugin_tier=None, payload_hint=None,
        habit_id=None, confidence=0.5, energy_estimate=0.0,
        rationale_code="default",
    )
    ok, reason = out.validate()
    assert not ok
    assert reason == "unknown_action"


def test_router_output_rejects_unknown_tier_value():
    out = RouterOutput(
        action="invoke", plugin="plug", plugin_tier="mesozoic",
        payload_hint=None, habit_id=None, confidence=0.5,
        energy_estimate=1.0, rationale_code="default",
    )
    ok, reason = out.validate()
    assert not ok
    assert reason == "unknown_plugin_tier"


def test_router_output_rejects_unknown_rationale():
    out = RouterOutput(
        action="noop", plugin=None, plugin_tier=None, payload_hint=None,
        habit_id=None, confidence=1.0, energy_estimate=0.0,
        rationale_code="vibes",
    )
    ok, reason = out.validate()
    assert not ok
    assert reason == "unknown_rationale_code"


def test_router_output_invoke_with_habit_id_is_invalid():
    """Cross-field: invoke must not carry habit_id."""
    out = RouterOutput(
        action="invoke", plugin="vision_plugin", plugin_tier="routine",
        payload_hint=None, habit_id="h1", confidence=0.5,
        energy_estimate=1.0, rationale_code="default",
    )
    ok, reason = out.validate()
    assert not ok
    assert reason == "invoke_with_habit_id"


def test_router_output_habit_with_plugin_fields_is_invalid():
    """Cross-field: habit must not carry plugin-side fields."""
    out = RouterOutput(
        action="habit", plugin="vision_plugin", plugin_tier=None,
        payload_hint=None, habit_id="h1", confidence=0.5,
        energy_estimate=1.0, rationale_code="habit_match",
    )
    ok, reason = out.validate()
    assert not ok
    assert reason == "habit_with_plugin_fields"


def test_router_output_coerce_to_noop_produces_valid_noop():
    bad = RouterOutput(
        action="invoke", plugin=None, plugin_tier=None, payload_hint=None,
        habit_id=None, confidence=0.5, energy_estimate=1.0,
        rationale_code="default",
    )
    coerced = RouterOutput.coerce_to_noop(bad, reason="invoke_missing_plugin")
    assert coerced.action == "noop"
    assert coerced.is_valid()
    assert coerced.rationale_code == "coerced_noop"


def test_router_output_roundtrip_each_action():
    for out in (_sample_invoke(), _sample_habit(), _sample_noop()):
        s = json.dumps(out.to_dict())
        back = RouterOutput.from_dict(json.loads(s))
        assert back == out


# ──────────────────────────────────────────────────────────────────────
# RouterRecord — schema versioning + round-trip
# ──────────────────────────────────────────────────────────────────────

def test_router_record_schema_version_is_set_by_default():
    rec = RouterRecord(
        router_input=_sample_router_input(),
        router_output=_sample_noop(),
        machine="cbp",
    )
    assert rec.schema_version == ROUTER_SCHEMA_VERSION
    assert rec.schema_version  # non-empty


def test_router_record_schema_version_must_be_non_empty():
    with pytest.raises(ValueError):
        RouterRecord(
            router_input=_sample_router_input(),
            router_output=_sample_noop(),
            schema_version="",
        )


def test_router_record_record_id_must_be_non_empty():
    with pytest.raises(ValueError):
        RouterRecord(
            router_input=_sample_router_input(),
            router_output=_sample_noop(),
            record_id="",
        )


def test_router_record_outcome_defaults_to_none():
    rec = RouterRecord(
        router_input=_sample_router_input(),
        router_output=_sample_noop(),
    )
    assert rec.outcome is None


def test_router_record_roundtrip_json():
    rec = RouterRecord(
        router_input=_sample_router_input(),
        router_output=_sample_invoke(),
        machine="sprout",
    )
    s = json.dumps(rec.to_dict())
    back = RouterRecord.from_dict(json.loads(s))
    assert back.record_id == rec.record_id
    assert back.schema_version == rec.schema_version
    assert back.timestamp == rec.timestamp
    assert back.machine == rec.machine
    assert back.router_input == rec.router_input
    assert back.router_output == rec.router_output
    assert back.outcome == rec.outcome


def test_router_record_with_outcome_is_non_mutating():
    rec = RouterRecord(
        router_input=_sample_router_input(),
        router_output=_sample_invoke(),
        machine="legion",
    )
    outcome = {"snarc_resolution": 0.4, "rpe_signal": 0.12}
    rec2 = rec.with_outcome(outcome)
    assert rec.outcome is None              # original untouched
    assert rec2.outcome == outcome
    assert rec2.record_id == rec.record_id  # identity preserved
    assert rec2 is not rec


def test_router_record_with_outcome_rejects_non_dict():
    rec = RouterRecord(
        router_input=_sample_router_input(),
        router_output=_sample_noop(),
    )
    with pytest.raises(TypeError):
        rec.with_outcome("not a dict")  # type: ignore[arg-type]


def test_router_record_rejects_wrong_nested_types():
    with pytest.raises(TypeError):
        RouterRecord(
            router_input={"not": "a RouterInput"},  # type: ignore[arg-type]
            router_output=_sample_noop(),
        )


# ──────────────────────────────────────────────────────────────────────
# Module-level import contract
# ──────────────────────────────────────────────────────────────────────

def test_public_exports_cover_acceptance_criterion():
    """Sprint doc acceptance: ``from sage.cognition.router import
    RouterInput, RouterOutput, Event, RouterRecord, PluginTier`` works.
    """
    from sage.cognition import router
    for name in ("RouterInput", "RouterOutput", "Event", "RouterRecord",
                 "PluginTier", "ROUTER_SCHEMA_VERSION"):
        assert hasattr(router, name), f"missing public export: {name}"


def test_valid_actions_matches_prd():
    assert VALID_ACTIONS == {"invoke", "habit", "noop"}


def test_valid_rationale_codes_cover_prd_examples():
    """PRD §3.2 calls out these rationale codes by name."""
    for code in ("high_novelty", "habit_match", "low_atp_rest",
                 "metacog_blocked", "goal_driven", "reflex",
                 "escalate_frontal", "federate_peer"):
        assert code in VALID_RATIONALE_CODES


# ──────────────────────────────────────────────────────────────────────
# Sprint 2 R1 — source-stamping via metadata
# ──────────────────────────────────────────────────────────────────────

import os
from sage.cognition.router.record import (
    VALID_RECORD_SOURCES,
    SOURCE_ENV_VAR,
    ROUTER_SCHEMA_VERSION,
)


def _make_minimal_record(**kwargs):
    """Helper: build a valid RouterRecord with minimal required args."""
    ri = RouterInput(
        tick=1, timestamp=123.0, goal_id=None,
        wm_state_key="abc", wm_slot_counts={}, wm_goal_active=False,
        wm_age_ticks=0, wm_pressure=0.0,
        sensory_modalities=[], sensory_novelty=0.0, sensory_urgency=0.0,
        snarc_surprise=0.0, snarc_novelty=0.0, snarc_arousal=0.0,
        snarc_reward=0.0, snarc_conflict=0.0,
        metabolic_state="wake", atp_level=50.0, atp_trend="stable",
        recall_count=0, recall_best_similarity=0.0, recall_best_outcome=None,
        habit_available=False, habit_confidence=0.0,
        prior_invoke=0.0, prior_habit=0.0, prior_noop=0.0,
        metacog_block_list=[],
        cartridge_recall_count=0, cartridge_recall_best_similarity=0.0,
        cartridge_recall_embedding=[0.0]*768,
    )
    ro = RouterOutput(
        action="noop", plugin=None, plugin_tier=None, payload_hint=None,
        habit_id=None, confidence=0.9, energy_estimate=0.0,
        rationale_code="low_atp_rest",
    )
    return RouterRecord(router_input=ri, router_output=ro, **kwargs)


def test_schema_version_bumped_to_v020():
    assert ROUTER_SCHEMA_VERSION == "v0.2.0"
    r = _make_minimal_record()
    assert r.schema_version == "v0.2.0"


def test_metadata_defaults_to_empty_dict():
    r = _make_minimal_record()
    assert r.metadata == {}


def test_metadata_roundtrip_preserves_source():
    r = _make_minimal_record(metadata={"source": "raising"})
    d = r.to_dict()
    assert d["metadata"] == {"source": "raising"}
    r2 = RouterRecord.from_dict(d)
    assert r2.metadata == {"source": "raising"}


def test_metadata_source_validator_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        _make_minimal_record(metadata={"source": "unknown_source"})


def test_metadata_source_validator_accepts_all_valid():
    for src in VALID_RECORD_SOURCES:
        r = _make_minimal_record(metadata={"source": src})
        assert r.metadata["source"] == src


def test_valid_record_sources_matches_prd():
    # The closed vocabulary. Sprint 2 R1 baseline plus the three sage_plays
    # harness sources added in the phase-1.5 "SAGE plays" convergence work:
    #   sage_plays       — offline re-scoring of captured records
    #   sage_plays_live  — teacher-forced live play
    #   sage_plays_self  — self-advance play; errors compound
    assert VALID_RECORD_SOURCES == {
        "raising", "gameplay", "idle", "interactive",
        "sage_plays", "sage_plays_live", "sage_plays_self",
    }


def test_from_dict_backward_compat_missing_metadata():
    # Simulate a v0.1.0 record that didn't have metadata.
    r = _make_minimal_record(metadata={"source": "idle"})
    d = r.to_dict()
    del d["metadata"]  # old-format record
    d["schema_version"] = "v0.1.0"
    r2 = RouterRecord.from_dict(d)
    assert r2.metadata == {}


def test_with_outcome_preserves_metadata():
    r = _make_minimal_record(metadata={"source": "gameplay"})
    r2 = r.with_outcome({"ok": True})
    assert r2.metadata == {"source": "gameplay"}
    # Non-mutating
    assert r.outcome is None
    assert r2.outcome == {"ok": True}


def test_shadow_build_metadata_from_env(monkeypatch):
    from sage.cognition.router.shadow import _build_record_metadata
    # Absent env → idle
    monkeypatch.delenv(SOURCE_ENV_VAR, raising=False)
    assert _build_record_metadata() == {"source": "idle"}
    # Valid value honored
    monkeypatch.setenv(SOURCE_ENV_VAR, "raising")
    assert _build_record_metadata() == {"source": "raising"}
    # Invalid value coerced to idle (rather than failing record construction)
    monkeypatch.setenv(SOURCE_ENV_VAR, "bogus")
    assert _build_record_metadata() == {"source": "idle"}
