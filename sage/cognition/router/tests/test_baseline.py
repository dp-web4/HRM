#!/usr/bin/env python3
"""
Unit tests for the programmatic baseline (Phase 0 Track 3).

Covers:
  - Every branch of ``_decide_action_type`` exercised (dream, habit,
    invoke, default-noop)
  - ``_compute_rationale`` yields a code in ``VALID_RATIONALE_CODES`` for
    every branch
  - Output always passes ``RouterOutput.validate()``
  - ``plugin_tier`` matches the registry assignment
  - Determinism: same (input, registry) → same output
  - Purity: ``router_input`` is not mutated
  - Performance: p99 < 1ms per decision
  - Edge cases: empty WM, zero salience, all components stubbed, ATP=0,
    metacog blocks everything, unknown modality, registry missing tier
  - Agreement with the *intent* of the existing dispatcher on a corpus
    of synthetic inputs (every branch hit ≥ once)

Run: ``python3 -m pytest sage/cognition/router/tests/test_baseline.py -v``
"""

import copy
import time
from typing import Any, Dict, List

import pytest

from sage.cognition.router import (
    RouterInput,
    RouterOutput,
    VALID_RATIONALE_CODES,
    PluginTier,
)
from sage.cognition.router.baseline import (
    HABIT_CONFIDENCE_THRESHOLD,
    MODALITY_MAP,
    NOOP_METABOLIC_STATES,
    SNARC_HIGH_NOVELTY_THRESHOLD,
    _compute_rationale,
    _decide_action_type,
    _select_plugin,
    _should_use_habit,
    programmatic_decide,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

def _registry() -> Dict[str, Dict[str, Any]]:
    """A plugin registry matching the live dispatcher's plugin set.

    Tiers chosen to align with PRD §1.2 and the existing orchestrator
    plugin types. ATP costs are illustrative — they drive
    ``energy_estimate`` but aren't sprint-critical.
    """
    return {
        "vision": {"tier": PluginTier.ROUTINE.value, "atp_cost": 5.0},
        "audio": {"tier": PluginTier.ROUTINE.value, "atp_cost": 4.0},
        "language": {"tier": PluginTier.FRONTAL_LOBE.value, "atp_cost": 80.0},
        "control": {"tier": PluginTier.REFLEX.value, "atp_cost": 0.5},
        "peer_sage": {"tier": PluginTier.FEDERATE.value, "atp_cost": 200.0},
        "specialized_solver": {
            "tier": PluginTier.SPECIALIZED.value,
            "atp_cost": 25.0,
        },
    }


def _router_input(**overrides) -> RouterInput:
    """Synthesize a valid RouterInput with sensible defaults.

    Defaults represent a low-salience wake tick with a vision observation
    and a goal active. Individual tests override to exercise branches.
    """
    defaults = dict(
        tick=1,
        timestamp=1_700_000_000.0,
        goal_id="goal-1",
        wm_state_key="0123456789abcdef",
        wm_slot_counts={"goal": 1},
        wm_goal_active=True,
        wm_age_ticks=0,
        wm_pressure=0.25,
        sensory_modalities=["vision"],
        sensory_novelty=0.2,
        sensory_urgency=0.2,
        snarc_surprise=0.1,
        snarc_novelty=0.2,
        snarc_arousal=0.2,
        snarc_reward=0.0,
        snarc_conflict=0.1,
        metabolic_state="wake",
        atp_level=80.0,
        atp_trend="stable",
        recall_count=0,
        recall_best_similarity=0.0,
        recall_best_outcome=None,
        habit_available=False,
        habit_confidence=0.0,
        prior_invoke=0.5,
        prior_habit=0.25,
        prior_noop=0.25,
        metacog_block_list=[],
    )
    defaults.update(overrides)
    return RouterInput(**defaults)


# ──────────────────────────────────────────────────────────────────────
# Branch coverage — top-level action type
# ──────────────────────────────────────────────────────────────────────

def test_dream_state_forces_noop():
    ri = _router_input(metabolic_state="dream")
    out = programmatic_decide(ri, _registry())
    assert out.action == "noop"
    assert out.rationale_code == "low_atp_rest"


def test_wake_with_vision_invokes_vision_plugin():
    ri = _router_input()
    out = programmatic_decide(ri, _registry())
    assert out.action == "invoke"
    assert out.plugin == "vision"
    assert out.plugin_tier == PluginTier.ROUTINE.value


def test_audio_modality_invokes_audio_first():
    # MODALITY_MAP['audio'] = ['audio', 'language'] — first wins.
    ri = _router_input(sensory_modalities=["audio"])
    out = programmatic_decide(ri, _registry())
    assert out.action == "invoke"
    assert out.plugin == "audio"
    assert out.plugin_tier == PluginTier.ROUTINE.value


def test_time_modality_emits_noop_default():
    # MODALITY_MAP['time'] = [] — no plugin, but no block either.
    ri = _router_input(sensory_modalities=["time"])
    out = programmatic_decide(ri, _registry())
    assert out.action == "noop"
    assert out.rationale_code == "default"


def test_empty_sensory_modalities_emits_noop_default():
    ri = _router_input(sensory_modalities=[])
    out = programmatic_decide(ri, _registry())
    assert out.action == "noop"
    assert out.rationale_code == "default"


def test_unknown_modality_emits_noop_default():
    ri = _router_input(sensory_modalities=["spooky_sensor"])
    out = programmatic_decide(ri, _registry())
    assert out.action == "noop"
    assert out.rationale_code == "default"


# ──────────────────────────────────────────────────────────────────────
# Habit branch
# ──────────────────────────────────────────────────────────────────────

def test_habit_available_high_confidence_chooses_habit():
    ri = _router_input(
        habit_available=True, habit_confidence=0.95
    )
    out = programmatic_decide(ri, _registry())
    assert out.action == "habit"
    assert out.habit_id == ri.wm_state_key
    assert out.rationale_code == "habit_match"
    assert out.plugin is None and out.plugin_tier is None


def test_habit_available_low_confidence_falls_through_to_invoke():
    ri = _router_input(
        habit_available=True, habit_confidence=HABIT_CONFIDENCE_THRESHOLD - 0.01
    )
    out = programmatic_decide(ri, _registry())
    # Low habit confidence → existing logic falls through to plugin invoke.
    assert out.action == "invoke"


def test_habit_unavailable_even_if_confidence_high():
    ri = _router_input(habit_available=False, habit_confidence=1.0)
    assert not _should_use_habit(ri)


def test_habit_at_threshold_fires():
    ri = _router_input(
        habit_available=True, habit_confidence=HABIT_CONFIDENCE_THRESHOLD
    )
    assert _should_use_habit(ri)


# ──────────────────────────────────────────────────────────────────────
# Metacog blocking
# ──────────────────────────────────────────────────────────────────────

def test_metacog_blocks_only_candidate_yields_metacog_blocked():
    ri = _router_input(
        sensory_modalities=["vision"],
        metacog_block_list=["vision"],
    )
    out = programmatic_decide(ri, _registry())
    assert out.action == "noop"
    assert out.rationale_code == "metacog_blocked"


def test_metacog_blocks_one_of_many_still_invokes():
    ri = _router_input(
        sensory_modalities=["audio"],         # → ['audio', 'language']
        metacog_block_list=["audio"],         # blocks first, 'language' remains
    )
    out = programmatic_decide(ri, _registry())
    assert out.action == "invoke"
    assert out.plugin == "language"


def test_metacog_blocks_everything_noop():
    ri = _router_input(
        sensory_modalities=["audio", "vision"],
        metacog_block_list=["audio", "language", "vision"],
    )
    out = programmatic_decide(ri, _registry())
    assert out.action == "noop"
    assert out.rationale_code == "metacog_blocked"


# ──────────────────────────────────────────────────────────────────────
# Rationale codes — full vocabulary coverage
# ──────────────────────────────────────────────────────────────────────

def test_frontal_lobe_tier_emits_escalate_frontal():
    # Message modality → language plugin → frontal_lobe tier.
    ri = _router_input(sensory_modalities=["message"])
    out = programmatic_decide(ri, _registry())
    assert out.action == "invoke"
    assert out.plugin == "language"
    assert out.plugin_tier == PluginTier.FRONTAL_LOBE.value
    assert out.rationale_code == "escalate_frontal"


def test_federate_tier_emits_federate_peer():
    registry = _registry()
    # Add federate plugin to a custom modality map by routing through a
    # direct-named modality. We don't overload the modality map so we
    # exercise the path via a plugin with federate tier that we route to.
    # Simpler: inject 'peer_sage' into a custom modality.
    registry["peer_sage"] = {
        "tier": PluginTier.FEDERATE.value,
        "atp_cost": 200.0,
    }
    # Monkeypatch MODALITY_MAP locally — we test the rationale path by
    # using _compute_rationale directly, which is one of the public
    # helpers listed in the Track 3 spec.
    ri = _router_input()
    rat = _compute_rationale(ri, "invoke", "peer_sage", registry)
    assert rat == "federate_peer"


def test_reflex_tier_emits_reflex():
    # Proprioception → control plugin (reflex tier in our fixture).
    ri = _router_input(sensory_modalities=["proprioception"])
    out = programmatic_decide(ri, _registry())
    assert out.action == "invoke"
    assert out.plugin == "control"
    assert out.plugin_tier == PluginTier.REFLEX.value
    assert out.rationale_code == "reflex"


def test_high_novelty_arousal_emits_high_novelty():
    # Specialized tier (not frontal/federate/reflex) + high SNARC → high_novelty.
    registry = _registry()
    # Attach a specialized-tier plugin to the vision modality so we
    # bypass the routine vision default.
    registry["vision"] = {
        "tier": PluginTier.SPECIALIZED.value,
        "atp_cost": 25.0,
    }
    ri = _router_input(
        snarc_novelty=SNARC_HIGH_NOVELTY_THRESHOLD + 0.1,
        snarc_arousal=SNARC_HIGH_NOVELTY_THRESHOLD + 0.1,
        wm_goal_active=False,  # exclude goal_driven precedence
    )
    out = programmatic_decide(ri, registry)
    assert out.action == "invoke"
    assert out.rationale_code == "high_novelty"


def test_goal_driven_path_emits_goal_driven():
    # Routine-tier invoke with goal active + perception modality → goal_driven.
    ri = _router_input(
        wm_goal_active=True,
        snarc_novelty=0.1,  # don't trigger high_novelty
        snarc_arousal=0.1,
    )
    out = programmatic_decide(ri, _registry())
    # vision plugin has routine tier in our fixture; rationale should be goal_driven.
    assert out.action == "invoke"
    assert out.plugin == "vision"
    assert out.rationale_code == "goal_driven"


def test_default_invoke_rationale_when_no_special_branch():
    # Routine-tier invoke, no goal, no high SNARC → default.
    ri = _router_input(
        wm_goal_active=False,
        snarc_novelty=0.1,
        snarc_arousal=0.1,
    )
    out = programmatic_decide(ri, _registry())
    assert out.action == "invoke"
    assert out.rationale_code == "default"


def test_default_noop_rationale_when_no_candidate():
    ri = _router_input(sensory_modalities=[])
    out = programmatic_decide(ri, _registry())
    assert out.action == "noop"
    assert out.rationale_code == "default"


# ──────────────────────────────────────────────────────────────────────
# Output validity — every code path
# ──────────────────────────────────────────────────────────────────────

def test_every_branch_produces_valid_output():
    """Exercise a corpus and assert every decision validates."""
    registry = _registry()
    known_plugins = set(registry.keys())
    corpus = [
        _router_input(),
        _router_input(metabolic_state="dream"),
        _router_input(metabolic_state="rest"),
        _router_input(metabolic_state="focus"),
        _router_input(metabolic_state="crisis"),
        _router_input(sensory_modalities=[]),
        _router_input(sensory_modalities=["time"]),
        _router_input(sensory_modalities=["message"]),
        _router_input(sensory_modalities=["proprioception"]),
        _router_input(habit_available=True, habit_confidence=1.0),
        _router_input(habit_available=True, habit_confidence=0.5),
        _router_input(metacog_block_list=["vision", "audio", "language", "control"]),
        _router_input(
            snarc_novelty=1.0, snarc_arousal=1.0, wm_goal_active=False
        ),
        _router_input(
            snarc_novelty=0.0, snarc_arousal=0.0, atp_level=0.1, atp_trend="falling"
        ),
        _router_input(wm_slot_counts={}, wm_goal_active=False, wm_pressure=0.0),
    ]
    for ri in corpus:
        out = programmatic_decide(ri, registry)
        ok, reason = out.validate(known_plugins=known_plugins)
        assert ok, f"invalid output: {reason} for {out!r}"
        assert out.rationale_code in VALID_RATIONALE_CODES


def test_rationale_code_always_in_valid_set_over_corpus():
    registry = _registry()
    # Broader sweep across SNARC + habit + metacog crosses.
    for novelty in (0.0, 0.3, 0.6, 0.9):
        for arousal in (0.0, 0.3, 0.6, 0.9):
            for habit_available in (False, True):
                for habit_conf in (0.0, 0.5, 0.9):
                    for metabolic in ("wake", "focus", "rest", "crisis", "dream"):
                        ri = _router_input(
                            snarc_novelty=novelty,
                            snarc_arousal=arousal,
                            habit_available=habit_available,
                            habit_confidence=habit_conf,
                            metabolic_state=metabolic,
                        )
                        out = programmatic_decide(ri, registry)
                        assert out.rationale_code in VALID_RATIONALE_CODES


def test_plugin_tier_matches_registry():
    registry = _registry()
    for modality, expected_plugins in MODALITY_MAP.items():
        if not expected_plugins:
            continue
        first = expected_plugins[0]
        if first not in registry:
            continue
        ri = _router_input(sensory_modalities=[modality])
        out = programmatic_decide(ri, registry)
        if out.action == "invoke":
            expected_tier = registry[out.plugin]["tier"]
            assert out.plugin_tier == expected_tier


# ──────────────────────────────────────────────────────────────────────
# Purity + determinism
# ──────────────────────────────────────────────────────────────────────

def test_deterministic_same_input_same_output():
    ri = _router_input()
    registry = _registry()
    out1 = programmatic_decide(ri, registry)
    out2 = programmatic_decide(ri, registry)
    assert out1 == out2


def test_pure_no_mutation_of_router_input():
    ri = _router_input(
        sensory_modalities=["vision", "audio"],
        metacog_block_list=["language"],
    )
    before = copy.deepcopy(ri)
    programmatic_decide(ri, _registry())
    # Verify core fields unchanged — RouterInput doesn't implement __eq__
    # in a deep sense, so compare attribute-by-attribute.
    for field in (
        "tick", "timestamp", "goal_id", "wm_state_key", "wm_slot_counts",
        "wm_goal_active", "sensory_modalities", "metacog_block_list",
        "snarc_novelty", "snarc_arousal", "metabolic_state", "atp_level",
    ):
        assert getattr(ri, field) == getattr(before, field), \
            f"router_input.{field} mutated"


def test_pure_no_mutation_of_plugin_registry():
    ri = _router_input()
    registry = _registry()
    before = copy.deepcopy(registry)
    programmatic_decide(ri, registry)
    assert registry == before


# ──────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────

def test_empty_wm_slot_counts():
    ri = _router_input(
        wm_slot_counts={}, wm_goal_active=False, wm_pressure=0.0,
    )
    out = programmatic_decide(ri, _registry())
    # Still picks vision since sensory_modalities=['vision'] by default.
    assert out.action in {"invoke", "noop"}


def test_zero_salience_still_produces_valid_output():
    ri = _router_input(
        snarc_surprise=0.0, snarc_novelty=0.0, snarc_arousal=0.0,
        snarc_reward=0.0, snarc_conflict=0.0,
        sensory_novelty=0.0, sensory_urgency=0.0,
    )
    out = programmatic_decide(ri, _registry())
    assert out.is_valid(known_plugins=set(_registry().keys()))
    assert 0.0 <= out.confidence <= 1.0


def test_atp_level_zero_does_not_crash():
    ri = _router_input(atp_level=0.0, atp_trend="falling")
    out = programmatic_decide(ri, _registry())
    # Note: ATP throttling is step 6 (Budget), not the router. Router
    # only emits energy_estimate; Budget may downgrade to noop.
    assert out.is_valid(known_plugins=set(_registry().keys()))


def test_registry_missing_tier_coerces_to_noop():
    registry = {"vision": {}}  # present but no tier → coerce
    ri = _router_input(sensory_modalities=["vision"])
    out = programmatic_decide(ri, registry)
    # The output must still validate; tier lookup returns None → validate
    # accepts None tier on an invoke IF the plugin is in known_plugins...
    # but we coerce explicitly since we can't trust the output.
    ok, _ = out.validate(known_plugins={"vision"})
    assert ok


def test_empty_plugin_registry_emits_noop():
    ri = _router_input()
    out = programmatic_decide(ri, {})
    assert out.action == "noop"
    assert out.rationale_code == "default"


def test_all_components_stubbed_still_valid():
    """Minimal RouterInput with everything at defaults — no habit, no
    recall, no prior data — baseline still produces a valid decision."""
    ri = _router_input(
        recall_count=0,
        recall_best_similarity=0.0,
        recall_best_outcome=None,
        habit_available=False,
        habit_confidence=0.0,
        prior_invoke=0.0,
        prior_habit=0.0,
        prior_noop=0.0,
        metacog_block_list=[],
    )
    out = programmatic_decide(ri, _registry())
    assert out.is_valid(known_plugins=set(_registry().keys()))


def test_habit_branch_does_not_require_plugin_registry():
    ri = _router_input(habit_available=True, habit_confidence=1.0)
    out = programmatic_decide(ri, {})
    assert out.action == "habit"
    assert out.habit_id == ri.wm_state_key


# ──────────────────────────────────────────────────────────────────────
# Performance — <1ms p99 budget
# ──────────────────────────────────────────────────────────────────────

def test_latency_under_1ms_p99():
    registry = _registry()
    # Varied corpus so we're not measuring CPU-cached single-path.
    inputs = [
        _router_input(),
        _router_input(metabolic_state="dream"),
        _router_input(habit_available=True, habit_confidence=1.0),
        _router_input(sensory_modalities=["audio", "message"]),
        _router_input(metacog_block_list=["vision"]),
        _router_input(snarc_novelty=0.9, snarc_arousal=0.9),
        _router_input(sensory_modalities=["time"]),
    ]

    n_iters = 5000
    durations_ns: List[int] = []
    # Warm-up: avoid measuring first-call import/cache effects.
    for ri in inputs:
        programmatic_decide(ri, registry)

    for i in range(n_iters):
        ri = inputs[i % len(inputs)]
        t0 = time.perf_counter_ns()
        programmatic_decide(ri, registry)
        durations_ns.append(time.perf_counter_ns() - t0)

    durations_ns.sort()
    p50 = durations_ns[len(durations_ns) // 2]
    p99 = durations_ns[int(len(durations_ns) * 0.99)]
    p50_us = p50 / 1000.0
    p99_us = p99 / 1000.0
    # <1ms p99 budget per Track 3 acceptance criteria.
    assert p99 < 1_000_000, (
        f"p99 latency {p99_us:.1f}µs exceeds 1ms budget "
        f"(p50={p50_us:.1f}µs)"
    )


# ──────────────────────────────────────────────────────────────────────
# Reproduces existing dispatcher behavior (behavioral parity)
# ──────────────────────────────────────────────────────────────────────

def test_dispatcher_parity_vision_modality_picks_vision_plugin():
    """Mirror of _get_plugins_for_modality('vision') → ['vision']."""
    ri = _router_input(sensory_modalities=["vision"])
    out = programmatic_decide(ri, _registry())
    assert out.action == "invoke" and out.plugin == "vision"


def test_dispatcher_parity_proprioception_picks_control():
    ri = _router_input(sensory_modalities=["proprioception"])
    out = programmatic_decide(ri, _registry())
    assert out.action == "invoke" and out.plugin == "control"


def test_dispatcher_parity_time_modality_no_plugin():
    ri = _router_input(sensory_modalities=["time"])
    out = programmatic_decide(ri, _registry())
    assert out.action == "noop"


def test_dispatcher_parity_message_routes_to_language():
    ri = _router_input(sensory_modalities=["message"])
    out = programmatic_decide(ri, _registry())
    assert out.action == "invoke" and out.plugin == "language"


def test_dispatcher_parity_audio_first_plugin_is_audio():
    """MODALITY_MAP['audio'] = ['audio', 'language'] per dispatcher."""
    ri = _router_input(sensory_modalities=["audio"])
    out = programmatic_decide(ri, _registry())
    assert out.action == "invoke" and out.plugin == "audio"


def test_dispatcher_parity_modality_order_preserved():
    """Top-of-list modality takes precedence (dispatcher sorts observations
    by salience; feature extractor in Track 2 surfaces them in that
    order)."""
    ri = _router_input(sensory_modalities=["proprioception", "vision"])
    out = programmatic_decide(ri, _registry())
    # First modality's plugin wins.
    assert out.action == "invoke" and out.plugin == "control"


# ──────────────────────────────────────────────────────────────────────
# Confidence + energy sanity
# ──────────────────────────────────────────────────────────────────────

def test_confidence_in_range():
    for ri in (
        _router_input(),
        _router_input(snarc_arousal=1.0, snarc_conflict=1.0),
        _router_input(snarc_arousal=0.0, snarc_conflict=0.0),
    ):
        out = programmatic_decide(ri, _registry())
        assert 0.0 <= out.confidence <= 1.0


def test_energy_estimate_matches_registry_cost_on_invoke():
    ri = _router_input(sensory_modalities=["vision"])
    out = programmatic_decide(ri, _registry())
    assert out.action == "invoke"
    assert out.energy_estimate == pytest.approx(5.0)  # vision atp_cost=5.0


def test_noop_energy_is_zero():
    ri = _router_input(metabolic_state="dream")
    out = programmatic_decide(ri, _registry())
    assert out.action == "noop"
    assert out.energy_estimate == 0.0


def test_habit_energy_is_zero():
    ri = _router_input(habit_available=True, habit_confidence=1.0)
    out = programmatic_decide(ri, _registry())
    assert out.action == "habit"
    assert out.energy_estimate == 0.0


# ──────────────────────────────────────────────────────────────────────
# Helper function contracts (spec-required exports)
# ──────────────────────────────────────────────────────────────────────

def test_helper_decide_action_type_returns_one_of_three():
    ri = _router_input()
    assert _decide_action_type(ri, _registry()) in {"invoke", "habit", "noop"}


def test_helper_select_plugin_returns_string_or_none():
    ri = _router_input()
    plugin = _select_plugin(ri, _registry())
    assert plugin is None or isinstance(plugin, str)


def test_helper_select_plugin_none_when_nothing_candidate():
    ri = _router_input(sensory_modalities=[])
    assert _select_plugin(ri, _registry()) is None


def test_helper_should_use_habit_bool():
    ri = _router_input()
    assert isinstance(_should_use_habit(ri), bool)


def test_helper_compute_rationale_returns_valid_code():
    ri = _router_input()
    for action, plugin in (
        ("noop", None),
        ("habit", None),
        ("invoke", "vision"),
    ):
        rat = _compute_rationale(ri, action, plugin, _registry())
        assert rat in VALID_RATIONALE_CODES


# ──────────────────────────────────────────────────────────────────────
# NOOP_METABOLIC_STATES coverage — dream hits the short-circuit
# ──────────────────────────────────────────────────────────────────────

def test_noop_metabolic_states_set_matches_dispatcher():
    """Existing dispatcher has max_active_plugins=0 only for DREAM state.
    This constant must match, or the baseline will diverge from the
    dispatcher under rest/crisis (both of which allow 1 plugin)."""
    # Explicit assertion: dream is in, rest/crisis are NOT in.
    assert "dream" in NOOP_METABOLIC_STATES
    assert "rest" not in NOOP_METABOLIC_STATES
    assert "crisis" not in NOOP_METABOLIC_STATES
    assert "wake" not in NOOP_METABOLIC_STATES
    assert "focus" not in NOOP_METABOLIC_STATES
