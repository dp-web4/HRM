"""Tests for phase1_training — agent-zero gates must fire on degenerate data."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Make parent package importable when tests run standalone.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sage.cognition.thalamic_router.phase1_training import (
    ACTION_CLASSES,
    MODAL_DUMMY_MARGIN_THRESHOLD_PP,
    SNARC_UTILITY_DELTA_THRESHOLD,
    MIN_SNARC_STD_PER_DIM,
    MIN_ENTROPY_NATS,
    HEAD_B_ACTION_CLASSES,
    HEAD_B_ACTION_NAMES,
    HEAD_B_MARGIN_THRESHOLD_PP,
    build_xy,
    build_xy_head_b,
    evaluate_phase1,
    evaluate_phase1_head_b,
    compute_diversity,
    rare_decision_recall,
    salience_weighted_accuracy,
)


# ───────────────────────────────────────────────────────────────────
# Record fixtures
# ───────────────────────────────────────────────────────────────────

def _make_record(
    action: str = "noop",
    snarc_arousal: float = 0.1,
    snarc_surprise: float = 0.1,
    snarc_reward: float = 0.0,
    snarc_conflict: float = 0.0,
    snarc_novelty: Optional[float] = None,
    source: str = "idle",
    has_audio: bool = False,
) -> dict:
    if snarc_novelty is None:
        snarc_novelty = snarc_arousal * 0.5
    return {
        "record_id": f"r{np.random.randint(1e9):09d}",
        "schema_version": "v0.2.0",
        "timestamp": 0.0,
        "machine": "test",
        "router_input": {
            "snarc_surprise": snarc_surprise,
            "snarc_novelty": snarc_novelty,
            "snarc_arousal": snarc_arousal,
            "snarc_reward": snarc_reward,
            "snarc_conflict": snarc_conflict,
            "sensory_novelty": 0.1,
            "sensory_urgency": 0.0,
            "atp_level": 50,
            "wm_goal_active": False,
            "wm_pressure": 0.0,
            "habit_available": False,
            "habit_confidence": 0.0,
            "sensory_modalities": (["audio"] if has_audio else []),
            "metabolic_state": "wake",
        },
        "router_output": {
            "action": action,
            "plugin": None,
            "plugin_tier": None,
            "habit_id": None,
            "confidence": 0.9,
            "energy_estimate": 0.0,
            "rationale_code": "low_atp_rest",
        },
        "outcome": None,
        "metadata": {"source": source},
    }


def _homogeneous_dataset(n: int = 200) -> list:
    """Degenerate: mostly noop, audio→invoke; SNARC constant. Should be INCONCLUSIVE."""
    rng = np.random.default_rng(0)
    records = []
    for _ in range(n):
        if rng.random() < 0.1:
            records.append(_make_record(action="invoke", has_audio=True,
                                        snarc_arousal=0.5, snarc_surprise=0.5))
        else:
            records.append(_make_record(action="noop", has_audio=False,
                                        snarc_arousal=0.1, snarc_surprise=0.1))
    return records


def _diverse_dataset(n: int = 400) -> list:
    """Varied: mixed classes, broad SNARC range on every dim, diverse sources."""
    rng = np.random.default_rng(1)
    records = []
    actions = ["noop", "invoke"]
    sources = ["raising", "gameplay", "idle", "interactive"]
    for i in range(n):
        action = rng.choice(actions, p=[0.55, 0.45])
        records.append(_make_record(
            action=action,
            snarc_arousal=float(rng.random()),
            snarc_surprise=float(rng.random()),
            snarc_novelty=float(rng.random()),
            snarc_reward=float(rng.uniform(-1, 1)),
            snarc_conflict=float(rng.random()),
            has_audio=bool(rng.random() > 0.5),
            source=rng.choice(sources),
        ))
    return records


# ───────────────────────────────────────────────────────────────────
# Gate tests — these are the reason this pipeline exists
# ───────────────────────────────────────────────────────────────────

def test_degenerate_data_produces_inconclusive():
    """100% accuracy on homogeneous data → INCONCLUSIVE (agent-zero defense)."""
    m = evaluate_phase1(_homogeneous_dataset(500), seed=0)
    assert m.verdict == "INCONCLUSIVE", f"expected INCONCLUSIVE, got {m.verdict}"
    # At least one of the three failure modes should fire:
    # - insufficient margin, OR collinear SNARC, OR no SNARC utility
    reasons_blob = " ".join(m.verdict_reasons)
    assert ("margin" in reasons_blob.lower()
            or "snarc" in reasons_blob.lower()
            or "collinear" in reasons_blob.lower()
            or "homogen" in reasons_blob.lower()), (
        f"no agent-zero reason found in: {m.verdict_reasons}"
    )


def test_margin_gate_fires_when_modal_dominates():
    """If modal-class dummy is already >= result-margin threshold, INCONCLUSIVE."""
    m = evaluate_phase1(_homogeneous_dataset(500), seed=0)
    assert m.margin_over_dummy < MODAL_DUMMY_MARGIN_THRESHOLD_PP


def test_snarc_ablation_gate_fires_on_trivial_routing():
    """If router-without-SNARC performs equivalently, delta is ~0 → gate fires."""
    m = evaluate_phase1(_homogeneous_dataset(500), seed=0)
    assert m.snarc_utility_delta is not None
    # Degenerate data makes SNARC irrelevant — delta should be ≈0
    assert m.snarc_utility_delta < SNARC_UTILITY_DELTA_THRESHOLD


def test_collinearity_gate_fires_when_snarc_is_constant():
    """If SNARC per-dim stddev is below threshold, gate fires."""
    m = evaluate_phase1(_homogeneous_dataset(500), seed=0)
    assert m.min_snarc_std < MIN_SNARC_STD_PER_DIM


# ───────────────────────────────────────────────────────────────────
# Positive-path tests — diverse data should not trigger INCONCLUSIVE
# spuriously (though LR may still FAIL to hit 98%, which is fine)
# ───────────────────────────────────────────────────────────────────

def test_diverse_data_passes_collinearity_check():
    """Varied SNARC values → per-dim stddev above threshold."""
    m = evaluate_phase1(_diverse_dataset(800), seed=1)
    assert m.min_snarc_std >= MIN_SNARC_STD_PER_DIM, (
        f"diverse data should not trigger collinearity gate: min_std={m.min_snarc_std}"
    )


def test_diverse_data_has_enough_entropy():
    """Balanced actions → entropy above threshold."""
    m = evaluate_phase1(_diverse_dataset(800), seed=1)
    assert m.decision_class_entropy >= MIN_ENTROPY_NATS


# ───────────────────────────────────────────────────────────────────
# Helper tests
# ───────────────────────────────────────────────────────────────────

def test_rare_decision_recall_on_obvious_case():
    """When rare class is perfectly recovered, recall = 1.0."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1])
    modal, rare, n = rare_decision_recall(y_true, y_pred)
    assert modal == "noop"  # class 0
    assert rare == 1.0
    assert n == 2


def test_salience_weighted_acc_uses_top_decile():
    """Top-decile-arousal subset drives the metric."""
    records = [
        _make_record(snarc_arousal=0.95),  # in top decile (threshold will be high)
        _make_record(snarc_arousal=0.1),
        _make_record(snarc_arousal=0.1),
        _make_record(snarc_arousal=0.1),
        _make_record(snarc_arousal=0.1),
        _make_record(snarc_arousal=0.1),
        _make_record(snarc_arousal=0.1),
        _make_record(snarc_arousal=0.1),
        _make_record(snarc_arousal=0.1),
        _make_record(snarc_arousal=0.1),
    ]
    y_true = np.array([0] * 10)
    y_pred = np.array([0] * 10)
    acc, n = salience_weighted_accuracy(records, y_true, y_pred)
    assert acc == 1.0
    assert n >= 1  # at least the top-arousal record included


def test_diversity_counts_sources_correctly():
    records = [
        _make_record(source="raising"),
        _make_record(source="raising"),
        _make_record(source="gameplay"),
        _make_record(source="idle"),
    ]
    d = compute_diversity(records)
    assert d["n_sources"] == 3
    assert d["sources"] == {"raising": 2, "gameplay": 1, "idle": 1}


def _head_b_record(action: int, arousal: float, source: str = "gameplay") -> dict:
    """Gameplay record with known_good_action in metadata."""
    return {
        "record_id": f"r{np.random.randint(1e9):09d}",
        "schema_version": "v0.2.0",
        "timestamp": 0.0,
        "machine": "test",
        "router_input": {
            "snarc_surprise": arousal * 0.5,
            "snarc_novelty": arousal * 0.3,
            "snarc_arousal": arousal,
            "snarc_reward": 0.0,
            "snarc_conflict": 0.0,
            "sensory_novelty": 0.1, "sensory_urgency": 0.1,
            "atp_level": 70, "wm_goal_active": True, "wm_pressure": 0.3,
            "habit_available": False, "habit_confidence": 0.0,
            "sensory_modalities": ["vision"],
            "metabolic_state": "focus",
        },
        "router_output": {"action": "invoke"},
        "metadata": {
            "source": source,
            "known_good_action": action,
            "game": "testgame",
            "known_good_level": 0,
        },
    }


def test_build_xy_head_b_drops_records_without_known_good():
    recs = [
        _head_b_record(action=1, arousal=0.1),
        {"router_input": {}, "router_output": {"action": "noop"}, "metadata": {"source": "idle"}},
        _head_b_record(action=6, arousal=0.5),
    ]
    X, y, kept = build_xy_head_b(recs, snarc=True)
    assert len(y) == 2
    assert list(y) == [1, 6]
    assert kept == [0, 2]


def test_build_xy_head_b_drops_out_of_range():
    recs = [
        _head_b_record(action=3, arousal=0.1),
        _head_b_record(action=99, arousal=0.1),  # out of range
    ]
    _, y, _ = build_xy_head_b(recs, snarc=True)
    assert list(y) == [3]


def test_head_b_recovers_signal_when_action_correlates_with_arousal():
    """If arousal perfectly predicts action class, LR should find it."""
    np.random.seed(0)
    recs = []
    # Low arousal → action 1 (UP). High arousal → action 6 (CLICK).
    for _ in range(75):
        recs.append(_head_b_record(action=1, arousal=float(np.random.uniform(0.0, 0.2))))
        recs.append(_head_b_record(action=6, arousal=float(np.random.uniform(0.7, 1.0))))
    m = evaluate_phase1_head_b(recs, seed=1)
    # Clean signal → model should substantially beat modal-dummy
    assert m.margin_over_dummy >= HEAD_B_MARGIN_THRESHOLD_PP, (
        f"expected margin ≥ {HEAD_B_MARGIN_THRESHOLD_PP}, got {m.margin_over_dummy:.4f}"
    )
    assert m.snarc_utility_delta is not None
    # SNARC ablation should hurt accuracy when arousal IS the signal
    assert m.snarc_utility_delta > 0


def test_head_b_flat_features_yield_inconclusive_or_fail():
    """When features don't vary, Head B can't beat the dummy."""
    np.random.seed(1)
    recs = [_head_b_record(action=i % 4, arousal=0.0) for i in range(200)]
    m = evaluate_phase1_head_b(recs, seed=1)
    assert m.verdict in ("INCONCLUSIVE", "FAIL"), f"got {m.verdict}"


def test_build_xy_ablation_zeros_snarc_only():
    """snarc=False zeros SNARC features but preserves non-SNARC."""
    recs = [_make_record(snarc_arousal=0.5, has_audio=True) for _ in range(5)]
    X_with, _ = build_xy(recs, snarc=True)
    X_without, _ = build_xy(recs, snarc=False)
    # SNARC columns are the first 5 (see ALL_FEATURE_NAMES ordering).
    assert not np.allclose(X_with[:, :5], 0.0)
    assert np.allclose(X_without[:, :5], 0.0)
    # Non-SNARC cols unchanged
    assert np.allclose(X_with[:, 5:], X_without[:, 5:])


if __name__ == "__main__":
    import sys
    failures = 0
    for name in list(globals()):
        if name.startswith("test_"):
            try:
                globals()[name]()
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                failures += 1
    if failures:
        print(f"\n{failures} failures")
        sys.exit(1)
    print("\nAll tests passed")
