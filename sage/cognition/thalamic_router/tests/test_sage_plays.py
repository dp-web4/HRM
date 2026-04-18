"""Smoke tests for sage_plays delta extraction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sage.cognition.thalamic_router.phase1_training import (
    HeadBAdapter, ALL_FEATURE_NAMES, HEAD_B_ACTION_CLASSES, HEAD_B_ACTION_NAMES,
)
from sage.cognition.thalamic_router.sage_plays import _compute_delta


def _trivial_adapter(preferred_action: int = 6) -> HeadBAdapter:
    """Build an adapter that strongly predicts one action regardless of input."""
    n_f = len(ALL_FEATURE_NAMES)
    W = np.zeros((n_f, HEAD_B_ACTION_CLASSES))
    b = np.zeros(HEAD_B_ACTION_CLASSES)
    b[preferred_action] = 10.0  # dominant bias
    return HeadBAdapter(
        head="B", machine="test", trained_at="0", train_commit=None,
        n_train_records=0, feature_names=list(ALL_FEATURE_NAMES),
        feature_mean=[0.0] * n_f, feature_std=[1.0] * n_f,
        weights=W.tolist(), bias=b.tolist(),
        n_classes=HEAD_B_ACTION_CLASSES,
        class_names=list(HEAD_B_ACTION_NAMES),
    )


def test_compute_delta_correct_case():
    ad = _trivial_adapter(preferred_action=6)
    d = _compute_delta([0.0] * len(ALL_FEATURE_NAMES), known_good=6, adapter=ad)
    assert d["proposed_action"] == 6
    assert d["correct"] is True
    assert d["predicted_rank_of_known_good"] == 0
    assert d["confidence_on_proposed"] == d["confidence_on_known_good"]
    assert d["entropy"] >= 0


def test_compute_delta_wrong_case_has_positive_rank():
    ad = _trivial_adapter(preferred_action=6)
    d = _compute_delta([0.0] * len(ALL_FEATURE_NAMES), known_good=1, adapter=ad)
    assert d["correct"] is False
    assert d["proposed_action"] == 6
    assert d["known_good_action"] == 1
    assert d["predicted_rank_of_known_good"] > 0
    # Confidence on the (wrongly) proposed must exceed confidence on known good
    assert d["confidence_on_proposed"] > d["confidence_on_known_good"]


def test_adapter_roundtrip(tmp_path):
    ad = _trivial_adapter(preferred_action=3)
    path = tmp_path / "adapter.json"
    ad.save(path)
    loaded = HeadBAdapter.load(path)
    assert loaded.machine == ad.machine
    assert loaded.n_classes == ad.n_classes
    # Prediction identical across round-trip
    features = [0.0] * len(ALL_FEATURE_NAMES)
    a, p = ad.predict(features)
    a2, p2 = loaded.predict(features)
    assert a == a2
    assert np.allclose(p, p2)


def test_softmax_probs_sum_to_one():
    ad = _trivial_adapter()
    d = _compute_delta([0.1] * len(ALL_FEATURE_NAMES), known_good=0, adapter=ad)
    s = sum(d["softmax_probs"])
    assert abs(s - 1.0) < 1e-6


if __name__ == "__main__":
    failures = 0
    for name in list(globals()):
        if name.startswith("test_"):
            try:
                fn = globals()[name]
                import inspect
                if "tmp_path" in inspect.signature(fn).parameters:
                    import tempfile
                    with tempfile.TemporaryDirectory() as td:
                        fn(Path(td))
                else:
                    fn()
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                failures += 1
    sys.exit(1 if failures else 0)
