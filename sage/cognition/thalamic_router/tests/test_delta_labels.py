"""Tests for delta_labels — substrate-derived retrain signal extraction.

These are unit tests against in-memory record dicts. Live-stream smoke
tests are run via `python -m sage.cognition.thalamic_router.delta_labels`
which exercises the file I/O end-to-end.
"""
from __future__ import annotations

import gzip
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sage.cognition.thalamic_router.delta_labels import (
    DeltaLabel, load_delta_labels, summarize_labels,
    _sage_plays_self_label, _gameplay_label, _sage_plays_live_label,
)


# ───────────────────────────────────────────────────────────────────
# Individual label extractors
# ───────────────────────────────────────────────────────────────────

def _rec(**meta):
    return {"metadata": {"game": "toy_b", "game_id": "toy_b-test", "step_index": 1, **meta}}


def test_sage_plays_self_stuck_triggered_is_invoke_positive():
    r = _rec(sage_plays_self={"stuck_triggered": True, "llm_invoked": False,
                              "levels_before": 0, "levels_after": 0})
    L = _sage_plays_self_label(r)
    assert L is not None
    assert L.invoke_target == 1.0
    assert L.reason == "stuck_triggered"


def test_sage_plays_self_llm_advanced_is_positive_with_action_override():
    r = _rec(sage_plays_self={
        "stuck_triggered": False, "llm_invoked": True, "llm_action": 3,
        "levels_before": 0, "levels_after": 1, "state_after": "NOT_FINISHED",
    })
    L = _sage_plays_self_label(r)
    assert L is not None
    assert L.invoke_target == 1.0
    assert L.action_target == 3
    assert "llm_invoke_advanced" in L.reason


def test_sage_plays_self_nn_correct_advance_is_negative():
    """NN played (not invoked) and levels advanced — don't escalate invoke."""
    r = _rec(sage_plays_self={
        "stuck_triggered": False, "llm_invoked": False,
        "levels_before": 0, "levels_after": 1,
    })
    L = _sage_plays_self_label(r)
    assert L is not None
    assert L.invoke_target == 0.0
    assert L.reason == "nn_correct_advance"


def test_sage_plays_self_no_signal_returns_none():
    """Record without decision field → no label contribution.

    (v7 behavior: true 'no signal'. v8 adds nn_played_no_stuck for records
    that DO have decision=play; see next test.)"""
    r = _rec(sage_plays_self={
        "stuck_triggered": False, "llm_invoked": False,
        "levels_before": 0, "levels_after": 0,
    })
    assert _sage_plays_self_label(r) is None


def test_sage_plays_self_nn_played_no_stuck_is_negative():
    """v8 rebalancing: decision=play + no stuck + state_ok → invoke=0.

    Counter-example to v7's invoke_head paranoia. If we only train on
    stuck/advance signals, the invoke_head learns "fire broadly." These
    negatives teach it "don't fire when NN is working fine."
    """
    r = _rec(sage_plays_self={
        "decision": "play", "stuck_triggered": False, "llm_invoked": False,
        "levels_before": 0, "levels_after": 0,
        "state_after": "NOT_FINISHED",
    })
    L = _sage_plays_self_label(r)
    assert L is not None
    assert L.invoke_target == 0.0
    assert L.reason == "nn_played_no_stuck"


def test_sage_plays_self_game_over_not_treated_as_negative():
    """If play resulted in GAME_OVER, it's NOT evidence we shouldn't have
    invoked. Leave the label unset rather than emit a misleading negative."""
    r = _rec(sage_plays_self={
        "decision": "play", "stuck_triggered": False, "llm_invoked": False,
        "levels_before": 0, "levels_after": 0,
        "state_after": "GAME_OVER",
    })
    assert _sage_plays_self_label(r) is None


def test_gameplay_known_good_is_action_only_no_invoke_signal():
    """Solver's known-good action is training signal for action_head, not
    for invoke_head. invoke_target should remain None so aggregation doesn't
    false-positive on it."""
    r = _rec(known_good_action=4)
    L = _gameplay_label(r)
    assert L is not None
    assert L.action_target == 4
    assert L.invoke_target is None
    assert L.reason == "solver_known_good"


def test_sage_plays_live_high_rank_boosts_invoke():
    """When NN ranked solver's pick 3rd or worse, invoke should be positive."""
    r = _rec(
        known_good_action=6,
        sage_plays_live={"predicted_rank_of_known_good": 3, "correct": False},
    )
    L = _sage_plays_live_label(r)
    assert L is not None
    assert L.invoke_target == 1.0
    assert L.action_target == 6


def test_sage_plays_live_low_rank_returns_none():
    r = _rec(
        known_good_action=6,
        sage_plays_live={"predicted_rank_of_known_good": 0, "correct": True},
    )
    assert _sage_plays_live_label(r) is None


# ───────────────────────────────────────────────────────────────────
# Aggregation via the full pipeline
# ───────────────────────────────────────────────────────────────────

def _write_stream(tmp_path: Path, name: str, records):
    path = tmp_path / name
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def test_load_delta_labels_aggregates_multiple_sources(tmp_path):
    # Same step-key appears in two records — one solver-known-good (action),
    # one sage_plays_self (stuck). Aggregation should merge both.
    records_a = [
        {"metadata": {"game": "toy_b", "game_id": "toy_b-a", "step_index": 5,
                      "known_good_action": 6}},
    ]
    records_b = [
        {"metadata": {"game": "toy_b", "game_id": "toy_b-a", "step_index": 5,
                      "sage_plays_self": {"stuck_triggered": True,
                                           "llm_invoked": False,
                                           "levels_before": 0, "levels_after": 0}}},
    ]
    pa = _write_stream(tmp_path, "a.jsonl.gz", records_a)
    pb = _write_stream(tmp_path, "b.jsonl.gz", records_b)
    labels = load_delta_labels([pa, pb])
    key = ("toy_b", "toy_b-a", 5)
    assert key in labels
    merged = labels[key]
    # Invoke signal from the stuck record
    assert merged.invoke_target == 1.0
    # Action target from the gameplay record
    assert merged.action_target == 6
    # Both sources counted
    assert merged.source_records == 2


def test_summarize_labels_counts_correctly():
    labels = {
        ("toy_b", "toy_b-a", 1): DeltaLabel(invoke_target=1.0, reason="stuck_triggered"),
        ("toy_b", "toy_b-a", 2): DeltaLabel(invoke_target=0.0, reason="nn_correct_advance"),
        ("toy_b", "toy_b-a", 3): DeltaLabel(invoke_target=None, action_target=4,
                                           reason="solver_known_good"),
    }
    s = summarize_labels(labels)
    assert s["total"] == 3
    assert s["invoke_positive"] == 1
    assert s["invoke_negative"] == 1
    assert s["invoke_unset"] == 1
    assert s["action_overrides"] == 1
    assert s["by_game"] == {"toy_b": 3}


if __name__ == "__main__":
    import tempfile
    failures = 0
    for name in list(globals()):
        if not name.startswith("test_"):
            continue
        fn = globals()[name]
        try:
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"  \u2713 {name}")
        except AssertionError as e:
            print(f"  \u2717 {name}: {e}")
            failures += 1
        except Exception as e:
            print(f"  \u2717 {name}: {type(e).__name__}: {e}")
            failures += 1
    if failures:
        print(f"\n{failures} failures")
        sys.exit(1)
    print("\nAll tests passed")
