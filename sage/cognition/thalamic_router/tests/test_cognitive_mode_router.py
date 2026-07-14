"""Tests for the commit_faith cognitive mode (+ goal_plausibility signal).

The faith mode is the complement of perseveration_break: persist toward a
plausible goal through a feedback desert; abandon only on disconfirmation
(conflict/surprise), never on mere absence of reward. Gated on goal_plausibility
(the threshold-faith result).

Pure functions (real select_mode / SNARCSignals / _goal_plausibility) — no mocks.
Runs under pytest if present, else as a plain script: `python3 test_cognitive_mode_router.py`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root (contains sage/)

from sage.cognition.thalamic_router.cognitive_mode_router import (  # noqa: E402
    SNARCSignals, RouterState, select_mode, get_mode, _goal_plausibility,
)

os.environ["SAGE_COGNITIVE_ROUTER"] = "1"   # router must be enabled to select modes


def test_commit_faith_fires_in_desert():
    """Plausible goal, coherent, no reward yet -> commit_faith."""
    s = SNARCSignals(goal_plausibility=0.9, conflict=0.0, surprise=0.0, reward=0.0)
    mode, score, _ = select_mode(s, RouterState())
    assert mode == "commit_faith", f"expected commit_faith, got {mode} ({score:.2f})"


def test_disconfirmation_beats_faith():
    """Same high plausibility but now DISCONFIRMED (conflict+arousal) -> the
    redirect modes win; faith is suppressed (the core 'abandon on disconfirmation,
    not absence of confirmation' rule)."""
    s = SNARCSignals(goal_plausibility=0.9, conflict=0.9, arousal=0.9, surprise=0.0)
    mode, score, _ = select_mode(s, RouterState())
    assert mode != "commit_faith", f"faith should yield to disconfirmation, got {mode}"
    assert mode == "perseveration_break", f"expected perseveration_break, got {mode}"


def test_faith_needs_plausibility():
    """Low plausibility -> faith stays below its threshold -> not selected."""
    s = SNARCSignals(goal_plausibility=0.2, conflict=0.0, surprise=0.0, reward=0.0)
    mode, _, _ = select_mode(s, RouterState())
    assert mode != "commit_faith", f"faith should not fire at low plausibility, got {mode}"


def test_high_surprise_suppresses_faith():
    """Predictions diverging (surprise) is disconfirmation -> not faith."""
    s = SNARCSignals(goal_plausibility=0.9, surprise=0.9, conflict=0.0)
    mode, _, _ = select_mode(s, RouterState())
    assert mode != "commit_faith", f"surprise should suppress faith, got {mode}"


def test_router_disabled_returns_default():
    os.environ["SAGE_COGNITIVE_ROUTER"] = "0"
    try:
        s = SNARCSignals(goal_plausibility=0.9)
        mode, _, reason = select_mode(s, RouterState())
        assert mode == "active" and reason == "router-disabled", (mode, reason)
    finally:
        os.environ["SAGE_COGNITIVE_ROUTER"] = "1"


def test_goal_plausibility_extractor():
    # no evidence -> conservative 0.0 (faith never fires spuriously)
    assert _goal_plausibility(object(), []) == 0.0

    # explicit world-model confidence wins
    class WMConf:
        goal_confidence = 0.7
    assert abs(_goal_plausibility(WMConf(), []) - 0.7) < 1e-6

    # coherent goal-directed stretch (win hypothesis + predictions tracking, no errors)
    class WMGoal:
        win_hypothesis = "deliver all tiles"
    log = [{"expect": "tile moves", "expect_passed": True} for _ in range(5)]
    p = _goal_plausibility(WMGoal(), log)
    assert p > 0.9, f"coherent stretch should be highly plausible, got {p}"

    # engine errors erode plausibility (incoherence)
    log_err = log + [{"engine_error": True} for _ in range(5)]
    p_err = _goal_plausibility(WMGoal(), log_err)
    assert p_err < p, f"errors should lower plausibility ({p_err} !< {p})"


def test_faith_prompt_override_injects_framing():
    mode = get_mode("commit_faith")
    assert mode is not None and mode.prompt_override is not None
    out = mode.prompt_override("BASE_PROMPT", {})
    assert "FAITH" in out and "BASE_PROMPT" in out
    assert "disconfirmation" in out.lower()


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
