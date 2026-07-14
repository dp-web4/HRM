"""Tests for the SituationReport perception-integrator V1.

V1 is instrumentation-only — it collects existing channel sources into
a unified report and logs coverage. These tests verify:

  - The report builds without crashing on a minimal WM
  - Each channel returns None when its source data is absent (clean
    "missing channel" semantics, not crash-or-empty-dict ambiguity)
  - Each channel populates when its source data is present
  - channel_coverage() and summary_one_line() format correctly
  - log_situation_report() runs without error (just prints)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from sage.cognition.thalamic_router.situation_report import (
    SituationReport, build_situation_report, log_situation_report,
)


# ── Minimal mock objects (no production WM dependency) ─────────────────

class _MockCandidate:
    def __init__(self, cid: str, trust: float = 0.3, plausibility: float = 0.5):
        self.id = cid
        self.trust = trust
        self.plausibility = plausibility
        self.strategic_value = plausibility
    def score(self):
        return self.trust * self.strategic_value


class _MockPortfolio:
    def __init__(self, candidates=None, commit_tau=0.45):
        self.candidates = {c.id: c for c in (candidates or [])}
        self.commit_tau = commit_tau
    def best_score(self):
        if not self.candidates:
            return 0.0
        return max(c.score() for c in self.candidates.values())


class _MockRule:
    def __init__(self, action: str, evidence_count: int = 1):
        self.action = action
        self.evidence_count = evidence_count


class _MockWM:
    def __init__(self, game="test", level=0, objects=None,
                 faith_portfolio=None, causal_rules=None,
                 cartridge_retrievals=None):
        self.game = game
        self.level = level
        self.objects = objects or {}
        self.faith_portfolio = faith_portfolio
        self.causal_rules = causal_rules or []
        if cartridge_retrievals is not None:
            self._last_cartridge_retrievals = cartridge_retrievals


class _MockFrameData:
    def __init__(self, frame=None, available_actions=None, frame_idx=0):
        self.frame = frame or [[[0]*64]*64]   # one frame, 64x64
        self.available_actions = available_actions or [1, 2, 3, 4]
        self.frame_idx = frame_idx


# ── Tests ──────────────────────────────────────────────────────────────

def test_empty_wm_builds_minimal_report():
    """A WM with no channels populated should still build a report — the
    `now` channel always has something (frame + actions); others are None."""
    wm = _MockWM()
    fd = _MockFrameData()
    sr = build_situation_report(wm, fd, cycle=0)
    assert sr.game == "test"
    assert sr.level == 0
    assert sr.cycle == 0
    # now channel always has at least available_actions
    assert sr.now is not None
    # All other channels missing
    assert sr.then is None
    assert sr.laws is None
    # self_ channel: faith_portfolio is None so no faith data, but level/
    # levels_completed make it non-empty
    assert sr.self_ is not None  # level=0 always present
    assert sr.salient is None  # V1 always None
    # Coverage: now + self_ = 2/5 = 0.4
    assert sr.channel_coverage() == 0.4


def test_full_wm_populates_all_channels_except_salient():
    """A WM with portfolio + rules + retrievals populates 4/5 channels."""
    portfolio = _MockPortfolio(candidates=[
        _MockCandidate("test__rule__UP", trust=0.7, plausibility=0.8),
        _MockCandidate("test__rule__DOWN", trust=0.4, plausibility=0.5),
    ])
    rules = [_MockRule("UP", evidence_count=2), _MockRule("DOWN", evidence_count=1)]
    retrievals = [
        {"preview": "Move down to align with goal", "src_level": 0},
        {"preview": "Click on target to advance", "src_level": 1},
    ]
    wm = _MockWM(faith_portfolio=portfolio, causal_rules=rules,
                 cartridge_retrievals=retrievals)
    fd = _MockFrameData(available_actions=[1, 2, 3, 4, 5])
    sr = build_situation_report(wm, fd, cycle=5)
    # 4 channels present: now, then, laws, self_; salient still None
    assert sr.now is not None
    assert sr.then is not None
    assert sr.then["n_cartridge_candidates"] == 2
    assert sr.laws is not None
    assert sr.laws["n_active_rules"] == 2
    assert "UP" in sr.laws["actions_with_rules"]
    assert sr.self_ is not None
    assert sr.self_["faith_n_candidates"] == 2
    assert sr.self_["faith_best_score"] == round(0.7 * 0.8, 3)
    # Best candidate's trust=0.7 >= commit_tau=0.45 → in commit_ready_ids
    assert "test__rule__UP" in sr.self_["faith_commit_ready_ids"]
    assert sr.salient is None
    # 4/5 channels = 0.8 coverage
    assert sr.channel_coverage() == 0.8


def test_summary_one_line_format():
    """summary_one_line should be a tight space-separated mark sequence."""
    sr = SituationReport(
        game="g50t", level=1, cycle=3,
        now={"x": 1}, then=None, laws={"a": 1, "b": 2},
        self_={"l": 0}, salient=None,
    )
    s = sr.summary_one_line()
    # now and self_ and laws have content; then and salient missing
    assert "now✓" in s
    assert "then·" in s
    assert "laws✓" in s
    assert "self✓" in s
    assert "salient·" in s


def test_channel_coverage_edge_cases():
    """0/5 → 0.0; 5/5 → 1.0; intermediate sane."""
    sr_empty = SituationReport()
    assert sr_empty.channel_coverage() == 0.0
    sr_full = SituationReport(
        now={"x": 1}, then={"y": 1}, laws={"z": 1},
        self_={"w": 1}, salient={"v": 1},
    )
    assert sr_full.channel_coverage() == 1.0


def test_log_situation_report_does_not_crash():
    """log_situation_report should print without raising; format check via
    capture not required for V1."""
    sr = SituationReport(
        game="r11l", level=0, cycle=2,
        now={"object_count": 3}, then=None, laws=None,
        self_={"level": 0}, salient=None,
    )
    # Should print without error
    log_situation_report(sr)


def test_missing_faith_portfolio_self_channel_handles_gracefully():
    """When wm.faith_portfolio is None, self_ channel still populates
    from level/levels_completed without crashing."""
    wm = _MockWM(faith_portfolio=None)
    wm.level = 2
    fd = _MockFrameData()
    sr = build_situation_report(wm, fd, cycle=0)
    assert sr.self_ is not None
    assert sr.self_["level"] == 2
    assert "faith_n_candidates" not in sr.self_


def test_empty_rules_returns_none_laws():
    """A WM with no causal rules (or all zero-evidence) should leave
    laws channel as None — not an empty dict."""
    rules = [_MockRule("UP", evidence_count=0)]  # zero evidence
    wm = _MockWM(causal_rules=rules)
    fd = _MockFrameData()
    sr = build_situation_report(wm, fd)
    assert sr.laws is None  # zero-evidence rules don't count as active


if __name__ == "__main__":
    tests = [
        test_empty_wm_builds_minimal_report,
        test_full_wm_populates_all_channels_except_salient,
        test_summary_one_line_format,
        test_channel_coverage_edge_cases,
        test_log_situation_report_does_not_crash,
        test_missing_faith_portfolio_self_channel_handles_gracefully,
        test_empty_rules_returns_none_laws,
    ]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"\n[OK] {len(tests)}/{len(tests)} SituationReport tests passed")
