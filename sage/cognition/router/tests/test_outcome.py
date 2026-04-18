#!/usr/bin/env python3
"""
Unit tests for OutcomeTracker — Phase 0 Track 6.
================================================

Covers the Track 6 deliverable matrix:

 1. Basic happy path: decision at tick N, SNARC at N+1..N+5 → complete
    outcome with correct trajectory length and resolution score.
 2. Tick gap > TRAJECTORY_TICKS → status='incomplete' with partial
    trajectory.
 3. Kernel restart: buffer persisted to disk, reloaded into a fresh
    tracker, backfill resumes.
 4. SNARC resolution score matches PRD §4.7.E formula (Δ dominant dim).
 5. Multiple concurrent decisions with overlapping windows.
 6. No-op: decision recorded but NO observations → flush emits
    'incomplete' with empty trajectory.
 7. Sidecar reader auto-merges outcomes onto main records.
 8. ``merge_outcomes=False`` yields main records untouched.
 9. Plugin failure mid-window marks status='failed' on that tick.
10. Out-of-order tick observations ignored + counted.
11. Max-pending cap evicts oldest as incomplete.
12. Atomic writes: tmp files cleaned up after persist.
13. Zero-salience SNARC at decision → resolution_score = None.
14. Duplicate register_decision rejected.
15. Outcome schema version stamped on every sidecar entry.

All tests are torch-free and use ``tmp_path`` for filesystem isolation.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from sage.cognition.router.outcome import (
    OUTCOME_SCHEMA_VERSION,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    TRAJECTORY_TICKS,
    OutcomeTracker,
    _snarc_resolution_score,
)
from sage.cognition.router.data import (
    RouterDatasetReader,
    RouterDatasetWriter,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ──────────────────────────────────────────────────────────────────────


def _fixed_clock(year: int = 2026, month: int = 4, day: int = 17):
    """Deterministic UTC clock for partition-naming assertions."""
    frozen = datetime(year, month, day, 12, 0, 0, tzinfo=timezone.utc)
    return lambda: frozen


def _snarc(**kwargs: float) -> Dict[str, float]:
    """Canonical SNARC dict with zero defaults."""
    out = {"arousal": 0.0, "conflict": 0.0, "surprise": 0.0,
           "reward": 0.0, "novelty": 0.0}
    out.update(kwargs)
    return out


def _read_outcome_lines(base_dir: Path, machine: str) -> list:
    """Read every sidecar entry back — uncompressed or gzipped."""
    records = []
    for path in sorted((base_dir / machine).glob("outcome_*.jsonl*")):
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
    return records


@pytest.fixture
def tracker_factory(tmp_path):
    """Produce tracker instances wired to a fresh tmp_path."""
    def _make(compress: bool = False, buffer_size: int = 1,
             max_pending: int = 4096, machine: str = "testmachine",
             clock=None) -> OutcomeTracker:
        return OutcomeTracker(
            base_dir=tmp_path,
            machine=machine,
            compress=compress,
            buffer_size=buffer_size,
            max_pending=max_pending,
            clock=clock or _fixed_clock(),
        )
    return _make


# ──────────────────────────────────────────────────────────────────────
# 1. Happy path — full 5-tick window
# ──────────────────────────────────────────────────────────────────────


def test_basic_full_window_outcome(tracker_factory, tmp_path):
    tracker = tracker_factory()

    # Decision at tick 100 with dominant arousal=0.9.
    tracker.register_decision(
        record_id="rec-1",
        tick_at_decision=100,
        snarc_at_decision=_snarc(arousal=0.9, conflict=0.2),
    )

    # Feed 5 post-decision ticks; arousal decays linearly.
    for i, a in enumerate([0.8, 0.6, 0.4, 0.2, 0.1], start=1):
        tracker.observe_tick(tick=100 + i, snarc=_snarc(arousal=a))

    tracker.close()

    outcomes = _read_outcome_lines(tmp_path, "testmachine")
    assert len(outcomes) == 1
    entry = outcomes[0]
    assert entry["record_id"] == "rec-1"
    oc = entry["outcome"]
    assert oc["status"] == STATUS_COMPLETE
    assert oc["outcome_schema_version"] == OUTCOME_SCHEMA_VERSION
    assert len(oc["snarc_trajectory"]) == TRAJECTORY_TICKS
    assert oc["tick_at_decision"] == 100
    assert oc["resolved_at_tick"] == 105
    # Resolution score = arousal_at_decision (0.9) - final_arousal (0.1) = 0.8
    assert oc["snarc_resolution_score"] == pytest.approx(0.8)

    stats = tracker.get_stats()
    assert stats["outcomes_emitted_complete"] == 1
    assert stats["outcomes_emitted_incomplete"] == 0
    assert stats["pending"] == 0


# ──────────────────────────────────────────────────────────────────────
# 2. Tick gap > window → incomplete
# ──────────────────────────────────────────────────────────────────────


def test_tick_gap_flushes_incomplete(tracker_factory, tmp_path):
    tracker = tracker_factory()
    tracker.register_decision("rec-gap", 50, _snarc(arousal=0.7))

    # Feed one valid tick, then a massive gap.
    tracker.observe_tick(tick=51, snarc=_snarc(arousal=0.65))
    tracker.observe_tick(tick=100, snarc=_snarc(arousal=0.5))

    tracker.close()

    outcomes = _read_outcome_lines(tmp_path, "testmachine")
    assert len(outcomes) == 1
    oc = outcomes[0]["outcome"]
    assert oc["status"] == STATUS_INCOMPLETE
    assert len(oc["snarc_trajectory"]) == 1  # only tick 51 captured
    assert oc["resolved_at_tick"] == 55  # window end (50 + 5)


# ──────────────────────────────────────────────────────────────────────
# 3. Kernel restart — buffer persistence
# ──────────────────────────────────────────────────────────────────────


def test_kernel_restart_resumes_buffer(tracker_factory, tmp_path):
    t1 = tracker_factory()
    t1.register_decision("rec-R", 10, _snarc(arousal=0.8))
    t1.observe_tick(11, _snarc(arousal=0.75))
    t1.observe_tick(12, _snarc(arousal=0.7))
    t1.persist()  # snapshot without flushing

    # Simulate kernel restart — DO NOT close t1 (that would flush).
    # Abandon t1 and construct a fresh tracker reading the same buffer.
    t2 = OutcomeTracker(
        base_dir=tmp_path,
        machine="testmachine",
        compress=False,
        buffer_size=1,
        clock=_fixed_clock(),
    )
    pending = t2.pending_record_ids()
    assert "rec-R" in pending

    # Complete the trajectory on t2.
    t2.observe_tick(13, _snarc(arousal=0.6))
    t2.observe_tick(14, _snarc(arousal=0.5))
    t2.observe_tick(15, _snarc(arousal=0.4))
    t2.close()

    outcomes = _read_outcome_lines(tmp_path, "testmachine")
    assert len(outcomes) == 1
    oc = outcomes[0]["outcome"]
    assert oc["status"] == STATUS_COMPLETE
    assert len(oc["snarc_trajectory"]) == TRAJECTORY_TICKS


def test_kernel_restart_stale_buffer_flushes_incomplete(tracker_factory, tmp_path):
    """Buffer entries older than last_observed_tick by > window are
    force-flushed as incomplete on load."""
    t1 = tracker_factory()
    t1.register_decision("rec-stale", 10, _snarc(arousal=0.8))
    # Pretend many ticks passed — set last_observed_tick forward via
    # an observation that doesn't touch our pending (wrong record_id).
    t1.observe_tick(100, _snarc(arousal=0.1))  # this flushes rec-stale
    t1.persist()

    # rec-stale was flushed on the gap check, not stored. Now register a
    # stale record, persist, and force last_observed_tick on reload.
    t1.register_decision("rec-stale2", 10, _snarc(arousal=0.8))
    # Don't observe — just persist with last_observed_tick stuck at 100.
    t1.persist()
    t1._writer.close()  # don't flush pending outcomes yet

    t2 = OutcomeTracker(
        base_dir=tmp_path,
        machine="testmachine",
        compress=False,
        buffer_size=1,
        clock=_fixed_clock(),
    )
    # rec-stale2 should have been emitted incomplete on load (100 > 10+5).
    assert "rec-stale2" not in t2.pending_record_ids()
    t2.close()

    outcomes = _read_outcome_lines(tmp_path, "testmachine")
    rec_ids = [o["record_id"] for o in outcomes]
    assert "rec-stale2" in rec_ids


# ──────────────────────────────────────────────────────────────────────
# 4. SNARC resolution score — PRD §4.7.E
# ──────────────────────────────────────────────────────────────────────


def test_resolution_score_picks_dominant_dim():
    # conflict is dominant at decision (0.9).
    snarc0 = _snarc(arousal=0.3, conflict=0.9, surprise=0.2)
    # Final trajectory sample: conflict reduced to 0.2, arousal risen.
    traj = [
        (1, _snarc(arousal=0.5, conflict=0.7, surprise=0.2)),
        (2, _snarc(arousal=0.6, conflict=0.5, surprise=0.2)),
        (3, _snarc(arousal=0.7, conflict=0.3, surprise=0.2)),
        (4, _snarc(arousal=0.8, conflict=0.25, surprise=0.2)),
        (5, _snarc(arousal=0.9, conflict=0.2, surprise=0.2)),
    ]
    score = _snarc_resolution_score(snarc0, traj)
    # Dominant is conflict (0.9). Final conflict = 0.2. Δ = 0.7.
    assert score == pytest.approx(0.7)


def test_resolution_score_none_when_all_zero():
    # Reward doesn't count as "dominant" — only arousal/conflict/surprise.
    snarc0 = _snarc(reward=0.9, novelty=0.8)
    traj = [(1, _snarc(reward=0.1))]
    assert _snarc_resolution_score(snarc0, traj) is None


def test_resolution_score_none_when_trajectory_empty():
    snarc0 = _snarc(arousal=0.9)
    assert _snarc_resolution_score(snarc0, []) is None


# ──────────────────────────────────────────────────────────────────────
# 5. Overlapping concurrent decisions
# ──────────────────────────────────────────────────────────────────────


def test_concurrent_decisions_overlapping_windows(tracker_factory, tmp_path):
    tracker = tracker_factory()
    # Three decisions spanning overlapping windows. Each gets observations
    # at EVERY subsequent tick so all three should complete cleanly.
    tracker.register_decision("a", 100, _snarc(arousal=0.8))
    tracker.register_decision("b", 102, _snarc(conflict=0.9))
    tracker.register_decision("c", 104, _snarc(surprise=0.7))
    # Ticks 101..109 — every decision gets up to 5 post-decision samples.
    for tick, snarc_vals in [
        (101, {"arousal": 0.7}),
        (102, {"arousal": 0.65, "conflict": 0.85}),
        (103, {"arousal": 0.6, "conflict": 0.8}),
        (104, {"arousal": 0.5, "conflict": 0.7}),
        (105, {"arousal": 0.4, "conflict": 0.6, "surprise": 0.6}),
        (106, {"arousal": 0.3, "conflict": 0.5, "surprise": 0.5}),
        (107, {"arousal": 0.2, "conflict": 0.4, "surprise": 0.4}),
        (108, {"conflict": 0.3, "surprise": 0.3}),
        (109, {"surprise": 0.2}),
    ]:
        tracker.observe_tick(tick, _snarc(**snarc_vals))
    tracker.close()

    outcomes = _read_outcome_lines(tmp_path, "testmachine")
    by_id = {o["record_id"]: o["outcome"] for o in outcomes}
    assert set(by_id.keys()) == {"a", "b", "c"}
    # Each gets exactly the 5 post-decision ticks in its window.
    for rid in ("a", "b", "c"):
        assert by_id[rid]["status"] == STATUS_COMPLETE, (
            f"{rid} not complete: {by_id[rid]}"
        )
        assert len(by_id[rid]["snarc_trajectory"]) == TRAJECTORY_TICKS


# ──────────────────────────────────────────────────────────────────────
# 6. No observations → incomplete on flush
# ──────────────────────────────────────────────────────────────────────


def test_noop_decision_flushed_incomplete(tracker_factory, tmp_path):
    tracker = tracker_factory()
    tracker.register_decision("lonely", 999, _snarc(arousal=0.5))
    tracker.close()  # flushes

    outcomes = _read_outcome_lines(tmp_path, "testmachine")
    assert len(outcomes) == 1
    oc = outcomes[0]["outcome"]
    assert oc["status"] == STATUS_INCOMPLETE
    assert oc["snarc_trajectory"] == []
    # Resolution score undefined without trajectory.
    assert oc["snarc_resolution_score"] is None


# ──────────────────────────────────────────────────────────────────────
# 7 + 8. Reader merges outcomes / opt-out
# ──────────────────────────────────────────────────────────────────────


def test_reader_merges_outcomes_by_record_id(tmp_path):
    """End-to-end: writer drops records, tracker emits outcomes, reader
    merges them at load time.
    """
    machine = "integ"
    clock = _fixed_clock(2026, 4, 17)
    writer = RouterDatasetWriter(
        base_dir=tmp_path, machine=machine,
        compress=False, buffer_size=1, clock=clock,
    )
    # Seed two main-partition records with outcome=None.
    for rid, tick, arousal in [("r1", 10, 0.8), ("r2", 20, 0.5)]:
        writer.append({
            "record_id": rid,
            "schema_version": "0.1.0",
            "timestamp": 1700000000.0,
            "machine": machine,
            "router_input": {"tick": tick},
            "router_output": {"action": "noop"},
            "outcome": None,
        })
    writer.close()

    # Tracker emits an outcome only for r1 — r2 stays without outcome.
    tracker = OutcomeTracker(
        base_dir=tmp_path, machine=machine,
        compress=False, buffer_size=1, clock=clock,
    )
    tracker.register_decision("r1", 10, _snarc(arousal=0.8))
    for i, a in enumerate([0.7, 0.6, 0.5, 0.4, 0.3], start=1):
        tracker.observe_tick(10 + i, _snarc(arousal=a))
    tracker.close()

    reader = RouterDatasetReader(base_dir=tmp_path)
    merged = {r["record_id"]: r for r in reader.read_partition(machine=machine)}
    assert merged["r1"]["outcome"] is not None
    assert merged["r1"]["outcome"]["status"] == STATUS_COMPLETE
    assert merged["r2"]["outcome"] is None


def test_reader_opt_out_preserves_main_state(tmp_path):
    machine = "integ2"
    clock = _fixed_clock(2026, 4, 17)
    writer = RouterDatasetWriter(
        base_dir=tmp_path, machine=machine,
        compress=False, buffer_size=1, clock=clock,
    )
    writer.append({
        "record_id": "r1",
        "schema_version": "0.1.0",
        "timestamp": 1700000000.0,
        "machine": machine,
        "outcome": None,
    })
    writer.close()

    tracker = OutcomeTracker(
        base_dir=tmp_path, machine=machine,
        compress=False, buffer_size=1, clock=clock,
    )
    tracker.register_decision("r1", 0, _snarc(arousal=0.9))
    for i, a in enumerate([0.1] * 5, start=1):
        tracker.observe_tick(i, _snarc(arousal=a))
    tracker.close()

    reader = RouterDatasetReader(base_dir=tmp_path)
    forensic = list(reader.read_partition(machine=machine, merge_outcomes=False))
    assert len(forensic) == 1
    assert forensic[0]["outcome"] is None  # no merge


# ──────────────────────────────────────────────────────────────────────
# 9. Plugin failure mid-window
# ──────────────────────────────────────────────────────────────────────


def test_plugin_failure_emits_failed_status(tracker_factory, tmp_path):
    tracker = tracker_factory()
    tracker.register_decision("boom", 5, _snarc(arousal=0.8))
    tracker.observe_tick(6, _snarc(arousal=0.7))
    tracker.observe_tick(7, _snarc(arousal=0.6), failed_record_ids=["boom"])
    tracker.close()

    outcomes = _read_outcome_lines(tmp_path, "testmachine")
    assert len(outcomes) == 1
    oc = outcomes[0]["outcome"]
    assert oc["status"] == STATUS_FAILED
    # Trajectory captured one sample (tick 6); tick-7 failure doesn't
    # extend trajectory.
    assert len(oc["snarc_trajectory"]) == 1


# ──────────────────────────────────────────────────────────────────────
# 10. Out-of-order ticks
# ──────────────────────────────────────────────────────────────────────


def test_out_of_order_ticks_ignored(tracker_factory):
    tracker = tracker_factory()
    tracker.register_decision("x", 10, _snarc(arousal=0.8))
    tracker.observe_tick(11, _snarc(arousal=0.7))
    # Regression — should be ignored.
    tracker.observe_tick(5, _snarc(arousal=0.9))
    assert tracker.get_stats()["out_of_order_observations"] == 1
    # Pending still has x, and trajectory only has the tick-11 sample.
    tracker.close()


# ──────────────────────────────────────────────────────────────────────
# 11. Max-pending eviction
# ──────────────────────────────────────────────────────────────────────


def test_max_pending_evicts_oldest(tracker_factory, tmp_path):
    tracker = tracker_factory(max_pending=2)
    tracker.register_decision("a", 1, _snarc(arousal=0.5))
    tracker.register_decision("b", 2, _snarc(arousal=0.5))
    tracker.register_decision("c", 3, _snarc(arousal=0.5))  # evicts a
    stats = tracker.get_stats()
    assert stats["evictions"] == 1
    assert "a" not in tracker.pending_record_ids()
    tracker.close()

    outcomes = _read_outcome_lines(tmp_path, "testmachine")
    ids_by_status = {o["record_id"]: o["outcome"]["status"] for o in outcomes}
    assert ids_by_status.get("a") == STATUS_INCOMPLETE


# ──────────────────────────────────────────────────────────────────────
# 12. Atomic writes — no leftover .tmp after persist
# ──────────────────────────────────────────────────────────────────────


def test_persist_buffer_cleans_up_tmp(tracker_factory, tmp_path):
    tracker = tracker_factory()
    tracker.register_decision("a", 1, _snarc(arousal=0.5))
    tracker.persist()
    leftover = list((tmp_path / "testmachine").glob("*.tmp"))
    assert leftover == []
    # Buffer file itself exists and is valid JSON.
    buf = (tmp_path / "testmachine" / "outcome_buffer.json")
    assert buf.exists()
    with open(buf) as f:
        data = json.load(f)
    assert data["outcome_schema_version"] == OUTCOME_SCHEMA_VERSION
    assert len(data["pending"]) == 1


# ──────────────────────────────────────────────────────────────────────
# 13. Duplicate register_decision
# ──────────────────────────────────────────────────────────────────────


def test_duplicate_register_returns_false(tracker_factory):
    tracker = tracker_factory()
    assert tracker.register_decision("dup", 1, _snarc(arousal=0.5)) is True
    assert tracker.register_decision("dup", 2, _snarc(arousal=0.7)) is False
    tracker.close()


# ──────────────────────────────────────────────────────────────────────
# 14. Schema version stamped on every sidecar entry
# ──────────────────────────────────────────────────────────────────────


def test_outcome_schema_version_stamped(tracker_factory, tmp_path):
    tracker = tracker_factory()
    tracker.register_decision("v", 1, _snarc(arousal=0.5))
    for i in range(1, 6):
        tracker.observe_tick(1 + i, _snarc(arousal=0.5 - i * 0.05))
    tracker.close()
    outcomes = _read_outcome_lines(tmp_path, "testmachine")
    assert all(
        o["outcome"]["outcome_schema_version"] == OUTCOME_SCHEMA_VERSION
        for o in outcomes
    )
    assert all(
        o["schema_version"] == OUTCOME_SCHEMA_VERSION
        for o in outcomes
    )


# ──────────────────────────────────────────────────────────────────────
# 15. Optional signals propagate through
# ──────────────────────────────────────────────────────────────────────


def test_rpe_and_level_up_signals_recorded(tracker_factory, tmp_path):
    tracker = tracker_factory()
    tracker.register_decision("s", 0, _snarc(arousal=0.9))
    tracker.observe_tick(1, _snarc(arousal=0.7), rpe_signal=0.3)
    tracker.observe_tick(2, _snarc(arousal=0.5), level_up=True)
    tracker.observe_tick(3, _snarc(arousal=0.4))
    tracker.observe_tick(4, _snarc(arousal=0.3))
    tracker.observe_tick(5, _snarc(arousal=0.2), rpe_signal=0.8)
    tracker.close()
    outcomes = _read_outcome_lines(tmp_path, "testmachine")
    assert len(outcomes) == 1
    oc = outcomes[0]["outcome"]
    # rpe is last-write-wins; level_up is OR-accumulate.
    assert oc["rpe_signal"] == pytest.approx(0.8)
    assert oc["level_up_observed"] is True


# ──────────────────────────────────────────────────────────────────────
# 16. Decision-tick SNARC sample itself doesn't count
# ──────────────────────────────────────────────────────────────────────


def test_decision_tick_sample_excluded(tracker_factory):
    tracker = tracker_factory()
    tracker.register_decision("z", 10, _snarc(arousal=0.9))
    # tick == tick_at_decision — must be ignored.
    tracker.observe_tick(10, _snarc(arousal=0.9))
    assert len(tracker._pending["z"].trajectory) == 0
    tracker.observe_tick(11, _snarc(arousal=0.7))
    assert len(tracker._pending["z"].trajectory) == 1


# ──────────────────────────────────────────────────────────────────────
# 17. Gzip sidecar round-trip through reader
# ──────────────────────────────────────────────────────────────────────


def test_gzip_sidecar_readable_by_reader(tmp_path):
    machine = "gz"
    clock = _fixed_clock(2026, 4, 17)
    writer = RouterDatasetWriter(
        base_dir=tmp_path, machine=machine,
        compress=True, buffer_size=1, clock=clock,
    )
    writer.append({
        "record_id": "g1",
        "schema_version": "0.1.0",
        "timestamp": 1700000000.0,
        "machine": machine,
        "outcome": None,
    })
    writer.close()
    tracker = OutcomeTracker(
        base_dir=tmp_path, machine=machine,
        compress=True, buffer_size=1, clock=clock,
    )
    tracker.register_decision("g1", 0, _snarc(arousal=0.9))
    for i, a in enumerate([0.8, 0.7, 0.6, 0.5, 0.4], start=1):
        tracker.observe_tick(i, _snarc(arousal=a))
    tracker.close()

    # Confirm gzip outcome file was written.
    gzs = list((tmp_path / machine).glob("outcome_*.jsonl.gz"))
    assert len(gzs) == 1

    reader = RouterDatasetReader(base_dir=tmp_path)
    merged = list(reader.read_partition(machine=machine))
    assert len(merged) == 1
    assert merged[0]["outcome"]["status"] == STATUS_COMPLETE
