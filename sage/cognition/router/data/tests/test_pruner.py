"""
Tests for the Phase 0 Track 9 SNARC-driven storage pruner.

Covers PRD §5.6 retention policy + §7.10 agent-zero defense:

  * Synthetic 1k-record partition with controlled SNARC distribution —
    retention rate per quintile per age bracket matches spec.
  * Pinned records preserved across all age brackets (including 90+).
  * Idempotency: re-running prune on already-pruned partition is a no-op.
  * Agent-zero check fires when pruning would collapse modal-class margin.
  * Concurrent safety: today's partition fails fast; mtime-recent partition
    is skipped; lock-file convention honored.
  * Dry-run reports expected retention without rewriting disk.
  * PruneStats surface: records, pinned, histograms, bytes, agent-zero check.
  * prune_all iterates partitions with failure isolation.

No torch, no network.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from sage.cognition.router.data import (
    RouterDatasetPruner,
    PruneStats,
    AGENT_ZERO_MARGIN_PP,
    RECOGNIZED_PIN_KINDS,
    SCHEMA_VERSION,
)


# ── helpers ────────────────────────────────────────────────────────


def _make_record(
    i: int,
    action: str,
    arousal: float,
    conflict: float = 0.0,
    reward: float = 0.0,
    pinned: Optional[str] = None,
    machine: str = "testmachine",
) -> Dict[str, Any]:
    """Synthetic record in the Track 4 payload envelope.

    The pinned kind is threaded through ``payload["pinned"]`` per the
    mission spec. Salience is max(arousal, conflict, |reward|) so callers
    can directly set the dimension they care about.
    """
    payload: Dict[str, Any] = {
        "router_input": {
            "tick": i,
            "snarc": {
                "surprise": 0.1,
                "novelty": 0.1,
                "arousal": arousal,
                "reward": reward,
                "conflict": conflict,
            },
        },
        "router_output": {
            "action": action,
            "confidence": 0.5,
        },
    }
    if pinned is not None:
        payload["pinned"] = pinned
    return {
        "record_id": f"rec-{i:08d}",
        "schema_version": SCHEMA_VERSION,
        "timestamp": 1_700_000_000.0 + i,
        "machine": machine,
        "payload": payload,
    }


def _write_partition(
    path: Path,
    records: List[Dict[str, Any]],
    compress: bool = False,
) -> None:
    """Write records as JSONL (or JSONL.gz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        with gzip.open(str(path), "wt", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")


def _read_partition(path: Path) -> List[Dict[str, Any]]:
    """Read records from a partition (plain or gzipped)."""
    if path.suffix == ".gz":
        opener = lambda: gzip.open(str(path), "rt", encoding="utf-8")
    else:
        opener = lambda: open(path, "r", encoding="utf-8")
    records: List[Dict[str, Any]] = []
    with opener() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _set_mtime(path: Path, when: datetime) -> None:
    """Back-date a file's mtime so the active-write guard lets us touch it."""
    ts = when.timestamp()
    os.utime(str(path), (ts, ts))


def _backdate_partition(path: Path, age_days: int, now: datetime) -> None:
    """Set mtime to ``age_days`` before ``now`` — outside active-write window."""
    _set_mtime(path, now - timedelta(days=age_days, hours=1))


def _fixed_clock(when: datetime):
    """Return a clock callable that yields a fixed UTC time."""
    return lambda: when


def _partition_path(base: Path, machine: str, partition_date, compress: bool = False) -> Path:
    suffix = ".jsonl.gz" if compress else ".jsonl"
    return base / machine / f"{partition_date.isoformat()}{suffix}"


# ── basic structural tests ────────────────────────────────────────


def test_stats_dataclass_roundtrip() -> None:
    stats = PruneStats(path="/x", age_days=10, bracket_rule="7-30d top-2-quintiles")
    stats.records_before = 100
    stats.records_after = 60
    stats.bytes_before = 1000
    stats.bytes_after = 400
    assert stats.records_dropped == 40
    assert stats.bytes_reclaimed == 600
    d = stats.to_dict()
    assert d["records_dropped"] == 40
    assert d["bytes_reclaimed"] == 600
    assert d["bracket_rule"] == "7-30d top-2-quintiles"


# ── retention rate per bracket ─────────────────────────────────────


@pytest.mark.parametrize(
    "age_days,expected_min_quintile",
    [
        (3, 0),     # 0-7d: keep all
        (15, 3),    # 7-30d: keep top 2 quintiles
        (60, 4),    # 30-90d: keep top 1 quintile
        (100, None),  # 90+d: pinned only
    ],
)
def test_retention_rate_per_bracket(
    tmp_path: Path, age_days: int, expected_min_quintile: Optional[int]
) -> None:
    """With a uniform SNARC distribution, retention rate matches the bracket rule.

    We build 1000 records with linearly-spaced arousal so the 5 quintiles
    are exactly 200 records each. Then prune and verify the survivors
    match the expected quintile slice.
    """
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    partition_date = now.date() - timedelta(days=age_days)
    path = _partition_path(tmp_path, "sprout", partition_date)

    # 1000 records, arousal evenly spaced in [0.001, 1.0].
    # Rotate action to stay away from modal-class collapse.
    actions = ["invoke", "habit", "noop"]
    records = [
        _make_record(
            i,
            action=actions[i % 3],
            arousal=(i + 1) / 1000.0,
        )
        for i in range(1000)
    ]
    _write_partition(path, records)
    _backdate_partition(path, age_days=age_days, now=now)

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats = pruner.prune_partition(path)

    assert stats.error is None, f"unexpected error: {stats.error}"
    assert stats.age_days == age_days

    # Derive expected survivor count from the retention rule.
    if expected_min_quintile is None:
        expected_kept = 0
    else:
        # 5-min_quintile quintiles × ~200 records each.
        expected_kept = (5 - expected_min_quintile) * 200

    # Agent-zero check should report "ok" — actions are balanced.
    assert stats.agent_zero_check == "ok", stats.to_dict()

    # Allow ±1% tolerance for quintile boundary tie-breaking.
    tol = max(20, int(expected_kept * 0.01))
    assert abs(stats.records_after - expected_kept) <= tol, (
        f"expected ~{expected_kept}, got {stats.records_after}; stats={stats.to_dict()}"
    )

    # Verify survivors land in the right quintile bucket(s).
    if expected_min_quintile is not None:
        surviving_quintiles = [
            i for i, c in enumerate(stats.salience_hist_after) if c > 0
        ]
        # All survivors must be at-or-above the threshold quintile.
        for q in surviving_quintiles:
            assert q >= expected_min_quintile, (
                f"survivors in quintile {q} below min {expected_min_quintile}; "
                f"hist={stats.salience_hist_after}"
            )


# ── pinned records ────────────────────────────────────────────────


def test_pinned_records_preserved_in_all_brackets(tmp_path: Path) -> None:
    """Pinned records survive at every age bracket including 90+ days."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)

    for age_days in (3, 15, 60, 120):
        partition_date = now.date() - timedelta(days=age_days)
        path = _partition_path(tmp_path, "sprout", partition_date)
        # 100 non-pinned low-salience records + 5 pinned low-salience records.
        # On 90+ bracket only the 5 pinned should remain. At 7-30 and 30-90,
        # the non-pinned low-salience records would drop; the 5 pinned
        # survive regardless.
        records = []
        actions = ["invoke", "habit", "noop"]
        for i in range(100):
            records.append(_make_record(i, action=actions[i % 3], arousal=0.01 + i * 0.001))
        # Add 5 pinned records at the LOWEST salience.
        for i in range(5):
            records.append(
                _make_record(
                    1000 + i,
                    action="invoke",
                    arousal=0.0,
                    pinned=RECOGNIZED_PIN_KINDS[i % len(RECOGNIZED_PIN_KINDS)],
                )
            )
        _write_partition(path, records)
        _backdate_partition(path, age_days=age_days, now=now)

        pruner = RouterDatasetPruner(clock=_fixed_clock(now))
        stats = pruner.prune_partition(path)
        assert stats.pinned_preserved == 5, (
            f"age={age_days}: pinned_preserved={stats.pinned_preserved}; stats={stats.to_dict()}"
        )

        # Read the rewritten file and confirm pinned flags survived.
        if stats.rewrote:
            survivors = _read_partition(path)
            pinned_survivors = [r for r in survivors if r["payload"].get("pinned")]
            assert len(pinned_survivors) == 5
            # Recognize all pinned kinds we emitted.
            kinds = {r["payload"]["pinned"] for r in pinned_survivors}
            assert kinds.issubset(set(RECOGNIZED_PIN_KINDS))


def test_pinned_with_unknown_kind_still_preserved(tmp_path: Path) -> None:
    """Unknown pin kind is preserved — fail-safe per the module docstring."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    partition_date = now.date() - timedelta(days=120)
    path = _partition_path(tmp_path, "sprout", partition_date)
    records = [
        _make_record(0, action="invoke", arousal=0.0, pinned="experimental_kind"),
        _make_record(1, action="habit", arousal=0.0),  # non-pinned low-salience
    ]
    _write_partition(path, records)
    _backdate_partition(path, 120, now)

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats = pruner.prune_partition(path)
    assert stats.pinned_preserved == 1
    assert "experimental_kind" in stats.pinned_by_kind


# ── agent-zero defense ────────────────────────────────────────────


def test_agent_zero_check_fires_when_would_collapse(tmp_path: Path) -> None:
    """Build a partition where pruning would collapse modal margin below 25pp."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    # 30-90d bracket: only top quintile survives.
    age_days = 60
    partition_date = now.date() - timedelta(days=age_days)
    path = _partition_path(tmp_path, "sprout", partition_date)

    # Mix:
    #   - 100 high-salience noop records (these would all survive the
    #     top-quintile-only rule).
    #   - 400 low-salience invoke records (these would all drop).
    #
    # BEFORE prune: modal = invoke (400/500=80%), next-most = noop (100/500=20%).
    #     margin = 60pp
    # AFTER prune (top quintile only): modal = noop (100%).
    #     margin with 1 class → None → collapse.
    records = []
    for i in range(400):
        records.append(_make_record(i, action="invoke", arousal=0.01 + i * 0.0005))
    for i in range(100):
        # High-salience noop records — top quintile only.
        records.append(_make_record(1000 + i, action="noop", arousal=0.9 + i * 0.0005))
    _write_partition(path, records)
    _backdate_partition(path, age_days, now)

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats = pruner.prune_partition(path)

    assert stats.agent_zero_check == "skipped_would_collapse", stats.to_dict()
    assert stats.skipped_reason == "agent_zero_collapse"
    assert stats.rewrote is False
    # The original file should be byte-identical.
    read_back = _read_partition(path)
    assert len(read_back) == 500


def test_agent_zero_margin_above_threshold_allows_prune(tmp_path: Path) -> None:
    """If post-prune margin stays above threshold, prune proceeds."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    age_days = 60
    partition_date = now.date() - timedelta(days=age_days)
    path = _partition_path(tmp_path, "sprout", partition_date)

    # Engineer the partition so even top-quintile has ≥2 classes with
    # sufficient margin. 500 records: arousal in [0.01, 1.0].
    # Assign classes so quintile 4 (top 20%, arousal > 0.8) has:
    #   * 60% invoke
    #   * 30% habit
    #   * 10% noop
    # Margin 30pp > threshold 25pp → allowed.
    records = []
    for i in range(500):
        ar = 0.01 + (i / 499.0) * 0.99
        if ar > 0.8:
            # Top quintile
            cycle = i % 10
            if cycle < 6:
                action = "invoke"
            elif cycle < 9:
                action = "habit"
            else:
                action = "noop"
        else:
            # Below top quintile — balance classes so pre-prune also has margin.
            action = ["invoke", "habit", "noop"][i % 3]
        records.append(_make_record(i, action=action, arousal=ar))
    _write_partition(path, records)
    _backdate_partition(path, age_days, now)

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats = pruner.prune_partition(path)
    assert stats.agent_zero_check == "ok", stats.to_dict()
    assert stats.rewrote is True


def test_agent_zero_not_applicable_on_single_class(tmp_path: Path) -> None:
    """Single-class partitions can't collapse — check is not_applicable."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    partition_date = now.date() - timedelta(days=15)
    path = _partition_path(tmp_path, "sprout", partition_date)
    records = [_make_record(i, action="noop", arousal=0.01 + i * 0.01) for i in range(100)]
    _write_partition(path, records)
    _backdate_partition(path, 15, now)

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats = pruner.prune_partition(path)
    assert stats.agent_zero_check == "not_applicable"


# ── idempotency ───────────────────────────────────────────────────


def test_idempotent_second_run_is_noop(tmp_path: Path) -> None:
    """Re-running prune on already-pruned partition does not drop records."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    age_days = 15
    partition_date = now.date() - timedelta(days=age_days)
    path = _partition_path(tmp_path, "sprout", partition_date)
    actions = ["invoke", "habit", "noop"]
    records = [
        _make_record(i, action=actions[i % 3], arousal=(i + 1) / 1000.0)
        for i in range(1000)
    ]
    _write_partition(path, records)
    _backdate_partition(path, age_days, now)

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats1 = pruner.prune_partition(path)
    assert stats1.rewrote is True
    kept_after_first = stats1.records_after

    # Re-backdate the file since rewrite reset mtime.
    _backdate_partition(path, age_days, now)

    stats2 = pruner.prune_partition(path)
    # Second run: quintiles re-computed from the (already top-2-quintile)
    # surviving distribution. Top 2 quintiles of that surviving set are
    # still the top 2 quintiles; no records drop.
    assert stats2.records_before == kept_after_first
    assert stats2.records_after == kept_after_first
    assert stats2.records_dropped == 0

    # Third run belt-and-suspenders.
    _backdate_partition(path, age_days, now)
    stats3 = pruner.prune_partition(path)
    assert stats3.records_dropped == 0


# ── concurrent safety ─────────────────────────────────────────────


def test_current_day_partition_fails_fast(tmp_path: Path) -> None:
    """Today's partition is skipped with a clear reason."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    partition_date = now.date()  # today
    path = _partition_path(tmp_path, "sprout", partition_date)
    records = [_make_record(0, action="invoke", arousal=0.5)]
    _write_partition(path, records)

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats = pruner.prune_partition(path)
    assert stats.skipped_reason == "current_day"
    assert stats.rewrote is False
    # File unchanged.
    assert len(_read_partition(path)) == 1


def test_mtime_recent_partition_is_skipped(tmp_path: Path) -> None:
    """A backdated-name file with a recent mtime is still treated as active."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    partition_date = now.date() - timedelta(days=15)
    path = _partition_path(tmp_path, "sprout", partition_date)
    records = [
        _make_record(i, action=["invoke", "habit", "noop"][i % 3], arousal=(i + 1) / 100.0)
        for i in range(100)
    ]
    _write_partition(path, records)
    # Leave mtime as-is (now). The filename says 15 days old, but the file
    # was written seconds ago → writer could still be appending.
    _set_mtime(path, now - timedelta(hours=1))  # within 24h window

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats = pruner.prune_partition(path)
    assert stats.skipped_reason == "active_write"
    assert stats.rewrote is False


def test_lock_file_blocks_prune(tmp_path: Path) -> None:
    """`.lock` sidecar skips the partition with a clear event."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    partition_date = now.date() - timedelta(days=15)
    path = _partition_path(tmp_path, "sprout", partition_date)
    records = [_make_record(0, action="invoke", arousal=0.5)]
    _write_partition(path, records)
    _backdate_partition(path, 15, now)

    # Place lock file per the convention.
    lock = path.with_suffix(path.suffix + ".lock")
    lock.write_text("pid=1234\n")

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats = pruner.prune_partition(path)
    assert stats.skipped_reason == "locked"
    assert stats.rewrote is False


# ── dry-run ───────────────────────────────────────────────────────


def test_dry_run_reports_without_rewriting(tmp_path: Path) -> None:
    """Dry-run returns a populated PruneStats but does not touch disk."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    age_days = 15
    partition_date = now.date() - timedelta(days=age_days)
    path = _partition_path(tmp_path, "sprout", partition_date)
    actions = ["invoke", "habit", "noop"]
    records = [_make_record(i, action=actions[i % 3], arousal=(i + 1) / 1000.0)
               for i in range(1000)]
    _write_partition(path, records)
    _backdate_partition(path, age_days, now)

    original_contents = path.read_bytes()
    original_mtime = path.stat().st_mtime

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats = pruner.prune_partition(path, dry_run=True)

    assert stats.dry_run is True
    assert stats.skipped_reason == "dry_run"
    assert stats.rewrote is False
    # Records-after reflects what WOULD be kept (top-2-quintile ≈ 400).
    assert 350 <= stats.records_after <= 450, stats.to_dict()
    # File untouched.
    assert path.read_bytes() == original_contents
    assert path.stat().st_mtime == original_mtime
    # Bytes-after is an estimate of the dry-run result.
    assert stats.bytes_after > 0 and stats.bytes_after < stats.bytes_before


# ── gzip round-trip ───────────────────────────────────────────────


def test_prunes_gzipped_partition(tmp_path: Path) -> None:
    """Gzipped partitions are re-read, pruned, and re-written gzipped."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    age_days = 15
    partition_date = now.date() - timedelta(days=age_days)
    path = _partition_path(tmp_path, "sprout", partition_date, compress=True)
    actions = ["invoke", "habit", "noop"]
    records = [
        _make_record(i, action=actions[i % 3], arousal=(i + 1) / 500.0)
        for i in range(500)
    ]
    _write_partition(path, records, compress=True)
    _backdate_partition(path, age_days, now)

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats = pruner.prune_partition(path)
    assert stats.rewrote is True
    # Output is still gzipped and readable.
    survivors = _read_partition(path)
    assert 0 < len(survivors) < 500


# ── prune_all ─────────────────────────────────────────────────────


def test_prune_all_iterates_partitions_with_failure_isolation(tmp_path: Path) -> None:
    """prune_all hits every partition and reports per-file stats."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    base = tmp_path / "dataset"

    # Three partitions across two machines, various ages.
    for machine, age_days in [("sprout", 3), ("sprout", 15), ("legion", 100)]:
        partition_date = now.date() - timedelta(days=age_days)
        path = _partition_path(base, machine, partition_date)
        actions = ["invoke", "habit", "noop"]
        records = [
            _make_record(i, action=actions[i % 3], arousal=(i + 1) / 100.0)
            for i in range(100)
        ]
        # For 100-day bracket, tag 3 records as pinned so at least some survive.
        if age_days == 100:
            for i in range(3):
                records.append(
                    _make_record(
                        1000 + i, action="invoke", arousal=0.0,
                        pinned="agent_zero_golden",
                    )
                )
        _write_partition(path, records)
        _backdate_partition(path, age_days, now)

    # Drop a junk file that doesn't match the YYYY-MM-DD convention —
    # pruner should report skipped_reason="non_partition_filename".
    junk = base / "sprout" / "README.jsonl"
    junk.write_text("{}\n")
    _backdate_partition(junk, 30, now)

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    results = pruner.prune_all(base)

    assert len(results) == 4
    # Junk file surfaces with a skip reason, NOT a raised exception.
    junk_stats = next(s for p, s in results.items() if p.name == "README.jsonl")
    assert junk_stats.skipped_reason == "non_partition_filename"

    # At age 3 no pruning happens (bracket keeps all).
    young_stats = next(
        s for p, s in results.items()
        if p.name.startswith((now.date() - timedelta(days=3)).isoformat())
    )
    assert young_stats.records_before == young_stats.records_after

    # Legion 100-day file: pinned-only. 3 pinned records survive.
    old_stats = next(
        s for p, s in results.items()
        if "legion" in p.parts
    )
    assert old_stats.pinned_preserved == 3


def test_prune_all_with_machine_filter(tmp_path: Path) -> None:
    """Machine filter narrows the pruner to one subdir."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    base = tmp_path / "dataset"
    for machine in ("sprout", "legion"):
        partition_date = now.date() - timedelta(days=15)
        path = _partition_path(base, machine, partition_date)
        records = [
            _make_record(i, action=["invoke", "habit", "noop"][i % 3],
                         arousal=(i + 1) / 100.0)
            for i in range(100)
        ]
        _write_partition(path, records)
        _backdate_partition(path, 15, now)

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    results = pruner.prune_all(base, machine="sprout")
    assert len(results) == 1
    assert all("sprout" in p.parts for p in results.keys())


# ── malformed input robustness ────────────────────────────────────


def test_corrupt_trailing_line_is_tolerated(tmp_path: Path) -> None:
    """A partial / corrupt trailing line is dropped without crashing."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    partition_date = now.date() - timedelta(days=15)
    path = _partition_path(tmp_path, "sprout", partition_date)
    actions = ["invoke", "habit", "noop"]
    records = [
        _make_record(i, action=actions[i % 3], arousal=(i + 1) / 100.0)
        for i in range(100)
    ]
    _write_partition(path, records)
    # Corrupt trailing content.
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"record_id": "bad", "pay')
    _backdate_partition(path, 15, now)

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats = pruner.prune_partition(path)
    assert stats.error is None
    # records_before reflects only the valid records (100), not 101.
    assert stats.records_before == 100


def test_missing_partition_returns_stats_not_exception(tmp_path: Path) -> None:
    """Non-existent path returns a populated stats with error."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats = pruner.prune_partition(tmp_path / "sprout" / "2026-01-01.jsonl")
    assert stats.skipped_reason == "missing"
    assert stats.error is not None


# ── PruneStats surface ────────────────────────────────────────────


def test_prune_stats_surfaces_required_fields(tmp_path: Path) -> None:
    """Verify the stats object carries the full observability surface."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    age_days = 15
    partition_date = now.date() - timedelta(days=age_days)
    path = _partition_path(tmp_path, "sprout", partition_date)
    actions = ["invoke", "habit", "noop"]
    records = [
        _make_record(i, action=actions[i % 3], arousal=(i + 1) / 1000.0)
        for i in range(1000)
    ]
    _write_partition(path, records)
    _backdate_partition(path, age_days, now)

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats = pruner.prune_partition(path)
    d = stats.to_dict()

    # Required observability fields from the mission spec.
    for field_name in (
        "records_before",
        "records_after",
        "records_dropped",
        "salience_hist_before",
        "salience_hist_after",
        "quintile_boundaries",
        "bytes_before",
        "bytes_after",
        "bytes_reclaimed",
        "agent_zero_check",
        "rewrote",
    ):
        assert field_name in d, f"missing field {field_name}"

    # Sanity: histograms sum to record counts.
    assert sum(stats.salience_hist_before) == stats.records_before
    assert sum(stats.salience_hist_after) == stats.records_after
    # Quintile boundaries always length-4.
    assert len(stats.quintile_boundaries) == 4


# ── Track 4 compatibility ─────────────────────────────────────────


def test_uses_salience_formula_compatible_with_sampler() -> None:
    """Pruner uses the same max(arousal, conflict, |reward|) as Sampler."""
    from sage.cognition.router.data.sampling import salience_score

    # Case: reward dominates.
    s = salience_score({"arousal": 0.1, "conflict": 0.2, "reward": -0.9})
    assert s == pytest.approx(0.9)
    # Case: arousal dominates.
    s = salience_score({"arousal": 0.8, "conflict": 0.1, "reward": 0.2})
    assert s == pytest.approx(0.8)
    # Case: conflict dominates.
    s = salience_score({"arousal": 0.1, "conflict": 0.7, "reward": 0.2})
    assert s == pytest.approx(0.7)


def test_atomic_rewrite_leaves_no_tmp_on_success(tmp_path: Path) -> None:
    """No `.tmp` sidecar remains after a successful prune."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    age_days = 15
    partition_date = now.date() - timedelta(days=age_days)
    path = _partition_path(tmp_path, "sprout", partition_date)
    actions = ["invoke", "habit", "noop"]
    records = [
        _make_record(i, action=actions[i % 3], arousal=(i + 1) / 100.0)
        for i in range(100)
    ]
    _write_partition(path, records)
    _backdate_partition(path, age_days, now)

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats = pruner.prune_partition(path)
    assert stats.rewrote is True
    # No tmp leftover.
    tmp_suffix = path.with_suffix(path.suffix + ".tmp")
    assert not tmp_suffix.exists()


def test_stats_records_pinned_by_kind(tmp_path: Path) -> None:
    """Pinned-by-kind histogram reflects the canonical kinds."""
    now = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    partition_date = now.date() - timedelta(days=120)
    path = _partition_path(tmp_path, "sprout", partition_date)
    records = [
        _make_record(0, action="invoke", arousal=0.0, pinned="agent_zero_golden"),
        _make_record(1, action="habit", arousal=0.0, pinned="agent_zero_golden"),
        _make_record(2, action="invoke", arousal=0.0, pinned="training_canon"),
        _make_record(3, action="habit", arousal=0.0, pinned="manual_review"),
        _make_record(4, action="noop", arousal=0.01),  # non-pinned, drops at 90+
    ]
    _write_partition(path, records)
    _backdate_partition(path, 120, now)

    pruner = RouterDatasetPruner(clock=_fixed_clock(now))
    stats = pruner.prune_partition(path)
    assert stats.pinned_preserved == 4
    assert stats.pinned_by_kind.get("agent_zero_golden") == 2
    assert stats.pinned_by_kind.get("training_canon") == 1
    assert stats.pinned_by_kind.get("manual_review") == 1
    assert stats.records_after == 4  # only the 4 pinned survive
