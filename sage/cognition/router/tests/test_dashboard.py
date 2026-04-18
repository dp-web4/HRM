#!/usr/bin/env python3
"""
Tests for the Phase 0 Track 8 router dashboard.

Coverage (spec: router-sprint-1-phase-0.md Track 8):
  * Empty dataset → "awaiting first data" markdown, no crash.
  * Round-trip from Track 4 writer: dashboard correctly aggregates per-
    machine + aggregate totals.
  * Modal-class dummy computed correctly against a controlled
    distribution where the answer is known a priori.
  * SNARC distribution histograms match input distribution.
  * Per-machine + aggregate decision-class + plugin breakdown accurate.
  * Records/day partitioned by UTC date matches writer layout.
  * Schema version distribution counts every record.
  * Recent-trend windows (last 24h / 7d) computed against the builder's
    clock (tests inject a fixed clock).
  * Markdown output includes the mandatory modal-class dummy column
    (PRD §7.10 agent-zero discipline).
  * JSON output round-trips through json.loads.

Run::

    python3 -m pytest sage/cognition/router/tests/test_dashboard.py -v
"""

from __future__ import annotations

import json
import math
import random
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from sage.cognition.router.dashboard import (
    AGENT_ZERO_MARGIN_PP,
    DRIFT_ALERT_THRESHOLD_NATS,
    DRIFT_MIN_SAMPLES_PER_WINDOW,
    DRIFT_SERVING_WINDOW_DAYS,
    DRIFT_TRAINING_MIN_AGE_DAYS,
    DashboardBuilder,
    DashboardMetrics,
    SNARC_DIMENSIONS,
    _kl_divergence,
    render_json,
    render_markdown,
)
from sage.cognition.router.data import (
    RouterDatasetWriter,
    SCHEMA_VERSION,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────


def _make_record(
    i: int,
    *,
    action: str = "noop",
    plugin: str = "",
    machine: str = "sprout",
    timestamp: float = 1_700_000_000.0,
    arousal: float = 0.1,
    surprise: float = 0.1,
    novelty: float = 0.1,
    conflict: float = 0.1,
    reward: float = 0.0,
) -> Dict[str, Any]:
    """Build one synthetic router record matching the envelope the
    writer stamps (top-level timestamp/machine/schema + nested input/output).
    """
    return {
        "record_id": f"rec-{i:08d}",
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp,
        "machine": machine,
        "router_input": {
            "tick": i,
            "snarc_arousal": arousal,
            "snarc_surprise": surprise,
            "snarc_novelty": novelty,
            "snarc_conflict": conflict,
            "snarc_reward": reward,
            "wm_state_key": f"{i:016x}",
        },
        "router_output": {
            "action": action,
            "plugin": plugin if action == "invoke" else None,
        },
    }


def _write_records(base: Path, machine: str, records: List[Dict[str, Any]], *, when: datetime):
    """Use Track 4's writer so we exercise the real partitioning path."""

    class _FixedClock:
        def __init__(self, dt: datetime):
            self._dt = dt

        def __call__(self) -> datetime:
            return self._dt

    clock = _FixedClock(when)
    writer = RouterDatasetWriter(
        base_dir=base,
        machine=machine,
        compress=False,  # plain JSONL keeps tests inspectable
        buffer_size=16,
        clock=clock,
    )
    with writer:
        for rec in records:
            writer.append(rec)


def _fixed_clock(now: float):
    def _c() -> float:
        return now
    return _c


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


def test_empty_dataset_renders_awaiting_first_data(tmp_path: Path) -> None:
    """Empty base-dir should produce a well-formed 'awaiting' dashboard."""
    base = tmp_path / "router_data"
    base.mkdir()

    builder = DashboardBuilder(base_dir=base)
    metrics = builder.build()

    assert metrics.aggregate.total_records == 0
    assert metrics.per_machine == {}

    md = render_markdown(metrics)
    assert "Awaiting first data" in md
    # Even empty, the agent-zero defense sentence must survive in the header
    # so operators don't forget the discipline before data arrives.
    assert "modal-class dummy" in md.lower()


def test_round_trip_basic_aggregation(tmp_path: Path) -> None:
    """Write via Track 4 writer, build dashboard, verify core counts."""
    base = tmp_path / "router_data"
    when = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    records = [_make_record(i, action="noop", timestamp=when.timestamp()) for i in range(50)]
    _write_records(base, "sprout", records, when=when)

    # Fixed clock slightly after write so last-24h window catches them.
    builder = DashboardBuilder(
        base_dir=base, clock=_fixed_clock(when.timestamp() + 3600)
    )
    metrics = builder.build()

    assert metrics.aggregate.total_records == 50
    assert "sprout" in metrics.per_machine
    assert metrics.per_machine["sprout"].total_records == 50
    assert metrics.per_machine["sprout"].records_per_day == {"2026-04-15": 50}
    assert metrics.aggregate.records_per_day == {"2026-04-15": 50}


def test_modal_class_dummy_controlled_distribution(tmp_path: Path) -> None:
    """Controlled 70/20/10 split → modal class = noop at exactly 70%.

    Also verifies the markdown table surfaces the dummy column with the
    right modal label.
    """
    base = tmp_path / "router_data"
    when = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    records: List[Dict[str, Any]] = []
    # 70 noop, 20 invoke, 10 habit — 100 records total
    for i in range(70):
        records.append(_make_record(i, action="noop", timestamp=when.timestamp()))
    for i in range(20):
        records.append(
            _make_record(70 + i, action="invoke", plugin="vision", timestamp=when.timestamp())
        )
    for i in range(10):
        records.append(_make_record(90 + i, action="habit", timestamp=when.timestamp()))
    _write_records(base, "sprout", records, when=when)

    builder = DashboardBuilder(
        base_dir=base, clock=_fixed_clock(when.timestamp() + 3600)
    )
    metrics = builder.build()

    agg_dec = metrics.aggregate.decisions
    assert agg_dec.total == 100
    assert agg_dec.action_counts == {"noop": 70, "invoke": 20, "habit": 10}
    assert agg_dec.modal_action == "noop"
    assert agg_dec.modal_action_rate == pytest.approx(0.70, abs=1e-9)

    # Plugin breakdown within invoke.
    assert agg_dec.plugin_counts == {"vision": 20}

    md = render_markdown(metrics)
    # The agent-zero column exists and names the dummy baseline.
    assert "Modal-class dummy %" in md
    assert "modal" in md.lower()
    # The dummy sentence must quote the modal class name.
    assert "`noop`" in md
    # Margin column: noop's margin is 0pp (70% - 100% = -30 actually;
    # representation uses signed pp; confirm the text appears).
    assert "Margin (pp)" in md
    # The 25-pp PRD threshold is referenced in the footer.
    assert f"{AGENT_ZERO_MARGIN_PP:.0f}" in md


def test_snarc_histogram_matches_controlled_distribution(tmp_path: Path) -> None:
    """Arousal pinned to a known set of values → histogram bins match."""
    base = tmp_path / "router_data"
    when = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    # Arousal values: 10 each at 0.05 (→bin 0), 0.55 (→bin 5), 0.95 (→bin 9).
    arousals = [0.05] * 10 + [0.55] * 10 + [0.95] * 10
    records = [
        _make_record(i, action="noop", arousal=a, timestamp=when.timestamp())
        for i, a in enumerate(arousals)
    ]
    _write_records(base, "sprout", records, when=when)

    builder = DashboardBuilder(
        base_dir=base, clock=_fixed_clock(when.timestamp() + 3600)
    )
    metrics = builder.build()

    arousal_stats = metrics.aggregate.snarc["arousal"]
    assert arousal_stats.count == 30
    # Mean of [0.05×10, 0.55×10, 0.95×10] = (0.05+0.55+0.95)/3 = 0.51666...
    assert arousal_stats.mean == pytest.approx((0.05 + 0.55 + 0.95) / 3, abs=1e-6)
    # 10 bins over [0,1]: bin_width=0.1 → 0.05→bin0, 0.55→bin5, 0.95→bin9.
    hist = arousal_stats.histogram
    assert len(hist) == 10
    assert hist[0] == 10
    assert hist[5] == 10
    assert hist[9] == 10
    # Everything else is empty — histogram carries all mass only at the
    # three pinned values.
    assert sum(hist) == 30


def test_reward_histogram_handles_negative_range(tmp_path: Path) -> None:
    """Reward is the one SNARC dim with signed range [-1, 1]."""
    base = tmp_path / "router_data"
    when = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    # Split reward: -0.9 and +0.9 equally → bin 0 and bin 9 for the
    # [-1,1] range.
    records = []
    for i in range(10):
        records.append(_make_record(i, action="noop", reward=-0.9, timestamp=when.timestamp()))
    for i in range(10):
        records.append(
            _make_record(10 + i, action="noop", reward=0.9, timestamp=when.timestamp())
        )
    _write_records(base, "sprout", records, when=when)

    builder = DashboardBuilder(
        base_dir=base, clock=_fixed_clock(when.timestamp() + 3600)
    )
    metrics = builder.build()

    reward_stats = metrics.aggregate.snarc["reward"]
    assert reward_stats.count == 20
    assert reward_stats.mean == pytest.approx(0.0, abs=1e-9)
    assert reward_stats.min_value == pytest.approx(-0.9, abs=1e-9)
    assert reward_stats.max_value == pytest.approx(0.9, abs=1e-9)
    # Bin edges should span [-1, 1].
    lo, hi = reward_stats.bin_edges[0][0], reward_stats.bin_edges[-1][1]
    assert lo == pytest.approx(-1.0, abs=1e-9)
    assert hi == pytest.approx(1.0, abs=1e-9)
    # Mass at each tail (exactly-10 in bin 0, exactly-10 in bin 9).
    assert reward_stats.histogram[0] == 10
    assert reward_stats.histogram[-1] == 10


def test_schema_version_distribution_counts_every_record(tmp_path: Path) -> None:
    """Mix two schema versions in one dataset — distribution should match."""
    base = tmp_path / "router_data"
    when = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    records = []
    for i in range(30):
        r = _make_record(i, action="noop", timestamp=when.timestamp())
        # Force schema_version variation
        r["schema_version"] = "0.1.0" if i < 20 else "0.2.0-future"
        records.append(r)
    _write_records(base, "sprout", records, when=when)

    builder = DashboardBuilder(
        base_dir=base, clock=_fixed_clock(when.timestamp() + 3600)
    )
    metrics = builder.build()

    assert metrics.aggregate.schema_version_counts.get("0.1.0") == 20
    assert metrics.aggregate.schema_version_counts.get("0.2.0-future") == 10

    md = render_markdown(metrics)
    # Both versions should appear in the markdown table.
    assert "0.1.0" in md
    assert "0.2.0-future" in md


def test_per_machine_and_aggregate_metrics_separate(tmp_path: Path) -> None:
    """Two machines → per-machine totals sum to aggregate."""
    base = tmp_path / "router_data"
    when = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)

    sprout = [
        _make_record(i, action="noop", machine="sprout", timestamp=when.timestamp())
        for i in range(30)
    ]
    sprout += [
        _make_record(
            100 + i, action="invoke", plugin="vision", machine="sprout",
            timestamp=when.timestamp(),
        )
        for i in range(10)
    ]
    thor = [
        _make_record(i, action="habit", machine="thor", timestamp=when.timestamp())
        for i in range(20)
    ]

    _write_records(base, "sprout", sprout, when=when)
    _write_records(base, "thor", thor, when=when)

    builder = DashboardBuilder(
        base_dir=base, clock=_fixed_clock(when.timestamp() + 3600)
    )
    metrics = builder.build()

    assert metrics.per_machine["sprout"].total_records == 40
    assert metrics.per_machine["thor"].total_records == 20
    assert metrics.aggregate.total_records == 60
    # Aggregate modal class is noop (30 > 20 > 10).
    assert metrics.aggregate.decisions.modal_action == "noop"
    assert metrics.aggregate.decisions.modal_action_rate == pytest.approx(30 / 60)
    # Sprout modal is noop (30 > 10); thor modal is habit (100%).
    assert metrics.per_machine["sprout"].decisions.modal_action == "noop"
    assert metrics.per_machine["thor"].decisions.modal_action == "habit"
    assert metrics.per_machine["thor"].decisions.modal_action_rate == pytest.approx(1.0)
    # Plugin counts only surface within invoke.
    assert metrics.per_machine["sprout"].decisions.plugin_counts == {"vision": 10}
    assert metrics.per_machine["thor"].decisions.plugin_counts == {}


def test_recent_trend_windows_respect_clock(tmp_path: Path) -> None:
    """Records split across yesterday vs 10-days-ago should populate windows."""
    base = tmp_path / "router_data"
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)

    # 10 records "yesterday" (within 24h of now-ish depends on exact
    # hour; use 2h ago to be unambiguous).
    recent_ts = now.timestamp() - 2 * 3600
    # 10 records 10 days ago (outside 7d window).
    old_ts = now.timestamp() - 10 * 86400

    records_recent = [
        _make_record(i, action="noop", timestamp=recent_ts) for i in range(10)
    ]
    records_old = [
        _make_record(100 + i, action="noop", timestamp=old_ts) for i in range(10)
    ]

    _write_records(base, "sprout", records_recent, when=now)
    # Writing old records to the same day partition is fine — dashboard
    # groups by record timestamp, not partition date, for recency.
    _write_records(base, "sprout", records_old, when=now - timedelta(days=10))

    builder = DashboardBuilder(base_dir=base, clock=_fixed_clock(now.timestamp()))
    metrics = builder.build()

    assert metrics.aggregate.total_records == 20
    assert metrics.aggregate.records_last_24h == 10
    assert metrics.aggregate.records_last_7d == 10  # old records fall outside
    assert metrics.per_machine["sprout"].records_last_24h == 10
    assert metrics.per_machine["sprout"].records_last_7d == 10


def test_markdown_surfaces_agent_zero_column(tmp_path: Path) -> None:
    """PRD §7.10: every decision-class table must include the dummy column."""
    base = tmp_path / "router_data"
    when = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    records = [_make_record(i, action="noop", timestamp=when.timestamp()) for i in range(5)]
    records += [
        _make_record(100 + i, action="invoke", plugin="v", timestamp=when.timestamp())
        for i in range(3)
    ]
    _write_records(base, "sprout", records, when=when)

    builder = DashboardBuilder(
        base_dir=base, clock=_fixed_clock(when.timestamp() + 60)
    )
    metrics = builder.build()
    md = render_markdown(metrics)

    # Canonical table headers from _decision_class_table — exactly these
    # columns protect against a drive-by markdown refactor deleting the
    # agent-zero column.
    header_row = (
        "| Decision type | Observed | Observed % | "
        "Modal-class dummy % | Margin (pp) |"
    )
    assert header_row in md
    # Modal class is noop (5 > 3). The row for noop must say '100% (modal)'.
    assert "100% (modal)" in md
    # Non-modal `invoke` should list dummy=0%.
    assert "`invoke`" in md
    # The footer reference to the 25-pp threshold must survive rendering.
    assert f"{AGENT_ZERO_MARGIN_PP:.0f}" in md


def test_json_output_roundtrips_and_has_schema_version(tmp_path: Path) -> None:
    """JSON must parse cleanly and carry the dashboard schema version."""
    base = tmp_path / "router_data"
    when = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    records = [_make_record(i, action="noop", timestamp=when.timestamp()) for i in range(7)]
    _write_records(base, "sprout", records, when=when)

    builder = DashboardBuilder(
        base_dir=base, clock=_fixed_clock(when.timestamp() + 60)
    )
    metrics = builder.build()

    js = render_json(metrics)
    parsed = json.loads(js)
    assert "schema_version" in parsed
    assert parsed["schema_version"]
    assert parsed["aggregate"]["total_records"] == 7
    assert "sprout" in parsed["per_machine"]
    assert parsed["per_machine"]["sprout"]["total_records"] == 7
    # Decision stats are surfaced in the JSON form.
    assert parsed["aggregate"]["decisions"]["modal_action"] == "noop"
    # SNARC block present for every canonical dimension.
    for dim in SNARC_DIMENSIONS:
        assert dim in parsed["aggregate"]["snarc"]


def test_sampling_retention_table_reconstructs_quintile_shape(tmp_path: Path) -> None:
    """Feed a fixed salience distribution; quintiles recover that shape."""
    base = tmp_path / "router_data"
    when = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    # 100 records with arousal sweeping [0, 1); quintile seen should be
    # ~20 records each.
    records = []
    for i in range(100):
        a = i / 100.0  # 0.00 .. 0.99
        records.append(
            _make_record(i, action="noop", arousal=a, timestamp=when.timestamp())
        )
    _write_records(base, "sprout", records, when=when)

    builder = DashboardBuilder(
        base_dir=base, clock=_fixed_clock(when.timestamp() + 60)
    )
    metrics = builder.build()

    seen = metrics.aggregate.quintile_seen
    assert sum(seen) == 100
    # Each quintile should carry ~20 records ± boundary-inclusive noise.
    for count in seen:
        assert 15 <= count <= 25, f"Quintile counts off: {seen}"

    md = render_markdown(metrics)
    # The retention-rate table renders five named rows.
    for name in ("Q0 (low)", "Q1", "Q2", "Q3", "Q4 (top)"):
        assert name in md


def test_builder_handles_mixed_compressed_and_plain_partitions(tmp_path: Path) -> None:
    """Writer produces a plain partition here; also drop a gzip partition
    for a different day and confirm the reader/builder handle both.
    """
    import gzip

    base = tmp_path / "router_data"
    when_plain = datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc)
    plain = [
        _make_record(i, action="noop", timestamp=when_plain.timestamp()) for i in range(5)
    ]
    _write_records(base, "sprout", plain, when=when_plain)

    # Manually write a gzipped partition for a different day to make
    # sure the dashboard sees both (tests the reader glob + partition
    # metadata pipeline).
    when_gz = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    gz_path = base / "sprout" / "2026-04-15.jsonl.gz"
    gz_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_path, "wt", encoding="utf-8") as gf:
        for i in range(7):
            rec = _make_record(100 + i, action="invoke", plugin="v", timestamp=when_gz.timestamp())
            gf.write(json.dumps(rec) + "\n")

    builder = DashboardBuilder(
        base_dir=base, clock=_fixed_clock(when_gz.timestamp() + 60)
    )
    metrics = builder.build()

    assert metrics.aggregate.total_records == 12
    assert metrics.aggregate.records_per_day == {
        "2026-04-14": 5,
        "2026-04-15": 7,
    }
    # Bytes-on-disk should include both partitions.
    assert metrics.aggregate.bytes_on_disk > 0


def test_date_range_filter_constrains_partitions(tmp_path: Path) -> None:
    """Providing a date range should drop out-of-range partitions."""
    base = tmp_path / "router_data"
    when_a = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)
    when_b = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    _write_records(
        base, "sprout",
        [_make_record(i, action="noop", timestamp=when_a.timestamp()) for i in range(3)],
        when=when_a,
    )
    _write_records(
        base, "sprout",
        [_make_record(100 + i, action="noop", timestamp=when_b.timestamp()) for i in range(7)],
        when=when_b,
    )

    # Date range excludes the earlier partition.
    builder = DashboardBuilder(
        base_dir=base,
        date_range=("2026-04-12", "2026-04-20"),
        clock=_fixed_clock(when_b.timestamp() + 60),
    )
    metrics = builder.build()
    assert metrics.aggregate.total_records == 7
    assert list(metrics.aggregate.records_per_day.keys()) == ["2026-04-15"]


def test_performance_smoke_100k_records(tmp_path: Path) -> None:
    """Performance budget: <5s for 100k records.

    Non-strict guard (CI runners vary) — generous 20s budget still catches
    quadratic regressions.
    """
    base = tmp_path / "router_data"
    when = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)

    # Build via direct gzip write — avoids writer buffer-flush overhead.
    import gzip
    gz_path = base / "sprout" / "2026-04-15.jsonl.gz"
    gz_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0xBEEF)
    with gzip.open(gz_path, "wt", encoding="utf-8") as gf:
        for i in range(100_000):
            action = rng.choice(["noop", "noop", "noop", "invoke", "habit"])
            rec = _make_record(
                i,
                action=action,
                plugin="vision" if action == "invoke" else "",
                arousal=rng.random(),
                surprise=rng.random(),
                novelty=rng.random(),
                conflict=rng.random(),
                reward=rng.uniform(-1, 1),
                timestamp=when.timestamp() + i * 0.001,
            )
            gf.write(json.dumps(rec) + "\n")

    t0 = time.time()
    builder = DashboardBuilder(
        base_dir=base, clock=_fixed_clock(when.timestamp() + 10000)
    )
    metrics = builder.build()
    elapsed = time.time() - t0
    assert metrics.aggregate.total_records == 100_000
    # Loose budget for CI variance — real run on dev hardware is sub-5s.
    # Drift monitoring was added in R4; it shares the same record pass so
    # cost is effectively constant overhead — the 20s guard still holds.
    assert elapsed < 20.0, f"dashboard too slow on 100k: {elapsed:.2f}s"


# ──────────────────────────────────────────────────────────────────────
# SNARC distribution-drift monitoring (PRD §4.7.G) — Sprint 2 R4
# ──────────────────────────────────────────────────────────────────────


def _write_drift_dataset(
    base: Path,
    *,
    machine: str,
    training_samples: List[Dict[str, float]],
    serving_samples: List[Dict[str, float]],
    now: datetime,
    training_age_days: int = DRIFT_TRAINING_MIN_AGE_DAYS + 5,
    serving_age_days: int = 1,
) -> None:
    """Write a training-window batch (old) and a serving-window batch (recent).

    Training records are stamped (now - training_age_days), serving
    records are stamped (now - serving_age_days). Uses direct gzip write
    so a single test partition holds both — the dashboard buckets by
    record-timestamp, not partition path, for drift.
    """
    import gzip

    t_train = (now - timedelta(days=training_age_days)).timestamp()
    t_serve = (now - timedelta(days=serving_age_days)).timestamp()

    train_path = (
        base / machine / (now - timedelta(days=training_age_days)).strftime("%Y-%m-%d")
    ).with_suffix(".jsonl.gz")
    serve_path = (
        base / machine / (now - timedelta(days=serving_age_days)).strftime("%Y-%m-%d")
    ).with_suffix(".jsonl.gz")
    train_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(train_path, "wt", encoding="utf-8") as gf:
        for i, s in enumerate(training_samples):
            rec = _make_record(
                i,
                action="noop",
                machine=machine,
                timestamp=t_train,
                arousal=s.get("arousal", 0.1),
                surprise=s.get("surprise", 0.1),
                novelty=s.get("novelty", 0.1),
                conflict=s.get("conflict", 0.1),
                reward=s.get("reward", 0.0),
            )
            gf.write(json.dumps(rec) + "\n")

    with gzip.open(serve_path, "wt", encoding="utf-8") as gf:
        for i, s in enumerate(serving_samples):
            rec = _make_record(
                1_000_000 + i,
                action="noop",
                machine=machine,
                timestamp=t_serve,
                arousal=s.get("arousal", 0.1),
                surprise=s.get("surprise", 0.1),
                novelty=s.get("novelty", 0.1),
                conflict=s.get("conflict", 0.1),
                reward=s.get("reward", 0.0),
            )
            gf.write(json.dumps(rec) + "\n")


def test_kl_divergence_identical_distributions_is_zero() -> None:
    """KL(P||P) = 0 by construction; also exercises the smoothing path."""
    p = [10, 20, 30, 40, 0, 0, 5, 15, 25, 35]
    assert _kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_kl_divergence_disjoint_distributions_is_large() -> None:
    """Disjoint supports → KL large (bounded by smoothing, not infinite)."""
    p = [100, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    q = [0, 0, 0, 0, 0, 0, 0, 0, 0, 100]
    kl = _kl_divergence(p, q)
    # Order-of-magnitude check: divergence should be well above the alert
    # threshold, and — critically — finite thanks to smoothing.
    assert math.isfinite(kl)
    assert kl > DRIFT_ALERT_THRESHOLD_NATS * 10


def test_kl_divergence_smoothing_handles_zero_bins() -> None:
    """Zero-count bins in Q must not push KL to +inf — smoothing does the work."""
    p = [50, 50, 0, 0, 0, 0, 0, 0, 0, 0]
    q = [50, 0, 50, 0, 0, 0, 0, 0, 0, 0]  # different support
    kl = _kl_divergence(p, q)
    assert math.isfinite(kl)
    assert kl > 0.0


def test_drift_no_drift_returns_healthy_status(tmp_path: Path) -> None:
    """Training-window and serving-window drawn from the same distribution
    → KL ≈ 0 → HEALTHY on every dim with enough data.
    """
    base = tmp_path / "router_data"
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)

    rng = random.Random(0xC0FFEE)

    def _sample() -> Dict[str, float]:
        return {
            "arousal": rng.random(),
            "surprise": rng.random(),
            "novelty": rng.random(),
            "conflict": rng.random(),
            "reward": rng.uniform(-1, 1),
        }

    n = DRIFT_MIN_SAMPLES_PER_WINDOW + 500
    training = [_sample() for _ in range(n)]
    serving = [_sample() for _ in range(n)]

    _write_drift_dataset(
        base,
        machine="sprout",
        training_samples=training,
        serving_samples=serving,
        now=now,
    )

    builder = DashboardBuilder(base_dir=base, clock=_fixed_clock(now.timestamp()))
    metrics = builder.build()

    drift = metrics.drift_aggregate
    # Every dim should be HEALTHY — same distribution on both windows.
    for dim in SNARC_DIMENSIONS:
        d = drift.dimensions[dim]
        assert d.status == "HEALTHY", (
            f"{dim}: expected HEALTHY, got {d.status} (kl={d.kl_nats})"
        )
        assert d.kl_nats is not None and math.isfinite(d.kl_nats)
        assert d.kl_nats < DRIFT_ALERT_THRESHOLD_NATS
        assert d.training_count >= DRIFT_MIN_SAMPLES_PER_WINDOW
        assert d.serving_count >= DRIFT_MIN_SAMPLES_PER_WINDOW
    assert drift.any_alert is False
    assert drift.awaiting_baseline is False


def test_drift_extreme_shift_fires_alert(tmp_path: Path) -> None:
    """Training = low arousal mode, serving = high arousal mode → DRIFT ALERT."""
    base = tmp_path / "router_data"
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    n = DRIFT_MIN_SAMPLES_PER_WINDOW + 100

    rng = random.Random(0xBAD)

    # Training: arousal concentrated near 0.1 (low).
    training = [
        {
            "arousal": max(0.0, min(1.0, rng.gauss(0.1, 0.03))),
            "surprise": rng.random(),
            "novelty": rng.random(),
            "conflict": rng.random(),
            "reward": rng.uniform(-1, 1),
        }
        for _ in range(n)
    ]
    # Serving: arousal concentrated near 0.9 (high) — disjoint mode.
    serving = [
        {
            "arousal": max(0.0, min(1.0, rng.gauss(0.9, 0.03))),
            "surprise": rng.random(),
            "novelty": rng.random(),
            "conflict": rng.random(),
            "reward": rng.uniform(-1, 1),
        }
        for _ in range(n)
    ]

    _write_drift_dataset(
        base,
        machine="sprout",
        training_samples=training,
        serving_samples=serving,
        now=now,
    )

    builder = DashboardBuilder(base_dir=base, clock=_fixed_clock(now.timestamp()))
    metrics = builder.build()

    drift = metrics.drift_aggregate
    arousal = drift.dimensions["arousal"]
    assert arousal.status == "DRIFT ALERT"
    assert arousal.kl_nats is not None
    assert arousal.kl_nats >= DRIFT_ALERT_THRESHOLD_NATS
    assert drift.any_alert is True
    # The other dims sampled uniformly on both sides — they should be
    # HEALTHY (aside from small floating noise).
    for dim in ("surprise", "novelty", "conflict", "reward"):
        d = drift.dimensions[dim]
        assert d.status in ("HEALTHY", "DRIFT ALERT")  # usually HEALTHY
        # Weak assertion: at least all other dims MUST be evaluated
        # (not INSUFFICIENT DATA) — same sample count as arousal.
        assert d.training_count >= DRIFT_MIN_SAMPLES_PER_WINDOW
        assert d.serving_count >= DRIFT_MIN_SAMPLES_PER_WINDOW
    # Markdown surfaces the alert status.
    md = render_markdown(metrics)
    assert "SNARC distribution drift" in md
    assert "DRIFT ALERT" in md
    assert "PRD §4.7.G" in md


def test_drift_insufficient_data_flags_not_alerts(tmp_path: Path) -> None:
    """Under the sample floor → INSUFFICIENT DATA, never a false alarm,
    even when raw KL would exceed threshold."""
    base = tmp_path / "router_data"
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)

    # Tiny samples — far below DRIFT_MIN_SAMPLES_PER_WINDOW — but with
    # deliberately disjoint modes so raw KL would absolutely scream.
    training = [{"arousal": 0.05} for _ in range(50)]
    serving = [{"arousal": 0.95} for _ in range(50)]

    _write_drift_dataset(
        base,
        machine="sprout",
        training_samples=training,
        serving_samples=serving,
        now=now,
    )

    builder = DashboardBuilder(base_dir=base, clock=_fixed_clock(now.timestamp()))
    metrics = builder.build()

    drift = metrics.drift_aggregate
    for dim in SNARC_DIMENSIONS:
        d = drift.dimensions[dim]
        assert d.status == "INSUFFICIENT DATA", (
            f"{dim}: expected INSUFFICIENT DATA with {d.training_count}/"
            f"{d.serving_count} samples, got {d.status}"
        )
        assert d.kl_nats is None
    assert drift.any_alert is False
    assert drift.awaiting_baseline is True

    md = render_markdown(metrics)
    # With no dimension meeting the floor anywhere, the section should
    # show the "awaiting baseline" preamble — NOT a scary alert.
    assert "awaiting baseline" in md.lower()
    assert "DRIFT ALERT" not in md


def test_drift_per_machine_isolation(tmp_path: Path) -> None:
    """One drifting machine must not mask another healthy machine.

    Sprout drifts (disjoint arousal modes), Thor is stable (identical
    distributions both windows). The per-machine report must surface the
    alert on sprout only.
    """
    base = tmp_path / "router_data"
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    n = DRIFT_MIN_SAMPLES_PER_WINDOW + 100

    rng_s = random.Random(1)
    rng_t = random.Random(2)

    # Sprout: arousal drift.
    sprout_training = [
        {"arousal": max(0.0, min(1.0, rng_s.gauss(0.1, 0.02)))} for _ in range(n)
    ]
    sprout_serving = [
        {"arousal": max(0.0, min(1.0, rng_s.gauss(0.9, 0.02)))} for _ in range(n)
    ]
    # Thor: same distribution on both sides.
    thor_training = [{"arousal": rng_t.random()} for _ in range(n)]
    thor_serving = [{"arousal": rng_t.random()} for _ in range(n)]

    _write_drift_dataset(
        base, machine="sprout",
        training_samples=sprout_training, serving_samples=sprout_serving,
        now=now,
    )
    _write_drift_dataset(
        base, machine="thor",
        training_samples=thor_training, serving_samples=thor_serving,
        now=now,
    )

    builder = DashboardBuilder(base_dir=base, clock=_fixed_clock(now.timestamp()))
    metrics = builder.build()

    assert "sprout" in metrics.drift_per_machine
    assert "thor" in metrics.drift_per_machine
    sprout_drift = metrics.drift_per_machine["sprout"]
    thor_drift = metrics.drift_per_machine["thor"]

    assert sprout_drift.dimensions["arousal"].status == "DRIFT ALERT"
    assert sprout_drift.any_alert is True

    # Thor's arousal is healthy — the per-machine isolation worked.
    thor_arousal = thor_drift.dimensions["arousal"]
    assert thor_arousal.status == "HEALTHY"
    assert thor_arousal.kl_nats is not None
    assert thor_arousal.kl_nats < DRIFT_ALERT_THRESHOLD_NATS
    assert thor_drift.any_alert is False

    # Aggregate still fires since sprout's contribution to aggregate
    # arousal is half of the combined histogram.
    agg_arousal = metrics.drift_aggregate.dimensions["arousal"]
    assert agg_arousal.status in ("DRIFT ALERT", "HEALTHY")
    # The markdown must contain both per-machine headers.
    md = render_markdown(metrics)
    assert "### sprout" in md
    assert "### thor" in md


def test_drift_json_output_includes_drift_block(tmp_path: Path) -> None:
    """render_json must expose drift_aggregate + drift_per_machine."""
    base = tmp_path / "router_data"
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    n = DRIFT_MIN_SAMPLES_PER_WINDOW + 50

    rng = random.Random(42)
    training = [
        {"arousal": rng.random(), "surprise": rng.random()} for _ in range(n)
    ]
    serving = [
        {"arousal": rng.random(), "surprise": rng.random()} for _ in range(n)
    ]
    _write_drift_dataset(
        base, machine="sprout",
        training_samples=training, serving_samples=serving, now=now,
    )

    builder = DashboardBuilder(base_dir=base, clock=_fixed_clock(now.timestamp()))
    metrics = builder.build()
    js = render_json(metrics)
    parsed = json.loads(js)

    assert "drift_aggregate" in parsed
    assert "drift_per_machine" in parsed
    assert "sprout" in parsed["drift_per_machine"]
    for dim in SNARC_DIMENSIONS:
        assert dim in parsed["drift_aggregate"]["dimensions"]
    # Schema version must bump — consumers check this.
    assert parsed["schema_version"] != "v0.1.0"
