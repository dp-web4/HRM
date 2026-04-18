"""
Tests for the Phase 0 Track 4 router data pipeline.

Coverage:
  * Round-trip write/read, 10k records, gzip on and off.
  * Partition path layout (machine + UTC date).
  * Buffering + explicit flush + close.
  * Failure isolation: disk-full / read-only disk does not crash caller.
  * Reader tolerates corrupt trailing lines and missing fields.
  * SNARC sampler quintile distribution within ±2% of spec.
  * Schema version skew: hypothetical v0.2.0 still surfaces.

Runs with plain `pytest`; no torch, no network.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import stat
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from sage.cognition.router.data import (
    RouterDatasetReader,
    RouterDatasetWriter,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    SnarcStratifiedSampler,
    salience_score,
)
# TODO: replace _RouterRecordStub with sage.cognition.router.RouterRecord
# once Track 1 merges.
from sage.cognition.router.data._stub import _RouterRecordStub


# ── helpers ────────────────────────────────────────────────────────


def _make_record(i: int, machine: str = "testmachine") -> Dict[str, Any]:
    """Synthetic record matching the Track 4 payload envelope."""
    return {
        "record_id": f"rec-{i:08d}",
        "schema_version": SCHEMA_VERSION,
        "timestamp": 1_700_000_000.0 + i,
        "machine": machine,
        "payload": {
            "router_input": {
                "tick": i,
                "snarc": {
                    "surprise": (i % 100) / 100.0,
                    "novelty": ((i + 7) % 100) / 100.0,
                    "arousal": ((i + 13) % 100) / 100.0,
                    "reward": (((i + 31) % 200) - 100) / 100.0,
                    "conflict": ((i + 17) % 100) / 100.0,
                },
                "wm_state_key": f"{i:016x}",
            },
            "router_output": {
                "action": ["invoke", "habit", "noop"][i % 3],
                "confidence": (i % 100) / 100.0,
            },
        },
    }


# ── round-trip ─────────────────────────────────────────────────────


@pytest.mark.parametrize("compress", [False, True])
def test_round_trip_10k_records(tmp_path: Path, compress: bool) -> None:
    """Write 10k records, read them back, verify exact match."""
    base = tmp_path / "dataset"
    records = [_make_record(i) for i in range(10_000)]

    writer = RouterDatasetWriter(
        base_dir=base, machine="testmachine",
        compress=compress, buffer_size=256,
    )
    try:
        for rec in records:
            assert writer.append(rec) is True
    finally:
        writer.close()

    # Verify partition naming + file presence
    machine_dir = base / "testmachine"
    files = list(machine_dir.iterdir())
    assert len(files) == 1, f"expected 1 partition, found {files}"
    f = files[0]
    if compress:
        assert f.suffix == ".gz"
        assert f.name.endswith(".jsonl.gz")
    else:
        assert f.suffix == ".jsonl"

    reader = RouterDatasetReader(base_dir=base)
    read_back = list(reader.read_file(f))
    assert len(read_back) == 10_000
    # Field-by-field match — ensures payload dict preserved in full
    for orig, got in zip(records, read_back):
        assert got == orig

    stats = writer.get_stats()
    assert stats["records_written"] == 10_000
    assert stats["records_dropped"] == 0


# ── partitioning ───────────────────────────────────────────────────


def test_partition_path_uses_machine_and_utc_date(tmp_path: Path) -> None:
    base = tmp_path / "dataset"
    fixed = datetime(2026, 4, 17, 23, 59, 30, tzinfo=timezone.utc)

    writer = RouterDatasetWriter(
        base_dir=base, machine="sprout", compress=False, clock=lambda: fixed,
    )
    writer.append(_make_record(1, machine="sprout"))
    writer.flush()
    assert writer.current_path() == base / "sprout" / "2026-04-17.jsonl"
    writer.close()


def test_partition_rolls_over_on_utc_date_boundary(tmp_path: Path) -> None:
    base = tmp_path / "dataset"
    # Mutable clock so we can advance across midnight.
    now_box: List[datetime] = [
        datetime(2026, 4, 17, 23, 59, 59, tzinfo=timezone.utc)
    ]

    writer = RouterDatasetWriter(
        base_dir=base, machine="thor", compress=False,
        buffer_size=1, clock=lambda: now_box[0],
    )
    writer.append(_make_record(1, machine="thor"))  # lands in 2026-04-17
    now_box[0] = datetime(2026, 4, 18, 0, 0, 1, tzinfo=timezone.utc)
    writer.append(_make_record(2, machine="thor"))  # should roll to 2026-04-18
    writer.close()

    thor_dir = base / "thor"
    files = sorted(p.name for p in thor_dir.iterdir())
    assert files == ["2026-04-17.jsonl", "2026-04-18.jsonl"], files


def test_reader_glob_across_machines_and_dates(tmp_path: Path) -> None:
    base = tmp_path / "dataset"
    # Pretend three machines each wrote three consecutive days.
    machines = ["sprout", "thor", "legion"]
    days = [datetime(2026, 4, d, 12, 0, 0, tzinfo=timezone.utc)
            for d in (15, 16, 17)]

    total = 0
    for m in machines:
        for d in days:
            writer = RouterDatasetWriter(
                base_dir=base, machine=m,
                buffer_size=1, clock=lambda d=d: d,
            )
            writer.append(_make_record(total, machine=m))
            writer.close()
            total += 1

    reader = RouterDatasetReader(base_dir=base)

    # All machines, all dates → 9 partitions × 1 record = 9
    all_records = list(reader.read_partition())
    assert len(all_records) == 9

    # Filter by machine
    sprout_records = list(reader.read_partition(machine="sprout"))
    assert len(sprout_records) == 3
    assert all(r["machine"] == "sprout" for r in sprout_records)

    # Filter by date range (inclusive)
    mid_day_records = list(reader.read_partition(
        date_range=("2026-04-16", "2026-04-16"),
    ))
    assert len(mid_day_records) == 3  # three machines, one day each

    # Combined filter
    sprout_mid = list(reader.read_partition(
        machine="sprout", date_range=("2026-04-16", "2026-04-17"),
    ))
    assert len(sprout_mid) == 2


# ── failure isolation ─────────────────────────────────────────────


def test_writer_failure_isolated_when_open_fails(tmp_path, caplog) -> None:
    """Simulate 'disk full' at open time — caller must not see an exception."""
    base = tmp_path / "dataset"
    writer = RouterDatasetWriter(base_dir=base, machine="sprout", buffer_size=1)

    # Patch the writer to inject an OSError at partition-open time.
    def raise_disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    writer._ensure_partition = lambda: (_ for _ in ()).throw(  # type: ignore[assignment]
        OSError(28, "No space left on device")
    )

    # append() swallows the error and reports success-queued (the record
    # sat in the buffer; the flush attempt failed isolated). Importantly,
    # NO exception propagates to the caller.
    assert writer.append(_make_record(1)) is True
    writer.flush()   # still safe
    writer.close()   # still safe

    # Caller can inspect stats if they care.
    stats = writer.get_stats()
    assert stats["records_written"] == 1


def test_writer_append_after_close_drops_and_warns(tmp_path, caplog) -> None:
    import logging
    base = tmp_path / "dataset"
    writer = RouterDatasetWriter(base_dir=base, machine="sprout")
    writer.append(_make_record(1))
    writer.close()
    with caplog.at_level(logging.WARNING,
                        logger="sage.cognition.router.data.writer"):
        ok = writer.append(_make_record(2))
    assert ok is False
    assert writer.get_stats()["records_dropped"] == 1


def test_writer_rejects_non_serializable_record_cleanly(tmp_path) -> None:
    base = tmp_path / "dataset"
    writer = RouterDatasetWriter(base_dir=base, machine="sprout")

    class Opaque:
        pass

    # append() must swallow and count as dropped, not raise.
    ok = writer.append(Opaque())
    assert ok is False
    assert writer.get_stats()["records_dropped"] == 1
    writer.close()


# ── reader robustness ─────────────────────────────────────────────


def test_reader_handles_corrupt_trailing_line(tmp_path) -> None:
    """A partial last line (crash mid-write) must not lose earlier records."""
    base = tmp_path / "dataset"
    writer = RouterDatasetWriter(
        base_dir=base, machine="sprout", buffer_size=1,
    )
    for i in range(5):
        writer.append(_make_record(i))
    writer.close()

    # Append a corrupt half-written line directly to the file.
    partition = next((base / "sprout").iterdir())
    with open(partition, "a", encoding="utf-8") as f:
        f.write('{"record_id": "partial", "schema_vers')  # no newline, invalid JSON

    reader = RouterDatasetReader(base_dir=base)
    records = list(reader.read_file(partition))
    assert len(records) == 5, f"expected 5 valid records, got {len(records)}"


def test_reader_handles_missing_file(tmp_path, caplog) -> None:
    reader = RouterDatasetReader(base_dir=tmp_path)
    # No error, just empty iterator.
    assert list(reader.read_file(tmp_path / "nope.jsonl")) == []
    assert list(reader.read_partition(machine="ghost")) == []


# ── schema version skew ───────────────────────────────────────────


def test_reader_handles_hypothetical_future_schema(tmp_path, caplog) -> None:
    base = tmp_path / "dataset"
    writer = RouterDatasetWriter(
        base_dir=base, machine="sprout", buffer_size=1,
    )
    # One v0.1.0 record.
    writer.append(_make_record(1))
    # One v0.2.0 record with an extra field the reader doesn't know.
    future = _make_record(2)
    future["schema_version"] = "0.2.0"
    future["payload"]["cartridge_recall_embedding"] = [0.1] * 768  # new field
    writer.append(future)
    writer.close()

    # Ensure assumption: our reader's SUPPORTED set is v0.1.0 only, so v0.2.0
    # must be "unknown" — this is what we want to test.
    assert "0.2.0" not in SUPPORTED_SCHEMA_VERSIONS

    reader = RouterDatasetReader(base_dir=base)
    partition = next((base / "sprout").iterdir())
    records = list(reader.read_file(partition))
    assert len(records) == 2
    assert records[0]["schema_version"] == "0.1.0"
    assert records[1]["schema_version"] == "0.2.0"
    # Extra field surfaces untouched — consumer decides whether to care.
    assert len(records[1]["payload"]["cartridge_recall_embedding"]) == 768


def test_reader_fills_missing_schema_version(tmp_path) -> None:
    base = tmp_path / "dataset" / "sprout"
    base.mkdir(parents=True)
    partition = base / "2026-04-17.jsonl"
    # Write a record with schema_version missing entirely.
    with open(partition, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "record_id": "legacy-1",
            "timestamp": 1.0,
            "machine": "sprout",
            "payload": {},
        }) + "\n")

    reader = RouterDatasetReader(base_dir=tmp_path / "dataset")
    records = list(reader.read_file(partition))
    assert len(records) == 1
    assert records[0]["schema_version"] == "0.1.0"


# ── stub integration ──────────────────────────────────────────────


def test_writer_accepts_record_stub(tmp_path) -> None:
    """Ensure the _RouterRecordStub shape round-trips through writer+reader.

    TODO: replace _RouterRecordStub with sage.cognition.router.RouterRecord
    once Track 1 merges.
    """
    base = tmp_path / "dataset"
    stub = _RouterRecordStub(
        record_id="stub-1",
        schema_version=SCHEMA_VERSION,
        timestamp=123.45,
        machine="testmachine",
        payload={"foo": "bar"},
    )
    writer = RouterDatasetWriter(base_dir=base, machine="testmachine")
    writer.append(stub)
    writer.close()

    reader = RouterDatasetReader(base_dir=base)
    partition = next((base / "testmachine").iterdir())
    records = list(reader.read_file(partition))
    assert len(records) == 1
    assert records[0]["record_id"] == "stub-1"
    assert records[0]["payload"] == {"foo": "bar"}


# ── salience / sampler ────────────────────────────────────────────


def test_salience_score_formula() -> None:
    # max(arousal, conflict, |reward|)
    assert salience_score({"arousal": 0.8, "conflict": 0.2, "reward": 0.1}) == 0.8
    assert salience_score({"arousal": 0.1, "conflict": 0.9, "reward": 0.1}) == 0.9
    # Negative reward: abs-value wins.
    assert salience_score({"arousal": 0.1, "conflict": 0.1, "reward": -0.7}) == 0.7
    # Missing keys → treated as zero.
    assert salience_score({}) == 0.0
    assert salience_score({"arousal": 0.5}) == 0.5


def test_sampler_determinism_with_seed() -> None:
    snarcs = [{"arousal": random.Random(i).random(),
               "conflict": random.Random(i + 1).random(),
               "reward": random.Random(i + 2).random()}
              for i in range(200)]

    s1 = SnarcStratifiedSampler(seed=42, warmup=10)
    s2 = SnarcStratifiedSampler(seed=42, warmup=10)
    r1 = [s1.should_keep(x) for x in snarcs]
    r2 = [s2.should_keep(x) for x in snarcs]
    assert r1 == r2


def test_sampler_distribution_matches_spec_within_tolerance() -> None:
    """Synthesize 10k ticks with uniform salience; verify per-quintile
    retention is within ±2% of the PRD §4.7.D spec.

    Uniform input → each quintile gets ~2000 samples; expected kept counts:
      q0: 2000 × 0.05 =  100
      q1: 2000 × 0.20 =  400
      q2: 2000 × 0.20 =  400
      q3: 2000 × 0.20 =  400
      q4: 2000 × 1.00 = 2000
    """
    sampler = SnarcStratifiedSampler(
        seed=20260417, warmup=500, window_size=2000,
    )
    rng = random.Random(20260417)
    n = 10_000
    for _ in range(n):
        # Uniform arousal in [0, 1], zero everything else → uniform salience.
        sampler.should_keep({
            "arousal": rng.random(), "conflict": 0.0, "reward": 0.0,
        })

    rates = sampler.stats.per_quintile_keep_rates()
    targets = [0.05, 0.20, 0.20, 0.20, 1.00]
    tolerance = 0.02

    # Top quintile has deterministic keep_rate=1.0, but the warmup period
    # kept everything across all quintiles. We expect CONVERGENCE to spec
    # once warmup is past, so we allow the small warmup-inflation on the
    # lower quintiles.
    #
    # With warmup=500 and 10k samples, the inflation is at most
    # 500/2000 ≈ 25% on any given quintile's count, but AT MOST
    # 100/2000 = 5pp on keep-rate because the warmup only adds ~100
    # warmup-kept samples to any one quintile. We test with 2pp tolerance
    # on non-warmup portion.
    #
    # Instead: reset stats after warmup and re-run.
    sampler.reset_stats()
    for _ in range(n):
        sampler.should_keep({
            "arousal": rng.random(), "conflict": 0.0, "reward": 0.0,
        })

    rates = sampler.stats.per_quintile_keep_rates()
    for q, (rate, target) in enumerate(zip(rates, targets)):
        assert abs(rate - target) <= tolerance, (
            f"Quintile {q}: keep rate {rate:.3f} outside ±{tolerance} of target {target}"
        )


def test_sampler_top_quintile_fully_captured() -> None:
    """Sanity: top-SNARC records MUST all be retained (spec normative)."""
    sampler = SnarcStratifiedSampler(seed=1, warmup=50, window_size=500)
    # Warm the window with uniform-salience noise.
    rng = random.Random(1)
    for _ in range(600):
        sampler.should_keep({"arousal": rng.random(), "conflict": 0.0, "reward": 0.0})

    sampler.reset_stats()
    # Now push 100 high-salience records (should all be kept).
    high_kept = 0
    for _ in range(100):
        if sampler.should_keep({"arousal": 0.99, "conflict": 0.99, "reward": 0.0}):
            high_kept += 1
    # Quintile 4 keep-rate is 1.0; all high-salience must pass.
    assert high_kept == 100, f"expected all 100 high-salience kept, got {high_kept}"


def test_sampler_stats_to_dict() -> None:
    sampler = SnarcStratifiedSampler(seed=1, warmup=0)
    for _ in range(10):
        sampler.should_keep({"arousal": 0.5, "conflict": 0.0, "reward": 0.0})
    d = sampler.stats.to_dict()
    assert "seen" in d
    assert "kept" in d
    assert "per_quintile_seen" in d
    assert sum(d["per_quintile_seen"]) == 10


# ── writer/reader ↔ sampler integration ───────────────────────────


def test_sampler_gates_writer_end_to_end(tmp_path) -> None:
    """Happy path: sampler decides which records get written, reader
    recovers exactly those."""
    base = tmp_path / "dataset"
    sampler = SnarcStratifiedSampler(seed=7, warmup=50, window_size=500)
    writer = RouterDatasetWriter(base_dir=base, machine="sprout", buffer_size=32)

    kept_ids = []
    for i in range(1000):
        snarc = {
            "arousal": ((i * 37) % 1000) / 1000.0,
            "conflict": 0.0, "reward": 0.0,
        }
        if sampler.should_keep(snarc):
            rec = _make_record(i, machine="sprout")
            writer.append(rec)
            kept_ids.append(rec["record_id"])
    writer.close()

    reader = RouterDatasetReader(base_dir=base)
    recovered = [r["record_id"] for r in reader.read_partition(machine="sprout")]
    assert recovered == kept_ids


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
