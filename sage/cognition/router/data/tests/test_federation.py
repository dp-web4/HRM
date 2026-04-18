"""
Tests for Router Sprint 2 R3 — federation aggregator.

Covers:
  * 3-machine fleet: aggregator pulls from all, dedups by record_id.
  * Machine-offline simulation: missing dir → log + continue.
  * Idempotency: re-run over same shards adds zero new records.
  * Schema-version skew: v0.1.0 and v0.2.0 records coexist.
  * Source-stamp preservation: records with metadata.source preserved.
  * Cross-machine collision: same record_id on two machines is logged.
  * Empty shard: 0-record partition handled gracefully.
  * Atomic write: .tmp file does not leak on success.
  * Per-peer summary: SNARC mean + decision-class counts correct.
  * Config parsing: PeerConfig / FederationConfig validation.
  * Dry-run: nothing written to disk.
  * CLI main(): smoke test (temp config, local peers, target date).

Runs with plain ``pytest``; no torch, no network.
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from sage.cognition.router.data import (
    FederationConfig,
    FleetAggregator,
    PeerConfig,
    RouterDatasetWriter,
    SCHEMA_VERSION,
)
from sage.cognition.router.data.federation import _iter_jsonl_gz, main


# ── helpers ────────────────────────────────────────────────────────


TARGET = date(2026, 4, 17)
TARGET_STR = TARGET.isoformat()
FIXED_CLOCK = lambda: datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)


def _make_record(
    i: int,
    machine: str,
    source: str = None,
    action: str = None,
) -> Dict[str, Any]:
    """Synthetic record matching the Track 4 payload envelope.

    If ``source`` is given, emit a v0.2.0 record with metadata.source
    (R1's source-stamping schema). Otherwise emit a v0.1.0 record.
    """
    rec: Dict[str, Any] = {
        "record_id": f"{machine}-rec-{i:06d}",
        "schema_version": "0.1.0" if source is None else "0.2.0",
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
                "action": action or ["invoke", "habit", "noop"][i % 3],
                "confidence": (i % 100) / 100.0,
            },
        },
    }
    if source is not None:
        rec["metadata"] = {"source": source}
    return rec


def _seed_machine_shard(
    base_dir: Path,
    machine: str,
    records: List[Dict[str, Any]],
    target_date: date = TARGET,
) -> Path:
    """Write ``records`` to ``{base_dir}/{machine}/{target_date}.jsonl.gz``."""
    fixed = datetime(
        target_date.year, target_date.month, target_date.day,
        12, 0, 0, tzinfo=timezone.utc,
    )
    writer = RouterDatasetWriter(
        base_dir=base_dir, machine=machine,
        compress=True, buffer_size=1, clock=lambda: fixed,
    )
    try:
        for rec in records:
            writer.append(rec)
    finally:
        writer.close()
    return base_dir / machine / f"{target_date.isoformat()}.jsonl.gz"


def _build_local_config(
    aggregate_dir: Path,
    machine_to_dir: Dict[str, Path],
) -> FederationConfig:
    peers = [
        PeerConfig(machine=m, transport="local", path=str(d))
        for m, d in machine_to_dir.items()
    ]
    return FederationConfig(
        peers=peers,
        aggregate_dir=str(aggregate_dir),
    )


def _read_aggregate(aggregate_dir: Path, target: date = TARGET) -> List[Dict[str, Any]]:
    path = aggregate_dir / f"{target.isoformat()}.jsonl.gz"
    return list(_iter_jsonl_gz(path))


# ── tests ──────────────────────────────────────────────────────────


def test_three_machine_fleet_aggregates_and_dedups(tmp_path: Path) -> None:
    """3 machines with distinct record_ids → aggregate has union."""
    aggregate = tmp_path / "aggregate"

    a_dir = tmp_path / "a"; b_dir = tmp_path / "b"; c_dir = tmp_path / "c"
    _seed_machine_shard(a_dir, "alpha", [_make_record(i, "alpha") for i in range(10)])
    _seed_machine_shard(b_dir, "beta", [_make_record(i, "beta") for i in range(10)])
    _seed_machine_shard(c_dir, "gamma", [_make_record(i, "gamma") for i in range(10)])

    config = _build_local_config(
        aggregate,
        {
            "alpha": a_dir,
            "beta": b_dir,
            "gamma": c_dir,
        },
    )
    agg = FleetAggregator(config=config, clock=FIXED_CLOCK)
    summary = agg.run(target_date=TARGET)

    assert summary.total_records == 30
    assert summary.total_new == 30
    assert summary.cross_machine_collisions == 0
    assert summary.aggregate_path is not None

    records = _read_aggregate(aggregate)
    assert len(records) == 30
    assert sorted(set(r["machine"] for r in records)) == ["alpha", "beta", "gamma"]

    # Per-peer summaries each saw 10 records.
    for peer_summary in summary.peers:
        assert peer_summary.available is True
        assert peer_summary.records_seen == 10
        assert peer_summary.records_new == 10
        assert peer_summary.duplicates == 0


def test_offline_peer_logged_and_skipped(tmp_path: Path, caplog) -> None:
    """Missing shard dir → logged, aggregation continues for others."""
    aggregate = tmp_path / "aggregate"
    live_dir = tmp_path / "live"
    missing_dir = tmp_path / "no-such-path"

    _seed_machine_shard(live_dir, "live", [_make_record(i, "live") for i in range(5)])

    config = FederationConfig(
        peers=[
            PeerConfig(machine="live", transport="local", path=str(live_dir)),
            PeerConfig(machine="offline", transport="local", path=str(missing_dir)),
        ],
        aggregate_dir=str(aggregate),
    )
    agg = FleetAggregator(config=config, clock=FIXED_CLOCK)

    with caplog.at_level(logging.WARNING):
        summary = agg.run(target_date=TARGET)

    assert summary.total_records == 5
    # live peer succeeded, offline peer reported errors.
    live_summary = next(p for p in summary.peers if p.machine == "live")
    off_summary = next(p for p in summary.peers if p.machine == "offline")
    assert live_summary.available is True
    assert off_summary.available is False
    assert any("shard dir missing" in err for err in off_summary.errors)


def test_idempotent_rerun(tmp_path: Path) -> None:
    """Re-running over same shards adds zero records."""
    aggregate = tmp_path / "aggregate"
    a_dir = tmp_path / "a"
    _seed_machine_shard(a_dir, "alpha", [_make_record(i, "alpha") for i in range(7)])

    config = _build_local_config(aggregate, {"alpha": a_dir})
    agg = FleetAggregator(config=config, clock=FIXED_CLOCK)

    s1 = agg.run(target_date=TARGET)
    s2 = agg.run(target_date=TARGET)

    assert s1.total_new == 7
    assert s2.total_new == 0
    assert s2.total_records == 7
    # Records on disk should equal first run.
    assert len(_read_aggregate(aggregate)) == 7


def test_schema_version_skew_both_preserved(tmp_path: Path) -> None:
    """Shards mixing v0.1.0 and v0.2.0 records → all preserved."""
    aggregate = tmp_path / "aggregate"
    a_dir = tmp_path / "a"; b_dir = tmp_path / "b"

    _seed_machine_shard(
        a_dir, "alpha",
        [_make_record(i, "alpha") for i in range(3)],  # v0.1.0
    )
    _seed_machine_shard(
        b_dir, "beta",
        [_make_record(i, "beta", source="raising") for i in range(3)],  # v0.2.0
    )

    config = _build_local_config(aggregate, {"alpha": a_dir, "beta": b_dir})
    agg = FleetAggregator(config=config, clock=FIXED_CLOCK)
    summary = agg.run(target_date=TARGET)

    records = _read_aggregate(aggregate)
    versions = sorted(set(r["schema_version"] for r in records))
    assert versions == ["0.1.0", "0.2.0"]

    # v0.2.0 records kept their metadata.source.
    v020 = [r for r in records if r["schema_version"] == "0.2.0"]
    assert len(v020) == 3
    assert all(r["metadata"]["source"] == "raising" for r in v020)
    assert summary.total_records == 6


def test_source_stamp_preservation(tmp_path: Path) -> None:
    """records with source=raising preserved alongside source=gameplay."""
    aggregate = tmp_path / "aggregate"
    a_dir = tmp_path / "a"
    records = [
        _make_record(1, "alpha", source="raising"),
        _make_record(2, "alpha", source="gameplay"),
        _make_record(3, "alpha", source="idle"),
        _make_record(4, "alpha", source="interactive"),
    ]
    _seed_machine_shard(a_dir, "alpha", records)

    config = _build_local_config(aggregate, {"alpha": a_dir})
    agg = FleetAggregator(config=config, clock=FIXED_CLOCK)
    summary = agg.run(target_date=TARGET)

    out = _read_aggregate(aggregate)
    sources = sorted(r["metadata"]["source"] for r in out)
    assert sources == ["gameplay", "idle", "interactive", "raising"]

    # Per-peer source_counts reflect distribution.
    peer_summary = summary.peers[0]
    assert peer_summary.source_counts == {
        "raising": 1, "gameplay": 1, "idle": 1, "interactive": 1,
    }


def test_cross_machine_collision_logged(tmp_path: Path, caplog) -> None:
    """Same record_id on two machines → logged, first-seen wins."""
    aggregate = tmp_path / "aggregate"
    a_dir = tmp_path / "a"; b_dir = tmp_path / "b"

    # Force identical record_id by using same index on both.
    shared_id = "shared-rec-000001"
    rec_a = _make_record(1, "alpha")
    rec_a["record_id"] = shared_id
    rec_a["payload"]["router_output"]["action"] = "invoke"
    rec_b = _make_record(1, "beta")
    rec_b["record_id"] = shared_id
    rec_b["payload"]["router_output"]["action"] = "habit"  # would differ

    _seed_machine_shard(a_dir, "alpha", [rec_a])
    _seed_machine_shard(b_dir, "beta", [rec_b])

    config = _build_local_config(aggregate, {"alpha": a_dir, "beta": b_dir})
    agg = FleetAggregator(config=config, clock=FIXED_CLOCK)

    with caplog.at_level(logging.WARNING):
        summary = agg.run(target_date=TARGET)

    assert summary.cross_machine_collisions == 1
    assert shared_id in agg.cross_machine_collisions
    assert sorted(agg.cross_machine_collisions[shared_id]) == ["alpha", "beta"]
    # First-seen (alpha) wins.
    records = _read_aggregate(aggregate)
    assert len(records) == 1
    assert records[0]["machine"] == "alpha"
    assert records[0]["payload"]["router_output"]["action"] == "invoke"


def test_empty_shard(tmp_path: Path) -> None:
    """Peer with 0-record partition → handled gracefully, no crash."""
    aggregate = tmp_path / "aggregate"
    a_dir = tmp_path / "a"
    # Shard dir exists but target-date partition does not. Writer with
    # empty list doesn't create the dir; simulate an operator-prepared
    # empty shard root (e.g., a machine that ran but emitted zero records
    # for this date).
    (a_dir / "alpha").mkdir(parents=True)

    config = _build_local_config(aggregate, {"alpha": a_dir})
    agg = FleetAggregator(config=config, clock=FIXED_CLOCK)
    summary = agg.run(target_date=TARGET)

    assert summary.total_records == 0
    assert summary.total_new == 0
    # Peer was reachable but had no partition for this date.
    peer_summary = summary.peers[0]
    assert peer_summary.available is True
    assert peer_summary.records_seen == 0
    # Error list notes the missing partition (informational, not fatal).
    assert any("no partition found" in err for err in peer_summary.errors)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    """--dry-run: aggregation runs, summary returns, but no disk write."""
    aggregate = tmp_path / "aggregate"
    a_dir = tmp_path / "a"
    _seed_machine_shard(a_dir, "alpha", [_make_record(i, "alpha") for i in range(4)])

    config = _build_local_config(aggregate, {"alpha": a_dir})
    agg = FleetAggregator(config=config, clock=FIXED_CLOCK)
    summary = agg.run(target_date=TARGET, dry_run=True)

    assert summary.dry_run is True
    assert summary.total_new == 4
    # Nothing on disk — aggregate dir shouldn't even be created.
    expected = aggregate / f"{TARGET_STR}.jsonl.gz"
    assert not expected.exists()


def test_per_peer_summary_snarc_and_decisions(tmp_path: Path) -> None:
    """Per-peer SNARC mean + decision-class counts correctly computed."""
    aggregate = tmp_path / "aggregate"
    a_dir = tmp_path / "a"
    # Hand-crafted records to make the math checkable.
    records = [
        _make_record(0, "alpha", action="invoke"),  # snarc[surprise]=0.0
        _make_record(1, "alpha", action="invoke"),  # snarc[surprise]=0.01
        _make_record(2, "alpha", action="habit"),   # snarc[surprise]=0.02
    ]
    _seed_machine_shard(a_dir, "alpha", records)

    config = _build_local_config(aggregate, {"alpha": a_dir})
    agg = FleetAggregator(config=config, clock=FIXED_CLOCK)
    summary = agg.run(target_date=TARGET)

    peer = summary.peers[0]
    assert peer.decision_class_counts == {"invoke": 2, "habit": 1}
    assert peer.records_seen == 3
    assert pytest.approx(peer.snarc_mean["surprise"], abs=1e-9) == (0.0 + 0.01 + 0.02) / 3


def test_atomic_write_no_tmp_leak(tmp_path: Path) -> None:
    """Successful run leaves aggregate file but NO .tmp."""
    aggregate = tmp_path / "aggregate"
    a_dir = tmp_path / "a"
    _seed_machine_shard(a_dir, "alpha", [_make_record(i, "alpha") for i in range(3)])

    config = _build_local_config(aggregate, {"alpha": a_dir})
    agg = FleetAggregator(config=config, clock=FIXED_CLOCK)
    agg.run(target_date=TARGET)

    files = sorted(p.name for p in aggregate.iterdir())
    assert f"{TARGET_STR}.jsonl.gz" in files
    # No .tmp leaks.
    assert not any(name.endswith(".tmp") for name in files)
    # Summary sidecar written.
    assert f"{TARGET_STR}.summary.json" in files


def test_summary_sidecar_contents(tmp_path: Path) -> None:
    """Summary sidecar is valid JSON with per-peer details."""
    aggregate = tmp_path / "aggregate"
    a_dir = tmp_path / "a"; b_dir = tmp_path / "b"
    _seed_machine_shard(a_dir, "alpha", [_make_record(i, "alpha") for i in range(5)])
    _seed_machine_shard(b_dir, "beta", [_make_record(i, "beta") for i in range(3)])

    config = _build_local_config(aggregate, {"alpha": a_dir, "beta": b_dir})
    agg = FleetAggregator(config=config, clock=FIXED_CLOCK)
    agg.run(target_date=TARGET)

    sidecar = aggregate / f"{TARGET_STR}.summary.json"
    data = json.loads(sidecar.read_text())
    assert data["target_date"] == TARGET_STR
    assert data["total_records"] == 8
    machine_names = sorted(p["machine"] for p in data["peers"])
    assert machine_names == ["alpha", "beta"]


def test_peer_config_validation() -> None:
    """PeerConfig rejects invalid transports / missing host / missing path."""
    with pytest.raises(ValueError, match="transport must be"):
        PeerConfig(machine="x", transport="ftp", path="/tmp")
    with pytest.raises(ValueError, match="requires 'host'"):
        PeerConfig(machine="x", transport="ssh", path="/tmp")
    with pytest.raises(ValueError, match="'path' must be"):
        PeerConfig(machine="x", transport="local", path="")
    with pytest.raises(ValueError, match="'machine'"):
        PeerConfig(machine="", transport="local", path="/tmp")
    # Happy-path ssh + local.
    p1 = PeerConfig(machine="a", transport="ssh", path="~/shards", host="a.local")
    p2 = PeerConfig(machine="b", transport="local", path="/var/router")
    assert p1.host == "a.local"
    assert p2.transport == "local"


def test_federation_config_from_file(tmp_path: Path) -> None:
    """FederationConfig parses a JSON file and round-trips peers."""
    cfg = {
        "peers": [
            {"machine": "alpha", "transport": "local", "path": "/tmp/alpha"},
            {"machine": "beta", "transport": "ssh", "host": "beta.local",
             "path": "~/shards"},
        ],
        "aggregate_dir": "/tmp/agg",
        "schedule_utc": "03:00",
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    fc = FederationConfig.from_file(p)
    assert fc.aggregate_dir == "/tmp/agg"
    assert fc.schedule_utc == "03:00"
    assert len(fc.peers) == 2
    assert fc.peers[0].transport == "local"
    assert fc.peers[1].host == "beta.local"


def test_cli_main_smoke(tmp_path: Path, capsys) -> None:
    """CLI main() runs end-to-end with a local-only config."""
    aggregate = tmp_path / "aggregate"
    a_dir = tmp_path / "a"
    _seed_machine_shard(a_dir, "alpha", [_make_record(i, "alpha") for i in range(2)])

    cfg = {
        "peers": [
            {"machine": "alpha", "transport": "local", "path": str(a_dir)},
        ],
        "aggregate_dir": str(aggregate),
        "schedule_utc": "02:00",
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    rc = main([
        "--run", "--config", str(cfg_path),
        "--date", TARGET_STR, "--days", "1",
    ])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    summary = json.loads(out.splitlines()[0])
    assert summary["target_date"] == TARGET_STR
    assert summary["total_records"] == 2


def test_no_record_id_preserved_not_dedup(tmp_path: Path) -> None:
    """Records without record_id are preserved but never dedup'd."""
    aggregate = tmp_path / "aggregate"
    a_dir = tmp_path / "a"
    rec1 = _make_record(1, "alpha")
    rec1.pop("record_id")
    rec2 = _make_record(2, "alpha")
    rec2.pop("record_id")
    _seed_machine_shard(a_dir, "alpha", [rec1, rec2])

    config = _build_local_config(aggregate, {"alpha": a_dir})
    agg = FleetAggregator(config=config, clock=FIXED_CLOCK)
    summary = agg.run(target_date=TARGET)
    # First run: both survived.
    assert summary.total_records == 2
    # Second run: no dedup possible → records doubled.
    # (Documented behavior: record_id is the dedup key; without it, any
    # re-pull duplicates. The aggregator logs nothing special because
    # there's nothing to compare against.)
    summary2 = agg.run(target_date=TARGET)
    assert summary2.total_records == 4
