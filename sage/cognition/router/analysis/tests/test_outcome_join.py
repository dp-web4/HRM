"""Tests for outcome_join — synthetic fleet-shape fixtures."""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

# Make parent package importable when tests run standalone.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from sage.cognition.router.analysis.outcome_join import (
    GameSession,
    OutcomeJoiner,
    load_game_sessions,
    enrich_records,
    _dir_start_ts,
    _classify_outcome,
)


# ───────────────────────────────────────────────────────────────────
# Helpers — synthetic fleet fixture
# ───────────────────────────────────────────────────────────────────

def _make_game_run(tmp_path: Path, game: str, run_name: str,
                   data: dict, mtime: float | None = None) -> Path:
    run_dir = tmp_path / game / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json = run_dir / "run.json"
    run_json.write_text(json.dumps(data))
    if mtime is not None:
        os.utime(run_json, (mtime, mtime))
    return run_dir


def _make_record(ts: float, action: str = "invoke", plugin: str = "gridvision",
                 source: str = "gameplay", machine: str = "thor") -> dict:
    return {
        "record_id": f"r{int(ts * 1000)}",
        "schema_version": "v0.2.0",
        "timestamp": ts,
        "machine": machine,
        "router_input": {"snarc_arousal": 0.5, "sensory_modalities": []},
        "router_output": {
            "action": action, "plugin": plugin,
            "plugin_tier": "routine", "payload_hint": None, "habit_id": None,
            "confidence": 0.8, "energy_estimate": 1.0,
            "rationale_code": "goal_driven",
        },
        "outcome": None,
        "metadata": {"source": source},
    }


def _write_records(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ───────────────────────────────────────────────────────────────────
# Pure unit tests — no fixtures
# ───────────────────────────────────────────────────────────────────

def test_dir_start_ts_parses_dated_run():
    ts = _dir_start_ts(Path("/tmp/run_20260410_115444"))
    assert ts is not None
    assert ts > 1_700_000_000


def test_dir_start_ts_returns_none_for_semantic_name():
    assert _dir_start_ts(Path("/tmp/run_L5_winning")) is None
    assert _dir_start_ts(Path("/tmp/run_legitimate_chain")) is None


def test_classify_outcome_win_levels_proved_style():
    lc, wl, fi, won = _classify_outcome({
        "win_levels_proved": 7, "final_level_index": 8,
    })
    assert wl == 7
    assert fi == 8
    assert won is False   # 7 < 8


def test_classify_outcome_explicit_result_win():
    lc, wl, fi, won = _classify_outcome({
        "result": "WIN", "levels_completed": 6, "final_level_index": 5,
    })
    assert won is True
    assert lc == 6


def test_joiner_finds_session_containing_ts():
    sessions = [
        GameSession(game="bp35", run_dir="/a", start_ts=100.0, end_ts=200.0,
                    total_steps=10, levels_completed=5, win_levels=5,
                    final_level_index=4, won=True),
        GameSession(game="lf52", run_dir="/b", start_ts=300.0, end_ts=400.0,
                    total_steps=8, levels_completed=3, win_levels=3,
                    final_level_index=5, won=False),
    ]
    joiner = OutcomeJoiner(sessions)
    assert joiner.find_session(150.0).game == "bp35"
    assert joiner.find_session(350.0).game == "lf52"
    assert joiner.find_session(250.0) is None
    assert joiner.find_session(500.0) is None
    assert joiner.find_session(50.0) is None


def test_joiner_empty_is_safe():
    joiner = OutcomeJoiner([])
    assert joiner.find_session(123.0) is None
    assert len(joiner) == 0


def test_joiner_handles_overlapping_sessions():
    sessions = [
        GameSession(game="a", run_dir="/a", start_ts=100.0, end_ts=250.0,
                    total_steps=10, levels_completed=1, win_levels=1,
                    final_level_index=1, won=True),
        GameSession(game="b", run_dir="/b", start_ts=200.0, end_ts=300.0,
                    total_steps=10, levels_completed=1, win_levels=1,
                    final_level_index=1, won=True),
    ]
    joiner = OutcomeJoiner(sessions)
    match = joiner.find_session(225.0)
    assert match is not None
    assert match.game == "b"


# ───────────────────────────────────────────────────────────────────
# Fixture-based integration tests
# ───────────────────────────────────────────────────────────────────

def test_load_game_sessions_skips_missing_run_json(tmp_path):
    (tmp_path / "bp35" / "run_20260410_115444").mkdir(parents=True)
    sessions = load_game_sessions(tmp_path)
    assert sessions == []


def test_load_game_sessions_basic(tmp_path):
    now = time.time()
    _make_game_run(
        tmp_path, "bp35", "run_20260410_115444",
        {"win_levels_proved": 7, "final_level_index": 7, "total_steps": 100},
        mtime=now,
    )
    _make_game_run(
        tmp_path, "lf52", "run_20260411_091000",
        {"win_levels_proved": 6, "final_level_index": 9, "total_steps": 50},
        mtime=now + 300,
    )
    sessions = load_game_sessions(tmp_path)
    games = {s.game for s in sessions}
    assert games == {"bp35", "lf52"}
    bp35 = next(s for s in sessions if s.game == "bp35")
    assert bp35.won is True


def test_enrich_records_joins_gameplay_by_timestamp(tmp_path):
    records_dir = tmp_path / "records"
    shard = records_dir / "thor" / "2026-04-18.jsonl.gz"
    _write_records(shard, [
        _make_record(150.0, source="gameplay"),
        _make_record(350.0, source="gameplay"),
        _make_record(500.0, source="gameplay"),
        _make_record(150.0, source="raising"),
        _make_record(150.0, source="idle"),
    ])

    sessions = [
        GameSession(game="bp35", run_dir="/a", start_ts=100.0, end_ts=200.0,
                    total_steps=10, levels_completed=5, win_levels=5,
                    final_level_index=4, won=True),
        GameSession(game="lf52", run_dir="/b", start_ts=300.0, end_ts=400.0,
                    total_steps=10, levels_completed=3, win_levels=3,
                    final_level_index=5, won=False),
    ]
    joiner = OutcomeJoiner(sessions)

    out_path = tmp_path / "enriched.jsonl"
    summary = enrich_records(records_dir, joiner, output_path=out_path)

    assert summary.records_seen == 5
    assert summary.gameplay_records == 3
    assert summary.gameplay_matched == 2
    assert summary.gameplay_unmatched == 1
    assert summary.sessions_touched == 2

    actions = summary.decisions_by_action_and_outcome
    assert actions["invoke"]["won"] == 1
    assert actions["invoke"]["lost"] == 1

    enriched = [json.loads(line) for line in out_path.read_text().splitlines()]
    gameplay_with_outcome = [r for r in enriched
                             if (r.get("metadata") or {}).get("game_outcome")]
    assert len(gameplay_with_outcome) == 2
    assert any(
        r["metadata"]["game_outcome"]["game"] == "bp35"
        for r in gameplay_with_outcome
    )


def test_enrich_records_does_not_mutate_originals(tmp_path):
    records_dir = tmp_path / "records"
    shard = records_dir / "thor" / "2026-04-18.jsonl.gz"
    original = _make_record(150.0, source="gameplay")
    _write_records(shard, [original])

    session = GameSession(
        game="bp35", run_dir="/a", start_ts=100.0, end_ts=200.0,
        total_steps=10, levels_completed=5, win_levels=5,
        final_level_index=4, won=True,
    )
    joiner = OutcomeJoiner([session])

    out_path = tmp_path / "enriched.jsonl"
    enrich_records(records_dir, joiner, output_path=out_path)

    with gzip.open(shard, "rt") as f:
        on_disk = json.loads(f.read().strip())
    assert on_disk == original
    assert "game_outcome" not in (on_disk.get("metadata") or {})


def test_enrich_records_summary_only_no_output(tmp_path):
    records_dir = tmp_path / "records"
    shard = records_dir / "thor" / "2026-04-18.jsonl.gz"
    _write_records(shard, [_make_record(150.0, source="gameplay")])

    session = GameSession(
        game="bp35", run_dir="/a", start_ts=100.0, end_ts=200.0,
        total_steps=10, levels_completed=5, win_levels=5,
        final_level_index=4, won=True,
    )
    joiner = OutcomeJoiner([session])

    summary = enrich_records(records_dir, joiner, output_path=None)
    assert summary.gameplay_matched == 1


def test_enrich_records_empty_records_dir(tmp_path):
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    summary = enrich_records(records_dir, OutcomeJoiner([]), output_path=None)
    assert summary.records_seen == 0
    assert summary.gameplay_matched == 0


def test_enrich_records_no_sessions(tmp_path):
    records_dir = tmp_path / "records"
    shard = records_dir / "thor" / "2026-04-18.jsonl.gz"
    _write_records(shard, [
        _make_record(150.0, source="gameplay"),
        _make_record(160.0, source="gameplay"),
    ])
    summary = enrich_records(records_dir, OutcomeJoiner([]), output_path=None)
    assert summary.gameplay_records == 2
    assert summary.gameplay_matched == 0
    assert summary.gameplay_unmatched == 2
