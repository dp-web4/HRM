"""Tests for the metacog cartridge writer."""

import json
from pathlib import Path

import pytest

from sage.cognition.metacog.core import Metacog, MetacogConfig, MetacogSignal
from sage.cognition.metacog.cartridge_writer import (
    CartridgeEntry,
    MetacogCartridgeWriter,
    signal_to_entry,
)


def _make_signal(
    signal: str = "perseveration",
    severity: float = 0.8,
    suggestion: str = "reframe",
    evidence: dict | None = None,
    tick: int = 42,
) -> MetacogSignal:
    return MetacogSignal(
        signal=signal,
        severity=severity,
        suggestion=suggestion,
        evidence=evidence or {"action_count": 3},
        tick=tick,
    )


class TestSignalToEntry:
    def test_basic_conversion(self):
        sig = _make_signal()
        entry = signal_to_entry(sig, game="toy_b", machine="nomad")
        assert entry.signal == "perseveration"
        assert entry.game == "toy_b"
        assert entry.machine == "nomad"
        assert "[role:metacog]" in entry.text
        assert "[signal:perseveration]" in entry.text
        assert "[game:toy_b]" in entry.text
        assert "[machine:nomad]" in entry.text

    def test_tags_in_canonical_format(self):
        entry = signal_to_entry(
            _make_signal(signal="budget_critical"),
            game="toy_a", machine="cbp",
        )
        # All tags should be [namespace:value] format
        assert "[role:metacog]" in entry.text
        assert "[signal:budget_critical]" in entry.text
        assert "[scope:moment]" in entry.text
        assert "[source:gameplay]" in entry.text

    def test_observation_text_includes_evidence(self):
        sig = _make_signal(evidence={"ratio": 0.625, "atp_balance": 10.0})
        entry = signal_to_entry(sig)
        assert "ratio=0.625" in entry.text
        assert "atp_balance=10.0" in entry.text

    def test_context_and_outcome_included(self):
        entry = signal_to_entry(
            _make_signal(),
            context="step 42 stuck on L2",
            outcome="reframed to different strategy, unstuck",
        )
        assert "step 42" in entry.text
        assert "unstuck" in entry.text

    def test_per_pattern_meta_has_signal_fields(self):
        sig = _make_signal(tick=99)
        entry = signal_to_entry(sig, game="toy_b")
        meta = entry.per_pattern_meta
        assert meta["signal"] == "perseveration"
        assert meta["severity"] == 0.8
        assert meta["tick"] == 99
        assert "evidence" in meta

    def test_no_game_still_valid(self):
        entry = signal_to_entry(_make_signal(), machine="nomad")
        assert "[game:" not in entry.text
        assert entry.game == ""

    def test_json_roundtrip(self):
        entry = signal_to_entry(
            _make_signal(), game="toy_b", machine="nomad"
        )
        text = entry.to_json()
        restored = json.loads(text)
        assert restored["signal"] == "perseveration"
        assert restored["game"] == "toy_b"
        assert "[role:metacog]" in restored["text"]


class TestMetacogCartridgeWriter:
    def test_observe_buffers_entries(self):
        writer = MetacogCartridgeWriter(
            output_dir="/tmp/test", machine="nomad"
        )
        writer.observe(_make_signal(), game="toy_b")
        writer.observe(_make_signal(signal="budget_anxiety"), game="toy_a")
        assert writer.pending == 2

    def test_flush_writes_jsonl(self, tmp_path: Path):
        writer = MetacogCartridgeWriter(
            output_dir=tmp_path, machine="nomad"
        )
        writer.observe(_make_signal(), game="toy_b", context="stuck on L2")
        writer.observe(
            _make_signal(signal="exploration_starvation"),
            game="toy_a", outcome="tried new area",
        )
        out = writer.flush()
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        # Each line is valid JSON
        for line in lines:
            d = json.loads(line)
            assert "text" in d
            assert "[role:metacog]" in d["text"]

    def test_flush_clears_buffer(self, tmp_path: Path):
        writer = MetacogCartridgeWriter(output_dir=tmp_path, machine="nomad")
        writer.observe(_make_signal(), game="toy_b")
        writer.flush()
        assert writer.pending == 0

    def test_flush_appends_not_overwrites(self, tmp_path: Path):
        writer = MetacogCartridgeWriter(output_dir=tmp_path, machine="nomad")
        writer.observe(_make_signal(), game="toy_b")
        writer.flush()
        writer.observe(_make_signal(signal="budget_anxiety"), game="toy_a")
        writer.flush()
        lines = (tmp_path / "metacog_observations.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_entries_returns_copy(self):
        writer = MetacogCartridgeWriter(output_dir="/tmp", machine="nomad")
        writer.observe(_make_signal(), game="toy_b")
        entries = writer.entries()
        assert len(entries) == 1
        entries.clear()
        assert writer.pending == 1  # original buffer untouched


class TestEndToEnd:
    def test_metacog_to_cartridge_pipeline(self, tmp_path: Path):
        """Full flow: metacog fires → writer observes → flush → valid JSONL
        with correct tags for membot consumption."""
        m = Metacog(config=MetacogConfig(
            perseveration_window=3, signal_cooldown_ticks=0
        ))
        writer = MetacogCartridgeWriter(
            output_dir=tmp_path, machine="nomad"
        )

        # Trigger perseveration
        action = {"type": "click", "target": (5, 5)}
        for tick in range(3):
            signals = m.observe_tick(tick, action_taken=action, state_delta=None)
        for sig in signals:
            writer.observe(sig, game="toy_b", context="step 42")

        # Trigger budget critical
        m.observe_tick(
            10, action_taken={"x": 1}, atp_cost=5.0,
            atp_balance=3.0, estimated_actions_to_goal=10.0,
        )
        for sig in m.active_signals():
            if sig.tick == 10:
                writer.observe(sig, game="toy_b",
                               outcome="switched to cheaper strategy")

        out = writer.flush()
        lines = out.read_text().strip().split("\n")
        assert len(lines) >= 2

        # Verify each line is membot-compatible
        for line in lines:
            d = json.loads(line)
            assert d["source"] == "metacog_observation"
            assert d["game"] == "toy_b"
            assert d["machine"] == "nomad"
            assert "[role:metacog]" in d["text"]
            assert d["signal"] in ("perseveration", "budget_critical")
            assert "per_pattern_meta" in d
            assert "severity" in d["per_pattern_meta"]
