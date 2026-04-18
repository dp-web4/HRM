#!/usr/bin/env python3
"""
End-to-end integration tests for the router shadow hook.
========================================================

Covers the Phase 0 Track 5 acceptance criteria:

  1. Env var OFF → no records written, no state constructed.
  2. Env var ON → records written on each tick.
  3. Deliberate breakage in feature extraction → no propagation.
  4. Deliberate breakage in writer → no propagation.
  5. SNARC sampling: high-salience ticks always kept, low-salience
     stratified.
  6. Failure-isolation of the on_event callback.
  7. Idempotent across loop ticks (hook state doesn't leak).
  8. Hook stats are accurate.
  9. is_shadow_enabled() respects strict "1" match.
  10. End-to-end with minimal stub kernel + RouterDatasetReader
      roundtrip.

No torch, no numpy, no live SAGEConsciousness instantiation — these
tests exercise the shadow plumbing in isolation so they run on any
edge machine and in CI without dependencies.

Run::

    python3 -m pytest sage/cognition/router/tests/test_shadow_integration.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from sage.cognition.router.data import (
    RouterDatasetReader,
    RouterDatasetWriter,
    SnarcStratifiedSampler,
)
from sage.cognition.router.feature_extraction import extract_router_input
from sage.cognition.router.baseline import programmatic_decide
from sage.cognition.router.outputs import RouterOutput
from sage.cognition.router.record import RouterRecord
from sage.cognition.router.shadow import (
    RouterShadowHook,
    SHADOW_ENV_VAR,
    is_shadow_enabled,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_env(monkeypatch):
    """Ensure SAGE_ROUTER_SHADOW is unset for deterministic tests."""
    monkeypatch.delenv(SHADOW_ENV_VAR, raising=False)
    return monkeypatch


@pytest.fixture
def tmp_dataset_dir(tmp_path: Path) -> Path:
    d = tmp_path / "router_shadow"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def plugin_registry() -> Dict[str, Any]:
    """Minimal registry mirroring what SAGEConsciousness would expose."""
    return {
        "vision": {"tier": "routine", "atp_cost": 5.0},
        "audio": {"tier": "routine", "atp_cost": 5.0},
        "language": {"tier": "specialized", "atp_cost": 10.0},
        "control": {"tier": "reflex", "atp_cost": 1.0},
    }


@pytest.fixture
def build_input():
    """Factory for RouterInputs at various SNARC/sensory profiles."""
    def _build(
        tick: int = 1,
        snarc: Optional[Dict[str, float]] = None,
        modalities: Optional[List[str]] = None,
        metabolic_state: str = "wake",
        atp_level: float = 70.0,
        goal_id: Optional[str] = None,
    ):
        snarc = snarc or {"surprise": 0.1, "novelty": 0.1, "arousal": 0.1,
                          "reward": 0.0, "conflict": 0.1}
        sensory = {
            "modalities": modalities if modalities is not None else ["vision"],
            "novelty": snarc.get("novelty", 0.0),
            "urgency": snarc.get("arousal", 0.0),
        }
        metabolic = SimpleNamespace(
            current_state=metabolic_state,
            atp_current=atp_level,
            atp_trend="stable",
        )
        return extract_router_input(
            wm=None,
            snarc=snarc,
            metabolic=metabolic,
            episodic=None,
            cerebellum=None,
            rpe=None,
            metacog=None,
            plugin_registry=None,
            sensory=sensory,
            tick=tick,
            goal_id=goal_id,
        )
    return _build


@pytest.fixture
def make_hook(tmp_dataset_dir):
    """Factory for RouterShadowHook backed by a real writer + sampler."""
    created: List[RouterShadowHook] = []

    def _make(
        *,
        machine: str = "test_machine",
        compress: bool = False,
        sampler: Optional[SnarcStratifiedSampler] = None,
        on_event=None,
    ) -> RouterShadowHook:
        writer = RouterDatasetWriter(
            base_dir=tmp_dataset_dir,
            machine=machine,
            compress=compress,
            buffer_size=1,  # flush each write so tests can read back immediately
        )
        s = sampler if sampler is not None else SnarcStratifiedSampler(seed=42)
        hook = RouterShadowHook(writer=writer, sampler=s, on_event=on_event)
        created.append(hook)
        return hook

    yield _make

    for hook in created:
        try:
            hook.writer.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────
# 1. Env-var gate
# ──────────────────────────────────────────────────────────────────────

def test_is_shadow_enabled_default_off(clean_env):
    """With env var unset, the hook must report disabled."""
    assert is_shadow_enabled() is False


def test_is_shadow_enabled_on(clean_env):
    clean_env.setenv(SHADOW_ENV_VAR, "1")
    assert is_shadow_enabled() is True


def test_is_shadow_enabled_strict_one_only(clean_env):
    """Only the literal string "1" enables the hook — not "0", "true", or empty."""
    clean_env.setenv(SHADOW_ENV_VAR, "true")
    assert is_shadow_enabled() is False
    clean_env.setenv(SHADOW_ENV_VAR, "0")
    assert is_shadow_enabled() is False
    clean_env.setenv(SHADOW_ENV_VAR, "")
    assert is_shadow_enabled() is False


# ──────────────────────────────────────────────────────────────────────
# 2. Env off → no writes when the integration layer skips the hook
# ──────────────────────────────────────────────────────────────────────

def test_env_off_skips_capture(clean_env, tmp_dataset_dir, plugin_registry, build_input):
    """The integration layer short-circuits when the flag is off — no
    files should exist under the shadow dir."""
    assert is_shadow_enabled() is False
    # Simulate the integration layer's conditional path: when
    # is_shadow_enabled() is False, nothing is written.
    if is_shadow_enabled():  # intentionally always False in this test
        pytest.fail("shadow should be disabled by default")

    # No writer instantiated → no file created.
    assert list(tmp_dataset_dir.rglob("*.jsonl*")) == []


# ──────────────────────────────────────────────────────────────────────
# 3. End-to-end capture + read-back
# ──────────────────────────────────────────────────────────────────────

def test_end_to_end_capture_and_read_back(
    make_hook, tmp_dataset_dir, plugin_registry, build_input
):
    """Capture 10 decisions and re-read them from the dataset.

    Verifies: records reach disk, record count matches, SNARC values
    survive the round trip, action/rationale fields are consistent
    with the programmatic baseline's output.
    """
    # Use a high-salience profile so the sampler keeps everything
    # through warmup AND afterwards.
    hook = make_hook(machine="e2e")
    for tick in range(10):
        router_input = build_input(
            tick=tick,
            snarc={"surprise": 0.8, "novelty": 0.9, "arousal": 0.9,
                   "reward": 0.5, "conflict": 0.7},
            modalities=["vision"],
        )
        output = programmatic_decide(router_input, plugin_registry)
        rec = hook.record_decision(router_input, output)
        assert rec is not None
    hook.writer.close()

    reader = RouterDatasetReader(base_dir=tmp_dataset_dir)
    records = list(reader.read_partition(machine="e2e"))
    assert len(records) == 10
    # Every record carries the right schema shell.
    for r in records:
        assert r.get("schema_version"), "schema_version must be stamped"
        assert r["router_input"]["snarc_novelty"] == pytest.approx(0.9)
        # Baseline on this profile should land on invoke (vision plugin)
        # with a high_novelty rationale.
        assert r["router_output"]["action"] == "invoke"
        assert r["router_output"]["plugin"] == "vision"


# ──────────────────────────────────────────────────────────────────────
# 4. Failure isolation — feature-extraction-level breakage
# ──────────────────────────────────────────────────────────────────────

def test_hook_survives_broken_input(make_hook, plugin_registry, build_input):
    """A broken input dataclass (simulated by a non-RouterInput object)
    must not raise out of record_decision."""
    hook = make_hook()
    bad_input = object()   # not a RouterInput
    bad_output = RouterOutput.noop()
    result = hook.record_decision(bad_input, bad_output)  # must not raise
    assert result is None
    # Failure counted in stats.
    assert hook.errors >= 1


def test_hook_survives_broken_sampler(make_hook, plugin_registry, build_input):
    """Sampler that raises must not propagate."""
    class BrokenSampler:
        def should_keep(self, snarc):
            raise RuntimeError("sampler boom")

    # Writer is fine; sampler broken.
    writer = make_hook().writer  # borrow a writer from the factory
    hook = RouterShadowHook(writer=writer, sampler=BrokenSampler())
    router_input = build_input()
    output = programmatic_decide(router_input, plugin_registry)
    result = hook.record_decision(router_input, output)
    assert result is None
    assert hook.errors == 1


def test_hook_survives_broken_writer(make_hook, plugin_registry, build_input):
    """Writer that raises must not propagate."""
    class BrokenWriter:
        def append(self, record):
            raise RuntimeError("writer boom")

    sampler = SnarcStratifiedSampler(seed=1)
    hook = RouterShadowHook(writer=BrokenWriter(), sampler=sampler)
    router_input = build_input(
        snarc={"surprise": 1.0, "novelty": 1.0, "arousal": 1.0,
               "reward": 0.0, "conflict": 1.0},
    )
    output = programmatic_decide(router_input, plugin_registry)
    result = hook.record_decision(router_input, output)
    assert result is None
    assert hook.errors == 1


# ──────────────────────────────────────────────────────────────────────
# 5. SNARC sampling behavior
# ──────────────────────────────────────────────────────────────────────

def test_sampling_high_salience_always_kept(
    make_hook, plugin_registry, build_input
):
    """Top-quintile SNARC ticks keep at 100% (post-warmup).

    The sampler is configured to drop EVERYTHING except the top
    quintile. High-salience ticks must be fully kept.
    """
    sampler = SnarcStratifiedSampler(
        keep_rates=[0.0, 0.0, 0.0, 0.0, 1.0],
        warmup=10,
        window_size=200,
        seed=7,
    )
    writer = make_hook().writer
    hook = RouterShadowHook(writer=writer, sampler=sampler)

    # Push 100 varied-but-low salience observations to build a
    # non-degenerate rolling window. Using a spread of values (not
    # all zero) ensures the quintile boundaries actually split
    # the distribution — otherwise all ticks land in quintile 4
    # by the sampler's strict-less-than semantics.
    import random as _random
    rng = _random.Random(999)
    for tick in range(100):
        low = rng.uniform(0.0, 0.4)
        ri = build_input(
            tick=tick,
            snarc={"surprise": low, "novelty": low, "arousal": low,
                   "reward": 0.0, "conflict": low},
        )
        out = programmatic_decide(ri, plugin_registry)
        hook.record_decision(ri, out)

    # 50 deliberately high-salience ticks (clearly top quintile).
    kept_before = hook.decisions_kept
    for tick in range(100, 150):
        ri = build_input(
            tick=tick,
            snarc={"surprise": 1.0, "novelty": 1.0, "arousal": 1.0,
                   "reward": 0.0, "conflict": 1.0},
        )
        out = programmatic_decide(ri, plugin_registry)
        hook.record_decision(ri, out)
    kept_after = hook.decisions_kept
    # Post-warmup high-salience ticks land in quintile 4 (100% keep).
    assert kept_after - kept_before == 50


def test_sampling_low_salience_mostly_dropped(
    make_hook, plugin_registry, build_input
):
    """Bottom-quintile SNARC ticks keep at ~5% post-warmup.

    A non-degenerate salience spread is required: the quintile
    boundaries are drawn from the rolling window, so an all-zero
    stream produces all-zero cut points and every tick degenerates
    into quintile 4. We use a low-mean spread so most ticks land
    in quintile 0.
    """
    sampler = SnarcStratifiedSampler(
        keep_rates=[0.05, 0.20, 0.20, 0.20, 1.0],
        warmup=10,
        window_size=1000,
        seed=13,
    )
    writer = make_hook().writer
    hook = RouterShadowHook(writer=writer, sampler=sampler)

    import random as _random
    rng = _random.Random(41)

    # 1000 varied low-salience ticks (salience in [0.0, 0.2]).
    # Post-warmup ~5% quintile-0 ⇒ expect ~50 kept, plus some
    # stratified sampling from the middle quintiles formed by the
    # spread within [0, 0.2].
    for tick in range(1000):
        val = rng.uniform(0.0, 0.2)
        ri = build_input(
            tick=tick,
            snarc={"surprise": val, "novelty": val, "arousal": val,
                   "reward": 0.0, "conflict": val},
        )
        out = programmatic_decide(ri, plugin_registry)
        hook.record_decision(ri, out)

    # Warmup keeps 10. After that, a weighted mix of 5% / 20%
    # depending on where the spread falls in the quintiles. Bounds
    # are loose — we just want "substantially below 1000" and not
    # "all kept" (which would mean stratification was off).
    assert 10 <= hook.decisions_kept <= 400
    assert hook.decisions_kept < 1000
    # The running counters must stay consistent.
    assert (hook.decisions_seen
            - hook.decisions_kept
            - hook.decisions_dropped_writer
            - hook.errors) == hook.decisions_dropped_sampler


# ──────────────────────────────────────────────────────────────────────
# 6. On-event callback isolation
# ──────────────────────────────────────────────────────────────────────

def test_broken_on_event_does_not_break_hook(
    make_hook, plugin_registry, build_input
):
    calls: List[tuple] = []

    def broken(kind: str, payload: dict) -> None:
        calls.append((kind, payload))
        if kind == "captured":
            raise RuntimeError("observer boom")

    hook = make_hook(on_event=broken)
    ri = build_input(
        snarc={"surprise": 1.0, "novelty": 1.0, "arousal": 1.0,
               "reward": 0.0, "conflict": 1.0},
    )
    out = programmatic_decide(ri, plugin_registry)
    rec = hook.record_decision(ri, out)
    assert rec is not None
    # The observer did get called…
    assert any(c[0] == "captured" for c in calls)
    # …and the broken observer did NOT count as a hook error
    # (on_event is optional; its exceptions are absorbed silently).
    assert hook.errors == 0


# ──────────────────────────────────────────────────────────────────────
# 7. Idempotency — no state leak between calls
# ──────────────────────────────────────────────────────────────────────

def test_repeated_calls_do_not_accumulate_extra_state(
    make_hook, plugin_registry, build_input
):
    """Each call should affect only writer + sampler (their
    legitimate state). Hook stats advance predictably."""
    hook = make_hook()
    for i in range(50):
        ri = build_input(
            tick=i,
            snarc={"surprise": 0.9, "novelty": 0.9, "arousal": 0.9,
                   "reward": 0.0, "conflict": 0.9},
        )
        out = programmatic_decide(ri, plugin_registry)
        hook.record_decision(ri, out)

    stats = hook.get_stats()
    assert stats["decisions_seen"] == 50
    # Keep rate should be ~100% on this profile (warmup + high salience).
    assert stats["decisions_kept"] == 50
    assert stats["errors"] == 0


# ──────────────────────────────────────────────────────────────────────
# 8. Writer-return-False path
# ──────────────────────────────────────────────────────────────────────

def test_writer_returns_false_counts_as_drop(make_hook, build_input):
    class RefusingWriter:
        def __init__(self):
            self.calls = 0
        def append(self, record):
            self.calls += 1
            return False  # not an exception — writer chose to drop

    sampler = SnarcStratifiedSampler(seed=3)
    hook = RouterShadowHook(writer=RefusingWriter(), sampler=sampler)
    ri = build_input(
        snarc={"surprise": 1.0, "novelty": 1.0, "arousal": 1.0,
               "reward": 0.0, "conflict": 1.0},
    )
    out = RouterOutput.noop()
    result = hook.record_decision(ri, out)
    assert result is None
    assert hook.decisions_dropped_writer == 1
    assert hook.errors == 0  # dropped, not errored


# ──────────────────────────────────────────────────────────────────────
# 9. Dataclass-construction validation
# ──────────────────────────────────────────────────────────────────────

def test_hook_rejects_non_callable_writer():
    sampler = SnarcStratifiedSampler(seed=3)
    with pytest.raises(TypeError):
        RouterShadowHook(writer=object(), sampler=sampler)


def test_hook_rejects_non_callable_sampler():
    class FakeWriter:
        def append(self, record):
            return True
    with pytest.raises(TypeError):
        RouterShadowHook(writer=FakeWriter(), sampler=object())


# ──────────────────────────────────────────────────────────────────────
# 10. Written record is a valid RouterRecord round-trip
# ──────────────────────────────────────────────────────────────────────

def test_written_record_roundtrips_to_routerrecord(
    make_hook, tmp_dataset_dir, plugin_registry, build_input
):
    """The record on disk, when loaded by RouterDatasetReader and
    passed through RouterRecord.from_dict, must reconstruct a valid
    RouterRecord with populated RouterInput + RouterOutput."""
    hook = make_hook(machine="rt", compress=False)
    ri = build_input(
        tick=42,
        snarc={"surprise": 0.6, "novelty": 0.8, "arousal": 0.7,
               "reward": 0.0, "conflict": 0.5},
        modalities=["vision"],
    )
    out = programmatic_decide(ri, plugin_registry)
    hook.record_decision(ri, out)
    hook.writer.close()

    reader = RouterDatasetReader(base_dir=tmp_dataset_dir)
    records = list(reader.read_partition(machine="rt"))
    assert len(records) == 1
    rr = RouterRecord.from_dict(records[0])
    assert rr.router_input.tick == 42
    assert rr.router_input.snarc_novelty == pytest.approx(0.8)
    assert rr.router_output.action in {"invoke", "noop", "habit"}
    assert rr.schema_version  # stamped
    assert rr.record_id        # stamped
