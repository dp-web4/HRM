#!/usr/bin/env python3
"""
Outcome tracking — Phase 0 Track 6.
===================================

``OutcomeTracker`` backfills every written ``RouterRecord`` with an
observed outcome: the 5-tick post-decision SNARC trajectory, a scalar
``snarc_resolution_score`` per PRD §4.7.E, an optional RPE signal, and
a ``level_up_observed`` flag (game-specific; optional for Phase 0).

Why this exists
---------------

Track 5's ``RouterShadowHook`` writes records to the main dataset
partition as soon as a decision is made. At write time, the outcome
cannot be known — it lives in the *next* 5 ticks. The Track 6 tracker
watches subsequent ticks, accumulates per-record SNARC trajectories,
and emits a standalone outcome record to a **sidecar** partition keyed
by ``record_id``. The reader merges the main partition with the sidecar
at load time (see ``sage.cognition.router.data.reader``).

Sidecar vs rewrite — design choice
----------------------------------

We deliberately do NOT rewrite the main partition to add outcomes.
Reasons:

1. **Append-only discipline of the writer**. ``RouterDatasetWriter``
   keeps a gzip handle open across flushes. Rewriting the partition
   means coordinating with a live writer — race-prone, expensive.
2. **Atomicity**. Rewriting a gzipped JSONL file means decoding,
   editing, recompressing, atomic-renaming. One bug drops the whole
   day's data. Sidecars never touch the main file.
3. **Reproducibility**. Main partition is immutable once a day ends.
   Forensic reads ("what did we know at decision time?") are trivially
   available. "What was the outcome?" is a join, not a mutation.
4. **Failure isolation**. A broken outcome pipeline can NEVER corrupt
   the primary dataset. Disable the tracker, delete the sidecars, re-
   ingest cleanly. This is the same discipline PRD §5.5 asks of the
   writer itself.
5. **Pruning independence**. Track 9 prunes main partitions by SNARC
   quintile. Outcomes for dropped records become orphans in sidecars,
   but the reader's merge is a left-outer join — orphan outcomes never
   surface without a matching record. A later pruner pass can sweep
   orphan outcome files by age.

Sidecar filename convention
---------------------------

    {base_dir}/{machine}/outcome_{YYYY-MM-DD}.jsonl[.gz]

Same daily rollover as main partitions. Uses ``RouterDatasetWriter``
under the hood with a ``machine`` subpath that embeds "outcome_" into
the filename — see ``_OutcomeWriter`` below.

Edge cases handled
------------------

* **Tick gap > window**. If we buffer a record at tick N and the next
  observation arrives at tick N+10 (window is 5), the outcome is
  emitted immediately on flush with ``status='incomplete'`` and
  whatever partial trajectory accumulated.
* **Kernel restart mid-buffer**. The in-memory buffer is persisted to
  ``{base_dir}/{machine}/outcome_buffer.json`` on every ``flush()``
  and on ``close()``. ``load_buffer()`` restores it. Resumption is
  best-effort — if the tick counter has advanced, stale records are
  flushed with ``status='incomplete'``.
* **Plugin failure mid-dispatch**. Callers can pass ``failure=True``
  to ``observe_tick`` (for the decision tick's own record) or later;
  we record ``status='failed'`` with whatever trajectory exists.
* **Ordering**. Observations must be monotonic in ``tick``. Out-of-
  order observations are logged and ignored. This matches the
  consciousness loop's single-threaded tick semantics.
* **No-op**. A decision whose observations never arrive flushes at
  ``close()`` with ``status='incomplete'`` so we always have a record
  on disk for every recorded decision.

Failure isolation
-----------------

Every method swallows exceptions. The tracker never propagates an
error into the consciousness loop. Disk failures, serialization
failures, and buffer-restore failures all log at WARNING and continue.

No torch dependency. Pure stdlib.

Spec:
  - PRD §4.7.E (SNARC trajectory as outcome signal)
  - PRD §5 (record shape, sidecar governance)
  - Sprint doc Track 6 (router-sprint-1-phase-0.md)
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from sage.cognition.router.data.writer import RouterDatasetWriter


# ──────────────────────────────────────────────────────────────────────
# Schema + constants
# ──────────────────────────────────────────────────────────────────────

# Outcome schema version. Distinct from the record-level
# ``ROUTER_SCHEMA_VERSION`` — outcome shape evolves independently.
# Consumers MUST key off this rather than the record schema.
OUTCOME_SCHEMA_VERSION: str = "v0.1.0"

# PRD §4.7.E says "5-tick post-decision SNARC trajectory". We collect
# ticks t+1 .. t+TRAJECTORY_TICKS inclusive — i.e. 5 samples after the
# decision tick. The decision-tick SNARC itself is the baseline (t0).
# Changing this is NOT a free knob — any change MUST bump
# ``OUTCOME_SCHEMA_VERSION`` and propagate to the reader's merge logic.
TRAJECTORY_TICKS: int = 5

# SNARC dimensions we track for resolution scoring. Subset of the
# SNARC vector PRD §3.1 declares on RouterInput: arousal/conflict/
# surprise are the "dominant features" PRD §4.7.E flags. Reward and
# novelty are recorded too (for trajectory completeness) but not
# counted toward ``snarc_resolution_score``.
RESOLUTION_DIMS: Tuple[str, ...] = ("arousal", "conflict", "surprise")
ALL_SNARC_DIMS: Tuple[str, ...] = (
    "arousal", "conflict", "surprise", "reward", "novelty",
)

# Status values for the outcome record.
STATUS_COMPLETE: str = "complete"      # full window observed
STATUS_INCOMPLETE: str = "incomplete"  # tick gap / flush before window closed
STATUS_FAILED: str = "failed"          # plugin failure reported during window

# Subdirectory + filename prefix for sidecar partitions. The underlying
# writer lays out ``{base_dir}/{machine}/`` — we encode "outcome_" into
# the prefix by using a writer-machine-name trick (see _OutcomeWriter).
OUTCOME_FILE_PREFIX: str = "outcome_"

# Buffer-persistence filename. Lives alongside main partitions per
# machine — simpler than a separate directory.
BUFFER_FILENAME: str = "outcome_buffer.json"

_log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data shapes
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _PendingOutcome:
    """An outcome record still accumulating its SNARC trajectory.

    Intentionally minimal — the buffer is hot at fleet scale (one per
    kept decision per machine per tick). Anything that isn't needed to
    materialize the final outcome stays off this dataclass.

    Fields
    ------
    record_id:
        ``RouterRecord.record_id`` — primary key of the sidecar write.
    tick_at_decision:
        Loop tick when the decision was made. Trajectory samples must
        have ``tick > tick_at_decision`` and are counted in order.
    snarc_at_decision:
        Dict of SNARC values at t0. Used to identify the dominant
        dimension for ``snarc_resolution_score`` per PRD §4.7.E.
    trajectory:
        List of (tick, {snarc_dim: float}) tuples, in observation order.
        Truncated to ``TRAJECTORY_TICKS`` samples at most.
    rpe_signal:
        Optional scalar RPE observed in the window. Phase 4+ wiring.
    level_up_observed:
        Optional bool; game-specific flag set via ``observe_tick``.
    failure:
        Optional bool; set True when a plugin failure is reported.
    decision_timestamp:
        Wall-clock seconds at decision — for staleness checks on
        resume. Not part of the emitted outcome.
    """

    record_id: str
    tick_at_decision: int
    snarc_at_decision: Dict[str, float]
    trajectory: List[Tuple[int, Dict[str, float]]] = field(default_factory=list)
    rpe_signal: Optional[float] = None
    level_up_observed: Optional[bool] = None
    failure: bool = False
    decision_timestamp: float = field(default_factory=time.time)

    def to_json(self) -> Dict[str, Any]:
        """Serialize for buffer persistence. NOT the outcome schema."""
        return {
            "record_id": self.record_id,
            "tick_at_decision": self.tick_at_decision,
            "snarc_at_decision": dict(self.snarc_at_decision),
            "trajectory": [(int(t), dict(s)) for t, s in self.trajectory],
            "rpe_signal": self.rpe_signal,
            "level_up_observed": self.level_up_observed,
            "failure": bool(self.failure),
            "decision_timestamp": float(self.decision_timestamp),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "_PendingOutcome":
        traj = [
            (int(t), {str(k): float(v) for k, v in s.items()})
            for t, s in data.get("trajectory", [])
        ]
        return cls(
            record_id=str(data["record_id"]),
            tick_at_decision=int(data["tick_at_decision"]),
            snarc_at_decision={str(k): float(v) for k, v in
                               data.get("snarc_at_decision", {}).items()},
            trajectory=traj,
            rpe_signal=data.get("rpe_signal"),
            level_up_observed=data.get("level_up_observed"),
            failure=bool(data.get("failure", False)),
            decision_timestamp=float(data.get("decision_timestamp", time.time())),
        )


# ──────────────────────────────────────────────────────────────────────
# Sidecar writer wrapper
# ──────────────────────────────────────────────────────────────────────


class _OutcomeWriter:
    """Thin shim that funnels outcome records to an ``outcome_{date}``
    sidecar partition using ``RouterDatasetWriter`` machinery.

    The underlying writer expects ``{base_dir}/{machine}/{date}.jsonl``.
    We achieve the ``outcome_`` filename prefix by passing a
    ``machine`` directory that points at the real machine subpath but
    passing a custom ``clock`` that prefixes the date. That is too
    clever. Simpler: we just write the JSONL ourselves, using the same
    atomic append+flush+gzip discipline, but keyed on the filename
    convention we need.

    This class is intentionally small — Track 4 already did the work of
    proving JSONL append-with-rollover correct. We copy the pattern
    rather than reusing the class to keep failure isolation tight: an
    outcome writer failure must not take the main writer down with it.
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        machine: str,
        compress: bool = True,
        buffer_size: int = 64,
        clock: Optional[Any] = None,
    ) -> None:
        if not machine or "/" in machine or "\\" in machine:
            raise ValueError(f"machine must be a simple name, got {machine!r}")
        # Delegate to a RouterDatasetWriter that thinks it's writing to
        # machine subdir ``{machine}`` but with partition names prefixed
        # by "outcome_". We can't change the writer's filename format
        # without invasive surgery, so we override via subclassing.
        self._inner = _PrefixedDatasetWriter(
            base_dir=base_dir,
            machine=machine,
            compress=compress,
            buffer_size=buffer_size,
            clock=clock,
            filename_prefix=OUTCOME_FILE_PREFIX,
        )

    def append(self, outcome_record: Dict[str, Any]) -> bool:
        """Queue one outcome dict. Never raises."""
        try:
            return bool(self._inner.append(outcome_record))
        except Exception as e:  # noqa: BLE001
            _log.warning("outcome sidecar append failed: %s", e)
            return False

    def flush(self) -> None:
        try:
            self._inner.flush()
        except Exception as e:  # noqa: BLE001
            _log.warning("outcome sidecar flush failed: %s", e)

    def close(self) -> None:
        try:
            self._inner.close()
        except Exception as e:  # noqa: BLE001
            _log.warning("outcome sidecar close failed: %s", e)

    def current_path(self) -> Optional[Path]:
        return self._inner.current_path()

    def get_stats(self) -> Dict[str, Any]:
        return self._inner.get_stats()


class _PrefixedDatasetWriter(RouterDatasetWriter):
    """RouterDatasetWriter that prepends a filename prefix to partitions.

    Same partition layout — ``{base_dir}/{machine}/{prefix}{date}.jsonl[.gz]``.
    Used by ``_OutcomeWriter`` to emit ``outcome_YYYY-MM-DD.jsonl[.gz]``
    alongside the main partitions without needing a separate subdir.
    """

    def __init__(self, *args: Any, filename_prefix: str = "", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._filename_prefix = filename_prefix

    def _ensure_partition(self) -> None:  # type: ignore[override]
        today = self._clock().strftime("%Y-%m-%d")
        ext = ".jsonl.gz" if self.compress else ".jsonl"
        fname = f"{self._filename_prefix}{today}{ext}"
        partition = self.base_dir / self.machine / fname

        if self._current_path == partition and self._current_handle is not None:
            return

        # Close previous handle.
        if self._current_handle is not None:
            try:
                self._current_handle.close()
            except Exception as e:  # pragma: no cover
                _log.exception("Error closing previous outcome partition: %s", e)
            self._current_handle = None
            self._current_path = None

        try:
            partition.parent.mkdir(parents=True, exist_ok=True)
            if self.compress:
                import gzip as _gzip
                handle: Any = _gzip.open(str(partition), "at", encoding="utf-8")
            else:
                handle = open(partition, "a", encoding="utf-8")
            self._current_handle = handle
            self._current_path = partition
        except Exception as e:
            _log.exception("Failed to open outcome partition %s: %s", partition, e)
            self._current_handle = None
            self._current_path = None


# ──────────────────────────────────────────────────────────────────────
# OutcomeTracker
# ──────────────────────────────────────────────────────────────────────


class OutcomeTracker:
    """Backfill ``RouterRecord`` outcomes via SNARC-trajectory observation.

    Lifecycle
    ---------

    1. ``RouterShadowHook`` writes a record to the main partition and
       calls ``tracker.register_decision(record_id, tick, snarc_at_decision)``
       to buffer a pending outcome.
    2. On each subsequent tick, the consciousness loop calls
       ``tracker.observe_tick(tick, snarc, rpe_signal=..., level_up=...)``.
       Every pending record whose tick window includes the current tick
       gets one more trajectory sample.
    3. When a pending record's window closes (i.e. ``TRAJECTORY_TICKS``
       samples collected, OR the next observation's tick exceeds
       ``tick_at_decision + TRAJECTORY_TICKS``), its outcome is emitted
       to the sidecar writer and the pending entry is discarded.
    4. ``tracker.flush()`` emits any remaining pending outcomes with
       ``status='incomplete'``. ``tracker.close()`` calls ``flush()``
       and closes the writer handle.

    Buffer persistence
    ------------------

    On every ``flush()`` / ``close()``, the pending buffer is
    serialized to ``{base_dir}/{machine}/outcome_buffer.json``. On
    construction, ``load_buffer()`` reads it back if it exists. A
    crashed kernel loses AT MOST one ``flush_interval`` of buffered
    pending outcomes; those records' outcomes will be absent from the
    sidecar (the reader surfaces them with ``outcome=None``).

    Parameters
    ----------
    base_dir:
        Root of the router dataset layout (same as the writer's).
    machine:
        Machine name — matches the writer's machine subdir.
    compress:
        Gzip sidecar partitions. Default True, matches writer default.
    buffer_size:
        Sidecar writer flush threshold (records). Default 64.
    max_pending:
        Soft cap on the in-memory pending buffer. When exceeded, the
        oldest pending entry is evicted + flushed with
        ``status='incomplete'``. Guards against an unbounded buffer
        if ``observe_tick`` stops being called for any reason.
    clock:
        Optional callable returning current UTC datetime. Forwarded
        to the sidecar writer for deterministic test partitions.

    Notes
    -----
    * The tracker is single-threaded by design. The consciousness loop
      runs it from the loop thread; no locking. If a future architecture
      calls ``observe_tick`` from multiple threads, wrap with a lock.
    * Out-of-order ticks are rejected (logged, ignored). The
      consciousness loop is monotonic in tick; an out-of-order tick
      indicates either a reset (handle by re-instantiating the tracker)
      or a bug.
    """

    # Maximum pending entries before the oldest is force-evicted.
    DEFAULT_MAX_PENDING: int = 4096

    def __init__(
        self,
        base_dir: Union[str, Path],
        machine: str,
        compress: bool = True,
        buffer_size: int = 64,
        max_pending: int = DEFAULT_MAX_PENDING,
        clock: Optional[Any] = None,
    ) -> None:
        if max_pending < 1:
            raise ValueError(f"max_pending must be >= 1, got {max_pending}")
        self.base_dir = Path(base_dir)
        self.machine = machine
        self.max_pending = max_pending

        # OrderedDict preserves insertion order — ``popitem(last=False)``
        # evicts the oldest entry when the cap is hit.
        self._pending: "OrderedDict[str, _PendingOutcome]" = OrderedDict()

        self._writer = _OutcomeWriter(
            base_dir=base_dir,
            machine=machine,
            compress=compress,
            buffer_size=buffer_size,
            clock=clock,
        )
        # Path to the persisted buffer.
        self._buffer_path = self.base_dir / machine / BUFFER_FILENAME

        # Observability counters.
        self.decisions_registered: int = 0
        self.outcomes_emitted_complete: int = 0
        self.outcomes_emitted_incomplete: int = 0
        self.outcomes_emitted_failed: int = 0
        self.ticks_observed: int = 0
        self.out_of_order_observations: int = 0
        self.evictions: int = 0
        self.errors: int = 0
        self._last_observed_tick: Optional[int] = None
        self._closed: bool = False

        # Best-effort restore — any failure logs + continues with an
        # empty buffer.
        try:
            self._load_buffer()
        except Exception as e:  # noqa: BLE001
            _log.warning("outcome buffer restore failed: %s", e)

    # ── context manager ──────────────────────────────────────────────

    def __enter__(self) -> "OutcomeTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── public API ───────────────────────────────────────────────────

    def register_decision(
        self,
        record_id: str,
        tick_at_decision: int,
        snarc_at_decision: Dict[str, float],
    ) -> bool:
        """Begin tracking a decision's post-decision SNARC trajectory.

        Returns True if registered. Returns False if the tracker is
        closed or the record_id is already pending (we DO NOT overwrite
        — an existing pending outcome must close first).

        Never raises.
        """
        try:
            if self._closed:
                _log.warning("OutcomeTracker.register_decision after close; ignored")
                return False
            if not isinstance(record_id, str) or not record_id:
                _log.warning("register_decision: bad record_id %r", record_id)
                return False
            if record_id in self._pending:
                _log.warning("register_decision: %s already pending", record_id)
                return False
            snarc_clean = _coerce_snarc(snarc_at_decision)
            pending = _PendingOutcome(
                record_id=record_id,
                tick_at_decision=int(tick_at_decision),
                snarc_at_decision=snarc_clean,
            )
            self._pending[record_id] = pending
            self.decisions_registered += 1

            # Enforce cap — oldest evicted + flushed as incomplete.
            while len(self._pending) > self.max_pending:
                _, oldest = self._pending.popitem(last=False)
                self.evictions += 1
                self._emit_outcome(oldest, status=STATUS_INCOMPLETE,
                                   resolved_at_tick=oldest.tick_at_decision)
            return True
        except Exception as e:  # noqa: BLE001
            _log.warning("register_decision failed: %s", e)
            self.errors += 1
            return False

    def observe_tick(
        self,
        tick: int,
        snarc: Dict[str, float],
        rpe_signal: Optional[float] = None,
        level_up: Optional[bool] = None,
        failed_record_ids: Optional[Iterable[str]] = None,
    ) -> int:
        """Feed one post-decision tick's SNARC (and optional signals).

        Parameters
        ----------
        tick:
            Current consciousness-loop tick. Must be monotonically
            non-decreasing across calls — out-of-order ticks are
            ignored.
        snarc:
            Dict of SNARC scalar values (arousal/conflict/surprise/
            reward/novelty). Missing dims default to 0.0.
        rpe_signal:
            Optional scalar RPE observed this tick. Applied to ALL
            currently-pending outcomes whose decision tick < tick.
            Phase 4+ wiring.
        level_up:
            Optional bool; game-specific. Applied to all currently-
            pending outcomes. ``None`` leaves the field untouched so a
            later True observation wins.
        failed_record_ids:
            Optional iterable of record_ids whose dispatch failed this
            tick. Those records' outcomes will emit with
            ``status='failed'`` regardless of trajectory completeness.

        Returns
        -------
        int
            Number of outcomes emitted (window-closed or failed) on
            this tick.

        Never raises. Errors are counted + logged.
        """
        emitted = 0
        try:
            if self._closed:
                _log.warning("observe_tick after close; ignored")
                return 0
            # Monotonic tick check — allow ties (same tick, multiple
            # calls for different signals) but not regression.
            if (self._last_observed_tick is not None
                    and tick < self._last_observed_tick):
                _log.warning(
                    "observe_tick: out-of-order tick %d (last was %d); ignored",
                    tick, self._last_observed_tick,
                )
                self.out_of_order_observations += 1
                return 0
            self._last_observed_tick = int(tick)
            self.ticks_observed += 1

            snarc_clean = _coerce_snarc(snarc)
            failed = set(failed_record_ids) if failed_record_ids else set()

            # We iterate a snapshot of the keys; emission mutates
            # ``self._pending``.
            for record_id in list(self._pending.keys()):
                pending = self._pending.get(record_id)
                if pending is None:  # emitted by another branch this loop
                    continue

                # Plugin failure? Emit as failed right now.
                if record_id in failed:
                    pending.failure = True
                    self._pending.pop(record_id, None)
                    self._emit_outcome(pending, status=STATUS_FAILED,
                                       resolved_at_tick=int(tick))
                    emitted += 1
                    continue

                # Decision-tick sample doesn't count — trajectory is
                # STRICTLY post-decision (PRD §4.7.E).
                if tick <= pending.tick_at_decision:
                    continue

                # Gap too large? Flush with the samples we have.
                window_end = pending.tick_at_decision + TRAJECTORY_TICKS
                if tick > window_end:
                    # No new sample for this record — the window closed
                    # silently on an earlier tick that simply never
                    # arrived. Emit the (possibly partial) trajectory.
                    self._pending.pop(record_id, None)
                    status = (STATUS_COMPLETE
                              if len(pending.trajectory) >= TRAJECTORY_TICKS
                              else STATUS_INCOMPLETE)
                    self._emit_outcome(pending, status=status,
                                       resolved_at_tick=window_end)
                    emitted += 1
                    continue

                # Append this tick's SNARC to the trajectory. Cap at
                # TRAJECTORY_TICKS samples (defensive; shouldn't
                # trigger given the window_end check above).
                if len(pending.trajectory) < TRAJECTORY_TICKS:
                    pending.trajectory.append((int(tick), dict(snarc_clean)))
                # Record optional signals (last-write-wins for scalars,
                # OR-accumulate for level_up bool).
                if rpe_signal is not None:
                    try:
                        pending.rpe_signal = float(rpe_signal)
                    except (TypeError, ValueError):
                        pass
                if level_up is not None:
                    pending.level_up_observed = bool(
                        pending.level_up_observed or level_up
                    )

                # Window fully observed — emit now.
                if len(pending.trajectory) >= TRAJECTORY_TICKS:
                    self._pending.pop(record_id, None)
                    self._emit_outcome(pending, status=STATUS_COMPLETE,
                                       resolved_at_tick=int(tick))
                    emitted += 1

            return emitted
        except Exception as e:  # noqa: BLE001
            _log.warning("observe_tick failed: %s", e)
            self.errors += 1
            return emitted

    def flush(self) -> int:
        """Emit all remaining pending outcomes with ``incomplete`` status.

        Also persists the (now-empty) buffer to disk and flushes the
        sidecar writer. Returns the count of outcomes emitted.

        Never raises.
        """
        count = 0
        try:
            for record_id in list(self._pending.keys()):
                pending = self._pending.pop(record_id, None)
                if pending is None:
                    continue
                status = (STATUS_COMPLETE
                          if len(pending.trajectory) >= TRAJECTORY_TICKS
                          else STATUS_INCOMPLETE)
                resolved = (pending.tick_at_decision
                            + len(pending.trajectory))
                self._emit_outcome(pending, status=status,
                                   resolved_at_tick=resolved)
                count += 1
            self._writer.flush()
            # Persist empty buffer (truncates stale state from a prior
            # run). Best-effort.
            self._persist_buffer()
        except Exception as e:  # noqa: BLE001
            _log.warning("flush failed: %s", e)
            self.errors += 1
        return count

    def persist(self) -> None:
        """Persist the current pending buffer WITHOUT flushing outcomes.

        Useful for graceful shutdown paths that want to preserve
        pending state for the next kernel start. ``flush()`` and
        ``close()`` both persist afterwards — this is a separate
        method for callers that want to snapshot without draining.
        """
        try:
            self._persist_buffer()
        except Exception as e:  # noqa: BLE001
            _log.warning("persist failed: %s", e)
            self.errors += 1

    def close(self) -> None:
        """Flush pending, close the sidecar writer. Idempotent."""
        if self._closed:
            return
        try:
            self.flush()
        finally:
            try:
                self._writer.close()
            except Exception as e:  # noqa: BLE001
                _log.warning("outcome writer close failed: %s", e)
            self._closed = True

    # ── introspection ────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Snapshot of tracker counters for observability."""
        return {
            "decisions_registered": self.decisions_registered,
            "outcomes_emitted_complete": self.outcomes_emitted_complete,
            "outcomes_emitted_incomplete": self.outcomes_emitted_incomplete,
            "outcomes_emitted_failed": self.outcomes_emitted_failed,
            "ticks_observed": self.ticks_observed,
            "out_of_order_observations": self.out_of_order_observations,
            "evictions": self.evictions,
            "errors": self.errors,
            "pending": len(self._pending),
            "last_observed_tick": self._last_observed_tick,
            "machine": self.machine,
        }

    def pending_record_ids(self) -> List[str]:
        """Return record_ids still waiting for more observations."""
        return list(self._pending.keys())

    # ── internals ────────────────────────────────────────────────────

    def _emit_outcome(
        self,
        pending: _PendingOutcome,
        status: str,
        resolved_at_tick: int,
    ) -> None:
        """Build and write the final outcome dict for one pending entry.

        The outcome dict shape matches Track 6 deliverable:

            {
                "record_id": ...,                 # primary key for merge
                "outcome": {
                    "snarc_trajectory": [...],    # list of dicts
                    "snarc_resolution_score": ..., # PRD §4.7.E
                    "rpe_signal": ...,
                    "level_up_observed": ...,
                    "resolved_at_tick": ...,
                    "tick_at_decision": ...,
                    "status": "complete"|"incomplete"|"failed",
                    "outcome_schema_version": "v0.1.0",
                },
                "machine": ...,
                "emitted_at": ...,
            }

        Never raises.
        """
        try:
            trajectory_dicts = [
                {"tick": int(t), **_project_snarc(s)}
                for t, s in pending.trajectory
            ]
            score = _snarc_resolution_score(
                snarc_at_decision=pending.snarc_at_decision,
                trajectory=pending.trajectory,
            )
            outcome_dict: Dict[str, Any] = {
                "snarc_trajectory": trajectory_dicts,
                "snarc_resolution_score": score,
                "rpe_signal": pending.rpe_signal,
                "level_up_observed": pending.level_up_observed,
                "resolved_at_tick": int(resolved_at_tick),
                "tick_at_decision": int(pending.tick_at_decision),
                "status": status,
                "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
            }
            # If a failure was reported but the status wasn't set to
            # failed, reflect it in the payload so downstream readers
            # can still see it.
            if pending.failure and status != STATUS_FAILED:
                outcome_dict["plugin_failure_observed"] = True

            sidecar_record = {
                "record_id": pending.record_id,
                "outcome": outcome_dict,
                "machine": self.machine,
                "emitted_at": time.time(),
                "schema_version": OUTCOME_SCHEMA_VERSION,
            }
            ok = self._writer.append(sidecar_record)
            if not ok:
                self.errors += 1
                return

            if status == STATUS_COMPLETE:
                self.outcomes_emitted_complete += 1
            elif status == STATUS_FAILED:
                self.outcomes_emitted_failed += 1
            else:
                self.outcomes_emitted_incomplete += 1
        except Exception as e:  # noqa: BLE001
            _log.warning("emit_outcome for %s failed: %s",
                         getattr(pending, "record_id", "?"), e)
            self.errors += 1

    def _persist_buffer(self) -> None:
        """Write the current pending buffer to disk atomically.

        Uses the standard write-tmp-rename pattern. Any failure is
        logged + swallowed (the buffer is lost on next restart, which
        is the already-documented degradation path).
        """
        try:
            self._buffer_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
                "machine": self.machine,
                "persisted_at": time.time(),
                "pending": [p.to_json() for p in self._pending.values()],
                "last_observed_tick": self._last_observed_tick,
            }
            tmp_path = self._buffer_path.with_suffix(
                self._buffer_path.suffix + ".tmp"
            )
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, separators=(",", ":"), default=str)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except (OSError, AttributeError):
                    # fsync unsupported on some FS (e.g. WSL mount) —
                    # the atomic rename is still the critical step.
                    pass
            os.replace(tmp_path, self._buffer_path)
        except Exception as e:  # noqa: BLE001
            _log.warning("persist_buffer failed: %s", e)
            self.errors += 1

    def _load_buffer(self) -> None:
        """Restore the pending buffer from disk, if one exists.

        Best-effort. Stale entries (whose ``tick_at_decision`` is older
        than ``last_observed_tick`` by more than the trajectory window)
        are flushed immediately as ``incomplete`` so they don't clog
        the buffer after a long downtime.
        """
        if not self._buffer_path.exists():
            return
        try:
            with open(self._buffer_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:  # noqa: BLE001
            _log.warning("load_buffer: unreadable %s: %s", self._buffer_path, e)
            return

        last_tick = payload.get("last_observed_tick")
        try:
            last_tick_int = int(last_tick) if last_tick is not None else None
        except (TypeError, ValueError):
            last_tick_int = None
        self._last_observed_tick = last_tick_int

        pending_entries = payload.get("pending", [])
        for entry in pending_entries:
            try:
                pending = _PendingOutcome.from_json(entry)
            except Exception as e:  # noqa: BLE001
                _log.warning("load_buffer: bad pending entry: %s", e)
                continue

            # If the last observed tick has already moved past this
            # pending's window, emit incomplete now.
            if (last_tick_int is not None
                    and last_tick_int > pending.tick_at_decision + TRAJECTORY_TICKS):
                self._emit_outcome(
                    pending,
                    status=STATUS_INCOMPLETE,
                    resolved_at_tick=pending.tick_at_decision
                    + len(pending.trajectory),
                )
                continue

            self._pending[pending.record_id] = pending


# ──────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────


def _coerce_snarc(snarc: Any) -> Dict[str, float]:
    """Project whatever the caller passed into the canonical SNARC dict.

    Missing dims → 0.0. Non-numeric values → 0.0 with a debug log.
    Always returns a dict with ALL keys in ``ALL_SNARC_DIMS`` present.
    """
    out: Dict[str, float] = {dim: 0.0 for dim in ALL_SNARC_DIMS}
    if not isinstance(snarc, dict):
        return out
    for k in ALL_SNARC_DIMS:
        v = snarc.get(k, 0.0)
        try:
            out[k] = float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def _project_snarc(snarc: Dict[str, float]) -> Dict[str, float]:
    """Return the SNARC dict projected to ALL_SNARC_DIMS with floats."""
    return {k: float(snarc.get(k, 0.0) or 0.0) for k in ALL_SNARC_DIMS}


def _snarc_resolution_score(
    snarc_at_decision: Dict[str, float],
    trajectory: List[Tuple[int, Dict[str, float]]],
) -> Optional[float]:
    """Compute the PRD §4.7.E resolution score.

    Score = Δ (dominant SNARC dim) = snarc_at_decision[dom] - snarc_final[dom]

    Where ``dom`` is the argmax of ``snarc_at_decision`` restricted to
    ``RESOLUTION_DIMS`` (arousal/conflict/surprise) — the "dominant
    features" PRD §4.7.E names. Positive score = dimension was reduced
    after the decision (good). Negative = amplified.

    Returns ``None`` when:
      * the trajectory is empty (no post-decision SNARC observed), or
      * all decision-time SNARC values in ``RESOLUTION_DIMS`` are zero
        (no dominant feature — score is ill-defined).
    """
    if not trajectory:
        return None

    # Identify dominant dimension at decision time.
    best_dim: Optional[str] = None
    best_val: float = -1.0
    for dim in RESOLUTION_DIMS:
        v = float(snarc_at_decision.get(dim, 0.0) or 0.0)
        if v > best_val:
            best_val = v
            best_dim = dim
    if best_dim is None or best_val <= 0.0:
        return None

    # Use the LAST observed trajectory sample as the post-decision
    # value. "Resolution" is the delta at the end of the window, not
    # a per-step average — PRD §4.7.E language is "reduces arousal/
    # conflict/surprise" which we interpret as end-of-window delta.
    final_tick, final_snarc = trajectory[-1]
    final_val = float(final_snarc.get(best_dim, 0.0) or 0.0)
    return best_val - final_val


__all__ = [
    "OutcomeTracker",
    "OUTCOME_SCHEMA_VERSION",
    "TRAJECTORY_TICKS",
    "RESOLUTION_DIMS",
    "ALL_SNARC_DIMS",
    "STATUS_COMPLETE",
    "STATUS_INCOMPLETE",
    "STATUS_FAILED",
    "OUTCOME_FILE_PREFIX",
    "BUFFER_FILENAME",
]
