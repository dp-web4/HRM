"""
RouterDatasetWriter — append-only JSONL with optional gzip.

Partition layout (per PRD §5.4, §5.5):

    {base_dir}/{machine}/{YYYY-MM-DD}.jsonl[.gz]

Design invariants:

1. **Append-only**: we never rewrite prior records. Compaction /
   migration is a separate offline concern.
2. **Failure isolation**: ANY exception raised during write is caught,
   logged, and swallowed. The consciousness loop must never die because
   the dataset partition filled up.
3. **Buffered writes**: records accumulate in an in-memory buffer and
   flush to disk in batches. Explicit `flush()` is always safe. `close()`
   flushes automatically.
4. **Gzip is per-file**, not per-record. The partition filename ends in
   `.jsonl.gz` when compression is on. A day's file is either fully
   compressed or fully plain; a daemon restart that flipped the flag
   would start a new partition on the next calendar day anyway (because
   the filename differs), so we never face a mixed file.
5. **JSON-only**: records are plain dicts. No dataclass smuggling. If a
   caller hands us a dataclass-with-`to_dict`, we call it; otherwise we
   try `json.dumps` directly and fail cleanly.

The writer is deliberately boring. Every clever optimization it might
want belongs in a later refactor with actual data in hand.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union


# Schema version the writer stamps on records missing a `schema_version`
# field. Readers must tolerate unknown versions — see reader.py.
SCHEMA_VERSION: str = "0.1.0"


# Module-level logger so tests can capture messages. Callers are free to
# attach handlers; default is the root configuration.
_log = logging.getLogger(__name__)


class RouterDatasetWriter:
    """Append-only JSONL writer with optional gzip and failure isolation.

    Parameters
    ----------
    base_dir:
        Root of the dataset layout. Per-machine subdir is created here.
    machine:
        Machine name — used as a subdirectory. Free-form but should match
        fleet manifest (sprout, legion, thor, ...).
    compress:
        If True (default), write ``.jsonl.gz`` files using stdlib gzip; else
        ``.jsonl``. Default is True because plain JSONL at fleet scale
        (~103 MB/day/machine) crosses Track 7's 100 MB/day alert threshold;
        gzip compresses ~70× on synthetic data and substantially less but
        still meaningful on real records. Pass ``compress=False`` for
        interactive debugging only.
    buffer_size:
        Number of records to accumulate before an automatic flush. Smaller
        → more disk syncs, larger → bigger window of in-memory data lost
        on crash. 64 is a deliberate middle — it matches roughly one
        second of fleet traffic at the planned ingestion rate.
    clock:
        Optional callable returning current UTC datetime. Tests inject a
        fixed clock; production leaves it None (uses `datetime.now(UTC)`).

    Notes
    -----
    * Partition rollover is by UTC date. A daemon running through local
      midnight but not UTC midnight will NOT roll over; this is
      intentional (UTC-based logs are globally comparable across fleet).
    * On partition rollover, the previous file handle is closed cleanly
      before the new one opens. If the rollover itself raises, we log
      and continue — the next `append()` will retry.
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        machine: str,
        compress: bool = True,
        buffer_size: int = 64,
        clock: Optional[Any] = None,
        subdir: Optional[str] = None,
    ):
        if not machine or "/" in machine or "\\" in machine:
            raise ValueError(f"machine must be a simple name, got {machine!r}")
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
        if subdir is not None and ("/" in subdir or "\\" in subdir):
            raise ValueError(f"subdir must be a simple name, got {subdir!r}")

        self.base_dir = Path(base_dir)
        self.machine = machine
        self.compress = compress
        self.buffer_size = buffer_size
        self.subdir = subdir
        self._clock = clock or (lambda: datetime.now(timezone.utc))

        self._buffer: List[str] = []
        self._current_path: Optional[Path] = None
        self._current_handle: Optional[Any] = None
        self._closed = False

        # Stats
        self.records_written: int = 0
        self.records_dropped: int = 0
        self.flushes: int = 0

    # ── context manager ───────────────────────────────────────────

    def __enter__(self) -> "RouterDatasetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── public API ────────────────────────────────────────────────

    def append(self, record: Any) -> bool:
        """Queue a record for writing.

        Returns True if queued successfully, False if dropped. NEVER
        raises — per PRD §5.5 failure-isolation mandate.

        Accepted shapes:
          * plain dict (preferred)
          * dataclass-like with ``to_dict``
          * dataclass decorated with ``@dataclass`` (uses ``asdict``)

        Any other type → dropped with a warn log.
        """
        if self._closed:
            _log.warning("RouterDatasetWriter.append called after close; dropping")
            self.records_dropped += 1
            return False

        try:
            serialized = self._serialize(record)
        except Exception as e:  # pragma: no cover — defensive
            _log.exception("Failed to serialize router record: %s", e)
            self.records_dropped += 1
            return False

        self._buffer.append(serialized)
        self.records_written += 1
        if len(self._buffer) >= self.buffer_size:
            self.flush()
        return True

    def flush(self) -> None:
        """Write buffered records to disk. Never raises."""
        if not self._buffer:
            return
        try:
            self._ensure_partition()
            if self._current_handle is None:
                return
            # Single write call — gzip and text handles both accept str.
            self._current_handle.write("".join(self._buffer))
            self._current_handle.flush()
            self._buffer.clear()
            self.flushes += 1
        except Exception as e:
            _log.exception("RouterDatasetWriter flush failed: %s", e)
            # Buffer retained for retry on next flush / close.

    def close(self) -> None:
        """Flush buffer and close current file handle. Safe to call twice."""
        if self._closed:
            return
        try:
            self.flush()
        finally:
            try:
                if self._current_handle is not None:
                    self._current_handle.close()
            except Exception as e:  # pragma: no cover
                _log.exception("Error closing writer handle: %s", e)
            self._current_handle = None
            self._current_path = None
            self._closed = True

    # ── introspection ─────────────────────────────────────────────

    def current_path(self) -> Optional[Path]:
        """Return the currently-open partition path (None if not opened yet)."""
        return self._current_path

    def get_stats(self) -> Dict[str, Any]:
        return {
            "records_written": self.records_written,
            "records_dropped": self.records_dropped,
            "flushes": self.flushes,
            "buffered": len(self._buffer),
            "current_path": str(self._current_path) if self._current_path else None,
            "compress": self.compress,
            "machine": self.machine,
        }

    # ── internals ─────────────────────────────────────────────────

    def _serialize(self, record: Any) -> str:
        """Normalize `record` to a dict, stamp schema_version if missing,
        and return a JSON line with trailing newline.
        """
        if isinstance(record, Mapping):
            d = dict(record)
        elif hasattr(record, "to_dict") and callable(record.to_dict):
            # TODO: replace _RouterRecordStub with
            # sage.cognition.router.RouterRecord once Track 1 merges.
            d = record.to_dict()
        elif is_dataclass(record):
            d = asdict(record)
        else:
            raise TypeError(
                f"router record must be dict, dataclass, or have to_dict(); "
                f"got {type(record).__name__}"
            )

        if "schema_version" not in d:
            d["schema_version"] = SCHEMA_VERSION
        if "timestamp" not in d:
            d["timestamp"] = time.time()
        if not d.get("machine"):
            d["machine"] = self.machine

        return json.dumps(d, separators=(",", ":"), default=str) + "\n"

    def _ensure_partition(self) -> None:
        """Open today's partition file if not already open, or roll over
        on UTC date boundary.
        """
        today = self._clock().strftime("%Y-%m-%d")
        ext = ".jsonl.gz" if self.compress else ".jsonl"
        machine_dir = self.base_dir / self.machine
        if self.subdir:
            machine_dir = machine_dir / self.subdir
        partition = machine_dir / f"{today}{ext}"

        if self._current_path == partition and self._current_handle is not None:
            return  # Still the right file.

        # Close old handle before opening new one.
        if self._current_handle is not None:
            try:
                self._current_handle.close()
            except Exception as e:  # pragma: no cover
                _log.exception("Error closing previous partition: %s", e)
            self._current_handle = None
            self._current_path = None

        try:
            partition.parent.mkdir(parents=True, exist_ok=True)
            if self.compress:
                # gzip append mode + text wrapper. `ab` + GzipFile is
                # slightly finicky; stdlib `gzip.open('at')` handles it.
                handle: Any = gzip.open(str(partition), "at", encoding="utf-8")
            else:
                handle = open(partition, "a", encoding="utf-8")
            self._current_handle = handle
            self._current_path = partition
        except Exception as e:
            _log.exception("Failed to open partition %s: %s", partition, e)
            self._current_handle = None
            self._current_path = None
