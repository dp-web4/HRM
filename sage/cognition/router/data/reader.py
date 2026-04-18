"""
RouterDatasetReader — replay reader for JSONL partitions.

Responsibilities:
  * Iterate records from plain `.jsonl` or gzipped `.jsonl.gz` files.
  * Tolerate schema drift: unknown schema_versions yield a warn log but
    still surface the record. Missing fields fall back to defaults from
    the version's known shape.
  * Read across partitions via glob patterns (machine + date range).
  * Handle partial / corrupt trailing lines without crashing — reads are
    best-effort; diagnostic info goes to a logger, not an exception.
  * Auto-merge Track 6 outcome sidecar partitions by ``record_id`` at
    read time. Main partitions remain append-only; outcomes live in
    parallel ``outcome_{date}.jsonl[.gz]`` files. Forensic readers that
    want pre-backfill state can opt out via ``merge_outcomes=False``.

Schema-version-awareness: the reader doesn't *migrate* old records to
new shapes (migrations are out of scope for Phase 0, per sprint doc).
It only ensures that consumers can read every record, see what version
it is, and fill defaults for fields the consumer expects but the record
lacks.
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


# Known schema versions the reader understands. Records stamped with a
# version OUTSIDE this set still yield; we just emit a warn so the
# operator can add the version to the registry when upstream evolves.
SUPPORTED_SCHEMA_VERSIONS: set = {"0.1.0"}


# Track 6 outcome sidecar: filename prefix and glob patterns. The reader
# auto-merges ``outcome_{date}.jsonl[.gz]`` into main-partition records
# when ``merge_outcomes=True`` (default). See module docstring.
OUTCOME_FILE_PREFIX: str = "outcome_"


# Default fill-ins for fields a v0.1.0 consumer might expect. Empty for
# now — Track 1 will populate this once RouterRecord solidifies. The
# slot is here so the hook point exists from day one.
_V010_DEFAULTS: Dict[str, Any] = {}


_log = logging.getLogger(__name__)


class RouterDatasetReader:
    """Replay reader for router dataset partitions.

    Parameters
    ----------
    base_dir:
        Root of the dataset layout — same directory the writer received.

    Methods
    -------
    read_file(path):
        Iterate records from a single file.
    read_partition(machine, date_range):
        Glob-style iterator across machines/dates.
    """

    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)

    # ── outcome sidecar helpers ───────────────────────────────────

    @staticmethod
    def _is_outcome_file(path: Path) -> bool:
        """True if ``path`` is an outcome sidecar partition."""
        return path.name.startswith(OUTCOME_FILE_PREFIX)

    # ── single-file iteration ─────────────────────────────────────

    def read_file(self, path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """Yield parsed dicts from one JSONL file.

        Gzip detection is by filename extension. A corrupt or truncated
        last line is logged and skipped; all prior valid lines still
        yield.

        Yields
        ------
        dict
            Schema-version-aware; missing fields filled per
            ``_V010_DEFAULTS``. See module docstring.
        """
        p = Path(path)
        if not p.exists():
            _log.warning("RouterDatasetReader: missing path %s", p)
            return

        opener: Any
        if p.suffix == ".gz":
            opener = lambda: gzip.open(str(p), "rt", encoding="utf-8")
        else:
            opener = lambda: open(p, "r", encoding="utf-8")

        try:
            with opener() as f:
                line_no = 0
                for raw in f:
                    line_no += 1
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        # Last-line corruption is expected after a crash
                        # mid-write. Log with enough detail to debug but
                        # continue — we want every complete record.
                        _log.warning(
                            "Skipping corrupt JSONL line in %s (line %d): %s",
                            p, line_no, e,
                        )
                        continue
                    yield self._hydrate(record)
        except (OSError, EOFError, gzip.BadGzipFile) as e:  # type: ignore[attr-defined]
            _log.warning("Error reading %s: %s", p, e)
        except Exception as e:  # pragma: no cover — defensive
            _log.exception("Unexpected error reading %s: %s", p, e)

    # ── partition iteration ───────────────────────────────────────

    def read_partition(
        self,
        machine: str = "*",
        date_range: Optional[Tuple[Union[str, date], Union[str, date]]] = None,
        merge_outcomes: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate records across multiple partition files.

        Parameters
        ----------
        machine:
            Machine name or ``"*"`` for all machines.
        date_range:
            Optional (start, end) inclusive, each as ``YYYY-MM-DD`` string
            or ``datetime.date``. None → no date filter.
        merge_outcomes:
            When True (default), records are merged with Track 6 outcome
            sidecar entries by ``record_id`` before being yielded. Set
            False for forensic replays that want the exact on-disk
            main-partition state (no post-hoc outcomes).

        Yields records in file-order (deterministic: sorted by path).
        """
        outcome_index: Dict[str, Dict[str, Any]] = {}
        if merge_outcomes:
            # Prefetch all outcome sidecars once for this query. Memory
            # footprint is bounded by the sampled partition size; the
            # outcome dict per record is small (one trajectory of 5
            # samples + a few scalars). This is the simpler correct
            # approach — if fleet-scale merges show pressure later, we
            # can switch to a per-partition date-aligned stream join.
            outcome_index = self._build_outcome_index(machine, date_range)

        for partition_path in self._resolve_partitions(machine, date_range):
            for record in self.read_file(partition_path):
                if merge_outcomes:
                    record = self._apply_outcome(record, outcome_index)
                yield record

    def list_partitions(
        self,
        machine: str = "*",
        date_range: Optional[Tuple[Union[str, date], Union[str, date]]] = None,
    ) -> List[Path]:
        """Return all (main) partition paths matching the filter, sorted.

        Outcome sidecar partitions are excluded. Use
        ``list_outcome_partitions`` to enumerate those separately.
        """
        return sorted(self._resolve_partitions(machine, date_range))

    def list_outcome_partitions(
        self,
        machine: str = "*",
        date_range: Optional[Tuple[Union[str, date], Union[str, date]]] = None,
    ) -> List[Path]:
        """Return all outcome sidecar partition paths matching the filter."""
        return sorted(self._resolve_outcome_partitions(machine, date_range))

    def read_outcomes(
        self,
        machine: str = "*",
        date_range: Optional[Tuple[Union[str, date], Union[str, date]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate raw outcome sidecar records (record_id + outcome dict).

        Useful for analytics / debugging. The normal training replay
        path should use ``read_partition(merge_outcomes=True)`` so the
        outcome arrives attached to its parent record.
        """
        for path in self._resolve_outcome_partitions(machine, date_range):
            yield from self.read_file(path)

    # ── internals ─────────────────────────────────────────────────

    def _resolve_partitions(
        self,
        machine: str,
        date_range: Optional[Tuple[Union[str, date], Union[str, date]]],
    ) -> List[Path]:
        """Glob + date filter. Returns sorted list (stable iteration).

        Outcome sidecar partitions (prefix ``outcome_``) are EXCLUDED.
        """
        return self._resolve_partitions_filtered(
            machine, date_range, outcome_only=False,
        )

    def _resolve_outcome_partitions(
        self,
        machine: str,
        date_range: Optional[Tuple[Union[str, date], Union[str, date]]],
    ) -> List[Path]:
        """Same filter as ``_resolve_partitions`` but only outcome files."""
        return self._resolve_partitions_filtered(
            machine, date_range, outcome_only=True,
        )

    def _resolve_partitions_filtered(
        self,
        machine: str,
        date_range: Optional[Tuple[Union[str, date], Union[str, date]]],
        outcome_only: bool,
    ) -> List[Path]:
        """Shared glob + date filter. ``outcome_only`` toggles which
        partition family is returned.
        """
        if not self.base_dir.exists():
            return []

        # Search both compressed and plain forms.
        patterns = [f"{machine}/*.jsonl", f"{machine}/*.jsonl.gz"]
        candidates: List[Path] = []
        for pat in patterns:
            candidates.extend(self.base_dir.glob(pat))

        # Split main from outcome sidecars.
        candidates = [
            p for p in candidates
            if self._is_outcome_file(p) == outcome_only
        ]

        if date_range is None:
            return sorted(candidates)

        start_d = _coerce_date(date_range[0])
        end_d = _coerce_date(date_range[1])

        def in_range(path: Path) -> bool:
            stem = path.name
            # Strip .jsonl or .jsonl.gz
            for suffix in (".jsonl.gz", ".jsonl"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            # Strip outcome_ prefix if present (for outcome partitions).
            if stem.startswith(OUTCOME_FILE_PREFIX):
                stem = stem[len(OUTCOME_FILE_PREFIX):]
            try:
                d = datetime.strptime(stem, "%Y-%m-%d").date()
            except ValueError:
                _log.debug("Skipping non-date-named file: %s", path)
                return False
            return start_d <= d <= end_d

        return sorted([p for p in candidates if in_range(p)])

    def _build_outcome_index(
        self,
        machine: str,
        date_range: Optional[Tuple[Union[str, date], Union[str, date]]],
    ) -> Dict[str, Dict[str, Any]]:
        """Load all outcome sidecar entries in the query range into a dict.

        Keyed by ``record_id``. On conflict (same record_id appears in
        multiple sidecars, e.g. a buggy retry) the LATER-emitted wins
        (by ``emitted_at``). Missing ``emitted_at`` sorts as 0, so a
        properly-timestamped later emission always overrides.
        """
        index: Dict[str, Dict[str, Any]] = {}
        timestamps: Dict[str, float] = {}
        for path in self._resolve_outcome_partitions(machine, date_range):
            for entry in self.read_file(path):
                rid = entry.get("record_id")
                if not rid:
                    continue
                ts = float(entry.get("emitted_at", 0.0) or 0.0)
                if rid in index and timestamps.get(rid, 0.0) >= ts:
                    continue
                index[rid] = entry
                timestamps[rid] = ts
        return index

    def _apply_outcome(
        self,
        record: Dict[str, Any],
        outcome_index: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge a sidecar outcome entry onto a record if present.

        Non-destructive: a copy is returned only when a merge is needed,
        otherwise the original dict is passed through. The ``outcome``
        field on the record wins if it is already populated and
        non-None — we never overwrite an existing outcome.
        """
        rid = record.get("record_id")
        if not rid:
            return record
        if record.get("outcome") is not None:
            return record
        sidecar = outcome_index.get(rid)
        if sidecar is None:
            return record
        merged = dict(record)
        merged["outcome"] = sidecar.get("outcome")
        return merged

    def _hydrate(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Fill defaults + log schema version mismatches."""
        version = record.get("schema_version")
        if version is None:
            # Records without schema_version are treated as the oldest
            # known version. Writer stamps it, so this should only occur
            # for records produced by an out-of-date writer.
            version = "0.1.0"
            record.setdefault("schema_version", version)

        if version not in SUPPORTED_SCHEMA_VERSIONS:
            # Forward-compat: a record written by a NEWER writer. We
            # surface it unchanged; consumer decides whether to care
            # about the extra fields. Emit at DEBUG to avoid log floods
            # during a rolling upgrade.
            _log.debug(
                "Router record has unknown schema_version %r; surfacing as-is",
                version,
            )

        # Fill v0.1.0 defaults for any expected field the record lacks.
        # _V010_DEFAULTS is the empty dict today; Track 1 may extend.
        for key, default in _V010_DEFAULTS.items():
            record.setdefault(key, default)

        return record


# ── module-level helpers ───────────────────────────────────────────


def _coerce_date(d: Union[str, date]) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(str(d), "%Y-%m-%d").date()
