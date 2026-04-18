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
    ) -> Iterator[Dict[str, Any]]:
        """Iterate records across multiple partition files.

        Parameters
        ----------
        machine:
            Machine name or ``"*"`` for all machines.
        date_range:
            Optional (start, end) inclusive, each as ``YYYY-MM-DD`` string
            or ``datetime.date``. None → no date filter.

        Yields records in file-order (deterministic: sorted by path).
        """
        for partition_path in self._resolve_partitions(machine, date_range):
            yield from self.read_file(partition_path)

    def list_partitions(
        self,
        machine: str = "*",
        date_range: Optional[Tuple[Union[str, date], Union[str, date]]] = None,
    ) -> List[Path]:
        """Return all partition paths matching the filter, sorted."""
        return sorted(self._resolve_partitions(machine, date_range))

    # ── internals ─────────────────────────────────────────────────

    def _resolve_partitions(
        self,
        machine: str,
        date_range: Optional[Tuple[Union[str, date], Union[str, date]]],
    ) -> List[Path]:
        """Glob + date filter. Returns sorted list (stable iteration)."""
        if not self.base_dir.exists():
            return []

        # Search both compressed and plain forms.
        patterns = [f"{machine}/*.jsonl", f"{machine}/*.jsonl.gz"]
        candidates: List[Path] = []
        for pat in patterns:
            candidates.extend(self.base_dir.glob(pat))

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
            try:
                d = datetime.strptime(stem, "%Y-%m-%d").date()
            except ValueError:
                _log.debug("Skipping non-date-named file: %s", path)
                return False
            return start_d <= d <= end_d

        return sorted([p for p in candidates if in_range(p)])

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
