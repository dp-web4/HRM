"""
RouterDatasetPruner — SNARC-driven rolling-window retention.
==============================================================

Implements PRD §5.6 retention policy:

    | Age         | Retention rule                                        |
    |-------------|-------------------------------------------------------|
    | 0-7 days    | Keep everything that passed sampling                  |
    | 7-30 days   | Keep top 2 SNARC quintiles (60% of sampled)           |
    | 30-90 days  | Keep top quintile only (20% of sampled)               |
    | 90+ days    | Keep only `pinned` records                            |

Two orthogonal axes govern the dataset per §4.7.D + §5.6:

  * **Sampling** (write-time, Track 4): controls which ticks ever land on
    disk. High-SNARC kept at 100%, low-SNARC downsampled to ~5%.
  * **Pruning** (read-time, this module): controls how long each record
    lives. Low-SNARC records age out; high-SNARC records stay.

Composed, the corpus grows roughly with novelty, not wall time.

Design choices (documented here because PRD §5.6 is somewhat silent):

* **Per-partition quintile boundaries** (NOT global). Each partition is a
  self-describing UTC-day file: salience quintiles are computed from the
  partition's own distribution. Reasons:

    1. Partition files are the unit of ownership, replication, and backup.
       A partition that can be pruned with its own data is portable across
       machines and robust to missing peers.
    2. Global quintiles would require a shared salience baseline that
       drifts as the fleet changes — a subtle coupling that we deliberately
       avoid in Phase 0.
    3. If the fleet ever needs cross-partition analysis, the salience
       histogram is captured in `PruneStats` — downstream tools can
       aggregate without the pruner needing a global view.

* **Agent-zero defense**: before rewriting, we compute the modal-class
  action-count on the would-be-kept records. If the modal-class score
  margin (modal - next-most) drops below the PRD §7.10 25-percentage-point
  threshold, we abort the prune with a warning. Pruning that collapses
  the dataset toward the majority class is the same hazard as the
  original Agent Zero story — a dataset that trains a router to "always
  noop" would pass naive metrics and fail in reality.

* **Concurrent safety** with the writer:

    1. We refuse to touch a file whose date stem == today (UTC).
    2. We refuse to touch a file whose mtime is within the last 24h —
       even on backfill, an active writer appending to an old file would
       be corrupted by a compacting rewrite.
    3. We respect an explicit `.lock` sidecar — if `{path}.lock` exists,
       the writer claims the file. Pruner skips with a clear event.
    4. Atomic rewrite via `.tmp` + `os.replace`. If we crash mid-rewrite,
       the original file is preserved and the next pass retries.

* **Pinned records** are exempt at every age bracket. The `pinned` flag
  lives inside `record.payload` (not a separate index), so partitions
  remain self-describing. Recognized pin kinds (PRD §5.6):

      - `agent_zero_golden`   — canon used by §7.10 evaluation
      - `training_canon`      — adapter training anchor set
      - `manual_review`       — human flag

  Pins with unknown kind are still preserved (fail-safe).

* **Failure isolation**: `prune_all` catches every exception per-partition
  and continues. A single bad file must never stall a nightly cron.

* **No torch** — pure stdlib.

Spec: phase2/brain-arch/router-sprint-1-phase-0.md Track 9
      phase2/brain-arch/thalamic-router-prd.md §5.6, §7.10
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from sage.cognition.router.data.sampling import salience_score


_log = logging.getLogger(__name__)


# ── defaults, all tunable but anchored to PRD spec ──────────────────

# Age brackets from PRD §5.6 (inclusive lower, exclusive upper).
# (lower_days, upper_days_or_None, min_quintile_to_keep_or_pinned_only)
#
# `min_quintile_to_keep` is the lowest quintile (0-indexed) that survives
# this bracket. For 90+ days, the convention is `None` meaning "pinned
# only". Using integers here keeps the table declarative.
AGE_BRACKETS: List[Tuple[int, Optional[int], Optional[int]]] = [
    (0, 7, 0),       # 0-7 days: keep everything (quintile >= 0)
    (7, 30, 3),      # 7-30 days: keep top 2 quintiles (quintile >= 3)
    (30, 90, 4),     # 30-90 days: keep top quintile only (quintile >= 4)
    (90, None, None),  # 90+ days: pinned only
]

# PRD §7.10 — margin-over-modal-class dummy threshold, in percentage
# points. A pruning pass that would leave a partition whose modal-class
# score doesn't exceed the next-most-common class by this margin indicates
# dataset collapse toward majority class.
AGENT_ZERO_MARGIN_PP: float = 25.0

# Files whose mtime is within this window are treated as possibly-being-
# written-to and NEVER touched. 24h covers UTC-date-boundary straddles and
# gives the writer room to finalize on kernel restart.
ACTIVE_WRITE_WINDOW_SECONDS: int = 24 * 60 * 60

# Recognized pin kinds (PRD §5.6). Unknown kinds are still preserved —
# this set is used only for reporting.
RECOGNIZED_PIN_KINDS: Tuple[str, ...] = (
    "agent_zero_golden",
    "training_canon",
    "manual_review",
)

# Schema version the pruner understands. Bumped in lockstep with the
# dataset writer. Records with unknown versions are preserved with a
# debug log.
PRUNER_VERSION: str = "0.1.0"


# ── data classes ────────────────────────────────────────────────────


@dataclass
class PruneStats:
    """Per-partition pruning statistics.

    Includes before/after histograms and the agent-zero decision. These
    fields are JSON-serializable so nightly runs can log them to
    observability surfaces.
    """

    path: str
    age_days: int
    bracket_rule: str           # human-readable bracket, e.g. "7-30d top2quintiles"
    records_before: int = 0
    records_after: int = 0
    pinned_preserved: int = 0
    pinned_by_kind: Dict[str, int] = field(default_factory=dict)
    # Salience histogram: 5 bins matching quintiles 0-4. Before-prune and
    # after-prune so the shape change is visible at a glance.
    salience_hist_before: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    salience_hist_after: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    quintile_boundaries: List[float] = field(default_factory=list)
    bytes_before: int = 0
    bytes_after: int = 0
    # Agent-zero check (PRD §7.10): "ok" / "skipped_would_collapse" /
    # "not_applicable" (too few classes to compute a margin, e.g. single-
    # class partition — pruning doesn't change the dummy so the check
    # can't fire).
    agent_zero_check: str = "ok"
    agent_zero_modal_before: Optional[str] = None
    agent_zero_margin_before_pp: Optional[float] = None
    agent_zero_margin_after_pp: Optional[float] = None
    rewrote: bool = False       # True if file was rewritten this pass
    skipped_reason: Optional[str] = None  # e.g. "active_write", "dry_run", ...
    dry_run: bool = False
    error: Optional[str] = None

    @property
    def bytes_reclaimed(self) -> int:
        return max(0, self.bytes_before - self.bytes_after)

    @property
    def records_dropped(self) -> int:
        return max(0, self.records_before - self.records_after)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "age_days": self.age_days,
            "bracket_rule": self.bracket_rule,
            "records_before": self.records_before,
            "records_after": self.records_after,
            "records_dropped": self.records_dropped,
            "pinned_preserved": self.pinned_preserved,
            "pinned_by_kind": dict(self.pinned_by_kind),
            "salience_hist_before": list(self.salience_hist_before),
            "salience_hist_after": list(self.salience_hist_after),
            "quintile_boundaries": list(self.quintile_boundaries),
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "bytes_reclaimed": self.bytes_reclaimed,
            "agent_zero_check": self.agent_zero_check,
            "agent_zero_modal_before": self.agent_zero_modal_before,
            "agent_zero_margin_before_pp": self.agent_zero_margin_before_pp,
            "agent_zero_margin_after_pp": self.agent_zero_margin_after_pp,
            "rewrote": self.rewrote,
            "skipped_reason": self.skipped_reason,
            "dry_run": self.dry_run,
            "error": self.error,
        }


# ── helpers ─────────────────────────────────────────────────────────


def _extract_snarc(record: Mapping[str, Any]) -> Mapping[str, float]:
    """Locate the SNARC dict wherever it lives in the record.

    Tolerant of both layouts:
      * Track 4 envelope: ``record["payload"]["router_input"]["snarc"]``
      * Track 1 RouterRecord: ``record["router_input"]["snarc"]``

    Empty dict fallback lets pruning proceed (record lands in quintile 0
    via salience_score=0) rather than crashing on a malformed record.
    """
    payload = record.get("payload")
    if isinstance(payload, Mapping):
        ri = payload.get("router_input")
        if isinstance(ri, Mapping):
            snarc = ri.get("snarc")
            if isinstance(snarc, Mapping):
                return snarc
    ri = record.get("router_input")
    if isinstance(ri, Mapping):
        snarc = ri.get("snarc")
        if isinstance(snarc, Mapping):
            return snarc
    snarc = record.get("snarc")
    if isinstance(snarc, Mapping):
        return snarc
    return {}


def _extract_action(record: Mapping[str, Any]) -> Optional[str]:
    """Best-effort decision-class extraction for the agent-zero check.

    Returns None if the record has no discernible action, in which case
    the record doesn't contribute to modal-class computation.
    """
    payload = record.get("payload")
    if isinstance(payload, Mapping):
        ro = payload.get("router_output")
        if isinstance(ro, Mapping):
            action = ro.get("action")
            if isinstance(action, str):
                return action
    ro = record.get("router_output")
    if isinstance(ro, Mapping):
        action = ro.get("action")
        if isinstance(action, str):
            return action
    action = record.get("action")
    if isinstance(action, str):
        return action
    return None


def _extract_pinned(record: Mapping[str, Any]) -> Optional[str]:
    """Extract the ``pinned`` flag.

    Returns:
      - None if record is NOT pinned.
      - str kind if pinned (e.g. "agent_zero_golden"); truthy-non-str
        values (True / 1) coerce to the literal "pinned" so the record
        is preserved even if the producer didn't name the kind.

    Canonical location: ``record["payload"]["pinned"]`` per the mission
    spec. Also accepts top-level ``record["pinned"]`` as a fallback for
    producers that write directly at the record root.
    """
    payload = record.get("payload")
    pinned: Any = None
    if isinstance(payload, Mapping) and "pinned" in payload:
        pinned = payload.get("pinned")
    elif "pinned" in record:
        pinned = record.get("pinned")
    if not pinned:
        return None
    if isinstance(pinned, str):
        return pinned
    # Truthy non-str (bool True, int 1, etc.) — preserve but name generically.
    return "pinned"


def _quintile_boundaries(scores: List[float]) -> List[float]:
    """Return [q20, q40, q60, q80] boundaries for the given score list.

    Mirrors ``SnarcStratifiedSampler.quintile_boundaries``: strict-less
    placement, so ties land in the lower quintile. Empty / tiny lists
    collapse to all-zeros, which means every record lands in quintile 0.
    """
    if len(scores) < 5:
        return [0.0, 0.0, 0.0, 0.0]
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    return [
        sorted_scores[int(n * 0.20)],
        sorted_scores[int(n * 0.40)],
        sorted_scores[int(n * 0.60)],
        sorted_scores[int(n * 0.80)],
    ]


def _quintile_of(score: float, boundaries: List[float]) -> int:
    """Map a score → quintile index 0-4 using partition boundaries."""
    for i, cut in enumerate(boundaries):
        if score < cut:
            return i
    return 4


def _parse_partition_date(path: Path) -> Optional[date]:
    """Extract YYYY-MM-DD from `{date}.jsonl` or `{date}.jsonl.gz`.

    Returns None on non-matching names — the pruner will skip with a
    clear reason rather than crash on stray files in the directory.
    """
    stem = path.name
    for suffix in (".jsonl.gz", ".jsonl"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    try:
        return datetime.strptime(stem, "%Y-%m-%d").date()
    except ValueError:
        return None


def _bracket_for_age(age_days: int) -> Tuple[int, Optional[int], Optional[int], str]:
    """Return (lower, upper, min_quintile_or_None, human_label) for an age."""
    for lower, upper, min_q in AGE_BRACKETS:
        if upper is None:
            if age_days >= lower:
                return lower, upper, min_q, f"{lower}d+ pinned-only"
        elif lower <= age_days < upper:
            if min_q is None:
                return lower, upper, min_q, f"{lower}-{upper}d pinned-only"
            keep_count = 5 - min_q
            if keep_count >= 5:
                return lower, upper, min_q, f"{lower}-{upper}d keep-all"
            return lower, upper, min_q, f"{lower}-{upper}d top-{keep_count}-quintiles"
    # Defensive fallback — should never hit given AGE_BRACKETS covers all.
    return AGE_BRACKETS[-1][0], AGE_BRACKETS[-1][1], AGE_BRACKETS[-1][2], "fallback"


def _open_read(path: Path):
    """Open partition for text read, transparent over `.gz`."""
    if path.suffix == ".gz":
        return gzip.open(str(path), "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _open_write(path: Path, compress: bool):
    """Open partition for text write, transparent over `.gz`."""
    if compress:
        return gzip.open(str(path), "wt", encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def _class_margin_pp(class_counts: Mapping[str, int]) -> Tuple[Optional[str], Optional[float]]:
    """Return (modal_class, margin_pp) — margin between top and next-most class.

    A partition with exactly one class has no "next-most" class; we return
    ``(modal, None)`` — callers interpret None as "not applicable" and
    skip the check.
    """
    if not class_counts:
        return None, None
    total = sum(class_counts.values())
    if total <= 0:
        return None, None
    sorted_counts = sorted(class_counts.items(), key=lambda kv: -kv[1])
    modal_class, modal_count = sorted_counts[0]
    if len(sorted_counts) == 1:
        return modal_class, None
    next_count = sorted_counts[1][1]
    margin = (modal_count - next_count) / total * 100.0
    return modal_class, margin


# ── pruner ──────────────────────────────────────────────────────────


class RouterDatasetPruner:
    """Nightly pruner for router dataset partitions.

    Parameters
    ----------
    clock:
        Optional callable returning current UTC datetime. Tests inject a
        fixed clock; production uses ``datetime.now(UTC)``.
    agent_zero_margin_pp:
        Agent-zero threshold in percentage points. Defaults to PRD §7.10
        value (25pp). Operator-tunable for experimentation; production
        must not deviate without cross-reference to PRD.
    active_write_window_seconds:
        How long after the last write (by mtime) a partition is treated
        as "possibly being written to". Defaults to 24h.
    age_brackets:
        Override the age → retention table. Only touch this for
        simulation / tests — the defaults are binding per PRD §5.6.

    Notes
    -----
    * The pruner is stateless between calls. All decisions are derived
      from the partition being pruned + the clock.
    * ``prune_partition`` never raises; failures surface as
      ``PruneStats.error`` and ``skipped_reason``.
    * ``prune_all`` iterates with failure-isolation: one bad partition
      never stalls the rest.
    """

    def __init__(
        self,
        clock: Optional[Any] = None,
        agent_zero_margin_pp: float = AGENT_ZERO_MARGIN_PP,
        active_write_window_seconds: int = ACTIVE_WRITE_WINDOW_SECONDS,
        age_brackets: Optional[List[Tuple[int, Optional[int], Optional[int]]]] = None,
    ):
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.agent_zero_margin_pp = float(agent_zero_margin_pp)
        self.active_write_window_seconds = int(active_write_window_seconds)
        self.age_brackets = age_brackets or AGE_BRACKETS

    # ── public API ──────────────────────────────────────────────────

    def prune_partition(
        self,
        path: Union[str, Path],
        dry_run: bool = False,
    ) -> PruneStats:
        """Re-read, filter by age + SNARC, rewrite compacted partition.

        Returns a ``PruneStats`` capturing what happened. Never raises —
        errors are reported in-band via ``PruneStats.error`` /
        ``skipped_reason``.

        Idempotency contract: re-running on an already-pruned partition
        is a no-op. Each pass recomputes quintiles from the surviving
        distribution; by construction the kept records in an age
        bracket's top-N quintiles still occupy the top-N quintiles
        after re-quintilization.
        """
        p = Path(path)
        now = self._clock()
        today = now.date() if isinstance(now, datetime) else now

        stats = PruneStats(path=str(p), age_days=-1, bracket_rule="unknown", dry_run=dry_run)

        if not p.exists():
            stats.skipped_reason = "missing"
            stats.error = f"path does not exist: {p}"
            _log.warning("Pruner: %s", stats.error)
            return stats

        partition_date = _parse_partition_date(p)
        if partition_date is None:
            stats.skipped_reason = "non_partition_filename"
            stats.error = f"filename does not match YYYY-MM-DD: {p.name}"
            _log.info("Pruner skip: %s", stats.error)
            return stats

        age_days = (today - partition_date).days
        stats.age_days = age_days

        # Current-day guard. Writer actively appends; never touch.
        if age_days <= 0:
            stats.skipped_reason = "current_day"
            stats.bracket_rule = "current-day (skip)"
            _log.info("Pruner skip: %s is current day (age=%d)", p, age_days)
            return stats

        # Lock-file convention: if {path}.lock sits next to the file,
        # the writer claims it. Skip.
        lock = p.with_suffix(p.suffix + ".lock")
        if lock.exists():
            stats.skipped_reason = "locked"
            stats.bracket_rule = "locked (skip)"
            _log.info("Pruner skip: %s has active lock", p)
            return stats

        # mtime guard: even for an old-dated partition, if a writer is
        # actively appending to it (e.g. late backfill), don't rewrite.
        try:
            mtime = p.stat().st_mtime
        except OSError as e:
            stats.skipped_reason = "stat_failed"
            stats.error = f"stat failed: {e}"
            _log.warning("Pruner: %s", stats.error)
            return stats
        now_epoch = now.timestamp() if isinstance(now, datetime) else time.time()
        if (now_epoch - mtime) < self.active_write_window_seconds:
            stats.skipped_reason = "active_write"
            stats.bracket_rule = f"mtime-within-{self.active_write_window_seconds}s (skip)"
            _log.info(
                "Pruner skip: %s mtime too recent (%.0fs ago < %ds)",
                p, now_epoch - mtime, self.active_write_window_seconds,
            )
            return stats

        lower, upper, min_quintile, bracket_rule = _bracket_for_age(age_days)
        stats.bracket_rule = bracket_rule

        # Stage 1: load all records (single pass; partitions are bounded by
        # per-day size, so in-memory is fine. If this breaks in the future,
        # streaming with two passes is the obvious refactor.)
        try:
            records, raw_lines = self._read_records(p)
        except Exception as e:
            stats.skipped_reason = "read_failed"
            stats.error = f"read failed: {e}"
            _log.exception("Pruner: %s", stats.error)
            return stats

        try:
            stats.bytes_before = p.stat().st_size
        except OSError:
            stats.bytes_before = sum(len(l.encode("utf-8")) for l in raw_lines)

        stats.records_before = len(records)
        if not records:
            stats.skipped_reason = "empty_partition"
            _log.info("Pruner skip: %s has no valid records", p)
            return stats

        # Stage 2: compute per-partition quintile boundaries.
        #
        # Idempotency is structural: the FIRST time a partition is seen,
        # we compute boundaries from its records and write a `.meta.json`
        # sidecar next to the partition. Every subsequent pass reads the
        # sidecar and uses the SAME boundaries, so repeat prunes within
        # the same bracket produce the same keep/drop decisions. A
        # partition that has already been pruned to "top 2 quintiles"
        # under the original boundaries has NO records below that cutoff
        # to drop on the next pass — even though re-quintilizing the
        # shrunken distribution would create a fresh quintile 0.
        #
        # This keeps the "partition is self-describing" property: the
        # boundaries live beside the partition, not in a global index.
        saliences: List[float] = [salience_score(_extract_snarc(r)) for r in records]
        meta = self._load_or_init_meta(p, saliences, len(records))
        boundaries = list(meta["quintile_boundaries"])
        stats.quintile_boundaries = boundaries

        for s in saliences:
            stats.salience_hist_before[_quintile_of(s, boundaries)] += 1

        # Pre-prune class distribution for the agent-zero check.
        before_classes: Dict[str, int] = {}
        for r in records:
            a = _extract_action(r)
            if a is not None:
                before_classes[a] = before_classes.get(a, 0) + 1
        modal_before, margin_before = _class_margin_pp(before_classes)
        stats.agent_zero_modal_before = modal_before
        stats.agent_zero_margin_before_pp = margin_before

        # Stage 3: apply retention rule. Pinned records ALWAYS survive.
        keep_records: List[Dict[str, Any]] = []
        keep_lines: List[str] = []
        for rec, line, sal in zip(records, raw_lines, saliences):
            pin = _extract_pinned(rec)
            if pin is not None:
                keep_records.append(rec)
                keep_lines.append(line)
                stats.pinned_preserved += 1
                key = pin if pin in RECOGNIZED_PIN_KINDS else (
                    "pinned_other" if pin == "pinned" else pin
                )
                stats.pinned_by_kind[key] = stats.pinned_by_kind.get(key, 0) + 1
                continue
            if min_quintile is None:
                # 90+ days: pinned-only — this non-pinned record drops.
                continue
            q = _quintile_of(sal, boundaries)
            if q >= min_quintile:
                keep_records.append(rec)
                keep_lines.append(line)

        # Stage 4: agent-zero check on the proposed-kept set.
        after_classes: Dict[str, int] = {}
        for r in keep_records:
            a = _extract_action(r)
            if a is not None:
                after_classes[a] = after_classes.get(a, 0) + 1
        modal_after, margin_after = _class_margin_pp(after_classes)
        stats.agent_zero_margin_after_pp = margin_after

        # Agent-zero decision (PRD §5.6 + §7.10):
        #
        # Fire ONLY when the prune is responsible for pushing the margin
        # below the threshold. Two cases:
        #
        #   (a) Pre-prune had ≥2 classes with a healthy margin
        #       (>= threshold) and post-prune has margin < threshold
        #       (or collapsed to a single class entirely).
        #
        #   (b) Pre-prune had ≥2 classes but margin already < threshold:
        #       the data was already collapsed by upstream sampling, not
        #       by pruning. Pruning is not responsible. Mark "ok" so the
        #       pass proceeds — the defense exists to prevent pruning
        #       from *causing* collapse, not to freeze every imbalanced
        #       dataset.
        #
        # The single-class-before case (margin_before is None) cannot be
        # made worse by pruning, so we mark not_applicable.
        if margin_before is None:
            stats.agent_zero_check = "not_applicable"
        else:
            collapse = False
            if margin_before >= self.agent_zero_margin_pp:
                # Pre-prune was above threshold; check whether prune dropped us below.
                if margin_after is None:
                    collapse = True  # lost all non-modal classes
                elif margin_after < self.agent_zero_margin_pp:
                    collapse = True
            # If margin_before < threshold, the data was already collapsed;
            # pruning didn't cause it. Don't block the pass.
            if collapse:
                stats.agent_zero_check = "skipped_would_collapse"
                stats.skipped_reason = "agent_zero_collapse"
                _log.warning(
                    "Pruner AGENT-ZERO SKIP: %s — margin before=%.2fpp after=%s < threshold=%.1fpp",
                    p, margin_before,
                    "None" if margin_after is None else f"{margin_after:.2f}pp",
                    self.agent_zero_margin_pp,
                )
                # Revert: keep the original distribution's histogram "after"
                # equal to "before" since we're not rewriting.
                stats.salience_hist_after = list(stats.salience_hist_before)
                stats.records_after = stats.records_before
                stats.bytes_after = stats.bytes_before
                return stats
            stats.agent_zero_check = "ok"

        # Populate after-histogram from the kept records.
        for rec, sal in zip(keep_records, [salience_score(_extract_snarc(r)) for r in keep_records]):
            stats.salience_hist_after[_quintile_of(sal, boundaries)] += 1
        stats.records_after = len(keep_records)

        # Stage 5: atomic rewrite (unless dry-run).
        if dry_run:
            stats.dry_run = True
            stats.skipped_reason = "dry_run"
            # Estimate bytes_after from the kept lines.
            stats.bytes_after = self._estimate_bytes(keep_lines, compress=(p.suffix == ".gz"))
            return stats

        # Short-circuit: if nothing changed, don't rewrite. This is
        # critical for idempotency — running prune on already-pruned
        # data must not churn files (or change mtime and thereby mark
        # the file as "active write").
        if len(keep_records) == len(records):
            stats.bytes_after = stats.bytes_before
            # Histogram may differ only if the record set is literally
            # identical — which it is in this branch.
            return stats

        try:
            self._atomic_rewrite(p, keep_lines)
        except Exception as e:
            stats.error = f"rewrite failed: {e}"
            stats.skipped_reason = "rewrite_failed"
            _log.exception("Pruner rewrite failed: %s", p)
            return stats

        try:
            stats.bytes_after = p.stat().st_size
        except OSError:
            stats.bytes_after = self._estimate_bytes(keep_lines, compress=(p.suffix == ".gz"))
        stats.rewrote = True
        return stats

    def prune_all(
        self,
        base_dir: Union[str, Path],
        machine: str = "*",
        dry_run: bool = False,
    ) -> Dict[Path, PruneStats]:
        """Prune every eligible partition under ``base_dir``.

        Iterates ``{base_dir}/{machine}/*.jsonl*`` in sorted order. Each
        partition is processed in isolation — failures are captured in
        ``PruneStats`` without propagating.

        Returns a dict mapping partition path to its stats, suitable for
        dashboard aggregation.
        """
        base = Path(base_dir)
        results: Dict[Path, PruneStats] = {}
        if not base.exists():
            _log.info("Pruner: base_dir %s does not exist; nothing to prune", base)
            return results

        patterns = [f"{machine}/*.jsonl", f"{machine}/*.jsonl.gz"]
        candidates: List[Path] = []
        for pat in patterns:
            candidates.extend(base.glob(pat))
        candidates = sorted(set(candidates))

        for partition in candidates:
            try:
                stats = self.prune_partition(partition, dry_run=dry_run)
            except Exception as e:  # pragma: no cover — prune_partition is non-raising
                stats = PruneStats(
                    path=str(partition),
                    age_days=-1,
                    bracket_rule="error",
                    error=f"unexpected exception: {e}",
                    skipped_reason="exception",
                    dry_run=dry_run,
                )
                _log.exception("Pruner unexpected exception on %s: %s", partition, e)
            results[partition] = stats

        return results

    # ── internals ───────────────────────────────────────────────────

    def _read_records(self, path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Read all records from ``path``.

        Returns ``(records, raw_lines)`` — parallel lists. We keep raw
        lines so the rewrite path preserves byte-exact representation,
        which matters for idempotency and for downstream hashing /
        signing if we ever add those.
        """
        records: List[Dict[str, Any]] = []
        raw_lines: List[str] = []
        with _open_read(path) as f:
            line_no = 0
            for raw in f:
                line_no += 1
                line = raw if raw.endswith("\n") else raw + "\n"
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError as e:
                    _log.warning(
                        "Pruner: corrupt JSONL line %d in %s: %s (dropped)",
                        line_no, path, e,
                    )
                    continue
                if not isinstance(rec, dict):
                    _log.warning(
                        "Pruner: line %d in %s is not a JSON object (dropped)",
                        line_no, path,
                    )
                    continue
                records.append(rec)
                raw_lines.append(line)
        return records, raw_lines

    def _meta_path(self, partition: Path) -> Path:
        """Sidecar path for a partition's pruner metadata."""
        return partition.with_suffix(partition.suffix + ".meta.json")

    def _load_or_init_meta(
        self,
        partition: Path,
        saliences: List[float],
        record_count: int,
    ) -> Dict[str, Any]:
        """Read the partition's `.meta.json` sidecar, or create it.

        On first-ever prune of a partition, the sidecar captures the
        salience quintile boundaries computed from the original
        distribution. Subsequent prunes read those same boundaries so
        the keep/drop decision is deterministic across passes — the
        foundation of idempotency.

        If the sidecar exists but is corrupt, we log and rebuild it
        from current data. This is conservative (a rebuild after partial
        pruning can shift boundaries) but never crashes the pass.
        """
        meta_path = self._meta_path(partition)
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if (
                    isinstance(meta, dict)
                    and isinstance(meta.get("quintile_boundaries"), list)
                    and len(meta["quintile_boundaries"]) == 4
                ):
                    return meta
                _log.warning(
                    "Pruner: meta sidecar malformed for %s; rebuilding", partition
                )
            except (OSError, json.JSONDecodeError) as e:
                _log.warning(
                    "Pruner: failed to read meta %s (%s); rebuilding", meta_path, e
                )
        # Build fresh.
        boundaries = _quintile_boundaries(saliences)
        meta = {
            "pruner_version": PRUNER_VERSION,
            "quintile_boundaries": boundaries,
            "original_record_count": record_count,
            "created_at": time.time(),
        }
        try:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, separators=(",", ":"))
            os.replace(str(tmp), str(meta_path))
        except OSError as e:
            _log.warning("Pruner: failed to write meta %s: %s", meta_path, e)
        return meta

    def _atomic_rewrite(self, path: Path, lines: List[str]) -> None:
        """Write ``lines`` to ``path`` atomically via ``.tmp`` + replace.

        Preserves compression inferred from the existing file's suffix.
        On any failure, the original file is untouched.
        """
        compress = path.suffix == ".gz"
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with _open_write(tmp, compress) as f:
                f.writelines(lines)
            # os.replace is atomic on POSIX and Windows (where available).
            os.replace(str(tmp), str(path))
        finally:
            # Best-effort cleanup if replace failed.
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:  # pragma: no cover
                    pass

    def _estimate_bytes(self, lines: List[str], compress: bool) -> int:
        """Estimate on-disk size of ``lines``.

        Exact for plain text; for gzip, runs the encoder in-memory. This
        exists for dry-run reporting so operators can see expected
        reclamation without touching disk.
        """
        if not compress:
            return sum(len(l.encode("utf-8")) for l in lines)
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            for l in lines:
                gz.write(l.encode("utf-8"))
        return buf.tell()


__all__ = [
    "RouterDatasetPruner",
    "PruneStats",
    "AGE_BRACKETS",
    "AGENT_ZERO_MARGIN_PP",
    "ACTIVE_WRITE_WINDOW_SECONDS",
    "RECOGNIZED_PIN_KINDS",
    "PRUNER_VERSION",
]
