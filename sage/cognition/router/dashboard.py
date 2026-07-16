"""
Router pipeline observability dashboard — Phase 0 Track 8.
===========================================================

Builds structured metrics from a Track 4 router dataset (JSONL partitions
under ``{base_dir}/{machine}/{YYYY-MM-DD}.jsonl[.gz]``) and renders a
markdown report for humans + optional JSON for tooling.

Design discipline — agent-zero defense (PRD §0.2, §7.10, §8):

    Every aggregate metric is reported alongside a **modal-class dummy**
    baseline — what an "always output the most common decision" policy
    would score on the same data. A dashboard that hides the dummy is
    exactly the Agent Zero failure mode: a mirror, not a measurement.

    Per PRD §7.10, a report that shows only aggregates without the dummy
    comparison is considered INCOMPLETE. This module surfaces the dummy
    alongside every decision-class metric by construction.

Reads (Track 4):
    * ``RouterDatasetReader`` for partition enumeration + JSONL parsing.

No torch dependency. No network. Pure-stdlib markdown + JSON output.
Target performance: <5s on 100k records, <30s on 1M records.

Spec: phase2/brain-arch/router-sprint-1-phase-0.md
      phase2/brain-arch/thalamic-router-prd.md
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from sage.cognition.router.data.reader import RouterDatasetReader
from sage.cognition.router.data.sampling import salience_score


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# Dimensions whose distribution we surface individually. Five canonical
# SNARC dims per PRD §2.2.
SNARC_DIMENSIONS: Tuple[str, ...] = (
    "arousal",
    "surprise",
    "novelty",
    "conflict",
    "reward",
)

# Bin edges for the SNARC histogram. 10 buckets over [0, 1] for positive
# dims; reward gets [-1, 1] remapped at render time. Fixed edges keep the
# output comparable across adapters and across time.
SNARC_HISTOGRAM_BINS: int = 10

# PRD §7.10 margin — the router (or Phase-0 programmatic baseline) must
# beat the modal-class dummy by at least 25 percentage points.
AGENT_ZERO_MARGIN_PP: float = 25.0

# Recent-trend windows surfaced on the dashboard.
RECENT_TREND_HOURS: int = 24
RECENT_TREND_DAYS: int = 7

# First-week storage-growth projection window. Dashboard will extrapolate
# from the most recent 24h of partitions on disk.
PROJECTION_WINDOW_HOURS: int = 24

# Dashboard schema version — bump on any structural change to the JSON
# output so downstream consumers can version-guard.
DASHBOARD_SCHEMA_VERSION: str = "v0.2.0"

# ── SNARC distribution drift (PRD §4.7.G) ─────────────────────────────
#
# Drift is measured per SNARC dimension as KL divergence between a stable
# training baseline distribution and a rolling serving distribution. The
# monitor is intentionally simple: histograms per dim, smoothed to avoid
# log(0), computed only when both windows carry enough samples to be
# meaningful.
#
# Window policy:
#   training-window  = records with timestamp ≥ DRIFT_TRAINING_MIN_AGE_DAYS
#                      old (stable baseline — old enough that training has
#                      plausibly happened on this distribution).
#   serving-window   = records with timestamp within
#                      DRIFT_SERVING_WINDOW_DAYS (rolling current behavior).
#
# Alert threshold per PRD §4.7.G: 0.1 nats.
#
# We deliberately do NOT make the alert threshold configurable: tuning
# knobs erode the PRD's single source of truth. If §4.7.G changes, the
# constant changes here.

DRIFT_TRAINING_MIN_AGE_DAYS: int = 30   # baseline: records ≥ 30d old
DRIFT_SERVING_WINDOW_DAYS: int = 7      # current: last 7d rolling
DRIFT_MIN_SAMPLES_PER_WINDOW: int = 1000  # per-dim sample floor
DRIFT_ALERT_THRESHOLD_NATS: float = 0.1   # PRD §4.7.G
# Laplace-smoothing pseudocount added to every histogram bin before the
# KL is evaluated. Prevents log(0) on zero bins without biasing toward
# any particular shape.
DRIFT_SMOOTHING_EPSILON: float = 1e-6


# ──────────────────────────────────────────────────────────────────────
# Aggregate dataclasses
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SnarcDimStats:
    """Per-SNARC-dimension mean/stddev/histogram."""

    dimension: str
    count: int = 0
    mean: float = 0.0
    stddev: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    # Histogram: list of counts per bin (length == SNARC_HISTOGRAM_BINS).
    histogram: List[int] = field(default_factory=list)
    # Bin edges ([lo, hi]) — same length as histogram.
    bin_edges: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "count": self.count,
            "mean": self.mean,
            "stddev": self.stddev,
            "min": self.min_value,
            "max": self.max_value,
            "histogram": list(self.histogram),
            "bin_edges": [list(e) for e in self.bin_edges],
        }


@dataclass
class DecisionStats:
    """Decision-class distribution + modal-class dummy baseline."""

    total: int = 0
    # action → count. Keys: 'invoke', 'habit', 'noop'.
    action_counts: Dict[str, int] = field(default_factory=dict)
    # plugin → count (only within invoke).
    plugin_counts: Dict[str, int] = field(default_factory=dict)

    # Modal-class dummy: what an always-output-most-common policy would
    # score (i.e. percentage of records it matches). Also captures which
    # class is modal so downstream code can trace the dummy.
    modal_action: Optional[str] = None
    modal_action_rate: float = 0.0  # [0, 1] — dummy's accuracy on this data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "action_counts": dict(self.action_counts),
            "plugin_counts": dict(self.plugin_counts),
            "modal_action": self.modal_action,
            "modal_action_rate": self.modal_action_rate,
        }


@dataclass
class MachineMetrics:
    """Per-machine (or aggregate) metrics bundle."""

    machine: str
    total_records: int = 0
    # YYYY-MM-DD → count
    records_per_day: Dict[str, int] = field(default_factory=dict)
    # schema_version → count
    schema_version_counts: Dict[str, int] = field(default_factory=dict)
    # dim → SnarcDimStats
    snarc: Dict[str, SnarcDimStats] = field(default_factory=dict)
    decisions: DecisionStats = field(default_factory=DecisionStats)
    # quintile index (0-4) → [seen, kept]. Phase 0 writes post-sampling
    # so we reconstruct the retention rate per quintile as the dataset
    # salience distribution vs. the uniform ideal (see _finalise).
    quintile_seen: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    # Storage size observed (bytes on disk).
    bytes_on_disk: int = 0
    # Storage-growth projection (bytes in first week extrapolated from
    # last PROJECTION_WINDOW_HOURS of data).
    projected_first_week_bytes: Optional[int] = None
    # Records/hour last 24h.
    records_last_24h: int = 0
    # Records/day last 7d.
    records_last_7d: int = 0
    # Earliest / latest observed timestamp.
    earliest_timestamp: Optional[float] = None
    latest_timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine": self.machine,
            "total_records": self.total_records,
            "records_per_day": dict(self.records_per_day),
            "schema_version_counts": dict(self.schema_version_counts),
            "snarc": {k: v.to_dict() for k, v in self.snarc.items()},
            "decisions": self.decisions.to_dict(),
            "quintile_seen": list(self.quintile_seen),
            "bytes_on_disk": self.bytes_on_disk,
            "projected_first_week_bytes": self.projected_first_week_bytes,
            "records_last_24h": self.records_last_24h,
            "records_last_7d": self.records_last_7d,
            "earliest_timestamp": self.earliest_timestamp,
            "latest_timestamp": self.latest_timestamp,
        }


@dataclass
class SnarcDriftMetrics:
    """Per-dimension KL-divergence drift between training and serving windows.

    ``status`` is one of:
      * ``"HEALTHY"`` — KL below alert threshold
      * ``"DRIFT ALERT"`` — KL ≥ alert threshold, retraining flag per §4.7.G
      * ``"INSUFFICIENT DATA"`` — at least one window under the sample floor

    Sample counts are always reported alongside the KL value; agent-zero
    discipline applied to drift — a KL of 0.3 off of 12 training samples is
    not a drift alert, it's a reporting artifact.
    """

    dimension: str
    status: str = "INSUFFICIENT DATA"
    kl_nats: Optional[float] = None       # None unless status != INSUFFICIENT DATA
    training_count: int = 0
    serving_count: int = 0
    training_histogram: List[int] = field(default_factory=list)
    serving_histogram: List[int] = field(default_factory=list)
    bin_edges: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "status": self.status,
            "kl_nats": self.kl_nats,
            "training_count": self.training_count,
            "serving_count": self.serving_count,
            "training_histogram": list(self.training_histogram),
            "serving_histogram": list(self.serving_histogram),
            "bin_edges": [list(e) for e in self.bin_edges],
        }


@dataclass
class SnarcDriftReport:
    """Per-machine (or aggregate) drift report for all SNARC dimensions."""

    machine: str
    # dim → SnarcDriftMetrics
    dimensions: Dict[str, SnarcDriftMetrics] = field(default_factory=dict)
    # True if any dimension has status == DRIFT ALERT.
    any_alert: bool = False
    # True if every dimension is INSUFFICIENT DATA.
    awaiting_baseline: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine": self.machine,
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "any_alert": self.any_alert,
            "awaiting_baseline": self.awaiting_baseline,
        }


@dataclass
class DashboardMetrics:
    """Full dashboard payload — per-machine + aggregate."""

    generated_at: float
    base_dir: str
    machine_filter: str
    date_range: Optional[Tuple[str, str]]
    schema_version: str = DASHBOARD_SCHEMA_VERSION
    per_machine: Dict[str, MachineMetrics] = field(default_factory=dict)
    aggregate: MachineMetrics = field(default_factory=lambda: MachineMetrics(machine="ALL"))
    # SNARC distribution-drift report (PRD §4.7.G).
    drift_aggregate: SnarcDriftReport = field(
        default_factory=lambda: SnarcDriftReport(machine="ALL")
    )
    drift_per_machine: Dict[str, SnarcDriftReport] = field(default_factory=dict)
    # Wall-clock build time (seconds) — useful when tuning nightly cron.
    build_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "base_dir": self.base_dir,
            "machine_filter": self.machine_filter,
            "date_range": list(self.date_range) if self.date_range else None,
            "build_seconds": self.build_seconds,
            "per_machine": {k: v.to_dict() for k, v in self.per_machine.items()},
            "aggregate": self.aggregate.to_dict(),
            "drift_aggregate": self.drift_aggregate.to_dict(),
            "drift_per_machine": {
                k: v.to_dict() for k, v in self.drift_per_machine.items()
            },
        }


# ──────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────


class DashboardBuilder:
    """Build dashboard metrics from a Track 4 dataset directory.

    Parameters
    ----------
    base_dir:
        Root of the router dataset (same dir the writer received).
    machine:
        Machine name to filter, or ``"*"`` for all. Per-machine stats
        are always reported; this filter limits which machines' partitions
        are read.
    date_range:
        Optional (start, end) inclusive. Each as ``YYYY-MM-DD`` string or
        ``datetime.date``. ``None`` → no date filter.
    reader:
        Optional injected :class:`RouterDatasetReader` — primarily for
        tests. Defaults to constructing one from ``base_dir``.
    clock:
        Optional callable returning current UTC time (float seconds).
        Tests inject a fixed clock; production uses ``time.time``.

    Performance
    -----------
    The builder makes ONE pass over records. Histograms use fixed bins so
    no second pass is needed. Running mean/stddev via Welford. On 100k
    records typical wall-time is well under 5 seconds on consumer hw.
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        machine: str = "*",
        date_range: Optional[Tuple[Union[str, date], Union[str, date]]] = None,
        reader: Optional[RouterDatasetReader] = None,
        clock: Optional[Any] = None,
    ):
        self.base_dir = Path(base_dir)
        self.machine_filter = machine
        self.date_range = _normalize_date_range(date_range)
        self.reader = reader or RouterDatasetReader(self.base_dir)
        self._clock = clock or time.time

    # ── public ────────────────────────────────────────────────────

    def build(self) -> DashboardMetrics:
        """Run one pass over partitions and return populated metrics."""
        t0 = self._clock()

        metrics = DashboardMetrics(
            generated_at=t0,
            base_dir=str(self.base_dir),
            machine_filter=self.machine_filter,
            date_range=self._date_range_as_strings(),
        )

        # ── Welford accumulators per machine × per SNARC dim. ────
        # welford[machine][dim] = (n, mean, M2, min, max, hist_bins_lo0to1,
        #                         hist_bins_reward)
        welford: Dict[str, Dict[str, _Welford]] = defaultdict(
            lambda: {dim: _Welford(dim) for dim in SNARC_DIMENSIONS}
        )

        # Enumerate partitions FIRST — bytes + per-day row count come from
        # the on-disk layout rather than from record contents (fast + a
        # faithful reflection of storage growth).
        partitions = self.reader.list_partitions(
            machine=self.machine_filter, date_range=self.date_range
        )
        partition_meta: List[Dict[str, Any]] = []
        for p in partitions:
            mach, day = _machine_and_date_from_partition(p, self.base_dir)
            if mach is None or day is None:
                # Not a machine/date-shaped partition — skip but don't
                # fail the build. `list_partitions` already filters by
                # machine; a stray file suggests an operator mistake,
                # but the dashboard is observability, not enforcement.
                continue
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            partition_meta.append(
                {"path": p, "machine": mach, "date": day, "size": size}
            )

        # Empty dataset: short-circuit but still return a well-formed
        # DashboardMetrics so downstream renderers don't have to special-
        # case None. The markdown layer uses `total_records == 0` to
        # switch to the "awaiting first data" preamble.
        if not partition_meta:
            metrics.build_seconds = self._clock() - t0
            return metrics

        # Seed per-machine bytes/per-day counts from partitions.
        for meta in partition_meta:
            mach = meta["machine"]
            day = meta["date"]
            mm = metrics.per_machine.setdefault(mach, MachineMetrics(machine=mach))
            mm.bytes_on_disk += meta["size"]
            # records_per_day incremented as records are read, not here.
            mm.records_per_day.setdefault(day, 0)
            metrics.aggregate.bytes_on_disk += meta["size"]
            metrics.aggregate.records_per_day.setdefault(day, 0)

        # ── Per-record pass. ──────────────────────────────────────
        cutoff_24h = t0 - RECENT_TREND_HOURS * 3600.0
        cutoff_7d = t0 - RECENT_TREND_DAYS * 86400.0
        cutoff_proj = t0 - PROJECTION_WINDOW_HOURS * 3600.0
        # ── SNARC drift windows (PRD §4.7.G). ─────────────────────
        # Training window: records older than DRIFT_TRAINING_MIN_AGE_DAYS —
        # used as the stable baseline distribution.
        # Serving window: records within DRIFT_SERVING_WINDOW_DAYS — the
        # rolling current behavior we compare against baseline.
        cutoff_training_upper = t0 - DRIFT_TRAINING_MIN_AGE_DAYS * 86400.0
        cutoff_serving_lower = t0 - DRIFT_SERVING_WINDOW_DAYS * 86400.0
        # histograms[machine][dim] = [train_hist, serve_hist]
        # train_hist / serve_hist are lists of SNARC_HISTOGRAM_BINS ints.
        drift_hist: Dict[str, Dict[str, Dict[str, List[int]]]] = defaultdict(
            lambda: {
                dim: {
                    "training": [0] * SNARC_HISTOGRAM_BINS,
                    "serving": [0] * SNARC_HISTOGRAM_BINS,
                }
                for dim in SNARC_DIMENSIONS
            }
        )
        # Salience scores collected per machine to compute quintile
        # boundaries AFTER the pass (one sort; avoids a second read).
        salience_per_machine: Dict[str, List[float]] = defaultdict(list)
        # Projection window byte counting needs file size + record-time
        # filtering. We bucket records by partition and at the end
        # proportionally allocate bytes.
        proj_records_per_partition: Dict[Path, int] = defaultdict(int)
        total_records_per_partition: Dict[Path, int] = defaultdict(int)

        for meta in partition_meta:
            path = meta["path"]
            mach = meta["machine"]
            day = meta["date"]
            mm = metrics.per_machine[mach]
            for rec in self.reader.read_file(path):
                total_records_per_partition[path] += 1
                mm.total_records += 1
                mm.records_per_day[day] = mm.records_per_day.get(day, 0) + 1
                metrics.aggregate.total_records += 1
                metrics.aggregate.records_per_day[day] = (
                    metrics.aggregate.records_per_day.get(day, 0) + 1
                )

                # schema version distribution
                sv = str(rec.get("schema_version", "unknown"))
                mm.schema_version_counts[sv] = (
                    mm.schema_version_counts.get(sv, 0) + 1
                )
                metrics.aggregate.schema_version_counts[sv] = (
                    metrics.aggregate.schema_version_counts.get(sv, 0) + 1
                )

                # timestamp window
                ts = _extract_timestamp(rec)
                if ts is not None:
                    mm.earliest_timestamp = (
                        ts if mm.earliest_timestamp is None
                        else min(mm.earliest_timestamp, ts)
                    )
                    mm.latest_timestamp = (
                        ts if mm.latest_timestamp is None
                        else max(mm.latest_timestamp, ts)
                    )
                    metrics.aggregate.earliest_timestamp = (
                        ts if metrics.aggregate.earliest_timestamp is None
                        else min(metrics.aggregate.earliest_timestamp, ts)
                    )
                    metrics.aggregate.latest_timestamp = (
                        ts if metrics.aggregate.latest_timestamp is None
                        else max(metrics.aggregate.latest_timestamp, ts)
                    )

                    if ts >= cutoff_24h:
                        mm.records_last_24h += 1
                        metrics.aggregate.records_last_24h += 1
                    if ts >= cutoff_7d:
                        mm.records_last_7d += 1
                        metrics.aggregate.records_last_7d += 1
                    if ts >= cutoff_proj:
                        proj_records_per_partition[path] += 1

                # SNARC
                snarc = _extract_snarc(rec)
                # Drift classification: which window does this record fall
                # into? Records with no timestamp are excluded from drift
                # (we need wall-clock to slot them). Records landing in the
                # no-man's-land between training and serving windows are
                # skipped — they are neither stable baseline nor current.
                drift_slot: Optional[str]
                if ts is None:
                    drift_slot = None
                elif ts <= cutoff_training_upper:
                    drift_slot = "training"
                elif ts >= cutoff_serving_lower:
                    drift_slot = "serving"
                else:
                    drift_slot = None

                for dim in SNARC_DIMENSIONS:
                    v = snarc.get(dim)
                    if v is None:
                        continue
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        continue
                    welford[mach][dim].push(v)
                    welford["__AGG__"][dim].push(v)
                    # Populate drift histogram for the appropriate window.
                    if drift_slot is not None:
                        bin_idx = _drift_bin_index(dim, v)
                        drift_hist[mach][dim][drift_slot][bin_idx] += 1
                        drift_hist["__AGG__"][dim][drift_slot][bin_idx] += 1

                # Salience → quintile bucket (post-hoc; boundaries from
                # the observed distribution, not a configured global).
                sal = salience_score(snarc)
                salience_per_machine[mach].append(sal)
                salience_per_machine["__AGG__"].append(sal)

                # Decision distribution
                action, plugin = _extract_decision(rec)
                if action is not None:
                    mm.decisions.action_counts[action] = (
                        mm.decisions.action_counts.get(action, 0) + 1
                    )
                    mm.decisions.total += 1
                    metrics.aggregate.decisions.action_counts[action] = (
                        metrics.aggregate.decisions.action_counts.get(action, 0) + 1
                    )
                    metrics.aggregate.decisions.total += 1
                    if action == "invoke" and plugin:
                        mm.decisions.plugin_counts[plugin] = (
                            mm.decisions.plugin_counts.get(plugin, 0) + 1
                        )
                        metrics.aggregate.decisions.plugin_counts[plugin] = (
                            metrics.aggregate.decisions.plugin_counts.get(plugin, 0) + 1
                        )

        # ── Finalise: Welford → stats, salience → quintile_seen. ──
        for mach, mm in metrics.per_machine.items():
            self._finalise_machine(mm, welford[mach], salience_per_machine[mach])
        self._finalise_machine(
            metrics.aggregate, welford["__AGG__"], salience_per_machine["__AGG__"]
        )

        # ── Finalise drift reports. ───────────────────────────────
        for mach in metrics.per_machine:
            metrics.drift_per_machine[mach] = _finalise_drift_report(
                machine=mach,
                per_dim=drift_hist.get(mach, {}),
            )
        metrics.drift_aggregate = _finalise_drift_report(
            machine="ALL",
            per_dim=drift_hist.get("__AGG__", {}),
        )

        # ── Byte-growth projection. ───────────────────────────────
        # For each partition, bytes are allocated proportionally based on
        # (records in projection window / records in partition). Then
        # aggregated per machine and scaled 24h → 7d (×7).
        for meta in partition_meta:
            path = meta["path"]
            mach = meta["machine"]
            total = total_records_per_partition.get(path, 0)
            in_window = proj_records_per_partition.get(path, 0)
            if total <= 0 or in_window <= 0:
                continue
            proj_bytes_24h = meta["size"] * (in_window / total)
            mm = metrics.per_machine[mach]
            mm.projected_first_week_bytes = int(
                (mm.projected_first_week_bytes or 0) + proj_bytes_24h * 7
            )
            metrics.aggregate.projected_first_week_bytes = int(
                (metrics.aggregate.projected_first_week_bytes or 0) + proj_bytes_24h * 7
            )

        metrics.build_seconds = self._clock() - t0
        return metrics

    # ── internals ─────────────────────────────────────────────────

    def _date_range_as_strings(self) -> Optional[Tuple[str, str]]:
        if self.date_range is None:
            return None
        start, end = self.date_range
        return (str(start), str(end))

    def _finalise_machine(
        self,
        mm: MachineMetrics,
        welford_by_dim: Dict[str, "_Welford"],
        salience_scores: List[float],
    ) -> None:
        for dim in SNARC_DIMENSIONS:
            w = welford_by_dim[dim]
            mm.snarc[dim] = w.as_stats(bins=SNARC_HISTOGRAM_BINS)

        # Quintile distribution from the observed salience scores.
        # 'seen' is the natural distribution post-sampling; retention
        # rate per quintile is surfaced in the markdown renderer (a
        # sanity check that the top SNARC quintile is represented).
        if salience_scores:
            sorted_s = sorted(salience_scores)
            n = len(sorted_s)
            cuts = [
                sorted_s[int(n * 0.20)],
                sorted_s[int(n * 0.40)],
                sorted_s[int(n * 0.60)],
                sorted_s[int(n * 0.80)],
            ]
            seen = [0, 0, 0, 0, 0]
            for s in salience_scores:
                idx = 4
                for i, cut in enumerate(cuts):
                    if s < cut:
                        idx = i
                        break
                seen[idx] += 1
            mm.quintile_seen = seen

        # Modal-class dummy baseline.
        if mm.decisions.total > 0:
            modal_action, modal_count = max(
                mm.decisions.action_counts.items(), key=lambda kv: kv[1]
            )
            mm.decisions.modal_action = modal_action
            mm.decisions.modal_action_rate = modal_count / mm.decisions.total


# ──────────────────────────────────────────────────────────────────────
# Markdown + JSON renderers
# ──────────────────────────────────────────────────────────────────────


def render_markdown(
    metrics: DashboardMetrics,
    *,
    include_drift: Union[bool, str] = "auto",
) -> str:
    """Render a DashboardMetrics object as human-readable markdown.

    Every aggregate number is paired with the modal-class dummy baseline
    per PRD §7.10 agent-zero discipline. When there is no data, renders a
    short "awaiting first data" preamble instead of empty tables.

    Parameters
    ----------
    include_drift:
        ``True`` / ``"on"`` → always render the SNARC drift section.
        ``False`` / ``"off"`` → skip it (e.g. when the caller renders a
        tight report without §4.7.G context).
        ``"auto"`` (default) → include when the dashboard has record
        data. Matches the CLI default.
    """
    lines: List[str] = []
    lines.append("# Router Pipeline Dashboard")
    lines.append("")
    gen_ts = datetime.fromtimestamp(metrics.generated_at, tz=timezone.utc)
    lines.append(
        f"_Generated {gen_ts.isoformat(timespec='seconds')} "
        f"(build {metrics.build_seconds*1000:.0f} ms)._"
    )
    lines.append("")
    lines.append(
        "This file is auto-regenerated by "
        "`scripts/router_dashboard_render.py`."
    )
    lines.append(
        "Every aggregate metric is paired with the **modal-class dummy** "
        "baseline (PRD §7.10 agent-zero discipline): a dashboard without "
        "that column is INCOMPLETE by construction."
    )
    lines.append("")
    lines.append(f"- **Base dir**: `{metrics.base_dir}`")
    lines.append(f"- **Machine filter**: `{metrics.machine_filter}`")
    if metrics.date_range:
        lines.append(
            f"- **Date range**: `{metrics.date_range[0]}` → `{metrics.date_range[1]}`"
        )
    else:
        lines.append("- **Date range**: (all partitions)")
    lines.append("")

    if metrics.aggregate.total_records == 0:
        lines.append("## Status")
        lines.append("")
        lines.append(
            "**Awaiting first data** — will be auto-updated when router "
            "shadow captures begin."
        )
        lines.append("")
        lines.extend(_references_section())
        return "\n".join(lines) + "\n"

    # ── Summary ──
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- **Total records**: {metrics.aggregate.total_records:,}"
    )
    earliest = _fmt_ts(metrics.aggregate.earliest_timestamp)
    latest = _fmt_ts(metrics.aggregate.latest_timestamp)
    lines.append(f"- **Earliest record**: {earliest}")
    lines.append(f"- **Latest record**: {latest}")
    lines.append(f"- **Machines observed**: {len(metrics.per_machine)}")
    agg_bytes = metrics.aggregate.bytes_on_disk
    lines.append(f"- **Bytes on disk (aggregate)**: {_fmt_bytes(agg_bytes)}")
    if metrics.aggregate.projected_first_week_bytes is not None:
        lines.append(
            "- **Projected first-week bytes** "
            f"(extrapolated from last {PROJECTION_WINDOW_HOURS}h): "
            f"{_fmt_bytes(metrics.aggregate.projected_first_week_bytes)}"
        )
    lines.append("")

    # ── Records/day (aggregate first, then per machine) ──
    lines.append("## Records per day")
    lines.append("")
    lines.append("### Aggregate")
    lines.append("")
    lines.append(_records_per_day_table(metrics.aggregate))
    lines.append("")
    for mach, mm in sorted(metrics.per_machine.items()):
        lines.append(f"### {mach}")
        lines.append("")
        lines.append(_records_per_day_table(mm))
        lines.append("")

    # ── Schema version distribution ──
    lines.append("## Schema version distribution")
    lines.append("")
    lines.append(_schema_version_table(metrics))
    lines.append("")

    # ── Decision-class distribution with modal-class dummy column ──
    lines.append("## Decision-class distribution")
    lines.append("")
    lines.append(
        "**Agent-zero discipline** (PRD §0.2, §7.10): the *modal-class "
        "dummy* column is the score an always-output-most-common policy "
        "would achieve on this data. The margin column shows each "
        "non-modal class's presence above the dummy's 0% on that class."
    )
    lines.append("")
    lines.append("### Aggregate")
    lines.append("")
    lines.append(_decision_class_table(metrics.aggregate))
    lines.append("")
    for mach, mm in sorted(metrics.per_machine.items()):
        lines.append(f"### {mach}")
        lines.append("")
        lines.append(_decision_class_table(mm))
        lines.append("")

    # ── Plugin breakdown (within invoke) ──
    lines.append("## Plugin breakdown within `invoke`")
    lines.append("")
    lines.append("### Aggregate")
    lines.append("")
    lines.append(_plugin_table(metrics.aggregate))
    lines.append("")
    for mach, mm in sorted(metrics.per_machine.items()):
        if not mm.decisions.plugin_counts:
            continue
        lines.append(f"### {mach}")
        lines.append("")
        lines.append(_plugin_table(mm))
        lines.append("")

    # ── SNARC distribution ──
    lines.append("## SNARC distribution per dimension")
    lines.append("")
    lines.append(
        "Mean, stddev, and a 10-bucket histogram over the full range. "
        "Verifies stratified sampling is working (PRD §4.7.D) — "
        "top-SNARC records should be visibly represented in the tail of "
        "the `arousal` / `conflict` histograms."
    )
    lines.append("")
    lines.append("### Aggregate")
    lines.append("")
    lines.append(_snarc_table(metrics.aggregate))
    lines.append("")
    for mach, mm in sorted(metrics.per_machine.items()):
        lines.append(f"### {mach}")
        lines.append("")
        lines.append(_snarc_table(mm))
        lines.append("")

    # ── Sampling retention (per quintile) ──
    lines.append("## Sampling retention by SNARC quintile")
    lines.append("")
    lines.append(
        "Computed from the salience distribution of records ALREADY on "
        "disk (post-sampling). Quintiles use per-scope boundaries; ratios "
        "approximate the sampler's realized keep rate per quintile and "
        "should track PRD §4.7.D (bottom ~5%, middle ~20%, top 100%)."
    )
    lines.append("")
    lines.append("### Aggregate")
    lines.append("")
    lines.append(_quintile_table(metrics.aggregate))
    lines.append("")
    for mach, mm in sorted(metrics.per_machine.items()):
        lines.append(f"### {mach}")
        lines.append("")
        lines.append(_quintile_table(mm))
        lines.append("")

    # ── Recent trend ──
    lines.append("## Recent trend")
    lines.append("")
    lines.append(_recent_trend_table(metrics))
    lines.append("")

    # ── SNARC distribution drift (PRD §4.7.G) ──
    if _should_include_drift(include_drift):
        lines.extend(_drift_section(metrics))

    lines.extend(_references_section())
    return "\n".join(lines) + "\n"


def _should_include_drift(flag: Union[bool, str]) -> bool:
    """Normalize the include_drift flag.

    ``True`` / ``"on"`` → include.
    ``False`` / ``"off"`` → skip.
    ``"auto"`` (default) → include — the drift section self-censors to
    "awaiting baseline" when there isn't enough data yet, so 'auto'
    effectively means 'always include but render gracefully'.
    """
    if isinstance(flag, bool):
        return flag
    val = str(flag).strip().lower()
    if val in ("on", "true", "yes", "1"):
        return True
    if val in ("off", "false", "no", "0"):
        return False
    # "auto" or anything else → include (graceful early-days render).
    return True


def render_json(metrics: DashboardMetrics, *, indent: int = 2) -> str:
    """Render metrics as a JSON document (for Track 7 scheduling / tooling)."""
    return json.dumps(metrics.to_dict(), indent=indent, default=_json_default)


# ──────────────────────────────────────────────────────────────────────
# Table helpers
# ──────────────────────────────────────────────────────────────────────


def _records_per_day_table(mm: MachineMetrics) -> str:
    if not mm.records_per_day:
        return "_No records on disk yet._"
    rows = [f"| Date | Records |", "|---|---|"]
    for day in sorted(mm.records_per_day.keys()):
        rows.append(f"| `{day}` | {mm.records_per_day[day]:,} |")
    rows.append(f"| **Total** | **{mm.total_records:,}** |")
    return "\n".join(rows)


def _schema_version_table(metrics: DashboardMetrics) -> str:
    # Aggregate across machines for the summary; a version skew is a
    # fleet-wide concern, not per-machine.
    rows = ["| Schema version | Records | Share |", "|---|---|---|"]
    total = metrics.aggregate.total_records
    for ver, count in sorted(
        metrics.aggregate.schema_version_counts.items(), key=lambda kv: -kv[1]
    ):
        share = (count / total) if total > 0 else 0.0
        rows.append(f"| `{ver}` | {count:,} | {share*100:.1f}% |")
    return "\n".join(rows)


def _decision_class_table(mm: MachineMetrics) -> str:
    total = mm.decisions.total
    if total == 0:
        return "_No decisions recorded._"
    modal = mm.decisions.modal_action
    modal_rate = mm.decisions.modal_action_rate
    rows = [
        "| Decision type | Observed | Observed % | Modal-class dummy % | Margin (pp) |",
        "|---|---|---|---|---|",
    ]
    # Stable column order: invoke, habit, noop, then anything else.
    preferred = ["invoke", "habit", "noop"]
    seen = set()
    ordered: List[str] = []
    for a in preferred:
        if a in mm.decisions.action_counts:
            ordered.append(a)
            seen.add(a)
    for a in mm.decisions.action_counts:
        if a not in seen:
            ordered.append(a)
    for action in ordered:
        count = mm.decisions.action_counts[action]
        obs_pct = 100.0 * count / total
        if action == modal:
            dummy_pct = 100.0
            dummy_label = f"100% (modal)"
        else:
            dummy_pct = 0.0
            dummy_label = "0%"
        margin = obs_pct - dummy_pct
        # Margin sign tells the story: positive on non-modal classes
        # (they show up where the dummy never would), exactly zero on
        # the modal class (dummy matches it 100%).
        rows.append(
            f"| `{action}` | {count:,} | {obs_pct:.1f}% | "
            f"{dummy_label} | {margin:+.1f} |"
        )
    rows.append("")
    rows.append(
        f"_Modal class is **`{modal}`** at {modal_rate*100:.1f}% — an "
        f"always-`{modal}` dummy policy would score {modal_rate*100:.1f}% "
        f"aggregate agreement on this slice. PRD §7.10 requires the "
        f"router to beat this by ≥{AGENT_ZERO_MARGIN_PP:.0f} percentage "
        f"points when evaluated end-to-end._"
    )
    return "\n".join(rows)


def _plugin_table(mm: MachineMetrics) -> str:
    if not mm.decisions.plugin_counts:
        return "_No `invoke` decisions recorded._"
    total_invoke = sum(mm.decisions.plugin_counts.values())
    rows = [
        "| Plugin | Invocations | Share of `invoke` |",
        "|---|---|---|",
    ]
    for plugin, count in sorted(
        mm.decisions.plugin_counts.items(), key=lambda kv: -kv[1]
    ):
        share = count / total_invoke
        rows.append(f"| `{plugin}` | {count:,} | {share*100:.1f}% |")
    return "\n".join(rows)


def _snarc_table(mm: MachineMetrics) -> str:
    rows = [
        "| Dimension | n | Mean | Stddev | Min | Max | Histogram |",
        "|---|---|---|---|---|---|---|",
    ]
    for dim in SNARC_DIMENSIONS:
        s = mm.snarc.get(dim)
        if s is None or s.count == 0:
            rows.append(f"| `{dim}` | 0 | — | — | — | — | _no data_ |")
            continue
        # Sparkline-ish bar via block characters, normalized per row.
        hist_str = _sparkline(s.histogram)
        rows.append(
            f"| `{dim}` | {s.count:,} | {s.mean:.3f} | {s.stddev:.3f} | "
            f"{s.min_value:.3f} | {s.max_value:.3f} | `{hist_str}` |"
        )
    return "\n".join(rows)


def _quintile_table(mm: MachineMetrics) -> str:
    total = sum(mm.quintile_seen)
    if total == 0:
        return "_No salience data yet._"
    rows = [
        "| Quintile | Records | Share | Expected (sampler target) |",
        "|---|---|---|---|",
    ]
    # Expected-share hints from the PRD §4.7.D target rates — these are
    # the realized shares AFTER sampling (i.e. what the on-disk corpus
    # should look like). We derive them from the default keep rates.
    # A natural uniform distribution (20% per quintile) times keep rates
    # [0.05, 0.20, 0.20, 0.20, 1.00] = [0.01, 0.04, 0.04, 0.04, 0.20],
    # renormalized → [3.0%, 12.1%, 12.1%, 12.1%, 60.6%].
    target_shares = [0.030, 0.121, 0.121, 0.121, 0.606]
    names = ["Q0 (low)", "Q1", "Q2", "Q3", "Q4 (top)"]
    for i in range(5):
        count = mm.quintile_seen[i]
        share = count / total
        rows.append(
            f"| {names[i]} | {count:,} | {share*100:.1f}% | "
            f"~{target_shares[i]*100:.1f}% |"
        )
    return "\n".join(rows)


def _recent_trend_table(metrics: DashboardMetrics) -> str:
    rows = [
        "| Scope | Records last 24h | Records last 7d |",
        "|---|---|---|",
    ]
    rows.append(
        f"| **Aggregate** | {metrics.aggregate.records_last_24h:,} | "
        f"{metrics.aggregate.records_last_7d:,} |"
    )
    for mach, mm in sorted(metrics.per_machine.items()):
        rows.append(
            f"| `{mach}` | {mm.records_last_24h:,} | "
            f"{mm.records_last_7d:,} |"
        )
    return "\n".join(rows)


def _drift_section(metrics: DashboardMetrics) -> List[str]:
    """Render the SNARC distribution-drift section (PRD §4.7.G).

    Always included when the dashboard has record data. If no window has
    enough samples yet (fresh deployment), renders an "awaiting baseline"
    preamble rather than a table of INSUFFICIENT DATA rows — operators
    shouldn't read the early-days output as a scary alert surface.
    """
    lines: List[str] = []
    lines.append("## SNARC distribution drift (PRD §4.7.G)")
    lines.append("")
    lines.append(
        "**Training window**: records ≥ "
        f"{DRIFT_TRAINING_MIN_AGE_DAYS} days old (stable baseline). "
        "**Serving window**: last "
        f"{DRIFT_SERVING_WINDOW_DAYS} days (rolling current). "
        f"**Alert threshold**: KL ≥ {DRIFT_ALERT_THRESHOLD_NATS} nats "
        "per dim (PRD §4.7.G). "
        f"**Sample floor**: {DRIFT_MIN_SAMPLES_PER_WINDOW:,} records per "
        "window per dim — fewer than this reports INSUFFICIENT DATA "
        "rather than a false alarm."
    )
    lines.append("")
    lines.append(
        "Laplace-style smoothing (ε="
        f"{DRIFT_SMOOTHING_EPSILON:g}) is applied to every histogram bin "
        "before computing KL(serving || training), so zero-count bins "
        "don't push the divergence to infinity."
    )
    lines.append("")

    drift_agg = metrics.drift_aggregate
    # True if NO machine (aggregate or per-machine) has crossed the
    # sample floor on any dimension. We use this to switch to the
    # "awaiting baseline" preamble rather than showing a wall of
    # INSUFFICIENT DATA rows during early-deployment days.
    per_mach_all_awaiting = all(
        rep.awaiting_baseline for rep in metrics.drift_per_machine.values()
    ) if metrics.drift_per_machine else True
    if drift_agg.awaiting_baseline and per_mach_all_awaiting:
        lines.append(
            "_Awaiting baseline — no dimension yet has "
            f"{DRIFT_MIN_SAMPLES_PER_WINDOW:,} records in BOTH the training "
            "and serving windows. Drift monitoring activates per dimension "
            "once the sample floor is met._"
        )
        lines.append("")
        return lines

    # Aggregate summary.
    lines.append("### Aggregate")
    lines.append("")
    lines.append(_drift_table(drift_agg))
    lines.append("")

    # Per-machine (only those with any non-insufficient row — or all if
    # operator wants full visibility; we show all so isolation is obvious).
    for mach in sorted(metrics.drift_per_machine):
        rep = metrics.drift_per_machine[mach]
        lines.append(f"### {mach}")
        lines.append("")
        lines.append(_drift_table(rep))
        lines.append("")

    # Summary line — quick-glance alert status.
    if drift_agg.any_alert:
        lines.append(
            "**STATUS**: DRIFT ALERT fired on aggregate — retraining "
            "flag per PRD §4.7.G. Inspect per-machine table to locate "
            "the drifting source(s)."
        )
    elif drift_agg.awaiting_baseline:
        lines.append(
            "**STATUS**: baseline still accumulating — no aggregate "
            "dimension has crossed the sample floor in both windows."
        )
    else:
        lines.append(
            "**STATUS**: healthy — all aggregate SNARC dimensions with "
            f"sufficient data report KL < {DRIFT_ALERT_THRESHOLD_NATS} nats."
        )
    lines.append("")
    return lines


def _drift_table(report: SnarcDriftReport) -> str:
    rows = [
        "| Dimension | Training n | Serving n | KL (nats) | Status |",
        "|---|---|---|---|---|",
    ]
    for dim in SNARC_DIMENSIONS:
        d = report.dimensions.get(dim)
        if d is None:
            rows.append(f"| `{dim}` | 0 | 0 | — | INSUFFICIENT DATA |")
            continue
        if d.status == "INSUFFICIENT DATA":
            rows.append(
                f"| `{dim}` | {d.training_count:,} | {d.serving_count:,} "
                f"| — | INSUFFICIENT DATA |"
            )
        else:
            kl_str = "—" if d.kl_nats is None else f"{d.kl_nats:.4f}"
            rows.append(
                f"| `{dim}` | {d.training_count:,} | {d.serving_count:,} "
                f"| {kl_str} | {d.status} |"
            )
    return "\n".join(rows)


def _references_section() -> List[str]:
    return [
        "## References",
        "",
        "- **PRD §8** (evaluation metrics, dashboard contract): "
        "`phase2/brain-arch/thalamic-router-prd.md`",
        "- **PRD §0.2, §7.10** (agent-zero discipline — "
        "modal-class dummy comparison)",
        "- **PRD §4.7.D, §4.7.F, §4.7.G** (SNARC sampling, SNARC "
        "ablation, distribution drift — drift monitor surfaces the §4.7.G "
        "KL comparison here at "
        f"alert threshold {DRIFT_ALERT_THRESHOLD_NATS} nats)",
        "- **Track 4** (dataset writer/reader — input source): "
        "`sage/cognition/router/data/`",
        "- **Track 9** (SNARC-driven storage pruning): "
        "`sage/cognition/router/data/pruner.py`",
        "- **Sprint doc**: "
        "`phase2/brain-arch/router-sprint-1-phase-0.md`",
        "",
        "_SNARC-utility delta (PRD §8) is intentionally NOT surfaced "
        "yet — it requires training signal from Phase 1 and isn't "
        "computable from Phase 0 data alone._",
    ]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _normalize_date_range(
    date_range: Optional[Tuple[Union[str, date], Union[str, date]]],
) -> Optional[Tuple[str, str]]:
    """Normalize a date range to (YYYY-MM-DD, YYYY-MM-DD)."""
    if date_range is None:
        return None
    start, end = date_range
    return (_date_to_str(start), _date_to_str(end))


def _date_to_str(d: Union[str, date]) -> str:
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    return str(d)


def _machine_and_date_from_partition(
    path: Path, base_dir: Path
) -> Tuple[Optional[str], Optional[str]]:
    """Infer (machine, YYYY-MM-DD) from the partition path.

    Layout is ``{base_dir}/{machine}/{YYYY-MM-DD}.jsonl[.gz]``.
    Returns (None, None) if the path doesn't match.
    """
    try:
        rel = path.relative_to(base_dir)
    except ValueError:
        return None, None
    parts = rel.parts
    if len(parts) < 2:
        return None, None
    machine = parts[-2]
    stem = parts[-1]
    for suffix in (".jsonl.gz", ".jsonl"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    try:
        datetime.strptime(stem, "%Y-%m-%d")
    except ValueError:
        return machine, None
    return machine, stem


def _extract_timestamp(rec: Mapping[str, Any]) -> Optional[float]:
    """Fetch the record's unix-seconds timestamp.

    Records may carry ``timestamp`` at the top level (writer-stamped) OR
    nested inside ``router_input`` (RouterInput.timestamp). Prefer top-
    level for recency filters — it reflects on-disk write time.
    """
    top = rec.get("timestamp")
    if isinstance(top, (int, float)):
        return float(top)
    nested = rec.get("router_input") if isinstance(rec.get("router_input"), dict) else None
    if nested is None:
        nested = rec.get("payload", {}).get("router_input") if isinstance(
            rec.get("payload"), dict
        ) else None
    if isinstance(nested, dict):
        inner = nested.get("timestamp")
        if isinstance(inner, (int, float)):
            return float(inner)
    return None


def _extract_snarc(rec: Mapping[str, Any]) -> Dict[str, float]:
    """Extract SNARC dict from a record.

    Looks in (in order):
      * ``rec['router_input']['snarc_*']`` — Track 1 schema — five flat keys
        (``snarc_surprise``, ``snarc_novelty``, ``snarc_arousal``,
        ``snarc_reward``, ``snarc_conflict``).
      * ``rec['router_input']['snarc']`` — nested dict form (used by
        Track 4's stub and any legacy callers).
      * ``rec['payload']['router_input'][...]`` — some capture paths wrap
        input+output in a ``payload`` envelope.
      * ``rec['snarc']`` — pruner-friendly top-level form.

    Missing dims are omitted (not zero-filled) so Welford ignores them
    rather than dragging the mean down for a dim that never captured.
    """
    candidates: List[Mapping[str, Any]] = []

    ri = rec.get("router_input")
    if isinstance(ri, dict):
        # Flat form wins when present.
        flat = {}
        for dim in SNARC_DIMENSIONS:
            key = f"snarc_{dim}"
            if key in ri:
                flat[dim] = ri[key]
        if flat:
            candidates.append(flat)
        nested = ri.get("snarc")
        if isinstance(nested, dict):
            candidates.append(nested)

    payload = rec.get("payload")
    if isinstance(payload, dict):
        ri2 = payload.get("router_input")
        if isinstance(ri2, dict):
            flat = {}
            for dim in SNARC_DIMENSIONS:
                key = f"snarc_{dim}"
                if key in ri2:
                    flat[dim] = ri2[key]
            if flat:
                candidates.append(flat)
            nested = ri2.get("snarc")
            if isinstance(nested, dict):
                candidates.append(nested)

    top_snarc = rec.get("snarc")
    if isinstance(top_snarc, dict):
        candidates.append(top_snarc)

    merged: Dict[str, float] = {}
    for cand in candidates:
        for dim in SNARC_DIMENSIONS:
            if dim not in merged and dim in cand:
                try:
                    merged[dim] = float(cand[dim])
                except (TypeError, ValueError):
                    pass
    return merged


def _extract_decision(
    rec: Mapping[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """Return (action, plugin_name) from a record, or (None, None)."""
    ro: Optional[Mapping[str, Any]] = None
    candidate = rec.get("router_output")
    if isinstance(candidate, dict):
        ro = candidate
    else:
        payload = rec.get("payload")
        if isinstance(payload, dict):
            c2 = payload.get("router_output")
            if isinstance(c2, dict):
                ro = c2
    if ro is None:
        return None, None
    action = ro.get("action")
    plugin = ro.get("plugin")
    if action is not None and not isinstance(action, str):
        action = str(action)
    if plugin is not None and not isinstance(plugin, str):
        plugin = str(plugin)
    return action, plugin


def _fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _fmt_bytes(b: Optional[int]) -> str:
    if b is None:
        return "—"
    if b < 1024:
        return f"{b} B"
    units = ["KB", "MB", "GB", "TB"]
    v = float(b)
    for u in units:
        v /= 1024.0
        if v < 1024.0 or u == units[-1]:
            return f"{v:.2f} {u}"
    return f"{b} B"


def _sparkline(counts: List[int]) -> str:
    """Turn a histogram list into an inline spark-string.

    Uses 8 block-intensity chars. Empty / all-zero → all spaces.
    """
    if not counts:
        return ""
    peak = max(counts)
    if peak <= 0:
        return " " * len(counts)
    glyphs = " ▁▂▃▄▅▆▇█"
    out_chars: List[str] = []
    for c in counts:
        # len(glyphs)-1 so full block only when c == peak.
        idx = int(round((c / peak) * (len(glyphs) - 1)))
        idx = max(0, min(idx, len(glyphs) - 1))
        out_chars.append(glyphs[idx])
    return "".join(out_chars)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


# ──────────────────────────────────────────────────────────────────────
# Welford helper — numerically stable mean/stddev + fixed-bin histogram
# ──────────────────────────────────────────────────────────────────────


class _Welford:
    """Running mean + variance via Welford's algorithm.

    Also accumulates a fixed-bin histogram. Range is [0, 1] except for
    ``reward`` which uses [-1, 1] — this is the only dim where the PRD
    allows negative values (§3.1).
    """

    __slots__ = (
        "dim",
        "n",
        "mean",
        "m2",
        "min_v",
        "max_v",
        "_hist",
        "_lo",
        "_hi",
    )

    def __init__(self, dim: str):
        self.dim = dim
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_v = float("inf")
        self.max_v = float("-inf")
        # Reward is the only dim with a signed range; treat everything
        # else as [0, 1]. See RouterInput doc-comments in Track 1.
        if dim == "reward":
            self._lo, self._hi = -1.0, 1.0
        else:
            self._lo, self._hi = 0.0, 1.0
        self._hist = [0] * SNARC_HISTOGRAM_BINS

    def push(self, x: float) -> None:
        x = float(x)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)
        if x < self.min_v:
            self.min_v = x
        if x > self.max_v:
            self.max_v = x
        # Histogram.
        rng = self._hi - self._lo
        if rng <= 0:
            idx = 0
        else:
            # Clamp to range — out-of-range values (rare; schema
            # guards against them but synthetic / forward-version
            # records may slip through) land in the edge bin.
            xc = max(self._lo, min(self._hi, x))
            bin_frac = (xc - self._lo) / rng
            idx = int(bin_frac * SNARC_HISTOGRAM_BINS)
            if idx >= SNARC_HISTOGRAM_BINS:
                idx = SNARC_HISTOGRAM_BINS - 1
        self._hist[idx] += 1

    def as_stats(self, bins: int = SNARC_HISTOGRAM_BINS) -> SnarcDimStats:
        if self.n == 0:
            return SnarcDimStats(
                dimension=self.dim,
                histogram=[0] * bins,
                bin_edges=_equal_bin_edges(self._lo, self._hi, bins),
            )
        var = self.m2 / self.n if self.n > 0 else 0.0
        return SnarcDimStats(
            dimension=self.dim,
            count=self.n,
            mean=self.mean,
            stddev=math.sqrt(var),
            min_value=self.min_v,
            max_value=self.max_v,
            histogram=list(self._hist),
            bin_edges=_equal_bin_edges(self._lo, self._hi, bins),
        )


def _equal_bin_edges(lo: float, hi: float, bins: int) -> List[Tuple[float, float]]:
    if bins <= 0:
        return []
    step = (hi - lo) / bins
    return [(lo + i * step, lo + (i + 1) * step) for i in range(bins)]


# ──────────────────────────────────────────────────────────────────────
# SNARC distribution-drift helpers (PRD §4.7.G)
# ──────────────────────────────────────────────────────────────────────


def _drift_dim_range(dim: str) -> Tuple[float, float]:
    """Return the (lo, hi) range for a SNARC dimension's drift histogram.

    Matches the Welford histogram convention: reward is the one signed
    dim, everything else is [0, 1]. Keeping the range aligned between the
    Welford summary and the drift histogram means operators can visually
    cross-reference without the x-axes disagreeing.
    """
    if dim == "reward":
        return (-1.0, 1.0)
    return (0.0, 1.0)


def _drift_bin_index(dim: str, value: float) -> int:
    """Bin a SNARC value into one of SNARC_HISTOGRAM_BINS equal-width bins.

    Out-of-range values clamp to the nearest edge bin (schema should
    prevent them; we defend-in-depth so a malformed record cannot raise).
    """
    lo, hi = _drift_dim_range(dim)
    rng = hi - lo
    if rng <= 0:
        return 0
    v = max(lo, min(hi, float(value)))
    frac = (v - lo) / rng
    idx = int(frac * SNARC_HISTOGRAM_BINS)
    if idx >= SNARC_HISTOGRAM_BINS:
        idx = SNARC_HISTOGRAM_BINS - 1
    if idx < 0:
        idx = 0
    return idx


def _kl_divergence(
    p_counts: List[int],
    q_counts: List[int],
    *,
    epsilon: float = DRIFT_SMOOTHING_EPSILON,
) -> float:
    """KL( P || Q ) in nats between two count histograms.

    Applies additive (Laplace-style) smoothing: every bin gets ``epsilon``
    added before normalization. This makes the KL well-defined when Q has
    zero-count bins — without it, a single empty bin in the serving
    histogram would push KL to infinity.

    Formula::

        p_i = (p_count_i + ε) / (Σp + N·ε)
        q_i = (q_count_i + ε) / (Σq + N·ε)
        KL  = Σ p_i · log(p_i / q_i)

    Returns a non-negative float. Identical distributions give exactly 0.0.
    """
    if len(p_counts) != len(q_counts):
        raise ValueError(
            f"histogram length mismatch: {len(p_counts)} vs {len(q_counts)}"
        )
    n_bins = len(p_counts)
    if n_bins == 0:
        return 0.0
    p_total = sum(p_counts) + epsilon * n_bins
    q_total = sum(q_counts) + epsilon * n_bins
    if p_total <= 0 or q_total <= 0:
        return 0.0
    kl = 0.0
    for pi, qi in zip(p_counts, q_counts):
        p = (pi + epsilon) / p_total
        q = (qi + epsilon) / q_total
        # p == 0 contributes 0 · log(0/q) = 0 by convention; with the
        # smoothing above p is always > 0 so the log is safe.
        kl += p * math.log(p / q)
    # Tiny negative values can arise from floating-point accumulation on
    # identical distributions; clamp at 0 to keep the reported metric
    # non-negative (KL is non-negative by construction).
    if kl < 0.0 and kl > -1e-12:
        return 0.0
    return kl


def _finalise_drift_report(
    *,
    machine: str,
    per_dim: Dict[str, Dict[str, List[int]]],
) -> SnarcDriftReport:
    """Build a SnarcDriftReport from per-dim {training,serving} histograms.

    Per-dim status (PRD §4.7.G discipline + agent-zero):

    - Either window < DRIFT_MIN_SAMPLES_PER_WINDOW records → INSUFFICIENT DATA
    - Otherwise compute KL(serving || training). ≥ threshold → DRIFT ALERT,
      else HEALTHY.

    Using KL(serving || training) (rather than the reverse) asks the
    question "how surprising is what the model sees now, given the
    baseline it learned?" — which is the directionality PRD §4.7.G
    implicitly calls for when it says "rolling-7-day serving distribution
    relative to training distribution".
    """
    report = SnarcDriftReport(machine=machine)
    any_alert = False
    all_insufficient = True

    for dim in SNARC_DIMENSIONS:
        dim_hists = per_dim.get(dim, {})
        training_hist = list(dim_hists.get("training", [0] * SNARC_HISTOGRAM_BINS))
        serving_hist = list(dim_hists.get("serving", [0] * SNARC_HISTOGRAM_BINS))
        training_count = sum(training_hist)
        serving_count = sum(serving_hist)
        lo, hi = _drift_dim_range(dim)
        bin_edges = _equal_bin_edges(lo, hi, SNARC_HISTOGRAM_BINS)

        if (
            training_count < DRIFT_MIN_SAMPLES_PER_WINDOW
            or serving_count < DRIFT_MIN_SAMPLES_PER_WINDOW
        ):
            report.dimensions[dim] = SnarcDriftMetrics(
                dimension=dim,
                status="INSUFFICIENT DATA",
                kl_nats=None,
                training_count=training_count,
                serving_count=serving_count,
                training_histogram=training_hist,
                serving_histogram=serving_hist,
                bin_edges=bin_edges,
            )
            continue

        all_insufficient = False
        kl = _kl_divergence(serving_hist, training_hist)
        status = "DRIFT ALERT" if kl >= DRIFT_ALERT_THRESHOLD_NATS else "HEALTHY"
        if status == "DRIFT ALERT":
            any_alert = True
        report.dimensions[dim] = SnarcDriftMetrics(
            dimension=dim,
            status=status,
            kl_nats=kl,
            training_count=training_count,
            serving_count=serving_count,
            training_histogram=training_hist,
            serving_histogram=serving_hist,
            bin_edges=bin_edges,
        )

    report.any_alert = any_alert
    report.awaiting_baseline = all_insufficient
    return report
