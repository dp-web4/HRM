"""
FleetAggregator — Sprint 2 R3. Legion (data czar) runs this nightly to
pull per-machine router shards and build a fleet-wide training corpus.

Design (per router-sprint-2-rollout-federation.md):

  * **Pull-model**. Legion walks each peer's shard dir, pulls records,
    writes a single partitioned aggregate at
    ``{aggregate_dir}/{YYYY-MM-DD}.jsonl.gz``.
  * **Transport agnostic** per peer: ``local`` reads the filesystem
    directly, ``ssh`` stages shards into a scratch dir via ``rsync``
    (no credential management — relies on ssh-agent).
  * **Deduplication by ``record_id``**. Records are unique across
    machines; collisions are logged as anomalies. First-seen wins so
    re-runs are idempotent (the aggregator loads existing aggregate
    record_ids into the seen-set before pulling).
  * **Failure isolation**: a peer being offline / unreadable / empty
    does not stop aggregation of other peers. Each peer is wrapped in
    ``try/except`` and its status lands in the run summary.
  * **Source-stamp preservation**: every record is written through
    verbatim. The R1 ``metadata.source`` field (and anything else) is
    preserved untouched — no re-mapping, no normalization.
  * **Schema-version tolerance**: v0.1.0 records (no metadata.source)
    and v0.2.0 records (with metadata.source from R1) are both passed
    through. The aggregator does not validate or rewrite the schema.
  * **Atomic writes**: aggregate is written to ``{date}.jsonl.gz.tmp``
    then ``os.replace``'d. A crash mid-run leaves the previous
    aggregate intact.
  * **Respects PRD §5.6**: pulls from ALREADY-pruned shards. The
    aggregator does not prune, does not consult pin-kinds — that's the
    per-machine pruner's job (tracks 9 / T8).

Intentional non-goals:
  * No torch dependency.
  * No network daemon / push path. Nightly cron, pull-only.
  * No cross-day reconciliation. One aggregate file per UTC date.
  * No outcome sidecar merge (reader does that at replay time).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Set, Tuple, Union

from sage.cognition.router.data.reader import RouterDatasetReader


_log = logging.getLogger(__name__)


# ── config model ───────────────────────────────────────────────────


@dataclass
class PeerConfig:
    """One peer's configuration entry.

    Attributes
    ----------
    machine:
        Peer machine name (matches fleet manifest + shard subdir name).
    transport:
        ``local`` or ``ssh``. ``local`` reads the directory directly;
        ``ssh`` stages it via ``rsync`` first.
    path:
        Shard root (may contain ``~`` or env vars — expanded at use time).
    host:
        SSH host (required when ``transport == "ssh"``).
    ssh_user:
        Optional SSH user override. Empty → use current user.
    ssh_port:
        Optional SSH port. None → default 22.
    """

    machine: str
    transport: str = "local"
    path: str = ""
    host: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_port: Optional[int] = None

    def __post_init__(self) -> None:
        if self.transport not in ("local", "ssh"):
            raise ValueError(
                f"peer {self.machine!r}: transport must be 'local' or 'ssh', "
                f"got {self.transport!r}"
            )
        if self.transport == "ssh" and not self.host:
            raise ValueError(
                f"peer {self.machine!r}: transport='ssh' requires 'host'"
            )
        if not self.machine:
            raise ValueError("peer entries must have 'machine' set")
        if not self.path:
            raise ValueError(
                f"peer {self.machine!r}: 'path' must be non-empty"
            )


@dataclass
class FederationConfig:
    """Top-level config loaded from ``fleet_shards.json``."""

    peers: List[PeerConfig] = field(default_factory=list)
    aggregate_dir: str = ""
    schedule_utc: str = "02:00"

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "FederationConfig":
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FederationConfig":
        peers = [PeerConfig(**entry) for entry in data.get("peers", [])]
        return cls(
            peers=peers,
            aggregate_dir=data.get("aggregate_dir", ""),
            schedule_utc=data.get("schedule_utc", "02:00"),
        )


# ── per-peer summary ───────────────────────────────────────────────


@dataclass
class PeerSummary:
    """Result of aggregating one peer's shard for the target date."""

    machine: str
    transport: str
    available: bool = False
    records_seen: int = 0
    records_new: int = 0
    duplicates: int = 0
    errors: List[str] = field(default_factory=list)
    snarc_mean: Dict[str, float] = field(default_factory=dict)
    decision_class_counts: Dict[str, int] = field(default_factory=dict)
    source_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateSummary:
    """Top-level result of one aggregation run."""

    target_date: str
    aggregate_path: Optional[str]
    total_records: int = 0
    total_new: int = 0
    duplicates: int = 0
    cross_machine_collisions: int = 0
    peers: List[PeerSummary] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # asdict recurses into peers → already serialized
        return d


# ── aggregator ─────────────────────────────────────────────────────


class FleetAggregator:
    """Pulls per-machine router shards and builds a fleet-wide corpus.

    Parameters
    ----------
    config:
        ``FederationConfig`` — list of peers + aggregate dir.
    clock:
        Optional callable returning current UTC datetime. Tests inject
        a fixed clock; production uses ``datetime.now(UTC)``.
    ssh_command:
        The ssh binary (default ``"ssh"``). Swap for tests.
    rsync_command:
        The rsync binary (default ``"rsync"``). Swap for tests.

    Notes
    -----
    * The aggregator is stateless across runs. All state it needs
      (seen record_ids, existing aggregate content) is loaded from
      disk at the start of each ``run``.
    * The default target date is YESTERDAY (UTC) — shards for "today"
      may still be getting appends from the peer's live daemon.
    """

    #: Per-record_id collision log — populated during a run, cleared
    #: between runs. Kept as an instance field for introspection tests.
    cross_machine_collisions: Dict[str, List[str]]

    def __init__(
        self,
        config: FederationConfig,
        clock: Optional[Any] = None,
        ssh_command: str = "ssh",
        rsync_command: str = "rsync",
    ) -> None:
        if not config.aggregate_dir:
            raise ValueError("FederationConfig.aggregate_dir must be set")
        self.config = config
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._ssh = ssh_command
        self._rsync = rsync_command
        self.cross_machine_collisions = {}

    # ── public entry points ────────────────────────────────────────

    def run(
        self,
        target_date: Optional[Union[str, date]] = None,
        dry_run: bool = False,
    ) -> AggregateSummary:
        """Aggregate one UTC date's shards from all peers.

        Parameters
        ----------
        target_date:
            ``YYYY-MM-DD`` or ``datetime.date``. Default: yesterday UTC.
        dry_run:
            If True, pull + enumerate but write nothing to disk.
        """
        started = time.time()
        target = _coerce_date(target_date) if target_date else self._yesterday()
        target_str = target.isoformat()
        self.cross_machine_collisions = {}

        summary = AggregateSummary(
            target_date=target_str,
            aggregate_path=None,
            started_at=started,
            dry_run=dry_run,
        )

        aggregate_path = self._aggregate_path_for(target)
        # Ensure aggregate dir exists (no-op in dry-run).
        if not dry_run:
            try:
                aggregate_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                _log.error("Cannot create aggregate dir %s: %s", aggregate_path.parent, e)
                summary.finished_at = time.time()
                return summary

        # Load existing aggregate record_ids so re-runs are idempotent.
        # `seen` accumulates record_ids observed ANYWHERE in this aggregation
        # (existing aggregate file + earlier peers in this run).
        # `seen_machine` maps record_id → first machine seen, for collision logs.
        seen: Set[str] = set()
        seen_machine: Dict[str, str] = {}
        existing_records: List[Dict[str, Any]] = []
        if aggregate_path.exists():
            for rec in _iter_jsonl_gz(aggregate_path):
                rid = rec.get("record_id")
                if not rid:
                    # No record_id → can't dedup. Preserve verbatim.
                    existing_records.append(rec)
                    continue
                if rid in seen:
                    continue
                seen.add(rid)
                seen_machine[rid] = rec.get("machine", "unknown")
                existing_records.append(rec)

        all_new_records: List[Dict[str, Any]] = []

        for peer in self.config.peers:
            peer_summary = self._aggregate_peer(
                peer=peer,
                target_date=target,
                seen=seen,
                seen_machine=seen_machine,
                new_records_out=all_new_records,
            )
            summary.peers.append(peer_summary)

        # Collision total: number of record_ids that showed up on more than
        # one machine this run.
        summary.cross_machine_collisions = len(self.cross_machine_collisions)

        total_records = len(existing_records) + len(all_new_records)
        summary.total_records = total_records
        summary.total_new = len(all_new_records)
        summary.duplicates = sum(p.duplicates for p in summary.peers)

        if dry_run:
            _log.info(
                "DRY-RUN aggregate %s: existing=%d new=%d peers=%d",
                target_str, len(existing_records), len(all_new_records),
                len(summary.peers),
            )
            summary.aggregate_path = str(aggregate_path)
            summary.finished_at = time.time()
            return summary

        # Write atomically: existing + new into a tmp, then os.replace.
        tmp_path = aggregate_path.with_suffix(aggregate_path.suffix + ".tmp")
        try:
            with gzip.open(str(tmp_path), "wt", encoding="utf-8") as f:
                for rec in existing_records:
                    f.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")
                for rec in all_new_records:
                    f.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")
            os.replace(tmp_path, aggregate_path)
            summary.aggregate_path = str(aggregate_path)
        except OSError as e:
            _log.exception("Failed to write aggregate %s: %s", aggregate_path, e)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

        # Write per-peer summary sidecar next to the aggregate.
        try:
            summary_path = aggregate_path.with_name(
                aggregate_path.name.replace(".jsonl.gz", ".summary.json")
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary.to_dict(), f, indent=2, default=str)
        except OSError as e:  # pragma: no cover — defensive
            _log.warning("Failed to write summary sidecar: %s", e)

        summary.finished_at = time.time()
        _log.info(
            "aggregate %s: total=%d new=%d peers=%d collisions=%d elapsed=%.2fs",
            target_str, summary.total_records, summary.total_new,
            len(summary.peers), summary.cross_machine_collisions,
            summary.finished_at - summary.started_at,
        )
        return summary

    # ── internals ──────────────────────────────────────────────────

    def _yesterday(self) -> date:
        """UTC yesterday — default aggregation target."""
        return (self._clock() - timedelta(days=1)).date()

    def _aggregate_path_for(self, target: date) -> Path:
        return Path(self.config.aggregate_dir) / f"{target.isoformat()}.jsonl.gz"

    def _aggregate_peer(
        self,
        peer: PeerConfig,
        target_date: date,
        seen: Set[str],
        seen_machine: Dict[str, str],
        new_records_out: List[Dict[str, Any]],
    ) -> PeerSummary:
        """Pull one peer's shard for the target date, appending
        not-yet-seen records to ``new_records_out``.

        Never raises — all errors go into ``peer_summary.errors`` and
        the peer is marked unavailable.
        """
        peer_summary = PeerSummary(
            machine=peer.machine,
            transport=peer.transport,
        )
        staging_dir: Optional[Path] = None
        try:
            if peer.transport == "local":
                shard_dir = Path(os.path.expanduser(os.path.expandvars(peer.path)))
                if not shard_dir.exists():
                    peer_summary.errors.append(f"shard dir missing: {shard_dir}")
                    _log.warning(
                        "peer %s: shard dir missing %s — skipping",
                        peer.machine, shard_dir,
                    )
                    return peer_summary
                source_base = shard_dir
            else:
                staging_dir = Path(tempfile.mkdtemp(prefix="fleetagg_"))
                ok = self._rsync_pull(peer, target_date, staging_dir, peer_summary)
                if not ok:
                    return peer_summary
                source_base = staging_dir

            peer_summary.available = True

            # Records live at {source_base}/{machine}/{target}.jsonl[.gz].
            # Be tolerant: the peer may write directly to {source_base}
            # without a machine subdir (self-rooted layout). Try both.
            candidate_dirs = [source_base / peer.machine, source_base]
            partitions: List[Path] = []
            for d in candidate_dirs:
                if not d.exists():
                    continue
                for ext in (".jsonl.gz", ".jsonl"):
                    p = d / f"{target_date.isoformat()}{ext}"
                    if p.exists():
                        partitions.append(p)

            if not partitions:
                peer_summary.errors.append(
                    f"no partition found for {target_date.isoformat()} "
                    f"under {source_base}"
                )
                _log.info(
                    "peer %s: no partition for %s — empty day (ok)",
                    peer.machine, target_date,
                )
                return peer_summary

            for part in partitions:
                for rec in _iter_partition(part):
                    peer_summary.records_seen += 1
                    self._tally_record(rec, peer_summary)

                    rid = rec.get("record_id")
                    if not rid:
                        # No record_id → keep it but cannot dedup.
                        new_records_out.append(rec)
                        peer_summary.records_new += 1
                        continue
                    if rid in seen:
                        peer_summary.duplicates += 1
                        first_machine = seen_machine.get(rid, "unknown")
                        rec_machine = rec.get("machine", peer.machine)
                        if first_machine != rec_machine:
                            # Same record_id, different machine → anomaly.
                            bucket = self.cross_machine_collisions.setdefault(rid, [first_machine])
                            if rec_machine not in bucket:
                                bucket.append(rec_machine)
                            _log.warning(
                                "cross-machine record_id collision %s: first=%s dup=%s",
                                rid, first_machine, rec_machine,
                            )
                        continue
                    seen.add(rid)
                    seen_machine[rid] = rec.get("machine", peer.machine)
                    new_records_out.append(rec)
                    peer_summary.records_new += 1

        except Exception as e:  # pragma: no cover — defensive
            _log.exception("peer %s: unexpected error: %s", peer.machine, e)
            peer_summary.errors.append(f"unexpected: {e}")
        finally:
            if staging_dir is not None and staging_dir.exists():
                try:
                    shutil.rmtree(staging_dir)
                except OSError:  # pragma: no cover
                    pass

        # Finalize mean SNARC (sum → mean after all records seen).
        if peer_summary.records_seen > 0 and peer_summary.snarc_mean:
            for k in list(peer_summary.snarc_mean.keys()):
                peer_summary.snarc_mean[k] /= peer_summary.records_seen

        return peer_summary

    def _rsync_pull(
        self,
        peer: PeerConfig,
        target_date: date,
        staging_dir: Path,
        peer_summary: PeerSummary,
    ) -> bool:
        """rsync peer's shard dir into ``staging_dir``. Returns True on
        success, False on failure (which is recorded in peer_summary).
        """
        remote_path = peer.path.rstrip("/") + "/"
        host = peer.host or ""
        user_prefix = f"{peer.ssh_user}@" if peer.ssh_user else ""
        src = f"{user_prefix}{host}:{remote_path}"

        ssh_flags = [self._ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        if peer.ssh_port:
            ssh_flags.extend(["-p", str(peer.ssh_port)])
        ssh_cmd = " ".join(ssh_flags)

        cmd = [
            self._rsync,
            "-az",
            "--timeout=30",
            "-e", ssh_cmd,
            src,
            str(staging_dir) + "/",
        ]
        try:
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, timeout=120,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            peer_summary.errors.append(f"rsync: {e}")
            _log.warning("peer %s: rsync failed (%s)", peer.machine, e)
            return False
        if result.returncode != 0:
            tail = (result.stderr or "").strip().splitlines()[-3:]
            peer_summary.errors.append(
                f"rsync rc={result.returncode}: {' | '.join(tail)}"
            )
            _log.warning(
                "peer %s: rsync rc=%d (%s)",
                peer.machine, result.returncode, tail,
            )
            return False
        return True

    @staticmethod
    def _tally_record(rec: Mapping[str, Any], peer_summary: PeerSummary) -> None:
        """Update running aggregates (SNARC sum, decision class, source)."""
        payload = rec.get("payload") or {}
        snarc = None
        ri = payload.get("router_input") if isinstance(payload, Mapping) else None
        if isinstance(ri, Mapping):
            snarc = ri.get("snarc")
        if isinstance(snarc, Mapping):
            for dim in ("surprise", "novelty", "arousal", "reward", "conflict"):
                v = snarc.get(dim)
                if isinstance(v, (int, float)):
                    peer_summary.snarc_mean[dim] = (
                        peer_summary.snarc_mean.get(dim, 0.0) + float(v)
                    )

        ro = payload.get("router_output") if isinstance(payload, Mapping) else None
        if isinstance(ro, Mapping):
            action = ro.get("action") or ro.get("decision_class")
            if isinstance(action, str):
                peer_summary.decision_class_counts[action] = (
                    peer_summary.decision_class_counts.get(action, 0) + 1
                )

        # R1 source stamp — metadata.source. Tolerate both v0.1.0 (missing)
        # and v0.2.0 (present) without complaining.
        metadata = rec.get("metadata") if isinstance(rec, Mapping) else None
        src_val: Optional[str] = None
        if isinstance(metadata, Mapping):
            s = metadata.get("source")
            if isinstance(s, str):
                src_val = s
        if src_val is None:
            src_val = "unknown"
        peer_summary.source_counts[src_val] = (
            peer_summary.source_counts.get(src_val, 0) + 1
        )


# ── module-level helpers ───────────────────────────────────────────


def _coerce_date(d: Union[str, date]) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(str(d), "%Y-%m-%d").date()


def _iter_partition(path: Path) -> Iterator[Dict[str, Any]]:
    """Iterate records from a single .jsonl or .jsonl.gz partition.

    Reuses RouterDatasetReader.read_file semantics — corrupt trailing
    lines are logged and skipped, not raised.
    """
    # Dummy base_dir; we only call read_file.
    reader = RouterDatasetReader(base_dir=path.parent)
    yield from reader.read_file(path)


def _iter_jsonl_gz(path: Path) -> Iterator[Dict[str, Any]]:
    """Iterate records from an existing aggregate (.jsonl.gz)."""
    try:
        with gzip.open(str(path), "rt", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    _log.warning(
                        "aggregate: corrupt line %s:%d — %s", path, line_no, e,
                    )
    except OSError as e:
        _log.warning("aggregate: error reading %s: %s", path, e)


# ── CLI ────────────────────────────────────────────────────────────


def _default_config_path() -> Path:
    """Default location for the fleet config (relative to SAGE repo)."""
    # sage/cognition/router/data/federation.py → repo root is 4 parents up.
    here = Path(__file__).resolve()
    repo_root = here.parents[4]
    return repo_root / "sage" / "gateway" / "fleet_shards.json"


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Cron line installed by ``scripts/router_federation_install.sh``::

        0 2 * * * /usr/bin/python3 -m sage.cognition.router.data.federation --run
    """
    parser = argparse.ArgumentParser(
        prog="python -m sage.cognition.router.data.federation",
        description="Fleet aggregator: pull per-machine router shards.",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Run aggregation (default if no other mode flag passed)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Enumerate and pull (for ssh peers) but write nothing",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to fleet_shards config. Default: "
             "SAGE/sage/gateway/fleet_shards.json",
    )
    parser.add_argument(
        "--date", default=None,
        help="Target UTC date (YYYY-MM-DD). Default: yesterday UTC.",
    )
    parser.add_argument(
        "--days", type=int, default=1,
        help="Number of consecutive days to aggregate, ending at --date "
             "(inclusive). Default: 1.",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase log verbosity (repeat for more).",
    )
    args = parser.parse_args(argv)

    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = Path(args.config) if args.config else _default_config_path()
    if not config_path.exists():
        _log.error("config not found: %s", config_path)
        return 2

    config = FederationConfig.from_file(config_path)
    aggregator = FleetAggregator(config=config)

    # Build date list — [date - days + 1, ..., date].
    end = _coerce_date(args.date) if args.date else aggregator._yesterday()
    days = max(1, args.days)
    dates = [end - timedelta(days=offset) for offset in range(days - 1, -1, -1)]

    # Exit 0 even if some peers were unavailable — partial aggregation
    # is the whole point of failure isolation. We only signal error
    # status when NO peer on the most recent aggregated date reported
    # `available=True` (total fleet outage).
    last_summary: Optional[AggregateSummary] = None
    for d in dates:
        summary = aggregator.run(target_date=d, dry_run=args.dry_run)
        last_summary = summary
        # Machine-readable stdout so cron / operators can grep.
        print(json.dumps(summary.to_dict(), default=str))

    if last_summary is not None and last_summary.peers and all(
        not p.available for p in last_summary.peers
    ):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
