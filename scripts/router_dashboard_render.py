#!/usr/bin/env python3
"""
router_dashboard_render.py — render the Phase 0 router-pipeline dashboard.
=========================================================================

Reads a Track 4 router dataset (JSONL partitions under
``{base_dir}/{machine}/{YYYY-MM-DD}.jsonl[.gz]``) and writes a markdown
dashboard (+ optional JSON) to disk. Idempotent: overwrites outputs on
every run.

Designed for Track 7 to wire as a cron every hour (or nightly). Exits 0
on success, non-zero on argument / I/O failure.

Usage
-----

::

    python3 scripts/router_dashboard_render.py \\
        --base-dir /var/sage/router \\
        --machine '*' \\
        --output shared-context/arc-agi-3/phase2/brain-arch/router-pipeline-dashboard.md

See ``python3 scripts/router_dashboard_render.py --help`` for all flags.

Spec: Track 8 of shared-context/arc-agi-3/phase2/brain-arch/router-sprint-1-phase-0.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple


# Make ``sage`` importable when the script is invoked directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sage.cognition.router.dashboard import (  # noqa: E402
    DashboardBuilder,
    render_json,
    render_markdown,
)


# Default dashboard location — Track 7 schedules from this path.
_DEFAULT_OUTPUT = (
    _REPO_ROOT.parent
    / "shared-context"
    / "arc-agi-3"
    / "phase2"
    / "brain-arch"
    / "router-pipeline-dashboard.md"
)


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="router_dashboard_render",
        description=(
            "Render the Phase 0 router-pipeline observability dashboard. "
            "Every aggregate metric in the output is paired with a "
            "modal-class dummy baseline per PRD §7.10."
        ),
    )
    p.add_argument(
        "--base-dir",
        required=True,
        type=Path,
        help="Root of the router dataset (per-machine subdirs).",
    )
    p.add_argument(
        "--machine",
        default="*",
        help="Machine filter (e.g. 'sprout', 'thor', or '*' for all).",
    )
    p.add_argument(
        "--date-range",
        default=None,
        help=(
            "Inclusive date filter as 'YYYY-MM-DD:YYYY-MM-DD' or "
            "'last:N' (e.g. 'last:7' for last 7 days). "
            "Omit to read all partitions."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=(
            "Markdown output path. Overwritten each run. "
            f"Default: {_DEFAULT_OUTPUT}"
        ),
    )
    p.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help=(
            "Optional JSON output path (same metrics, JSON form). "
            "Skipped if not provided."
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Cron-friendly mode: suppress all stderr progress logging "
            "UNLESS a SNARC drift alert fires. Non-zero-signal output "
            "only — your inbox will thank you."
        ),
    )
    p.add_argument(
        "--snarc-drift",
        dest="snarc_drift",
        default=None,
        choices=("on", "off", "auto"),
        help=(
            "Control the SNARC drift section (PRD §4.7.G). "
            "'on' always emits, 'off' suppresses, 'auto' (default) emits "
            "when any dimension has enough data to evaluate."
        ),
    )
    return p.parse_args(argv)


def _resolve_date_range(spec: Optional[str]) -> Optional[Tuple[str, str]]:
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    if spec.startswith("last:"):
        try:
            n = int(spec.split(":", 1)[1])
        except ValueError as e:
            raise SystemExit(f"Invalid --date-range 'last:N' value: {spec}") from e
        if n < 0:
            raise SystemExit("--date-range last:N must be non-negative")
        end = datetime.utcnow().date()
        start = end - timedelta(days=max(0, n - 1))
        return (start.isoformat(), end.isoformat())
    if ":" not in spec:
        raise SystemExit(
            f"Invalid --date-range: {spec!r}. "
            "Use 'YYYY-MM-DD:YYYY-MM-DD' or 'last:N'."
        )
    start_s, end_s = spec.split(":", 1)
    # Validate parseable.
    try:
        datetime.strptime(start_s.strip(), "%Y-%m-%d")
        datetime.strptime(end_s.strip(), "%Y-%m-%d")
    except ValueError as e:
        raise SystemExit(f"Invalid --date-range dates: {e}") from e
    return (start_s.strip(), end_s.strip())


def _log(msg: str, *, quiet: bool, force: bool = False) -> None:
    """Emit a log line unless quiet is set.

    ``force=True`` is reserved for drift-alert signalling — in --quiet
    (cron) mode the ONLY stderr output should be a real event.
    """
    if force or not quiet:
        print(msg, file=sys.stderr, flush=True)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    date_range = _resolve_date_range(args.date_range)

    _log(
        f"[dashboard] base_dir={args.base_dir} machine={args.machine} "
        f"date_range={date_range}",
        quiet=args.quiet,
    )
    t0 = time.time()

    builder = DashboardBuilder(
        base_dir=args.base_dir,
        machine=args.machine,
        date_range=date_range,
    )
    metrics = builder.build()

    # --snarc-drift: 'auto' (default) includes the section and the section
    # itself self-censors to "awaiting baseline" when not enough data.
    include_drift: bool
    if args.snarc_drift in (None, "auto"):
        include_drift = True
    elif args.snarc_drift == "on":
        include_drift = True
    else:  # "off"
        include_drift = False

    md = render_markdown(metrics, include_drift=include_drift)

    # Idempotent overwrite — parent dir may not exist on first run.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Write via temp + rename so a killed cron doesn't leave a half-file.
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(md, encoding="utf-8")
    os.replace(tmp, args.output)

    if args.json_output is not None:
        js = render_json(metrics)
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        tmp_js = args.json_output.with_suffix(args.json_output.suffix + ".tmp")
        tmp_js.write_text(js, encoding="utf-8")
        os.replace(tmp_js, args.json_output)

    elapsed = time.time() - t0
    _log(
        f"[dashboard] wrote {args.output} "
        f"(records={metrics.aggregate.total_records}, "
        f"machines={len(metrics.per_machine)}, {elapsed*1000:.0f} ms)",
        quiet=args.quiet,
    )

    # Drift alert signalling — in --quiet mode this is the ONLY stderr
    # output the operator will see. Cron catches it via MAILTO / logging.
    if metrics.drift_aggregate.any_alert:
        drifting = sorted(
            dim for dim, d in metrics.drift_aggregate.dimensions.items()
            if d.status == "DRIFT ALERT"
        )
        _log(
            f"[dashboard] SNARC DRIFT ALERT (PRD §4.7.G) on aggregate: "
            f"dims={','.join(drifting)} — retraining flag",
            quiet=args.quiet,
            force=True,
        )
    for mach, rep in sorted(metrics.drift_per_machine.items()):
        if rep.any_alert:
            drifting = sorted(
                dim for dim, d in rep.dimensions.items()
                if d.status == "DRIFT ALERT"
            )
            _log(
                f"[dashboard] SNARC DRIFT ALERT (PRD §4.7.G) on {mach}: "
                f"dims={','.join(drifting)}",
                quiet=args.quiet,
                force=True,
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
