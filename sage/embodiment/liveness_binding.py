"""Sprout embodiment → six-rung liveness ladder: the schema binding.

This is the WIRING JOB the transfer map's step 2 names (Thor #2: lift
organism/liveness.py verbatim, bind artifact schemas per organ here — never
inside the lifted modules). It reads the embodiment pipeline's EXISTING
artifacts; it does not touch the live cortex.

Binding (organ: vision → raising, on Sprout):

  rung        source artifact                          meaning here
  --------    -------------------------------------    ---------------------------
  enabled     ~/.sprout/perception.json exists         the organ is installed
  entered     perception.json mtime fresh (<30s)       the organ is sensing NOW
  produced    perception_journal.jsonl 'salient' rows  it produced salient percepts
  admitted    presence_log.jsonl 'noticed' rows        presence delivered a wake to
                                                       the being (past bar+cooldown)
  used        U/S — needs the daemon-side binding      (did sage-daemon record it
                                                       as experience? next wiring)
  affected    U/S — needs raising-outcome binding      (did a session's behavior
                                                       change because of it?)

U/S is per INSTRUMENT_SCAN's closing rule: an instrument with no source reads
UNSERVICEABLE, not zero. Rungs 5-6 unbound is the honest current state — the
exact gap ("vision was on, but did it matter?") this ladder exists to close,
now visible on a panel instead of unaskable.

Flow accounting (Rule 2, silent-resolution guard): salient events are the IN,
presence wakes are the OUT, and the dropped difference is attributed to the
presence filter's declared reasons (bar / cooldown / hourly cap) rather than
silently resolved.

Usage:
    python -m sage.embodiment.liveness_binding [--window-h 24] [--out PATH]
    python -m sage.organism.scan --liveness ~/.sprout/liveness.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

from sage.organism.liveness import Ladder

PERCEPTION = os.path.expanduser("~/.sprout/perception.json")
JOURNAL = os.path.expanduser("~/.sprout/perception_journal.jsonl")
PRESENCE_LOG = os.path.expanduser("~/.sprout/presence_log.jsonl")
OUT_DEFAULT = os.path.expanduser("~/.sprout/liveness.json")

FRESH_S = 30.0  # 'entered' bar: cortex writes at ~4Hz; presence treats >10s as stale


def _jsonl(path: str, since: float):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if d.get("ts", 0) >= since:
                yield d


def bind(window_h: float = 24.0, now: float | None = None) -> Ladder:
    now = time.time() if now is None else now
    since = now - window_h * 3600
    lad = Ladder(run_id=f"sprout-embodiment-{int(now)}")

    organ = "vision->raising"

    # rung 1: enabled — the organ is installed
    if os.path.exists(PERCEPTION):
        lad.mark(organ, "enabled", detail="perception.json present")
    else:
        return lad  # nothing else can be read; ladder honestly stops at rung 0

    # rung 2: entered — sensing now
    age = now - os.path.getmtime(PERCEPTION)
    if age < FRESH_S:
        lad.mark(organ, "entered", detail=f"perception.json age {age:.1f}s")

    # rung 3: produced — salient percepts in the window
    salient = [d for d in _jsonl(JOURNAL, since) if d.get("kind") == "salient"]
    if salient:
        lad.mark(organ, "produced", n=len(salient),
                 detail=f"{len(salient)} salient events in {window_h:.0f}h")

    # rung 4: admitted — presence delivered wakes to the being
    noticed = [d for d in _jsonl(PRESENCE_LOG, since) if d.get("kind") == "noticed"]
    if noticed:
        lad.mark(organ, "admitted", n=len(noticed),
                 detail=f"{len(noticed)} wakes delivered in {window_h:.0f}h")

    # rungs 5-6: U/S — no source bound yet. Deliberately NOT marked; the scan
    # renders the ladder stopping at 'admitted', which is the honest reading.
    # Next bindings: 'used' <- sage-daemon experience records referencing a wake;
    # 'affected' <- raising-session outcome deltas attributable to a percept.

    # flow (in -> out -> dropped), with the drop attributed, not silently resolved
    lad.flow(organ, n_in=len(salient), n_out=len(noticed),
             reason="presence filter: salience bar + 300s cooldown + 6/h cap")
    return lad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window-h", type=float, default=24.0)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    lad = bind(window_h=args.window_h)
    lad.save(args.out)
    print(lad.report())
    print(f"\nliveness -> {args.out}")
    print("panel:      python -m sage.organism.scan --liveness " + args.out)


if __name__ == "__main__":
    main()
