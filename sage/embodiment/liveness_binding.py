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
  used        instances/sprout-*/experience_buffer_    the being recorded the wake
              rs.jsonl joined to wakes                 as experience (M1 binding)
  affected    U/S — needs raising-outcome binding      (did a session's behavior
                                                       change because of it?)

M1 attribution rule (STATED BEFORE FIRST COMPUTATION, 2026-07-29, per PRD M2
discipline applied early): a wake is `used` iff an experience record exists with
  timestamp in [wake.ts, wake.ts + 90s]     (daemon /chat responds inside
                                             presence's 60s timeout + margin)
  AND word_overlap(descriptor, prompt) >= 0.5  (Jaccard on lowercased word sets;
                                             the daemon stores the wake message
                                             as `prompt`, possibly re-prefixed)
Each experience matches at most one wake (nearest earlier). Unmatched wakes are
dropped-with-reason; experiences with no wake in range are the daemon's own
cycles, not evidence about this channel. The rule is versioned here; changing it
requires changing this text, not just the code.

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
# the being's experience buffer (sage-daemon, sage-rs main.rs::experience_path)
EXPERIENCE_GLOB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "instances", "sprout-*", "experience_buffer_rs.jsonl")

FRESH_S = 30.0  # 'entered' bar: cortex writes at ~4Hz; presence treats >10s as stale
USED_WINDOW_S = 90.0   # M1 rule: experience must land within this of its wake
USED_OVERLAP = 0.5     # M1 rule: word-Jaccard(descriptor, prompt) floor


def _word_jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _match_wakes_to_experiences(wakes: list[dict], experiences: list[dict]):
    """Apply the M1 attribution rule. Returns (matched_pairs, unmatched_wakes)."""
    unclaimed = sorted(experiences, key=lambda e: e.get("timestamp", 0))
    matched, unmatched = [], []
    for w in sorted(wakes, key=lambda d: d.get("ts", 0)):
        hit = None
        for e in unclaimed:
            dt = e.get("timestamp", 0) - w.get("ts", 0)
            if 0 <= dt <= USED_WINDOW_S and \
               _word_jaccard(w.get("descriptor", ""), e.get("prompt", "")) >= USED_OVERLAP:
                hit = e
                break
        if hit is not None:
            unclaimed.remove(hit)  # each experience claims at most one wake
            matched.append((w, hit))
        else:
            unmatched.append(w)
    return matched, unmatched


def _jsonl_ts(path: str, since: float, ts_key: str = "ts"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if d.get(ts_key, 0) >= since:
                yield d


def _jsonl(path: str, since: float):
    yield from _jsonl_ts(path, since, ts_key="ts")


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

    # rung 5: used — the being recorded the wake as experience (M1 rule above)
    import glob as _glob
    experiences = []
    for path in _glob.glob(EXPERIENCE_GLOB):
        experiences.extend(d for d in _jsonl_ts(path, since, ts_key="timestamp"))
    matched, unmatched = _match_wakes_to_experiences(noticed, experiences)
    if matched:
        lad.mark(organ, "used", n=len(matched),
                 detail=f"{len(matched)}/{len(noticed)} wakes became experience "
                        f"(rule: <={USED_WINDOW_S:.0f}s + overlap>={USED_OVERLAP})")

    # rung 6: U/S — no source bound yet ('affected' <- raising-session outcome
    # deltas attributable to a percept; PRD M2, needs a second witness).

    # flow (in -> out -> dropped), with each drop attributed, not silently resolved
    lad.flow(organ, n_in=len(salient), n_out=len(noticed),
             reason="presence filter: salience bar + 300s cooldown + 6/h cap")
    lad.flow(organ, n_in=len(noticed), n_out=len(matched),
             reason="M1 join: no experience recorded (measured 2026-07-29: the "
                    "being's capture gate — wakes sal>=0.53 all recorded, "
                    "<=0.50 none; daemon downtime also lands here)")
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
