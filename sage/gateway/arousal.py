"""
Arousal — a metabolic response to world input.

dp, 2026-09-07: *"we should have metabolic response to events. beat is default idle state.
world inputs require engagement."*

That is a correction to what the heartbeat had become. A 30-minute timer is a fine IDLE
rhythm — it exists so a being has a reason to look around when nothing is happening — but
it had become the ONLY rhythm, which makes every arriving thing wait an average of fifteen
minutes for attention regardless of what it is. dp posted a turn and immediately asked how
frequent the beats were, which is the question you ask when the system has no answer to
"something just happened".

So: the timer is the floor, not the clock. A world input carries salience, and salience
above the engagement threshold wakes the being now.

WHAT MAKES THIS METABOLIC RATHER THAN AN INTERRUPT.

  * It is GRADED. Not everything that arrives deserves a beat. dp speaking directly is not
    the same event as a seat leaving a note, and neither is the same as a peer digest
    moving. Each carries a weight, and only weights above the threshold spend a beat.
  * It is REFRACTORY. After engaging, there is a period in which the being does not engage
    again, however loud the world gets. Without it a burst of five turns is five beats, the
    GPU thrashes, and the being's attention is shredded across fragments of the same
    conversation — the opposite of engagement.
  * It COSTS something. A beat is ~18 minutes of the only GPU on this machine. Waking is
    an expenditure and the record says what it was spent on, so "was that worth a beat"
    stays an answerable question instead of a feeling.

WHAT IT IS NOT: a way for anything outside to seize the being's attention on demand. The
threshold, the refractory period and the weights live here, in the seat's code, not in the
event. A caller says what happened; this decides what it is worth.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Optional

# What a kind of world input is worth, 0..1. These are a starting posture, not a finding:
# they are the seat's guess at what deserves ~18 minutes of GPU, and they should move when
# the record says they are wrong.
SALIENCE = {
    # The operator speaking directly. dp is asynchronous by rule, so a turn from dp is rare
    # and is the strongest signal the being gets that someone is actually present.
    "dp_turn": 0.9,
    # A peer being reaching across the mesh: another body, which is the only source of
    # facts this being cannot gather itself.
    "peer_turn": 0.7,
    # An operator ruling on something it asked for — it has been waiting, sometimes days.
    "scope_decided": 0.7,
    # A seat leaving a turn. Deliberately BELOW the threshold: a seat can already reach the
    # being at the next beat and usually just did something on its behalf. Waking for our
    # own convenience would be us spending its attention on us.
    "seat_turn": 0.4,
    # Ambient fleet movement. Real information, no urgency.
    "digest": 0.1,
}

ENGAGE_AT = 0.6          # at or above this, spend a beat now
REFRACTORY_S = 8 * 60    # after engaging, do not engage again this soon
IMMINENT_S = 4 * 60      # a beat already this close: let it arrive rather than racing it

UNIT = "sage-heartbeat.service"
TIMER = "sage-heartbeat.timer"


def _sh(*args: str) -> str:
    try:
        return subprocess.run(args, text=True, capture_output=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def beat_running() -> bool:
    return _sh("systemctl", "--user", "is-active", UNIT) in ("active", "activating")


def seconds_to_next_beat() -> Optional[int]:
    """From `list-timers --output=json`, which reports raw microseconds. The `show -p`
    forms are not usable here: NextElapseUSecRealtime is empty for an OnUnitActiveSec
    timer, and the monotonic one renders a human-readable duration."""
    try:
        rows = json.loads(_sh("systemctl", "--user", "list-timers", TIMER, "--output=json") or "[]")
        return int(int(rows[0]["next"]) / 1_000_000 - time.time())
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def last_beat_end(instance: Path) -> Optional[float]:
    try:
        rec = json.loads((Path(instance) / "heartbeats.jsonl")
                         .read_text(errors="replace").strip().splitlines()[-1])
        return float(rec["t0"]) + float(rec.get("elapsed_s") or 0)
    except Exception:
        return None


def decide(instance: Path, kind: str, *, now: Optional[float] = None) -> dict:
    """Should this event spend a beat? Returns the decision AND its reasoning, because a
    wake policy that cannot say why it declined is indistinguishable from one that is
    broken — the failure this codebase keeps meeting from the other side."""
    now = now if now is not None else time.time()
    sal = SALIENCE.get(kind, 0.2)
    d = {"kind": kind, "salience": sal, "engage": False, "reason": ""}

    if sal < ENGAGE_AT:
        d["reason"] = (f"salience {sal} is below the engagement threshold {ENGAGE_AT}; "
                       f"it will be read at the next scheduled beat")
        return d
    if beat_running():
        d["reason"] = "a beat is already running and will see this when it reads its state"
        return d

    since = None
    end = last_beat_end(instance)
    if end is not None:
        since = now - end
        if since < REFRACTORY_S:
            d["reason"] = (f"refractory: only {int(since)}s since the last beat ended "
                           f"({REFRACTORY_S}s). Engaging again this soon shreds attention "
                           f"across fragments of the same exchange")
            d["refractory_s_left"] = int(REFRACTORY_S - since)
            return d

    nxt = seconds_to_next_beat()
    if nxt is not None and 0 <= nxt <= IMMINENT_S:
        d["reason"] = f"a beat is already due in {nxt}s; racing it would waste one"
        d["next_beat_s"] = nxt
        return d

    d["engage"] = True
    d["reason"] = f"salience {sal} >= {ENGAGE_AT} and the being is idle"
    if since is not None:
        d["idle_s"] = int(since)
    return d


def respond(instance: Path, kind: str, *, descriptor: str) -> dict:
    """Register a world input and, if it earns one, wake the being now.

    The wake marker is written whatever the decision, so a beat that arrives on the timer
    still learns that something specific happened and what it was. Only the systemd start
    is conditional."""
    from sage.gateway.being_join import write_wake_marker
    d = decide(instance, kind)
    d["descriptor"] = descriptor
    try:
        write_wake_marker(descriptor, d["salience"])
    except Exception as e:
        d["marker_error"] = f"{type(e).__name__}: {e}"
    if d["engage"]:
        out = _sh("systemctl", "--user", "start", "--no-block", UNIT)
        d["started"] = True
        if out:
            d["systemctl"] = out
    return d
