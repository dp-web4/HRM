"""Metabolic response to world input.

dp, 2026-09-07: "we should have metabolic response to events. beat is default idle state.
world inputs require engagement."

The 30-minute timer had become the only rhythm, so everything that arrived waited an
average of fifteen minutes for attention regardless of what it was. These pin the three
properties that make the response metabolic rather than an interrupt: it is GRADED, it is
REFRACTORY, and it always says WHY.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.gateway import arousal  # noqa: E402


def _inst(last_beat_ended_s_ago=None):
    d = Path(tempfile.mkdtemp(prefix="arousal-"))
    if last_beat_ended_s_ago is not None:
        t0 = time.time() - last_beat_ended_s_ago - 100
        (d / "heartbeats.jsonl").write_text(json.dumps({"t0": t0, "elapsed_s": 100}) + "\n")
    return d


def _quiet(monkey_running=False, next_s=3600):
    arousal.beat_running = lambda: monkey_running
    arousal.seconds_to_next_beat = lambda: next_s


def setup_function(_):
    _quiet()


def teardown_function(_):
    import importlib
    importlib.reload(arousal)


def test_it_is_graded_not_binary():
    """Not everything that arrives deserves ~18 minutes of the only GPU on the machine.
    A seat is deliberately BELOW the threshold: a seat can already reach the being at the
    next beat, so waking for one spends the being's attention on us."""
    inst = _inst(last_beat_ended_s_ago=3600)
    assert arousal.decide(inst, "dp_turn")["engage"] is True
    assert arousal.decide(inst, "peer_turn")["engage"] is True
    assert arousal.decide(inst, "scope_decided")["engage"] is True

    seat = arousal.decide(inst, "seat_turn")
    assert seat["engage"] is False and "below the engagement threshold" in seat["reason"]
    assert arousal.decide(inst, "digest")["engage"] is False
    # an unknown kind is quiet by default, never loud
    unknown = arousal.decide(inst, "something-new")
    assert unknown["engage"] is False and unknown["salience"] < arousal.ENGAGE_AT


def test_it_is_refractory_so_a_burst_is_not_five_beats():
    """Without this, five turns in a minute are five beats: the GPU thrashes and the
    being's attention is shredded across fragments of one exchange."""
    just_finished = _inst(last_beat_ended_s_ago=10)
    d = arousal.decide(just_finished, "dp_turn")
    assert d["engage"] is False
    assert "refractory" in d["reason"]
    assert d["refractory_s_left"] > 0

    rested = _inst(last_beat_ended_s_ago=arousal.REFRACTORY_S + 60)
    assert arousal.decide(rested, "dp_turn")["engage"] is True


def test_it_does_not_race_a_beat_that_is_already_coming():
    inst = _inst(last_beat_ended_s_ago=3600)
    _quiet(next_s=60)
    d = arousal.decide(inst, "dp_turn")
    assert d["engage"] is False and "already due" in d["reason"] and d["next_beat_s"] == 60


def test_a_running_beat_is_not_interrupted():
    inst = _inst(last_beat_ended_s_ago=3600)
    _quiet(monkey_running=True)
    d = arousal.decide(inst, "dp_turn")
    assert d["engage"] is False and "already running" in d["reason"]


def test_every_decision_says_why():
    """A wake policy that cannot say why it declined is indistinguishable from one that is
    broken — the failure this codebase keeps meeting from the other side."""
    inst = _inst(last_beat_ended_s_ago=3600)
    for kind in list(arousal.SALIENCE) + ["unknown-kind"]:
        d = arousal.decide(inst, kind)
        assert d["reason"], f"{kind} decided nothing out loud"
        assert isinstance(d["engage"], bool)
        assert 0.0 <= d["salience"] <= 1.0


def test_no_history_is_not_a_reason_to_refuse():
    """A being that has never beaten has no last-beat time. That must read as 'rested',
    not as 'unknown, therefore no' — a first world input should still land."""
    fresh = _inst(last_beat_ended_s_ago=None)
    assert arousal.decide(fresh, "dp_turn")["engage"] is True
