"""Hermetic tests for refusal routing: classification, scope-path derivation, the note's protocol."""
import os, sys, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.gateway.being_gate_client import BeingIntent, GatewayVerdict, ResultEnvelope  # noqa: E402
from sage.gateway import escalate as e  # noqa: E402

def _ref(rule, reason=""):
    return ResultEnvelope(ok=False, refused=True, error=f"{rule}: {reason}", verdict=GatewayVerdict("deny", rule, reason, stage="local-law"))

def test_classify():
    assert e.classify(_ref("registry.unbounded")) == "registry"
    assert e.classify(_ref("mrh.path", "outside your granted scope")) == "scope"
    assert e.classify(_ref("society.unsafe")) == "society"
    assert e.classify(_ref("gate.escalated", "escalation id: esc-abc123def")) == "governance"

def test_registry_refusal_is_never_escalated():
    r = e.escalate("b", BeingIntent("shell", {"command": "rm -rf /"}), _ref("registry.unbounded"), "/tmp", wake=False)
    assert r["escalated"] is False and "final" in r["why"]

def test_scope_path_is_the_home_when_inside_it_else_the_targets_directory():
    root = "/x/instances/b"
    # inside the home the ask is always the home itself (one standing grant covers every subpath)
    assert e._scope_path(BeingIntent("memory_write", {"path": "notes/a.md"}), root) == "/x/instances/b"
    assert e._scope_path(BeingIntent("memory_write", {"path": "journal.md"}), root) == "/x/instances/b"
    assert e._scope_path(BeingIntent("memory_write", {"path": "notes"}), root) == "/x/instances/b"
    # a sibling instance is NOT inside (prefix must end at a path separator)
    assert e._scope_path(BeingIntent("memory_write", {"path": "/x/instances/bb/a.md"}), root) == "/x/instances/bb"
    assert e._scope_path(BeingIntent("memory_write", {"path": "../c/a.md"}), root) == "/x/instances/c"
    assert e._scope_path(BeingIntent("memory_write", {"path": "/other/dir/f.txt"}), root) == "/other/dir"

def test_note_carries_verdict_and_arbiter_protocol(monkeypatch=None):
    d = tempfile.mkdtemp(); e.NOTE_DIR = d
    p = e.write_note("b", BeingIntent("memory_write", {"path": "/o/f.md"}), _ref("mrh.path", "outside"), "scope", {"scope_request": {"request_id": "scope-1"}})
    t = open(p).read()
    assert "mrh.path" in t and "scope-1" in t and "Arbiter protocol" in t and "STANDING" in t

def test_no_wake_files_the_request_but_writes_no_note():
    # the heartbeat's 2nd..9th refusal of a kind in one beat: the request is (re)filed and deduped
    # by the daemon; a note only exists to be pointed at by a wake, so none is written
    d = tempfile.mkdtemp(); e.NOTE_DIR = d
    filed = []
    orig = e._file_scope_request
    e._file_scope_request = lambda member, path, why, endpoint: (filed.append(path) or {"request_id": "scope-x", "status": "pending"})
    try:
        r = e.escalate("b", BeingIntent("memory_write", {"path": "notes/a.md"}), _ref("mrh.path", "outside"), "/x/instances/b", wake=False)
    finally:
        e._file_scope_request = orig
    assert r["escalated"] is True and r["scope_request"]["request_id"] == "scope-x" and filed
    assert "note" not in r and "wake" not in r and os.listdir(d) == []

def test_two_notes_in_one_second_get_distinct_files():
    d = tempfile.mkdtemp(); e.NOTE_DIR = d
    orig = e.time.strftime
    e.time.strftime = lambda *_: "2026-09-05-000000"
    try:
        i, v = BeingIntent("memory_write", {"path": "/o/f.md"}), _ref("mrh.path", "outside")
        a = e.write_note("b", i, v, "scope", {}); b = e.write_note("b", i, v, "scope", {}); c = e.write_note("b", i, v, "scope", {})
    finally:
        e.time.strftime = orig
    assert len({a, b, c}) == 3 and a.endswith("-000000.md") and b.endswith("-000000-2.md") and c.endswith("-000000-3.md")

def test_seat_is_the_beings_own_machine_not_sprout():
    # identity.json names the machine -> that seat; no file -> the member prefix
    d = tempfile.mkdtemp()
    assert e.seat_for("legion-being", d) == "legion"
    assert e.seat_for("sprout-being", d) == "sprout"
    assert e.seat_for("cbp-being") == "cbp"
    import json
    open(os.path.join(d, "identity.json"), "w").write(json.dumps({"identity": {"machine": "Thor"}}))
    assert e.seat_for("legion-being", d) == "thor"
    # the gate workspace is this checkout, whatever its name (Legion: ~/ai-workspace/SAGE)
    assert os.path.isdir(os.path.join(e.WORKSPACE, "sage", "gateway"))

def test_heartbeat_routes_refusals_by_default():
    import sage.gateway.heartbeat as hb, inspect
    src = inspect.getsource(hb.main)
    assert "--no-escalate" in src and "_esc.escalate(" in src
    assert "egress_drain.drain_once(" in src   # the being's parked mesh acts leave every beat

if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"PASS {name}")
    print(f"\n{n} passed")
