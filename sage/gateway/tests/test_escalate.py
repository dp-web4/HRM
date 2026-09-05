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

def test_scope_path_is_the_directory_of_the_target():
    root = "/x/instances/b"
    assert e._scope_path(BeingIntent("memory_write", {"path": "notes/a.md"}), root) == "/x/instances/b/notes"
    assert e._scope_path(BeingIntent("memory_write", {"path": "/other/dir/f.txt"}), root) == "/other/dir"

def test_note_carries_verdict_and_arbiter_protocol(monkeypatch=None):
    d = tempfile.mkdtemp(); e.NOTE_DIR = d
    p = e.write_note("b", BeingIntent("memory_write", {"path": "/o/f.md"}), _ref("mrh.path", "outside"), "scope", {"scope_request": {"request_id": "scope-1"}})
    t = open(p).read()
    assert "mrh.path" in t and "scope-1" in t and "Arbiter protocol" in t and "STANDING" in t

if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"PASS {name}")
    print(f"\n{n} passed")
