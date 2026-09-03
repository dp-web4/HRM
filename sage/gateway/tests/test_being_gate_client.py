"""Hermetic tests for the society-safety fail-closed boundary of BeingGateClient.

No live hestia gate required: we bypass __init__ and inject fake _core/_mech, so
this exercises the Stage-2 policy in isolation. Runnable under pytest or directly
(`python3 test_being_gate_client.py`).
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.gateway.being_gate_client import BeingGateClient, BeingIntent  # noqa: E402


def _client(mech):
    """A client whose local law always ALLOWs, with an injected society mechanism."""
    c = BeingGateClient.__new__(BeingGateClient)
    c.member_id = "test-being"
    c.workspace = "/tmp/ws"
    c._import_error = ""
    c.host_session_id = None
    c._profile = object()
    c._mech = mech
    c._core = SimpleNamespace(
        NormalizedEvent=lambda **kw: SimpleNamespace(raw=kw.get("raw", {}), tool=kw.get("tool")),
        evaluate=lambda ev, prof, ws, policy=None: SimpleNamespace(
            decision="allow", rule="", reason="ok", innate=False),
    )
    return c


PEER = BeingIntent("peer_ask", {"to": "legion", "body": "hi"})       # consequential
WRITE = BeingIntent("memory_write", {"path": "/tmp/ws/n.md", "content": "x"})  # consequential
READ = BeingIntent("memory_read", {"path": "/tmp/ws/n.md"})          # observational
WIT = BeingIntent("witness", {"event": "x"})                          # observational

# The mechanism's real signature: query_society_safety(event, *, plugin_id, host_agent, ...)
# -> SafetyVerdict(allow, decided, message, ...). `allow` is the only field acted on.
def _mech(fn):
    return SimpleNamespace(query_society_safety=lambda event, **kw: fn(event, kw))

_raises = _mech(lambda e, kw: (_ for _ in ()).throw(TimeoutError("down")))
_denies = _mech(lambda e, kw: SimpleNamespace(allow=False, decided=True, message="nope"))
_noverd = _mech(lambda e, kw: SimpleNamespace(allow=False, decided=False, message="no verdict"))
_allows = _mech(lambda e, kw: SimpleNamespace(allow=True, decided=True, message="ok"))


def test_mech_absent_consequential_denies():
    v = _client(None).gate(PEER)
    assert v.blocks and v.rule == "society.unavailable", v


def test_mech_absent_observational_softpasses():
    v = _client(None).gate(READ)
    assert v.decision == "allow", v


def test_mech_raises_consequential_denies():
    v = _client(_raises).gate(WRITE)
    assert v.blocks and v.rule == "society.unreachable", v


def test_mech_raises_observational_softpasses():
    v = _client(_raises).gate(WIT)
    assert v.decision == "allow", v


def test_mech_denies_blocks():
    v = _client(_denies).gate(PEER)
    assert v.blocks and v.rule == "society.unsafe", v


def test_mech_allows_consequential_allows():
    v = _client(_allows).gate(PEER)
    assert v.decision == "allow", v


def test_mech_no_verdict_fails_closed_distinctly():
    v = _client(_noverd).gate(PEER)
    assert v.blocks and v.rule == "society.no_verdict", v


def test_mech_is_called_with_real_contract():
    seen = {}
    m = _mech(lambda e, kw: seen.update(event=e, **kw) or SimpleNamespace(allow=True, decided=True, message="ok"))
    c = _client(m)
    c.host_session_id = "run-1"
    c.gate(PEER)
    assert seen["event"]["tool_name"] == "peer_ask" and seen["event"]["tool_input"]["to"] == "legion"
    assert seen["plugin_id"] == "test-being" and seen["host_agent"] == "sage-gateway"
    assert seen["host_session_id"] == "run-1"


def test_unregistered_effector_denies_before_gate():
    v = _client(_allows).gate(BeingIntent("shell", {"command": "rm -rf /"}))
    assert v.blocks and v.rule == "registry.unbounded" and v.stage == "registry", v


def test_no_core_fails_closed():
    c = _client(_allows)
    c._core = None
    v = c.gate(PEER)
    assert v.blocks and v.rule == "gate.unreachable" and v.innate, v


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"PASS {name}")
    print(f"\n{n} passed")


def test_relative_memory_path_is_judged_at_the_being_memory_root():
    """The gate must judge the SAME path the dispatcher will touch: a relative memory
    path is rooted at the being's memory root (its instance dir), not the cwd or the
    workspace. Captures the NormalizedEvent the law is handed."""
    seen = {}
    c = _client(_allows)
    c.memory_root = "/tmp/being-home"
    c._core = SimpleNamespace(
        NormalizedEvent=lambda **kw: seen.update(kw) or SimpleNamespace(raw=kw.get("raw", {}), tool=kw.get("tool")),
        evaluate=lambda ev, prof, ws, policy=None: SimpleNamespace(
            decision="allow", rule="", reason="ok", innate=False),
    )
    c.gate(BeingIntent("memory_write", {"path": "notes/x.md", "content": "x"}))
    assert seen["paths"] == ["/tmp/being-home/notes/x.md"], seen["paths"]
    c.gate(BeingIntent("memory_read", {"path": "/tmp/being-home/notes/x.md"}))
    assert seen["paths"] == ["/tmp/being-home/notes/x.md"]
