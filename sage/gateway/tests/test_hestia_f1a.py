"""Hermetic tests for the real F1a dispatcher: an injected fake MCP scripts the daemon."""
import os, sys, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.gateway.being_gate_client import BeingIntent, GatewayVerdict  # noqa: E402
from sage.gateway.hestia_f1a import HestiaF1aDispatcher  # noqa: E402

_ALLOW = GatewayVerdict("allow")


class FakeMcp:
    """Scripts tools/call results; records every call."""
    def __init__(self, notify=None, connect=None):
        self.calls = []
        self._notify = notify if notify is not None else {"result": {"structuredContent": {"notice_id": 42}}}
        self._connect = connect if connect is not None else {"result": {"structuredContent": {"sessionId": "s-1"}}}
    def init(self): pass
    def call(self, name, args):
        self.calls.append((name, args))
        if name == "hestia_connect": return self._connect
        if name == "hestia_member_notify": return self._notify
        if name == "hestia_member_inbox": return {"result": {"structuredContent": {"notices": []}}}
        return {"result": {"structuredContent": {}}}


def _disp(mcp=None, **kw):
    d = tempfile.mkdtemp(prefix="f1a-")
    return HestiaF1aDispatcher("sprout-being", "sage-raising", memory_root=d, mcp=mcp or FakeMcp(),
                               peer_pointer_dir=os.path.join(d, "shared-context", "forum"), **kw), d


def test_mesh_connects_then_notifies_with_session_and_routed_address():
    disp, _ = _disp()
    env = disp(BeingIntent("mesh", {"to": "legion", "kind": "coordination", "pointer_uri": "shared-context/forum/x.md"}), _ALLOW)
    assert env.ok and env.witness_id == "42", env
    names = [c[0] for c in disp._mcp.calls]
    assert names == ["hestia_connect", "hestia_member_notify"]
    a = disp._mcp.calls[1][1]
    assert a["to_plugin_id"] == "legion/claude-code" and a["kind"] == "coordination"
    assert a["pointer_uri"] == "shared-context/forum/x.md" and a["session_id"] == "s-1"


def test_mesh_accepts_legacy_pointer_key_and_refuses_missing_pointer():
    disp, _ = _disp()
    ok = disp(BeingIntent("mesh", {"to": "legion", "kind": "reply", "pointer": "p.md"}), _ALLOW)
    assert ok.ok and disp._mcp.calls[-1][1]["pointer_uri"] == "p.md"
    bad = disp(BeingIntent("mesh", {"to": "legion", "kind": "reply"}), _ALLOW)
    assert not bad.ok and bad.error.startswith("hestia.member_notify_missing_pointer")


def test_daemon_error_surfaces_hestia_code_as_key():
    err = {"result": {"structuredContent": {"_hestia_error": {"code": "hestia.member_notify_self", "message": "no self-notify"}}}}
    disp, _ = _disp(FakeMcp(notify=err))
    env = disp(BeingIntent("mesh", {"to": "sprout-being", "kind": "ack", "pointer_uri": "x"}), _ALLOW)
    assert not env.ok and env.error.startswith("hestia.member_notify_self")


def test_connect_failure_is_fail_closed_with_code():
    err = {"result": {"structuredContent": {"_hestia_error": {"code": "hestia.internal_error", "message": "missing host_agent"}}}}
    disp, _ = _disp(FakeMcp(connect=err))
    env = disp(BeingIntent("mesh", {"to": "legion", "kind": "reply", "pointer_uri": "x"}), _ALLOW)
    assert not env.ok and env.error.startswith("hestia.internal_error")


def test_peer_ask_composes_pointer_plus_coordination_notify():
    disp, d = _disp()
    env = disp(BeingIntent("peer_ask", {"to": "legion", "body": "Are you still you?"}), _ALLOW)
    assert env.ok, env
    a = disp._mcp.calls[-1][1]
    assert a["to_plugin_id"] == "legion/claude-code" and a["kind"] == "coordination"
    assert a["pointer_uri"].startswith("shared-context/forum/sprout-being-asks-legion-")
    posted = os.path.join(d, a["pointer_uri"])
    assert os.path.exists(posted) and "Are you still you?" in open(posted).read()
    assert env.result["asked"] == "legion"


def test_channel_egress_is_honest_pending():
    disp, _ = _disp()
    env = disp(BeingIntent("channel_egress", {"to": "x", "body": "y"}), _ALLOW)
    assert env.pending and not env.ok and "not built" in env.note


def test_memory_and_witness_delegate_to_reference():
    disp, d = _disp()
    note = os.path.join(d, "n.md")
    w = disp(BeingIntent("memory_write", {"path": note, "content": "kept"}), _ALLOW)
    assert w.ok and "kept" in open(note).read()
    # witness -> reference -> real hestia witness_fn; with no daemon it falls back to the local log
    v = disp(BeingIntent("witness", {"event": "x"}), _ALLOW)
    assert v.ok and v.witness_id


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"PASS {name}")
    print(f"\n{n} passed")
