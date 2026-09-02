"""Hermetic tests for the egress drain worker: fake MCP scripts the daemon, fake sender stands in for hub-notify."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.gateway.egress_drain import drain_once  # noqa: E402

def _sc(d): return {"result": {"structuredContent": d}}

class FakeMcp:
    def __init__(self, pending=None, egress_error=None):
        self.calls = []; self.pending = pending or []; self.egress_error = egress_error
    def init(self): pass
    def call(self, name, args):
        self.calls.append((name, dict(args)))
        if name == "hestia_connect": return _sc({"sessionId": "s-9"})
        if name == "hestia_egress_pending":
            if self.egress_error: return _sc({"_hestia_error": self.egress_error})
            if "mark_forwarded" in args or "mark_failed" in args: return _sc({"ok": True})
            return _sc({"pending": self.pending, "total": len(self.pending), "drain_contract": {}})
        return _sc({})

ROW = {"id": "n-1", "forward_on": "61525719-def6-475c-a030-917f24a9dbf2", "forward_on_is_lct": True,
       "kind": "coordination", "pointer_uri": "shared-context/forum/x.md", "attempts": 0}

def test_forwards_and_marks_forwarded():
    m = FakeMcp(pending=[ROW]); sent = []
    r = drain_once(mcp=m, sender=lambda to, kind, ptr: (sent.append((to, kind, ptr)) or (True, "ledger=77")), log=lambda *_: None)
    assert r["forwarded"] == 1 and r["failed"] == 0 and not r["empty"]
    assert sent == [("61525719-def6-475c-a030-917f24a9dbf2", "coordination", "shared-context/forum/x.md")]
    marks = [a for n, a in m.calls if n == "hestia_egress_pending" and "mark_forwarded" in a]
    assert marks and marks[0]["mark_forwarded"] == "n-1" and marks[0]["session_id"] == "s-9"

def test_sender_failure_marks_failed_with_reason():
    m = FakeMcp(pending=[ROW])
    r = drain_once(mcp=m, sender=lambda to, kind, ptr: (False, "hub refused: kind gate"), log=lambda *_: None)
    assert r["failed"] == 1 and r["forwarded"] == 0
    marks = [a for n, a in m.calls if n == "hestia_egress_pending" and "mark_failed" in a]
    assert marks and marks[0]["mark_failed"] == "n-1" and "kind gate" in marks[0]["reason"]

def test_empty_queue_is_empty_not_error():
    r = drain_once(mcp=FakeMcp(pending=[]), sender=lambda *a: (True, ""), log=lambda *_: None)
    assert r["empty"] and r["error"] is None and r["forwarded"] == 0

def test_refused_call_is_error_never_silence():
    r = drain_once(mcp=FakeMcp(egress_error={"code": "hestia.egress_unattributed", "message": "x"}),
                   sender=lambda *a: (True, ""), log=lambda *_: None)
    assert r["error"] and r["error"]["code"] == "hestia.egress_unattributed" and not r["empty"]

if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"PASS {name}")
    print(f"\n{n} passed")
