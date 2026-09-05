"""Hermetic tests for HestiaF1aDispatcher: a fake MCP records the calls the daemon would
see, so the three measured contract deltas (pointer_uri, kind enum, live session_id) and the
r1 envelope (hestia.<code> error keys) are pinned without a running daemon."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.gateway.being_gate_client import BeingIntent, GatewayVerdict  # noqa: E402
from sage.gateway.hestia_dispatch import HestiaF1aDispatcher  # noqa: E402

_ALLOW = GatewayVerdict("allow")


class FakeMcp:
    """Answers like the daemon: connect -> sessionId; member_notify -> receipt or error."""
    calls = []

    def __init__(self, endpoint, plugin_id, notify_reply=None):
        self.plugin_id = plugin_id
        self.notify_reply = notify_reply

    def init(self):
        pass

    def call(self, name, args):
        FakeMcp.calls.append((name, args))
        if name == "hestia_connect":
            body = {"sessionId": "sid-1", "softLct": "lct:web4:session:x"}
        elif name == "hestia_member_notify":
            if not args.get("pointer_uri"):
                body = {"_hestia_error": {"code": "hestia.member_notify_missing_pointer", "message": "no pointer"}}
            elif args.get("to_plugin_id") == self.plugin_id:
                body = {"_hestia_error": {"code": "hestia.member_notify_self", "message": "self"}}
            else:
                to = args["to_plugin_id"]
                body = {"queued_id": 7, "witnessEntryHash": "abc123", "to_plugin_id": to,
                        "egress_queued_to": to.split("/")[0] if "/" in to else None,
                        "recipient_liveness": "unknown"}
        elif name == "hestia_member_inbox":
            body = {"total": 1, "notices": [{"id": 9, "kind": "reply", "pointer_uri": "x"}], "evicted": 0}
        elif name == "hestia_request_scope":
            body = {"request_id": "scope-1", "status": "pending", "witnessEntryHash": "rs-hash",
                    "next": "operator decides", "on_timeout": "refused"}
        else:
            body = {}
        return {"result": {"structuredContent": body}}


class FakeMembot(FakeMcp):
    """Answers like membot (fastmcp streamable HTTP): text in content + structuredContent.
    `fail` names tools that answer as membot does when they fail — a JSON-RPC `error` or
    a tool result with `isError: true` — the two shapes Sprout reproduced on #36."""
    def __init__(self, endpoint, plugin_id, fail=None):
        super().__init__(endpoint, plugin_id)
        self.fail = dict(fail or {})

    def call(self, name, args):
        if name not in ("memory_search", "memory_store", "save_cartridge"):
            return super().call(name, args)
        FakeMcp.calls.append((name, args))
        how = self.fail.get(name)
        if how == "rpc":
            return {"jsonrpc": "2.0", "id": 1, "error": {"code": -32603, "message": f"{name} exploded"}}
        if how == "isError":
            return {"result": {"content": [{"type": "text", "text": f"Error calling tool {name}"}],
                               "isError": True}}
        text = {"memory_search": "1. something remembered",
                "memory_store": "Stored memory abc",
                "save_cartridge": f"Saved cartridge {args.get('name')}"}[name]
        return {"result": {"content": [{"type": "text", "text": text}],
                           "structuredContent": {"result": text}}}


def _mdisp(fail=None):
    FakeMcp.calls = []
    root = tempfile.mkdtemp(prefix="hd-")
    factory = lambda ep, pid: FakeMembot(ep, pid, fail=fail)  # noqa: E731
    return HestiaF1aDispatcher("sprout-being", root, mcp_factory=factory), root


def _mb_calls(name):
    return [a for n, a in FakeMcp.calls if n == name]


def _disp(**kw):
    FakeMcp.calls = []
    root = tempfile.mkdtemp(prefix="hd-")
    factory = lambda ep, pid: FakeMcp(ep, pid)  # noqa: E731
    return HestiaF1aDispatcher("sprout-being", root, mcp_factory=factory, **kw), root


def _notify_args():
    return [a for n, a in FakeMcp.calls if n == "hestia_member_notify"][-1]


def test_mesh_routes_remote_with_pointer_uri_and_session():
    d, _ = _disp()
    env = d(BeingIntent("mesh", {"to": "legion", "kind": "reply", "pointer": "shared-context/forum/x.md"}), _ALLOW)
    assert env.ok and env.witness_id == "abc123", env
    a = _notify_args()
    assert a["to_plugin_id"] == "legion/claude-code"          # peer/member routed address
    assert a["pointer_uri"] == "shared-context/forum/x.md"    # the field is pointer_uri, not pointer
    assert "pointer" not in a
    assert a["session_id"] == "sid-1"                         # live session from hestia_connect
    assert env.result["egress_queued_to"] == "legion" and env.result["queued_id"] == 7


def test_mesh_local_member_stays_bare():
    d, _ = _disp(local_members={"claude-code"})
    d(BeingIntent("mesh", {"to": "claude-code", "kind": "coordination", "pointer": "p"}), _ALLOW)
    assert _notify_args()["to_plugin_id"] == "claude-code"


def test_mesh_explicit_routed_address_passes_through():
    d, _ = _disp()
    d(BeingIntent("mesh", {"to": "thor/thor-sage", "kind": "ack", "pointer": "p"}), _ALLOW)
    assert _notify_args()["to_plugin_id"] == "thor/thor-sage"


def test_mesh_bad_kind_refused_before_round_trip():
    d, _ = _disp()
    env = d(BeingIntent("mesh", {"to": "legion", "kind": "pr_review_request", "pointer": "p"}), _ALLOW)
    assert not env.ok and "kind" in env.error
    assert not any(n == "hestia_member_notify" for n, _ in FakeMcp.calls)


def test_mesh_missing_pointer_is_hestia_keyed_error():
    d, _ = _disp()
    env = d(BeingIntent("mesh", {"to": "legion", "kind": "reply"}), _ALLOW)
    assert not env.ok and env.error.startswith("hestia.member_notify_missing_pointer")


def test_daemon_error_envelope_becomes_hestia_keyed_error():
    d, _ = _disp(local_members={"sprout-being"})
    env = d(BeingIntent("mesh", {"to": "sprout-being", "kind": "reply", "pointer": "p"}), _ALLOW)
    assert not env.ok and env.error.startswith("hestia.member_notify_self")


def test_session_is_connected_once_and_reused():
    d, _ = _disp()
    for _ in range(3):
        d(BeingIntent("mesh", {"to": "legion", "kind": "reply", "pointer": "p"}), _ALLOW)
    assert sum(1 for n, _ in FakeMcp.calls if n == "hestia_connect") == 1


def test_peer_ask_without_publisher_is_pending_not_fabricated():
    d, _ = _disp()
    env = d(BeingIntent("peer_ask", {"to": "legion", "body": "what is the envelope?"}), _ALLOW)
    assert not env.ok and env.pending and "publisher" in env.note
    assert not any(n == "hestia_member_notify" for n, _ in FakeMcp.calls)


def test_peer_ask_composes_publish_then_coordination_notify():
    published = []
    def pub(to, body):
        published.append((to, body)); return f"shared-context/forum/being/q-{to}.md"
    d, _ = _disp(publish_fn=pub)
    env = d(BeingIntent("peer_ask", {"to": "legion", "body": "what is the envelope?"}), _ALLOW)
    assert env.ok and published == [("legion", "what is the envelope?")]
    a = _notify_args()
    assert a["kind"] == "coordination" and a["pointer_uri"] == "shared-context/forum/being/q-legion.md"
    assert env.result["question_at"] == a["pointer_uri"]


def test_drain_inbox_returns_notices():
    d, _ = _disp()
    env = d.drain_inbox()
    assert env.ok and env.result["total"] == 1 and env.result["notices"][0]["id"] == 9
    assert [a for n, a in FakeMcp.calls if n == "hestia_member_inbox"][-1]["session_id"] == "sid-1"


def test_channel_egress_pending_not_built():
    d, _ = _disp()
    env = d(BeingIntent("channel_egress", {"to": "x", "body": "y"}), _ALLOW)
    assert not env.ok and env.pending and "send-side" in env.note


def test_local_verbs_delegate_to_reference():
    d, root = _disp()
    note = os.path.join(root, "n.md")
    w = d(BeingIntent("memory_write", {"path": note, "content": "kept"}), _ALLOW)
    r = d(BeingIntent("memory_read", {"path": note}), _ALLOW)
    assert w.ok and r.ok and "kept" in r.result
    wit = d(BeingIntent("witness", {"event": "x"}), _ALLOW)
    assert wit.ok and wit.witness_id


# ---- ported from the folded hestia_f1a tests (2026-09-02) --------------------------------
from sage.gateway.being_gate_client import BeingIntent as _BI, GatewayVerdict as _GV  # noqa: E402
from sage.gateway.hestia_dispatch import HestiaF1aDispatcher as _HD, make_forum_publisher  # noqa: E402
_ALLOW_ = _GV("allow")


def _forum_repo():
    """A shared-context clone with a bare origin, like the seat's checkout: the publisher must
    land the doc on ORIGIN, not merely write it — that gap is the beat-12 defect (2026-09-05)."""
    import subprocess
    base = tempfile.mkdtemp(prefix="hd-forum-")
    bare = os.path.join(base, "origin.git")
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", bare], check=True)
    root = os.path.join(base, "shared-context")
    subprocess.run(["git", "clone", "-q", bare, root], check=True)
    g = lambda *a: subprocess.run(["git", "-C", root, *a], check=True, capture_output=True, text=True).stdout  # noqa: E731
    g("config", "user.name", "t"); g("config", "user.email", "t@t")
    g("commit", "-q", "--allow-empty", "-m", "root"); g("push", "-q", "-u", "origin", "main")
    return base, root, bare


def _on_origin(bare, rel):
    import subprocess
    return subprocess.run(["git", "-C", bare, "cat-file", "-e", f"main:{rel}"], capture_output=True).returncode == 0


def test_peer_ask_with_forum_publisher_lands_on_origin_then_notifies():
    base, root, bare = _forum_repo()
    d, _ = _disp(publish_fn=make_forum_publisher(os.path.join(root, "forum"), "sprout-being"))
    env = d(_BI("peer_ask", {"to": "legion", "body": "Are you still you?"}), _ALLOW_)
    assert env.ok, env
    notify = [a for n, a in FakeMcp.calls if n == "hestia_member_notify"][-1]
    assert notify["to_plugin_id"] == "legion/claude-code" and notify["kind"] == "coordination"
    assert notify["pointer_uri"].startswith("shared-context/forum/sprout-being-asks-legion-")
    posted = os.path.join(base, notify["pointer_uri"])
    assert os.path.exists(posted) and "still you" in open(posted).read()
    rel = notify["pointer_uri"][len("shared-context/"):]
    assert _on_origin(bare, rel), "the pointer doc must be on origin before the notice fires"


def test_peer_ask_publisher_rebases_over_a_concurrent_push():
    """Origin moved after the clone (a sibling seat pushed): the publisher rebases and lands,
    instead of a rejected push that leaves the doc local."""
    import subprocess
    base, root, bare = _forum_repo()
    other = os.path.join(base, "other")
    subprocess.run(["git", "clone", "-q", bare, other], check=True)
    subprocess.run(["git", "-C", other, "-c", "user.name=o", "-c", "user.email=o@o",
                    "commit", "-q", "--allow-empty", "-m", "sibling"], check=True)
    subprocess.run(["git", "-C", other, "push", "-q", "origin", "main"], check=True)
    d, _ = _disp(publish_fn=make_forum_publisher(os.path.join(root, "forum"), "sprout-being"))
    env = d(_BI("peer_ask", {"to": "legion", "body": "still there?"}), _ALLOW_)
    assert env.ok, env
    rel = _notify_args()["pointer_uri"][len("shared-context/"):]
    assert _on_origin(bare, rel)


def test_peer_ask_publisher_outside_a_repo_is_an_error_envelope_and_no_notice():
    """A doc that cannot land is refused BEFORE any notice: the peer is never fired on a
    pointer it cannot read."""
    root_dir = tempfile.mkdtemp(prefix="hd-norepo-")
    d, _ = _disp(publish_fn=make_forum_publisher(os.path.join(root_dir, "shared-context", "forum"), "sprout-being"))
    env = d(_BI("peer_ask", {"to": "legion", "body": "anyone?"}), _ALLOW_)
    assert not env.ok and "git" in (env.error or ""), env
    assert not [n for n, _ in FakeMcp.calls if n == "hestia_member_notify"]


def test_peer_ask_publisher_push_false_only_writes():
    root_dir = tempfile.mkdtemp(prefix="hd-nopush-")
    d, _ = _disp(publish_fn=make_forum_publisher(os.path.join(root_dir, "shared-context", "forum"), "sprout-being", push=False))
    env = d(_BI("peer_ask", {"to": "legion", "body": "local only"}), _ALLOW_)
    assert env.ok, env
    assert os.path.exists(os.path.join(root_dir, _notify_args()["pointer_uri"]))


def test_mesh_witness_id_falls_back_to_queued_id():
    class NoHash(FakeMcp):
        def call(self, name, args):
            r = super().call(name, args)
            sc = r["result"]["structuredContent"]
            if name == "hestia_member_notify":
                sc.pop("witnessEntryHash", None)
            return r
    FakeMcp.calls = []
    d = _HD("sprout-being", tempfile.mkdtemp(prefix="hd-"), mcp_factory=lambda ep, pid: NoHash(ep, pid))
    env = d(_BI("mesh", {"to": "legion", "kind": "ack", "pointer_uri": "x.md"}), _ALLOW_)
    assert env.ok and env.witness_id == "7", env


# -- long-term memory (membot) and request_scope: the three verbs #36 adds ------------
def test_recall_sends_query_and_clamps_top_k():
    d, _ = _mdisp()
    env = d(BeingIntent("recall", {"query": "what was I doing", "top_k": 99}), _ALLOW)
    assert env.ok and env.result == "1. something remembered" and env.witness_id, env
    assert _mb_calls("memory_search") == [{"query": "what was I doing", "top_k": 20}]
    d(BeingIntent("recall", {"query": "x", "top_k": -3}), _ALLOW)
    assert _mb_calls("memory_search")[-1]["top_k"] == 1
    d(BeingIntent("recall", {"query": "x", "top_k": 0}), _ALLOW)   # 0/absent => the default
    assert _mb_calls("memory_search")[-1]["top_k"] == 5
    d(BeingIntent("recall", {"query": "x", "top_k": "lots"}), _ALLOW)
    assert _mb_calls("memory_search")[-1]["top_k"] == 5
    env = d(BeingIntent("recall", {"query": "  "}), _ALLOW)
    assert not env.ok and "query" in env.error and len(_mb_calls("memory_search")) == 4


def test_remember_stores_then_saves_the_seat_fixed_cartridge():
    """The cartridge name is the seat's (membot_cartridge or plugin_id); a being-supplied
    `name` never reaches save_cartridge — that is what bounds remember's reach."""
    d, _ = _mdisp()
    env = d(BeingIntent("remember", {"content": "lesson one", "tags": "a,b",
                                     "name": "someone-elses-cartridge"}), _ALLOW)
    assert env.ok and env.witness_id, env
    assert _mb_calls("memory_store") == [{"content": "lesson one", "tags": "a,b"}]
    assert _mb_calls("save_cartridge") == [{"name": "sprout-being"}]
    order = [n for n, _ in FakeMcp.calls if n in ("memory_store", "save_cartridge")]
    assert order == ["memory_store", "save_cartridge"]
    d2, _ = _mdisp()
    d2.membot_cartridge = "seat-named"
    d2(BeingIntent("remember", {"content": "x"}), _ALLOW)
    assert _mb_calls("save_cartridge") == [{"name": "seat-named"}]
    env = d2(BeingIntent("remember", {"content": ""}), _ALLOW)
    assert not env.ok and "content" in env.error


def test_membot_rpc_error_on_store_is_not_witnessed_as_a_memory():
    d, root = _mdisp(fail={"memory_store": "rpc"})
    env = d(BeingIntent("remember", {"content": "x"}), _ALLOW)
    assert not env.ok and env.witness_id is None and "memory_store exploded" in env.error, env
    assert _mb_calls("save_cartridge") == []          # no save after a failed store
    assert not os.path.exists(d._local.witness_log)   # nothing witnessed


def test_membot_iserror_on_save_is_not_witnessed_as_a_memory():
    d, _ = _mdisp(fail={"save_cartridge": "isError"})
    env = d(BeingIntent("remember", {"content": "x"}), _ALLOW)
    assert not env.ok and env.witness_id is None, env
    assert "Error calling tool save_cartridge" in env.error
    assert not os.path.exists(d._local.witness_log)


def test_membot_iserror_on_search_is_an_error_not_a_recall():
    d, _ = _mdisp(fail={"memory_search": "isError"})
    env = d(BeingIntent("recall", {"query": "x"}), _ALLOW)
    assert not env.ok and env.witness_id is None and "memory_search" in env.error, env
    assert not os.path.exists(d._local.witness_log)


def test_request_scope_carries_plugin_path_reason_and_live_session():
    d, _ = _mdisp()
    env = d(BeingIntent("request_scope", {"path": "/home/dp/notes", "reason": "to read my notes"}), _ALLOW)
    assert env.ok and env.witness_id == "rs-hash", env
    assert env.result["request_id"] == "scope-1" and env.result["status"] == "pending"
    assert env.result["path"] == "/home/dp/notes" and "mode" not in env.result
    (sent,) = [a for n, a in FakeMcp.calls if n == "hestia_request_scope"]
    assert sent["plugin_id"] == "sprout-being" and sent["path"] == "/home/dp/notes"
    assert sent["session_id"] == "sid-1"                # the live session from hestia_connect
    assert sent["reason"] == "[sprout-being] to read my notes"
    assert set(sent) == {"plugin_id", "path", "reason", "session_id"}  # no mode, no permits_read


def test_request_scope_refuses_relative_path_and_empty_reason_before_any_round_trip():
    d, _ = _mdisp()
    env = d(BeingIntent("request_scope", {"path": "notes", "reason": "why"}), _ALLOW)
    assert not env.ok and "absolute" in env.error, env
    env = d(BeingIntent("request_scope", {"path": "/home/dp/notes", "reason": "  "}), _ALLOW)
    assert not env.ok and "reason" in env.error, env
    assert FakeMcp.calls == []   # not even hestia_connect


def test_request_scope_daemon_error_is_keyed():
    class Refuses(FakeMembot):
        def call(self, name, args):
            if name == "hestia_request_scope":
                FakeMcp.calls.append((name, args))
                return {"result": {"structuredContent": {"_hestia_error": {
                    "code": "hestia.scope_request_unknown_member", "message": "who"}}}}
            return super().call(name, args)
    FakeMcp.calls = []
    d = HestiaF1aDispatcher("sprout-being", tempfile.mkdtemp(prefix="hd-"),
                            mcp_factory=lambda ep, pid: Refuses(ep, pid))
    env = d(BeingIntent("request_scope", {"path": "/x", "reason": "y"}), _ALLOW)
    assert not env.ok and env.error.startswith("hestia.scope_request_unknown_member"), env


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"PASS {name}")
    print(f"\n{n} passed")
