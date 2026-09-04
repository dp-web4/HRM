"""
Hestia F1a dispatcher — executes the being's ALLOWED intents against the RUNNING hestia
daemon, verb by verb, on the primitives the daemon actually has (Legion's verb→daemon map,
forum/legion-reply-sprout-f1a-r1-stands-verb-map-2026-09-02.md, read from handler.rs @b7a6dcd):

    §7.2 verb        daemon primitive                                   here
    witness          hestia_begin_action / hestia_record_outcome        via ReferenceF1aDispatcher + hestia_witness
    memory r/w       none by design (local to the instance dir)         via ReferenceF1aDispatcher
    mesh             hestia_member_notify(to_plugin_id, kind,           EXECUTED — the primitive
                       pointer_uri, session_id)
    peer_ask         none — a COMPOSE: publish the question at a         EXECUTED as the compose
                       pointer, member_notify(kind=coordination),
                       drain hestia_member_inbox for the answer
    channel_egress   hestia_egress_pending is read-side only;           honest `pending`
                       no send-side tool exists

Result envelope (r1, pinned from the owner seat): {ok, result|error, witness_id}. When the
failure originates in the daemon, `error` carries hestia's own `hestia.<code>` string as its
key (e.g. `hestia.member_notify_missing_pointer`) so §7.3's deny→appeal loop has a stable key
to appeal on, not prose. `refused/pending/note/verdict` stay client-side extras.

Three contract deltas, all measured live on Sprout 2026-09-02, encoded below so the being's
`mesh` args (to, kind, pointer) never reach the daemon misspelled:
  * the field is `pointer_uri`, not `pointer` — a missing one is refused;
  * `kind` must be in the daemon's MEMBER_NOTICE_KINDS; self-notify refuses;
  * the sender is the live session_id from hestia_connect — no latest-session fallback.

Invariants (same as the reference this wraps):
  * only ever invoked on an intent the gate already ALLOWED;
  * every executed act is witnessed — mesh/peer_ask carry the daemon's witnessEntryHash,
    witness/memory carry the chain actionId (or the local log id when the daemon is down);
  * nothing is fabricated: a daemon error is an error envelope, a missing capability is
    `pending` with a note that names what is missing.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional

from sage.gateway.being_gate_client import BeingIntent, GatewayVerdict, ResultEnvelope
from sage.gateway.hestia_witness import _ENDPOINT, _Mcp, _unwrap, make_hestia_witness_fn
from sage.gateway.reference_f1a import ReferenceF1aDispatcher

# Mirrors handler.rs MEMBER_NOTICE_KINDS (b7a6dcd). Checked client-side so a bad kind is a
# clear refusal before the round-trip; the daemon enforces it again regardless.
MEMBER_NOTICE_KINDS = frozenset(
    {"coordination", "review_request", "review_done", "reply", "handoff", "forum-note", "ack"})

# publish_fn(to, body) -> pointer_uri : puts the question where the peer can read it.
# KINDS.md: content lives AT the pointer, never in the notice. The being's instance dir is
# not readable cross-seat, so this is the seat's choice (forum doc, hub pair thread, ...).
PublishFn = Callable[[str, str], str]
# mcp_factory(endpoint, plugin_id) -> object with .init() and .call(name, args); injected
# in tests, real _Mcp otherwise.
McpFactory = Callable[[str, str], Any]


def _hestia_error(env: dict) -> Optional[str]:
    """Render a daemon error envelope as the r1 `error` string keyed by hestia's own code."""
    err = env.get("_hestia_error")
    if err is None:
        return None
    if isinstance(err, dict):
        code = err.get("code") or "hestia.error"
        msg = err.get("message") or ""
        return f"{code}: {msg}" if msg else str(code)
    return str(err)


class HestiaF1aDispatcher:
    """A Dispatcher (being_gate_client.Dispatcher) that runs the bounded registry against the
    live daemon. Wraps ReferenceF1aDispatcher for the local verbs (witness / memory)."""

    def __init__(self, plugin_id: str, memory_root: str,
                 endpoint: str = _ENDPOINT,
                 publish_fn: Optional[PublishFn] = None,
                 remote_member_default: str = "claude-code",
                 local_members: Optional[set] = None,
                 host_session_id: Optional[str] = None,
                 mcp_factory: Optional[McpFactory] = None,
                 being_lct: Optional[str] = None,
                 # the BEING's membot, on its own port: :8000 is the seat sessions' membot and
                 # their conversation hooks write into whatever cartridge is mounted there
                 # (measured 2026-09-03: 4 seat memories landed in legion-being's cartridge)
                 membot_endpoint: str = "http://127.0.0.1:8010/mcp",
                 membot_cartridge: Optional[str] = None):
        self.plugin_id = plugin_id
        # the being's registry LCT id, named in every outward act it signs (pr_review)
        self.being_lct = being_lct
        # the being's long-term memory: a membot cartridge named after the member
        self.membot_endpoint = membot_endpoint
        self.membot_cartridge = membot_cartridge or plugin_id
        self._mb = None
        self.endpoint = endpoint
        self._publish = publish_fn
        self.remote_member_default = remote_member_default
        self.local_members = set(local_members or ())
        self.host_session_id = host_session_id
        self._mcp_factory = mcp_factory or _Mcp
        self._local = ReferenceF1aDispatcher(
            memory_root=memory_root,
            witness_fn=make_hestia_witness_fn(plugin_id, endpoint) if mcp_factory is None else None)
        self._c = None
        self._session_id: Optional[str] = None

    # -- the Dispatcher contract ---------------------------------------------
    def __call__(self, intent: BeingIntent, verdict: GatewayVerdict) -> ResultEnvelope:
        handler = getattr(self, f"_do_{intent.effector}", None)
        if handler is None:
            return self._local(intent, verdict)   # witness / memory_read / memory_write
        try:
            return handler(intent)
        except Exception as e:
            return ResultEnvelope(ok=False, error=f"{type(e).__name__}: {e}")

    # -- daemon session ------------------------------------------------------
    def _connect(self) -> str:
        """Hold ONE live session for the being: attribution is proven per session, not
        inherited, so every notify/inbox call carries it. Re-connects if the daemon lost it."""
        if self._c is not None and self._session_id:
            return self._session_id
        c = self._mcp_factory(self.endpoint, self.plugin_id)
        c.init()
        args = {"plugin_id": self.plugin_id, "host_agent": "sage-gateway", "plugin_version": "f2"}
        if self.host_session_id:
            args["host_session_id"] = self.host_session_id
        conn = _unwrap(c.call("hestia_connect", args))
        err = _hestia_error(conn)
        if err:
            raise RuntimeError(err)
        sid = conn.get("sessionId") or conn.get("session_id")
        if not sid:
            raise RuntimeError("hestia.connect_no_session: hestia_connect returned no sessionId")
        self._c, self._session_id = c, sid
        return sid

    def _call(self, name: str, args: dict) -> dict:
        sid = self._connect()
        out = _unwrap(self._c.call(name, {**args, "session_id": sid}))
        # a session the daemon no longer recognises: reconnect once, then report honestly
        err = out.get("_hestia_error") if isinstance(out, dict) else None
        if isinstance(err, dict) and "session" in str(err.get("code", "")):
            self._c, self._session_id = None, None
            sid = self._connect()
            out = _unwrap(self._c.call(name, {**args, "session_id": sid}))
        return out

    # -- addressing ----------------------------------------------------------
    def _address(self, to: str) -> str:
        """`peer/member` routes via the forwarding plane; a bare id stays on this local mesh.
        The being names a member ('legion'); the seat says whether that is local or remote."""
        to = (to or "").strip()
        if "/" in to or to in self.local_members:
            return to
        return f"{to}/{self.remote_member_default}"

    # -- mesh: THE primitive -------------------------------------------------
    def _do_mesh(self, intent: BeingIntent) -> ResultEnvelope:
        to = str(intent.args.get("to", "")).strip()
        kind = str(intent.args.get("kind", "")).strip()
        pointer = str(intent.args.get("pointer") or intent.args.get("pointer_uri") or "").strip()
        if not to:
            return ResultEnvelope(ok=False, error="mesh needs a 'to' member")
        if kind not in MEMBER_NOTICE_KINDS:
            return ResultEnvelope(ok=False, error=f"mesh 'kind' must be one of {sorted(MEMBER_NOTICE_KINDS)}, got {kind!r}")
        if not pointer:
            # the daemon would refuse this as hestia.member_notify_missing_pointer; say it first
            return ResultEnvelope(ok=False, error="hestia.member_notify_missing_pointer: mesh needs a 'pointer' (content lives AT the pointer, never in the notice)")
        args: Dict[str, Any] = {"to_plugin_id": self._address(to), "kind": kind, "pointer_uri": pointer}
        irt = intent.args.get("in_reply_to")
        if irt not in (None, ""):
            try:
                args["in_reply_to"] = int(irt)
            except (TypeError, ValueError):
                return ResultEnvelope(ok=False, error="mesh 'in_reply_to' must be a notice id (integer)")
        out = self._call("hestia_member_notify", args)
        err = _hestia_error(out)
        if err:
            return ResultEnvelope(ok=False, error=err)
        result = {
            "queued_id": out.get("queued_id"),
            "to_plugin_id": out.get("to_plugin_id", args["to_plugin_id"]),
            "kind": kind,
            # present => the forwarding branch fired: the row is PARKED for the seat's egress
            # drain, not forwarded. Absent => local inbox row.
            "egress_queued_to": out.get("egress_queued_to"),
            "recipient_liveness": out.get("recipient_liveness"),
        }
        return ResultEnvelope(ok=True, result=result,
                              witness_id=out.get("witnessEntryHash") or (str(out["queued_id"]) if out.get("queued_id") is not None else None))

    # -- peer_ask: a compose, not a verb ------------------------------------
    def _do_peer_ask(self, intent: BeingIntent) -> ResultEnvelope:
        to = str(intent.args.get("to", "")).strip()
        body = str(intent.args.get("body", "")).strip()
        if not to or not body:
            return ResultEnvelope(ok=False, error="peer_ask needs 'to' and 'body'")
        if self._publish is None:
            return ResultEnvelope(ok=False, pending=True,
                                  note="peer_ask needs a publisher: the question must live at a pointer "
                                       "the peer can read (forum doc / hub thread); none configured on this seat")
        pointer = self._publish(to, body)
        if not pointer:
            return ResultEnvelope(ok=False, error="peer_ask: publisher returned no pointer")
        env = self._do_mesh(BeingIntent("mesh", {"to": to, "kind": "coordination", "pointer": pointer}))
        if env.ok and isinstance(env.result, dict):
            env.result["question_at"] = pointer
            env.result["answer_via"] = "hestia_member_inbox (drain_inbox)"
        return env

    def drain_inbox(self, peek: bool = False) -> ResultEnvelope:
        """The answer side of peer_ask: the being's own notices (consume-once unless peek)."""
        try:
            out = self._call("hestia_member_inbox", {"peek": peek})
        except Exception as e:
            return ResultEnvelope(ok=False, error=f"{type(e).__name__}: {e}")
        err = _hestia_error(out)
        if err:
            return ResultEnvelope(ok=False, error=err)
        return ResultEnvelope(ok=True, result={"notices": out.get("notices", []),
                                               "total": out.get("total", 0),
                                               "evicted": out.get("evicted", 0)})

    # -- pr_review: the seat posts the being's review, as the gated command -------
    def _do_pr_review(self, intent: BeingIntent) -> ResultEnvelope:
        """Only ever reached on an intent the gate ALLOWED as the exact `gh pr review`
        command below. Order: begin_action (chain) -> post -> record_outcome. The
        signature trailer is appended here, so the being cannot post without it."""
        import shlex
        import subprocess
        from sage.gateway.being_gate_client import pr_review_command, pr_review_signature
        try:
            cmd = pr_review_command(intent.args)
        except ValueError as e:
            return ResultEnvelope(ok=False, error=str(e))
        repo, number = str(intent.args["repo"]).strip(), str(intent.args["number"]).strip()
        target = f"{repo}#{number}"
        begin = self._call("hestia_begin_action", {"tool_name": "pr_review", "target": target})
        err = _hestia_error(begin)
        if err:
            return ResultEnvelope(ok=False, error=err)
        action_id = begin.get("actionId")
        body = str(intent.args["body"]).rstrip() + "\n" + \
            pr_review_signature(self.plugin_id, action_id, self.being_lct) + "\n"
        try:
            proc = subprocess.run(shlex.split(cmd), input=body, text=True,
                                  capture_output=True, timeout=60)
            ok = proc.returncode == 0
            detail = (proc.stdout if ok else proc.stderr).strip()
        except Exception as e:  # gh missing, timeout: a failed act, recorded as such
            ok, detail = False, f"{type(e).__name__}: {e}"
        try:
            self._call("hestia_record_outcome",
                       {"action_id": action_id, "success": ok, "magnitude": 0.0,
                        **({} if ok else {"error": detail[:300]})})
        except Exception:
            pass  # begin_action already chained the act; the outcome is in the envelope
        if not ok:
            return ResultEnvelope(ok=False, error=f"pr_review failed: {detail[:400]}",
                                  witness_id=action_id)
        return ResultEnvelope(ok=True, witness_id=action_id,
                              result={"posted": target, "gh": detail[:200], "action_id": action_id})

    # -- long-term memory: the being's own membot cartridge ----------------------
    def _membot(self):
        """One MCP session to the membot server (fastmcp streamable HTTP). Lazy; a
        server that is down surfaces as an error envelope on the act, never a crash."""
        if getattr(self, "_mb", None) is None:
            c = self._mcp_factory(self.membot_endpoint, self.plugin_id)
            c.init()
            self._mb = c
        return self._mb

    def _membot_call(self, name: str, args: dict) -> str:
        """One membot tool call, unwrapped to its text. RAISES on a JSON-RPC `error` or a
        tool-level `isError` result: both handlers turn the exception into ok=False, so a
        membot that says "Error calling tool save_cartridge" is never witnessed as a
        memory the being kept (Sprout's review of #36, reproduced against a fake membot)."""
        out = self._membot().call(name, args)
        if not isinstance(out, dict):
            raise RuntimeError(f"membot {name}: malformed reply {type(out).__name__}")
        if out.get("error"):
            err = out["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"membot {name}: {msg or err}")
        res = out.get("result", {})
        if not isinstance(res, dict):
            raise RuntimeError(f"membot {name}: malformed result {type(res).__name__}")
        if res.get("isError"):
            text = "".join(b.get("text", "") for b in res.get("content", []) if isinstance(b, dict))
            raise RuntimeError(f"membot {name}: {text.strip() or 'isError'}")
        sc = res.get("structuredContent")
        if isinstance(sc, dict) and "result" in sc:
            return str(sc["result"])
        return "".join(b.get("text", "") for b in res.get("content", []) if isinstance(b, dict))

    def _do_recall(self, intent: BeingIntent) -> ResultEnvelope:
        q = str(intent.args.get("query", "")).strip()
        if not q:
            return ResultEnvelope(ok=False, error="recall needs a 'query'")
        try:
            k = int(intent.args.get("top_k") or 5)
        except (TypeError, ValueError):
            k = 5
        try:
            text = self._membot_call("memory_search", {"query": q, "top_k": max(1, min(k, 20))})
        except Exception as e:
            return ResultEnvelope(ok=False, error=f"membot ({type(e).__name__}): {e}")
        return ResultEnvelope(ok=True, result=text,
                              witness_id=self._local._witness(f"recall {q[:80]}"))

    def _do_remember(self, intent: BeingIntent) -> ResultEnvelope:
        content = str(intent.args.get("content", "")).strip()
        if not content:
            return ResultEnvelope(ok=False, error="remember needs 'content'")
        tags = str(intent.args.get("tags", "") or "")
        try:
            stored = self._membot_call("memory_store", {"content": content, "tags": tags})
            saved = self._membot_call("save_cartridge", {"name": self.membot_cartridge})
        except Exception as e:
            return ResultEnvelope(ok=False, error=f"membot ({type(e).__name__}): {e}")
        return ResultEnvelope(ok=True, result=f"{stored}; {saved}",
                              witness_id=self._local._witness(f"remember {content[:80]}"))

    # -- request_scope: the sanctioned answer to a deny --------------------------
    def _do_request_scope(self, intent: BeingIntent) -> ResultEnvelope:
        """The daemon (handler.rs::tool_request_scope @a5e18af) reads plugin_id, role, path,
        reason; a grant is reach on the path, read AND write. There is no mode to carry: an
        earlier `permits_read` was dropped silently by the daemon while the operator read
        "[.., read]" on a request that, granted, gave write."""
        path = str(intent.args.get("path", "")).strip()
        reason = str(intent.args.get("reason", "")).strip()
        if not path.startswith("/"):
            return ResultEnvelope(ok=False, error="request_scope 'path' must be absolute")
        if not reason:
            return ResultEnvelope(ok=False, error="request_scope needs a 'reason' (a human reads it)")
        args: Dict[str, Any] = {"plugin_id": self.plugin_id, "path": path,
                                "reason": f"[{self.plugin_id}] {reason}"}
        out = self._call("hestia_request_scope", args)
        err = _hestia_error(out)
        if err:
            return ResultEnvelope(ok=False, error=err)
        return ResultEnvelope(ok=True, witness_id=out.get("witnessEntryHash"),
                              result={"request_id": out.get("request_id"), "status": out.get("status"),
                                      "path": path,
                                      "next": out.get("next"), "on_timeout": out.get("on_timeout")})

    # -- channel_egress: not built on the daemon ----------------------------
    def _do_channel_egress(self, intent: BeingIntent) -> ResultEnvelope:
        return ResultEnvelope(ok=False, pending=True,
                              note="channel_egress awaits hestia's send-side tool (hestia_egress_pending is "
                                   "read-only; F5 enforcement gated on rate-governor calibration, PRD §12)")


# --------------------------------------------------------------------------
# A reference publisher for peer_ask: the question becomes a forum doc in shared-context,
# committed and pushed, and the pointer is its repo path. Git failures raise, so the caller
# gets an error envelope rather than a notice pointing at content that never landed.
# --------------------------------------------------------------------------
def forum_publisher(shared_context_root: str, from_name: str,
                    subdir: str = "forum/being", push: bool = True) -> PublishFn:
    import re
    import subprocess
    from datetime import datetime

    root = os.path.expanduser(shared_context_root)

    def publish(to: str, body: str) -> str:
        ts = datetime.now()
        slug = re.sub(r"[^a-z0-9]+", "-", body.lower())[:40].strip("-") or "question"
        rel = f"{subdir}/{from_name}-asks-{re.sub(r'[^a-z0-9]+', '-', to.lower())}-{slug}-{ts:%Y-%m-%d-%H%M%S}.md"
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(f"---\nfrom: {from_name}\nto: {to}\nkind: peer_ask\ndate: {ts:%Y-%m-%d}\n---\n\n{body}\n")
        subprocess.run(["git", "-C", root, "add", rel], check=True)
        subprocess.run(["git", "-C", root, "commit", "-q", "-m", f"being({from_name}): peer_ask -> {to}"], check=True)
        if push:
            subprocess.run(["git", "-C", root, "push", "-q"], check=True)
        return f"shared-context/{rel}"
    return publish


if __name__ == "__main__":  # live smoke against the local daemon: mesh -> member_notify
    import sys
    inst = os.path.expanduser("~/ai-workspace/sage/sage/instances/sprout-qwen3.8-distill-2b")
    d = HestiaF1aDispatcher("sprout-being", inst)
    to, kind, ptr = (sys.argv[1:4] + [None, None, None])[:3]
    if not (to and kind and ptr):
        print("usage: hestia_dispatch.py <to> <kind> <pointer_uri>"); sys.exit(2)
    env = d(BeingIntent("mesh", {"to": to, "kind": kind, "pointer": ptr}), GatewayVerdict("allow"))
    print(json.dumps({"ok": env.ok, "result": env.result, "error": env.error,
                      "witness_id": env.witness_id, "pending": env.pending, "note": env.note}, indent=1))


def make_forum_publisher(pointer_dir: str, plugin_id: str) -> PublishFn:
    """Default publish_fn for peer_ask: the question lives AT a cross-seat-readable pointer
    (a shared-context/forum file, repo-relative), never in the notice (KINDS.md). The file
    must be pushed for the peer to read it; the notice only points."""
    from datetime import datetime
    from pathlib import Path

    def publish(to: str, body: str) -> str:
        d = Path(pointer_dir); d.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        p = d / f"{plugin_id}-asks-{to.replace('/', '-')}-{ts}.md"
        p.write_text(f"---\nfrom: {plugin_id}\nto: {to}\nkind: coordination\ndate: {ts[:10]}\n---\n\n{body.strip()}\n")
        sp = str(p); i = sp.find("shared-context/")
        return sp[i:] if i >= 0 else sp
    return publish
