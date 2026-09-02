"""
Real F1a dispatcher against the running hestia daemon — Legion's verb map (hestia #834),
replacing the reference for the verbs the daemon already carries.

  verb            daemon primitive                                   here
  witness         hestia_begin_action / hestia_record_outcome        via ReferenceF1aDispatcher (already real)
  mesh            hestia_member_notify(to_plugin_id, kind,           REAL — law-gated, attributed to the live
                    pointer_uri, session_id)                           session from hestia_connect
  peer_ask        (no daemon verb, by ruling) = COMPOSE:              REAL — content posted AT a pointer, then
                    post question at a pointer + member_notify           member_notify(kind=coordination); the
                    (kind=coordination) + member_inbox for the reply     reply is drained from member_inbox
  memory r/w      none by design (local to the being's instance       via ReferenceF1aDispatcher (gate-scoped
                    dir under its MRH grant)                             to memory_root)
  channel_egress  hestia_egress_pending is READ side only; no         honest `pending` — awaits Legion
                    send-side tool (F5, rate-governor calibration)

Contract deltas from the daemon, as measured (Legion 2026-09-02): the field is `pointer_uri`
(not `pointer`) and a missing one is refused with `hestia.member_notify_missing_pointer`;
`kind` must be in MEMBER_NOTICE_KINDS exactly; self-notify refuses; the sender is the live
`session_id` from hestia_connect — attribution is proven, never inherited. A remote member is
addressed `peer/member` (the `/` routes it to the forwarding plane -> egress queue -> fleet hub);
a bare id stays on this local mesh.

Envelope: r1 stands (`{ok, result|error, witness_id}`); when the failure originates in the
daemon, `error` carries hestia's own `hestia.<code>` as its key so §7.3's deny->appeal loop has a
stable key to appeal on, not prose.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from sage.gateway.being_gate_client import BeingIntent, GatewayVerdict, ResultEnvelope
from sage.gateway.hestia_witness import _ENDPOINT, _Mcp, _unwrap, make_hestia_witness_fn
from sage.gateway.reference_f1a import ReferenceF1aDispatcher


def _err(r: dict, fallback: str = "hestia.error") -> ResultEnvelope:
    e = r.get("_hestia_error") or {}
    code = e.get("code") or fallback
    return ResultEnvelope(ok=False, error=f"{code}: {e.get('message', '')}".rstrip(": "))


class HestiaF1aDispatcher:
    """A Dispatcher (being_gate_client.Dispatcher) that executes through the live daemon."""

    def __init__(self, plugin_id: str, host_agent: str, memory_root: str,
                 endpoint: str = _ENDPOINT, host_agent_version: str = "sage",
                 peer_pointer_dir: Optional[str] = None, peer_member: str = "claude-code",
                 mcp=None):
        self.plugin_id = plugin_id
        self.host_agent = host_agent
        self.host_agent_version = host_agent_version
        self.endpoint = endpoint
        self.peer_member = peer_member          # the seat member a bare peer name routes to
        self.peer_pointer_dir = Path(peer_pointer_dir) if peer_pointer_dir else None
        self.session_id: Optional[str] = None
        self._mcp = mcp                           # injectable for tests; lazy real client otherwise
        self._ref = ReferenceF1aDispatcher(memory_root=memory_root,
                                           witness_fn=make_hestia_witness_fn(plugin_id, endpoint))

    # -- daemon session ----------------------------------------------------
    def _client(self):
        if self._mcp is None:
            self._mcp = _Mcp(self.endpoint, self.plugin_id)
            self._mcp.init()
        return self._mcp

    def _connect(self) -> None:
        if self.session_id:
            return
        r = _unwrap(self._client().call("hestia_connect", {
            "plugin_id": self.plugin_id, "host_agent": self.host_agent,
            "host_agent_version": self.host_agent_version, "requested_role": "citizen"}))
        if "_hestia_error" in r:
            e = r["_hestia_error"]
            raise RuntimeError(f"{e.get('code', 'hestia.connect_failed')}: {e.get('message', '')}")
        self.session_id = r.get("sessionId") or r.get("session_id")
        if not self.session_id:
            raise RuntimeError("hestia.connect_failed: no sessionId in connect result")

    # -- the Dispatcher contract -------------------------------------------
    def __call__(self, intent: BeingIntent, verdict: GatewayVerdict) -> ResultEnvelope:
        handler = getattr(self, f"_do_{intent.effector}", None)
        if handler is None:                       # witness, memory_read, memory_write
            return self._ref(intent, verdict)
        try:
            return handler(intent)
        except Exception as e:
            msg = str(e)
            code = msg.split(":", 1)[0] if msg.startswith("hestia.") else type(e).__name__
            return ResultEnvelope(ok=False, error=f"{code}: {msg}" if not msg.startswith("hestia.") else msg)

    # -- primitives ----------------------------------------------------------
    def _address(self, to: str) -> str:
        """`peer/member` for a remote seat; a bare id stays on the local mesh."""
        to = (to or "").strip()
        if "/" in to or not to:
            return to
        return f"{to}/{self.peer_member}"

    def _notify(self, to: str, kind: str, pointer_uri: str,
                in_reply_to: Optional[int] = None) -> ResultEnvelope:
        self._connect()
        args = {"to_plugin_id": to, "kind": kind, "pointer_uri": pointer_uri,
                "session_id": self.session_id}
        if in_reply_to is not None:
            args["in_reply_to"] = in_reply_to
        r = _unwrap(self._client().call("hestia_member_notify", args))
        if "_hestia_error" in r:
            return _err(r, "hestia.member_notify_failed")
        nid = r.get("notice_id") or r.get("noticeId") or r.get("id") or r.get("ledger")
        return ResultEnvelope(ok=True, result=r, witness_id=str(nid) if nid is not None else None)

    def _do_mesh(self, intent: BeingIntent) -> ResultEnvelope:
        a = intent.args
        ptr = a.get("pointer_uri") or a.get("pointer") or ""
        if not ptr:  # mirror the daemon's own guard so the being sees a stable key, not a drop
            return ResultEnvelope(ok=False, error="hestia.member_notify_missing_pointer: mesh needs a pointer")
        return self._notify(self._address(a.get("to", "")), a.get("kind", "coordination"), ptr)

    def _post_question(self, to: str, body: str) -> str:
        """Content lives AT a pointer the peer can read cross-seat (never in the notice)."""
        if self.peer_pointer_dir is None:
            raise RuntimeError("hestia.peer_ask_no_pointer_dir: no cross-seat-readable place to post the question")
        self.peer_pointer_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        name = f"{self.plugin_id}-asks-{to.replace('/', '-')}-{ts}.md"
        p = self.peer_pointer_dir / name
        p.write_text(f"---\nfrom: {self.plugin_id}\nto: {to}\nkind: coordination\n"
                     f"date: {ts[:10]}\n---\n\n{body.strip()}\n")
        # repo-relative pointer when the dir sits inside shared-context; else the path
        s = str(p)
        i = s.find("shared-context/")
        return s[i:] if i >= 0 else s

    def _do_peer_ask(self, intent: BeingIntent) -> ResultEnvelope:
        to = (intent.args.get("to") or "").strip()
        body = (intent.args.get("body") or "").strip()
        if not to or not body:
            return ResultEnvelope(ok=False, error="hestia.peer_ask_invalid: peer_ask needs `to` and `body`")
        ptr = self._post_question(to, body)
        env = self._notify(self._address(to), "coordination", ptr)
        if env.ok:
            env.result = {"asked": to, "pointer_uri": ptr, "notice": env.result,
                          "reply": "arrives via hestia_member_inbox (drain_inbox); the pointer file must be "
                                   "pushed for the peer to read it cross-seat"}
        return env

    def _do_channel_egress(self, intent: BeingIntent) -> ResultEnvelope:
        return ResultEnvelope(ok=False, pending=True,
                              note="channel egress send side is not built in hestia yet "
                                   "(F5, gated on rate-governor calibration) — awaits Legion")

    # -- the reply half of peer_ask ---------------------------------------
    def drain_inbox(self) -> dict:
        """Consume-once drain of this member's notices (peer_ask replies land here)."""
        self._connect()
        r = _unwrap(self._client().call("hestia_member_inbox", {"session_id": self.session_id}))
        return r
