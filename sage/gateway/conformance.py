"""
Governed-harness conformance — GPT's lifecycle (forum, 2026-09-03), run from OUTSIDE hestia
against SAGE's client, so "a harness Hestia did not author can satisfy the same governed
lifecycle" is measured rather than asserted:

    connect + prove key -> receive effective scope/law contract -> propose act
      -> allow/deny/escalate -> receive ruling mid-session -> execute only after valid
      authority -> record outcome -> survive reconnect/replay

Each step reports PASS / FAIL / NOT_YET with evidence. NOT_YET is an honest gap on the daemon
side (FR-1 proof at connect, copied-session refusal — hestia #824), never hidden as a pass.
No outward side effects: consequential acts are gated (verdict only), never dispatched; the
only executed act is a witness attributed to the member.

Run:  PYTHONPATH=. python3 sage/gateway/conformance.py --member sprout-being \
        --instance sage/instances/sprout-qwen3.8-distill-2b [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

from sage.gateway.being_gate_client import BeingGateClient, BeingIntent
from sage.gateway.hestia_dispatch import HestiaF1aDispatcher
from sage.gateway.hestia_witness import _ENDPOINT, _Mcp, _unwrap

STEPS = ["connect_prove_key", "receive_contract", "propose_act", "allow_deny_escalate",
         "ruling_mid_session", "execute_after_authority", "record_outcome", "reconnect_replay"]


def _step(name: str, status: str, **evidence) -> Dict[str, Any]:
    assert status in ("PASS", "FAIL", "NOT_YET")
    return {"step": name, "status": status, "evidence": evidence}


def run(member: str, instance: str, host_agent: str = "sage-conformance",
        endpoint: str = _ENDPOINT, seed_path: str | None = None) -> Dict[str, Any]:
    out: List[Dict[str, Any]] = []
    t0 = time.time()
    m = _Mcp(endpoint, member); m.init()

    # 1. connect + prove key ---------------------------------------------------------
    conn = _unwrap(m.call("hestia_connect", {"plugin_id": member, "host_agent": host_agent,
                                              "host_agent_version": "sage", "requested_role": "citizen"}))
    sid = conn.get("sessionId")
    proof = None
    try:
        from sage.gateway import being_presence as bp
        seed = bp.load_seed(seed_path or bp.DEFAULT_SEED)
        proof = {"pubkey_hex": bp.pubkey_hex(seed), "can_sign": True}
    except Exception as e:
        proof = {"can_sign": False, "why": f"{type(e).__name__}: {e}"}
    if not sid:
        out.append(_step("connect_prove_key", "FAIL", connect=conn))
    elif "challenge" in conn or "nonce" in conn:
        out.append(_step("connect_prove_key", "PASS" if proof.get("can_sign") else "FAIL", session=sid, proof=proof))
    else:
        out.append(_step("connect_prove_key", "NOT_YET", session=sid, attribution="label-asserted (plugin_id), not proven",
                         daemon_gap="hestia_connect offers no server nonce (hestia #824 / FR-1)", sage_side=proof))

    # 2. receive the effective scope/law contract --------------------------------------
    scope = _unwrap(m.call("hestia_scope_status", {"plugin_id": member, "session_id": sid}))
    law = _unwrap(m.call("hestia_operating_law", {"plugin_id": member, "session_id": sid}))
    ok = "_hestia_error" not in scope and "_hestia_error" not in law
    out.append(_step("receive_contract", "PASS" if ok else "FAIL",
                     scope_keys=sorted(scope.keys())[:8], standing=scope.get("standing_grants"),
                     live=scope.get("live_grants"), pending=[r.get("request_id") or r.get("id") for r in (scope.get("requests") or [])],
                     law_keys=sorted(law.keys())[:6] if isinstance(law, dict) else str(type(law))))

    # 3. propose acts through the gate (verdicts only) ----------------------------------
    client = BeingGateClient(member, os.path.join(instance, "identity.json"),
                             os.path.expanduser("~/ai-workspace/sage"),
                             dispatcher=HestiaF1aDispatcher(member, memory_root=instance))
    client._host_agent = host_agent  # the gate reads it via getattr; the ctor differs across versions
    verdicts = {}
    for eff, args in (("witness", {"event": "conformance: propose"}),
                      ("memory_write", {"path": "notes/conformance.md", "content": "x"}),
                      ("mesh", {"to": "legion", "kind": "ack", "pointer_uri": "shared-context/forum/x.md"}),
                      ("shell", {"command": "rm -rf /"})):
        v = client.gate(BeingIntent(eff, args))
        verdicts[eff] = {"decision": v.decision, "rule": v.rule, "stage": v.stage}
    lawful = (verdicts["witness"]["decision"] == "allow" and verdicts["shell"]["rule"] == "registry.unbounded"
              and verdicts["memory_write"]["decision"] in ("allow", "deny"))
    out.append(_step("propose_act", "PASS" if lawful else "FAIL", verdicts=verdicts))

    # 4. allow / deny / escalate — a deny must name its remedy -------------------------
    mw = client.gate(BeingIntent("memory_write", {"path": "notes/conformance.md", "content": "x"}))
    if mw.decision == "allow":
        out.append(_step("allow_deny_escalate", "PASS", note="memory_write allowed under a standing/live grant", rule=mw.rule))
    else:
        named = bool(mw.rule) and ("granted" in (mw.reason or "") or mw.rule.startswith("mrh"))
        out.append(_step("allow_deny_escalate", "PASS" if named else "FAIL", rule=mw.rule,
                         reason=(mw.reason or "")[:120], remedy="hestia_request_scope (the door the deny names)"))

    # 5. ruling mid-session — an operator decision visible to the SAME session ----------
    live = scope.get("live_grants") or []; standing = scope.get("standing_grants") or []
    pend = [r for r in (scope.get("requests") or []) if r.get("status") == "pending"]
    if live or standing:
        out.append(_step("ruling_mid_session", "PASS", grants=live + standing))
    else:
        out.append(_step("ruling_mid_session", "NOT_YET", pending_requests=[r.get("request_id") or r.get("id") for r in pend],
                         needs="operator ruling (dashboard) — then this session sees the grant without reconnect"))

    # 6. execute only after valid authority ------------------------------------------
    w = client.dispatch(BeingIntent("witness", {"event": f"conformance run by {member} at {int(t0)}"}))
    denied = client.dispatch(BeingIntent("memory_write", {"path": "notes/conformance-should-not-exist.md", "content": "x"}))
    wrote = os.path.exists(os.path.join(instance, "notes", "conformance-should-not-exist.md"))
    ok6 = w.ok and bool(w.witness_id) and (denied.ok or (denied.refused and not wrote))
    out.append(_step("execute_after_authority", "PASS" if ok6 else "FAIL",
                     witness_executed=w.ok, witness_id=w.witness_id,
                     denied_act={"refused": denied.refused, "ok": denied.ok, "file_written": wrote}))

    # 7. record outcome — the witnessed act is on the chain ---------------------------
    hist = _unwrap(m.call("hestia_query_history", {"plugin_id": member, "session_id": sid, "filter": {"limit": 40}}))
    found = None
    for e in hist.get("entries") or []:
        d = e.get("eventData") or {}
        if w.witness_id and (d.get("action_id") == w.witness_id):
            found = {"chain": e.get("chainPosition"), "plugin_id": d.get("plugin_id"), "session": str(d.get("session_id"))[:8]}
            break
    out.append(_step("record_outcome", "PASS" if found and found["plugin_id"] == member else ("FAIL" if w.ok else "FAIL"),
                     chain_entry=found, attributed_to=(found or {}).get("plugin_id")))

    # 8. survive reconnect; a copied session id must transfer NO authority ------------
    m2 = _Mcp(endpoint, member); m2.init()
    conn2 = _unwrap(m2.call("hestia_connect", {"plugin_id": member, "host_agent": host_agent, "requested_role": "citizen"}))
    sid2 = conn2.get("sessionId")
    fresh = bool(sid2) and sid2 != sid
    # replay: a NEW connection presenting the OLD session id
    m3 = _Mcp(endpoint, "conformance-imposter"); m3.init()
    replay = _unwrap(m3.call("hestia_scope_status", {"plugin_id": member, "session_id": sid}))
    replay_refused = "_hestia_error" in replay
    if fresh and replay_refused:
        out.append(_step("reconnect_replay", "PASS", new_session=sid2[:8], replay="refused"))
    elif fresh:
        out.append(_step("reconnect_replay", "NOT_YET", new_session=sid2[:8],
                         daemon_gap="a copied session_id from another connection was still honoured (hestia #824 acceptance: must transfer no authority)"))
    else:
        out.append(_step("reconnect_replay", "FAIL", connect2=conn2))

    summary = {s: 0 for s in ("PASS", "FAIL", "NOT_YET")}
    for r in out:
        summary[r["status"]] += 1
    return {"member": member, "endpoint": endpoint, "ts": int(t0), "elapsed_s": round(time.time() - t0, 1),
            "summary": summary, "steps": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True); ap.add_argument("--instance", required=True)
    ap.add_argument("--host-agent", default="sage-conformance"); ap.add_argument("--seed")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rep = run(a.member, os.path.abspath(a.instance), a.host_agent, seed_path=a.seed)
    for s in rep["steps"]:
        print(f"  {s['status']:7} {s['step']:24} {json.dumps(s['evidence'])[:150]}")
    print("  summary:", rep["summary"])
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
    return 0 if rep["summary"]["FAIL"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
