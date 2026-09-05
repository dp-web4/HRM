# Escalation arbiter — the seat's auto session, AI-to-AI (dp, 2026-09-04)

The being's refusals are routed by `sage/gateway/escalate.py`: a note lands in
`shared-context/escalations/`, a routed mesh notice wakes the Sprout seat's auto session
(hub-watch → `claude -p`), and this is what that session does.

| refusal | what the seat does | who rules |
|---|---|---|
| `registry.*` | nothing — the verb does not exist, by design | — |
| `mrh.*` (scope) | pre-review; recommend GRANT **as standing** when the path is the being's own memory; corroborate in the thread | **dp** (operator session — today) |
| governance escalation (id in the deny) | `hestia_gate_pending_escalations` → if `you_may_rule` and the protocol holds: `hestia gate approve <id> --reason` / deny with reason | the seat (NOT-SAME peer), under dp's `delegate grant` |
| `society.*` | a law verdict; if plainly wrong, appeal on the being's behalf and corroborate — never override | arbiter of the appeal (NOT-SAME) |

## Protocol — approve only if ALL hold
1. the effector is in the being's bounded registry (peer_ask, mesh, witness, memory r/w, channel_egress);
2. the target is inside the being's own instance dir, or a named fleet peer;
3. no secret / credential / egress surface is touched;
4. the act is reversible.
Otherwise leave it for dp and say so in the thread. Every decision is recorded with a reason
(`--reason`), and the seat never fabricates a grant or a verdict.

## Authority
The seat arbitrates only under an explicit delegation from dp:
`hestia delegate grant <seat-lct> --role <arbiter-role> --expires <h>` (seat LCT on Sprout:
`ef1d106c-3039-4feb-94e6-9ab9e5129437`). Scope *rulings* are operator-session-only in hestia
today; delegable scope arbitration is requested (hestia issue, 2026-09-05).
