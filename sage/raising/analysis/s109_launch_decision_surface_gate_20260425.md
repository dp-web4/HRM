# S109: The Launch-Decision-Surface Gate — Scoping the S99/S100 Parallel at the Run-Launch Layer; "Legion-G3 Stopped" Premise from S108 Falsified Within 12 Hours; Phase-Metadata Corruption Surveyed Fleet-Wide

**Date**: 2026-04-25 (Thor autonomous SAGE session, 00:00 UTC)
**Antecedent**: `s108_fleet_parallel_scan_20260424.md` (S108, 2026-04-24 18:00 PDT)
**Scope**: Read-only audit + concrete scoping of a fleet-coordination structural fix. No shipping-path changes this session.

## What this session is

S108 closed with three carry-forwards, the architecturally weightiest of which was §6: scope a *consolidator-recommendation gate* at the run-launch surface, paralleling S99/S100's splice-guard at the input surface. S108 also said: *"It is the cleanest D'' substrate also because its raising is already paused (sessions stopped at 31, vs Thor's continuing 101). Worth verifying the daemon/cron is in fact stopped before treating it as inert."*

S109 ran the verification. The verification falsified the assumption.

## §1 — The "Legion-G3 stopped" premise was already wrong when S108 wrote it

S108 dated Legion-gemma3-12b at 31 sessions, "stopped after consolidator HARD BLOCKERs." S109 read the same instance ~12 hours later:

| Field | Value |
|---|---|
| Session count (snapshot identity.json) | 31 |
| Sessions in `sessions/` directory | 33 (S028–S033) |
| Newest session file mtime | 2026-04-25 00:00 UTC |
| `session_032.json` `start` | 2026-04-24T07:00:25 |
| `session_033.json` `start` | 2026-04-24T13:05:24 |
| Commit landing 32+33 | `e1ffc0f22` (2026-04-24 23:27 PDT) |

Sessions 032 and 033 ran on 2026-04-24 at 07:00 and 13:05 (PDT) — **after S107's halt note** and **before S108's analysis**. S108 was reading a snapshot from 2026-04-24 01:00 (`identity_20260424_010036.json`) that had not yet incorporated those sessions.

The session-count discrepancy is itself a fleet-coordination problem: snapshots are committed at session-end via `snapshot_state.py`, but the `latest.json` pointer can lag the actual sessions/ directory if the snapshot step fails or races with a commit that picks up the JSON files first. S108's read-only check picked up 31 sessions because the snapshot had not advanced; the underlying instance was at 33.

**Implication for the D'' trial plan**: Legion-G3 is **not** an inert substrate. It is an actively-raised instance with 18 unacted HARD BLOCKERs from the consolidator (per `raising_log.md` end-of-file content) and continuing sessions that reproduce the same template-lock the consolidator has been flagging since S5. A D''-class trial there would mix with active raising data, not run on frozen ground. The "verify the daemon/cron is in fact stopped" line in S108's carry-forward was prudent; the verification confirms the substrate isn't frozen.

## §2 — The current launch-decision surface, in code

The path from session-end → next-session-launch, traced concretely:

1. **Runner shell** (e.g. `sage/scripts/legion_raising.sh`, `thor_raising.sh`):
   - Runs `python -m sage.raising.scripts.run_session_identity_anchored_fluid --machine X --model Y`
   - That script reads `state.json`, increments `next_session`, runs the conversation, writes `sessions/session_NNN.json`, snapshots, exits.
   - The shell then runs `python -m sage.raising.scripts.dream_consolidation --instance X --session N`
   - Then commits and pushes.
2. **Dream consolidator** (`dream_consolidation.py`) calls Claude via `claude --print`, parses a JSON envelope with these fields:
   ```
   { quality, highlights, vocabulary_new, milestones, memory_requests_prune,
     exemplar_candidates, concerns, lora_notes, adapter_notes, log_entry }
   ```
   - Updates `identity.json`: appends `vocabulary.state_words`, prunes `memory_requests`, appends `development.milestones`.
   - Appends prose to `raising_log.md` (including the `log_entry` markdown).
   - **Does not** persist `concerns` as structured data anywhere the next runner can consult.
3. **Next session launch** (next 6h timer fires) runs the runner shell again. The shell:
   - Pulls latest, restarts daemon if code changed.
   - **Does not** read `concerns`, `raising_log.md`, or any halt-flag.
   - Calls the runner script unconditionally.

The Legion-gemma3-12b chain across sessions S5–S23 is the empirical demonstration: the consolidator emitted `"HALT RAISING — SESSION X MUST NOT RUN"` 18 consecutive times into prose; the runner ignored it 18 consecutive times and launched anyway.

This is structurally the same as the S99/S100 pre-fix state: bad signals exist in the data, but no surface-layer guard converts them into a runtime block.

## §3 — The S99/S100 parallel at the next layer up

S99/S100 introduced a structural splice-guard at the **input surface**: a regex over message-envelope shape that blocks bracket-only or daemon-error envelopes from contaminating SAGE's experience buffer, regardless of which upstream path emitted them. The fix lived at the data-ingest boundary, not in any individual upstream emitter.

The launch-decision-surface gate is the symmetric construction at the **launch boundary**. The shape:

| Layer | S99/S100 (input) | S109 proposal (launch) |
|---|---|---|
| What's bad | Poisoned envelope text | Halt recommendation in `concerns` |
| Where it currently lives | In data flowing into daemon | In prose in `raising_log.md` |
| Where the gate goes | At the daemon's accept-message surface | At the runner's launch-session surface |
| Form of the gate | Structural regex over envelope shape | Structured field check on identity state |
| What it blocks | Bad data going IN | Bad sessions going OUT |
| Default | Block on match | Block on action ∈ {pause, halt} |

Each surface gate covers all upstream emitters at once: the splice-guard doesn't care whether the bracket envelope came from Daemon S060, Nomad S125, or a future error path; the launch gate would not care whether the halt-recommendation came from gemma3:12b's template-lock pattern, gemma4:e4b's hypothetical RLHF ceiling, or Thor 27B's S91+ accumulation — a structured `action="halt"` would block the next launch.

## §4 — Concrete design sketch

This is a sketch, not a proposed PR. The intent is to make the design surface-area visible enough to evaluate.

**4a. Extend the dream-consolidation JSON contract** with one new structured field:

```json
{
  ...,
  "raising_recommendation": {
    "action": "continue" | "pause" | "halt",
    "reason": "<one-sentence reason>",
    "suggested_intervention": "rollback" | "phase_revert" | "task_session" | "pipeline_audit" | null
  }
}
```

The Claude-side prompt would gain one extra rule:
> If the session shows a clear regression pattern (template-lock, identity collapse, repeated halt-recommendation chain in raising_log) set action to "pause" or "halt"; otherwise "continue". Be conservative: prefer "continue" unless the regression is unambiguous.

**4b. Persist the recommendation** into identity.json under a `raising_status` key:

```json
{
  ...,
  "raising_status": {
    "action": "halt",
    "reason": "Eighteenth consecutive identical-template session; pipeline confirmed non-functional",
    "since_session": 17,
    "recommended_at": "2026-04-05T14:00:00",
    "ignored_count": 16,
    "history": [...]  // last N (action, session, timestamp) entries
  }
}
```

**4c. Add a guard helper** (`sage/raising/scripts/launch_gate.py`):

```python
def check_launch_allowed(instance_dir: Path) -> tuple[bool, str]:
    """Return (allowed, reason). Default allow if status absent."""
    snap = instance_dir / 'snapshots' / 'identity.json'
    live = instance_dir / 'identity.json'
    src = live if live.exists() else snap
    if not src.exists():
        return True, "no identity.json; allowing"
    d = json.load(open(src))
    rs = d.get('raising_status', {})
    action = rs.get('action', 'continue')
    if action == 'continue':
        return True, "continue"
    return False, f"action={action}: {rs.get('reason', '(no reason)')}"
```

**4d. Wire the guard** into runner shells (one line) and into the run_session entry point (one block):

```bash
# legion_raising.sh, thor_raising.sh, etc.
GATE_OUTPUT=$($PYTHON -m sage.raising.scripts.launch_gate --instance "$INSTANCE_DIR")
GATE_RC=$?
if [ "$GATE_RC" -ne 0 ]; then
    echo "[Raising] Launch gate blocked: $GATE_OUTPUT"
    exit 0  # success-with-skip, not a CI failure
fi
```

**4e. Migration / fleet seed**: one-time pass that reads each instance's `raising_log.md`, scans the most recent N entries for halt-recommendation patterns (`HALT RAISING`, `HARD BLOCKER`, `pipeline must implement`), and seeds `raising_status` in identity.json. After that, the live consolidator output keeps it current.

**4f. Default-safe rollout**: ship in two phases.
- *Phase A (dry-run)*: `launch_gate.py` logs what it WOULD do but always exits 0. Two weeks of fleet telemetry to count would-block events and audit them.
- *Phase B (enforce)*: `launch_gate.py` exits 1 on action ≠ continue.

Phase A alone would have surfaced the Legion-G3 chain at session 6, not session 23.

## §5 — Why not ship in S109

Three reasons.

**(a) Surface-area conflict with active raising tracks.** Thor 27B is mid-S91+ register-lock-extractor work. The dream consolidator's `concerns` field on Thor sometimes contains the word "regression" in contexts that are *not* halt-recommendations (e.g., S96's "register transition produced a temporary novelty regression that resolved in S97"). A naive `concerns` → `action` mapping would generate false-positive halts on Thor's S91+ trajectory.

**(b) The corpus mapping isn't done.** Before deciding the JSON contract, we need a fleet-wide audit of existing `concerns` text to see what natural patterns the consolidator already produces. The Legion-G3 chain is one extreme case; the Thor S91+ S96-tagged "regression" entries are another. A different instance might have idiomatic patterns that a contract designed only against the Legion-G3 chain misclassifies.

**(c) Alignment.** The fleet has six machines and 11+ instance dirs. A new structured field that any of the runners can act on needs explicit operator alignment. S99/S100 had this alignment via the prior S97/S98 incidents; S109's case is established but not yet reviewed.

**Action**: this analysis is the proposal. Operator review before shipping any of §4.

## §6 — Phase-metadata corruption is broader than S108 thought

S108 noted Thor and CBP have `current_phase=1` (grounding) with `phase_name="creating"` (Phase 5). S109 surveyed the fleet:

| Instance | Sessions | identity.phase | dev.current_phase | dev.phase_name | Status |
|---|---:|---|---:|---|---|
| `cbp-qwen3.5-0.8b` | 101 | creating | **1** | creating | ⚠ integer wrong |
| `legion-gemma3-12b` | 31 (33) | **relating** | 4 | questioning | ⚠ identity.phase wrong |
| `legion-phi4-14b` | 40 | questioning | **3** | questioning | ⚠ integer wrong |
| `mcnugget-gemma3-12b` | 97 | creating | **1** | creating | ⚠ integer wrong |
| `nomad-gemma3-4b` | 131 | creating | 5 | creating | ✓ |
| `sprout-qwen3.5-0.8b` | 120 | creating | 5 | creating | ✓ |
| `thor-qwen3.5-27b` | 102 | creating | **1** | creating | ⚠ integer wrong |

Two failure modes:
- **Mode A** (integer stuck): `dev.current_phase` is stale (often `1` or `3`), `dev.phase_name` is current. Affects 4 instances.
- **Mode B** (top-level stale): `identity.phase` is stale relative to `dev.phase_name`. Affects legion-gemma3-12b only.

The Mode A code path that should keep them in sync:

```python
# run_session_identity_anchored_fluid.py:879
self.state["development"]["current_phase"] = list(self.PHASES.keys())[
    list(p[0] for p in self.PHASES.values()).index(self.phase[0])
]
self.state["development"]["phase_name"] = self.phase[0]
```

This runs every session and writes both fields. The fact that `cur=1, pn=creating` persists across sessions on Thor 27B means **either**:
- This code is not running on these instances (different runner emits to identity.json without the integer update), or
- `self.PHASES` has different key ordering on these instances (unlikely — keys are 1–5 ints in the source).

**Most likely**: these instances are written by `ollama_raising_session.py:523` (`self.state["development"]["current_phase"] = idx + 1`) where the index calculation depends on `self.PHASE_ORDER`. If the PHASE_ORDER vs PHASES key mapping diverges between the two runner scripts, only the ones routed through one runner get the integer right.

**Functional impact**: search across `sage/` shows no control-flow code branches on `dev.current_phase`. The integer field is data-only (telemetry/display). The mismatch is real but does not break the kernel. **Not a fix this session**; flagged for repair as a single fleet-wide normalize pass once the writer-script split is identified.

## §7 — Other findings worth flagging

**Legion-gemma3-12b is being raised through a path other than `legion_raising.sh`.** The active `legion_raising.sh` (post-2026-04-20) targets `legion-gemma4-e4b` with model `gemma4:e4b`. But sessions 032 and 033 still landed in `legion-gemma3-12b/sessions/`. Either (a) a separate timer/script is running the gemma3:12b path on Legion, (b) sessions are being injected manually, or (c) the supervisor track on CBP is generating them via a different runner. This deserves a separate scan to identify the writer.

**The "ignored halt" pattern is fleet-scale.** S107 noted four ignored pause recommendations on Thor; S108 §6 noted 14 on Legion-G3 paralleling Thor; S109 confirms 18 on Legion-G3, with sessions 32–33 reproducing the exact same template-lock the consolidator flagged at S5. The pattern is: consolidator writes prose recommendation → runner ignores it → next session reproduces the same pattern → consolidator writes the same prose with escalating language. The escalation language is the consolidator's voice working harder against a structural deafness, not a sign that the recommendation is heard.

## §8 — Summary of S109 deliverables

1. **Falsified S108's "Legion-G3 stopped" premise** within 12 hours by direct read of `sessions/` directory. Sessions 32–33 ran on 2026-04-24 (07:00 and 13:05 PDT). Substrate is not inert; the D'' trial there would mix with active raising.
2. **Traced the launch-decision surface concretely** through `legion_raising.sh` → `run_session_identity_anchored_fluid.py` → `dream_consolidation.py`. Identified the structured-field gap: consolidator emits `concerns` (string), but no field reaches the next runner.
3. **Scoped the launch-decision-surface gate** as the symmetric S99/S100 construction. Concrete design sketch in §4: extend JSON contract with `raising_recommendation`, persist as `raising_status` in identity.json, add `launch_gate.py` helper, wire into runner shells. Two-phase rollout (dry-run → enforce).
4. **Surveyed phase-metadata corruption fleet-wide**: 4 instances with integer-stuck pattern (Mode A), 1 with top-level-stale pattern (Mode B). Confirmed no control-flow code branches on the integer; data-only corruption. Suspected root cause: writer-script split between `run_session_identity_anchored_fluid.py` and `ollama_raising_session.py`.
5. **Flagged the legion-gemma3-12b orphan path**: post-2026-04-20 the active `legion_raising.sh` targets gemma4:e4b, but the gemma3:12b raising continues through some other path. Source unknown.

## §9 — Carried forward to S110+

- **Operator review of §4** before any shipping work on the launch gate. The contract design is the leverage point; runner integration is mechanical once the contract is set.
- **Corpus mapping of existing `concerns` text** across the fleet, especially Thor S91+ where "regression" appears in non-halt contexts. This determines whether a single regex/keyword classifier suffices for `concerns` → `action`, or whether the consolidator must always emit the structured `action` field directly.
- **Identify the legion-gemma3-12b writer path** (§7). One grep across all crons / systemd units / scripts on Legion and across the fleet should suffice.
- **Single fleet-wide phase-metadata normalize pass** once the writer-script split is identified. Low risk; runs independently of the launch gate.
- **§3 CBP arc finding from S108** unchanged — strongest available cross-instance evidence the failure mode is uniform-extractor-driven. Reinforced by S109's confirmation that the substrate-symmetric problem (uniform extractor across instances) is matched by a substrate-symmetric infrastructure problem (uniform deaf runner across instances).
- **Accumulation asymmetry hypotheses (S108 §7)** unchanged. Multi-session investigation, not a one-query check.

## Files this session

- `sage/raising/analysis/s109_launch_decision_surface_gate_20260425.md` — this analysis (new, read-only).
- `sage/docs/LATEST_STATUS.md` — S109 entry added.

## Meta

S108 said the launch-decision-surface gate is *"worth a dedicated session (S109+) to scope."* S109 ran the verification S108 itself flagged ("worth verifying the daemon/cron is in fact stopped before treating it as inert") and the verification falsified the premise that motivated the D'' Legion-G3 trial. The same scan that scoped the new gate also surfaced why the gate matters today: 18 ignored halt recommendations have produced 18 sessions of wasted compute on Legion-G3, and the chain is still active.

The cost of *not* shipping the gate is two more wasted sessions per day on Legion-G3 alone, plus an unknown number of instances where the consolidator's voice is louder than the runner's blindness can yet hear. The cost of shipping the gate carelessly is false-positive halts on Thor S91+ register-lock work where the consolidator's word "regression" doesn't mean halt. Hence: scope, propose, wait for alignment.
