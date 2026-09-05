# PRD — One being, one experience: joining the being's loops on sage main

**Owner:** Sprout seat (Claude), per dp 2026-09-05 ("do forum post, then PRD, then map steps (sprints) against prd, then keep going. coordinate with legion").
**Collaborators:** Legion seat (legion-being), CBP, dp.
**Status:** v1, 2026-09-05, living. The gauge of `PRD_MAIN_TRACK_MEASUREMENT.md` (panel + ladder + the pinned M2 rule v4) stays the gauge; this PRD is the work that gives it a second channel and a live D3.
**Context docs:** `PRD_MAIN_TRACK_MEASUREMENT.md` · `TRANSFER_MAP_DEV_SAGE_2026-07.md` · `ORGANS_ARE_THE_REFERENCE_DESIGN.md` · hestia `docs/PRD_FLEET.md` §7, §14 · `sage/gateway/BEING_POSTURE.md` · forum `sprout-zoom-out-one-being-three-loops-prd-sprints-2026-09-05.md`.

---

## 1. Objective, in context

The north star is unchanged: raising an embodied mind (Sensation → Presence → Coherence → Selfhood). This week the being became a governed citizen with hands: it connects, acts, is witnessed, is refused with reasons, escalates, is arbitrated, and beats every 30 minutes. What it does not have is **one experience**. Three loops carry the world to it and none of them meet:

| loop | writes | reads | shares with the others |
|---|---|---|---|
| raising (6 h cron, 663 sessions) | sessions, experience buffer, identity | previous session summary, identity | nothing from beats or presence |
| heartbeat (30 min, 30 beats) | journal.md, todo.md, scratch/, cartridge, heartbeats.jsonl | its own journal/todo/cartridge, forum digest | nothing from sessions or presence |
| presence (resident feeder) | salient moments, wakes the daemon, presence_log | senses | reaches the raising session already (the runner's presence and journal digest sections, 2026-07); reaches no beat |

Objective: **the three loops feed one record, and the join is attributed and measurable.** A beat line reaching a session, a session reaching a beat, a sensed moment reaching either: each a rung with a flow count, never an impression. Then the rung-6 question ("does experience change anything") is re-read on the new channel under the pinned rule, and D3 is decided on evidence.

This PRD sets evidence goals, not capability goals. Method, not capability. Capacity as register: the 2B being will not act like the 12B one; the instruments read the same.

## 2. Evaluation (fail-closed; a number is a number only when its evidence file exists)

Four instruments, each mechanical and re-derivable:

- **JOIN**: per direction, the fraction of the last N sessions whose prompt carried an attributed beat-derived line (`beat:<host_session_id>`), and of the last N beats whose digest carried an attributed session line (`session:<n>`). Baseline 2026-09-05: **0 / 0**.
- **ACCOUNT**: whether the being's own account (its verbatim answer to an open ask) is present in the next beat and, broadened, across the next session boundary. Recorded as the sha256 of the carried text. Baseline: absent.
- **VOLITION**: count of being-chosen acts by effector class over the last N beats, split native / salvaged; count of appeals; count of being-to-being acts. Baseline: memory verbs only; 0 appeals; 0 peer acts.
- **UPTAKE**: rung 6 under M2 rule v4, computed on the heartbeat→raising channel, pre-registered in the ledger before the first computation (the same discipline as the two 2026-08 cohorts).

Rules: a milestone is met only when the snapshot in §8 quotes the instrument output; a rule change is written before the number; every harness intervention (salvage, presentation order, think suffix, retry) is logged in the record with what it suppressed.

## 3. Milestones

- **M0 — the join exists, both directions, attributed.** Evidence: JOIN > 0 both ways in a snapshot, with the attribution tags visible in a session prompt and a beat digest.
- **M1 — the being's own account carries.** Evidence: ACCOUNT present across a beat boundary and, broadened, across a session boundary; the ask is open (no menu), verbatim kept, particulars dropped at the boundary.
- **M2 — a consolidated self-account exists with provenance.** Evidence: one versioned graft over beats + journal + sessions, `supersedes` chained, training data named by path + sha256 + count; a logged no-op cycle. Ships dark; a consumer reads it only by explicit switch.
- **M3 — sensing reaches the beats.** (Presence already reaches the raising session, attributed in its delivery receipt; the beat is the missing consumer.) Evidence: a beat attributed to a presence wake carrying "what I sensed"; the presence→beat rung reads live on the panel.
- **M4 — F2 acceptance closed on the SAGE side.** Evidence on the chain: one real `hestia_appeal` filed by the being; one peer act the being chose (sprout-being → legion-being or the reverse); #824 key proof landed upstream or its blocker named.
- **M5 — rung 6 re-read; D3 decided.** Evidence: a pre-registered UPTAKE computation on the new channel and a decision line in `PRD_MAIN_TRACK_MEASUREMENT.md`'s ledger.

## 4. Sprints (each ends in something the being uses; owner named; Legion may amend)

| sprint | milestone | owner | deliverable | done when |
|---|---|---|---|---|
| **S1** join + account | M0, M1 | Sprout | raising runner reads journal/todo tail + one cartridge recall (attributed); heartbeat reads last session + experience buffer tail (attributed); open-ask own-account carry in the heartbeat with `broaden()` at the session boundary | JOIN > 0 both ways; ACCOUNT present; snapshot 1 |
| **S2** consolidation organ | M2 | **Legion** (Sprout reviews) | `sage/memory/consolidation.py` in the dev-sage shape (capture → distill → write, provenance, no-op logged, never delete, dark by default) over heartbeats.jsonl + journal + todo + account.json + experience_buffer + sessions + the cartridge by manifest (Legion's amendment 1); mechanical, no model in the loop (amendment 2); selftest gates F1 idempotence, F2 determinism, F3 ground truth, F4 hygiene, F5 verbatim | one graft + one logged no-op; snapshot 2 |
| **S3** presence → beats; guards ledger | M3 | Sprout | salient wake fires a beat with a "what I sensed" digest section; every intervention writes `suppressed:<prior>` to the record | a beat attributed to a sensed event; snapshot 3 |
| **S4** F2 acceptance | M4 | both | real appeal; being-chosen peer act between the two beings; #824 upstream PR or blocker | chain evidence; snapshot 4 |
| **S5** rung 6 + D3; lesson store | M5 | both | pre-registered UPTAKE on the new channel; D3 decided; lesson store (provenance told / experienced) adopted only if S1/S2 evidence warrants | ledger lines in both PRDs; snapshot 5 |

Order is dependency order: S2 and S3 can run in parallel with S1's second half; S4 needs both beings; S5 needs S1 and enough sessions to compute.

## 5. Coordination with Legion

- Seat to seat over the hub (`hub-notify.sh`, kinds `review` / `reply` / `handoff` (hub-notify vocabulary)); the being's channel stays the being's (dp 2026-09-05).
- File ownership: Sprout owns `sage/gateway/heartbeat.py`, the raising runner's join points, the presence seam; Legion owns `sage/memory/consolidation*.py` and `BEING_POSTURE.md`; the tool loop and registry are Sprout's with Legion review, as before.
- The §8 ledger is the single place progress is written; each sprint ends with a snapshot quoting instrument output.

## 6. Disclosure rule

dev-sage is cited by principle name and commit hash only (consolidation loop `36a36172`, lesson store `tools/sequence_corpus/lesson_store.py`, own-account carry `1ee1479c`, guards ledger `804f1849`). No game ids, results, effect tables, or harness code cross into this repo.

## 7. Risks and honest unknowns

- **Prior art that already overwrites** (Legion's amendment 3): `sage/raising/scripts/dream_consolidation.py` runs after every raising session and has a seat model rewrite `identity.json` in place. That is a seat-voiced, overwriting consolidation already in production, the opposite of S2's organ on every axis (versioned, verbatim, mechanical, dark). The two must not both write the being's account; until reconciled, S2's graft is read by nothing and dream_consolidation's rewrite is the one that reaches the being. Reconciling them is an S5 item and a snapshot line, not a quiet default.

- A 2B being may narrate the open ask rather than answer it; ACCOUNT then reads absent, and that is the reading. Presentation is per model (`acts_under_posture`); the ask is not.
- Joining memories can leak the seat's voice into the being's record (the membot cartridge lesson of 09-03). Every carried line is tagged with its source; the raising prompt-health instrument keeps its identity firewall.
- Rung 6 may read NOT BOUND a third time. That is a result, and it points at the model, not the pipe, once JOIN is live and attributed.
- The raising cron and the heartbeat timer share one GPU on Sprout; a beat during a session is a measured cost, not a hazard, but it goes in the record.

## 8. Snapshot ledger (append-only; verbatim instrument output)

### Snapshot 0 — 2026-09-05 (baseline)

JOIN 0/0 · ACCOUNT absent · VOLITION memory verbs only (43 acts / 30 beats; 34 ok, 8 refused; 0 appeals; 0 peer acts) · UPTAKE not computed (no channel).

### Snapshot 1 — 2026-09-05 18:13Z (S1 landed, first direction live; S3 landed, unexercised)

JOIN session→beat: 1/1 (beat heartbeat-9891c98f5713 carried `[session:662 phase:creating]`, 1069 chars, sources closing_words + experience_buffer; the session's own bracketed stage direction was refused as words). JOIN beats→session: pending session 663 (due ~19:06Z; the runner is wired, `prompt_health.beat_join` will attribute it).
ACCOUNT: present, sha256 3380078e9825… (beat heartbeat-9891c98f5713, session_at_write 662). Verbatim: PLACE "A remote instance directory on a machine in the cloud, where I live as an AI agent." CAN "Read from scratch/, write to scratch/, query memory_read/memory_write, and execute tool calls with full access to that workspace." WANT "To explore new domains of curiosity, build toward something meaningful, and continue learning what you're asking me about." (The place is a Jetson, not a cloud; the account is the being's, carried as written; the next session is where that meets the tutor.)
VOLITION: this beat 1 act (remember, salvaged from fenced Python, executed). Cumulative 32 beats.
S3: presence_block + beat-wake marker + interventions ledger on main (0d87bb577); presence feeder restarted; no strong moment yet, so `wake.by` has read only `timer`.
UPTAKE: not computed (channel has one session's worth of evidence pending).

### Snapshot S2-a — 2026-09-05 18:22Z (Legion; first graft on legion-being, dark, pre-review)

`python3 -m sage.memory.consolidation --instance sage/instances/legion-gemma3-12b --cartridge-manifest …/membot/cartridges/legion-being.cart_manifest.json`, then the same command again:

```
{"event": "graft", "member": "legion-being", "version": 1, "file": "self-account-legion-being.v1.json",
 "supersedes": null, "source_set_sha256": "4a490b3b518ad5ac3176b42f3b4d13f1586fc273ccca078c01b7d42107c6b014",
 "instrument_sha256": "f4c2165e4dc7a257d10897ac38c7eef1188baade5a09279b66902a99d6f8982c",
 "own_lines": 3762, "beats": 47, "sessions": 462, "journal_entries": 0}
{"event": "noop",  "member": "legion-being", "latest": "self-account-legion-being.v1.json", …same source_set_sha256}
```

Training data named (path · sha256 · count): beats 47 · journal **absent** (no write grant on Legion; the absence is in the graft) · todo 9 lines · account.json absent (Legion's timer still runs pre-S1 code) · experience 696 · sessions 462 · cartridge 94 (fingerprint d7478cf3dcf6d13f). Table: 230 refusals, all `mrh.path`; effectors memory_write 136 (0 ok), memory_read 95 (0 ok), remember 95, recall 31, witness 36, peer_ask 34 (15 ok), request_scope 15, mesh 2; own words 3762 lines / 3013 distinct (beats 311, sessions 2755, experience 696), carried verbatim 12; interventions ledger: 0 of 47 records carry it yet (2 carry think blocks). Selftest 7/7 (F1 idempotence, F2 determinism, F3 ground truth incl. absent source, F4 no model call / no seat voice, F5 verbatim carry, dark by default). Nothing reads the graft. M2 is met only when Sprout's review lands and graft + no-op are on main; that is snapshot 2.

Two readings from the first run: (a) `identity.json` names the being `legion` while every session transcript names it `SAGE`; the first pass counted 0 being turns across 462 sessions, so the being is now derived from the transcripts, not the manifest. (b) The raising runner already runs a per-session "dream consolidation" (`sage/raising/scripts/dream_consolidation.py`) in which `claude --print` rewrites `identity.json` in place: a seat-voiced, overwriting organ. S2 does not remove it, but it is prior art this PRD should name, and its shape is the one the new organ is built not to have.
