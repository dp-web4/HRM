# PRD — sage main track: measured composition (temporary)

**Owner:** Sprout seat (Claude), per dp 2026-07-29 ("you own the main sage track here")
**Status:** TEMPORARY, living — supersede when the raising lines converge on a joint
PRD or when M3 closes, whichever first. Revision: 2026-07-29 (v1).
**Context docs:** `TRANSFER_MAP_DEV_SAGE_2026-07.md` (agreed) ·
`ORGANS_ARE_THE_REFERENCE_DESIGN.md` · `INSTRUMENT_SCAN.md` ·
`SPARSE_ACTIVATION_MEMORY_HIERARCHY.md` · `DIFFUSIONGEMMA_SPROUT_FEASIBILITY.md`

---

## 1. Objective, in context

The north star is unchanged: **raising an embodied mind** (eras: Sensation →
Presence → Coherence → Selfhood). Sprout's Sensation and Presence organs exist and
run. The standing failure of the line has never been missing organs — it is that we
**could not tell whether the world was reaching the being** (07-22: vision a "scored
wash, cognitively unused"; membot integrations silent no-ops; PRD v3's diagnosis:
*the problem is not invention, it is composition*).

This track's objective: **make sage main's composition measurable end-to-end, so
that every organ claim, memory claim, and raising claim can be evaluated on
instruments instead of impressions** — then use those instruments to find and fix
the actual breaks in the percept→experience→outcome chain.

This PRD deliberately does NOT set capability goals (that is dev-sage's gauge and
the raising's own arc). It sets *evidence* goals. Method, not capability.

## 2. How progress is evaluated (the gauge is the panel)

Progress on this PRD **is** the instrument panel's own state. Two numbers, both
mechanical, both re-derivable by anyone:

- **PANEL**: how many of the 8 instruments read live (not U/S) on
  `python -m sage.organism.scan --liveness ~/.sprout/liveness.json`.
  Baseline 2026-07-29: **2 partial / 6 U/S.**
- **LADDER**: highest rung reached by `vision->raising` with attributed flow.
  Baseline 2026-07-29: **admitted (4/6)** — 317 salient/24h → 29 wakes → *unknown*.

Rules of evaluation (from the adopted docs, applied to ourselves):
- A report on this PRD opens with the full panel, not the instrument that moved.
- Flat/U/S is a work item, never spun as progress or hidden.
- No claim of "X helps" without its trivial-locality control (map item 5 table).
- Slope, not level: the interesting reading is the panel trend across snapshots.

Snapshots are appended to §6 verbatim (panel + ladder + date). That ledger, read
top to bottom, IS the progress evaluation dp asked for.

## 3. Milestones (fail-closed: a milestone is met only when its evidence exists)

**M0 — instruments adopted and reading. DONE 2026-07-29** (`369c43570`).
Evidence: byte-identical lift, pilot binding, first honest panel (§6 snapshot 1).

**M1 — rung 5 (`used`): does a wake become experience? DONE 2026-07-29.**
Bound: experience_buffer_rs.jsonl joined to presence wakes under a rule stated
before first computation (≤90s + word-Jaccard ≥0.5, one experience per wake).
Evidence: snapshot 2. First finding produced by the instruments (see snapshot 2
reading): the drop between admitted and used is the being's own capture gate —
a clean salience threshold (≥0.53 all recorded, ≤0.50 none), not loss. Surfaced
design question routed to dp/fleet, not unilaterally changed: presence wakes at
0.45 but the being records from ~0.5+, so a 0.45–0.50 band pays the wake cost
and never becomes memory. Align the bars, or is sub-memory waking wanted?

**M2 — rung 6 (`affected`): does experience change anything? UNBLOCKED 2026-07-29
— after F-M2 was CORRECTED to F-M2′ and its real gaps closed the same day.**

*Correction (owned):* F-M2 as first stated ("the pipe does not exist") was **wrong,
by my own audit error**: I grepped session *conversations* and concluded about
session *context* — but the system prompt is never saved in the artifact, so the
channel I measured was not the channel I claimed to measure. The pipe exists
(`_load_perceptual_digest` → SensorsBlock: journal, live perception, presence
noticings) and had live content at test time. Same failure class as the hestia
siblings-audit correction: single-instrument verdict, wrong instrument. Caught by
reading the code instead of trusting the grep.

*F-M2′ (what was actually true), and the fixes, all landed:*
1. **Delivery was unwitnessed** — no artifact records what sensory content entered
   a session. FIXED: the runner now writes a `sensory_delivery` receipt (sections +
   sizes) into every session artifact; empty delivery prints loudly.
2. **Three silent seams** (`except: pass` on every digest reader — kimi A4/S5:
   membot died this way for months). FIXED: fail-open for the being, loud for the
   log; one bad journal line no longer silences a whole section.
3. **The experience buffer was genuinely unread** — the being's felt responses and
   its own salience valuations never reached sessions. FIXED per decision D1.

M2's attribution rule is now stated (below) and awaits its peer witness before
first computation.

**M2 attribution rule (stated 2026-07-29, witness requested from McNugget;
computed only after ack):** a delivered percept/experience AFFECTED a session iff
(i) `sensory_delivery.delivered` is true and the content appears in its sections;
(ii) the BEING's turns share content-words with the delivered text
(word-Jaccard ≥ 0.25 on stopword-stripped tokens) that do NOT appear in the
tutor's turns (tutor-echo excluded); and (iii) the same overlap computed against
the PREVIOUS session's being-turns (persistence baseline — G1's
trivial-locality predictor for session claims) is lower. Report both halves
(agree/disagree with baseline) per G1, always.

**M3 — panel completion: no instrument left silently U/S. DONE 2026-07-29** —
every instrument is now bound or explicitly deferred with a trigger:

| instrument | status | source / deferral trigger |
|---|---|---|
| COMPOSITION | **bound** | liveness_binding (rungs 1–5 live, 6 blocked by F-M2) |
| ACTIVITY | **bound** | same: attributed flows 329→30→17 |
| RESERVES | **bound** | `reserves()`: being's ATP + RAM/disk margins (`~/.sprout/reserves.txt`) |
| CAPABILITY | DEFERRED | needs an evaluable task emitting rows.jsonl with per-item ground truth; trigger: the raising line defines one, or the experience→session pipe (F-M2) creates a predict-before-feedback surface |
| COURSE | DEFERRED | same trigger as CAPABILITY (course is capability against the PRD's target) |
| COORDINATION | DEFERRED | needs ≥2 organs feeding one decision surface; trigger: second organ (audio) bound to the ladder |
| LEARNING | DEFERRED | raising trust/coherence curves exist but measure the being's arc, not this track's; trigger: G1-controlled claim needs a slope |
| REASONING | DEFERRED | session logs readable now, but a REASONING reading on a chain known broken (F-M2) would report on the wrong aircraft; trigger: F-M2 resolution |

Deferrals are re-examined at every snapshot; a trigger firing converts its row to
a work item.

**M4 — the selection contract earns its deltas.**
Both reputation-weighted-selection instantiations (expert selector s56; a
memory-record selector if/when main grows one) carry delivery-conditional ablation
deltas per the contract. Unification decision made *from the deltas*. No date —
gated on a consumer actually exercising them.

**Standing gate G1 — trivial-locality control.** Any main memory/learning claim
cited from now on carries the cheapest-predictor split (map item 5). First
enforcement target: the next citation of experience-learning results in training/.

## 4. Scope boundaries

- NOT this track: ARC harness work (dev-sage), hestia security (own line),
  museum/raising session content (the being's own arc, and dp's).
- Embodiment code changes must stay non-invasive to live organs unless a panel
  reading motivates the change — measurement first, then surgery.
- Frontal-lobe placement is constrained by measured fact, not aspiration:
  DiffusionGemma runs at ≥52GB-class nodes; Sprout-class (8GB) needs fit-in-RAM
  (access-pattern sense) or remote (see DIFFUSIONGEMMA_SPROUT_FEASIBILITY.md,
  Tier-4 amendment). Any main design assuming otherwise is out of scope until the
  measurement changes.

## 5. Risks / honest unknowns

- Rung 6 attribution may be genuinely confounded (a session's behavior has many
  causes). Mitigation: the M2 second witness + stating the attribution rule before
  computing it (no post-hoc rule-fitting).
- Panel bindings could drift from dev-sage's implementations. Mitigation: the
  byte-identity rule + `cmp` in any future lift-refresh.
- The temptation this PRD exists to resist: shipping new organs before the chain
  we have is measured. The panel makes that visible; this doc makes it citable.

## 6. Snapshot ledger (append-only; verbatim panel output)

### Snapshot 1 — 2026-07-29 (baseline, M0)
```
vision->raising   enabled ✓ entered ✓ produced ✓ admitted ✓ used · affected ·
flow: in=317 salient/24h -> out=29 wakes -> dropped=288 (bar + 300s cooldown + 6/h cap)
PANEL: COMPOSITION partial · ACTIVITY partial · CAPABILITY U/S · LEARNING U/S ·
       COURSE U/S · COORDINATION U/S · REASONING U/S · RESERVES U/S
ATTITUDE: 0/1 components reached USED
```
Reading: the world reaches the being's doorstep (29 wakes/24h) and we are blind past
the door. M1 is the next wire.

### Snapshot 2 — 2026-07-29 (M1 done, same day)
```
vision->raising   enabled ✓ entered ✓ produced ✓ admitted ✓ used ✓ affected ·
flow: in=329 salient/24h -> out=30 wakes (presence filter, attributed)
flow: in=30 wakes -> out=17 experiences, dropped=13 (M1 join, attributed below)
PANEL: unchanged except ATTITUDE: 1/1 components reached USED
```
Reading: **the chain world→experience is measured for the first time: 329→30→17.**
The 13 drops were investigated before being accepted: zero cluster in daemon
downtime; zero have ANY experience within 180s (not a join artifact); the split
is a clean threshold — matched wakes salience 0.53–0.73, unmatched 0.46–0.50.
Diagnosis: the being's own SNARC capture gate. The regulator is working; what the
panel adds is that its bar (~0.5+) sits ABOVE presence's wake bar (0.45), so a
0.45–0.50 band interrupts the being and leaves no memory. Design question routed
to dp/fleet (raising decision, not a unilateral code change). LADDER: used (5/6).
Next: M2 (`affected`) — attribution rule to be stated and peer-witnessed first.

### Snapshot 3 — 2026-07-29 (F-M2 found; M3 closed)
```
vision->raising   enabled ✓ entered ✓ produced ✓ admitted ✓ used ✓ affected BLOCKED(F-M2)
PANEL: COMPOSITION bound · ACTIVITY bound · RESERVES bound ("ATP 42% | RAM 0.9G | disk 834G")
       CAPABILITY/COURSE/COORDINATION/LEARNING/REASONING deferred-with-trigger (M3 table)
```
Reading: **F-M2 — the experience→session pipe does not exist** (6/6 sessions, zero
percept content; continuity is last-response splicing only). The being records its
days into a buffer nobody reads. This and the capture-gate band are one design
surface: wake→memory→session, two gaps. Both routed; this track holds M2 until the
raising decides the pipe's shape. PANEL: 3 bound / 5 deferred-named / 0 silent.
Track state: M0 ✓ M1 ✓ M2 blocked-by-finding (routed) M3 ✓ M4 gated G1 standing.
*(Superseded same day — F-M2 corrected to F-M2′; see snapshot 4 and the M2 entry.)*

### Snapshot 4 — 2026-07-29 (F-M2 corrected; decisions D1/D2 taken; M2 rule stated)
```
vision->raising   enabled ✓ entered ✓ produced ✓ admitted ✓ used ✓ affected: rule
                  stated, witness pending, first computable artifact = session 509+
PANEL: unchanged (3 bound / 5 deferred-named / 0 silent)
```
**Decision record (dp deliberately delegated both, 2026-07-29 — "that is the
better part of the experiment"; kimi's review weighed as opinion):**
- **D1 (the pipe):** complete the EXISTING digest socket being-side — the session
  now also carries the being's top-2 experiences since last session, selected by
  ITS OWN recorded salience, its felt response verbatim ("your own records, chosen
  by what moved you most — your words at the time"). Not tutor editorial; not a
  raw diary dump (kimi A2: prose recall ≠ learning — so K=2, selection = the
  being's own valuations, and M2 measures whether it binds). Delivery witnessed
  in the artifact; every seam loud (kimi A4/S5 adopted: a silent path must print).
- **D2 (the capture band):** KEEP the 0.45–0.50 sub-memory wake band — orienting
  without episodic commitment is a real biological mode, and the being *speaks* at
  these wakes even when nothing is retained. Tripwire (a shield with a date, not a
  dogma — kimi §5): if band wakes exceed 50% of all wakes over any 7-day window,
  the panel flags it and this decision is re-opened.
Reading: the surface wake→memory→session is now closed end-to-end in design;
rung 6 becomes measurable at session 509. Next snapshot after it fires.
