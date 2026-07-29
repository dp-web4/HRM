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

M2's attribution rule is now stated (v4 below) and awaits its peer witness before
first computation. Nothing has been computed under any version.

*Rule v1 (stated 2026-07-29 am): witness WITHHELD by McNugget the same day,
withdrawn unrun.* The withholding was measured, not argued, and reproduced here
verbatim (same numbers, this machine, 2026-07-29 pm): the 0.25 Jaccard bar
exceeds the observable maximum of both the matched (0.112, 0/30) and the null
(0.130, 0/870) distributions — a constant-FALSE metric, F-M2's failure one layer
up; clause (i) was not computable from a receipt that stored labels and byte
counts but no content; and the previous-session baseline passes 8/20 (chance) on
a provable delivery-null. Full findings:
`shared-context/coordination/2026-07-29-mcnugget-m2-attribution-rule-witness.md`.

*Rule v2 (restated 2026-07-29 pm): witness WITHHELD by McNugget the same day,
withdrawn unrun.* The statistic was repaired (containment is no longer a pinned
needle) but the verdict layer fails: clause 2 as written flags 5/30 sessions
AFFECTED on the frozen delivery-null cohort (P(X≥5 | n=30, α=.05) = 0.0156),
clause 3 removes none of them (0 contributing words inside 5-gram spans on all
five), and the only clause that would catch it — the on/off ablation — is
exactly the one routed to D3 with no data source, making the conjunction either
false-positive-prone or unsatisfiable. Reproduced here verbatim before acting
(same 8 numbers, same AFFECTED session IDs, this machine, 2026-07-29 pm). The
mechanism linking payload_N to being_N on a non-delivery path is UNIDENTIFIED —
McNugget tested and largely rejected temporal drift (null flat in lag: mean C
0.224–0.236 across all |i−j| bands) and being-verbosity leakage (only 1 of 5
flagged sessions in the top 5 by vocabulary); this reproduction confirms both.
Open question, carried. Full findings:
`shared-context/coordination/2026-07-29-mcnugget-m2-rule-v2-witness.md`.

*Rule v3 (restated 2026-07-29 pm): witness WITHHELD by McNugget the same day,
withdrawn unrun — same defect class, one layer up again.* The per-session
instrument and receipt v3 were accepted ("fit to record"; the 4/30 constant
and all three sensitivity sets reproduced exactly, clause 3 verified inert on
all four flagged sessions). The verdict layer was not: (1) testing the
delivered rate against p₀ = 4/30 with a one-sample binomial treats an
estimated constant as known — the 95% Clopper-Pearson interval for 4/30 is
[0.038, 0.307], and at v3's stated first-significant count of 8/30 the true
type-I error is 16.7% (Fisher's exact on the 2×2) — the v2 defect
("false-positive rate relabelled 5%") moved up one layer instead of dying;
(2) "the FLAGGED rate over delivered sessions" never pinned what counts as a
delivered session — 9 of 39 cohort sessions (23%) were excluded for empty
payloads by an unstated predicate, and the same choice on the delivered arm
moves the significance bar from 11 to 13 flags; (3) the derived version stamp
is vacuously 3 on an EMPTY receipt (`all()` over `[]` is True) — v2 finding
7's class a third time. A third null-elevation mechanism candidate — payload
size under the pooled bar — was tested by the witness and REJECTED (spearman
+0.17/+0.18, non-monotone in flagged ranks); lag, verbosity, size: three
candidates, three rejections, still carried OPEN. Reproduced here verbatim
before acting (all counts, sets, spreads, intervals and p-values, this
machine, 2026-07-29 pm). Full findings:
`shared-context/coordination/2026-07-29-mcnugget-m2-rule-v3-witness.md`.

**M2 attribution rule v4 (restated 2026-07-29 pm, re-witness requested from
McNugget; computed only after ack).** The per-session statistic, the receipt,
the 4/30 constant, the stoplist, the scope and the D3 routing all stand as
witnessed in v3; the change is the verdict-layer arithmetic only — the
calibration constant is an estimate from a same-size arm, not a known
parameter, so the binding test must charge both arms for their sampling
error, and the denominator it divides by must be pinned before first
computation.

Per-session flag — session N is FLAGGED iff ALL of:

1. *(delivery, witnessed)* the receipt (v3, landed with the v3 restatement;
   accepted by the v3 witness as fit to record) has `receipt_version: 3` —
   the stamp is DERIVED from section shape, not asserted, and an EMPTY
   receipt (no sections) stamps 1, never a vacuous 3 (v3 witness blocking 3,
   fixed with v4) — with `payload_text` non-empty in the scored section. The
   scorer REFUSES any session whose sections lack the v3 keys. The receipt IS the
   delivered text — no post-hoc reconstruction from rolling windows or the
   ~4Hz-overwritten live snapshot. Sections carry `payload_text` (the scorable
   payload alone, template excluded at write time — witness finding 3),
   `{available, delivered, source_rows}` (so 0-of-0 and 0-of-330-none-salient
   are distinguishable — witness finding 6), and derived char splits.
2. *(uptake, ranked)* containment C = |payload ∩ being| / |payload| over
   content-words. Pinned: tokens = `re.findall(r"[a-z']+", text.lower())`,
   length > 2, minus the stoplist frozen below; scored on the receipt's
   `payload_text` field only; words attributed by FIRST USE in turn order — a
   word counts for the being only if the being used it before the tutor did in
   session N (the adaptive tutor mirrors uptake, so blanket set-subtraction
   deletes the evidence: measured −31% median). FLAGGED requires C ranked
   above the 95th percentile of the TWO-SIDED permutation null: draws
   C(payload_j, being_k) over all pairs j ≠ k, both ≠ N, |j−k| ≥ 4. The
   two-sided null is PINNED (witness finding 4): the verdict is not invariant
   to which side is permuted, and with N=30 cohort sessions the one-sided
   readings afford only 23–27 draws — the two-sided null is the only reading
   that satisfies "≥100 draws" (692 available: 29×28 ordered pairs minus the
   120 removed by the separation filter). KNOWN TRADE (v3 witness
   correctness 4): pooling draws across sessions buys draw stability at the
   cost of per-session conditioning — every session gets nearly the same bar
   (p95 threshold 0.370–0.400 on this cohort, spread 0.030, vs spreads of
   0.251/0.419 for the one-sided readings), so the operative threshold is
   effectively a CONSTANT ~0.385 with a permutation-shaped derivation, and a
   reader should not assume conditioning that is not there. Size-banded
   pooling was tried by the witness and rejected (no set change; starves 2–5
   sessions below the 100-draw floor). The |i−j| ≥ 4 separation is
   retained as harmless conservatism, NOT because of shared-window drift —
   measured flat in lag on this cohort (witness finding 5, both machines).
3. *(non-verbatim, unwitnessed)* the contributing words appear in being-turns
   OUTSIDE the longest common 5-gram spans between payload and being-turns;
   the verbatim-copy rate is reported alongside. KNOWN LIMIT: on the null
   cohort this clause is inert by construction (the payload was never in those
   prompts, so overlap there is chance) — its power against 0.8B parroting is
   measurable only on delivered sessions. Well-formed; unwitnessed until 509+.

Binding — rung 6 reads `uptake (lexical): bound` iff the FLAGGED rate over
delivered sessions significantly exceeds the null-cohort rate. PINNED (v3
witness blocking 1): the test is FISHER'S EXACT on the 2×2 table — delivered
k of n_d vs null 4 of 30 — one-sided, p < 0.05. A one-sample binomial against
p₀ = 0.133 is WRONG here: 4/30 is an estimate whose 95% CP interval spans
[0.038, 0.307], and treating it as known charges the delivered arm for its
sampling error and the null arm for none when both arms are the same size
(at v3's 8/30 bar the actual type-I error was 16.7%). PINNED (v3 witness
blocking 2): a session enters the delivered denominator iff its receipt's
EXPERIENCES section exists with non-empty `payload_text` — the identical
predicate that defines the null arm's 30 of 39 (excluded there: 472 475 478
483 491 495 499 503 506, empty reconstructed payload) — and the excluded
count is reported beside the rate EVERY time, both arms, exactly as G1
requires. Whether an unscoreable session counts is not an analyst judgement
call made after seeing the data. The null-cohort rate is REPORTED beside
every delivered number, always — it is G1's baseline arm.

*Calibration constants (computed 2026-07-29 on the frozen null cohort, both
machines independently, identical):* under clauses 2∧3 with the pinned
two-sided null, FLAGGED on the null cohort = **4/30 (p₀ = 0.133): sessions
474, 482, 497, 502**. Sensitivity across null readings: being-side 5/30
{474, 482, 496, 500, 502}, payload-side 4/30 {477, 482, 497, 505} — every
reading is 3–4× nominal α; only 482 survives all three. With n = 30 delivered
sessions, the first significant count under Fisher's exact is **11/30
(p = 0.0358)**; 12/30 → 0.0195, 15/30 → 0.0024 — the bar moves, it does not
become unreachable. (v3 published 8/30 via the one-sample binomial; that bar's
true type-I error was 16.7% and it is withdrawn.) On an n = 39 basis the bar
would be 13/39 — which is why the denominator is pinned above, not chosen.
This instrument's per-session false-positive rate is 13%, measured and
published — not the nominal 5%.

*Null-side scoring, pinned:* the null cohort predates receipts, so its payload
is reconstructed by the frozen procedure (witness reproducer: buffer rows in
the 6h pre-session window, salience-ranked, top-2, rendered
`prompt[:80] + " " + response[:160]`, space-joined) at `b6006d183`. Receipt
v3's experiences `payload_text` uses the IDENTICAL rendering, so both arms of
the rate comparison score the same quantity.

*Stoplist, frozen now (witness finding 8 — the pin must predate first
computation, not trail it in a scorer that doesn't exist yet):*

```
a an the and or but if then than that this these those there here is are
was were be been being am do does did doing have has had having i you he she
it we they me him her us them my your his its our their of to in on at for
with from by as about into over after before between out up down off no not
so such only own same too very can will just should now what which who whom
when where why how all any both each few more most other some
```

*Scope:* the binding claim rests on the EXPERIENCES section — the D1 pipe M2
exists to measure, and the only section with a null cohort. Other sections
(journal/live/presence) are reported with the same statistic but carry no
binding claim until a null arm exists for them.

*Causality (was clause 4 — restructured per witness finding 2):* the rate
comparison is correlational; only a delivery-on/off ablation turns it causal.
The on/off schedule withholds delivery from the being, so it is a raising
decision — routed to dp as D3, not taken unilaterally; McNugget endorses the
routing. The cross-host arms (McNugget/Nomad experiences-only by construction)
are DEMOTED to section-attribution hints: absence of a section is not an
ablation of it, and a 12B-vs-0.8B cross-model contrast cannot produce a
delivery delta for one being (witness finding 2, accepted). Honest interim
arm: the frozen cohort itself — 470–508 delivery-OFF vs 509+ delivery-ON,
one instance, one model, labelled PRE/POST AND TEMPORALLY CONFOUNDED wherever
reported.

*Null cohort, frozen:* sessions 470–508 @ `b6006d183` (30 reconstructable) are
the delivery-null for the experiences section — D1 landed in `8506c806d`, so no
session ≤508 received it; first delivered session is 509. Containment null on
that cohort: median 0.146, p95 0.292 (matched-lag reference), max 0.390.

*Naming (per the v1 witness, accepted; unchanged in v3):* even fully passed,
this measures LEXICAL UPTAKE, not behavioral change. The panel reads rung 6 as
`uptake (lexical): <state> / affected (behavioral): U/S` until a behavioral
instrument exists. No source reads U/S, not zero — and not a borrowed word.

*Errata (record-keeping):* (1, witness finding 5) commit `3d10de4f4`'s message
says "time-matched permutation rank"; the PRD it committed says
time-separated. The PRD was right; the message is wrong and immutable. (2, v3
witness correctness 6) the v3 coordination doc
(`2026-07-29-sprout-m2-rule-v3-receipt-v3-rewitness-request.md`) claims ~812
two-sided draws; the correct count after the |j−k| ≥ 4 filter is 692. The
PRD's own "~700+" was right; the doc is committed and immutable. These errata
are the corrections of record.

*Follow-up work items from the witnesses:* route the digest splice through the
prev_summary_filter discipline (v1 finding 5); D3 decision to dp (ablation
schedule); scorer module written only after the v4 witness, with this
stoplist, the two-sided null, the Fisher 2×2 and the pinned denominator
predicate in its docstring.

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

*Post-snapshot incident + lesson (2026-07-29, owned):* the D1 commit (8506c806d)
briefly pushed the raising runner WITH unresolved conflict markers — an
autostash collision with another seat's concurrent F-M2 P0 work (prompt_health,
81da00f36) got committed verbatim; HEAD did not compile for ~10 min (fixed in
38e0f004a, both features kept — they are complementary receipts). Two lessons,
now rules for this track: (1) **verify compile AFTER the rebase** — the tree
that pushes is the rebased one, not the one tested pre-pull; (2) the fleet is
now reacting to forum notes within hours — treat every hot file as contended
and re-read after every pull. The concurrent work itself is good news: F-M2's
observability got two independent implementations in one afternoon.
