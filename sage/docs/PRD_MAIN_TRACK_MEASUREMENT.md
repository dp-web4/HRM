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

**M1 — rung 5 (`used`): does a wake become experience?**
Bind sage-daemon experience records to presence wakes. Done when the ladder can
show `wakes → experiences recorded` as attributed flow, and the panel's ATTITUDE
row can read ≥1 component reached USED (or we learn, on instruments, that zero do —
which is a finding, not a failure of the milestone).

**M2 — rung 6 (`affected`): does experience change anything?**
Bind raising-session outcomes (session content referencing percepts; trust/coherence
movements) to experiences. Done when `affected` has a defined, computed source —
including a written statement of what would count as attribution, reviewed by one
fleet peer (this is the hardest and most gameable rung; it gets a second witness).

**M3 — panel completion: no instrument left silently U/S.**
Each of the 6 U/S instruments is either (a) bound to a main source, or (b) explicitly
DEFERRED in this doc with a reason and a trigger. Done when the scan output contains
no U/S that this PRD does not name. (Honest deferral is completion; silence is not.)

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
