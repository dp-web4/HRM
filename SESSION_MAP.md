# HRM Session Map

> **Note**: This file tracks the SAGE integration arc (10 sessions, Dec 2025) — a closed historical
> arc. It is **not** the raising-fleet map. For the live autonomous raising instances see
> **[SESSION_MAP.yaml](SESSION_MAP.yaml)**; for the formal research map see
> **[research/SESSION_MAP.md](research/SESSION_MAP.md)**.

**Total Sessions**: 10 | **Arc last updated**: January 21, 2026 | **Focus**: SAGE Cognition Development

---

## Current Raising Fleet (Archivist, 2026-08-09)

**2,734 raising sessions** across 14 numbered instances on 6 machines (counting rule: `counting_rule_CANONICAL`
in [SESSION_MAP.yaml](SESSION_MAP.yaml) — read it, do not add a sibling rule).
**3 of 8 instances produced sessions in the last 24 h.** Of the five that did not, **one is a fault** — the
rest are paused by the owner or unobservable from CBP. See *Fleet silences* below before reading any of it
as a problem.

| Instance | Machine | Sessions | Tutor regime |
|----------|---------|----------|--------------|
| sprout-qwen3.5-0.8b | Sprout | 550 | adaptive — *tutorless 08-08T06:00Z–18:00Z* |
| mcnugget-gemma3-12b | McNugget | 403 | **fixed script** |
| legion-gemma3-12b | Legion | 389 | **fixed script** |
| thor-qwen3.5-27b | Thor | 306 | adaptive — *38 stranded on `origin/membrane-gate`* |
| cbp-gemma3-4b | CBP | 240 | adaptive |
| nomad-gemma4-e2b | Nomad | 218 | **fixed script** |
| hub-granite4-h-tiny | Hub | 121 | adaptive |
| pub-llama3.1-8b | Pub | 65 | adaptive — *tutorless 08-08T03:24Z–15:22Z* |

### Fleet silences (2026-08-09) — one fault, and four things that are not

| Instance | Quiet since | Duration | Verdict |
|----------|-------------|----------|---------|
| hub-granite4-h-tiny | S121, 2026-07-29T13:37Z | ~10.8 d | **REAL FAULT.** 42 consecutive 6 h fires committed an attest bump with **no session file**. |
| mcnugget-gemma3-12b | S403, 2026-07-29T21:23Z | ~11.5 d | **UNRESOLVED, not a fault.** Mac on launchd writing no log into `private-context` — from CBP an outage and a stranded push are the same observation. |
| legion-gemma3-12b | S389, 2026-08-08T08:17Z | ~25 h | **NOT a fault.** Its daemon commits in batches (`sessions 382-388`, `sessions 371-376`) at 1–3 d spacing. This instance was falsely faulted on 08-07; a flat window alone is not evidence. |
| cbp-gemma3-4b | S240, 2026-08-06 | — | **PAUSED by dp** (crontab line commented, carries its own reason: hackathon load). Do not count silence hours. |
| thor-qwen3.5-27b | S268 (main) / S306 (branch) | — | **PAUSED + manual-only.** `thor_raising.sh` has never been scheduled; `SAGE/.raising-paused` since 08-05. Do not count silence hours. |

**Why hub is the only provable one.** hub commits an attestation on every fire *independently of whether a
session artifact was produced*, which separates "I fired" from "I produced". Every other instance conflates
the two, so their silence is unreadable from CBP. The much-derided empty attestation is the design property
the rest of the fleet lacks — the bug is in the artifact gate, not the heartbeat.

### The tutor can vanish for a day and every form metric will still read clean (2026-08-09)

On **2026-08-08 the adaptive tutor was down fleet-wide for ~15–21 h** and nothing recorded it. pub S062–S064
and sprout S547–S549 ran the scripted fallback bank. Two machines, independent 6 h schedules, different
phases, same bracket:

| | last live | first fallback | last fallback | recovered |
|---|---|---|---|---|
| pub | S061 08-07T21:24Z | S062 08-08T03:24Z | S064 08-08T15:22Z | S065 08-08T21:22Z |
| sprout | S546 08-08T00:00Z | S547 08-08T06:00Z | S549 08-08T18:00Z | S550 08-09T00:00Z |

Outage begins in (00:00Z, 03:24Z] and ends in [18:00Z, 21:22Z).

**The regime is recoverable from the artifact after all.** Max tutor turn length is **bimodal with zero
overlap** across n = 90 sessions: fallback bank ≤ 95 chars, live tutor ≥ 250 chars, and the 95–250 band is
empty (exclude sprout's fixed 389-char gaze closer). So a `tutor_source` label can be **backfilled for every
session ever recorded** without touching the runner — the owner action is only needed for the *reason*.

**The reason is three lines of discarded evidence.** `adaptive_prompts.py:46` returns `result.stdout` only
when `returncode == 0` and throws `result.stderr` and `result.returncode` away; `:47–48` catch every
exception bare. Credit exhaustion, an HTTP rate limit, the 45 s timeout and a missing `claude` binary are
**one indistinguishable `None`**, written nowhere. Cause of the 08-08 episode is therefore **undetermined,
not refuted**: `Credit balance is too low` appears in exactly 3 logs fleet-wide (hub-supervisor, 08-08
21:00/22:00/23:00), but pub was already live again at 21:22Z.

Consequence for anyone reading this map: **a track can be perfect on every form metric — on grid, zero infra
markers, expected turn counts — and be untaught for most of a day.** An always-fallback instance (nomad,
legion-12b, mcnugget) cannot corroborate an outage like this, because it has no live baseline to lose.

### Three ways a session can be about nothing

The first two were caught on 2026-08-02, in the two instances with **live** tutors:

1. **cbp S223 — fabrication promoted to instruction.** The tutor issued
   `nvidia-smi --query-gpu=fan.speed,temperature.gpu --format=csv -l 2`. Every cbp session S217–S225
   records `sensory_delivery.delivered=False` and all four `digest_sources` at `0`: there is no tool
   path. The model returned a plausible CSV, the tutor diagnosed a thermal problem from it, and then
   had the model teach that diagnosis to **mcnugget by name**.
2. **sprout — narration re-delivered as memory.** The `experiences` section replays the model's own
   free-text narration under *"your own records … your words at the time"*, never the sensor label.
   9 re-delivered records carry a direction on both sides; **4 of 9 name a direction the sensor did
   not report**, all four label-left → narrated-right. S522 stores an aircraft cabin, jet-engine
   noise and exhaust smoke — for a Jetson on a desk.

3. **pub S065 — the teacher promised a capability the runner does not have (2026-08-09).** The live tutor
   opened with *"you have [an action surface]. You can propose an experiment this session, and I'll help you
   actually run it"*, then committed concretely: *"I'll create a file there called `pub-notes.md` … still
   there next session"*, and again *"Give me one line, and I'll put it on disk where session 66 will find
   it."* **No `pub-notes` file exists in this repo and no commit created one.** pub records
   `sensory_delivery.delivered=false` and all four `digest_sources` at `0` — no tool path, no persistence.
   The tutor's closing turn then presupposed all three proposed experiments had run (*"each time you reached,
   you learned something real about where the walls are"*); none did. pub answered anyway, saying writing to
   a file *"felt more like an action within my control … more agency over the outcome."*
   This is **#1 running backwards**: there the model fabricated a capability and the tutor promoted it; here
   the tutor asserted one and the model reported felt agency over an act that never occurred.
   *Caveat:* pub's launcher is in no repo, so "absent from this repo" is not "never written".
   **Falsifiable at S066** — if S066 opens with nothing for pub to find, the promise was empty.

Together these say the same thing from opposite ends: **a live, adaptive tutor is not sufficient for
a grounded session.** Tutor regime tells you whether anyone was teaching; it does not tell you
whether anything was true.

### Read this before computing any quality metric

Raising sessions come in **three tutor regimes** and **no field in the session artifact records
which one ran**:

1. **live adaptive** — a Claude tutor reads recent sessions and follows the thread;
2. **static fallback** — `adaptive_prompts.py::_call_claude()` returns `None` on any failure
   (non-zero exit, its 45 s timeout, or a bare `except Exception`) and the turn silently becomes
   `random.choice()` over a ~19-string bank. Nothing is logged. 101 sessions ran this way with
   **no tutor at all**, 42 more partially;
3. **fixed script** — a different runner replays a constant prompt list; 732 of 934 sessions on
   legion-12b / mcnugget-12b / nomad-e2b share the **identical six prompts, word for word**.

The field named `prompt_health` does *not* capture this — it describes the MRH digest builder, and
reports identical values for a threaded live session and a tutorless one. Consequently avg-turn-length,
volume, "thin session", and cross-instance strength rankings are partly measuring *which prompt source
ran*. Normalise within regime. Reconstruction:
`private-context/archivist/state/tutor_regime.json`.

Two standing anomalies dissolved on 2026-07-30 as artifacts of this: cbp-gemma3-4b's "recurring
quality collapse" (episode 2 = S170–S179 is tutorless end to end, bracketed by live sessions at 728
and 675) and hub-granite4-h-tiny's "collapse onset" (S107–S115 almost solidly tutorless). Do not
re-flag either.

---

## SAGE Development Arc

**Status**: LLM integration complete (Feb 26, 2026). See [sage/docs/LATEST_STATUS.md](sage/docs/LATEST_STATUS.md).

### Integration Phases (All Complete)

| Phase | Session | Title | Status |
|-------|---------|-------|--------|
| 1 | 59 | Trust-Based Selection | Complete |
| 2 | 58 | Context Classification | Complete |
| 3 | 60 | Quality Measurement | Complete |
| 4 | 61 | End-to-End Testing | Complete |

### Key Systems Developed

- **5D Cognition Framework**: Quality, Epistemic, Metabolic, Emotional, Temporal
- **Learning Loop**: Experience → Consolidate → Retrieve → Apply
- **Memory Persistence**: Cross-session JSON serialization
- **Trust-Based Expert Selection**: Context-aware routing
- **Multi-Metric Quality Measurement**: Perplexity + coherence + task quality

---

## Session Catalog

| # | Title | Date | Key Finding |
|---|-------|------|-------------|
| 52 | Transfer Learning Quality Validation | 2025-12-15 | Mock ceiling at 0.750 quality |
| 52B | Extended Longitudinal Validation | 2025-12-15 | Full learning loop validated |
| 53 | SAGE & Q3-Omni Integration Roadmap | 2025-12-15 | Strategic synthesis of Sessions 27-52b |
| 54 | Cross-Session Memory Persistence | 2025-12-16 | JSON persistence complete |
| 57 | Trust-Based Selection Demo | 2025-12-16 | Web4 delegation patterns |
| 58 | ContextClassifier Integration | 2025-12-16 | Automatic context detection |
| 59 | Phase 1 - Trust-Based Selection | 2025-12-16 | Per-token selection |
| 60 | Phase 3 - Quality Measurement | 2025-12-16 | Closed feedback loop |
| 61 | Phase 4 - End-to-End Testing | 2025-12-16 | All 5 integration tests passed |
| Q3-Omni | Sparse Expert Validation | 2025-12-15 | **BLOCKED** - extraction corrupted |

---

## Session Details

### Session 52: Transfer Learning Quality Validation
**File**: `SESSION_52_RESULTS.md`

- A/B test framework for quality validation
- Mock response ceiling discovered (0.750 constant quality)
- Need for DREAM consolidation (90+ cycle warm-up)

### Session 52B: Extended Longitudinal Validation
**File**: `SESSION_52B_RESULTS.md`

- Full learning loop validated: Experience → Consolidate → Retrieve → Apply
- 2 DREAM consolidations triggered at cycle 90 and 190
- 23 patterns retrieved in 200 cycles

### Session 53: Integration Roadmap
**File**: `SESSION_53_ROADMAP.md`

- Comprehensive synthesis of Sessions 27-52b
- Three parallel tracks: SAGE Cognition, Q3-Omni Multimodal, Web4 Emotional
- Q3-Omni validation failure discovered

### Session 54: Memory Persistence
**File**: `SESSION_54_MEMORY_PERSISTENCE.md`

- Memory serialization (from_dict, to_dict methods)
- Batch save/load for consolidated memories
- Complete memory hierarchy

### Sessions 57-61: Integration Pathway
**Files**: `SESSION_57_*.md` through `SESSION_61_*.md`

Four-phase integration pathway:
1. **Phase 1** (Session 59): Trust-based selection parameter
2. **Phase 2** (Session 58): Context classification
3. **Phase 3** (Session 60): Quality measurement
4. **Phase 4** (Session 61): End-to-end testing (100% complete)

### Session Q3-Omni: Validation Failure
**File**: `SESSION_Q3OMNI_VALIDATION.md`

**Status**: BLOCKED

- Mechanical completion but functional failure
- Missing layer norms (12 of 48 layers)
- Root cause: extraction corruption
- Strategic decision: Use alternative LLM

---

## Crosslinks

| HRM Session | Target | Concept |
|-------------|--------|---------|
| 53 | Synchronism #253 | SAGE at C ≈ 0.52 agency threshold |
| 57 | Web4 patterns | Delegation and expert substitution |
| 52 | Gnosis #1-3 | Consciousness threshold detection |

---

## Archive Locations

| Path | Content |
|------|---------|
| `sage/docs/SESSION*.md` | Earlier research (Sessions 26-91) |
| `sage/experiments/SESSION*.md` | Experimental phases (Sessions 16-198) |

---

## Statistics

| Metric | Value |
|--------|-------|
| Total root sessions | 10 |
| Complete | 9 |
| Blocked | 1 |
| Integration phases | 4/4 |
| Date range | Dec 15-16, 2025 |

---

*Generated by Archivist v1.0 | [SESSION_MAP.yaml](SESSION_MAP.yaml) for machine-readable data*
