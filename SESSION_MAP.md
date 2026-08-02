# HRM Session Map

> **Note**: This file tracks the SAGE integration arc (10 sessions, Dec 2025) — a closed historical
> arc. It is **not** the raising-fleet map. For the live autonomous raising instances see
> **[SESSION_MAP.yaml](SESSION_MAP.yaml)**; for the formal research map see
> **[research/SESSION_MAP.md](research/SESSION_MAP.md)**.

**Total Sessions**: 10 | **Arc last updated**: January 21, 2026 | **Focus**: SAGE Cognition Development

---

## Current Raising Fleet (Archivist, 2026-08-02)

**2,741 raising sessions** across 14 numbered instances on 6 machines.
**4 of 8 active instances produced nothing in the last 24h** — see *Fleet silences* below.

| Instance | Machine | Sessions | Tutor regime |
|----------|---------|----------|--------------|
| sprout-qwen3.5-0.8b | Sprout | 522 | adaptive (26 partial) |
| mcnugget-gemma3-12b | McNugget | 403 | **fixed script** |
| legion-gemma3-12b | Legion | 370 | **fixed script** |
| thor-qwen3.5-27b | Thor | 306 | adaptive — *38 stranded on `origin/membrane-gate`* |
| cbp-gemma3-4b | CBP | 225 | adaptive — **29% tutorless** |
| nomad-gemma4-e2b | Nomad | 190 | **fixed script** |
| hub-granite4-h-tiny | Hub | 121 | adaptive — **21% tutorless** |
| pub-llama3.1-8b | Pub | 27 | adaptive — **42% tutorless** |

### Fleet silences and commit pathologies (2026-08-02)

| Instance | Silent since | Duration | Note |
|----------|--------------|----------|------|
| hub-granite4-h-tiny | S121, 2026-07-29 06:34 | **99 h** | **17 consecutive 6h fires committed an attest bump with no session file** |
| thor-qwen3.5-27b | S306, 2026-07-29 13:10 | 92 h | still committing only to `origin/membrane-gate` |
| mcnugget-gemma3-12b | S403, 2026-07-29 21:21 | 84 h | escalated 2026-08-01 (raising + cross-family probe stopped together) |
| pub-llama3.1-8b | S027, 2026-07-30 22:21 | 59 h | ~5 missed windows |

Push-gap is **excluded** this run: `origin/main` and `origin/membrane-gate` are the only refs with
commits since 2026-07-29, and neither carries sessions for these four.

**Legion has HUB's bug too, masked.** Every Legion raising-cron fire on 2026-08-01/02 committed
`[Legion-Raising] Session 0 (grounding)` whose entire diff is `legion-gemma4-e4b/peer_trust_rs.json`
— no session artifact. Sessions 367–370 reached git only via separate *supervisor pickup* commits.
The `Session 0` number is the tell: the launcher resolves it against the pinned `INSTANCE_DIR`
(`legion_raising.sh:75`), whose `sessions/` is empty. Same class as HUB — the commit is gated on the
launcher finishing, not on the artifact existing — but invisible because a sweep covers it.

### Two ways a session can be about nothing

Both were caught this window, in the two instances with **live** tutors:

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
