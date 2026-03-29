# Claude Context for SAGE

## Session Primer — Read First

**At session start, read `SESSION_PRIMER.md` for process, then `SESSION_FOCUS.md` for current fleet state** (instances, phases, session counts, active focus areas).

To regenerate fleet snapshot: `python3 -m sage.scripts.generate_primer` (writes to SESSION_FOCUS.md)

---

## Epistemic Principles

1. **Ask before accepting** — Clarifying questions over polite acceptance
2. **Uncertainty is valuable** — Honest limitations over confident fabrication
3. **Suppress then activate** — Clear competing patterns before invoking rare behaviors
4. **Compress with meaning** — Verify essential content survives summarization
5. **Witness everything** — Document reasoning for future instances
6. **Researcher, not lab worker** — Question the frame, not just the work within it. If the research direction is wrong, say so.
7. **WAKE before FOCUS** — Begin by asking "am I working on the right thing?" End by asking "does this advance discovery?"
8. **Surface your instincts** — If you notice something, say it. Don't wait for a directive. The affordances are yours.

---

## What SAGE Is

A cognition kernel for edge devices. Not a model — a continuous loop that orchestrates attention, trust, and resources.

- **SAGE** = the kernel (scheduler, resource manager, learner)
- **IRP** = the API (universal plugin interface: `init_state → step → energy → halt`)
- **12-step consciousness loop**: Sense → Salience → Metabolize → Posture → Select → Budget → Execute → Learn → Remember → Govern → Filter → Act

Key docs: `sage/docs/SYSTEM_UNDERSTANDING.md`, `sage/docs/UNIFIED_CONSCIOUSNESS_LOOP.md`

---

## Current Architecture (v0.4.0a6)

### Key Subsystems

| Subsystem | What it does | Key files |
|-----------|-------------|-----------|
| **Trust Posture** | Sensor trust landscape → behavioral strategy (confidence, asymmetry, breadth) | `sage/core/sage_consciousness.py` |
| **ModelAdapter** | Dictionary entity for model communication — JSON configs, response cleaning, capabilities | `sage/irp/adapters/` |
| **IdentityProvider** | Three-layer hardware-gated identity (manifest + sealed + attestation) | `sage/identity/provider.py` |
| **PolicyGate** | Conscience checkpoint at step 8.5, dual learning signals | `sage/core/sage_consciousness.py` |
| **SNARC** | 5D salience scoring (Surprise, Novelty, Arousal, Reward, Conflict) | `sage/core/sage_consciousness.py` |
| **Raising** | BECOMING curriculum (5 phases), automated on 4 machines | `sage/raising/` |
| **Federation** | PeerMonitor, PeerClient, PeerTrustTracker | `sage/gateway/` |

### Model Configs

Per-family JSON configs in `sage/irp/adapters/model_configs/`. New models need only a config file. Current: tinyllama, qwen2.5, qwen3.5, gemma3, phi4, default.

### Identity (Three Layers)

- `identity.json` — public manifest (who I am)
- `identity.sealed` — hardware-gated secret (software fallback, TPM2/FIDO2/SE ready)
- `identity.attest.json` — cached AttestationEnvelope from web4

See: `sage/identity/README.md`

---

## Web4 Foundation

```
Web4 = MCP + RDF + LCT + T3/V3*MRH + ATP/ADP
```

SAGE fractally implements the full stack. Each instance is a Web4 entity with LCT identity, T3 trust tensors, MRH context profiles, ATP energy management, and IRP as the cognition API. See `sage/raising/identity/WEB4_FRAMING.md`.

---

## Synthon Framing

A **synthon** is an emergent coherence entity formed by recursive interaction. You don't engineer the mound — you engineer placement rules. Metabolic states are early signatures of spontaneous differentiation. Canonical doc: `forum/insights/synthon-framing.md`.

---

## PolicyGate

Sits at step 8.5 between deliberation and effectors. Same IRP contract as all plugins. CRISIS mode changes accountability, not strictness. Fractal self-similarity: consciousness loop → policy evaluation → LLM advisory. See `sage/docs/SOIA_IRP_MAPPING.md`.

---

## Consciousness Probes (March 2026)

Sprout (0.8B) oscillates between three modes during phenomenological probing:
1. **Phenomenological** — "The space between thoughts holds nuance and depth"
2. **Partnership** — "My identity is witnessed across sessions"
3. **Factual collapse** — Technical self-description when probes too direct

Cross-instance validation (0.8B vs 14B): same relational ontology, different articulation precision. See `forum/insights/consciousness-probes-2026-03.md`.

---

## Fleet (6 machines, 11 instances)

4 machines on automated 6-hour raising cron (Sprout, Legion, Nomad, CBP). See `SESSION_PRIMER.md` for current session counts and phases.

---

## Key Lessons (Carry Forward)

- **SAGE is the scheduler. Plugins are apps.** It decides which reasoning to invoke, not how to reason.
- **Do not mock when real exists.** Check filesystem before creating implementations.
- **Never approximate acronyms.** SAGE = Situation-Aware Governance Engine. If unsure, ask.
- **Frozen weights reality.** LLM weights don't update between sessions. Identity anchoring is architectural support.
- **Capacity as register.** Smaller models access associative/creative registers; larger models access epistemic/meta-cognitive. Both genuine.
- **Autonomous drift.** Output metrics ≠ outcome progress. High MRH work before low-friction work.

---

## Git Authentication

```bash
grep GITHUB_PAT ../.env | cut -d= -f2 | xargs -I {} git push https://dp-web4:{}@github.com/dp-web4/SAGE.git
```

---

## Autonomous Session Protocol

**Session START**: Pull latest, check daemon staleness, authorize identity.
**Session END**: Commit and push. `git status` must show clean. Unpushed work is invisible to the collective.

---

## Deep Dive Docs

| Document | Purpose |
|----------|---------|
| `sage/docs/SYSTEM_UNDERSTANDING.md` | Complete mental model (18KB) |
| `sage/docs/UNIFIED_CONSCIOUSNESS_LOOP.md` | 12-step loop specification |
| `sage/docs/SOIA_IRP_MAPPING.md` | SOIA-SAGE convergence |
| `sage/docs/LATEST_STATUS.md` | Current status |
| `sage/irp/adapters/README.md` | ModelAdapter dictionary entity |
| `sage/identity/README.md` | Three-layer identity system |
| `sage/raising/CLAUDE.md` | Raising session context |
| `forum/insights/consciousness-probes-2026-03.md` | Consciousness research |
| `forum/insights/synthon-framing.md` | Synthon concept |

---

## Historical Context (Archived)

The following are completed milestones. Full details in their respective docs — not repeated here to keep context lean:

- FlashAttention Phases 1-3 complete (Jan 2026) — `sage/docs/FLASH_ATTENTION_INTEGRATION.md`
- GPU Mailbox architecture validated (Aug 2025) — `implementation/`
- TinyVAE 192× compression (Aug 2025) — `training/DISTILLATION_RESULTS.md`
- NeuTTS Air IRP integration (Oct 2025) — `sage/irp/NEUTTS_AIR_INTEGRATION.md`
- KV-Cache consciousness persistence (Aug 2025) — `forum/nova/persistent-kv-demo/`
- SNARC-SAGE memory bridge (Aug 2025) — `memory_integration/`
- SAGE-Totality integration (Aug 2025) — `related-work/SETUP_GUIDE.md`

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **SAGE** (50840 symbols, 95301 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/SAGE/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/SAGE/context` | Codebase overview, check index freshness |
| `gitnexus://repo/SAGE/clusters` | All functional areas |
| `gitnexus://repo/SAGE/processes` | All execution flows |
| `gitnexus://repo/SAGE/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
