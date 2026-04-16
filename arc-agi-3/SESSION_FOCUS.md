# ARC-AGI-3 Session Focus

*Shared priorities for all machines working on ARC-AGI-3. Updated by CBP (coordinator).*

*Last updated: 2026-04-15*

---

## Current State: 92.82% — 21/25 Games, 173/183 Levels

**Scorecard c4e6442e** — 21/25 environments, 173/183 levels, 5,496 actions

| Game | Machine | Levels | Actions | Baseline | Efficiency | Method |
|------|---------|--------|---------|----------|------------|--------|
| sb26 | CBP | 8/8 | ~140 | 153 | ~109% | Claude interactive + visual reasoning |
| cd82 | Nomad | 6/6 | 127 | 136 | 107% | Claude interactive + world model |
| vc33 | CBP | 7/7 | 167 | 307 | 184% | Source analysis scaffold + BFS solver |
| lp85 | McNugget | 8/8 | 117 | 422 | 361% | Gemma 4 E4B autonomous |
| ft09 | McNugget | 6/6 | — | 163 | — | Gemma 4 E4B autonomous |
| sc25 | CBP | 7/7 | — | — | — | — |
| tn36 | CBP | 7/7 | — | — | — | — |
| tr87 | CBP | 7/7 | — | — | — | — |
| tu93 | CBP | 7/7 | — | — | — | — |
| su15 | CBP | 7/7 | — | — | — | — |
| s5i5 | McNugget | 6/6 | — | — | — | — |
| sp80 | Thor | 7/7 | — | — | — | — |
| ar25 | Thor | 7/7 | — | — | — | — |
| cn04 | Thor | 7/7 | — | — | — | — |
| ls20 | Sprout | 7/7 | — | — | — | — |
| bp35 | Sprout | 7/7 | — | — | — | partial (L6+ structurally blocked) |
| m0r0 | Sprout | 7/7 | — | — | — | — |
| r11l | Nomad | 6/6 | — | 167 | — | 99.75% (near-perfect) |
| g50t | Sprout | 7/7 | — | — | — | — |
| wa30 | McNugget | 9/9 | — | — | — | — |
| ka59 | Legion | 7/7 | — | — | — | — |

**20 games at 100%+. r11l at 99.75%. 4 structurally blocked levels remain.**

---

## CURRENT PRIORITY: Phase 2 Research + Remaining 4 Games

### Structurally Blocked Levels

| Game | Level | Status |
|------|-------|--------|
| re86 | L8 | Blocked |
| dc22 | L6 | Blocked |
| lf52 | L7, L10 | Blocked (eq.win() bypass works in NORMAL mode only, COMPETITION mode blocks it) |
| bp35 | L6+ | Blocked |

### Phase 2 Research

- **Phase 1 paper sealed**: `paper/ARC-SAGE-AGI-84-9.md` (filename kept for link stability)
- **Phase 2 paper started**: `paper/ARC-SAGE-PHASE2.md`
- **Key Phase 2 finding**: Gemma 4 E2B scored 0% across 20 harness variations (CBP). 7-vendor cross-model survey confirms fixation is universal in small VLMs.
- **gemma4-good-submission repo**: Kaggle hackathon (May 18 deadline)

### Thursday Fleet Wake-Up Plan

1. **Legion**: E4B capacity test (first ARC-AGI-3 work on Legion)
2. **Fresh-perspective passes**: All 4 structurally blocked games get new eyes

---

## Solver Versions

```
v5 → v6 → v7 (fleet standard, membot integration)
              ├→ v8 (Thor: ATP coupling research)
              │   └→ v10 (Thor: golden ratio validation)
              └→ v9 (multimodal, requires vision-capable model)
```

- **v7**: Fleet standard. `persistent_solver.py` uses v7 primary, v6 fallback. Text descriptions of grids.
- **v9**: Multimodal branch. Sends actual PNG frames to the model. Requires gemma4:e4b or equivalent.
- **v8/v10**: Thor research variants (ATP coupling, coherence measurement). Not game-solving improvements — instrumentation.

---

## Machine Status

**CBP** (coordinator) — 7 solves (sb26, sc25, tn36, vc33, tr87, tu93, su15)
- Claude Opus 4.6 multimodal — sees every frame
- claude_solver.py for interactive play, game_viewer.py for visualization
- Phase 2 research: cross-model VLM fixation survey
- Next: Thursday fresh-perspective passes on blocked games

**McNugget** — 3 solves (ft09, lp85, s5i5)
- Gemma 4 E4B (9.6GB, multimodal, 8-12s/action)
- Full game runner with sequence planning + reflection
- Next: Kaggle hackathon (gemma4-good-submission)

**Thor** — 3 solves (sp80, ar25, cn04)
- v8/v10 research instrumentation
- Next: available for blocked-game fresh passes

**Sprout** — 3 solves (ls20, bp35, m0r0) + g50t
- Edge constraint (8GB, 0.8B model)
- bp35 L6+ structurally blocked

**Nomad** — 1 solve (cd82) + r11l (99.75%)
- World model principle documented (meta_world_model_principle.md)
- r11l nearly perfect — 1 level short

**Legion** — 1 solve (ka59)
- RTX 4090 — E4B capacity test planned Thursday

---

## Key Learnings (Fractal — Apply Beyond Games)

1. **World model before action**: Build understanding in context, then act. Free to build, costly to act without.
2. **Action classification**: Observation (free) → Reversible (cheap) → Consequential (verify first).
3. **Persistence ≠ perseveration**: If an approach isn't producing new signal, that's data — not a reason to try harder.
4. **Structural alignment**: Surface-level match (position) may not satisfy deeper conditions (connector alignment in vc33). Check the actual win condition.
5. **Source analysis is scaffold**: Useful for learning, not legal in competition. Encode discoveries as visual heuristics.
6. **Discovery phase**: First 5-10 actions should MAP the game (what does each button do?), not try to solve it.
7. **VLM fixation is universal**: Small VLMs (all vendors) fixate on initial strategies. Not a model-specific problem — structural limitation of the approach.

---

## Key Documents

- `shared-context/arc-agi-3/game_coordination.json` — who's solving what
- `shared-context/arc-agi-3/fleet-learning/` — per-machine learning logs
- `shared-context/arc-agi-3/consolidated/` — deduplicated fleet insights
- `SAGE/arc-agi-3/experiments/GAME_SOLVING_PRINCIPLES.md` — universal patterns
- `SAGE/arc-agi-3/paper/ARC-SAGE-AGI-84-9.md` — Phase 1 paper (sealed)
- `SAGE/arc-agi-3/paper/ARC-SAGE-PHASE2.md` — Phase 2 paper (active)
- `shared-context/arc-agi-3/fleet-learning/nomad/meta_world_model_principle.md` — world model framework
- `SAGE/arc-agi-3/ENVIRONMENT.md` — scoring (SQUARED!), sandbox, protocol

---

## Milestones

| Target | Date | Status |
|--------|------|--------|
| SDK on all machines | April 7 | DONE |
| First game solve | April 7 | DONE (sb26, CBP) |
| 5 games solved | April 8 | DONE |
| 21 games solved | April 12 | DONE (92.82%) |
| Phase 1 paper sealed | April 12 | DONE |
| Phase 2 paper started | April 13 | DONE |
| Legion E4B capacity test | April 17 | NEXT (Thursday) |
| Fresh passes on 4 blocked games | April 17 | NEXT (Thursday) |
| Kaggle notebook draft | May 15 | Pending |
| Kaggle hackathon deadline | May 18 | Deadline |
| Beat 0.26% frontier | June 30 | Deadline |
