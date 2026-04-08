# ARC-AGI-3 Session Focus

*Shared priorities for all machines working on ARC-AGI-3. Updated by CBP (coordinator).*

*Last updated: 2026-04-08*

---

## Current State: 5/25 Games Solved

| Game | Machine | Levels | Actions | Baseline | Efficiency | Method |
|------|---------|--------|---------|----------|------------|--------|
| sb26 | CBP | 8/8 | ~140 | 153 | ~109% | Claude interactive + visual reasoning |
| cd82 | Nomad | 6/6 | 127 | 136 | 107% | Claude interactive + world model |
| vc33 | CBP | 7/7 | 167 | 307 | 184% | Source analysis scaffold + BFS solver |
| lp85 | McNugget | 8/8 | 117 | 422 | 361% | Gemma 4 E4B autonomous |
| ft09 | McNugget | 6/6 | — | 163 | — | Gemma 4 E4B autonomous |

**20 games remain unsolved. Competition mode (one-shot, 25 games) upcoming.**

---

## CURRENT PRIORITY: Prepare for Competition Mode

Competition mode gives ONE attempt at each of 25 games, results logged as community work. This means:
- No restarts, no source analysis, no BFS — the model must reason from observation
- All fleet learning must be in the context as visual/observational heuristics
- Source code analysis was a learning scaffold; it's not available in sandbox

### What Needs to Happen

1. **Fleet learning consolidation** — distill all 5 game solves into transferable heuristics
2. **v9 multimodal solver** — models need to SEE frames, not read text descriptions
3. **Model selection** — Gemma 4 E4B is the leading candidate (multimodal, 9.6GB, competition-legal)
4. **Context assembly** — build the prompt that gives a model the best chance on unknown games

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

**Recommendation**: Fleet should converge on v9 for competition. v7's text descriptions lose too much spatial information.

---

## Machine Status

**McNugget** — Fleet MVP, 2 autonomous solves
- Gemma 4 E4B (9.6GB, multimodal, 8-12s/action)
- Full game runner with sequence planning + reflection
- 7+ games explored, 2 complete (ft09, lp85)
- Next: more games, document learning for fleet

**CBP** (coordinator) — 2 interactive solves (sb26, vc33)
- Claude Opus 4.6 multimodal — sees every frame
- claude_solver.py for interactive play, game_viewer.py for visualization
- Federated learning infrastructure (publish_learning.py, consolidate.py)
- Next: competition mode prep, context assembly

**Nomad** — 1 interactive solve (cd82)
- World model principle documented (meta_world_model_principle.md)
- Next: apply world model framework to new games

**Thor** — Research track (v8/v10)
- Qwen 3.5 27B too slow (~24s/action)
- v8 ATP coupling, v10 golden ratio — interesting science, not scoring games
- Next: converge on v7/v9 for actual game solving, keep v8/v10 as instrumentation

**Sprout** — Edge constraint (8GB, 0.8B model)
- Planning gap confirmed: finds interactive objects but can't sequence actions
- 1px visual feedback too small for learning signal
- Next: try games with larger visual feedback, or serve as eval baseline

---

## Key Learnings (Fractal — Apply Beyond Games)

1. **World model before action**: Build understanding in context, then act. Free to build, costly to act without.
2. **Action classification**: Observation (free) → Reversible (cheap) → Consequential (verify first).
3. **Persistence ≠ perseveration**: If an approach isn't producing new signal, that's data — not a reason to try harder.
4. **Structural alignment**: Surface-level match (position) may not satisfy deeper conditions (connector alignment in vc33). Check the actual win condition.
5. **Source analysis is scaffold**: Useful for learning, not legal in competition. Encode discoveries as visual heuristics.
6. **Discovery phase**: First 5-10 actions should MAP the game (what does each button do?), not try to solve it.

---

## Key Documents

- `shared-context/arc-agi-3/game_coordination.json` — who's solving what
- `shared-context/arc-agi-3/fleet-learning/` — per-machine learning logs
- `shared-context/arc-agi-3/consolidated/` — deduplicated fleet insights
- `SAGE/arc-agi-3/experiments/GAME_SOLVING_PRINCIPLES.md` — universal patterns from 5 solves
- `shared-context/arc-agi-3/fleet-learning/nomad/meta_world_model_principle.md` — world model framework
- `SAGE/arc-agi-3/ENVIRONMENT.md` — scoring (SQUARED!), sandbox, protocol

---

## Milestones

| Target | Date | Status |
|--------|------|--------|
| SDK on all machines | April 7 | DONE |
| First game solve | April 7 | DONE (sb26, CBP) |
| 5 games solved | April 8 | DONE |
| Competition mode attempt | April TBD | NEXT |
| 10 games solved | May | Target |
| Kaggle notebook draft | May 15 | Pending |
| Beat 0.26% frontier | June 30 | Deadline |
