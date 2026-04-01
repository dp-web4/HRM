# SAGE TODO

*Global task list across active workstreams. Updated by CBP (coordinator).*

*Last updated: 2026-04-01*

---

## ARC-AGI-3 (Deadline: June 30)

### Adapter Layer (April 7-14)
- [x] Install SDK on Thor, Sprout, McNugget — DONE (all 3 verified 2026-03-31)
- [x] Benchmark: action iteration speed per machine — DONE (SDK: 3-6k steps/sec, NOT bottleneck)
- [ ] End-to-end test: push frame → consciousness loop → action dispatch
- [x] Wire GridVisionIRP into consciousness loop — DONE (config-gated in `_gather_observations()`)
- [x] GameActionEffector — DONE (`sage/irp/plugins/game_action_effector.py`)
- [x] GridVisionIRP skeleton — DONE (`sage/irp/plugins/grid_vision_irp.py`)
- [x] GridVisionIRP interface spec — DONE (private-context/plans/)
- [x] Environment spec (ENVIRONMENT.md) — DONE
- [x] StochasticGoose analysis — DONE

### Perception Layer (Andy, April)
- [ ] GridCartridge paired lattice format (even: embedded state, odd: SAGE reasoning)
- [ ] Connected component analysis → structured GridObservation
- [ ] Frame embedding (CLIP or custom CNN — TBD, memory budget matters on 32GB)
- [ ] Cross-level cartridge assembly
- [x] SDK installed, frames pulling — DONE (Andy confirmed 2026-03-30)

### Scoring (ACTIVE — replace "Integration" milestone)
- [x] First SAGE-played game — DONE 2026-03-31 (7 games scoring, 50 levels in best run)
- [x] Experience DB — DONE (all 25 games surveyed, action effectiveness mapped)
- [x] Smart clicker (no LLM) — DONE (explore→exploit, scores on pure-click games)
- [x] Spatial reasoning layer — DONE (`arc_spatial.py`, object tracking + interactive detection)
- [x] Action-effect model — DONE (learns what each action does from observation)
- [x] Hybrid architecture design — DONE (`HYBRID_ARCHITECTURE.md` — Driver + Navigator + Membot)
- [ ] Implement hybrid runner (`sage_hybrid_runner.py`)
- [ ] First movement+click level — 11+ games currently at 0
- [ ] Membot cartridge integration end-to-end (read on start, write on level completion)
- [ ] Serialization format for GridObservation (msgpack vs JSON vs shared mem) — Andy handoff
- [ ] GridCartridgeIRP — memory search wrapper (transport-agnostic: MCP or direct)

### Fast Loop (May)
- [ ] Strip consciousness loop to game speed (~100ms/cycle)
- [ ] Action efficiency tracker (information-to-action ratio)
- [ ] Goal inference from environment state changes
- [ ] Level-context memory (ephemeral within environment)

### Submission (May-June)
- [ ] Kaggle notebook draft (submittable, even if score = 0) — target May 15
- [ ] Model selection for competition (fits in 32GB with SAGE + perception)
- [ ] Beat 0.26% frontier — June 30

### Architecture Decisions (Pending)
- [ ] Grid representation: one-hot 16-channel vs structured features vs both
- [ ] Action selection: pure consciousness loop vs hybrid with spatial CNN
- [ ] Session-level vs step-level loop
- [ ] Model for reasoning: size/speed tradeoff per game phase
- [ ] Memory between levels: dream consolidation vs hash dedup vs cartridge

---

## Membot Integration

- [ ] Read pattern-0-v2-spec.md from membot upstream
- [ ] Build segment builder (SNARC DB → .seg.npz)
- [ ] Federated assembly with merged Pattern 0
- [ ] MemoryCartridgeIRP transport-agnostic backend (MCP for orchestration, direct for embedded)
- [ ] PR to membot upstream
- [x] RFC #5 submitted and approved — DONE
- [x] Dual-write experiment (SNARC + membot) — DONE
- [x] Semantic reach test (7/7) — DONE
- [x] Divergent tail analysis (30.5%) — DONE
- [x] REST bridge — DONE

---

## 4-gov Presentation (April 24)

- [ ] Live demo plugin — build during the talk (pieces already exist)
- [x] 37+ blocks across 4 parts — DONE
- [x] Start Here page — DONE
- [x] Orchestrator comparison (5 tools + matrix) — DONE
- [x] Nova review incorporated — DONE

---

## SAGE Consciousness Loop

- [ ] Effector system (#7) — ~1,660 LOC, unblocks PolicyGate + TTS/motor
- [ ] PolicyGate Phase 2 (#8) — depends on effector system
- [ ] LLM as Temporal Sensor (#6) — SensorHub unification
- [ ] MemoryHub / RLLF gathering (#9) — SQLite backend
- [ ] Multimodal Phase 5: cross-modal trust confirmation
- [x] Trust posture system — DONE (all 7 changes)
- [x] Multimodal Phases 0-3 + defensive trust — DONE
- [x] ModelAdapter + model configs — DONE
- [x] Tool use integration (#10) — DONE

---

## Fleet & Infrastructure

- [x] Create shared-context repo for Andy collaboration — DONE (`dp-web4/sage-arc-collab`, 2026-04-01)
- [x] Grant Andy collaborator access to SAGE — DONE (agro23 invited to SAGE + sage-arc-collab)
- [ ] Update fleet track registry for ARC-AGI-3 pivot
- [ ] Nomad schema v1→v2 migration
- [x] Fleet track registry (SQLite) — DONE
- [x] Supervisor v2.0 — DONE
- [x] Session governance R7 split (primer/focus) — DONE
- [x] Raising deep dive + 7 fixes — DONE

---

## External

- [ ] ARIA proposal follow-up (if needed)
- [ ] The Byte article — pitch submitted, awaiting response
- [x] Hermes Agent forked and analyzed — DONE

---

## Key Documents

| Document | Location |
|----------|----------|
| ARC-AGI-3 strategy | `arc-agi-3/README.md` |
| ARC-AGI-3 environment | `arc-agi-3/ENVIRONMENT.md` |
| StochasticGoose analysis | `arc-agi-3/STOCHASTICGOOSE_ANALYSIS.md` |
| ARC-AGI-3 session focus | `arc-agi-3/SESSION_FOCUS.md` |
| GridVisionIRP interface spec | `private-context/plans/arc-agi-3-gridvision-irp-interface-spec.md` |
| Collaboration proposal (Andy) | `private-context/plans/arc-agi-3-collaboration-proposal-andy.md` |
| Entry plan | `private-context/plans/arc-agi-3-entry-2026-03-29.md` |
| Fleet priorities pivot | `private-context/plans/fleet-priorities-arc-agi-3-pivot-2026-03-30.md` |
| Parallel plans tracker | `private-context/plans/PARALLEL_PLANS_TRACKER.md` |
