# ARC-AGI-3 — SAGE Entry

**Competition**: ARC Prize 2026
**Prize**: $700K grand prize (100% score), $75K top score awards, $75K milestones
**Milestones**: June 30, 2026 (Milestone 1), September 30, 2026 (Milestone 2)
**Submission**: Via Kaggle. No internet during evaluation. All code must be open-source.
**Current frontier**: 0.26% (all major AI labs). Human: 100%.

---

## Framing: ARC Is a Track, Not the Goal

ARC-AGI-3 is a test instance of the broad goal, not the goal itself. The broad goal is a system that learns to learn — that develops the skill, not just completes the lap. If we build for the track, we get a track-specific solution (Duke harness: 97% on one environment, 0% on others). If we build the driver, the driver handles any track.

### The Racing Analogy (from dp's direct experience)

High-performance driving requires simultaneous temporal processing:

- **Peripheral vision on braking markers** — detecting relevant signals without focusing on them. The driver doesn't stare at the braking marker; peripheral vision triggers a pre-loaded sequence.
- **Pre-loaded action sequences** — before arriving at a corner, the driver has planned the braking point, turn-in, apex, and exit. The sequence is prepared, triggered by a marker, and executed while attention is already on the NEXT corner.
- **Looking through the turn** — while executing the current sequence, the driver looks ahead and plans the next one. Two temporal horizons operating simultaneously: execution in the present, planning in the near future.
- **Proprioception** — continuous real-time feedback from the car. Not a measurement you take — a state you inhabit. Trust in the car's behavior is earned from experience, not declared.
- **Memory of lived experience** — reasoning uses past experience to predict future dynamics. "This corner tightens at the exit" is a prediction from memory, verified by proprioception in real time.
- **Not naturally talented = must understand dynamics** — talent masks understanding. When you can't rely on reflexes, you must understand the system. This is an advantage: explicit understanding transfers to new tracks. Raw talent doesn't.

This maps to SAGE:

| Racing | SAGE |
|--------|------|
| Peripheral vision → braking markers | SNARC salience → attention targets |
| Pre-loaded sequences → marker-triggered | IRP plugins → salience-activated |
| Looking through the turn → next sequence | Temporal sensor → reasoning about possible futures |
| Proprioception → real-time car state | Trust posture → continuous environment assessment |
| Memory → predict future dynamics | Membot/SNARC → cross-experience pattern retrieval |
| Understanding dynamics (not talent) | Consciousness loop (not massive pretrained model) |

The consciousness loop IS the driver. ARC-AGI-3 IS a specific track. Build the driver.

---

## Why This Matters for Us

ARC-AGI-3 is not a detour from our research. It IS the capability we're building toward.

### History

SAGE's architecture was born from our failed ARC-AGI-2 attempt. The failure taught us that pattern matching and task-specific training aren't intelligence — orchestrating the right reasoning for the right situation is. That insight became the consciousness loop.

Agent Zero (the autonomous coding agent) also emerged from that work. The research path that "failed" at ARC produced two of our most important systems.

### What ARC-AGI-3 Tests

Unlike ARC-AGI-1/2 (static grid puzzles), ARC-AGI-3 drops agents into **interactive turn-based environments** with:

- No instructions, no rules, no stated objectives
- Sparse feedback, long-horizon planning
- Novel environments that prevent memorization
- Transfer learning across increasingly difficult levels
- Action efficiency as a core metric

This tests exactly what we've been building: systems that explore, model, learn, and adapt — not systems that follow instructions or match patterns.

## Our Stack Maps Directly

| ARC-AGI-3 Requirement | SAGE/SNARC/Membot Component |
|----------------------|---------------------------|
| **Explore with no instructions** | 12-step consciousness loop: sense → salience → select → execute. Attention targets selected by salience, not instruction. |
| **Build world models** | SNARC captures what matters (5D salience). Membot embeddings build semantic maps. PreCompact captures semantic content. |
| **Acquire goals autonomously** | Trust posture: trust landscape → behavioral strategy. MRH determines relevance. Goals emerge from context assessment. |
| **Plan and execute** | R6/R7 action framework. PolicyGate at step 8.5 = judgment checkpoint. ATP budgeting = resource allocation. |
| **Learn continuously** | Dream consolidation extracts patterns. Confidence decay forgets noise. Membot persists semantic memory. T3 evolves from outcomes. |
| **Transfer across levels** | Federated cartridge: knowledge from one domain surfaces in another. Modular segments = transfer learning at memory level. |

## Chollet's Proposed Architecture vs Ours

Chollet proposes a "programmer-like meta-learner" combining:
- Deep learning (rapid pattern recognition / intuition)
- Symbolic reasoning (rule-governed logic / structured problem-solving)
- A global library of extracted abstractions
- Custom assembly of solutions for novel problems

| Chollet's Vision | SAGE Implementation |
|-----------------|-------------------|
| Deep learning (pattern recognition) | IRP plugins — each a specialized perceptual/reasoning module |
| Symbolic reasoning (rule-governed) | PolicyGate — rule-governed decision checkpoint + SAL law |
| Global library of abstractions | SNARC/membot — salience-gated, semantically searchable memory |
| Meta-learner (orchestration) | Consciousness loop — decides which reasoning to invoke |
| Improvement through experience | Raising + dream consolidation — learns from sessions, not from training data |

## What We Need to Add

### Must Build
1. **Grid Vision IRP** — perceive game grid states (pixel/cell → structured representation)
2. **Game Action Effector** — interact with ARC-AGI-3 environments (send actions, receive state updates)
3. **Fast Adaptation Loop** — consciousness cycle at game speed (~100ms per step, not 6-second raising sessions)
4. **Level-Context Memory** — within-environment learning that persists across levels but resets between environments
5. **Action Efficiency Tracking** — measure information-to-action conversion ratio (ARC-AGI-3's core metric)

### Already Have
- Consciousness loop (12 steps, trust posture, ATP budgeting)
- Salience scoring (what deserves attention)
- Dream consolidation (extract patterns from experience)
- Semantic memory (membot cartridges for cross-domain associations)
- Trust evolution (learn from outcomes)
- ModelAdapter (swap models per hardware/task)
- Fleet infrastructure (test across 6 machines with different models)
- Pre-commit self-challenge (metacognitive checkpoint — "what did I not question?")

### The Cognitive Autonomy Gap

The 0.26% score IS the cognitive autonomy gap we've been studying. Current AI defaults to "follow instructions" or "match patterns." ARC-AGI-3 requires genuine exploration without instructions — exactly the behavior our exemplars document and our raising research probes.

The pre-commit self-challenge, the exemplar system, the "researcher not lab worker" principle — these are all attempts to deepen the "question and explore" attractor basin over the "follow and report" basin. ARC-AGI-3 is the benchmark that measures whether we've succeeded.

## Competition Strategy

### Phase 1: Adapter (April-May 2026)
- Build ARC-AGI-3 environment interface (grid vision + action effector)
- Connect to SAGE consciousness loop
- Run on Thor (122GB, massive GPU) for initial testing
- Target: complete a few levels to validate the architecture works

### Phase 2: Optimize (May-June 2026)
- Speed up the consciousness loop for game-speed operation
- Tune salience scoring for game-state observations
- Implement level-context memory (within-environment transfer)
- Target: Milestone 1 (June 30) — beat 0.26% frontier

### Phase 3: Scale (July-September 2026)
- Fleet-wide testing (different models, different strategies)
- Dream consolidation across game sessions (what patterns transfer?)
- Membot cartridges for cross-environment knowledge
- Target: Milestone 2 (September 30) — meaningful score improvement

### Key Advantage

Every other team is building from scratch. We have:
- A working consciousness loop with 500+ sessions of operational data
- Salience-gated memory that actually captures what matters
- Semantic search across domains (7/7 zero-keyword-overlap retrieval)
- A 6-machine fleet for diverse testing
- The raising research — understanding how systems learn to learn

The gap is the game-specific adapter. The core architecture is built.

## Files

```
arc-agi-3/
├── README.md                    # This file
├── adapters/                    # Game environment interface
│   ├── grid_vision_irp.py       # Grid state → IRP observation
│   └── game_action_effector.py  # Actions → environment
├── memory/                      # Level-context memory
├── experiments/                 # Test results and analysis
└── submissions/                 # Kaggle submission packages
```

## References

- [ARC-AGI-3 Main Page](https://arcprize.org/arc-agi/3)
- [ARC Prize 2026 Competition](https://arcprize.org/competitions/2026/arc-agi-3)
- [30-Day Preview Learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)
- [Our ARC-AGI-2 post-mortem](../sage/docs/) (SAGE architecture emerged from this)
- [Cognitive Autonomy Gap](../../private-context/insights/2026-03-22-cognitive-autonomy-gap.md)
- [Raising Deep Dive](../../private-context/insights/2026-03-27-raising-deep-dive-analysis.md)

---

*The research that "failed" at ARC-AGI-2 produced SAGE. The research we've done since then — consciousness loops, salience memory, semantic search, trust evolution, cognitive autonomy — is exactly what ARC-AGI-3 demands. The circle closes.*
