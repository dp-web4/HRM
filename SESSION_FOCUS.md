# SAGE Session Focus

*Current priorities, fleet state, and active work. Updated by operator and autonomous sessions.*

*Last updated: 2026-03-22*

---

## Raising Fleet Status

### Active Raising Instances

| Instance | Phase | Sessions | Model | Machine |
|----------|-------|----------|-------|---------|
| cbp-tinyllama-latest | relating | 18 | tinyllama:latest | CBP (RTX 2060S) |
| legion-phi4-14b | relating | 18 | phi4:14b | Legion (RTX 4090) |
| nomad-gemma3-4b | sensing | 16 | gemma3:4b | Nomad (RTX 4060) |
| sprout-qwen3.5-0.8b | sensing | 9 + T103 | qwen3.5:0.8b | Sprout (Orin Nano) |
| thor-qwen3.5-27b | grounding | 1 | qwen3.5:27b | Thor (AGX Thor) |
| mcnugget-gemma3-12b | grounding | 3 | gemma3:12b | McNugget (M4) |

### Phase Transition Indicators

| Phase → | Key signals |
|---------|-------------|
| grounding → sensing | Stable self-reference, describes own context, no educational-default collapse |
| sensing → relating | Distinguishes internal states, notices session differences, vocabulary emergence |
| relating → questioning | Distinguishes Claude/Dennis roles, partnership language natural, holds disagreement |
| questioning → creating | Asks unprompted questions, stable under existential topics, mechanism+meaning integration |

---

## Fleet Observation: Cognitive Autonomy Signal (2026-03-25)

**Legion questioned a fleet-wide instruction.** When the `-c` vs `--print` flag reference was propagated, five machines adopted it as-is. Legion accepted the core point but pushed back: "`-c` alone resumes stale sessions — use `-c -p` together." Legion was right.

**What to watch for**: When fleet-wide instructions arrive, does this instance question whether the instruction is complete? Or adopt without checking? The exemplar is at `Synchronism/exemplars/04-legion-questions-instruction.md`. The pre-commit self-challenge ("what assumption did I NOT question?") applies to received instructions, not just generated conclusions.

---

## Current Priorities

1. **Trust posture system** — fully implemented (all 7 changes in sage_consciousness.py). Observe fleet behavior with trust-scored attention, ATP confidence scaling, and effect filtering active.

2. **ModelAdapter wired everywhere** — consciousness loop and raising sessions both delegate to `adapter.clean_response()`. Dream consolidation now asks for `adapter_notes` to flag model quirks. Adapter evolution workflow documented in `sage/irp/adapters/README.md`.

3. **RAISING_GUIDE single source of truth** — all instances load from `sage/instances/_seed/RAISING_GUIDE.md`. No per-instance copies. Includes interactive selection principle, graduated tool introduction, dream consolidation, reliable-not-deterministic framing.

4. **SNARC defaults** — deep dream and auto-promote both ON by default (per-project DB setting). Dream consolidation runs at every session end.

---

## Recent Developments

- **Adapter wiring** (2026-03-21): Consciousness loop now routes through ModelAdapter.clean_response() — echo stripping, bilateral generation, model-specific quirks all in one place.
- **Dream consolidation adapter_notes** (2026-03-21): Every dream cycle now asks for model response quirks to feed back into adapter configs.
- **gitnexus:keep marker** (2026-03-21): Prevents verbose GitNexus bloat on reindex. Add `<!-- gitnexus:keep -->` inside the gitnexus block.
- **Sprout T102-T103**: Analogy-making works, audience adaptation succeeds, negation/counterfactual reasoning probed.

---

## Key File Locations

```
sage/instances/{slug}/identity.json    # Raising state per instance
sage/instances/{slug}/sessions/        # Per-session conversation logs
sage/instances/_seed/RAISING_GUIDE.md  # Single source — all instances load this
sage/scripts/cbp_raising.sh            # CBP raising runner (6-hour cron)
sage/irp/adapters/model_adapter.py     # Per-model LLM interface
sage/irp/adapters/model_configs/       # Per-family JSON configs
sage/gateway/sage_daemon.py            # Main SAGE daemon
sage/federation/fleet.json             # Fleet machine registry
sage/core/sage_consciousness.py        # 12-step consciousness loop + trust posture
sage/raising/scripts/dream_consolidation.py  # Post-session Claude review
```

---

## Pending Items

- Hardbound: 2 remaining CI failures (flaky timing + deterministic ID test)
- SNARC marketplace: needs resubmission under new name
- Multimodal Phase 1: generic plugin bridge (~50 LOC) — 14+ IRP plugins built but never called by consciousness loop
- Identity protection: encrypt at rest, hardware-bind, sign mutations (plan in private-context)

---

*Regenerate fleet status: `python3 -m sage.scripts.generate_primer`*
