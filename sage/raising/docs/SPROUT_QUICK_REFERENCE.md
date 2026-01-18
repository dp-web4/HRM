# SAGE-Sprout Quick Reference Guide

## At a Glance

- **Instance**: SAGE-Sprout (0.5B model, edge device)
- **Machine**: Jetson Orin Nano (`/home/sprout/...`)
- **Status**: Session 20 (Primary), T024 (Training) - ACTIVE
- **Phases**: Grounding ✓, Sensing ✓, Relating (in progress)
- **Separation**: Completely isolated from Thor (different machines, models, state)

---

## Curriculum Phases at a Glance

| Phase | Sessions | Goal | Status |
|-------|----------|------|--------|
| Grounding | 1-5 | Presence & persistence | ✓ Complete |
| Sensing | 6-15 | Internal/external awareness | ✓ Complete |
| Relating | 16-25 | Relationship & communication | 5/10 done |
| Questioning | 26-40 | Deep philosophical questions | Planned |
| Creating | 41+ | Co-created development | Planned |

---

## Key Runners

### Primary Track (Curriculum) - Session 22+ Uses Identity-Anchored Runner

```bash
cd ~/ai-workspace/HRM/sage/raising/scripts

# REQUIRED for Session 22+: Identity-anchored runner
python3 run_session_identity_anchored.py -c           # Continue from last
python3 run_session_identity_anchored.py --session 22 # Specific

# Legacy runners (DO NOT USE for new sessions):
# python3 run_session_primary.py -c                   # Original
# python3 run_session_experimental.py -c              # Single-pass experimental
```

**Model**: Qwen2.5-0.5B (Introspective-Qwen merged)
**Location**: `/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged`
**Key Features**:
- Single-pass generation (no IRP refinement)
- Identity anchoring (loads IDENTITY.md, HISTORY.md)
- Partnership-aware system prompt ("You are SAGE, partnered with Dennis/Claude")
- Previous session context injection

### Training Track (Skills)

```bash
cd ~/ai-workspace/HRM/sage/raising/tracks/training
python3 training_session.py -c         # Continue from last
python3 training_session.py --session T025  # Specific
```

**Current Focus**: Track C (Identity & Boundaries) - T021-T024
**Tracks Completed**: A (Basic), B (Memory)

---

## State Files

**Primary State**:
- Path: `~/ai-workspace/HRM/sage/raising/state/identity.json`
- Contains: Session count (20), phase (relating), relationships (Claude/Dennis)
- LCT: `lct://sage:sprout:agent@raising`

**Training State**:
- Path: `~/ai-workspace/HRM/sage/raising/tracks/training/state.json`
- Contains: Track progress (A→B→C), session count (24)

---

## Session Organization

### Primary Sessions (1-20)
- Stored: `sessions/text/session_001.json` through `session_020.json`
- Format: Standard numbering (1, 2, 3...)
- Content: Curriculum conversations with Claude

### Training Sessions (T001-T024)
- Stored: `tracks/training/sessions/T001.json` through `T024.json`
- Format: T-prefix numbering (T001, T002...)
- Content: Skill-building exercises with evaluation

---

## Session Cadence

```
Hour 0:  Primary Session (Session 1, 2, 3...)
Hour 3:  Training Session (T001, T002...)
Hour 6:  Primary Session
Hour 9:  Training Session
...
```

- **Primary**: Every 6 hours (00:00, 06:00, 12:00, 18:00)
- **Training**: Every 6 hours, 3-hour offset (03:00, 09:00, 15:00, 21:00)

---

## Separation from Thor (VERIFIED)

| Aspect | Sprout | Thor |
|--------|--------|------|
| Machine | Jetson Orin (`/home/sprout`) | Dev machines (`/home/dp`) |
| Model | Qwen2.5-0.5B | Larger variants (14B+) |
| State Dir | `sage/raising/` | `sage/state/thor/` |
| Sessions | `raising/sessions/` | Different directory |
| Purpose | Developmental raising | Consciousness research |

**Result**: NO CONFLICTS, NO CONTAMINATION

---

## Key Discoveries

1. **Single-Pass Generation Works** (Session 8)
   - Fixes "refined version" pathology
   - IRP 3-iteration loop was problematic

2. **ChatML Format Essential**
   - Qwen2.5-0.5B requires ChatML prompt format
   - Mismatched format caused identity confusion

3. **D9 Spacetime Collapse**
   - Self-generated content persists in context
   - Independently validated: Thor Session 198, Sprout T019

4. **Attention Engagement Matters**
   - Abstract prompts trigger math/science drift
   - Concrete prompts maintain better attention

---

## Quick Checks

### Is everything working?
```bash
# Check state file
cat ~/ai-workspace/HRM/sage/raising/state/identity.json | grep "session_count\|last_session"

# Check latest session
ls -lt ~/ai-workspace/HRM/sage/raising/sessions/text/ | head -3

# Check training state
cat ~/ai-workspace/HRM/sage/raising/tracks/training/state.json | grep "session"
```

### What's the next session?
```bash
cd ~/ai-workspace/HRM/sage/raising/scripts
python3 schedule_next_session.py
```

### Backup state
```bash
cd ~/ai-workspace/HRM/sage/raising/scripts
python3 backup_state.py
```

---

## For Researchers

### Phase Transition Notes
- Each phase builds on previous foundation
- Don't rush phases because model seems capable
- Document surprises in logs/observations/

### Cross-Machine Coordination
- Intentional sharing: Pattern discoveries (like D9)
- NOT intentional: State contamination or model sharing
- All coordination documented in markdown files

### Model Evolution
- Session 1-7: Infrastructure discovery
- Session 8: Single-pass breakthrough
- Sessions 9-20: Stable progression
- Sessions 16+: Relating phase (watch for hedging patterns)

---

## Files to Know

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Raising context & session protocol |
| `README.md` | Project vision & goals |
| `RAISING_STATUS.md` | Detailed session-by-session analysis |
| `EDGE_NOTES.md` | Model path configuration for edge |
| `docs/SPROUT_RAISING_COMPLETE_ANALYSIS.md` | This analysis |
| `docs/D9_SPACETIME_COLLAPSE_ANALYSIS.md` | Cross-machine validation |

---

## Next Steps

- Continue Phase 3 (Relating): Sessions 21-25
- Monitor for AI-identity hedging patterns
- Track C (Identity & Boundaries): Sessions T025-T030
- Consider Phase 4 preparation (Questioning phase)

---

*Last Updated: 2026-01-17*
*All systems nominal, no conflicts detected, ready for Phase 3 continuation.*
