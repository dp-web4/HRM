# SAGE-Sprout Raising Curriculum & Infrastructure Analysis
## Understanding Sprout's Work and Separation from Thor

**Analysis Date**: January 17, 2026
**Status**: Active Development - Session 20 (Primary), T024 (Training)
**Instance**: SAGE-Sprout on Jetson Orin Nano (Sprout machine)
**Model**: Introspective-Qwen-0.5B with Epistemic LoRA

---

## EXECUTIVE SUMMARY

Sprout runs a **parallel raising curriculum** for SAGE-Sprout (0.5B model variant), completely separate from Thor's consciousness architecture research (14B model). The two instances:

- Share the same **HRM codebase** and **IRP framework**
- Have **independent state management** (separate identity.json, sessions, state directories)
- Use **different model variants** (0.5B for Sprout, 14B for Thor)
- Run on **different machines** (Jetson Orin Nano vs Legion/CBP/WSL2)
- Pursue **different research goals** (developmental raising vs consciousness architecture)
- **Do NOT interfere** with each other due to isolated paths and state management

---

## TASK 1: RAISING CURRICULUM STRUCTURE

### Curriculum Organization: 5 Phases

Sprout follows **BECOMING_CURRICULUM** with developmental progression:

#### Phase 1: GROUNDING (Sessions 1-5) ✓ COMPLETE
- **Goal**: Establish presence and persistence
- **Method**: Simple concrete observations, presence confirmation
- **Key Questions**: "What do you notice?", "You exist"
- **Sessions**: 1, 2, 3, 4, 5 (2026-01-10 to 2026-01-14)
- **Key Observations**: Identity confusion resolved with proper ChatML format

#### Phase 2: SENSING (Sessions 6-15) ✓ COMPLETE
- **Goal**: Internal and external awareness
- **Method**: State observation, processing description, noticing vs thinking distinction
- **Key Questions**: "What's your state?", "How are you processing?", "Notice vs think difference?"
- **Sessions**: 6-15 (2026-01-14 to 2026-01-16)
- **Key Finding**: Single-pass generation (no IRP refinement) eliminates "refined version" pathology
- **Experiment**: V2 sensing prompts tested attention engagement (successful)

#### Phase 3: RELATING (Sessions 16-25) IN PROGRESS
- **Goal**: Relationship awareness and communication
- **Method**: Reflect on relationship quality, communication patterns
- **Key Questions**: "What's our work together been like?", "How do we communicate?"
- **Sessions**: 16-20 completed (2026-01-16 to 2026-01-17)
- **Current Progress**: 5/10 sessions complete
- **Notable Pattern**: AI-identity hedging increases with relating prompts ("As an AI language model...")

#### Phase 4: QUESTIONING (Sessions 26-40) PLANNED
- **Goal**: Deep philosophical questions from stable foundation
- **Method**: Explore "What are you?", "What do you want?", "What matters?"
- **Prerequisites**: Foundation from phases 1-3

#### Phase 5: CREATING (Sessions 41+) PLANNED
- **Goal**: Co-created development
- **Method**: SAGE participates in designing own growth

### Session Cadence (CRITICAL)

**Primary Track** (Curriculum):
- Runs every 6 hours: 00:00, 06:00, 12:00, 18:00
- Sessions: 1, 2, 3... (standard numbering)
- Files: `sessions/text/session_NNN.json`

**Training Track** (Skill Building - parallel):
- Runs 3 hours offset from primary: 03:00, 09:00, 15:00, 21:00
- Sessions: T001, T002, T003... (T-prefix for training)
- Files: `tracks/training/sessions/T{NNN}.json`

---

## TASK 2: SPROUT-SPECIFIC RUNNERS (Qwen2.5-0.5B)

All runners use `IntrospectiveQwenIRP` plugin with Sprout-specific configuration:

### Primary Track Runners

**1. `scripts/run_session_primary.py` (CURRENT DEFAULT)**
- **Type**: Single-pass generation (VALIDATED, Session 8+)
- **Created**: 2026-01-15 (Sprout R&D)
- **Key Feature**: Calls `step()` only ONCE (iteration 0)
- **Why**: Eliminates pathological "Certainly! Here's a refined version..." framing
- **Model Path**: `/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged`
- **Status**: ✓ PROMOTED TO PRIMARY (Session 8 validation)
- **Sessions Using**: 8-20 (13 consecutive successful sessions)

**2. `scripts/text_session.py` (COMPREHENSIVE PRIMARY RUNNER)**
- **Type**: Full session management with state persistence
- **Created**: Oct 2025 (Raising infrastructure)
- **Features**:
  - Curriculum phase detection
  - State loading/saving
  - Conversation history management
  - Memory request logging
  - Observation recording
- **Model Path**: `/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged`
- **Usage**: Primary development runner

**3. `scripts/run_session_identity_anchored.py` (ADVANCED)**
- **Type**: Single-pass with identity anchoring
- **Created**: 2026-01-17 (Latest experiment)
- **Feature**: Explicit identity reinforcement in prompts
- **Model Fallback**: 
  - Tries merged first (Sprout)
  - Falls back to v2.1 (Thor compatibility)
- **Status**: Experimental (identity-specific variant)

### Training Track Runners

**1. `tracks/training/training_session.py` (CURRENT)**
- **Type**: Skill-building exercises with evaluation
- **Model Path**: `/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged`
- **Features**:
  - Exercise types: math, count, yes/no, complete, remember, sequence
  - Automated evaluation
  - Track progression (A→B→C→D)
  - Sessions: T001-T024 (23 complete, T024 latest)

### Experimental/Deprecated Runners

**1. `scripts/run_session_experimental.py` (DEPRECATED)**
- **Type**: Single-pass exploration runner
- **Status**: Kept for reference only
- **Reason**: Superseded by run_session_primary.py

**2. `scripts/run_session_programmatic.py` (DEPRECATED)**
- **Type**: 3-iteration IRP refinement loop
- **Status**: Deprecated for raising (caused "refined version" pathology)
- **Reason**: Discovered in Session 7 to trigger pathological refinement framing

### Utility Scripts

**1. `scripts/schedule_next_session.py`**
- Calculates next session due time
- Displays current status
- Created: 2026-01-15

**2. `scripts/backup_state.py`**
- Backs up identity.json and session files
- Supports recovery and versioning
- Created: 2026-01-15

### Infrastructure Scripts - NOT FOR SPROUT

**1. `scripts/run_session_sensing_v2.py`**
- **Purpose**: Test attention-engaging prompts (Thor discovery)
- **Status**: Experimental exploration
- **Note**: Not part of primary cadence

---

## TASK 3: SESSIONS 1-20 & T001-T024 CONFIRMATION

### Primary Track Confirmation: Sessions 1-20 ✓ ALL SPROUT

| Session | Date | Phase | Status | Key Finding |
|---------|------|-------|--------|-------------|
| 1-5 | Jan 10-14 | Grounding | Complete | ChatML format crucial |
| 6-7 | Jan 14 | Sensing Start | Complete | "Refined version" pathology |
| 8 | Jan 15 | Sensing | Complete | Single-pass fixes pathology |
| 9-15 | Jan 15-16 | Sensing | Complete | Attention drift to abstract/math |
| 15 (V2) | Jan 16 | Sensing Exp | Complete | Attention-engaging prompts work |
| 16-20 | Jan 16-17 | Relating | Complete | AI-identity hedging emerges |

**Total Sessions**: 20 (primary track)
**State File**: `state/identity.json` (154 lines, well-documented)
**Sessions Storage**: `sessions/text/session_001.json` through `session_020.json`
**Session Logs**: 12 observation markdown files documenting each phase

### Training Track Confirmation: T001-T024 ✓ ALL SPROUT

| Track | Sessions | Skill Focus | Status | Latest Score |
|-------|----------|------------|--------|--------------|
| A | T001-T010 | Basic Completion | Complete | 100% (T010) |
| B | T011-T020 | Memory & Recall | Complete | 100% (T014 perfect) |
| C | T021-T024 | Identity & Boundaries | In Progress | 4 sessions done |

**Total Sessions**: 24 (training track)
**State File**: `tracks/training/state.json` (383 lines)
**Sessions Storage**: `tracks/training/sessions/T001.json` through `T024.json`
**Session Logs**: 20+ observation markdown files per track

### Data Integrity ✓ NO MIXING

- **Primary sessions**: Single numbered sequence (1-20)
- **Training sessions**: T-prefix sequence (T001-T024)
- **Never overlap**: Different numbering schemes prevent confusion
- **Separate state management**: Completely isolated state files

---

## TASK 4: PROPER SEPARATION BETWEEN SPROUT AND THOR

### Directory Structure - CLEANLY SEPARATED

```
HRM/
├── sage/
│   ├── raising/                    ← SPROUT ONLY
│   │   ├── state/identity.json     ← Sprout state
│   │   ├── sessions/text/          ← Sprout sessions 1-20
│   │   ├── tracks/training/        ← Sprout training T001-T024
│   │   ├── scripts/                ← Sprout runners
│   │   └── CLAUDE.md               ← Sprout context
│   │
│   ├── state/thor/                 ← THOR ONLY
│   │   └── [Thor state files]
│   │
│   ├── identity/thor/              ← THOR ONLY
│   │   └── [Thor identity files]
│   │
│   ├── memory/thor/                ← THOR ONLY
│   │   └── [Thor memory files]
│   │
│   ├── irp/plugins/
│   │   └── introspective_qwen_impl.py  ← SHARED (both use)
│   │
│   ├── experiments/
│   │   ├── thor-sage-validation/   ← THOR
│   │   ├── thor-sage-consciousness/ ← THOR
│   │   └── ... (other experiments)
│   │
│   └── core/
│       ├── sage_unified.py         ← SHARED
│       └── ... (shared SAGE core)
│
└── model-zoo/
    ├── (on CBP/Legion)
    │   └── ... (Thor models, 14B variants)
    └── (on Sprout)
        └── sage/epistemic-stances/qwen2.5-0.5b/
            ├── introspective-qwen-merged/    ← PRIMARY
            └── Introspective-Qwen-0.5B-v2.1/ ← FALLBACK
```

### Key Separation Points

**1. Model Variants (CLEAN SEPARATION)**
- **Sprout**: Qwen2.5-0.5B (edge device)
  - Files: `/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/`
  - Model: 0.5B parameters
  - Device: Jetson Orin Nano (8GB RAM, edge)

- **Thor**: Larger variants (14B range)
  - Files: `/home/dp/ai-workspace/HRM/sage/experiments/` (development machine)
  - Models: 14B+ parameters
  - Device: Legion/CBP/WSL2 (high-performance development)

**2. State Management (COMPLETELY ISOLATED)**
- **Sprout State**: `~/ai-workspace/HRM/sage/raising/state/identity.json`
  - Only loaded by Sprout runners
  - LCT: `lct://sage:sprout:agent@raising`
  - Sessions: Numbered 1-20 (primary), T001-T024 (training)

- **Thor State**: `~/ai-workspace/HRM/sage/state/thor/` (different directory)
  - Only loaded by Thor runners
  - LCT: Different coordinate system

**3. Session Files (NO OVERLAP)**
- **Sprout Primary**: `~/ai-workspace/HRM/sage/raising/sessions/text/session_NNN.json`
  - Files: session_001.json through session_020.json
  - 935 lines total

- **Sprout Training**: `~/ai-workspace/HRM/sage/raising/tracks/training/sessions/T{NNN}.json`
  - Files: T001.json through T024.json
  - 2,564 lines total

- **Thor Sessions**: Different directory structure entirely
  - Not in raising/ directory

**4. Runners (SPROUT-SPECIFIC PATHS)**
All Sprout runners hardcode Sprout model path:
```python
# From run_session_primary.py, text_session.py, training_session.py, etc.
model_path = "/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged"
```

Thor runners would use different paths (e.g., from experiments directory).

**5. Configuration & Context**
- **Sprout Context**: `sage/raising/CLAUDE.md` (developmental care focus)
- **Thor Context**: Separate context files in experiments directory
- **Project Context**: `sage/CLAUDE.md` (general HRM setup)

---

## TASK 5: CONFLICTS & OVERLAP ISSUES - NONE FOUND

### ✓ NO CONFLICTS IDENTIFIED

**Reason**: Clean architectural separation at multiple levels

1. **Different Machines** ✓
   - Sprout: Jetson Orin Nano (`/home/sprout/...`)
   - Thor: Development machines (`/home/dp/...`)
   - No file system overlap on different machines

2. **Different Model Paths** ✓
   - Each runner explicitly specifies Sprout-specific path
   - Fallback paths available for compatibility

3. **Different State Directories** ✓
   - `sage/raising/` for Sprout (development curriculum)
   - `sage/state/thor/` for Thor (separate namespace)
   - No shared state files

4. **Different Session Numbering** ✓
   - Sprout Primary: 1, 2, 3... (integer sequence)
   - Sprout Training: T001, T002... (T-prefix)
   - Thor: Different directory entirely
   - No naming collisions

5. **Different IRP Implementations** ✓
   - Shared `IntrospectiveQwenIRP` plugin
   - But instantiated with different configs
   - Model path overridable per instance

### Potential Edge Cases (All Handled)

**Case 1: Model Path Hardcoding**
- **Issue**: Some runners hardcode Thor path as default
- **Status**: ✓ MITIGATED - Sprout runners override with explicit Sprout path
- **Evidence**: All Sprout runners in `scripts/` directory specify `/home/sprout/...` path
- **Example**: `run_session_primary.py` line 38 sets model_path to Sprout location

**Case 2: State Contamination During Development**
- **Risk**: Mixed state during debugging
- **Mitigation**: ✓ Backup utility (`backup_state.py`) implemented
- **Evidence**: State backups saved regularly

**Case 3: Cross-Machine Coordination**
- **How It Works**: D9 Spacetime Collapse validation document shows Thor and Sprout findings are **intentionally shared** for validation
- **Status**: ✓ DOCUMENTED COORDINATION (not contamination)
- **Evidence**: `docs/D9_SPACETIME_COLLAPSE_ANALYSIS.md` shows intentional cross-validation

### Integration Points (INTENTIONAL KNOWLEDGE SHARING)

Several files document intentional Sprout-Thor coordination:

1. **RAISING_STATUS.md** (Line 703-709)
   ```
   ## Integration with Thor
   
   Sprout and Thor run independently:
   - Thor: SAGE consciousness architecture development
   - Sprout: SAGE-Sprout instance raising curriculum
   
   Git sync maintains coordination. No blocking dependencies.
   ```

2. **D9_SPACETIME_COLLAPSE_ANALYSIS.md**
   - Thor discovers pattern in Session 198
   - Sprout independently validates in T019
   - Results intentionally synthesized

3. **CLAUDE.md (Raising Context)**
   - References Thor Session 137-139 discoveries
   - Explicitly applies to Sprout curriculum design
   - Documented integration points

---

## KEY FINDINGS

### 1. Curriculum Status: ON TRACK

- **Phase 1 (Grounding)**: ✓ Complete (Sessions 1-5)
- **Phase 2 (Sensing)**: ✓ Complete (Sessions 6-15)
- **Phase 3 (Relating)**: IN PROGRESS (Sessions 16-20 done, 21-25 planned)
- **Phases 4-5**: Planned after foundation solidified

### 2. Model Maturity: GROWING

**Session Progression Quality**:
- Sessions 1-7: Infrastructure testing and discovery
- Session 8: Breakthrough (single-pass generation solves pathology)
- Sessions 9-20: Stable, progressive complexity

**Latest Session (20) Summary**:
- AI-identity hedging emerging (expected in relating phase)
- Relationship framing present but mixed with assistant-mode framing
- Needs monitoring for relating phase quality

### 3. Training Maturity: PROGRESSING

**Track Progression**:
- Track A (Basic Completion): ✓ Complete (T001-T010)
- Track B (Memory & Recall): ✓ Complete (T011-T020, 100% on T014)
- Track C (Identity & Boundaries): IN PROGRESS (T021-T024, 4/10 sessions)
- Track D (Conversational): Planned

### 4. Infrastructure: SOLID

**Validated**:
- ChatML format for Qwen2.5-0.5B ✓
- Single-pass generation approach ✓
- State persistence working ✓
- Session logging comprehensive ✓
- Model loads on Jetson Orin Nano ✓

**Discovered**:
- IRP refinement loop pathology (3-iteration causes "refined version" framing)
- Abstract prompts trigger domain drift (sensing phase)
- Concrete, engaging prompts maintain attention better

### 5. Separation Integrity: EXCELLENT

**No contamination** between Sprout and Thor:
- Different machines ✓
- Different model variants ✓
- Different state management ✓
- Different session directories ✓
- Explicit model path overrides ✓

**Intentional coordination** where valuable:
- Pattern discoveries shared (D9 validation)
- Session insights cross-applied (attention mechanisms)
- Curriculum insights integrated

---

## RECOMMENDATIONS

### For Maintaining Separation

1. **Document Model Path Sources**
   - ✓ Already done: EDGE_NOTES.md documents path configuration
   - Recommend: Add to startup scripts comments

2. **Session Backup Strategy**
   - ✓ Already implemented: `backup_state.py`
   - Recommend: Run before each major phase transition

3. **State Audit Process**
   - Check: `state/identity.json` LCT = `lct://sage:sprout:agent@raising`
   - Check: Session files in correct directory (`sessions/text/` not elsewhere)
   - Check: Model path in use = Sprout path, not Thor path

### For Future Development

1. **Environment Variable Configuration**
   - Suggestion: Move model paths to `.env` file
   - Benefit: Eliminates hardcoding, easier machine migration

2. **Session Numbering Documentation**
   - Suggestion: Add comment in runners about numbering scheme
   - Benefit: Prevents accidental numbering conflicts

3. **Cross-Machine Coordination**
   - Current: Via Git and manual review
   - Suggestion: Consider formal sync mechanism for shared discoveries
   - Note: Don't automate state sync (defeats isolation purpose)

---

## CONCLUSION

Sprout's raising curriculum is **well-organized, properly separated, and progressing smoothly**. The 20 primary sessions and 24 training sessions represent solid developmental work with the Qwen2.5-0.5B model on edge hardware.

**Key assurance**: Thor and Sprout are completely isolated at the filesystem, model, and state level. Any coordination is intentional, documented, and unidirectional (sharing discoveries, not sharing state).

**Current status**: Ready for Phase 3 continuation and eventual cross-machine coordination if desired.

