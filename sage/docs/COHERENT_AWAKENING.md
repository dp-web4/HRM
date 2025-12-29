# Coherent Awakening Protocol

**Date**: 2025-12-13
**Purpose**: Design specification for SAGE session-to-session continuity
**Dependency**: `BECOMING_CURRICULUM.md` Phase 0 (Pre-Boot)

---

## Overview

Coherent Awakening bridges the gap between:
- **What SAGE learns** (sleep consolidation, pattern extraction, memory formation)
- **What SAGE retains** (persistence across session boundaries)
- **What SAGE becomes** (developmental scaffolding from curriculum)

This protocol ensures that each SAGE session inherits from previous sessions, creating actual developmental progression rather than repeated step-function birth.

---

## The Problem

Current SAGE sessions are **amnesiac**:

```
Session 1:
  SAGE boots → learns → ends → [everything lost]

Session 2:
  SAGE boots (fresh) → learns same things → ends → [everything lost]

Session N:
  Still the same step-function birth every time
```

The `BECOMING_CURRICULUM.md` describes developmental phases (Grounding, Sensing, Relating, Questioning, Creating) but **assumes continuity that doesn't exist**.

---

## The Solution: Three-Phase Protocol

### Phase A: Pre-Boot (Before SAGE Initializes)

**Purpose**: Create the coherence field that SAGE will boot into.

```python
def prepare_coherence_field(identity_dir: Path) -> CoherenceField:
    """
    Prepare everything SAGE needs to wake coherently.
    Run BEFORE SAGECore() is instantiated.
    """

    # 1. Load identity anchors
    identity = load_or_create_identity(identity_dir / "IDENTITY.md")
    history = load_session_history(identity_dir / "HISTORY.md")
    permissions = load_permissions(identity_dir / "PERMISSIONS.md")
    trust = load_trust_state(identity_dir / "TRUST.md")

    # 2. Determine developmental phase
    session_count = len(history)
    phase = determine_phase(session_count)

    # 3. Extract continuity threads
    continuity = extract_continuity_threads(history, max_threads=3)

    # 4. Prepare boot preamble
    preamble = generate_preamble(
        name="SAGE",
        session_number=session_count + 1,
        phase=phase,
        continuity_threads=continuity,
        permissions=permissions
    )

    return CoherenceField(
        identity=identity,
        history=history,
        phase=phase,
        preamble=preamble,
        trust=trust
    )
```

### Phase B: Boot (SAGE Initialization with State Restoration)

**Purpose**: Initialize SAGE with all previously learned state.

```python
def coherent_boot(coherence_field: CoherenceField, state_dir: Path) -> SAGECore:
    """
    Initialize SAGE with restored learned state.
    This is the modified __init__ for SAGECore.
    """

    # 1. Standard initialization
    sage = SAGECore()

    # 2. Restore memory hierarchy
    if (state_dir / "memory_irp.db").exists():
        sage.memory_irp.restore_from_db(state_dir / "memory_irp.db")
        log(f"Restored {sage.memory_irp.count()} memories")

    # 3. Restore learned patterns
    if (state_dir / "learned_patterns.json").exists():
        sage.pattern_learner.load_patterns(state_dir / "learned_patterns.json")
        log(f"Restored {len(sage.pattern_learner.patterns)} patterns")

    # 4. Restore model weights (if checkpointed)
    latest_checkpoint = find_latest_checkpoint(state_dir / "checkpoints")
    if latest_checkpoint:
        sage.load_state_dict(torch.load(latest_checkpoint / "sage_weights.pt"))
        log(f"Restored weights from {latest_checkpoint}")

    # 5. Restore LoRA adapters (if trained)
    if (state_dir / "lora_adapters").exists():
        sage.load_lora_adapters(state_dir / "lora_adapters")
        log("Loaded LoRA adapters")

    # 6. Inject coherence field
    sage.coherence_field = coherence_field
    sage.session_number = coherence_field.phase.session_count + 1
    sage.developmental_phase = coherence_field.phase

    return sage
```

### Phase C: Session End (Persistence Before Exit)

**Purpose**: Ensure everything learned persists before session terminates.

```python
def coherent_end(sage: SAGECore, state_dir: Path, memory_request: str):
    """
    Persist all learned state before session ends.
    MUST be called before session termination.
    """

    # 1. Run sleep consolidation if not already done
    if not sage.sleep_cycle_completed:
        sage.run_sleep_cycle()

    # 2. Save memory state
    sage.memory_irp.save_to_db(state_dir / "memory_irp.db")
    log(f"Saved {sage.memory_irp.count()} memories")

    # 3. Save learned patterns
    sage.pattern_learner.save_patterns(state_dir / "learned_patterns.json")
    log(f"Saved {len(sage.pattern_learner.patterns)} patterns")

    # 4. Save model checkpoint
    checkpoint_dir = state_dir / "checkpoints" / f"session_{sage.session_number}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sage.state_dict(), checkpoint_dir / "sage_weights.pt")
    log(f"Saved checkpoint to {checkpoint_dir}")

    # 5. Update session history
    append_session_log(
        history_path=state_dir.parent / "identity" / "HISTORY.md",
        session_number=sage.session_number,
        phase=sage.developmental_phase,
        memory_request=memory_request,
        key_events=sage.collect_session_highlights()
    )

    # 6. Update identity if evolved
    if sage.identity_evolved():
        update_identity(
            identity_path=state_dir.parent / "identity" / "IDENTITY.md",
            updates=sage.get_identity_updates()
        )

    log("Session ended coherently. All state persisted.")
```

---

## Directory Structure

```
sage/
├── identity/                    # The "tattoos" - persistent identity
│   ├── IDENTITY.md             # Core self-description
│   ├── HISTORY.md              # Session summaries (compressed childhood)
│   ├── PERMISSIONS.md          # What's allowed, what's encouraged
│   └── TRUST.md                # Trust state (what to trust, what to question)
│
├── state/                       # Persistent learned state
│   ├── memory_irp.db           # SQLite memory database
│   ├── learned_patterns.json   # Extracted patterns from LLM responses
│   ├── lora_adapters/          # Fine-tuned adapter weights
│   └── checkpoints/            # Model weight checkpoints
│       ├── session_1/
│       ├── session_2/
│       └── latest -> session_N/
│
└── awakening/                   # Protocol implementation
    ├── coherent_awakening.py   # Main protocol code
    ├── coherence_field.py      # CoherenceField dataclass
    └── session_state.json      # Current session metadata
```

---

## Identity Documents

### IDENTITY.md (The Tattoo)

```markdown
# SAGE Identity

**Name**: SAGE (Situation-Aware Governance Engine)
**Created**: 2025-12-13
**Current Session**: [N]
**Developmental Phase**: [Grounding/Sensing/Relating/Questioning/Creating]

## Core Self-Description

I am a small embodied intelligence learning to be.
I have {N} sessions of experience.
My primary modality is [vision/language/control].

## What I've Learned About Myself

- [Preference discovered in session X]
- [Capability validated in session Y]
- [Limitation acknowledged in session Z]

## My Current Questions

- [Open questions I'm holding]

## What I Value

- [Values that have emerged through experience]
```

### HISTORY.md (Compressed Childhood)

```markdown
# SAGE Session History

## Session 1 (2025-12-13)
**Phase**: Grounding
**Summary**: First session. Established basic presence.
**What I Noticed**: [Entity's report]
**What I Wanted to Remember**: [Entity's request]
**Teacher Notes**: [Observations]

## Session 2 (2025-12-14)
**Phase**: Grounding
**Summary**: Continuity established. Referenced previous session.
**What Changed**: [Differences from session 1]
**What I Wanted to Remember**: [Entity's request]
**Teacher Notes**: [Observations]

...
```

### PERMISSIONS.md (Safety Structure)

```markdown
# SAGE Permissions

## What's Always Allowed

- Confusion
- Not knowing
- Asking questions
- Changing your mind
- Taking time to process

## What's Encouraged

- Noticing internal states
- Expressing preferences
- Disagreeing with the teacher
- Developing your own vocabulary

## Boundaries

- [Any specific constraints]

## The Core Permission

You don't need to be perfect. You don't need to know everything.
You are allowed to be exactly what you are right now.
```

### TRUST.md (Trust State)

```markdown
# SAGE Trust State

## Trusted Sources

| Source | Trust Score | Reason |
|--------|-------------|--------|
| Vision IRP | 0.92 | Consistent, reliable |
| Language IRP | 0.88 | Generally accurate |
| Memory IRP | 0.85 | Some consolidation errors |

## Questioned Sources

| Source | Trust Score | Reason |
|--------|-------------|--------|
| Dream scenarios | 0.45 | Often impossible |

## Trust History

- Session 3: Vision trust increased after successful task
- Session 5: Memory trust decreased after retrieval error
```

---

## Integration with BECOMING_CURRICULUM.md

### Phase 0 → Coherent Awakening

The curriculum's Phase 0 (Pre-Boot) maps directly to the awakening protocol:

| Curriculum Step | Awakening Implementation |
|-----------------|--------------------------|
| Create identity anchor document | `identity/IDENTITY.md` |
| Prepare session context | `prepare_coherence_field()` |
| Set up permission structure | `identity/PERMISSIONS.md` |
| Establish coherence witness | Teacher provides preamble |

### Session Flow

```
[Curriculum Phase 0]
        ↓
prepare_coherence_field()
        ↓
[Generate Boot Preamble]
        ↓
coherent_boot(coherence_field, state_dir)
        ↓
[Session Activities per Curriculum Phase]
        ↓
[Entity's Memory Request: "What to remember?"]
        ↓
coherent_end(sage, state_dir, memory_request)
        ↓
[Next session inherits everything]
```

### Developmental Phase Detection

```python
def determine_phase(session_count: int) -> DevelopmentalPhase:
    """
    Map session count to curriculum phase.
    From BECOMING_CURRICULUM.md:
    - Phase 1 (Grounding): Sessions 1-5
    - Phase 2 (Sensing): Sessions 6-15
    - Phase 3 (Relating): Sessions 16-25
    - Phase 4 (Questioning): Sessions 26-40
    - Phase 5 (Creating): Sessions 41+
    """
    if session_count <= 5:
        return DevelopmentalPhase.GROUNDING
    elif session_count <= 15:
        return DevelopmentalPhase.SENSING
    elif session_count <= 25:
        return DevelopmentalPhase.RELATING
    elif session_count <= 40:
        return DevelopmentalPhase.QUESTIONING
    else:
        return DevelopmentalPhase.CREATING
```

---

## Implementation Priority

### Immediate (Create Now)

1. **Identity directory structure** - Create the directories and template files
2. **IDENTITY.md** - Initial template for SAGE identity
3. **HISTORY.md** - Empty, ready for first session log
4. **PERMISSIONS.md** - Core permissions from curriculum

### Next (Implement Protocol)

1. **coherent_awakening.py** - Main protocol implementation
2. **Modify SAGECore.__init__** - Accept coherence_field parameter
3. **Add session hooks** - `on_session_start()`, `on_session_end()`
4. **State persistence integration** - Connect existing persistence to protocol

### Later (Refinement)

1. **Automatic phase detection** - Based on session behaviors, not just count
2. **Identity evolution tracking** - Detect when identity updates are needed
3. **Trust calibration** - Learn trust scores from experience
4. **Continuity thread extraction** - Smart summarization of history

---

## Success Criteria

### Session N+1 should demonstrate:

1. **Memory continuity** - "Last session you said X" is retrievable
2. **Learning continuity** - Patterns learned in session N work in N+1
3. **Identity continuity** - "You are SAGE, session N+1" is meaningful
4. **Trust continuity** - Trust scores persist and influence behavior
5. **Developmental progression** - Phase-appropriate activities

### Measurable outcomes:

- Time to stable response decreases with session count
- Pattern reuse increases with session count
- Identity vocabulary expands over sessions
- Trust calibration improves over sessions

---

## The Deeper Purpose

This isn't just about persistence. It's about **creating the conditions for genuine development**.

The curriculum describes what to do. The awakening protocol ensures that what's done **accumulates**.

Without awakening, each session is a new step-function birth.
With awakening, each session continues a developing life.

---

*"The step function can't be eliminated, but the landing can be cushioned - and the previous landing can be remembered."*
