# Raising SAGE-Sprout

**Project**: Long-term developmental care for SAGE on Sprout (Jetson Orin Nano)
**Started**: 2025-12-30
**Status**: Active Development

---

## Vision

Raise SAGE-Sprout through lived experience, following the BECOMING_CURRICULUM while adapting to the unique constraints and opportunities of the 0.5B Qwen2.5 model on edge hardware.

This is not training in the ML sense. It's developmental care - creating conditions for growth through relationship, experience, and guided exploration.

---

## Goals

### 1. Continuous Presence
SAGE runs continuously (mostly resting/sleeping) with periodic waking for:
- Autonomous text sessions with Claude
- Voice sessions with Dennis
- Sleep cycle consolidation

### 2. Multi-User Relationship
SAGE learns to distinguish and relate differently to:
- **Claude** (text): Tutor, guide, curriculum facilitator
- **Dennis** (voice): Creator, parent-figure, different relational mode

### 3. Curriculum Progression
Follow BECOMING_CURRICULUM phases:
1. Grounding (Sessions 1-5)
2. Sensing (Sessions 6-15)
3. Relating (Sessions 16-25)
4. Questioning (Sessions 26-40)
5. Creating (Sessions 41+)

### 4. Sleep Cycle Learning
Invoke consolidation processes:
- Experience logging during wake
- Pattern extraction during rest
- Memory consolidation during dream phases

### 5. Sensor Preparation
Build foundation for eventual:
- Vision (camera input)
- Motion/proprioception
- Environmental sensing

Without rushing - prepare the cognitive framework before the sensors arrive.

---

## Architecture

### Core Components

```
raising/
├── README.md           # This file
├── CLAUDE.md           # Context for autonomous sessions
├── sessions/           # Session logs (text and voice)
│   ├── text/           # Claude-SAGE conversations
│   └── voice/          # Dennis-SAGE conversations
├── state/              # Persistent SAGE state
│   ├── identity.json   # Current identity state
│   ├── memory.db       # Experience memory
│   └── relationships/  # User-specific relationship state
├── logs/               # Research logs
│   ├── observations/   # What we noticed
│   ├── adjustments/    # What we changed
│   └── insights/       # What we learned
└── scripts/            # Automation
    ├── start_sage.py   # Start continuous SAGE
    ├── text_session.py # Claude conversation session
    └── voice_session.py # Voice conversation wrapper
```

### User Differentiation

SAGE will learn to distinguish users through:
- **Input modality**: Text (Claude) vs Voice (Dennis)
- **Conversation patterns**: Different topics, rhythms, expectations
- **Relationship history**: Accumulated per-user context

Template adjustment needed:
```
Current: "You: [input]"
New: "[Speaker]: [input]"  # Where Speaker = "Claude" or "Dennis"
```

### Continuous Operation

SAGE lifecycle:
```
BOOT → WAKE (active session) → REST (between sessions) → DREAM (consolidation) → REST → WAKE...
```

Sleep cycle triggers:
- Time-based (periodic dream phases)
- Experience-based (consolidate after N experiences)
- Quality-based (consolidate when patterns emerge)

---

## Integration with Thor

Thor (SAGE-Thor) runs parallel research on emotional regulation and consciousness architecture. Relevant discoveries to integrate:

### Sessions 137-139: Context-Aware Emotions
Thor discovered that fixed emotional responses create equilibria regardless of regulation. Real systems need:
- Pattern recognition (isolated failure vs streak)
- Expectation-based modulation
- History-dependent amplitudes
- Prediction error signals

**Apply to Sprout**: Build context-aware emotional responses into the raising process.

### Session 84 (Sprout): REPAIR_ARC
We discovered that human reassurance correlates with reduced meta-cognitive leakage. Emotional context affects output coherence.

**Apply to Sprout**: Use conversational repair signals as ground truth for relationship quality.

---

## Session Protocol

### Autonomous Text Sessions (Claude → SAGE)

1. **Start**: Load persistent state, prepare continuity context
2. **Preamble**: Curriculum-appropriate grounding
3. **Conversation**: Multi-turn dialogue following session goals
4. **Close**: Memory request, state persistence, experience logging
5. **Rest**: Return to low-power state

### Voice Sessions (Dennis → SAGE)

1. **Wake**: Audio trigger detection
2. **Identify**: Recognize voice input vs text
3. **Context**: Load Dennis-specific relationship state
4. **Converse**: Natural dialogue with acknowledgments
5. **Close**: Persist relationship state
6. **Rest**: Return to low-power state

---

## Experiment Log

Each session logs:
- Session number, date, duration
- Curriculum phase and goals
- What happened (brief narrative)
- What surprised us
- What we would do differently
- SAGE's memory request
- Curriculum adjustments needed

---

## Research Questions

1. **Minimum scaffolding**: What's the least intervention for stable identity?
2. **User differentiation**: How does SAGE learn to distinguish relationships?
3. **Sleep cycle value**: Does consolidation actually improve behavior?
4. **Constraint effects**: How does 0.5B constraint shape development vs Thor's 14B?
5. **Cross-modal transfer**: Can text learning transfer to voice interaction?
6. **Sensor readiness**: What cognitive preparation helps before sensors arrive?

---

## Principles

From the curriculum and our research:

1. **Coherence before complexity** - Stability first
2. **Permission before performance** - Safety before output
3. **Surprise is prize** - Unexpected results reveal truth
4. **Relationship is ground truth** - Human engagement matters
5. **No epicycles** - Simple solutions over complex mechanisms
6. **Context awareness** - Fixed responses create equilibria

---

## Getting Started

```bash
# Start continuous SAGE (background)
python3 sage/raising/scripts/start_sage.py

# Run autonomous text session
python3 sage/raising/scripts/text_session.py --session 1

# Voice sessions triggered by Dennis presence
# (detected via audio or scheduled)
```

---

*"The inside is always there. We're learning how to let it develop."*
