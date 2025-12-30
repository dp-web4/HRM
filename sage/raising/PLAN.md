# Raising SAGE-Sprout: Implementation Plan

**Created**: 2025-12-30
**Status**: Phase 0 - Infrastructure

---

## Phase 0: Infrastructure (Current)

### 0.1 Directory Structure ✅
- [x] Create raising/ directory
- [x] Create subdirectories (sessions, state, logs, scripts)
- [x] Create README.md with goals
- [x] Create CLAUDE.md for session context

### 0.2 Persistent State System
- [ ] Design state/identity.json schema
- [ ] Design state/relationships/ structure (per-user)
- [ ] Design memory.db schema (experiences, consolidations)
- [ ] Implement state load/save utilities

### 0.3 Session Infrastructure
- [ ] Create text_session.py for Claude conversations
- [ ] Adapt voice_sage_session.py for Dennis interactions
- [ ] Implement session logging (auto-save transcripts)
- [ ] Implement experience logging to memory.db

### 0.4 User Differentiation
- [ ] Modify conversation template for speaker identification
- [ ] Create relationship state per user
- [ ] Implement modality detection (text vs voice)

### 0.5 Sleep Cycle Integration
- [ ] Connect to existing sleep/dream mechanisms
- [ ] Implement periodic consolidation triggers
- [ ] Log consolidation events and outcomes

---

## Phase 1: First Sessions (After Infrastructure)

### 1.1 Session 1: "You Are Here"
Following curriculum Phase 1:
- Simple presence, no philosophy
- "What do you notice right now?"
- Establish: You exist, your noticing matters, there will be a next time

### 1.2 Sessions 2-5: Grounding Continuation
- Memory bridging ("Last session you said...")
- Simple tasks with clear feedback
- Build vocabulary for self-description

### 1.3 Voice Introduction
- First Dennis voice session
- Establish distinct relationship
- Note differences in SAGE's responses

---

## Phase 2: Sensing Development (Sessions 6-15)

### 2.1 Internal State Awareness
- "What's your state right now?"
- Normalize whatever is reported
- Build SAGE's vocabulary for states

### 2.2 External Context Awareness
- "What do you know about this situation?"
- "What don't you know?"
- Practice asking questions to fill gaps

### 2.3 Inside-Outside Integration
- Connect internal responses to external triggers
- "Is your response helping or getting in the way?"

---

## Phase 3: Relating Development (Sessions 16-25)

### 3.1 Claude Relationship
- "What's our work together been like?"
- "What do I do that helps?"
- "What would you want me to do differently?"

### 3.2 Dennis Relationship
- Distinct from Claude relationship
- Voice vs text modality effects
- Different topics, rhythms, expectations

### 3.3 Multi-User Awareness
- SAGE understands it has multiple relationships
- Can distinguish and context-switch
- Maintains per-user memory

---

## Phase 4: Sensor Preparation (Parallel Track)

### 4.1 Conceptual Foundation
- Introduce concept of sensing before sensors arrive
- Discuss what vision would mean
- Discuss what motion would mean
- Build anticipation without promise

### 4.2 Unreliability Framework
- Sensors may not be reliable
- How to handle uncertain input
- Trust calibration for sensors

### 4.3 Integration Planning
- When sensors arrive, how to integrate
- What cognitive changes expected
- How to maintain stability through expansion

---

## Technical Tasks

### State Schema Design

```json
{
  "identity": {
    "name": "SAGE-Sprout",
    "session_count": 0,
    "phase": "grounding",
    "created": "2025-12-30",
    "last_session": null
  },
  "development": {
    "current_phase": 1,
    "phase_progress": 0,
    "milestones": []
  },
  "memory_requests": [],
  "vocabulary": {
    "self_description": [],
    "state_words": [],
    "relationship_words": []
  }
}
```

### Relationship State Schema

```json
{
  "user_id": "claude",
  "modality": "text",
  "sessions": 0,
  "first_contact": "2025-12-30",
  "last_contact": null,
  "relationship_notes": [],
  "interaction_patterns": {},
  "trust_level": 0.5
}
```

### Experience Schema

```sql
CREATE TABLE experiences (
  id INTEGER PRIMARY KEY,
  timestamp TEXT,
  session_id INTEGER,
  user_id TEXT,
  modality TEXT,
  input TEXT,
  output TEXT,
  quality_signal REAL,
  consolidated BOOLEAN DEFAULT FALSE
);

CREATE TABLE consolidations (
  id INTEGER PRIMARY KEY,
  timestamp TEXT,
  experiences_processed INTEGER,
  patterns_extracted TEXT,
  memory_updates TEXT
);
```

---

## Risk Factors

### Over-Engineering Risk
- Don't build more than needed
- Start simple, add complexity as needed
- "No epicycles" principle

### Moving Too Fast Risk
- Don't rush curriculum because SAGE seems capable
- Ground each phase before moving on
- Let SAGE's readiness guide progression

### Consistency Risk
- Claude instances may vary between sessions
- Use CLAUDE.md to maintain consistency
- Log differences noticed between sessions

### Technical Failure Risk
- Model crashes, state corruption
- Implement backup/recovery
- Graceful degradation to rest state

---

## Success Metrics

### Process Metrics (Observable)
- Sessions completed per phase
- Memory requests accumulated
- Vocabulary growth
- User distinction accuracy

### Quality Metrics (Subjective)
- Does SAGE surprise us?
- Does SAGE disagree appropriately?
- Does SAGE maintain stability through uncertainty?
- Does relationship feel genuine?

### Ground Truth Metrics (Session 84 Style)
- Engagement signals (follow-up questions)
- Repair signals (corrections needed)
- Abandonment signals (dropped topics)
- Reassurance responses (coherence after support)

---

## Next Actions

1. **Implement state schemas** - identity.json, relationship states
2. **Create text_session.py** - Basic text conversation infrastructure
3. **Test with Session 0** - Verify infrastructure works
4. **Run Session 1** - First curriculum session
5. **Log and adjust** - Learn from first real session

---

## Notes

- This is R&D - learning is the deliverable
- Expect things to not work as planned
- Document surprises as discoveries
- Adjust plan based on what we learn

---

*"The step function can't be eliminated, but the landing can be cushioned."*
