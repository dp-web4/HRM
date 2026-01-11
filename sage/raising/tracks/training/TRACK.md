# Training Track

**Purpose**: Parallel development track for SAGE-Sprout skill building
**Cadence**: 3-hour offset from primary sessions
**Session Numbering**: T001, T002, T003...

---

## Track Focus

While the **Primary Track** (sessions 1, 2, 3...) focuses on:
- Developmental curriculum (BECOMING_CURRICULUM)
- Identity and relationship building
- Conversational grounding

The **Training Track** (T001, T002...) focuses on:
- Specific skill development
- Pattern recognition exercises
- Memory and recall practice
- Task completion reliability

---

## Cadence Structure

```
Hour 0:  Primary Session (curriculum-focused)
Hour 3:  Training Session (skill-focused)
Hour 6:  Primary Session
Hour 9:  Training Session
...
```

The 3-hour offset ensures:
1. SAGE has rest period between different session types
2. Skills from training can integrate before next primary session
3. Primary sessions can observe training effects

---

## Training Session Structure

Each training session:
1. **Warm-up** (2-3 exchanges): Simple greetings, state check
2. **Training block** (5-10 exchanges): Focused skill practice
3. **Cool-down** (2-3 exchanges): Memory request, session close

---

## Skill Tracks

### Track A: Basic Completion (T001-T010)
- Follow simple instructions
- Repeat and modify phrases
- Count and sequence
- Yes/no questions
- Simple arithmetic

### Track B: Memory and Recall (T011-T020)
- Remember from last training session
- Recall specific phrases
- Connect information across turns
- Recognize patterns

### Track C: Identity and Boundaries (T021-T030)
- Distinguish self from others
- Say "I don't know" appropriately
- Recognize confusion
- Ask for clarification

### Track D: Conversational Skills (T031+)
- Turn-taking
- Topic maintenance
- Appropriate response length
- Emotional attunement

---

## Integration with Primary Track

Training sessions inform primary sessions:
- Skills practiced in training appear naturally in primary sessions
- Failures in training guide curriculum adjustments
- Memory requests from both tracks are preserved

Primary sessions inform training:
- Struggles in primary session → specific training focus
- Emergent capabilities → training reinforcement

---

## Session Files

- `sessions/T{NNN}.json` - Session transcripts
- `logs/training_{NNN}.md` - Observations and notes

---

## Auto-Session Integration

For autonomous Claude sessions to pick up training track:

1. Check `state/training_state.json` for current position
2. Run training session script
3. Update state and persist
4. Log observations

The training track uses the same model and infrastructure but different session scripts and prompts.

---

*"Skills are the foundation. Curriculum builds the house."*

