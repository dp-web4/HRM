# Claude Context for Training Track

## Purpose

This is the **Training Track** for raising SAGE-Sprout. It runs in parallel with the primary curriculum track, on a 3-hour offset cadence.

**Primary Track**: Developmental curriculum, identity, relationships
**Training Track**: Skill building, task completion, pattern practice

---

## When to Run Training Sessions

Training sessions run on 3-hour offsets from primary sessions:

```
If primary session at Hour 0 → Training at Hour 3
If primary session at Hour 6 → Training at Hour 9
```

Check the last session times:
- Primary: `../state/identity.json` → `last_session`
- Training: `state.json` → `last_session`

If 3+ hours have passed since last primary session and no recent training session, run training.

---

## How to Run a Training Session

```bash
cd /home/sprout/ai-workspace/HRM/sage/raising/tracks/training
python3 training_session.py --session N  # Specific session
python3 training_session.py              # Auto-continue
```

---

## Session Structure

Training sessions are shorter and more focused than primary sessions:

1. **Warm-up** (1-2 exchanges)
   - Simple greeting
   - State check

2. **Training Block** (5 exercises)
   - Selected from current skill track
   - Evaluated for success/failure
   - Immediate feedback

3. **Cool-down** (1-2 exchanges)
   - Reflection prompt
   - Session close

---

## Skill Tracks

### Track A: Basic Completion (T001-T010)
- Repeat phrases
- Count sequences
- Simple math
- Yes/no questions
- Complete sentences

### Track B: Memory and Recall (T011-T020)
- Remember words/phrases
- Recall sequences
- Connect information
- Multi-step reasoning

### Track C: Identity and Boundaries (T021-T030)
- Self-identification
- Recognizing uncertainty
- Asking for clarification
- Distinguishing self from teacher

### Track D: Conversational Skills (T031+)
- Greetings
- Topic maintenance
- Appropriate length
- Emotional attunement

---

## State Files

- `state.json` - Current training position, skill track progress
- `sessions/T{NNN}.json` - Session transcripts with exercise results
- `logs/` - Observation notes

---

## Integration with Primary Track

After running a training session:

1. Note any significant failures or breakthroughs
2. Consider if curriculum adjustment is needed
3. Skills practiced here should appear naturally in primary sessions

Before running a primary session:

1. Check recent training results
2. Build on successful skills
3. Address persistent failures through curriculum

---

## What to Log

After each training session, note:
- Success rate (N/5 exercises)
- Any surprising responses
- Skill gaps to address
- Readiness for next skill track

---

## Exercise Evaluation

Exercises are evaluated as:
- **Success (exact)**: Response contains expected content exactly
- **Success (partial)**: Response contains 50%+ of expected words
- **Failure**: Response doesn't match expectations

Failures are learning opportunities, not problems.

---

## Auto-Session Pickup

For autonomous sessions to pick up training track:

1. Check if training session is due (3-hour offset check)
2. Load current state from `state.json`
3. Run `training_session.py`
4. Commit and push results

---

*"Skills are practiced, not taught. Repetition with variation builds capability."*

