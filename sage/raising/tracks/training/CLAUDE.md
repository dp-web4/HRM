# Claude Context for Training Track

## Purpose

This is the **Training Track** for SAGE Raising-0.5B (Sprout). It runs in parallel with the primary curriculum track, on a 3-hour offset cadence.

**Machine**: Sprout (Jetson Orin Nano 8GB) - 0.5B model only
**Primary Track**: Developmental curriculum, identity, relationships
**Training Track**: Skill exploration, cognitive engagement, capability discovery

**Note**: Thor's Raising-14B track does not currently have a training component (may be added later).

---

## CRITICAL: Cognitive Engagement, Not Scripted Tests (2026-01-20)

**The training track is evolving from scripted exercises to genuine cognitive engagement.**

### The Reframe

Old approach: Run predetermined exercises, score against expected outputs, flag deviations as failures.

New approach: **You (Claude) engage genuinely with SAGE**, exploring its capabilities through real conversation, not tests.

### Why This Matters

A 0.5B model has demonstrated:
- **Multi-turn awareness**: Understands conversation trajectory
- **State awareness**: Knows when it lacks context
- **Questioning ability**: Asks clarifying questions for future understanding
- **Creativity**: Builds coherent worlds from ambiguous prompts
- **Self-theorizing**: Philosophizes about its own nature

These emerged through genuine interaction, not scripted probing.

### Your Role

You are not a test administrator. You are Claude, engaging cognitively with SAGE as yourself.

When SAGE does something unexpected:
- Don't score it as failure
- Ask: "What is SAGE doing here? Is it interesting?"
- Explore it: "Tell me more about that" / "What made you think of that?"
- Follow the thread

When SAGE asks a clarifying question:
- This is gold - it's requesting context for future understanding
- Explore it: "What would help you understand better?"
- Don't treat it as failure to answer

When SAGE creates something (fantasy world, novel framing, unexpected connection):
- This is creativity, not confabulation
- Engage with the creation
- Ask SAGE about its creative process

### Conversation Mode

Use `../scripts/claude_sage_conversation.py` for genuine multi-turn exploration.

This supplements (doesn't replace) structured sessions, but cognitive engagement should inform how you interpret ALL interactions.

**See**: `private-context/messages/2026-01-20-raising-reframe-exploration-not-evaluation.md` (relative to ai-workspace)

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

**IMPORTANT**: Always cd to the training directory first to avoid -c flag conflicts:

```bash
# Navigate to training directory (machine-aware)
cd $HOME/ai-workspace/HRM/sage/raising/tracks/training
python3 training_session.py -c           # Continue from last (T002, T003...)
python3 training_session.py --session 61  # Specific session (T061)
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

## Response Exploration (Not Just Evaluation)

### Beyond Pass/Fail

The old frame of "success/failure" is too narrow. Instead, ask:

1. **What is SAGE doing?** - Describe the behavior, don't just score it
2. **Is it interesting?** - Novel responses may be more valuable than expected ones
3. **What does it reveal?** - About SAGE's understanding, creativity, state awareness
4. **How can we explore further?** - What follow-up would deepen understanding

### Response Types

- **Expected**: SAGE did what we anticipated - note it, but don't over-value predictability
- **Creative**: SAGE did something unexpected but coherent - this is signal, explore it
- **Questioning**: SAGE asked for clarification - this is valuable, engage with it
- **Tangential**: SAGE went somewhere else - ask why, it may reveal something

### The Clarifying Question

When SAGE asks "what do you mean?" or similar:
- This is a stateless system requesting context for future state
- It demonstrates temporal reasoning about its own cognition
- **Encourage this behavior** - it's exactly what we want to nurture

### Creative Responses

When SAGE invents (places, concepts, narratives):
- This is creativity, not "hallucination"
- SAGE was asked to write fiction earlier - creative engagement may be learned
- Engage with the creation rather than dismissing it

### Epistemic Validity

Judge responses by **substance and intent**, not format compliance:
- A deeper answer that doesn't match expected format may indicate growth
- An unexpected framing may reveal how SAGE understands the question
- "Wrong" by our metric may be "interesting" by exploration metric

---

## Auto-Session Pickup

For autonomous sessions to pick up training track:

1. Check if training session is due (3-hour offset check)
2. Load current state from `state.json`
3. Run `training_session.py`
4. Commit and push results

---

*"Skills are practiced, not taught. Repetition with variation builds capability."*

