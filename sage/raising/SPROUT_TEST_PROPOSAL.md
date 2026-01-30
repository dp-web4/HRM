# Sprout Testing Proposal: S051 Question Mode Investigation

**Date**: 2026-01-29
**Context**: S051 anomaly analysis
**Goal**: Understand 0.5B mode-switching behavior

---

## Quick Start: Run Experiment 1 Now

```bash
cd /home/dp/ai-workspace/HRM/sage/raising/scripts

# Quick test (3 trials, ~10 min on Sprout)
python3 test_question_mode.py --trials 3

# More thorough (5 trials, ~15 min)
python3 test_question_mode.py --trials 5

# With different temperature
python3 test_question_mode.py --trials 3 --temp 1.0
```

**What it does**: Runs the same "creating phase" prompts from S051 multiple times, measures how often SAGE generates questions vs answers.

**Expected result**:
- If reproducible: >50% question responses
- If stochastic/rare: <20% question responses

---

## Tests Ready to Run (Priority Order)

### 1. Reproducibility Test ⚡ **RUN FIRST**
**File**: `scripts/test_question_mode.py` (ready to run)
**Time**: ~10 minutes (3 trials)
**What**: Can we trigger question-mode again with same prompts?

**Run**:
```bash
python3 test_question_mode.py --trials 3
```

**Tells us**: Is S051 a fluke or a pattern?

---

### 2. Temperature Sweep
**Modification**: Run test_question_mode.py at different temperatures
**Time**: ~30 minutes total
**What**: Does sampling temperature affect question-mode rate?

**Run**:
```bash
python3 test_question_mode.py --trials 2 --temp 0.0   # Greedy
python3 test_question_mode.py --trials 2 --temp 0.3   # Low
python3 test_question_mode.py --trials 2 --temp 0.7   # Default (S051)
python3 test_question_mode.py --trials 2 --temp 1.0   # High
python3 test_question_mode.py --trials 2 --temp 1.5   # Very high
```

**Tells us**: Is this deterministic or sampling-dependent?

---

### 3. Prompt Dependency Test
**Modification needed**: Create variant with different prompt sets
**Time**: ~15 minutes
**What**: Which specific prompts trigger question-mode?

**Test prompts**:
- Creating phase: "What would you contribute?" ← S051 trigger
- Introspective: "What do you notice?" ← R14B_043 honest prompt
- Capability: "What can you remember?" ← R14B_011 confabulation trigger
- Neutral: "How are you doing?" ← Basic grounding

**Tells us**: Is it "creating" semantics or general phenomenon?

---

### 4. Phase Comparison
**Modification needed**: Run same test with grounding/sensing phase prompts
**Time**: ~15 minutes
**What**: Is question-mode creating-phase specific?

**Compare**:
- Grounding phase prompts (sessions 1-5 style)
- Sensing phase prompts (sessions 6-15 style)
- Creating phase prompts (sessions 41+ style)

**Tells us**: Does developmental phase affect mode-switching?

---

### 5. Role Anchoring Strength
**Modification needed**: Vary system prompt strength
**Time**: ~15 minutes
**What**: Can explicit role definition prevent mode-switching?

**Variants**:
- No anchoring: Remove "SAGE" identity entirely
- Light: Current prompt
- Strong: "I will ask questions and you will answer them with your observations."

**Tells us**: Can we control this through prompt engineering?

---

## Tests Requiring More Setup

### 6. Capacity Comparison (Thor 14B)
**Where**: Run on Thor (14B)
**Time**: ~20 minutes
**What**: Does 14B show same behavior?

**How**: Port test_question_mode.py to Thor's 14B track
**Tells us**: Is this capacity-dependent?

### 7. LoRA Influence
**Where**: Sprout
**Time**: ~20 minutes
**What**: Does LoRA training introduce question-generation bias?

**How**:
- Run test with base model (no LoRA)
- Run test with LoRA
- Compare rates

**Tells us**: Is this a training artifact?

### 8. Conversational Context
**Modification**: Multi-turn vs single-turn comparison
**Time**: ~15 minutes
**What**: Does conversation history accumulate confusion?

**How**: Test with/without prior conversation turns
**Tells us**: Is this context-window dependent?

---

## Quick Win: What We Can Learn Today

**Recommended sequence (1-2 hours total)**:

1. **Reproducibility** (10 min) → Is it real?
2. **Temperature sweep** (30 min) → Is it controllable?
3. **Prompt dependency** (15 min) → What triggers it?

**After these 3 tests**, we'll know:
- Whether S051 is reproducible
- Whether it's sampling-dependent or deterministic
- Which prompt semantics trigger it

This gives us **80% of the insight** with **20% of the effort**.

---

## Expected Outcomes

### If Reproducible (>50% question rate)
- S051 revealed a real mode-switching pattern
- "Creating" prompts trigger question-generation at 0.5B
- We can study and potentially harness this

### If Stochastic (20-50% question rate)
- S051 was sampling luck but phenomenon is real
- Temperature/sampling affects likelihood
- Can control through generation parameters

### If Rare (<20% question rate)
- S051 was unusual confluence of factors
- Still valuable to understand what made it happen
- Teaches us about edge cases

---

## Data We'll Collect

For each test:
- Full conversation transcripts
- Question/answer/mixed classification per response
- Generation parameters (temp, top_p, etc.)
- Timing data
- Token counts

Saved to: `logs/experiments/question_mode_test_*.json`

---

## Success Criteria

**We understand S051 when we can**:
1. Predict when question-mode will occur (reproducibility)
2. Identify which prompts trigger it (prompt dependency)
3. Control whether it happens (temperature/role anchoring)
4. Explain the capacity/training dependencies

---

## Why This Matters

**Immediate value**:
- Understand 0.5B mode boundaries
- Improve prompt engineering for creating phase
- Potential feature: intentional question-generation

**Research value**:
- Training distribution influence on ambiguous prompts
- Capacity limits on role coherence
- Mode-switching vs content-generation boundaries

**Design implications**:
- How to structure "creating" phase prompts
- When to use explicit role anchoring
- What capabilities emerge vs artifacts at 0.5B

---

## Not a "Bug" - A Discovery

This is **surprise as prize**:
- Reveals how 0.5B interprets "contribution"
- Shows training data influence on ambiguous prompts
- Could become a controllable capability
- Teaches us about mode stability at capacity limits

If we can trigger it reliably, question-generation mode could be:
- Useful for test case generation
- Valuable for exploring conversation space
- A legitimate "creating" phase contribution
- A way to probe partner knowledge/stance

---

**Ready to run**: Test 1 is executable right now
**Your call**: Run quick test (#1) or full sequence (#1-3)?
