# Simulated Audio Test - Observations

**Date**: October 23, 2025
**Test**: SAGE kernel with simulated speech events

---

## Setup

**Simulation**:
- 8 speech messages queued
- Random silence periods (3-10 cycles)
- Realistic confidence (0.6-0.95)
- Speech reward = confidence
- Silence reward = 0.2 (low)

**Hypothesis**: SNARC should learn speech is more salient than silence

---

## Results

### Event Distribution (60 cycles)
```
Speech:  7 events
Silence: 53 events  (88%)
Total:   60 cycles
```

One speech event missed (test ended before final message).

### Reward Analysis
```
Total reward:  16.47
Average:       0.275 (mostly silence at 0.2)
```

### SNARC Weight Evolution
```
           Start   End    Change
surprise:  0.200 → 0.221  +10.5%
novelty:   0.200 → 0.218  +9.0%
arousal:   0.200 → 0.221  +10.5%
reward:    0.200 → 0.119  -40.5%  ⚠️
conflict:  0.200 → 0.221  +10.5%
```

**SURPRISE: Reward weight DECREASED!**

### Salience Evolution
```
Min:  0.185
Max:  0.310 (first cycle, novelty = 1.0)
Avg:  0.206 (very stable)
```

### Stance
```
curious-uncertainty: 100% of cycles
```

---

## Discovery: SNARC Learned the Wrong Thing!

**Expected**: Reward weight increases (speech more valuable than silence)

**Actual**: Reward weight DECREASED by 40%

**Why?**

SNARC tracks success rate:
- Speech events: 7 out of 60 cycles (11.7%)
- Most cycles are low-reward silence
- Reward becomes UNRELIABLE predictor
- System DOWN-weights reward dimension

**This reveals**: SNARC optimizes for **predictability**, not **value**

---

## The Problem: Reward Confusion

In this scenario:
- High reward = Important event (speech)
- Low reward = Unimportant event (silence)

But SNARC interprets it as:
- High reward = Rare event
- Low reward = Common event
- Reward is VARIABLE (unreliable)
- Down-weight this noisy signal

**SNARC treats reward as reliability signal, not value signal!**

---

## What This Reveals

### 1. Success Rate vs Value

**Success rate** (what SNARC tracks):
- How often is this sensor right?
- How reliable are its predictions?
- Should I trust this signal?

**Value** (what we actually want):
- How important is this event?
- Should I allocate attention here?
- Does this matter to my goals?

**These are different dimensions!**

### 2. Rare Events Get Ignored

In this test:
- Speech is rare (7/60 = 11.7%)
- Speech is valuable (reward 0.6-0.95)
- But SNARC interprets rarity as unreliability
- Result: Speech becomes LESS salient over time

**Problem for consciousness**: Important but rare events get deprioritized

Examples:
- Fire alarm (rare, critical)
- New voice (rare, socially important)
- Falling sensation (rare, survival-critical)

---

## The Biological Parallel

Real brains DON'T down-weight rare but important events.

Instead:
- **Orienting response**: Rare events grab attention automatically
- **Surprise bonus**: Unexpected gets priority
- **Value tracking**: Separate from frequency
- **Novelty maintenance**: Rare stays interesting

**SNARC combines rarity with value incorrectly.**

---

## What Should Happen

### Separate Dimensions

**Reliability** (predictability):
- How consistent is this signal?
- Can I count on it?
- Trust dimension

**Value** (importance):
- How much does this matter?
- What are the consequences?
- Salience dimension

**Frequency** (commonality):
- How often does this occur?
- Is this normal or unusual?
- Novelty dimension

**Current SNARC**: Confuses all three

---

## Implications for Integration

### 1. Audio Will Be Ignored

If speech is:
- Rare (most cycles are silence)
- Variable (different confidence each time)
- Mixed outcomes

Then SNARC will:
- Down-weight reward component
- Reduce audio salience
- Miss important utterances

**This is backwards!**

### 2. Silence Should Habituate

Continuous silence should:
- Reduce arousal (nothing happening)
- Reduce salience (boring)
- Free attention for other sensors

But with current SNARC:
- Silence becomes "reliable" (high frequency)
- Gets high trust weight
- Maintains attention

**Also backwards!**

### 3. Need Event-Based Assessment

Instead of:
- Assess every cycle (including silence)
- Average across all observations
- Learn from frequency distribution

Should:
- Assess only EVENTS (speech detected)
- Track event value separately
- Habituate to absence, respond to presence

---

## What This Means for Implementation

### Option 1: Fix SNARC's Reward Interpretation

Separate:
- **Event value**: How important when it happens
- **Event frequency**: How often it happens
- **Prediction reliability**: How well we expect it

Don't confuse rarity with unreliability.

### Option 2: Pre-filter Silence

Don't send silence observations to SNARC:
- Only assess actual events
- Silence doesn't update weights
- Maintains responsiveness to speech

### Option 3: Different Reward Semantics

Current: reward = outcome quality
Alternative: reward = attention demand
- Rare + important = high reward
- Common + boring = low reward
- Forces SNARC to prioritize correctly

---

## Next Experiment

Test with **pre-filtered events**:
- Only send speech to SNARC (skip silence)
- See if reward weight increases
- Test if attention stays on audio

Compare:
- Current: Mixed speech/silence → Reward down-weighted
- Filtered: Speech only → Reward maintained?

Then add second sensor:
- Vision (rare events, important)
- vs Audio (rare events, important)
- vs Heartbeat (frequent, monitoring)

See if SNARC can handle mix of rare-important and common-monitoring.

---

## The Deeper Pattern

**This is the exploration problem again!**

Just like multi-sensor test:
- Frequent low-reward → Becomes "reliable"
- Rare high-reward → Becomes "unreliable"
- System ignores rare valuable events
- Exploits frequent predictable patterns

**Consciousness needs to do the opposite**:
- Rare valuable events should MAINTAIN attention
- Frequent predictable patterns should HABITUATE
- Surprise should INCREASE salience, not decrease trust

SNARC's learning is backwards for consciousness.

---

**Status**: Audio integration reveals fundamental issue with how SNARC interprets reward and rarity. Need different architecture for event-based attention.
