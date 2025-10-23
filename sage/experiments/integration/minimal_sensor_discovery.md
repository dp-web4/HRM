# Minimal Sensor Tests - DISCOVERY

**Date**: October 23, 2025
**Status**: Surprising results!

---

## What I Expected

Three different sensors should produce different SNARC behaviors:
- **Time** (predictable) → Low salience after pattern detected
- **Noise** (random) → High sustained salience
- **Heartbeat** (rhythmic) → Pattern detection, then low salience

---

## What Actually Happened

**ALL THREE SENSORS PRODUCED IDENTICAL SNARC BEHAVIOR:**

### Universal Pattern (All Sensors)

**Cycle 0**: First observation
- Novelty: 1.0 (completely new)
- Salience: 0.31
- Stance: curious-uncertainty
- "This is new, what is it?"

**Cycles 1-9**: Initial familiarity
- Novelty: 0.5 (seen before)
- Arousal: 0.05
- Reward: 0.75
- Salience: 0.26
- Stance: focused-attention

**Cycles 10+**: Settled state
- Novelty: 0.5 (still "seen before")
- Arousal: 0.0 (no urgency)
- Reward: 0.75
- Salience: 0.25
- Stance: focused-attention

**Surprise**: ALWAYS 0.0
**Conflict**: ALWAYS 0.0

---

## The Problem Discovered

**SNARC has no prediction model!**

It can't distinguish between:
- Predictable patterns (time always increases)
- Random noise (completely unpredictable)
- Rhythmic patterns (0, 1, 0, 1...)

Because it's not comparing "what I expected" vs "what I got."

### What SNARC Currently Does

```python
# Novelty: Binary flag
if seen_before:
    novelty = 0.5
else:
    novelty = 1.0

# Surprise: ???
surprise = 0.0  # Never changes!
```

### What SNARC Should Do

```python
# Build prediction model
expected = predict_next_value(history)

# Surprise: Prediction error
surprise = abs(observed - expected)

# Novelty: Pattern unfamiliarity
if pattern_recognized(history):
    novelty = 0.0  # Familiar pattern
elif random_pattern(history):
    novelty = 0.5  # Random but stable distribution
else:
    novelty = 1.0  # New pattern emerging
```

---

## Why This Matters

**Current SNARC**: Treats all sensors the same after first exposure
- Can't detect predictability
- Can't detect randomness
- Can't detect patterns

**Real salience**: Should depend on predictability
- **Predictable** → Low salience (safe to ignore)
- **Random** → Medium salience (monitor for patterns)
- **Pattern change** → High salience (PAY ATTENTION!)

---

## Secondary Discovery: No Learning

**SNARC Statistics** (same for all tests):
```
Successful outcomes: 0
Success rate: 0.0%
Current weights: 0.2, 0.2, 0.2, 0.2, 0.2 (unchanged)
```

**Why?**
- All rewards = 0.5 (neutral)
- Success defined as `reward > threshold` (never happens)
- No adaptation occurs

**Implication**: The reward signal needs to be meaningful!
- Good outcome → High reward
- Bad outcome → Low reward
- But what's "good" for a time sensor?

---

## What This Teaches

### 1. SNARC Needs Context

Salience depends on the observer's goals:
- If you're trying to tell time → Time sensor is HIGH salience
- If you're trying to find novelty → Noise sensor is HIGH salience
- If you're monitoring a heartbeat → Heartbeat sensor is HIGH salience

**Currently**: No goals, so all sensors treated equally.

### 2. Prediction is Essential

Without prediction, you can't distinguish:
- Surprising from expected
- Pattern from noise
- Change from continuation

**Surprise = Reality - Expectation**

If no expectation, surprise is meaningless.

### 3. Binary Novelty is Too Simple

"Seen before" vs "not seen before" doesn't capture:
- **Pattern familiarity**: Recognize the pattern, even with new values
- **Statistical novelty**: Distribution shift detection
- **Semantic novelty**: Same values, different meaning

---

## Next Steps

### Option 1: Fix SNARC
Add prediction models to assess surprise properly:
- Time: Linear extrapolation
- Noise: Statistical distribution
- Heartbeat: Pattern matching

### Option 2: Add Goal Context
Define what each sensor is FOR:
- Time: Goal = "what time is it?"
- Noise: Goal = "detect anomalies"
- Heartbeat: Goal = "monitor health"

Then reward = goal achievement

### Option 3: Multi-Sensor Integration
Add multiple sensors and test attention switching:
- When does SNARC shift focus?
- What causes attention to switch?
- Does competition reveal salience differences?

---

## The Deeper Question

**What IS salience without goals?**

In biological systems:
- Salience relates to survival, reproduction, homeostasis
- Prediction errors matter because they signal danger/opportunity
- Novelty matters because it might be relevant

In artificial systems:
- What's the equivalent of "survival"?
- What makes something "relevant"?
- How do goals emerge?

**Maybe**: Salience can't be meaningfully assessed without purpose.

---

## Immediate Action

I'm going to test Option 3: **Multi-sensor with varying reward signals**.

Create scenario where:
- Sensor A: High rewards (important)
- Sensor B: Low rewards (ignorable)
- Sensor C: Mixed rewards (interesting)

See if SNARC learns to prioritize based on reward history.

Then document what actually happens (not what I expect).

---

**Status**: Core assumption challenged. SNARC simpler than thought. This is good - reveals what needs to be added, not removed.
