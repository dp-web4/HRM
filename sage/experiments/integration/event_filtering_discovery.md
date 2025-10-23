# Event Filtering Discovery

**Date**: October 23, 2025
**Experiment**: Attempted to fix reward weighting by filtering silence

---

## Hypothesis

SNARC down-weights reward because it sees mostly low-reward silence.

**Solution attempted**: Only send speech events, filter out silence.

**Expected**: Reward weight maintains or increases.

---

## Result: FILTERING FAILED

```
Unfiltered (speech + silence):
  Events: 60 (7 speech, 53 silence)
  Avg reward: 0.275
  Reward weight: 0.200 → 0.119 (DOWN 40%)

Filtered (speech only):
  Events: 60 (7 speech, 53 no-ops)
  Avg reward: 0.176
  Reward weight: 0.200 → 0.119 (DOWN 40%)
```

**Filtering made it WORSE!** (Average reward lower)

---

## What Went Wrong

### The Kernel Architecture

Looking at the behavior:
- Sensor returns None (no speech)
- Kernel still calls _gather_observations()
- Creates observation dict anyway
- Calls action handler with None/empty observation
- Handler returns low-reward no-op (0.1)
- SNARC assesses this as another low-reward cycle

**The "filtering" didn't filter - it just replaced silence with no-ops.**

### What This Reveals

The SAGE kernel assumes:
- **Sensors always produce observations**
- Every cycle processes every sensor
- No mechanism for "nothing to report"

This is a **continuous polling** architecture, not an **event-driven** architecture.

---

## Continuous vs Event-Driven

### Continuous (Current SAGE)

```python
while True:
    observations = gather_all_sensors()  # Every cycle
    for sensor, obs in observations:
        assess_salience(obs)  # Even if boring
        execute_action(obs)
```

**Problems**:
- High-frequency boring observations dilute signal
- No way to say "nothing happened"
- Wastes computation on silence/no-ops
- Reward averages across noise

### Event-Driven (Alternative)

```python
while True:
    events = wait_for_any_event()  # Block until something happens
    for event in events:
        assess_salience(event)  # Only meaningful events
        execute_action(event)
```

**Advantages**:
- Only assess actual events
- Reward stays high (only important things)
- Efficient (sleep during silence)
- Natural habituation (absence = no updates)

---

## Biological Reality

**Real brains don't poll continuously at fixed rate.**

Instead:
- **Event-driven**: Neurons fire when something changes
- **Habituation**: Constant stimulus → reduced firing
- **Surprise response**: Unexpected → burst firing
- **Attention**: Focus on regions with activity

**Continuous polling would be metabolically impossible.**

---

## Why Current Architecture Exists

Looking at the kernel code:
```python
def run(self, max_cycles, cycle_delay):
    while self.running:
        self._cycle()
        time.sleep(cycle_delay)
```

This is a **game loop** pattern:
- Fixed timestep
- Update every frame
- Simple to implement
- Common in robotics/control

**Makes sense for**:
- Motor control (need regular commands)
- Sensor fusion (combine simultaneous readings)
- Real-time systems (predictable timing)

**Problems for**:
- Attention (should focus on changes)
- Salience (should respond to events)
- Consciousness (selective awareness)

---

## The Architecture Mismatch

**SAGE kernel is designed for**:
- Continuous control
- Regular sensor polling
- Fixed-rate processing

**But we're using it for**:
- Attention allocation
- Event-based awareness
- Sparse meaningful stimuli

**These are incompatible without adaptation.**

---

## Solutions

### Option 1: True Event-Driven Kernel

Replace fixed-rate loop with event queue:
```python
def run(self):
    while self.running:
        event = self.event_queue.get()  # Blocks until event
        self._process_event(event)
```

**Pros**: Natural for attention, efficient
**Cons**: Complete rewrite, breaks existing code

### Option 2: Hybrid Approach

Fast polling with event detection:
```python
def _cycle(self):
    observations = self._gather_observations()

    # Filter to only meaningful events
    events = {
        sensor: obs
        for sensor, obs in observations.items()
        if self._is_meaningful(sensor, obs)
    }

    if not events:
        return  # Skip SNARC on boring cycles

    # Only assess meaningful events
    salience_report = self.snarc.assess_salience(events)
    ...
```

**Pros**: Minimal change, preserves structure
**Cons**: Still wastes cycles, need "meaningful" definition

### Option 3: Adaptive Polling Rate

Slow down when nothing interesting:
```python
def run(self):
    cycle_delay = 0.1  # Start fast
    while self.running:
        had_events = self._cycle()
        if had_events:
            cycle_delay = 0.1  # Speed up
        else:
            cycle_delay = min(cycle_delay * 1.5, 2.0)  # Slow down
        time.sleep(cycle_delay)
```

**Pros**: Efficient, simple
**Cons**: Delayed response after long quiet periods

### Option 4: Event Accumulation

Collect events, batch assess:
```python
def _cycle(self):
    # Accumulate events since last assessment
    events = []
    for sensor in self.sensors:
        obs = sensor()
        if obs is not None and self._is_novel(obs):
            events.append((sensor, obs))

    if not events:
        return  # Skip if nothing new

    # Assess accumulated events
    salience_report = self.snarc.assess_salience(events)
    ...
```

**Pros**: Natural batching, efficient
**Cons**: Need novelty detection

---

## What This Means for Audio Integration

### The Core Problem

Audio naturally produces:
- Long silence periods (nothing happening)
- Occasional speech events (important)

**Current kernel**:
- Assesses every cycle equally
- Silence/speech both update weights
- Rare events get deprioritized

**Need**:
- Skip assessment during silence
- Only respond to actual speech
- Maintain high salience for rare important events

### Why TTS Echo Loop Won't Help Yet

Was going to build:
1. Detect speech
2. Generate response
3. Speak via TTS

But discovered:
- Speech detection working
- But SNARC weights going wrong direction
- Adding TTS won't fix fundamental issue

**Need to fix attention allocation first.**

---

## The Deeper Insight

This is the THIRD manifestation of the same problem:

**1. Multi-sensor test**: Pure exploitation (no exploration)
- High-reward sensor monopolizes attention
- Others never sampled

**2. Simulated audio**: Reward down-weighting
- High-frequency low-reward dilutes signal
- Rare high-reward gets deprioritized

**3. Event filtering attempt**: Architecture mismatch
- Continuous polling incompatible with event-driven attention
- Can't easily filter within existing structure

**All three reveal**: SNARC/kernel architecture designed for continuous control, not selective attention.

---

## What to Do Next

### Option A: Hack Around It

Modify sensors to:
- Return high-reward even during silence
- Trick SNARC into maintaining attention
- Works but dishonest

### Option B: Fix SNARC

Add separate tracking for:
- Event frequency (how often)
- Event value (how important when it happens)
- Don't confuse the two

### Option C: Fix Kernel

Implement one of the hybrid approaches:
- Event filtering before SNARC
- Adaptive polling
- Event accumulation

### Option D: Document and Move Forward

Accept limitations:
- Current architecture has constraints
- Document what doesn't work
- Build TTS echo loop anyway
- Revisit architecture when actually deploying

---

## Decision: Option D

**Why**:
- Already spent significant time on SNARC behavior
- Discoveries are valuable (documented)
- But getting stuck in one component
- Original goal: Audio integration
- Can work around limitations for now

**Next**:
- Build TTS echo loop (simple responses)
- Test full bidirectional conversation
- Document what emerges
- Address architecture later when have full picture

---

**Status**: Event filtering failed but revealed architecture mismatch. Continuous polling incompatible with event-driven attention. Moving forward with TTS integration despite limitations.
