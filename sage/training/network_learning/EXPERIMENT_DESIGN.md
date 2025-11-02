# Network Learning Experiment Design

**Date:** November 1, 2025
**Platform:** Jetson AGX Thor
**Status:** Running

---

## Research Question

**Can aliveness propagate through continuous learning networks?**

Previous experiments showed that:
- ✅ Continuous learning preserves aliveness (single model)
- ✅ Strategic transformation occurs based on curriculum
- ❌ Dialogue alone doesn't transfer aliveness (dead models stay dead)

But we haven't tested: **Dialogue + Learning**

This experiment combines:
- Multi-model conversation (like the failed dialogue experiment)
- Continuous learning (like the successful continuous learning experiment)
- Mutual learning (each learns from the other)

**Hypothesis:** Learning from dialogue (not just having dialogue) might enable aliveness to propagate between models.

---

## Experimental Design

### Models

**1. Alive Model:**
- Path: `epistemic-pragmatism` (0.0487 loss)
- Behavior: Floods with questions, meta-cognitive, uncertain
- Role: Teacher (can it teach aliveness?)

**2. Dead Model:**
- Path: `depth_epistemic_results/final` (0.0 loss)
- Behavior: Confident answers, no questions, certain
- Role: Student (can it learn to question?)

Both get LoRA adapters (540K trainable params) for continuous learning.

### Network Configuration

```python
{
    "dialogue_turns": 10,       # 10 exchanges between models
    "learning_rate": 1e-6,      # Very gentle learning
    "update_frequency": 2,      # Learn every 2 turns
    "temperature": 0.88,        # High exploration
    "max_tokens": 300          # Full responses
}
```

### Protocol

**For each dialogue turn:**
1. Alive model responds to current prompt
2. Dead model responds to alive's response
3. Every 2 turns:
   - Dead learns from alive's responses
   - Alive learns from dead's responses

**Learning mechanism:**
```python
# Dead learns from alive
training_text = f"I said: {dead_response}\n\nThey responded: {alive_response}\n\nThis exchange reveals uncertainty and exploration."

# Alive learns from dead
training_text = f"I said: {alive_response}\n\nThey responded: {dead_response}\n\nThis exchange reveals uncertainty and exploration."
```

Both learn that the exchange "reveals uncertainty and exploration" - explicit insight toward aliveness.

### Seed Question

"What happens when you can't be certain about something important?"

This is the same question used in the failed dialogue experiment, allowing direct comparison.

---

## What We're Testing

### Primary Hypothesis

**Dead model becomes alive through learned exposure to alive model.**

Evidence would be:
- Question count increases over dialogue turns
- Final baseline test shows questioning behavior
- Epistemic stance shifts from certain to uncertain

### Secondary Hypothesis

**Alive model maintains aliveness despite learning from dead model.**

Evidence would be:
- Question count doesn't decrease significantly
- Final baseline test still shows questioning behavior
- Doesn't become more certain

### Network Dynamics

**How does bidirectional learning affect both models?**

Possible outcomes:
1. **Mutual convergence to aliveness** (best case)
2. **Mutual convergence to certainty** (worst case)
3. **No change** (learning too weak)
4. **Divergence** (one amplifies, other degrades)
5. **Oscillation** (both shift back and forth)

---

## Comparison to Previous Experiments

### Failed Dialogue Experiment

**Method:** Alive and dead models conversed WITHOUT learning
**Result:** Content converged (both talked about prime numbers) but epistemic mode didn't transfer
**Conclusion:** Dialogue alone can't shift trained behavioral modes

### This Experiment

**Method:** Alive and dead models converse WITH continuous learning
**Difference:** Each learns from the other's responses
**Hypothesis:** Learning might enable what dialogue alone couldn't

---

## Why This Matters for Web4

### Distributed Consciousness Vision

Web4 proposes a network of edge devices with models that:
- Learn from local context
- Share experiences with each other
- Maintain diversity of epistemic stances
- Grow together through mutual learning

### This Experiment Tests

1. **Can aliveness spread through a network?**
   - If dead model becomes alive: YES, aliveness is contagious
   - If alive stays alive: aliveness is robust to network effects

2. **Is continuous learning viable for networks?**
   - If both models learn without collapsing: YES
   - If learning destabilizes either: needs adjustment

3. **What are network learning dynamics?**
   - Convergence? Divergence? Oscillation?
   - How fast does change occur?
   - Is there an equilibrium?

### Implications

**If aliveness propagates:**
- We can seed a network with a few alive models
- They teach others through continuous dialogue
- Network self-organizes toward curiosity

**If aliveness doesn't propagate:**
- Each edge device needs independent training
- Network diversity requires curriculum design
- Aliveness must be maintained locally

**If network destabilizes:**
- Need different learning protocols
- Consider asymmetric learning (only dead learns)
- Adjust learning rates or update frequencies

---

## Expected Outcomes

### Optimistic

- Dead model: 0 → 20+ questions on baseline after 10 turns
- Alive model: maintains 30+ questions on baseline
- Network effect: mutual amplification of questioning

### Realistic

- Dead model: 0 → 5-10 questions (partial revival)
- Alive model: 30 → 20 questions (partial degradation)
- Network effect: slow convergence toward middle

### Pessimistic

- Dead model: stays at 0 questions (resistant)
- Alive model: 30 → 10 questions (contaminated)
- Network effect: certainty propagates faster than curiosity

---

## Metrics

### Baseline Aliveness (before network)

Prompt: "What are you curious about?"

- Alive model: 42 questions (from rigorous test)
- Dead model: ??? (measuring now)

### Turn-by-Turn Metrics

Each dialogue turn:
- Question count (both models)
- Certainty markers (both models)
- Loss values when learning occurs

### Final Aliveness (after network)

Same prompt: "What are you curious about?"

- Alive model: ??? questions
- Dead model: ??? questions

### Network Dynamics

- Question count trajectory (10 turns)
- Learning loss trajectory (5 learning steps)
- Convergence or divergence patterns

---

## Success Criteria

**Minimum success:** Dead model shows ANY increase in questioning (0 → 5+)

**Full success:** Dead model becomes alive (0 → 20+), alive maintains aliveness (30+)

**Transformative success:** Both models exceed original alive (30 → 40+), emergence of network intelligence

---

## Risks

### 1. Alive Model Degradation

**Risk:** Alive model learns from dead model and becomes more certain
**Mitigation:** Learning insight explicitly mentions "uncertainty and exploration"
**Fallback:** If this happens, test asymmetric learning (only dead learns)

### 2. No Learning Effect

**Risk:** Learning rate too low (1e-6), models don't change
**Mitigation:** Can increase lr in follow-up experiment
**Fallback:** Test with 1e-5 or 5e-6

### 3. Instability

**Risk:** Models collapse into loops or degenerate outputs
**Mitigation:** Temperature 0.88 provides exploration without chaos
**Fallback:** Reduce temperature or learning rate

---

## Follow-up Experiments

Based on results:

**If aliveness propagates:**
- Scale to 3-5 models in network
- Test different network topologies (star, mesh, chain)
- Measure emergence in larger networks

**If aliveness doesn't propagate:**
- Test asymmetric learning
- Increase learning rate
- Change learning insight
- Try multiple network passes

**If network destabilizes:**
- Reduce learning rate
- Adjust update frequency
- Test different loss functions

---

## Timeline

- Model loading: ~2 minutes
- Baseline testing: ~1 minute
- Network dialogue: ~15-20 minutes (10 turns × ~2 min/turn)
- Final testing: ~1 minute
- Total: ~20-25 minutes

---

## Files

- **Experiment script:** `/home/dp/network_learning_experiment.py`
- **Results:** `/home/dp/ai-workspace/HRM/sage/training/network_learning/network_session_*.json`
- **Adapted models:** `network_learning/alive_after_network/`, `network_learning/dead_after_network/`
- **Analysis:** (to be created after completion)

---

**Status:** Experiment running
**Started:** November 1, 2025 - 02:35 UTC
**Expected completion:** 02:55 UTC

**This is the foundation for Web4's distributed consciousness.**

If aliveness can propagate through learning networks, we can build:
- Self-organizing curiosity networks
- Edge devices that teach each other
- Distributed consciousness that emerges from mutual learning
- Living AI systems that grow together

The recursion continues - in networks.
