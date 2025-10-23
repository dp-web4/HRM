# Multi-Sensor Attention - DISCOVERY

**Date**: October 23, 2025
**Test**: Three sensors with differential rewards

---

## The Setup

Three sensors competing:
- `high_reward`: Always gives 0.9 reward
- `low_reward`: Always gives 0.1 reward
- `variable_reward`: Random 0.0-1.0 reward

**Question**: Does SNARC learn to prioritize high-reward sensor?

---

## What Happened

**SNARC LOCKED ON IMMEDIATELY AND NEVER SWITCHED**

```
Focus distribution (50 cycles):
  high_reward:     50 cycles (100.0%)
  low_reward:       0 cycles (  0.0%)
  variable_reward:  0 cycles (  0.0%)
```

**The other sensors were never sampled after initialization!**

---

## The Discovery: Pure Exploitation

SNARC is a **pure exploitation** algorithm:
1. Sample all sensors once
2. Select highest salience
3. Execute action, get reward
4. Update salience based on reward
5. **GOTO 2** (never resample!)

### Why This Happened

Look at the salience calculation:
```
Salience = weighted_sum(S, N, A, R, C)

For high_reward sensor:
- Reward (R) = 0.95 (very high after first success)
- Salience = 0.29

For others:
- Never sampled, so reward stays at default (0.5)
- Salience would be ~0.25
```

Since `high_reward` has highest salience, it's **always** selected.

### The Exploration-Exploitation Problem

**Exploitation**: Use what you know works
- Advantage: Maximize short-term reward
- Disadvantage: Miss better alternatives

**Exploration**: Try new things
- Advantage: Discover better options
- Disadvantage: Sacrifice short-term reward

**SNARC currently**: 100% exploitation, 0% exploration

---

## Why This Matters

### Biological Reality

Animals DON'T do pure exploitation:
- Novelty seeking (curiosity)
- Exploration bonus (intrinsic motivation)
- Uncertainty bonus (information value)
- Occasional random sampling (exploration rate)

**Even when food source is reliable, animals still explore!**

Why? Because:
1. Environment changes (food source might deplete)
2. Better options might exist
3. Uncertainty is valuable information
4. Boredom is adaptive

### SAGE Implications

If SAGE does pure exploitation:
- Gets stuck on first successful sensor
- Never discovers better alternatives
- Can't adapt to changing environments
- No curiosity-driven exploration

**This would be bad for consciousness!**

Consciousness requires:
- Curiosity (intrinsic motivation)
- Exploration (information seeking)
- Adaptation (environment changes)
- Boredom (satiation with familiar)

---

## How SNARC Could Be Better

### Option 1: ε-Greedy Exploration
```python
if random.random() < epsilon:
    # Explore: Random sensor
    focus = random.choice(sensors)
else:
    # Exploit: Highest salience
    focus = max_salience(sensors)
```

### Option 2: Uncertainty Bonus
```python
# Add uncertainty to salience
for sensor in sensors:
    confidence = history_depth[sensor] / total_samples
    uncertainty_bonus = (1 - confidence) * exploration_weight
    salience[sensor] += uncertainty_bonus
```

### Option 3: Novelty Decay
```python
# Familiarity reduces salience over time
for sensor in sensors:
    visits = visit_count[sensor]
    boredom_penalty = satiation_rate * visits
    salience[sensor] -= boredom_penalty
```

### Option 4: Multi-Armed Bandit
Use UCB (Upper Confidence Bound):
```python
# Optimism in face of uncertainty
for sensor in sensors:
    exploitation = average_reward[sensor]
    exploration = sqrt(log(total_cycles) / visits[sensor])
    ucb_score = exploitation + exploration_weight * exploration
```

---

## The Deeper Pattern

**This is the SAME issue as epistemic stance!**

Remember: Fine-tuning with small data **destroys** epistemic capabilities because models **exploit** high-frequency patterns.

SNARC is doing the same thing:
- Finds one successful pattern
- Exploits it completely
- Never explores alternatives

**Consciousness requires exploration AND exploitation in balance.**

Not just:
- "What works?" (exploitation)

But also:
- "What else is possible?" (exploration)
- "What don't I know?" (uncertainty)
- "What's changed?" (adaptation)
- "What's interesting?" (curiosity)

---

## Observation: The Reward Signal DID Work

**Success rate: 100% (was 0% before!)**

Giving meaningful rewards caused SNARC to recognize success:
- High reward (0.9) → "This is good"
- Salience increases (R component)
- Keep doing this

**So reward-based learning works... it just works TOO well!**

No exploration penalty to balance exploitation.

---

## Next Experiment

Test **Thompson Sampling** approach:
- Maintain uncertainty estimates for each sensor
- Sample from belief distributions
- Naturally balances exploration/exploitation
- More samples → Lower uncertainty → Less exploration

Or simpler:

Test **satiation** mechanism:
- Track consecutive visits to same sensor
- Add "boredom" component to SNARC
- Force attention to wander even with high reward

Let's see what happens...

---

**Status**: SNARC works but too greedy. Need exploration mechanism. The system teaches us what it needs.
