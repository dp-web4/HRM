# StochasticGoose Analysis — 1st Place ARC-AGI-3 Preview (12.58%)

**Source**: github.com/DriesSmit/ARC3-solution
**Team**: Tufa Labs (Dries Smit + Jack Cole)
**Score**: 12.58%, 18 levels completed

---

## What It Does

A CNN learns which actions cause frame changes in the game grid. That's it. No world model, no goal inference, no planning, no memory of past states. Pure "does action X at state S cause a change?" binary classification, updated continuously from experience.

## Why It Beat LLMs

LLMs try to reason about unseen game mechanics from text descriptions of grid states. StochasticGoose learns action-state interactions empirically. Given 288K possible steps and a simple binary learning signal (frame changed: yes/no), it discovers which actions matter faster than an LLM can reason about them.

The key insight: **you don't need to understand what the game does. You need to learn what causes state changes and do more of that.**

## Architecture

- **CNN**: 4 conv layers (16→32→64→128→256 channels, all 3x3), two heads:
  - Action head: MaxPool → Dense → 5 logits (ACTION1-5)
  - Coordinate head: 3 more conv layers → 4,096 logits (64×64 grid positions for ACTION6)
- **~2M parameters**, fits easily in GPU memory
- Forward pass: ~100ms on modern GPU

## The Learning Signal

```python
reward = 1.0 if not np.array_equal(prev_frame, current_frame) else 0.0
```

Pixel-exact frame comparison. If any pixel changed, the action is labeled as "useful." Supervised binary cross-entropy loss, trained every 5 actions on batches of 64 from the experience buffer.

## Experience Buffer

- 200K capacity, FIFO eviction
- **MD5 hash deduplication**: only stores unique (state, action) pairs
- This is the critical engineering: with 4,101 possible actions per state, deduplication compresses the experience space dramatically

## Exploration: Stochastic Hierarchical Sampling

Not epsilon-greedy. Not UCB. Sigmoid probabilities on all 4,101 actions, fairly weighted:

```python
coord_probs_scaled = coord_probs / 4096  # Prevent ACTION6 from dominating
all_probs = cat([action_probs, coord_probs_scaled])
all_probs = all_probs / all_probs.sum()
selected = multinomial(all_probs)
```

As the model learns, it naturally shifts from uniform exploration to biased exploitation — actions predicted to cause changes get higher probability.

## Level Reset Strategy

Full model + buffer reset when score changes (new level). Reasoning: new level = new mechanics. Old experience is negative transfer. The agent starts fresh at each level.

**This is a deliberate tradeoff**: throws away potentially useful knowledge to avoid overfitting to previous mechanics. The TODO comment in the code suggests the author considered NOT resetting.

## Where It Succeeds

- Games where "does this action change the frame?" is a useful learning signal
- Games where spatial patterns in the grid predict actionable positions
- Games with clear cause-effect relationships between actions and state changes
- Games where 288K exploration steps is enough to discover the mechanics

## Where It Fails (Inferred)

- **Invisible state changes**: Games with internal counters or hidden state where pixels don't change but game state does
- **Temporal reasoning**: Games requiring memory of past positions or sequences of actions. The agent is fully Markovian — no history.
- **Goal inference**: The agent doesn't know WHAT it's trying to achieve. It maximizes frame changes, which correlates with progress but isn't the same thing.
- **Precise sequencing**: Games requiring specific action orders. Stochastic sampling may never find the right sequence.
- **Subtle mechanics**: Games where the learning signal is too sparse (most actions don't change frames at all).

## What SAGE Can Do Differently

| StochasticGoose | SAGE Approach |
|----------------|---------------|
| Binary frame-change signal | SNARC 5D salience (Surprise, Novelty, Arousal, Reward, Conflict) — richer learning signal |
| No memory of past states | Consciousness loop maintains attention history, membot persists semantic patterns |
| No world model | Trust posture builds an implicit world model (what's reliable, what's uncertain) |
| No goal inference | MRH scoping + salience-driven attention = emergent goal acquisition |
| Per-level reset (forget everything) | Dream consolidation extracts transferable patterns before reset |
| Pure exploration | Pre-commit self-challenge: "what assumption am I not questioning?" |
| 2M param CNN, one signal | IRP plugins: multiple specialized modules orchestrated by consciousness loop |

## Key Takeaways for Our Approach

1. **Frame change detection works.** It's a simple, universal signal. We should use it too — but as ONE of SNARC's input signals, not the only one.

2. **Hash deduplication is essential.** With 4,101 actions per state, the action space explodes. Deduplication makes learning tractable. Our SNARC already has this concept (seen-before set).

3. **Per-level adaptation is important but wasteful.** Resetting everything per level loses knowledge. Dream consolidation before reset would preserve transferable patterns.

4. **Stochastic exploration > deterministic search.** The multinomial sampling naturally transitions from exploration to exploitation as the model learns. No epsilon schedule needed. This is conceptually similar to trust-weighted attention selection.

5. **Small CNN beats large LLM for this task.** 2M parameters, ~200MB GPU memory, ~100ms per step. Our consciousness loop needs to be similarly lightweight for game-speed operation. The heavy model (Gemma 12B) should be used for reasoning at decision points, not for every action.

6. **The 12.58% ceiling is NOT from compute.** It's from the learning signal's limitations (binary frame-change, Markovian, no goals). A richer signal (salience scoring) and richer architecture (consciousness loop with memory) should push past this ceiling.

---

*We don't copy this approach. We learn from what made it work (simple signal, efficient exploration, spatial bias) and address what made it fail (no memory, no goals, no transfer). SAGE's architecture is the answer to StochasticGoose's limitations.*
