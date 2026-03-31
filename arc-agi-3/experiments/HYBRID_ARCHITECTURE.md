# ARC-AGI-3 Hybrid Architecture Design
## Thor Implementation (March 2026)

## Problem Statement

**Exploration approach (sage_learner.py):**
- ✓ Fast (5,910 steps/sec)
- ✓ Learns action effectiveness
- ✓ Cross-session experience accumulation
- ✗ **No goal detection** (100k steps, 0 levels completed)

**LLM reasoning approach (sage_game_runner_v2.py):**
- ✓ Hypothesis formation
- ✓ Goal awareness
- ✓ Pattern recognition
- ✗ **Too slow** (~24s/action with Qwen 3.5 27B, ~4s with Gemma 3 12B)

**Required:** Combine fast action generation with goal-directed reasoning and cross-session memory.

---

## Hybrid Architecture: "Driver + Navigator"

### Core Concept

```
┌─────────────────────────────────────────────────────────┐
│ NAVIGATOR (LLM - Phi-4 14B or Gemma 3 12B)             │
│  - Runs every 10-20 actions                             │
│  - Assesses: "Are we making progress toward the goal?" │
│  - Generates: Strategic hypothesis + action bias       │
│  - Queries membot for similar patterns                  │
│  - Time budget: <5s per reflection                      │
└─────────────────────────────────────────────────────────┘
                         ↓ (action biases)
┌─────────────────────────────────────────────────────────┐
│ DRIVER (Exploration with learned biases)               │
│  - Fast action loop (5k steps/sec baseline)            │
│  - Biases: LLM guidance + learned effectiveness        │
│  - Detects: Level completion, new mechanics            │
│  - Stores: State→action patterns in experience DB      │
│  - Time budget: <1ms per action                         │
└─────────────────────────────────────────────────────────┘
                         ↓ (winning sequences)
┌─────────────────────────────────────────────────────────┐
│ MEMBOT (Cross-session memory)                          │
│  - Cartridge per game family                           │
│  - Stores: Winning sequences, goal patterns, action    │
│    effectiveness, strategic insights                    │
│  - Loaded at game start, written on level completion   │
└─────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Driver (Fast Action Loop)

**File:** `arc-agi-3/experiments/sage_driver.py` (extends sage_learner.py)

**Action Selection:**
```python
def select_action(state_hash, llm_bias=None):
    # Blend three signals:
    # 1. Learned effectiveness (from experience DB)
    # 2. LLM strategic bias (from Navigator)
    # 3. Exploration noise (decay over time)

    effectiveness_score = get_effectiveness(state_hash, action)
    llm_score = llm_bias.get(action, 0.5) if llm_bias else 0.5

    combined = 0.6 * effectiveness_score + 0.4 * llm_score
    return softmax_sample(combined)
```

**Speed Target:** <1ms per action (10k+ actions/sec without LLM overhead)

**Responsibilities:**
- Execute actions at maximum speed
- Track state→action→outcome
- Detect level completion (frame_data.levels_completed change)
- Trigger Navigator reflection every N actions
- Update experience DB

---

### 2. Navigator (LLM Reflection)

**File:** `arc-agi-3/experiments/sage_navigator.py`

**Reflection Trigger:**
- Every 10-20 actions (adaptive based on progress)
- On level completion
- On 50+ steps with no progress

**Reflection Prompt:**
```
You are analyzing a grid puzzle game. Recent actions and grid changes:

[Last 10 actions with grid diffs]

Current grid: [ASCII representation]

Goal: Complete the level (levels_completed will increase when successful)

Question: What pattern do you notice? What actions should we bias toward?

Respond in JSON:
{
  "hypothesis": "brief pattern description",
  "action_bias": {"UP": 0.8, "SELECT": 0.2, ...},  // sum = 1.0
  "confidence": 0.7,
  "strategy": "exploration" | "exploitation"
}
```

**Model Selection:**
- **Phi-4 14B** if <6s average reflection time (best reasoning)
- **Gemma 3 12B** if Phi-4 >8s (speed matters)

**Time Budget:** Reflection must complete in <5s to maintain overall throughput

---

### 3. Membot Integration

**Cartridge Structure:**
```json
{
  "game_family": "sc25",
  "winning_sequences": [
    {
      "level": 1,
      "actions": ["UP", "UP", "UP", "ACTION6", ...],
      "state_hashes": ["a3f4e2", "b8c9d1", ...],
      "learned": "2026-03-31T16:00:00Z"
    }
  ],
  "goal_patterns": [
    {
      "description": "Fill row with specific color",
      "indicators": ["color 14 disappears", "row becomes uniform"],
      "effective_actions": ["ACTION6", "UP"]
    }
  ],
  "action_effectiveness": {
    "global": {"UP": 0.95, "SELECT": 0.3, ...},
    "state_dependent": {
      "a3f4e2": {"UP": 1.0, "SELECT": 0.1}
    }
  },
  "strategic_insights": [
    "Movement always changes grid",
    "ACTION6 only effective after 3×UP sequence",
    "Level 1 goal: clear columns 62-63"
  ],
  "total_attempts": 127,
  "best_score": {"levels": 0, "steps": 77}
}
```

**Membot Operations:**

**On game start:**
```python
cartridge = membot.read(f"arc-agi-3/{game_family}")
if cartridge:
    driver.load_effectiveness(cartridge["action_effectiveness"])
    navigator.load_insights(cartridge["strategic_insights"])
    if cartridge["winning_sequences"]:
        # Try known winning sequence first (exploitation)
        try_sequence(cartridge["winning_sequences"][0]["actions"])
```

**On level completion:**
```python
cartridge["winning_sequences"].append({
    "level": current_level,
    "actions": recent_actions,
    "state_hashes": recent_states,
    "learned": now()
})
membot.write(f"arc-agi-3/{game_family}", cartridge)
```

**On Navigator reflection:**
```python
if navigator.confidence > 0.8 and hypothesis_is_novel:
    cartridge["strategic_insights"].append(navigator.hypothesis)
    membot.write(f"arc-agi-3/{game_family}", cartridge)
```

---

## Control Flow

### Game Loop

```python
def play_game_hybrid(env, max_steps=10000):
    # Initialize
    driver = SageDriver(available_actions)
    navigator = SageNavigator(model="phi4:14b")  # or gemma3:12b
    cartridge = membot.read(f"arc-agi-3/{game_family}")

    driver.load_experience(cartridge)
    llm_bias = navigator.get_initial_bias(cartridge)

    step = 0
    actions_since_reflection = 0
    levels_before = frame_data.levels_completed

    while step < max_steps:
        # Driver: Fast action loop
        state_hash = grid_hash(grid)
        action = driver.select_action(state_hash, llm_bias)

        frame_data = env.step(action)
        grid = np.array(frame_data.frame)
        changed = not np.array_equal(prev_grid, grid)

        driver.record(action, state_hash, changed)
        step += 1
        actions_since_reflection += 1

        # Navigator: Periodic reflection
        if (actions_since_reflection >= 20 or
            frame_data.levels_completed > levels_before or
            (step > 50 and actions_since_reflection >= 10)):

            reflection = navigator.reflect(
                recent_actions=driver.recent_actions,
                recent_grids=driver.recent_grids,
                current_grid=grid,
                cartridge_insights=cartridge["strategic_insights"]
            )

            llm_bias = reflection["action_bias"]
            actions_since_reflection = 0

            # Update cartridge with new insights
            if reflection["confidence"] > 0.7:
                cartridge["strategic_insights"].append(reflection["hypothesis"])

        # Level completion
        if frame_data.levels_completed > levels_before:
            winning_sequence = driver.recent_actions[-30:]
            cartridge["winning_sequences"].append({
                "level": frame_data.levels_completed,
                "actions": winning_sequence,
                "learned": time.time()
            })
            membot.write(f"arc-agi-3/{game_family}", cartridge)

            levels_before = frame_data.levels_completed
            driver.reset_for_new_level()

    # Save final experience
    driver.save_experience()
    cartridge["total_attempts"] += 1
    cartridge["best_score"] = max(cartridge["best_score"],
                                   {"levels": frame_data.levels_completed, "steps": step})
    membot.write(f"arc-agi-3/{game_family}", cartridge)
```

---

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Actions/sec (without LLM) | 1,000+ | Driver should be fast |
| Navigator reflection time | <5s | Keep overall throughput high |
| Overall throughput | 100+ actions/sec | Reflection every 20 actions = 5s/20 = 0.25s overhead |
| Time to first level | <5 minutes | 10k actions @ 100/s = 100s baseline + learning |

**Comparison to baselines:**
- Pure exploration: 5,910 steps/sec, 0 levels in 100k steps
- Pure LLM: 0.04 steps/sec (24s/action), 0 levels in 77 steps (McNugget)
- **Hybrid target:** 100 steps/sec, first level in <5 minutes

---

## Model Selection Decision Tree

```
Phi-4 14B available?
├─ Yes → Test reflection time
│   ├─ <6s → USE PHI-4 (best reasoning)
│   └─ >8s → USE GEMMA 3 12B (speed matters)
└─ No → USE GEMMA 3 12B (fallback)
```

Once selected, stick with that model for the session to avoid reload overhead.

---

## Success Metrics

**Milestone 1: First level completion**
- Any game, any level
- Proves goal detection works

**Milestone 2: Consistent level completion**
- >50% success rate on first level across 10 games
- Proves learning transfers

**Milestone 3: Multi-level games**
- Complete 2+ levels in a single game
- Proves adaptation to new mechanics

**Milestone 4: Cross-session improvement**
- Second run on same game faster than first
- Proves membot integration works

---

## Implementation Plan

### Phase 1: Core Components (while models download)
- [ ] Write `sage_driver.py` (extend sage_learner.py)
- [ ] Write `sage_navigator.py` (reflection prompts + JSON parsing)
- [ ] Write `membot_cartridge.py` (read/write game cartridges)

### Phase 2: Integration Test (once models ready)
- [ ] Test Phi-4 14B reflection speed (3-action test)
- [ ] Test Gemma 3 12B reflection speed (3-action test)
- [ ] Select model based on <6s criterion

### Phase 3: Hybrid Runner (main implementation)
- [ ] Write `sage_hybrid_runner.py` (combines driver + navigator + membot)
- [ ] Run on single game (tu93 or sc25)
- [ ] Measure: actions/sec, time to first level

### Phase 4: First Score
- [ ] Run hybrid on 5-10 games
- [ ] Track level completions
- [ ] Update SESSION_FOCUS.md with results

---

## Open Questions

1. **Navigator frequency:** 10 vs 20 actions? Adaptive based on confidence?
2. **State hash granularity:** 8×8 downsample enough? Or try 4×4 for finer control?
3. **Membot cartridge size:** Keep last 20 winning sequences or all of them?
4. **Exploitation vs exploration:** Try known winning sequence first, or always explore?
5. **Multi-level reset:** Reset driver stats between levels or keep global knowledge?

These will be answered empirically during Phase 3-4.

---

## Next Steps (Immediate)

While models download (~20-60 min):
1. Implement `sage_driver.py` (fast action loop with dual biasing)
2. Implement `sage_navigator.py` (reflection prompts + model integration)
3. Implement `membot_cartridge.py` (JSON schema + read/write)

Once Phi-4 14B or Gemma 3 12B ready:
4. Test reflection speed on 3 actions
5. Build `sage_hybrid_runner.py`
6. Run first game test
7. **Get first score**

---

*Design completed: 2026-03-31*
*Target machine: Thor (122GB RAM, Qwen 3.5 27B baseline)*
*Objective: Get first level completion with cross-session memory*
