# ARC-AGI-3 Environment Specification

**Source**: https://docs.arcprize.org/ (read 2026-03-29)
**SDK**: `pip install arc-agi` (v0.9.3)
**Agent repo**: https://github.com/arcprize/ARC-AGI-3-Agents

---

## SCORING — READ THIS FIRST

We misunderstood ARC-AGI-2 scoring (thought partial pixel matches earned partial credit; exact match was required). Do NOT repeat that mistake.

### RHAE (Relative Human Action Efficiency)

**Per-level:**
```
level_score = min(1.0, human_baseline / ai_actions) ^ 2
```

- **SQUARED** — 50% efficiency = 25% score, NOT 50%
- Capped at 1.0 — no bonus for beating humans
- 2x human actions = 0.25 score
- 3x = 0.11
- 5x = 0.04
- 10x = 0.01
- Unsolved levels = 0 (no partial credit)
- Hard cutoff at **5x human baseline actions per level** — terminated

**Human baseline**: 2nd-best human (fewest actions), not average. From 486 participants in 90-minute first-play sessions.

**Per-game aggregation** — weighted by level index:
```
game_score = sum(level_score[i] * i) / sum(range(1, num_levels+1))
```
Level 5 gets 5x the weight of level 1. Solving tutorials is nearly worthless. Hard levels are everything.

**Final**: Average of all game scores. Range 0-100%.

### Implication for SAGE

**Action efficiency is the ONLY metric that matters.** Every wasted exploration action costs quadratically. The consciousness loop must be extremely selective about what actions to take. "Explore everything" is a losing strategy. "Explore the minimum needed to build a world model, then act precisely" is the winning strategy.

---

## Kaggle Sandbox

| Resource | Spec |
|----------|------|
| GPU | RTX 5090 (32GB GDDR7) |
| Time limit | 8 hours total |
| Internet | **NONE** — no API calls to external models |
| Model weights | YES — upload as Kaggle datasets |
| Custom data | YES — membot cartridges, pre-computed data allowed |
| Packages | Standard Kaggle Python + `pip install` allowed |
| Submission | Kaggle notebook, one-click runnable |

### What this means
- Model must be LOCAL. No Claude, no GPT, no Gemini during evaluation.
- Must fit in 32GB VRAM alongside game environment + vision processing
- Good candidates: Gemma 12B, Phi-4 14B, Qwen 3.5 27B (quantized)
- Membot cartridges CAN be uploaded as Kaggle datasets
- SNARC databases CAN be uploaded
- Pre-computed pattern libraries CAN be uploaded

---

## Interaction Protocol

### Observation
- **64x64 grid**, cells are integers 0-15 (16 colors)
- Response contains `frame` — array of 1-N grids (multi-frame for animations)
- Coordinate system: (0,0) top-left, (x,y) format

### Actions (7 total)
| Action | Type | Typical meaning |
|--------|------|-----------------|
| RESET | Special | Restart level (or new game if no actions taken) |
| ACTION1 | Simple | Up |
| ACTION2 | Simple | Down |
| ACTION3 | Simple | Left |
| ACTION4 | Simple | Right |
| ACTION5 | Simple | Interact (game-specific) |
| ACTION6 | Complex | Click at (x,y) 0-63. Agent must discover valid coordinates. |
| ACTION7 | Simple | Undo — **NOT AVAILABLE in competition** |

Each game defines which actions are available via `env.action_space`.

### Game States
| State | Meaning |
|-------|---------|
| NOT_STARTED | Initial |
| NOT_FINISHED | Active, awaiting action |
| WIN | Level complete |
| GAME_OVER | Terminated (max actions or failure) |

### Game Loop
```python
import arc_agi
from arc_agi import Arcade, GameAction, GameState, OperationMode

arc = Arcade(operation_mode=OperationMode.COMPETITION)
env = arc.make("game_id")
obs = env.reset()

while obs.state != GameState.WIN:
    # SAGE consciousness loop decides action
    action = sage_decide(obs.frame, env.action_space)

    action_data = None
    if action == GameAction.ACTION6:
        x, y = sage_decide_coordinates(obs.frame)
        action_data = {"x": x, "y": y}

    obs = env.step(action, data=action_data, reasoning={"thought": "..."})

    if obs.state == GameState.GAME_OVER:
        obs = env.reset()  # Level reset in competition mode
```

### Environments
- ~135 total (25 public demo + 55 semi-private + 55 fully private)
- 8-10 levels per environment, 5 scored
- Progressive difficulty within each environment
- "Core Knowledge priors" only — no language, numbers, letters, cultural symbols

---

## Competition Mode Constraints

```python
arc = Arcade(operation_mode=OperationMode.COMPETITION)
```

- All environments scored, regardless of what you interact with
- Only level resets permitted (game resets auto-convert)
- Each environment can only be `make()`'d ONCE
- Only ONE scorecard
- Cannot check scorecard mid-run
- Kaggle track forces this mode

---

## CRITICAL GOTCHAS

1. **Session cookies (AWSALB*)** — MUST persist across all API requests or sessions break
2. **Score is SQUARED** — this changes everything. Efficiency > completion.
3. **Level weighting** — later levels worth more. Don't optimize for tutorials.
4. **ACTION7 (undo) NOT available in competition** — don't depend on it
5. **ACTION6 doesn't tell you valid coordinates** — must discover them
6. **Scorecard auto-closes at 15 minutes** — don't leave gaps between actions
7. **Only state-changing actions count** — internal reasoning is free
8. **Two consecutive RESETs = full game restart** — careful with reset logic
9. **~2000 FPS locally** — compute per step is NOT the bottleneck. Reasoning quality IS.
10. **Vision matters enormously** — Claude Opus scores 97% WITH vision, 0% WITHOUT on individual environments. Grid visualization may be critical.

---

## What We Can Bring to Kaggle

| Asset | Upload method | Use |
|-------|--------------|-----|
| Model weights (Gemma/Qwen/Phi) | Kaggle dataset | Local LLM for reasoning |
| Membot cartridges (.npz) | Kaggle dataset | Cross-environment semantic memory |
| SNARC databases | Kaggle dataset | Pattern library from raising sessions |
| Pre-computed game analysis | Kaggle dataset | Knowledge from demo environments |
| Grid pattern library | Kaggle dataset | Common visual patterns pre-encoded |
| SAGE consciousness loop code | Submission notebook | The orchestrator |

---

## Preview Competition Results (for calibration)

| Team | Score | Approach |
|------|-------|---------|
| StochasticGoose | 12.58% | CNN + RL predicting action→frame changes |
| Blind Squirrel | 6.71% | State graphs + ResNet18 value model |
| Fluxonian | 8.04% | DSL + LLM combination |

**What worked**: Detecting which actions cause state changes (frame diffs). State deduplication via hashing. Learning action-consequence mappings.

**What didn't work**: LLM-only (crashed). Random search (fails on well-designed levels). Pure vision-language (insufficient).

---

## Files in This Directory

```
arc-agi-3/
├── README.md              # Strategy and stack mapping
├── ENVIRONMENT.md          # This file — scoring, sandbox, protocol
├── adapters/               # Grid vision + action effector
├── memory/                 # Level-context memory
├── experiments/            # Test results
└── submissions/            # Kaggle packages
```
