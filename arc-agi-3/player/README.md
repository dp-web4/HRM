# ARC-AGI-3 Competition Player

This directory contains the **fleet-maintained seed identity** for the competition player. Any machine can load it. All machines contribute to it.

## Files

- `seed_identity.json` — Fused identity from 284 fleet raising sessions. Updated by any machine that improves it.
- `game_knowledge/` — Accumulated per-game KB from fleet play (populated during sweeps).

## How to Use

```python
import json
seed = json.load(open("arc-agi-3/player/seed_identity.json"))
# Load into your runner/agent as the player identity
```

## How to Update

Any machine can improve the seed. Push changes to this file. The identity is the fleet's collective, not any single instance.

Things that should be updated:
- `memory_requests` — as the fleet learns what matters for game-playing
- `vocabulary` — as new game-relevant concepts emerge
- `federation.game_knowledge` — as more games are solved
- `federation.key_insight` — as strategic understanding deepens

## Model

Target: `gemma4:e4b` (Google Gemma 4, 9.6GB, thinking model)

Alternatives: any model the fleet has. The identity is model-agnostic — the game knowledge and strategic insights transfer across models.
