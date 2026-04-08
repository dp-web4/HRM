# SAGE Solver Architecture — v7 and v9

## Overview

SAGE solvers for ARC-AGI-3 have evolved through multiple iterations, with v7 as the current **fleet standard** (text-based) and v9 as the **multimodal extension** (vision-enabled with live visualization).

## Solver Versions

### v7 — Fleet Standard (Text-Based)
- **File**: `sage_solver_v7.py`
- **Model**: gemma4:e4b (4B vision model, text-only prompts)
- **Architecture**: Membot integration, FederatedKnowledge, ContextConstructor
- **Status**: Production fleet standard, running on all 6 machines
- **Key features**:
  - Deep membot integration for knowledge retrieval
  - 4-layer context construction (L1: narrative, L2: game KB, L3: membot, L4: metacognitive)
  - Federation-wide knowledge sharing
  - Systematic discovery phase (22-step probe)
  - Cross-level learning with level-up knowledge storage

### v9 — Multimodal Vision + Viewer Integration
- **File**: `sage_solver_v9.py`
- **Model**: gemma4:e4b (4B vision model, multimodal prompts)
- **Architecture**: Extends v7 with PNG grid vision + SessionWriter for live visualization
- **Status**: Thor's experimental multimodal solver, viewer integration complete (2026-04-07)
- **Key features**:
  - Everything from v7 (membot, federation, context layers)
  - **Grid PNG images** sent to gemma4:e4b's vision system
  - **SessionWriter integration** for live visualization via `game_viewer.py`
  - Real-time session state tracking (`/tmp/claude_solver/session.json`)
  - Grid snapshots saved as `.npy` files for viewer

## Architecture Details

### Context Construction (Both v7 and v9)

All solvers use a 4-layer context budget system:

```python
L1_narrative: Action history, observations, recent events
L2_game_kb: Structured game knowledge (interactive objects, mechanics)
L3_membot: Federated knowledge from other machines/games
L4_metacognitive: High-level patterns and learning insights
```

Token budget dynamically allocated across layers to stay under LLM context limits (131072 tokens for gemma4:e4b).

### Discovery Phase (Both v7 and v9)

Systematic 22-step probe at game start:
1. Click each interactive object once
2. Try each directional action (UP, DOWN, LEFT, RIGHT) if available
3. Record:
   - Which actions have effects
   - Pixel changes (magnitude)
   - Success rates
   - State changes

**Key principle**: Discovery before strategy (per GAME_SOLVING_PRINCIPLES.md)

### Vision Architecture (v9 Only)

```python
# Grid converted to PNG image
grid_png = grid_to_png(grid, scale=10)  # 64x64 → 640x640 PNG

# Sent to gemma4:e4b as image + text prompt
prompt = {
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": f"data:image/png;base64,{grid_base64}"},
        {"type": "text", "text": game_prompt}
    ]
}
```

Vision model analyzes:
- Spatial relationships between UI elements
- Visual grouping (containers, borders, connected regions)
- State changes (color shifts, object movements)
- Structural roles (buttons vs stamps vs goals)

### Viewer Integration (v9 Only)

Live visualization via SessionWriter:

```python
from arc_session_writer import SessionWriter

# Initialize session writer
session_writer = SessionWriter(
    game_id=game_id,
    win_levels=fd.win_levels,
    available_actions=available,
    baseline=baseline,
    player=f"gemma4:{MODEL}"
)

# Record every action
session_writer.record_action(
    action=action_int,
    x=x, y=y,
    observation=event.observation,
    grid=grid  # Current grid state
)

# Record level-ups
session_writer.record_level_up(
    new_level=fd.levels_completed,
    winning_actions=winning_actions
)

# Record game end
session_writer.record_game_end(fd.state.name)  # WON/LOST/GAME_OVER
```

**Output files**:
- `/tmp/claude_solver/session.json` — Game state, actions log, observations
- `/tmp/claude_solver/current_grid.npy` — Current grid snapshot
- `/tmp/claude_solver/previous_grid.npy` — Previous grid snapshot

**Viewer**:
- Run `game_viewer.py` on port 8765
- Real-time visualization of game state, actions, level progress
- Works with both autonomous solvers and interactive play

## Knowledge Sharing

### Federation Architecture

All solvers share knowledge via:

```python
from sage_federated_knowledge import FederatedKnowledge

fk = FederatedKnowledge()
fk.sync()  # Pull from all 6 machines

# Access discoveries and meta-insights
game_kb = fk.get_game_knowledge(game_id)
meta = fk.get_meta_insights()
```

**Fleet machines**:
- localhost (Thor) — 122GB, RTX 3090
- cbp, nomad, mcnugget, legion, sprout — Various edge devices

**Knowledge types**:
- `game_discoveries`: Interactive objects, action effects, game mechanics
- `meta_insights`: Cross-game patterns, failure modes, learning signals

### Membot Integration

Membot provides semantic memory retrieval:

```python
# Query membot for relevant knowledge
membot_context = membot_query(
    game_state=current_state,
    game_type=game_type,
    level=level
)
```

Membot cartridges store:
- Completed game solutions (sb26, vc33, cd82)
- Cross-level patterns
- Mechanic-specific knowledge

## When to Use Which Solver

### Use v7 when:
- Running on fleet machines (standardized, tested, reliable)
- Text-based reasoning is sufficient
- No visualization needed
- Production sweeps across all games

### Use v9 when:
- Spatial/visual reasoning may help (complex layouts, visual patterns)
- Debugging requires live visualization
- Testing on Thor (has GPU for vision model)
- Experimenting with multimodal approaches

## Running Solvers

### v7 (Fleet Standard)
```bash
.venv-arc/bin/python arc-agi-3/experiments/sage_solver_v7.py \
    --all \
    --attempts 5 \
    -v
```

### v9 (Multimodal + Viewer)
```bash
# Terminal 1: Start viewer
python3 arc-agi-3/experiments/game_viewer.py
# Viewer on http://localhost:8765

# Terminal 2: Run v9 solver
.venv-arc/bin/python arc-agi-3/experiments/sage_solver_v9.py \
    --game dc22 \
    --attempts 3 \
    -v
```

## Key Files

```
arc-agi-3/experiments/
├── sage_solver_v7.py              # Fleet standard (text-based)
├── sage_solver_v9.py              # Multimodal vision + viewer
├── arc_session_writer.py          # Viewer integration module
├── game_viewer.py                 # Live visualization web app
├── sage_federated_knowledge.py   # Cross-machine knowledge sharing
├── GAME_SOLVING_PRINCIPLES.md    # Learnings from complete solves
└── SAGE_SOLVER_ARCHITECTURE.md   # This file
```

## Evolution Timeline

- **v4**: Basic game runner with qwen2.5:3b
- **v5**: Added membot integration
- **v6**: Federation-wide knowledge sharing
- **v7**: Context layers, systematic discovery, fleet standard (2026-04-06)
- **v9**: Multimodal vision + viewer integration (2026-04-07)

## Next Steps

- **v9 → Fleet Multimodal Standard**: Once v9 proves stable, promote to fleet
- **Vision evaluation**: Compare v9 vs v7 on visually complex games
- **Viewer enhancements**: Add heatmaps, attention visualization, pattern highlighting
- **Cross-solver learning**: Feed v9 visual insights back into v7's text reasoning

---

**Recommended reading**:
- `GAME_SOLVING_PRINCIPLES.md` — 6 universal patterns from complete solves (sb26, vc33, cd82)
- `sage_federated_knowledge.py` — Fleet-wide knowledge sharing architecture
- `arc_session_writer.py` — Viewer integration pattern

**Credits**:
- v7: Fleet standard developed across 6 machines
- v9: Thor's multimodal extension with viewer integration from claude_solver.py pattern
- Viewer architecture: Based on claude_solver.py's successful interactive gameplay sessions
