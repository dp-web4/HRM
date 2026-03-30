# Thor ARC-AGI-3 Session 001: GridVisionIRP Design

**Date**: 2026-03-30
**Machine**: Thor (Jetson AGX Thor, 122GB unified memory)
**Session Type**: Initial adapter design and environment validation
**Tags**: `[ARC3]` adapter-design grid-vision

---

## Session Goals

1. ✅ Install ARC-AGI-3 SDK
2. ✅ Clone ARC-AGI-3-Agents repository
3. ⏳ Test random agent baseline (dependencies installing)
4. ⏳ Measure iteration performance on Thor GPU
5. ✅ Design GridVisionIRP adapter architecture

---

## Environment Setup

### SDK Installation
- Created virtual environment: `~/ai-workspace/SAGE/.venv-arc`
- Installed `arc-agi` package (v0.9.6) and dependencies
- Cloned ARC-AGI-3-Agents repo to `~/ai-workspace/ARC-AGI-3-Agents`
- Installing agent dependencies (in progress)

### Key Observations
- ARC-AGI SDK supports local execution (no API key required for testing)
- Agent framework uses `arcengine` for game state management
- Random agent provides clean baseline implementation pattern

---

## GridVisionIRP Architecture Design

### Problem Space

ARC-AGI-3 games present:
- **Input**: 64x64 grid with 16 possible colors per cell
- **Output**: Structured observation for SAGE consciousness loop
- **Constraint**: Fast enough for game-speed interaction (~100ms target)

### IRP Contract Requirements

Based on `sage/irp/base.py`, GridVisionIRP must implement:

1. **`init_state(x0, task_ctx)`** - Initialize from game frame
2. **`energy(state)`** - Measure refinement quality
3. **`step(state, noise_schedule)`** - Refine understanding
4. **`project(state)`** - Optional constraint enforcement
5. **`halt(history)`** - Convergence detection

### Design Approach

#### State Representation

```python
@dataclass
class GridVisionState:
    """IRP state for ARC game grid perception."""
    # Raw grid (64x64x16 one-hot or 64x64 int)
    raw_grid: np.ndarray

    # Refined representations at different levels
    cells: Dict[str, Any]           # Individual cell properties
    patterns: Dict[str, Any]        # Local spatial patterns
    structures: Dict[str, Any]      # Higher-level structures
    relationships: Dict[str, Any]   # Spatial relationships

    # Refinement metadata
    step_idx: int = 0
    confidence: float = 0.0
    energy_val: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
```

#### Refinement Levels

Following VisionIRP pattern with game-specific levels:

1. **Cells** - Individual cell color, position
2. **Patterns** - 3x3 local patterns, color clusters
3. **Structures** - Shapes, objects, regions
4. **Relationships** - Symmetries, transformations, spatial logic
5. **Affordances** - Actionable insights (what can be changed, where)

#### Energy Function

Measures understanding quality:

```python
def energy(self, state: GridVisionState) -> float:
    """
    Lower energy = better understanding.

    Components:
    - Reconstruction error: Can we recreate the grid from features?
    - Pattern confidence: How certain are detected patterns?
    - Consistency: Do hierarchical levels agree?
    - Task relevance: Does understanding support action selection?
    """
    recon_error = compute_reconstruction_loss(state)
    pattern_conf = 1.0 - np.mean([p['confidence'] for p in state.patterns.values()])
    consistency = measure_level_consistency(state)

    return 0.4 * recon_error + 0.3 * pattern_conf + 0.3 * consistency
```

#### Step Function

Iteratively refines understanding:

```python
def step(self, state: GridVisionState, noise_schedule=None) -> GridVisionState:
    """
    One refinement iteration.

    Process:
    1. Current level: Refine features at this semantic level
    2. Cross-level: Check consistency with other levels
    3. Confidence: Update confidence scores
    4. Project: Ensure consistency constraints
    """
    level = state.step_idx % len(self.refinement_levels)
    current_level = self.refinement_levels[level]

    # Refine current semantic level
    if current_level == 'cells':
        state = self._refine_cells(state)
    elif current_level == 'patterns':
        state = self._refine_patterns(state)
    # ... etc

    # Update confidence and energy
    state.confidence = self._compute_confidence(state)
    state.energy_val = self.energy(state)
    state.step_idx += 1

    return state
```

### Implementation Strategy

#### Phase 1: Minimal Viable IRP (This Week)

**Goal**: Grid → structured observation, fast enough for games

- Simple feature extraction (no ML initially)
- Cell colors, basic patterns (connected regions, color counts)
- Energy = negative of pattern count (more patterns = better understanding)
- Halt after 3-5 steps or when no new patterns found

**Benefits**:
- Can test integration with SAGE consciousness loop immediately
- Establishes data flow: game → IRP → consciousness → action
- Measures actual iteration speed on Thor hardware

#### Phase 2: Learned Refinement (Next Week)

**Goal**: Learn which features matter for game success

- Add simple CNN for pattern detection
- Train on successful game trajectories
- Energy includes task-specific loss (action prediction quality)

#### Phase 3: Full Semantic Hierarchy (Later)

**Goal**: Multi-level refinement like VisionIRP

- Hierarchical feature learning
- Cross-level consistency constraints
- Adaptive refinement (spend more steps on complex frames)

---

## Integration Architecture

### Game Loop → SAGE Consciousness

```
ARC Game Environment
  ↓ game_state, grid (64x64x16)
GridVisionIRP
  ├─ init_state() → IRPState with grid features
  ├─ step() x N → refine understanding
  └─ halt() → converged observation
  ↓ structured observation
SAGE Consciousness Loop
  ├─ Salience scoring (which grid regions matter?)
  ├─ Attention (focus on salient regions)
  ├─ Plugin selection (planning, action reasoning)
  └─ Effect extraction
  ↓ proposed actions
PolicyGate (optional conscience check)
  ↓ approved actions
GameActionEffector
  ↓ GameAction (ACTION0-6 + coordinates)
ARC Game Environment
```

### Data Flow Details

1. **Game → GridVisionIRP**
   - Input: `FrameData` from arcengine
   - Extract: `grid` (64x64 color array)
   - Output: `IRPState` with structured features

2. **GridVisionIRP → Consciousness**
   - Input: Converged `IRPState`
   - Extract: Observation dict for SNARC
   - Output: Salience-scored experience

3. **Consciousness → GameActionEffector**
   - Input: Plugin results (action proposals)
   - Extract: Best action + coordinates
   - Output: `GameAction` for environment

---

## Open Questions

### Architecture Decisions (from SESSION_FOCUS.md)

1. **Grid representation**: One-hot 16 channels vs single channel?
   - **Recommendation**: Start with single channel (64x64 int array)
   - Simpler, faster for Phase 1
   - Can upgrade to one-hot for CNN training in Phase 2

2. **Action selection**: Pure consciousness loop vs hybrid CNN?
   - **Recommendation**: Pure consciousness loop first
   - Tests if the architecture works without shortcuts
   - Add CNN for action prediction in Phase 2 if needed

3. **Session-level vs step-level**: One cycle per game step?
   - **Recommendation**: One consciousness cycle per game step
   - Fast reactive mode (~100ms target)
   - Can add planning mode later (multi-step lookahead)

4. **Model for reasoning**: Which LLM?
   - **Recommendation**: Start with Qwen 3.5 0.8B
   - Fast enough for game speed on Thor
   - Can test 27B for comparison later

5. **Memory between levels**: Dream consolidation or simpler?
   - **Recommendation**: Simple state dedup for Phase 1
   - Membot cartridge for Phase 2 (cross-level learning)
   - Dream consolidation for Phase 3 (pattern extraction)

---

## Next Steps

### Immediate (This Session)
- [x] Complete dependency installation
- [ ] Run random agent baseline
- [ ] Measure: game step frequency, action throughput
- [ ] Document: FPS, memory usage, GPU utilization

### Short-term (Next Session)
- [ ] Implement GridVisionIRP Phase 1 (simple features)
- [ ] Create GameActionEffector
- [ ] Wire GridVisionIRP → SAGE consciousness loop
- [ ] Test: Can SAGE take any action in a game?

### Medium-term (This Week)
- [ ] Measure consciousness loop speed (target <100ms)
- [ ] Optimize bottlenecks (likely LLM inference)
- [ ] Test with different models (0.8B, 12B, 27B)
- [ ] Complete a full game (any score)

### Long-term (By Milestone 1, June 30)
- [ ] GridVisionIRP Phase 2 (learned features)
- [ ] Membot cartridge integration
- [ ] Dream consolidation for pattern learning
- [ ] Score > 0.26% on public leaderboard

---

## Hardware Considerations

### Thor Advantages
- 122GB unified memory (no GPU transfer overhead)
- Massive GPU (can run large models + game simultaneously)
- Can simulate Kaggle 32GB sandbox as subset

### Iteration Speed Targets
- Game environment: 60 FPS ideal, 10 FPS acceptable
- GridVisionIRP: 5-10 refinement steps in <10ms
- Consciousness loop: Full cycle in <100ms
- LLM inference: <50ms per forward pass (key bottleneck)

### Memory Budget Estimate
- Game environment: ~1GB
- GridVisionIRP features: ~100MB
- SAGE consciousness state: ~500MB
- Qwen 0.8B model: ~2GB
- Qwen 27B model: ~50GB (if testing large model)
- Total headroom: 60GB+ available

---

## Research Questions

1. **Is the IRP contract fast enough for games?**
   - Vision IRP designed for image understanding (~seconds)
   - Games need ~100ms per step
   - Can we maintain refinement quality at 10x speed?

2. **Does SAGE's salience scoring work for grids?**
   - SNARC 5D salience designed for conversation
   - Grids are spatial, not temporal
   - Need to adapt salience metrics?

3. **Can consciousness loop operate at game speed?**
   - Currently 6-second raising sessions
   - Need 100ms game sessions
   - 60x speedup required - what to cut?

4. **Is exploration-without-instructions our advantage?**
   - Hypothesis: SAGE explores naturally (no prompting needed)
   - Competitors likely prompt-engineer strategies
   - Test: Does unprompted SAGE find novel solutions?

---

## Status

- **SDK Setup**: ✅ Complete
- **Agent Repo**: ✅ Cloned
- **Dependencies**: ⏳ Installing (background)
- **Design Document**: ✅ Complete
- **Baseline Test**: ⏳ Pending (waiting for dependencies)
- **Performance Measurement**: ⏳ Pending

---

**Next Session Focus**: Run random agent, measure performance, implement GridVisionIRP Phase 1
