# IRP (Iterative Refinement Protocol) Architecture Analysis

**Version**: 1.0
**Date**: 2025-10-12
**Location**: `/home/dp/ai-workspace/HRM/sage/irp/`
**Analysis Type**: Deep dive based on actual codebase

---

## Executive Summary

The IRP (Iterative Refinement Protocol) is a universal computational pattern for intelligence as **iterative denoising toward coherence**. It provides a consistent interface for different cognitive modalities (vision, language, control, memory, audio) to refine noisy/incomplete states into clean/complete states through energy-guided iteration.

**Core Insight**: All intelligence can be modeled as iterative refinement from uncertainty to coherence, with trust emerging from convergence stability.

---

## Table of Contents

1. [Core Abstractions](#core-abstractions)
2. [Plugin Interface Contract](#plugin-interface-contract)
3. [Data Structures](#data-structures)
4. [Data Exchange Mechanisms](#data-exchange-mechanisms)
5. [IRP Cycle Anatomy](#irp-cycle-anatomy)
6. [Energy/Trust/Convergence](#energy-trust-convergence)
7. [Plugin Examples](#plugin-examples)
8. [Orchestration](#orchestration)
9. [Data Flow Diagrams](#data-flow-diagrams)
10. [Integration Patterns](#integration-patterns)

---

## Core Abstractions

### 1. IRPState (Data Container)

**Location**: `/home/dp/ai-workspace/HRM/sage/irp/base.py:15-22`

```python
@dataclass
class IRPState:
    """State container for IRP refinement process."""
    x: Any                      # Plugin-specific state (latent, tokens, trajectory, etc.)
    step_idx: int = 0          # Current iteration index
    energy_val: Optional[float] = None  # Cached energy value
    meta: Dict[str, Any] = field(default_factory=dict)  # Metadata
    timestamp: float = field(default_factory=time.time)  # Creation time
```

**Key Properties**:
- **x**: The actual state being refined (type varies by plugin)
  - Vision: `torch.Tensor` (latent space)
  - Language: `torch.Tensor` (token IDs)
  - Control: `Dict[str, torch.Tensor]` (states + actions)
  - Memory: `torch.Tensor` (memory embeddings)
  - TTS: `TTSState` (custom dataclass with audio)
- **step_idx**: Monotonically increasing iteration counter
- **energy_val**: Cached to avoid recomputation
- **meta**: Plugin-specific metadata (task context, history, etc.)
- **timestamp**: Used for telemetry and timing analysis

### 2. IRPPlugin (Base Interface)

**Location**: `/home/dp/ai-workspace/HRM/sage/irp/base.py:24-286`

The base class defines the universal IRP contract that all plugins must implement.

---

## Plugin Interface Contract

Every IRP plugin must define **four invariants**:

1. **State Space**: The representation being refined
2. **Noise Model**: How uncertainty is represented
3. **Energy/Distance Metric**: Measure of refinement progress
4. **Coherence Contribution**: Impact on system-level coherence

### Required Methods

#### 1. `init_state(x0, task_ctx) -> IRPState`

**Purpose**: Initialize refinement state from raw input

**Contract**:
- Takes raw input `x0` (image, text, trajectory spec, etc.)
- Takes `task_ctx` (dict with objectives, constraints, hints)
- Returns `IRPState` with initial state in plugin's representation

**Example (Vision)**:
```python
def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
    # Convert to tensor
    if isinstance(x0, np.ndarray):
        x0 = torch.from_numpy(x0).float()

    # Encode to latent space
    latent = self.encoder(x0)

    # Store original and context
    meta = {
        'original_image': x0,
        'task_ctx': task_ctx,
        'current_level': 0,
        'confidence_history': []
    }

    return IRPState(x=latent, step_idx=0, meta=meta)
```

#### 2. `energy(state) -> float`

**Purpose**: Compute energy/distance metric for current state

**Contract**:
- Takes `IRPState`
- Returns scalar float (lower = better refinement)
- Must be fast (called frequently)
- Should be differentiable (for optimization)

**Common Patterns**:
- **Reconstruction loss**: Distance from target/original
- **Task loss**: Objective-specific metric (classification confidence, etc.)
- **Constraint violation**: Penalty for infeasibility
- **Perplexity**: Prediction uncertainty

**Example (Language)**:
```python
def energy(self, state: IRPState) -> float:
    tokens = state.x
    embeddings = self.embedder(tokens)

    # Pass through denoiser
    denoised = self.denoiser(embeddings)

    # Perplexity as energy
    original_embeddings = self.embedder(state.meta['original_tokens'])
    perplexity = nn.functional.mse_loss(denoised, original_embeddings).item()

    return float(np.log(perplexity + 1e-6))
```

#### 3. `step(state, noise_schedule=None) -> IRPState`

**Purpose**: Execute one refinement iteration

**Contract**:
- Takes current `IRPState`
- Returns new `IRPState` with refined state
- Should move toward lower energy
- May accept noise schedule for diffusion

**Common Patterns**:
- **Gradient descent**: Move in direction of energy reduction
- **Denoising**: Remove noise gradually
- **Constraint projection**: Apply corrections
- **Progressive refinement**: Move through semantic levels

**Example (Control)**:
```python
def step(self, state: IRPState, noise_schedule: Any = None) -> IRPState:
    # Get refinement gradient from network
    action_update = self.refiner(trajectory_flat)

    # Apply update with learning rate
    lr = self.lr_schedule[step_idx]
    new_actions = actions - lr * action_update

    # Add exploration noise (decreasing)
    noise_scale = 0.1 * (1.0 - step_idx / max_steps)
    new_actions = new_actions + torch.randn_like(new_actions) * noise_scale

    return IRPState(x=new_trajectory, step_idx=step_idx + 1, meta=state.meta)
```

### Optional Methods

#### 4. `project(state) -> IRPState`

**Purpose**: Enforce constraints on state

**Default**: Pass-through (no constraints)

**Use Cases**:
- Dynamics feasibility (control)
- Bound constraints (state/action limits)
- Obstacle avoidance
- Latent range clamping

**Example (Control)**:
```python
def project(self, state: IRPState) -> IRPState:
    trajectory = state.x
    actions = trajectory['actions']

    # Project actions to bounds
    actions = torch.clamp(actions, -1.0, 1.0)

    # Forward simulate to ensure dynamics consistency
    projected_states = torch.zeros_like(states)
    projected_states[:, 0] = start

    for t in range(horizon - 1):
        next_state = self.dynamics_model(projected_states[:, t], actions[:, t])
        projected_states[:, t + 1] = next_state

    state.x = {'states': projected_states, 'actions': actions}
    return state
```

#### 5. `halt(history) -> bool`

**Purpose**: Determine if refinement should stop

**Default Implementation** (in base class):
- Energy slope < epsilon for K consecutive steps
- Maximum iterations reached

**Common Extensions**:
- Task-specific confidence threshold
- Feasibility achieved
- All constraints satisfied
- Perplexity stabilized

**Example (Vision)**:
```python
def halt(self, history: List[IRPState]) -> bool:
    # Check base halting (energy slope)
    if super().halt(history):
        return True

    # Check task confidence
    if history:
        latest_confidence = history[-1].meta['confidence_history'][-1]
        if latest_confidence >= self.confidence_threshold:
            return True  # Early stop on high confidence

    return False
```

#### 6. `emit_telemetry(state, history) -> Dict`

**Purpose**: Generate telemetry for monitoring/Web4 integration

**Default Implementation**: Provides standard telemetry

**Output Format**:
```json
{
  "entity_id": "vision_irp_v1",
  "plugin": "vision",
  "step_idx": 17,
  "E": 0.482,
  "dE": -0.0123,
  "steps": 18,
  "halt_reason": "slope<eps",
  "trust": {
    "monotonicity_ratio": 0.93,
    "dE_variance": 0.004,
    "convergence_rate": 0.021
  },
  "budget": {
    "ATP_spent": 1.7,
    "time_ms": 43.2,
    "memory_mb": 12.5
  },
  "LRC_context": null
}
```

---

## Data Structures

### 1. IRPState Flow

```
Raw Input (x0)
    ↓
init_state()
    ↓
IRPState {
    x: Plugin-specific representation,
    step_idx: 0,
    energy_val: None,
    meta: {task_ctx, ...}
}
    ↓
step() → IRPState {x: refined, step_idx: 1, ...}
    ↓
step() → IRPState {x: more_refined, step_idx: 2, ...}
    ↓
... (until halt())
    ↓
Final IRPState
    ↓
extract() → Output
```

### 2. Plugin-Specific State Types

#### Vision: Latent Tensor
```python
x: torch.Tensor  # Shape: [B, latent_dim]
meta: {
    'original_image': torch.Tensor,
    'current_level': str,  # 'edges', 'textures', 'objects', etc.
    'confidence_history': List[float]
}
```

#### Language: Token IDs + Masks
```python
x: torch.Tensor  # Shape: [B, seq_len], dtype=torch.long
meta: {
    'original_tokens': torch.Tensor,
    'masked_tokens': torch.Tensor,
    'mask_positions': torch.BoolTensor,
    'refinement_level': str,  # 'surface', 'syntactic', 'semantic'
    'perplexity_history': List[float]
}
```

#### Control: Trajectory Dictionary
```python
x: Dict[str, torch.Tensor] = {
    'states': torch.Tensor,   # [B, horizon, state_dim]
    'actions': torch.Tensor   # [B, horizon, action_dim]
}
meta: {
    'start_state': torch.Tensor,
    'goal_state': torch.Tensor,
    'cost_history': List[float],
    'feasibility_history': List[bool]
}
```

#### Memory: Hierarchical Embeddings
```python
x: torch.Tensor  # Shape varies by abstraction level
meta: {
    'raw_experiences': List[Dict],
    'current_level': str,  # 'episodic', 'semantic', 'procedural', etc.
    'compression_history': List[float],
    'retrieval_accuracy_history': List[float]
}
```

#### TTS: Audio State
```python
x: TTSState = dataclass {
    text: str,
    ref_audio: Optional[np.ndarray],
    audio_waveform: Optional[np.ndarray],
    prosody_params: Dict[str, float],
    iteration: int,
    confidence: float
}
meta: {
    'task': 'tts',
    'voice_id': str
}
```

---

## Data Exchange Mechanisms

### 1. Between Plugin and Orchestrator

**Interface**: Plugins don't communicate directly. Orchestrator mediates.

```python
# Orchestrator calls plugin
result: PluginResult = orchestrator.run_plugin(
    plugin_name='vision',
    plugin=vision_plugin,
    input_data=image,
    budget=10.0  # ATP budget
)

# PluginResult contains:
{
    'plugin_name': 'vision',
    'final_state': IRPState,
    'history': List[IRPState],
    'telemetry': Dict,
    'budget_used': float,
    'execution_time': float
}
```

### 2. Between Plugins (via Orchestrator)

Plugins exchange data through the orchestrator's `integrate_results()` method:

```python
# Plugin outputs extracted
vision_output = vision.get_semantic_representation(final_state)
language_output = language.get_understanding(final_state)

# Orchestrator integrates
integrated = {
    'plugin_outputs': {
        'vision': vision_output,
        'language': language_output
    },
    'system_coherence': computed_from_energies,
    'total_energy': sum_of_plugin_energies
}
```

**Key Point**: Plugins are **loosely coupled**. They don't depend on each other's internals, only on the orchestrator's integration.

### 3. Cross-Modal Information Flow

**Future Enhancement** (mentioned in README):
> "Cross-Modal Refinement: Share information between plugins"

**Current Pattern**: Via memory layers
```python
# Plugin A stores result in meta
state.meta['shared_latent'] = latent_vector

# Plugin B accesses via orchestrator context
task_ctx['vision_latent'] = orchestrator.get_plugin_output('vision', 'latent')
```

---

## IRP Cycle Anatomy

An IRP "cycle" consists of:

### 1. Initialization Phase
```python
state = plugin.init_state(x0, task_ctx)
state.energy_val = plugin.energy(state)
history = [state]
```

### 2. Refinement Loop
```python
for step in range(max_iterations):
    if plugin.halt(history):
        break  # Converged!

    # One refinement step
    state = plugin.step(state, noise_schedule=None)

    # Apply constraints
    state = plugin.project(state)

    # Update tracking
    state.step_idx = step + 1
    state.energy_val = plugin.energy(state)
    state.timestamp = time.time()

    history.append(state)
```

### 3. Extraction Phase
```python
final_state = history[-1]
output = plugin.extract(final_state)  # Plugin-specific
telemetry = plugin.emit_telemetry(final_state, history)
```

### Complete Cycle Visualization

```
┌─────────────────────────────────────────────────────────┐
│                    IRP CYCLE                             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. INIT: x0 → init_state() → state₀                    │
│            ↓                                             │
│  2. LOOP:  state → step() → project() → energy()        │
│            ↓                                             │
│         state₁, state₂, ..., stateₙ                     │
│            ↓                                             │
│  3. HALT:  energy slope < ε  OR  confidence > θ         │
│            ↓                                             │
│  4. EXTRACT: stateₙ → extract() → output                │
│                                                          │
└─────────────────────────────────────────────────────────┘

Timeline:
t=0    t=dt   t=2dt        t=n·dt
│      │      │            │
state₀ state₁ state₂  ...  stateₙ
E₀=1.0 E₁=0.7 E₂=0.5  ...  Eₙ=0.1  (converged!)
```

---

## Energy/Trust/Convergence

### Energy Function Philosophy

**Core Principle**: Energy measures distance from ideal state
- Lower energy = better refinement
- Energy decrease = progress
- Energy plateau = convergence

### Trust Metrics

**Trust emerges from convergence behavior**:

```python
def compute_trust_metrics(self, history: List[IRPState]) -> Dict[str, float]:
    energies = [s.energy_val for s in history]
    dE_values = [energies[i+1] - energies[i] for i in range(len(energies)-1)]

    # 1. Monotonicity: Fraction of steps with energy decrease
    monotonic_steps = sum(1 for dE in dE_values if dE < 0)
    monotonicity = monotonic_steps / len(dE_values)

    # 2. Variance: Stability of energy changes
    dE_variance = np.var(dE_values)

    # 3. Convergence rate: Average energy decrease per step
    convergence_rate = (energies[0] - energies[-1]) / len(history)

    return {
        'monotonicity_ratio': monotonicity,    # High = trustworthy
        'dE_variance': dE_variance,            # Low = stable
        'convergence_rate': convergence_rate   # High = efficient
    }
```

**Trust Interpretation**:
- **High monotonicity (>0.9)**: Reliable convergence, high trust
- **Low dE variance (<0.01)**: Stable refinement, predictable
- **High convergence rate**: Efficient, reaches goal quickly

**Trust-Based Resource Allocation** (in orchestrator):
```python
def allocate_budgets(self, available_ATP: float) -> Dict[str, float]:
    total_trust = sum(self.trust_weights.values())

    budgets = {}
    for name, plugin in self.plugins.items():
        weight = self.trust_weights[name] / total_trust
        budgets[name] = available_ATP * weight

    return budgets
```

### Convergence Detection

**Default Mechanism** (from base class):
```python
def halt(self, history: List[IRPState]) -> bool:
    eps = self.config.get('halt_eps', 1e-4)
    K = self.config.get('halt_K', 3)
    max_iter = self.config.get('max_iterations', 100)

    # Maximum iterations reached
    if len(history) >= max_iter:
        return True

    # Not enough history yet
    if len(history) < K + 1:
        return False

    # Check energy slope over last K steps
    recent_energies = [s.energy_val for s in history[-(K+1):]]
    slope = abs(recent_energies[-1] - recent_energies[0]) / len(recent_energies)

    return slope < eps  # Converged if slope flat
```

**Halt Reasons**:
- `"slope<eps"`: Energy converged
- `"max_steps"`: Iteration limit
- `"confidence"`: Task-specific threshold (plugin override)
- `"feasible"`: Constraint satisfaction (control)
- `"unmasked"`: All tokens revealed (language understanding)

---

## Plugin Examples

### Example 1: Vision IRP

**File**: `/home/dp/ai-workspace/HRM/sage/irp/vision.py`

**State Space**: Learned latent space (not pixel space!)

**Noise Model**: Gaussian noise in latent dimensions

**Energy**: Reconstruction loss + task loss
```python
energy = mse_loss(reconstructed, original) + task_weight * (-confidence)
```

**Refinement Levels** (progressive semantic understanding):
1. `edges` - Low-level features
2. `textures` - Surface properties
3. `objects` - Object detection
4. `relationships` - Spatial relationships
5. `affordances` - Action possibilities
6. `meaning` - Semantic understanding

**Key Feature**: Early stops when task confidence exceeds threshold

**Example Usage**:
```python
vision = VisionIRP({
    'latent_dim': 256,
    'max_iterations': 50,
    'halt_eps': 1e-4,
    'confidence_threshold': 0.95
})

image = torch.randn(3, 224, 224)
final_state, history = vision.refine(image, {'target': 'objects'})
results = vision.get_semantic_representation(final_state)
# → {'level': 'objects', 'confidence': 0.96, 'predictions': [...]}
```

### Example 2: Language IRP

**File**: `/home/dp/ai-workspace/HRM/sage/irp/language.py`

**State Space**: Token IDs with mask positions

**Noise Model**: Masked spans (span-based masking)

**Energy**: Log perplexity
```python
energy = log(mse_loss(denoised_embeddings, original_embeddings) + 1e-6)
```

**Refinement Process**:
1. **Surface** (steps 0-16): Unmask based on local context
2. **Syntactic** (steps 17-33): Grammar-aware unmasking
3. **Semantic** (steps 34-50): Meaning-driven unmasking

**Progressive Unmasking**:
```python
# Predict tokens at masked positions
predicted_tokens = argmax(model(embeddings))

# Unmask high-confidence predictions
high_conf_mask = (confidence > 0.7) & mask_positions
new_tokens[high_conf_mask] = predicted_tokens[high_conf_mask]
```

**Key Feature**: Halts when all masks removed (understanding mode)

**Example Usage**:
```python
language = LanguageIRP({
    'vocab_size': 50000,
    'hidden_dim': 512,
    'max_iterations': 50
})

text = "The quick brown fox jumps over the lazy dog"
final_state, history = language.refine(text, {'mode': 'understand'})
results = language.get_understanding(final_state)
# → {'refined_tokens': [...], 'final_perplexity': 2.1, 'masks_remaining': 0}
```

### Example 3: Control IRP

**File**: `/home/dp/ai-workspace/HRM/sage/irp/control.py`

**State Space**: Trajectory (states + actions over horizon)

**Noise Model**: Random trajectory initialization

**Energy**: Action cost + terminal cost + dynamics violation
```python
energy = sum(actions²) + ||final_state - goal||² + ||predicted - actual||²
```

**Constraint Projection**:
```python
def project(state):
    # 1. Clamp actions to bounds
    actions = clamp(actions, -1, 1)

    # 2. Forward simulate with dynamics
    for t in range(horizon):
        states[t+1] = dynamics_model(states[t], actions[t])

    # 3. Apply obstacle avoidance
    states = avoid_obstacles(states)

    return state
```

**Key Feature**: Only halts when trajectory is BOTH feasible AND converged

**Example Usage**:
```python
control = ControlIRP({
    'state_dim': 4,
    'action_dim': 2,
    'horizon': 50,
    'max_iterations': 100
})

x0 = {
    'start': np.array([0, 0, 0, 0]),  # [x, y, vx, vy]
    'goal': np.array([5, 5, 0, 0])
}
final_state, history = control.refine(x0, {'avoid_obstacles': True})
traj = control.get_trajectory(final_state)
# → {'is_feasible': True, 'final_cost': 12.3, 'terminal_error': 0.01}
```

### Example 4: Memory IRP

**File**: `/home/dp/ai-workspace/HRM/sage/irp/memory.py`

**State Space**: Memory embeddings at different abstraction levels

**Noise Model**: Augmentation-based variations

**Energy**: Negative of (compression × retrieval_accuracy)
```python
energy = -(compression_ratio * retrieval_accuracy)
```

**Abstraction Hierarchy**:
1. `episodic` - Specific events (256 dim)
2. `semantic` - General knowledge (256 dim)
3. `procedural` - How-to knowledge (256 dim)
4. `conceptual` - Abstract principles (128 dim, 2x compression)
5. `strategic` - Meta-level patterns (64 dim, 4x compression)

**Augmentation Types**:
- `temporal_shift`: Shift temporal aspects
- `feature_dropout`: Randomly drop features
- `noise_injection`: Add Gaussian noise
- `permutation`: Permute dimensions

**Key Feature**: Progressive abstraction with SQLite verbatim storage

**Example Usage**:
```python
memory = MemoryIRP({
    'memory_dim': 256,
    'max_iterations': 100,
    'db_path': 'memory.db'
})

experiences = [
    {'embedding': np.random.randn(256), 'timestamp': i}
    for i in range(100)
]
final_state, history = memory.refine(experiences, {'goal': 'extract_patterns'})
consolidated = memory.get_consolidated_memory(final_state)
# → {'final_level': 'strategic', 'compression_achieved': 4.0, 'value_created': 2.3}
```

### Example 5: TinyVAE IRP

**File**: `/home/dp/ai-workspace/HRM/sage/irp/plugins/tinyvae_irp_plugin.py`

**State Space**: VAE latent codes

**Noise Model**: VAE reparameterization (μ + σε)

**Energy**: Reconstruction error + KL divergence
```python
energy = mse_loss(reconstruction, input) + beta_kl * kl_divergence
```

**Special Feature**: Single-pass (no iteration), instant compression

**Trust Metric**:
```python
trust_recon = max(0, 1.0 - recon_error * 10.0)
trust_kl = 1.0 / (1.0 + kl_div)
trust = 0.5 * trust_recon + 0.5 * trust_kl
```

**Example Usage**:
```python
tinyvae = create_tinyvae_irp(latent_dim=64, device='cuda')

crop = image[y:y+64, x:x+64]  # 64x64 crop
latent, telemetry = tinyvae.refine(crop)
# → latent.shape = [1, 64], telemetry['trust'] = 0.92
```

### Example 6: NeuTTS Air IRP

**File**: `/home/dp/ai-workspace/HRM/sage/irp/plugins/neutts_air_impl.py`

**State Space**: Audio waveform + prosody parameters

**Noise Model**: Initial synthesis noise, prosody uncertainty

**Energy**: Audio quality metrics
```python
# Spectral flatness (lower = more tonal = better)
energy = (1 - confidence) * 0.5 + spectral_flatness * 0.5
```

**Refinement Process**:
1. **Step 0**: Initial synthesis with voice cloning
2. **Steps 1-N**: Prosody refinement, quality improvement

**Key Feature**: Voice cloning from reference audio

**Example Usage**:
```python
tts = NeuTTSAirIRP({
    'backbone_repo': 'neuphonic/neutts-air-q4-gguf',
    'sample_rate': 24000
})

x0 = {
    'text': "Hello, this is a test.",
    'ref_audio': 'samples/dave.wav',
    'ref_text': "So I'm live on radio."
}
final_state, history = tts.refine(x0, {})
audio_data = tts.extract(final_state)
# → {'audio': np.ndarray, 'sample_rate': 24000, 'confidence': 0.8}
```

---

## Orchestration

### HRMOrchestrator

**File**: `/home/dp/ai-workspace/HRM/sage/irp/orchestrator.py`

**Purpose**: Asynchronous coordination of multiple IRP plugins with trust-based resource allocation

### Key Features

1. **Parallel Execution**
   ```python
   async def process_async(inputs: Dict[str, Any]) -> Dict[str, Any]:
       # Create futures for parallel execution
       futures = {}
       for name, plugin in self.plugins.items():
           if name in inputs:
               future = loop.run_in_executor(
                   self.executor,
                   self.run_plugin,
                   name, plugin, inputs[name], budgets[name]
               )
               futures[name] = future

       # Collect results as they complete
       while futures:
           done, pending = await asyncio.wait(
               futures.values(),
               return_when=asyncio.FIRST_COMPLETED
           )
           # Process completed plugins...
   ```

2. **Trust-Weighted Budget Allocation**
   ```python
   def allocate_budgets(self, available_ATP: float) -> Dict[str, float]:
       total_trust = sum(self.trust_weights.values())

       for name, plugin in self.plugins.items():
           weight = self.trust_weights[name] / total_trust
           budgets[name] = available_ATP * weight

       return budgets
   ```

3. **Dynamic Reallocation**
   ```python
   # When plugin finishes early
   freed_ATP = budgets[plugin_name] - result.budget_used

   # Reallocate to active plugins
   additional = self.reallocate_budget(freed_ATP, active_plugins)
   for name, extra in additional.items():
       budgets[name] += extra
   ```

4. **Trust Weight Updates**
   ```python
   def update_trust_weights(self, results, integrated):
       for name, result in results.items():
           monotonicity = result.telemetry['trust']['monotonicity_ratio']
           contribution = result.telemetry['trust']['contribution_to_H']
           efficiency = 1.0 - (result.budget_used / max_budget)

           new_trust = (
               0.7 * old_trust +
               0.2 * monotonicity * system_modifier +
               0.1 * efficiency
           )

           self.trust_weights[name] = clip(new_trust, 0.1, 10.0)
   ```

### Orchestrator Configuration

```python
config = {
    'total_ATP': 100.0,           # Total energy budget
    'max_workers': 4,             # Parallel execution threads
    'trust_update_rate': 0.1,     # Learning rate for trust
    'telemetry_interval': 10,     # Telemetry emission frequency

    # Plugin configs
    'enable_vision': True,
    'vision_config': {
        'latent_dim': 256,
        'max_iterations': 50
    },

    'enable_language': True,
    'language_config': {
        'hidden_dim': 512,
        'max_iterations': 50
    },

    'enable_control': True,
    'control_config': {
        'horizon': 50,
        'max_iterations': 100
    }
}

orchestrator = HRMOrchestrator(config)
```

### Usage Example

```python
# Prepare inputs for each plugin
inputs = {
    'vision': image_tensor,
    'language': "Process this text",
    'control': {
        'start': np.array([0, 0, 0, 0]),
        'goal': np.array([5, 5, 0, 0])
    }
}

# Run orchestrated processing
results = orchestrator.process(inputs)

# Access integrated results
print(f"System coherence: {results['system_coherence']}")
print(f"Total ATP used: {results['total_ATP_used']}")
print(f"Trust weights: {results['trust_weights']}")

# Get plugin-specific outputs
vision_output = results['plugin_outputs']['vision']
language_output = results['plugin_outputs']['language']
control_output = results['plugin_outputs']['control']
```

---

## Data Flow Diagrams

### Single Plugin Flow

```
┌──────────────────────────────────────────────────────────────┐
│                      SINGLE PLUGIN FLOW                       │
└──────────────────────────────────────────────────────────────┘

Input (x0) + Context (task_ctx)
    │
    ├─→ init_state()
    │       │
    │       ↓
    │   IRPState₀ {x: initial, E₀: high}
    │       │
    ├─→ ┌──step()──┐
    │   │          │
    │   ↓          ↓
    │   IRPState₁  IRPState₂  ...  IRPStateₙ
    │   E₁ < E₀    E₂ < E₁          Eₙ << E₀
    │       │
    │   ┌───halt()───┐
    │   │            │
    │   ↓            ↓
    │   Continue?    Stop!
    │                │
    ├─→ extract()    │
    │       │        │
    │       ↓        ↓
    └─→ Output + Telemetry
```

### Multi-Plugin Orchestration Flow

```
┌──────────────────────────────────────────────────────────────┐
│                  ORCHESTRATOR DATA FLOW                       │
└──────────────────────────────────────────────────────────────┘

Inputs = {vision: img, language: txt, control: traj}
    │
    ├─→ allocate_budgets(total_ATP) → {vision: 30, language: 40, control: 30}
    │
    ├─→ Parallel Execution:
    │       ┌─────────────┬─────────────┬─────────────┐
    │       │             │             │             │
    │       ↓             ↓             ↓             │
    │   VisionIRP     LanguageIRP   ControlIRP       │
    │   (refine)      (refine)      (refine)         │
    │       │             │             │             │
    │       ↓ (halts)     │             │             │
    │   Result₁           │             │             │
    │   ATP: 25           │             │             │
    │   freed: 5          │             │             │
    │       │             │             │             │
    │       └─→ reallocate(5) ─→┴─────→┘             │
    │                     │                           │
    │                     ↓ (halts)                   │
    │                 Result₂                         │
    │                 ATP: 45                         │
    │                     │                           │
    │                     │             ↓ (halts)     │
    │                     │         Result₃           │
    │                     │         ATP: 32           │
    │                     │             │             │
    ├─→ integrate_results([Result₁, Result₂, Result₃])
    │       │
    │       ↓
    │   Integrated Output {
    │       plugin_outputs: {vision: {...}, language: {...}, control: {...}},
    │       system_coherence: 0.87,
    │       total_ATP_used: 102,
    │       trust_weights: {vision: 1.2, language: 0.9, control: 1.1}
    │   }
    │       │
    ├─→ update_trust_weights()
    │       │
    │       ↓
    └─→ Final Results + Updated Trust
```

### Cross-Modal Information Flow (Future)

```
┌──────────────────────────────────────────────────────────────┐
│              CROSS-MODAL INFORMATION SHARING                  │
└──────────────────────────────────────────────────────────────┘

Vision Plugin                    Language Plugin
    │                                │
    ├─→ refine(image)                │
    │       │                        │
    │       ↓                        │
    │   visual_latent                │
    │       │                        │
    │       └─→ store_shared('vision_latent', latent)
    │                                │
    │                                ├─→ refine(text)
    │                                │       │
    │                                │   retrieve_shared('vision_latent')
    │                                │       │
    │                                │       ↓
    │                                │   language_latent + visual_context
    │                                │       │
    │                                │       ↓
    │                                │   multimodal_understanding
    │                                │
    └────────────────────────────────┴───────────────┐
                                                     │
                                                     ↓
                                             Integrated Output
```

### Energy Trajectory Visualization

```
Energy Over Time (Vision IRP Example)

E │
  │ ●                                      Initial state (E=1.0)
1 │  ╲
  │   ╲
  │    ●                                   Step 1 (E=0.8)
  │     ╲
0.8│      ╲
  │       ●                                Step 2 (E=0.6)
  │        ╲___
0.6│            ●                          Step 3 (E=0.5)
  │             ╲___
  │                 ●___                   Step 4 (E=0.4)
0.4│                     ●___              Step 5 (E=0.35)
  │                         ●___           Step 6 (E=0.32)
  │                             ●━━━━━━━━━ Converged! (slope < ε)
0.2│
  │
0 ├─────────────────────────────────────────────────→ Steps
  0   1   2   3   4   5   6   7   8   9   10

Trust Indicators:
● Monotonic decrease → High trust (monotonicity_ratio = 1.0)
━ Flat energy slope → Convergence (dE < ε)
```

---

## Integration Patterns

### Pattern 1: Sequential Processing

```python
# Process inputs one at a time
vision_state, _ = vision.refine(image, {})
vision_latent = vision.get_semantic_representation(vision_state)['latent']

# Feed vision output to language
language_input = {
    'text': caption,
    'visual_context': vision_latent
}
language_state, _ = language.refine(language_input, {})
```

### Pattern 2: Parallel Processing (Orchestrator)

```python
# All plugins run simultaneously
inputs = {
    'vision': image,
    'language': text,
    'control': trajectory_spec
}

results = orchestrator.process(inputs)
# All plugins complete with dynamic budget reallocation
```

### Pattern 3: Hierarchical Processing (HRM L/H Modules)

```python
# L-module: Fine-grained IRP iterations
def l_module_step():
    for plugin in active_plugins:
        state = plugin.step(state)  # One IRP iteration
        if plugin.energy(state) < threshold:
            yield state  # Report to H-module

# H-module: Orchestration and resource allocation
def h_module():
    while True:
        states = collect_l_module_outputs()
        integrated = integrate(states)
        budgets = allocate_budgets(integrated)
        dispatch_to_l_modules(budgets)
```

### Pattern 4: Memory-Augmented Processing

```python
# Store experiences during day
for experience in daily_interactions:
    memory.store_episodic(experience)

# Consolidate during sleep
sleep_config = {'consolidation_goal': 'extract_patterns'}
consolidated_state, _ = memory.refine(
    daily_experiences,
    sleep_config
)

# Retrieve during next day
query = encode_query("How do I solve this?")
relevant_memory = memory.retrieve(query, level='procedural')
```

### Pattern 5: Audio Conversation Loop

```python
# Continuous awareness cycle
class SproutAwarenessLoop:
    async def awareness_cycle(self):
        while True:
            # Listen for speech
            audio_state, _ = await audio_in.refine()
            transcription = audio_in.extract(audio_state)

            # Process with SAGE
            sage_response = sage.process(
                transcription['text'],
                context={'memory': memory_state, 'vision': vision_state}
            )

            # Speak response
            tts_input = {'text': sage_response.text}
            audio_out_state, _ = await tts.refine(tts_input)
            tts.save_audio(audio_out_state, '/tmp/response.wav')
            play_audio('/tmp/response.wav')
```

---

## Key Insights

### 1. Universal Pattern

**"Intelligence as Iterative Denoising"**: All cognitive processes can be modeled as refinement from noisy/incomplete to clean/complete states.

Examples:
- Vision: Noisy pixels → clear semantic understanding
- Language: Masked tokens → complete meaning
- Control: Random trajectory → optimal path
- Memory: Raw experiences → compressed wisdom
- Speech: Text → natural audio

### 2. Energy-Guided Convergence

**Energy functions provide unified optimization target**:
- Reconstruction loss (how well does state match target?)
- Task loss (how well does state solve the problem?)
- Constraint violation (how feasible is the state?)

**Convergence detection is automatic**:
- Slope < epsilon → stop
- No need for task-specific heuristics

### 3. Trust Emerges from Behavior

**Trust is not preset, it's earned**:
- Monotonic energy decrease → reliable plugin
- Low variance → stable plugin
- High convergence rate → efficient plugin

**Trust drives resource allocation**:
- High trust → more ATP budget
- Low trust → less resources until proven

### 4. Loose Coupling

**Plugins are independent**:
- No direct communication between plugins
- Orchestrator mediates all interaction
- Easy to add new plugins (just implement interface)

**Benefits**:
- Plugins can be developed separately
- Easy to swap implementations
- Testing is isolated

### 5. Early Stopping = Efficiency

**Adaptive iteration depth**:
- Stop as soon as "good enough"
- Don't waste compute on diminishing returns
- Task-specific thresholds (confidence, feasibility, etc.)

**Benchmarks show**:
- 32.3% token reduction with early stopping
- 2.8-4.4x speed improvement
- Maintains quality (sometimes improves it!)

---

## Practical Considerations

### Edge Deployment (Jetson)

```python
config = {
    'device': 'cuda',
    'use_fp16': True,           # Half precision
    'latent_dim': 64,           # Smaller latents
    'max_iterations': 20,       # Hard cap
    'halt_eps': 1e-3,           # Looser threshold
    'batch_size': 1             # Single sample
}
```

**Optimizations**:
- GroupNorm instead of BatchNorm (stable with batch=1)
- Depthwise separable convolutions (fewer params)
- Adaptive pooling (flexible input sizes)
- FP16 inference (2x faster on Jetson)

### Workstation/Cloud

```python
config = {
    'device': 'cuda',
    'use_fp16': False,          # Full precision
    'latent_dim': 256,          # Larger latents
    'max_iterations': 100,      # Adaptive
    'halt_eps': 1e-5,           # Tight threshold
    'batch_size': 32            # Batch processing
}
```

**Capabilities**:
- Full diffusion models (heavier backends)
- Real-time telemetry streaming
- Multi-GPU orchestration

### Telemetry Integration

```python
# Stream telemetry to monitoring system
def on_telemetry_emit(telemetry):
    # Send to Web4 dashboard
    web4_client.send_metric(telemetry)

    # Log for analysis
    logger.info(f"Plugin {telemetry['entity_id']}: "
                f"E={telemetry['E']:.3f}, "
                f"trust={telemetry['trust']['monotonicity_ratio']:.2f}")

    # Alert on anomalies
    if telemetry['trust']['monotonicity_ratio'] < 0.5:
        alert_service.warn(f"Low trust plugin: {telemetry['plugin']}")

plugin.on_telemetry = on_telemetry_emit
```

---

## Summary

The IRP system provides a **universal interface for iterative refinement** across diverse cognitive modalities. Key components:

**Core Abstractions**:
- `IRPState`: Container for refinement state
- `IRPPlugin`: Base interface with 4 invariants

**Plugin Contract**:
- `init_state()`: Initialize from raw input
- `energy()`: Measure refinement quality
- `step()`: Execute one refinement iteration
- `project()`: Enforce constraints (optional)
- `halt()`: Detect convergence (optional)

**Data Structures**:
- Plugin-specific state representations
- Metadata for tracking history
- Telemetry for monitoring

**Data Exchange**:
- Via orchestrator (loose coupling)
- Shared memory/context (future)
- Trust-weighted budget allocation

**Energy/Trust/Convergence**:
- Energy guides optimization
- Trust emerges from behavior
- Automatic convergence detection

**Orchestration**:
- Parallel plugin execution
- Dynamic resource reallocation
- Trust-based ATP budgeting

The architecture enables **"intelligence as iterative denoising"** - a universal computational pattern that unifies vision, language, control, memory, and speech under a single refinement protocol.

---

## References

### Code Files Analyzed

1. `/home/dp/ai-workspace/HRM/sage/irp/base.py` - Core abstractions
2. `/home/dp/ai-workspace/HRM/sage/irp/orchestrator.py` - Orchestration
3. `/home/dp/ai-workspace/HRM/sage/irp/vision.py` - Vision plugin
4. `/home/dp/ai-workspace/HRM/sage/irp/language.py` - Language plugin
5. `/home/dp/ai-workspace/HRM/sage/irp/control.py` - Control plugin
6. `/home/dp/ai-workspace/HRM/sage/irp/memory.py` - Memory plugin
7. `/home/dp/ai-workspace/HRM/sage/irp/plugins/tinyvae_irp_plugin.py` - TinyVAE
8. `/home/dp/ai-workspace/HRM/sage/irp/plugins/neutts_air_impl.py` - TTS
9. `/home/dp/ai-workspace/HRM/sage/irp/test_irp.py` - Test suite
10. `/home/dp/ai-workspace/HRM/sage/irp/README.md` - Documentation
11. `/home/dp/ai-workspace/HRM/sage/irp/BIDIRECTIONAL_AUDIO_ARCHITECTURE.md` - Audio integration

### Key Documentation

- IRP Protocol Specification: `../../IRP_PROTOCOL.md`
- Diffusion Architecture: `../../DIFFUSION_ARCHITECTURE.md`
- TinyVAE Compression Trust: `../../docs/tinyvae_compression_trust.md`

---

**Document Status**: Complete based on actual codebase analysis
**Next Steps**: Deploy plugins, integrate with HRM L/H modules, test orchestration
