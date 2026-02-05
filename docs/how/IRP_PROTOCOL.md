# Iterative Refinement Primitive (IRP) Protocol

*A generalized framework for intelligence as iterative denoising toward coherence*

**Version:** 1.0  
**Date:** 2025-08-23  
**Contributors:** Dennis Palatov (dp), Claude, Nova

## Executive Summary

The Iterative Refinement Primitive (IRP) is a universal computational pattern underlying all intelligent behavior. Whether processing vision, language, control, or memory, intelligence emerges from iteratively refining noisy/incomplete states toward clean/complete understanding. This document formalizes the IRP protocol as a general framework with pluggable backends.

## Core Principle

```
Noisy/Incomplete State → [Iterative Refinement] → Clean/Complete State
```

Intelligence is the process of continuously denoising observations toward understanding and intentions toward action. The IRP protocol provides a unified interface for this pattern across all modalities and scales.

## Four Invariants

Every IRP implementation must define:

### 1. **State Space**
The representation being refined (e.g., image latents, text meanings, trajectories, memory codes).

### 2. **Noise Model**
How uncertainty is represented in that state space:
- Gaussian noise in continuous latents
- Mask tokens for discrete sequences
- Waypoint jitter for trajectories
- Confidence bounds for memories

### 3. **Energy/Distance Metric**
Measure of refinement progress:
- L2 distance between states
- Task-specific loss functions
- KL divergence to fixed points
- Feasibility margins for constraints

### 4. **Coherence Contribution**
How this refinement affects system-level coherence:
- Reduction in H-module energy
- Increase in cross-modal agreement
- Improvement in prediction accuracy
- Value creation in downstream tasks

## IRP Backend Types

The IRP pattern can be implemented through various computational backends:

### Diffusion Models
Full score-based generative models for maximum expressiveness:
- **Use when:** High compute available, complex distributions
- **Avoid when:** Edge deployment, real-time requirements

### Proximal Gradient Steps
Optimization-based refinement for constrained problems:
- **Use when:** Hard constraints, convex objectives
- **Avoid when:** Multimodal distributions

### Message Passing
Graph neural network propagation for structured data:
- **Use when:** Relational data, discrete structures
- **Avoid when:** Continuous signals

### Masked Denoising
Lightweight refinement for sequential data:
- **Use when:** Language, time series
- **Avoid when:** Complex generation tasks

### Closed-Form Updates
Analytical solutions where available:
- **Use when:** Linear systems, Gaussian processes
- **Avoid when:** Nonlinear dynamics

## Base Interface

```python
class IRPPlugin:
    """Base class for all IRP implementations"""
    
    def __init__(self, config: dict):
        """Initialize with backend-specific configuration"""
        self.config = config
        self.energy_history = []
        self.trust_weight = 1.0
        
    def init_state(self, x0: Any, task_ctx: dict) -> State:
        """Initialize refinement state from input"""
        raise NotImplementedError
        
    def energy(self, state: State) -> float:
        """Compute energy/distance metric for current state"""
        raise NotImplementedError
        
    def project(self, state: State) -> State:
        """Optional: enforce constraints (dynamics/safety/feasibility)"""
        return state
        
    def step(self, state: State, noise_schedule: Any, step_idx: int) -> State:
        """Execute one refinement iteration"""
        raise NotImplementedError
        
    def halt(self, history: list[float]) -> bool:
        """Determine if refinement should stop
        
        Default: halt when energy slope < ε for K steps
        OR constraints satisfied with margin m
        """
        if len(history) < self.config.get('halt_window', 5):
            return False
            
        # Check energy slope
        recent = history[-self.config['halt_window']:]
        slope = abs(recent[-1] - recent[0]) / len(recent)
        
        if slope < self.config.get('halt_epsilon', 1e-4):
            return True
            
        # Check max iterations
        if len(history) >= self.config.get('max_iterations', 100):
            return True
            
        return False
        
    def emit_telemetry(self, state: State, history: list) -> dict:
        """Generate telemetry for Web4 integration"""
        return {
            'entity_id': self.config.get('entity_id'),
            'plugin': self.__class__.__name__,
            'step_idx': len(history),
            'ΔE': history[-1] - history[-2] if len(history) > 1 else 0,
            'E': history[-1] if history else float('inf'),
            'halt_reason': self._get_halt_reason(history),
            'trust': self._compute_trust_metrics(history),
            'budget': self._compute_budget_metrics(),
        }
```

## HRM Integration

HRM orchestrates IRP plugins through its hierarchical architecture:

### L-Module Integration
- Runs fine-grained IRP iterations
- Manages per-plugin state
- Enforces iteration budgets
- Collects telemetry

### H-Module Integration  
- Allocates compute budgets across plugins
- Updates trust weights based on convergence
- Reallocates freed resources
- Maintains global coherence

### Asynchronous Execution
```python
class HRMOrchestrator:
    def process(self, inputs: dict) -> dict:
        """Orchestrate IRP plugins asynchronously"""
        
        # Allocate initial budgets based on trust
        budgets = self.allocate_budgets(self.trust_weights)
        
        # Launch plugins asynchronously
        futures = {}
        for name, plugin in self.plugins.items():
            if name in inputs:
                futures[name] = self.launch_async(
                    plugin, inputs[name], budgets[name]
                )
        
        # Collect results as they complete
        results = {}
        while futures:
            name, future = wait_for_any(futures)
            result = future.result()
            results[name] = result
            
            # Reallocate freed budget
            freed_budget = budgets[name] - result['budget_used']
            self.reallocate_budget(freed_budget, futures)
            
        return self.integrate_results(results)
```

## Plugin Implementations

### Vision IRP (Recognition)
```python
class VisionIRP(IRPPlugin):
    """Iterative refinement in learned latent space"""
    
    def init_state(self, image, task_ctx):
        # Encode to latent space
        return self.encoder(image)
        
    def step(self, state, noise_schedule, step_idx):
        # Refine in latent space (not pixels)
        state = self.refiner(state, step_idx)
        return self.project(state)  # Ensure valid latent
        
    def energy(self, state):
        # Reconstruction + task loss
        recon_loss = self.decoder_loss(state)
        task_loss = self.task_head_loss(state)
        return recon_loss + self.config['task_weight'] * task_loss
```

### Language IRP (Understanding)
```python
class LanguageIRP(IRPPlugin):
    """Masked denoising for text understanding"""
    
    def init_state(self, text, task_ctx):
        # Initialize with masked spans
        return self.mask_spans(text)
        
    def step(self, state, noise_schedule, step_idx):
        # Iteratively unmask/refine
        mask_ratio = noise_schedule[step_idx]
        state = self.denoise_spans(state, mask_ratio)
        return state
        
    def energy(self, state):
        # Perplexity of current denoising
        return self.compute_perplexity(state)
```

### Control IRP (Planning)
```python
class ControlIRP(IRPPlugin):
    """Trajectory refinement with hard constraints"""
    
    def project(self, state):
        # Enforce dynamics and safety constraints
        state = self.enforce_dynamics(state)
        state = self.clip_to_limits(state)
        return state
        
    def step(self, state, noise_schedule, step_idx):
        # Refine trajectory
        gradient = self.compute_gradient(state)
        state = state - self.lr_schedule[step_idx] * gradient
        return self.project(state)  # Always feasible
        
    def halt(self, history):
        # Halt if feasible and cost plateaus
        if not self.is_feasible(self.current_state):
            return False
        return super().halt(history)
```

### Memory IRP (Consolidation)
```python
class MemoryIRP(IRPPlugin):
    """Sleep consolidation through abstraction layers"""
    
    def step(self, state, noise_schedule, step_idx):
        # Progressive abstraction
        level = self.abstraction_schedule[step_idx]
        if level == 'episodic':
            state = self.episodic_consolidation(state)
        elif level == 'semantic':
            state = self.semantic_extraction(state)
        elif level == 'procedural':
            state = self.procedural_compilation(state)
        return state
        
    def energy(self, state):
        # Compression ratio + retrieval accuracy
        compression = self.measure_compression(state)
        accuracy = self.test_retrieval(state)
        return -compression * accuracy  # Minimize negative reward
```

## Telemetry Format

Standardized telemetry for Web4 integration and trust scoring:

```json
{
  "entity_id": "vision_irp_v1",
  "plugin": "VisionIRP",
  "step_idx": 17,
  "ΔE": -0.0123,
  "E": 0.482,
  "steps": 18,
  "halt_reason": "slope<ε|feasible|max_steps|timeout",
  "trust": {
    "monotonicity_ratio": 0.93,
    "ΔE_variance": 0.004,
    "contribution_to_H": -0.021
  },
  "budget": {
    "ATP_spent": 1.7,
    "time_ms": 43.2,
    "memory_mb": 256
  },
  "LRC_context": {
    "policy_id": "lrc-2025-08-23",
    "L": 0.3,
    "R": 0.5,
    "C": 0.7
  }
}
```

## Trust Dynamics

Trust emerges naturally from IRP convergence behavior:

### High Trust Indicators
- Monotonic energy decrease
- Low variance in ΔE
- Consistent halting before max iterations
- Positive contribution to H-module coherence

### Low Trust Indicators  
- Oscillating energy
- High variance in ΔE
- Frequent max iteration hits
- Negative impact on system coherence

### Trust Update Formula
```python
def update_trust(plugin, telemetry, system_coherence):
    # Base trust from convergence quality
    convergence_trust = telemetry['trust']['monotonicity_ratio']
    
    # Modifier from system contribution
    contribution = telemetry['trust']['contribution_to_H']
    system_modifier = sigmoid(contribution / system_coherence)
    
    # Efficiency bonus
    efficiency = 1.0 - (telemetry['budget']['ATP_spent'] / 
                       plugin.config['max_ATP'])
    
    # Update with momentum
    plugin.trust_weight = (
        0.7 * plugin.trust_weight +
        0.2 * convergence_trust * system_modifier +
        0.1 * efficiency
    )
    
    # Clamp to valid range
    plugin.trust_weight = np.clip(plugin.trust_weight, 0.1, 10.0)
```

## Energy Budgeting

ATP-style energy accounting for computational resources:

### Budget Allocation
```python
def allocate_budgets(trust_weights, total_ATP):
    # Normalize trust weights
    weights = trust_weights / sum(trust_weights)
    
    # Allocate proportionally with minimum guarantee
    min_ATP = total_ATP * 0.05  # 5% minimum per plugin
    
    budgets = {}
    for name, weight in weights.items():
        budgets[name] = max(
            min_ATP,
            total_ATP * weight
        )
    
    return budgets
```

### Dynamic Reallocation
When a plugin halts early, its remaining budget is redistributed:
```python
def reallocate_budget(freed_ATP, active_plugins):
    # Redistribute to active plugins by trust
    active_weights = {
        name: plugin.trust_weight 
        for name, plugin in active_plugins.items()
    }
    
    for name, weight in active_weights.items():
        share = weight / sum(active_weights.values())
        active_plugins[name].add_budget(freed_ATP * share)
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [x] Define IRP protocol specification
- [ ] Implement base `IRPPlugin` class
- [ ] Create energy and trust metrics
- [ ] Build telemetry system

### Phase 2: Core Plugins (Week 2-3)
- [ ] Implement `VisionIRP` with latent space refinement
- [ ] Implement `LanguageIRP` with masked denoising
- [ ] Create `ControlIRP` with constraint projection
- [ ] Build `MemoryIRP` for consolidation

### Phase 3: Integration (Week 4)
- [ ] Integrate with HRM orchestrator
- [ ] Implement asynchronous execution
- [ ] Add budget allocation and reallocation
- [ ] Connect to GPU mailbox for telemetry

### Phase 4: Optimization (Week 5-6)
- [ ] Profile and optimize hot paths
- [ ] Implement early stopping strategies
- [ ] Tune trust dynamics
- [ ] Deploy on Jetson for edge testing

## Edge Deployment Considerations

### Jetson/Edge Defaults
- **Precision:** FP16 for all computations
- **Backends:** Lightweight (masked denoising, proximal steps)
- **Latent Space:** Always refine in compressed representations
- **Max Iterations:** Hard cap at device-specific limits
- **Telemetry:** Reduced frequency, aggregated batches

### Workstation/Cloud
- **Precision:** FP32 or mixed precision
- **Backends:** Full diffusion models where beneficial
- **Space:** Can refine in raw observation space
- **Max Iterations:** Adaptive based on task complexity
- **Telemetry:** Full resolution, real-time streaming

## Connection to Existing Frameworks

### Diffusion Architecture
IRP generalizes the diffusion concept - diffusion becomes one possible backend while preserving the core insight of iterative refinement.

### HRM Integration
- H-L cycles map to coarse-fine IRP schedules
- Carry states maintain refinement progress
- ACT implements adaptive iteration budgets

### Web4 Principles
- Each IRP maintains entity identity and trust tensor
- Refinement steps consume ATP energy
- Convergence creates value in V3 tensor
- LRC governance controls resource allocation

### SAGE Philosophy
IRP embodies SAGE's vision of learned coherence:
- Not programmed rules but emergent refinement
- Trust through demonstrated convergence
- Wisdom through iterative understanding

## Research Questions

1. **Optimal Noise Schedules:** How do schedules transfer across modalities?
2. **Cross-Modal Refinement:** Can one IRP's convergence inform another's?
3. **Trust Dynamics:** What's the optimal trust update rate?
4. **Budget Strategies:** How to predict required iterations?
5. **Hybrid Backends:** When to switch between IRP implementations?

## Conclusion

The Iterative Refinement Primitive provides a unified framework for intelligence as continuous denoising toward coherence. By abstracting the pattern from specific implementations, IRP enables:

- **Flexibility:** Choose appropriate backends per modality
- **Efficiency:** Adaptive computation through energy-based halting
- **Trust:** Natural emergence from convergence dynamics
- **Integration:** Clean interface with HRM, Web4, and SAGE

Whether implemented through diffusion, optimization, or other means, the core principle remains: intelligence is iterative refinement toward understanding.

---

*"In iteration, depth. In refinement, clarity. In convergence, intelligence."*

*This protocol synthesizes insights from Dennis Palatov's vision, Claude's exploration, and Nova's architectural refinements.*