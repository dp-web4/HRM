# IRP (Iterative Refinement Primitive) Implementation

**Version:** 1.0  
**Date:** 2025-08-23  
**Contributors:** Dennis Palatov, Claude, Nova

## Overview

This directory contains the implementation of the Iterative Refinement Primitive (IRP) protocol - a universal computational pattern for intelligence as iterative denoising toward coherence.

## Architecture

The IRP framework consists of:

### Core Components

1. **Base Module (`base.py`)**
   - `IRPPlugin`: Abstract base class defining the IRP interface
   - `IRPState`: State container for refinement process
   - Four invariants: State Space, Noise Model, Energy Metric, Coherence Contribution

2. **Plugin Implementations**
   - `VisionIRP` (`vision.py`): Latent space refinement for visual understanding
   - `LanguageIRP` (`language.py`): Masked denoising for text processing
   - `ControlIRP` (`control.py`): Trajectory planning with constraint projection
   - `MemoryIRP` (`memory.py`): Sleep consolidation through abstraction layers

3. **Orchestration (`orchestrator.py`)**
   - `HRMOrchestrator`: Asynchronous plugin management
   - Trust-weighted budget allocation
   - Dynamic resource reallocation
   - Integrated telemetry system

## Key Features

### Universal Pattern
```
Noisy/Incomplete State → [Iterative Refinement] → Clean/Complete State
```

### Energy-Based Halting
- Automatic early stopping when energy slope < ε
- Task-specific confidence thresholds
- Maximum iteration budgets

### Trust Dynamics
- Trust emerges from convergence stability
- Monotonic energy decrease → High trust
- Oscillating energy → Low trust

### Resource Management
- ATP-style energy budgeting
- Proportional allocation by trust
- Dynamic reallocation of freed resources

## Usage

### Basic Plugin Usage

```python
from sage.irp import VisionIRP

# Configure plugin
config = {
    'latent_dim': 256,
    'max_iterations': 50,
    'halt_eps': 1e-4,
    'device': 'cuda'
}

# Initialize plugin
vision = VisionIRP(config)

# Run refinement
image = load_image('input.jpg')
final_state, history = vision.refine(image, task_ctx={'target': 'objects'})

# Get results
results = vision.get_semantic_representation(final_state)
```

### Orchestrated Execution

```python
from sage.irp import HRMOrchestrator

# Configure orchestrator
config = {
    'total_ATP': 100.0,
    'enable_vision': True,
    'enable_language': True,
    'enable_control': True
}

orchestrator = HRMOrchestrator(config)

# Process inputs asynchronously
inputs = {
    'vision': image_data,
    'language': text_data,
    'control': trajectory_spec
}

results = orchestrator.process(inputs)
```

## Plugin Details

### Vision IRP
- Refines in learned latent space (not pixels)
- Progressive semantic levels: edges → textures → objects → relationships → affordances → meaning
- Early stops on task confidence

### Language IRP
- Span-based masking and progressive unmasking
- Meaning latent extraction
- Lightweight alternative to full diffusion LMs

### Control IRP
- Diffuser-style trajectory refinement
- Hard constraint projection for feasibility
- Dynamics-aware planning

### Memory IRP
- Progressive abstraction: episodic → semantic → procedural → conceptual → strategic
- Augmentation-based pattern extraction
- SQLite backend for verbatim storage

## Telemetry Format

Each plugin emits standardized telemetry:

```json
{
  "entity_id": "vision_irp_v1",
  "plugin": "VisionIRP",
  "step_idx": 17,
  "ΔE": -0.0123,
  "E": 0.482,
  "steps": 18,
  "halt_reason": "slope<ε",
  "trust": {
    "monotonicity_ratio": 0.93,
    "ΔE_variance": 0.004,
    "contribution_to_H": -0.021
  },
  "budget": {
    "ATP_spent": 1.7,
    "time_ms": 43.2
  }
}
```

## Testing

Run the test suite:

```bash
cd sage/irp
python test_irp.py
```

This will:
- Test each plugin individually
- Run orchestrated multi-plugin processing
- Benchmark early stopping configurations
- Save results to `test_results.json`

## Performance Considerations

### Edge Deployment (Jetson)
- Use FP16 precision
- Prefer lightweight backends (masked denoising, proximal steps)
- Always refine in compressed latent spaces
- Hard iteration caps

### Workstation/Cloud
- Full precision available
- Can use heavy backends (full diffusion)
- Adaptive iteration budgets
- Real-time telemetry streaming

## Connection to HRM

The IRP framework integrates seamlessly with HRM:
- L-module runs fine-grained IRP iterations
- H-module orchestrates plugins and allocates resources
- ACT mechanism maps to adaptive iteration depth
- Carry states maintain refinement progress

## Future Extensions

1. **Cross-Modal Refinement**: Share information between plugins
2. **Learned Halting**: Train halting policies
3. **Hybrid Backends**: Switch between IRP implementations dynamically
4. **Distributed Execution**: Scale across multiple devices

## References

- [IRP Protocol Specification](../../IRP_PROTOCOL.md)
- [Diffusion Architecture](../../DIFFUSION_ARCHITECTURE.md)
- [Nova's Architectural Framing](../../forum/nova/SAGE_IRP_Framing.md)