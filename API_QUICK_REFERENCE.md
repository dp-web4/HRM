# API Quick Reference

## Most Common Operations

### 1. Basic Vision Refinement
```python
from sage.irp.plugins.vision_impl import create_vision_irp

vision_irp = create_vision_irp()
refined_image, telemetry = vision_irp.refine(image_tensor, early_stop=True)
print(f"Saved {telemetry['compute_saved']*100:.1f}% compute")
```

### 2. Basic Language Refinement
```python
from sage.irp.plugins.language_impl import create_language_irp

language_irp = create_language_irp()
refined_tokens, telemetry = language_irp.refine(token_ids, early_stop=True)
print(f"Iterations: {telemetry['iterations']}")
```

### 3. Parallel Multi-Modal Processing
```python
from sage.orchestrator.hrm_orchestrator import HRMOrchestrator
import asyncio

orchestrator = HRMOrchestrator(initial_atp=1000.0)
orchestrator.register_plugin("vision", vision_irp)
orchestrator.register_plugin("language", language_irp)

async def process():
    results = await orchestrator.execute_parallel({
        "vision": image_tensor,
        "language": token_ids
    })
    return results

results = asyncio.run(process())
```

### 4. Memory-Guided Refinement
```python
from sage.memory.irp_memory_bridge import IRPMemoryBridge, MemoryGuidedIRP

memory_bridge = IRPMemoryBridge()
guided_vision = MemoryGuidedIRP(vision_irp, memory_bridge)

# Refines with memory guidance
refined, telemetry = guided_vision.refine(image_tensor)
```

### 5. Full System Setup
```python
import torch
from sage.orchestrator.hrm_orchestrator import HRMOrchestrator
from sage.memory.irp_memory_bridge import IRPMemoryBridge, MemoryGuidedIRP
from sage.irp.plugins.vision_impl import create_vision_irp
from sage.irp.plugins.language_impl import create_language_irp

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
memory_bridge = IRPMemoryBridge()

# Create and wrap plugins
vision_guided = MemoryGuidedIRP(create_vision_irp(device), memory_bridge)
language_guided = MemoryGuidedIRP(create_language_irp(device), memory_bridge)

# Create orchestrator
orchestrator = HRMOrchestrator(initial_atp=1000.0)
orchestrator.register_plugin("vision", vision_guided)
orchestrator.register_plugin("language", language_guided)

# Use
async def process_multimodal(image, text):
    return await orchestrator.execute_parallel({
        "vision": image,
        "language": text
    })
```

## Key Parameters

### Vision IRP
- `max_iterations`: 50 (default)
- `eps`: 0.01 (convergence threshold)
- `device`: "cuda" or "cpu"

### Language IRP
- `max_iterations`: 30 (default)
- `mask_ratio`: 0.5 (initial mask)
- `model_variant`: "nano", "micro", or "tiny"

### Orchestrator
- `initial_atp`: 1000.0 (budget)
- `max_concurrent`: 4 (parallel limit)
- `initial_trust`: 1.0 (per plugin)

### Memory Bridge
- `buffer_size`: 100 (circular buffer)
- `consolidation_threshold`: 50
- `snarc_capacity`: 1000

## Expected Performance

| Operation | Time | Speedup | Quality |
|-----------|------|---------|---------|
| Vision Refinement | 3-5ms | 25x | 99.9% |
| Language Refinement | 2-4ms | 15x | Stable |
| Parallel Multi-Modal | <10ms | N/A | High |
| Memory Consolidation | 100ms | N/A | N/A |

## Common Telemetry Fields

All refinement operations return telemetry with:
```python
{
    "iterations": int,          # 2-50
    "compute_saved": float,     # 0.0-0.96
    "energy_trajectory": list,  # Energy per iteration
    "trust": float,            # 0.0-1.0
    "converged": bool,         # Did it converge?
    "time_ms": float           # Execution time
}
```

## Error Handling

```python
try:
    refined, telemetry = plugin.refine(input_data)
except Exception as e:
    print(f"Refinement failed: {e}")
    # Fallback logic here
```

## Testing Your Integration

```python
# Quick test
from demos.full_system_demo import full_system_demo
import asyncio

results = asyncio.run(full_system_demo())
print(f"Vision speedup: {results['vision_performance']['compute_saved']*100:.1f}%")
print(f"Language speedup: {results['language_performance']['compute_saved']*100:.1f}%")
```

---

*For complete documentation, see [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)*