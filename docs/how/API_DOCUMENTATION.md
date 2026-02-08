# HRM/SAGE API Documentation

*Last Updated: August 23, 2025*

## Table of Contents
1. [IRP Plugin System](#irp-plugin-system)
2. [Orchestrator API](#orchestrator-api)
3. [Memory Bridge API](#memory-bridge-api)
4. [Cross-System Integration](#cross-system-integration)
5. [Parameter Reference](#parameter-reference)

---

## IRP Plugin System

### Base IRP Plugin Interface

All IRP plugins must inherit from `IRPPlugin` and implement the core refinement interface.

```python
from sage.irp.base import IRPPlugin
from typing import Any, Dict, Tuple, Optional

class IRPPlugin:
    """Base class for all IRP plugins"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize IRP plugin
        
        Args:
            config: Optional configuration dictionary with plugin-specific settings
        """
        self.entity_id = "base_irp"
        self.config = config or {}
        
    def refine(self, x: Any, early_stop: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Core refinement method - must be implemented by subclasses
        
        Args:
            x: Input to refine (tensor, tokens, etc.)
            early_stop: Whether to use early stopping based on energy convergence
            
        Returns:
            Tuple of:
                - refined: The refined output
                - telemetry: Dictionary containing:
                    - iterations: int - Number of refinement iterations
                    - energy_trajectory: List[float] - Energy values per iteration
                    - compute_saved: float - Fraction of compute saved (0-1)
                    - trust: float - Trust score for this refinement (0-1)
                    - converged: bool - Whether refinement converged
                    - energy_delta: float - Final energy change
        """
        raise NotImplementedError
```

### Vision IRP Plugin

```python
from sage.irp.plugins.vision_impl import VisionIRPImpl

# Creation
vision_irp = VisionIRPImpl(config={
    "max_iterations": 50,      # Maximum refinement iterations
    "eps": 0.01,               # Convergence threshold
    "device": "cuda",          # Computation device
    "vae_path": None          # Optional path to saved VAE model
})

# Usage
refined_image, telemetry = vision_irp.refine(
    x=image_tensor,           # Shape: [B, 3, 224, 224]
    early_stop=True           # Enable early stopping
)

# Telemetry contains:
# - iterations: 2-50
# - compute_saved: 0.0-0.96 (typically >0.9)
# - energy_trajectory: List of energy values
# - trust: Confidence in refinement quality
```

### Language IRP Plugin

```python
from sage.irp.plugins.language_impl import LanguageIRPImpl

# Creation
language_irp = LanguageIRPImpl(config={
    "max_iterations": 30,      # Maximum refinement iterations
    "mask_ratio": 0.5,        # Initial masking ratio
    "min_mask_ratio": 0.1,    # Final masking ratio
    "model_variant": "tiny",   # "nano", "micro", or "tiny"
    "device": "cuda"
})

# Usage
refined_tokens, telemetry = language_irp.refine(
    x=token_ids,              # Shape: [B, seq_len]
    early_stop=True
)

# Telemetry contains:
# - iterations: 2-30
# - compute_saved: 0.0-0.93
# - meaning_drift: Semantic similarity measure
# - mask_ratio_final: Final mask ratio used
```

### TinyVAE IRP Plugin

```python
from sage.irp.plugins.tinyvae_irp_plugin import create_tinyvae_irp

# Creation
tinyvae = create_tinyvae_irp(
    latent_dim=16,           # Dimension of latent space
    input_channels=1,         # 1 for grayscale, 3 for RGB
    device="cuda"            # Computation device
)

# Usage - Encode image crop to latent
crop = image[y:y+64, x:x+64]  # 64x64 crop from attention region
latent, telemetry = tinyvae.refine(
    x=crop,                   # Input crop (numpy or tensor)
    early_stop=True          # Not used (single pass)
)

# Get reconstruction
reconstruction = tinyvae.get_reconstruction()  # Returns tensor

# Batch operations
latents = tinyvae.encode_batch(batch_tensor)  # [B, C, H, W] -> [B, latent_dim]
recons = tinyvae.decode_batch(latents)        # [B, latent_dim] -> [B, C, H, W]

# Telemetry contains:
# - reconstruction_error: MSE between input and reconstruction
# - kl_divergence: Regularization term
# - latent_norm: L2 norm of latent vector
# - time_ms: Inference time
```

---

## Orchestrator API

### HRM Orchestrator

The orchestrator manages parallel execution of multiple IRP plugins with ATP (Allocation Transfer Packet) budget management.

```python
from sage.orchestrator.hrm_orchestrator import HRMOrchestrator

# Creation
orchestrator = HRMOrchestrator(
    initial_atp=1000.0,       # Initial ATP budget
    max_concurrent=4,         # Max parallel plugins
    reallocation_interval=0.1, # Seconds between reallocation checks
    device=None               # Optional torch device
)

# Plugin Registration
orchestrator.register_plugin(
    plugin_id="vision",       # Unique identifier
    plugin=vision_irp,        # IRP plugin instance
    initial_trust=1.0         # Initial trust weight (0-1)
)

# Synchronous Execution
results = orchestrator.execute(
    tasks={
        "vision": image_tensor,
        "language": token_ids
    },
    early_stop=True
)

# Asynchronous Parallel Execution
import asyncio
results = await orchestrator.execute_parallel(
    tasks={
        "vision": image_tensor,
        "language": token_ids
    },
    early_stop=True,
    timeout=5.0               # Optional timeout in seconds
)

# Results Format
for result in results:
    print(f"Plugin: {result.plugin_id}")
    print(f"State: {result.state}")  # completed/halted_early/failed
    print(f"ATP consumed: {result.atp_consumed}")
    print(f"Trust score: {result.trust_score}")
    print(f"Efficiency: {result.efficiency}")
    print(f"Output shape: {result.output.shape}")
```

### ATP Budget Management

```python
# Get current budget status
summary = orchestrator.get_orchestration_summary()

# Summary contains:
{
    "total_execution_time": float,
    "plugins_executed": int,
    "successful": int,
    "early_stopped": int,
    "average_efficiency": float,
    "budget_report": {
        "total_budget": float,
        "total_allocated": float,
        "total_consumed": float,
        "utilization": float,  # 0-1
        "per_plugin": {
            "vision": {
                "allocated": float,
                "consumed": float,
                "utilization": float
            },
            # ... other plugins
        }
    },
    "plugin_results": {
        # Detailed per-plugin metrics
    }
}
```

---

## Memory Bridge API

### IRP Memory Bridge

Connects IRP refinement with SNARC selective memory for experience-guided optimization.

```python
from sage.memory.irp_memory_bridge import IRPMemoryBridge, MemoryGuidedIRP

# Create Memory Bridge
memory_bridge = IRPMemoryBridge(
    buffer_size=100,          # Circular buffer capacity
    snarc_capacity=1000,      # SNARC memory capacity
    consolidation_threshold=50, # Consolidate after N memories
    device=torch.device("cuda")
)

# Wrap IRP Plugin with Memory Guidance
guided_plugin = MemoryGuidedIRP(
    irp_plugin=vision_irp,    # Any IRP plugin
    memory_bridge=memory_bridge
)

# Usage (same as regular IRP)
refined, telemetry = guided_plugin.refine(x, early_stop=True)

# Telemetry includes additional memory fields:
# - memory_guidance: Dict with suggested parameters
# - memory_efficiency: Efficiency score from memory
# - convergence_rate: How quickly it converged
```

### Memory Recording and Retrieval

```python
# Manual Memory Recording
memory = memory_bridge.record_refinement(
    plugin_id="vision_irp",
    initial_state=input_tensor,
    final_state=output_tensor,
    energy_trajectory=[0.5, 0.3, 0.1],
    telemetry={
        "iterations": 3,
        "compute_saved": 0.9,
        "trust": 0.8
    }
)

# Retrieve Guidance from Memory
guidance = memory_bridge.retrieve_guidance(
    plugin_id="vision_irp",
    current_state=new_input,
    k=5  # Number of similar memories to consider
)

# Guidance contains:
{
    "max_iterations": int,     # Suggested based on past success
    "early_stop_threshold": float,
    "trust_weight": float,
    "pattern": Optional[Dict], # Extracted pattern if available
    "similar_memories": int    # How many memories informed this
}

# Consolidation (extract patterns)
memory_bridge.consolidate()

# Get Memory Statistics
stats = memory_bridge.get_memory_stats()
# Returns:
{
    "total_memories": int,
    "pending_consolidation": int,
    "patterns_extracted": int,
    "plugins_with_patterns": List[str],
    "avg_efficiency": float,
    "avg_convergence": float,
    "avg_iterations": float
}
```

### SNARC Integration

When SNARC is available, the memory bridge uses:

```python
# SNARC Components (automatically initialized if available)
from SNARC.circular_buffer import CircularScratchpad, VerbatimStorage
from SNARC.full_implementation.snarc_core import SNARCGate, FastWeightMemory

# Configuration
SNARCConfig(
    gate_threshold=0.3,       # Salience threshold for storage
    weight_surprise=0.3,      # Weight for surprise component
    weight_novelty=0.15,      # Weight for novelty component
    weight_arousal=0.5,       # Weight for arousal component
    decay_rate=0.995          # Memory decay rate
)
```

---

## Cross-System Integration

### Complete System Setup

```python
import torch
from sage.orchestrator.hrm_orchestrator import HRMOrchestrator
from sage.memory.irp_memory_bridge import IRPMemoryBridge, MemoryGuidedIRP
from sage.irp.plugins.vision_impl import create_vision_irp
from sage.irp.plugins.language_impl import create_language_irp

# 1. Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Create Memory Bridge
memory_bridge = IRPMemoryBridge(
    buffer_size=50,
    consolidation_threshold=10
)

# 3. Create IRP Plugins
vision_irp = create_vision_irp(device)
language_irp = create_language_irp(device)

# 4. Wrap with Memory Guidance
vision_guided = MemoryGuidedIRP(vision_irp, memory_bridge)
language_guided = MemoryGuidedIRP(language_irp, memory_bridge)

# 5. Create Orchestrator
orchestrator = HRMOrchestrator(
    initial_atp=1000.0,
    max_concurrent=2
)

# 6. Register Plugins
orchestrator.register_plugin("vision", vision_guided, initial_trust=1.0)
orchestrator.register_plugin("language", language_guided, initial_trust=1.0)

# 7. Execute Tasks
async def process_multimodal(image, text):
    results = await orchestrator.execute_parallel({
        "vision": image,
        "language": text
    })
    return results

# 8. Periodic Consolidation
def consolidate_memories():
    memory_bridge.consolidate()
    stats = memory_bridge.get_memory_stats()
    return stats
```

### Data Flow

```
Input Data
    ↓
Orchestrator (ATP allocation)
    ↓
Memory-Guided IRP Plugin
    ├→ Memory Bridge (retrieve guidance)
    ├→ IRP Refinement (iterative processing)
    └→ Memory Bridge (record experience)
         ↓
    SNARC Evaluation (salience scoring)
         ↓
    Selective Storage
         ↓
    Pattern Extraction (consolidation)
         ↓
    Future Guidance
```

---

## Parameter Reference

### Common Parameters Across Systems

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `device` | str/torch.device | "cpu"/"cuda" | "cuda" if available | Computation device |
| `max_iterations` | int | 1-100 | 30-50 | Maximum refinement iterations |
| `early_stop` | bool | True/False | True | Enable convergence-based early stopping |
| `eps` | float | 0.001-0.1 | 0.01 | Convergence threshold |

### IRP-Specific Parameters

| Plugin | Parameter | Type | Range | Default | Description |
|--------|-----------|------|-------|---------|-------------|
| Vision | `vae_channels` | int | 16-64 | 32 | VAE base channels |
| Vision | `latent_dim` | int | 128-512 | 256 | Latent space dimension |
| Vision | `latent_size` | int | 7-14 | 7 | Spatial size of latent |
| Language | `model_variant` | str | nano/micro/tiny | tiny | Model size |
| Language | `mask_ratio` | float | 0.1-0.9 | 0.5 | Initial mask ratio |
| Language | `vocab_size` | int | 1000-50000 | 10000 | Vocabulary size |

### Orchestrator Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `initial_atp` | float | 100-10000 | 1000.0 | Initial ATP budget |
| `max_concurrent` | int | 1-8 | 4 | Maximum parallel plugins |
| `reallocation_interval` | float | 0.01-1.0 | 0.1 | Seconds between budget checks |
| `timeout` | float | 1-60 | None | Execution timeout in seconds |

### Memory Bridge Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `buffer_size` | int | 10-1000 | 100 | Circular buffer capacity |
| `snarc_capacity` | int | 100-10000 | 1000 | SNARC memory capacity |
| `consolidation_threshold` | int | 10-100 | 50 | Memories before consolidation |
| `gate_threshold` | float | 0.1-0.9 | 0.3 | SNARC salience threshold |

### Performance Targets

| Metric | Target | Achieved | Notes |
|--------|--------|----------|-------|
| Vision Speedup | 20x | 25x | With <1% quality loss |
| Language Speedup | 10x | 15x | With stable meaning |
| Memory Overhead | <100MB | ~50MB | Including SNARC |
| Convergence | 2-5 iterations | 2-3 | Typical refinement |
| ATP Utilization | >50% | 40-60% | Depends on workload |

---

## Error Handling

### Common Exceptions

```python
# IRP Plugin Errors
class IRPConvergenceError(Exception):
    """Raised when refinement fails to converge"""
    
class IRPConfigError(Exception):
    """Raised when plugin configuration is invalid"""

# Orchestrator Errors  
class ATPBudgetExceeded(Exception):
    """Raised when ATP budget is exhausted"""
    
class PluginRegistrationError(Exception):
    """Raised when plugin registration fails"""

# Memory Bridge Errors
class MemoryConsolidationError(Exception):
    """Raised when consolidation fails"""
    
class SNARCInitializationError(Exception):
    """Raised when SNARC components fail to initialize"""
```

### Error Recovery

```python
try:
    results = await orchestrator.execute_parallel(tasks)
except ATPBudgetExceeded:
    # Increase budget or reduce task complexity
    orchestrator.budget.add_budget(500)
    results = await orchestrator.execute_parallel(tasks)
except asyncio.TimeoutError:
    # Task took too long
    results = orchestrator.get_partial_results()
```

---

## Testing and Validation

### Unit Tests

```bash
# Test individual components
python3 -m pytest sage/tests/test_irp_plugins.py
python3 -m pytest sage/tests/test_orchestrator.py
python3 -m pytest sage/tests/test_memory_bridge.py
```

### Integration Tests

```bash
# Test full system
python3 demos/full_system_demo.py
python3 demos/sleep_cycle_demo.py
python3 benchmarks/baseline_jetson.py
```

### Performance Validation

```python
# Validate performance targets
from benchmarks.validate_performance import validate_all

results = validate_all()
assert results["vision_speedup"] >= 20
assert results["language_speedup"] >= 10
assert results["memory_overhead_mb"] < 100
```

---

## Version Compatibility

| Component | Version | Requirements |
|-----------|---------|--------------|
| Python | 3.8+ | 3.10 recommended |
| PyTorch | 2.0+ | 2.3+ for Jetson |
| CUDA | 11.8+ | 12.1 for full features |
| SNARC | Latest | From Memory project |
| HRM Core | 1.0+ | This repository |

---

*For implementation examples, see the `demos/` directory.*
*For performance benchmarks, see `benchmarks/` directory.*
*For troubleshooting, see `TROUBLESHOOTING.md`.*