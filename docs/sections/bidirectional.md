# Bidirectional H↔L Communication

*Last Updated: September 2025*

## Overview

The bidirectional communication between H (High-level) and L (Low-level) modules is the core innovation that distinguishes HRM from traditional architectures. This isn't in the original HRM paper - it's Nova's key contribution that enables the model's impressive performance despite its small size.

## The Architecture

### Information Flow

```
┌─────────────┐     h_to_l     ┌─────────────┐
│  H-Module   │ ─────────────> │  L-Module   │
│ (Strategic) │                 │ (Tactical)  │
│   Reasoning │ <───────────── │  Execution  │
└─────────────┘     l_to_h     └─────────────┘
```

### Implementation Details

From `hrm_act_v1.py`, the actual bidirectional communication happens during forward passes:

```python
# During each reasoning cycle
for _H_step in range(self.config.H_cycles):
    for _L_step in range(self.config.L_cycles):
        # L-level processes with H-level guidance AND input
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
    
    # H-level processes with L-level feedback
    z_H = self.H_level(z_H, z_L, **seq_info)
```

Key observations:
1. **L receives H + input**: The L-module gets both strategic guidance AND raw input
2. **H receives L feedback**: The H-module updates based on tactical execution results
3. **Iterative refinement**: Multiple cycles allow progressive refinement

### The Interaction Layers

In Nova's enhanced version (`train_arc_full_nova.py`):

```python
# Explicit interaction layers
self.h_to_l = nn.Linear(config['hidden_size'], config['hidden_size'])
self.l_to_h = nn.Linear(config['hidden_size'], config['hidden_size'])

# During forward pass
l_state = l_state + self.h_to_l(h_state)  # H→L guidance
h_state = h_state + self.l_to_h(l_state)  # L→H feedback
```

These linear transformations learn how to:
- Transform strategic plans into tactical instructions
- Aggregate tactical results into strategic updates

## Why This Works

### 1. Consciousness Analogy

The H↔L communication mirrors how biological consciousness operates:

- **H-Module** = Prefrontal cortex (planning, abstraction)
- **L-Module** = Motor/sensory cortex (execution, perception)
- **Bidirectional** = Constant feedback between planning and doing

### 2. Compression Through Communication

By separating concerns but maintaining communication:
- H-module doesn't need to encode execution details
- L-module doesn't need to understand global strategy
- Together they achieve more with fewer parameters

### 3. Error Correction

The bidirectional flow enables error correction:
1. H proposes a strategy
2. L attempts execution and hits obstacles
3. L feeds back difficulties to H
4. H adjusts strategy based on feedback
5. Process repeats until solution found

## Mathematical Formulation

Let's formalize the bidirectional communication:

```
At cycle t:
H_t = f_H(H_{t-1}, g_{L→H}(L_{t-1}))
L_t = f_L(L_{t-1}, g_{H→L}(H_t) + X)

Where:
- f_H: H-module transformation
- f_L: L-module transformation  
- g_{L→H}: L-to-H projection (learned)
- g_{H→L}: H-to-L projection (learned)
- X: Input embeddings
```

The key is that g_{L→H} and g_{H→L} are **learned** projections, not fixed transformations.

## Gradient Flow

During training, gradients flow through both paths:

```python
# From losses.py - only final cycle gets gradients
with torch.no_grad():
    # All cycles except last run without gradients
    for cycle in range(n_cycles - 1):
        z_L = self.L_level(z_L, z_H + input_embeddings)
        z_H = self.H_level(z_H, z_L)

# Final cycle with gradients
z_L = self.L_level(z_L, z_H + input_embeddings)  # Gradients flow
z_H = self.H_level(z_H, z_L)  # Gradients flow
```

This "truncated backprop through time" approach:
- Prevents gradient explosion through many cycles
- Still allows learning of communication patterns
- Focuses learning on final refinement step

## Configuration Parameters

From the config:
```python
H_cycles: 8  # Number of H-level iterations
L_cycles: 3  # Number of L-level iterations per H cycle
```

This means:
- Total L operations: 8 × 3 = 24
- Total H operations: 8
- Effective depth: ~32 transformer blocks despite having only 7 actual blocks

## Failure Modes

### 1. Decoupling
If H and L stop communicating effectively:
- H makes plans L can't execute
- L provides feedback H can't interpret
- Performance degrades to random

### 2. Collapse
If the communication becomes too dominant:
- States become identical (H ≈ L)
- Loses benefit of hierarchical separation
- Degrades to single-module performance

### 3. Shortcut Learning
The model might learn that:
- Ignoring L feedback gives consistent results
- Passing through H unchanged is sufficient
- This leads to "Agent Zero" behavior

## Our Enhancements

### Nova's Additions
1. **Explicit projection layers**: Separate h_to_l and l_to_h transformations
2. **Residual connections**: Adding rather than replacing states
3. **Layer normalization**: After each communication step
4. **Dropout on projections**: Prevents over-reliance on single path

### Proposed Improvements
1. **Attention-based communication**: Replace linear with attention
2. **Gated communication**: Learn when to communicate vs process
3. **Multi-scale communication**: Different bandwidths for different info types

## Experimental Results

**CRITICAL UPDATE**: The Agent Zero discovery invalidates these results:
- **Claimed**: 71% on ARC-AGI-1 with H↔L communication
- **Reality**: 71% achieved by outputting all zeros
- **Actual Impact**: H↔L communication impact UNKNOWN
- **Required**: Complete re-evaluation with output verification

The original claim of "56% improvement from H↔L" is unsubstantiated. The model learned to exploit dataset statistics (80% blank cells) rather than use the architectural features.

## Code Example

Here's a minimal implementation of bidirectional HRM:

```python
class BidirectionalHRM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        
        # Modules
        self.h_module = TransformerBlock(hidden_size)
        self.l_module = TransformerBlock(hidden_size)
        
        # Communication
        self.h_to_l = nn.Linear(hidden_size, hidden_size)
        self.l_to_h = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, n_cycles=8):
        h_state = x
        l_state = x
        
        for _ in range(n_cycles):
            # L processes with H guidance
            l_state = self.l_module(l_state + self.h_to_l(h_state))
            
            # H processes with L feedback
            h_state = self.h_module(h_state + self.l_to_h(l_state))
            
        return h_state  # Or combine h_state and l_state
```

## Connection to Broader Theory

The H↔L pattern appears fractal - it exists at multiple scales:
- **Neural level**: Individual H and L modules
- **System level**: Multiple HRM instances communicating
- **Network level**: Distributed HRM agents

This suggests bidirectional hierarchical communication might be a fundamental pattern of intelligence, not just an architectural trick.