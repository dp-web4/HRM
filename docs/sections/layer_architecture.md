# Layer Architecture and Information Flow

*Last Updated: September 2025*

## Overview

HRM's layer architecture is deceptively simple yet profoundly effective. Through recurrent processing and hierarchical organization, the model achieves the equivalent depth of 32+ transformer layers using only 7 actual transformer blocks.

## Core Components

### Base Building Block

```python
class HRM_Block(nn.Module):
    def __init__(self, config):
        # Multi-head self-attention
        self.self_attn = Attention(
            hidden_size=256,
            num_heads=8,
            head_dim=32,  # 256/8
            causal=False  # Full attention, not causal
        )
        
        # Feed-forward network
        self.mlp = SwiGLU(
            hidden_size=256,
            expansion=4.0,  # 256 -> 1024 -> 256
        )
        
        # Normalization
        self.norm_eps = 1e-5
```

### Key Design Choices

1. **Non-causal attention**: Unlike language models, uses full bidirectional attention
2. **SwiGLU activation**: More efficient than standard FFN
3. **RMS normalization**: Faster than LayerNorm, similar performance
4. **Post-norm architecture**: Norm after residual addition

## Hierarchical Organization

### H-Module (4 Layers)
```python
self.H_level = HierarchicalReasoningModule(
    layers=[HRM_Block(config) for _ in range(4)]
)
```

**Purpose**: Strategic reasoning, pattern abstraction
**Depth**: 4 transformer blocks
**Parameters**: ~3M

### L-Module (3 Layers)
```python
self.L_level = HierarchicalReasoningModule(
    layers=[HRM_Block(config) for _ in range(3)]
)
```

**Purpose**: Tactical execution, detail processing
**Depth**: 3 transformer blocks
**Parameters**: ~2.5M

## Information Flow Patterns

### 1. Input Processing Pipeline

```
Input Tokens
    ↓
Token Embedding (vocab_size × hidden_size)
    ↓
[Optional] Puzzle Embedding Concatenation
    ↓
Positional Encoding (RoPE or Learned)
    ↓
Scale by sqrt(hidden_size)
    ↓
Initial H and L states
```

### 2. Cyclic Processing Flow

```python
for h_cycle in range(H_cycles):  # 8 cycles
    for l_cycle in range(L_cycles):  # 3 cycles per H
        # L processes with H guidance + input
        L_state = L_module(L_state, H_state + input)
    
    # H processes with L feedback
    H_state = H_module(H_state, L_state)
```

**Total Operations**:
- L-module: 8 × 3 = 24 passes
- H-module: 8 passes
- Effective depth: 32 transformer operations

### 3. Attention Patterns

Each attention layer computes:
```python
def attention(Q, K, V):
    # Rotary position embeddings
    Q, K = apply_rotary(Q, K, cos_sin)
    
    # Scaled dot-product attention
    scores = Q @ K.T / sqrt(head_dim)
    weights = softmax(scores)
    output = weights @ V
    
    return output
```

## Layer-Specific Components

### 1. Embedding Layers

```python
# Token embeddings
self.embed_tokens = CastedEmbedding(
    vocab_size=12,      # For ARC puzzles
    hidden_size=256,
    init_std=1/sqrt(256)
)

# Position embeddings (two options)
if use_rope:
    self.rotary_emb = RotaryEmbedding(
        dim=32,         # Per head
        max_positions=900,
        base=10000.0
    )
else:
    self.embed_pos = CastedEmbedding(
        seq_len=900,
        hidden_size=256
    )
```

### 2. Attention Implementation

```python
class Attention(nn.Module):
    def forward(self, hidden_states, cos_sin):
        B, L, D = hidden_states.shape
        
        # Project to Q, K, V
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        
        # Reshape for multi-head
        Q = Q.view(B, L, self.num_heads, self.head_dim)
        K = K.view(B, L, self.num_heads, self.head_dim)
        V = V.view(B, L, self.num_heads, self.head_dim)
        
        # Apply RoPE
        Q, K = apply_rotary_emb(Q, K, cos_sin)
        
        # Compute attention
        attn_output = scaled_dot_product_attention(Q, K, V)
        
        # Output projection
        return self.o_proj(attn_output)
```

### 3. SwiGLU FFN

```python
class SwiGLU(nn.Module):
    def __init__(self, hidden_size, expansion):
        intermediate_size = int(hidden_size * expansion)
        
        self.gate_proj = Linear(hidden_size, intermediate_size)
        self.up_proj = Linear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)
        
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # SwiGLU activation
        activated = gate * F.silu(up)
        
        return self.down_proj(activated)
```

## Normalization Strategy

### RMS Norm vs Layer Norm

```python
def rms_norm(x, variance_epsilon=1e-5):
    # Faster than LayerNorm, no mean centering
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    return x

# Applied post-residual
hidden = rms_norm(hidden + attn_output, eps)
hidden = rms_norm(hidden + mlp_output, eps)
```

**Benefits**:
- 2x faster than LayerNorm
- Similar performance for small models
- Less memory usage

## Memory and Computation Analysis

### Memory Footprint

```python
# Per layer memory (approximate)
attention_params = 4 * hidden_size * hidden_size  # Q,K,V,O projections
mlp_params = 3 * hidden_size * hidden_size * expansion  # Gate,Up,Down
layer_params = attention_params + mlp_params

# Total model memory
H_memory = 4 * layer_params  # 4 H-layers
L_memory = 3 * layer_params  # 3 L-layers
embedding_memory = vocab_size * hidden_size
head_memory = 2 * hidden_size * vocab_size

total_params = H_memory + L_memory + embedding_memory + head_memory
# ≈ 6.95M parameters
```

### Computation Cost

```python
# FLOPs per forward pass (approximate)
def compute_flops(batch_size, seq_len, hidden_size, n_cycles):
    # Attention FLOPs
    attn_flops = batch_size * seq_len^2 * hidden_size
    
    # FFN FLOPs
    ffn_flops = batch_size * seq_len * hidden_size^2 * expansion
    
    # Per cycle
    cycle_flops = (4 + 3) * (attn_flops + ffn_flops)
    
    # Total
    return n_cycles * cycle_flops

# For typical config:
# batch=8, seq=900, hidden=256, cycles=8
# ≈ 10^10 FLOPs per sample
```

## Optimizations

### 1. Flash Attention (When Available)

```python
if has_flash_attention:
    attn_output = flash_attn_func(
        Q, K, V,
        dropout_p=0.1,
        causal=False
    )
else:
    # Standard attention
    attn_output = standard_attention(Q, K, V)
```

### 2. Gradient Checkpointing

```python
if training and use_checkpointing:
    # Trade compute for memory
    hidden = checkpoint(layer_forward, hidden, *args)
else:
    hidden = layer_forward(hidden, *args)
```

### 3. Mixed Precision

```python
# Use bfloat16 for forward pass
with autocast(dtype=torch.bfloat16):
    output = model(input)

# But compute loss in float32
loss = loss_fn(output.float(), target)
```

## Layer Initialization

### Weight Initialization

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        # Truncated normal initialization
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=1/sqrt(hidden_size))
```

### Special Initializations

```python
# H and L initial states
self.H_init = trunc_normal_(empty(hidden_size), std=1)
self.L_init = trunc_normal_(empty(hidden_size), std=1)

# Q-head bias (prefer continuing)
self.q_head.bias.fill_(-5)
```

## Information Bottlenecks

### 1. Fixed Hidden Size
All information must flow through 256-dimensional vectors:
- Limits representational capacity
- Forces efficient encoding
- Prevents overfitting

### 2. Recurrent State
Information persists only through H/L states:
- No external memory
- Must compress history
- Enables generalization

### 3. Communication Bandwidth
H↔L communication is linear projection:
- Simple transformation
- Limited expressiveness
- Could benefit from attention

## Comparison with Standard Architectures

| Aspect | HRM | GPT-style | BERT-style |
|--------|-----|-----------|------------|
| Depth | 7 blocks × 8 cycles | 12-48 blocks | 12-24 blocks |
| Attention | Bidirectional | Causal | Bidirectional |
| Processing | Recurrent | Single-pass | Single-pass |
| Parameters | 6.95M | 100M-175B | 100M-340M |
| Specialization | Visual reasoning | Language | Language |

## Key Insights

1. **Depth through recurrence**: Achieves deep processing without deep architecture
2. **Hierarchical separation**: Different layers for different reasoning levels
3. **Efficient design**: Every component serves specific purpose
4. **Bottleneck learning**: Constraints force meaningful representations
5. **Simplicity wins**: Basic components combined cleverly outperform complexity