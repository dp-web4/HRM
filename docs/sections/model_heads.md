# Model Heads and Output Layers

*Last Updated: September 2025*

## Overview

HRM uses multiple specialized output heads for different objectives: token prediction (LM head), halting decisions (Q heads), and optional puzzle embeddings. Each head serves a specific purpose in the model's reasoning and decision-making process.

## Head Architecture

### 1. Language Model Head (Primary Output)

```python
self.lm_head = CastedLinear(
    hidden_size,     # 256
    vocab_size,      # 12 for ARC (0-9 + special tokens)
    bias=False
)
```

**Purpose**: Predicts next token in sequence
**Input**: H-module final state
**Output**: Logits over vocabulary

**Key Details:**
- No bias term (following modern LM design)
- Shares embeddings with input in some variants
- Cast to appropriate dtype (bfloat16 typically)

### 2. Q-Value Heads (Halting Control)

```python
self.q_head = CastedLinear(
    hidden_size,     # 256
    2,               # [halt_value, continue_value]
    bias=True
)
```

**Purpose**: Estimates value of halting vs continuing
**Input**: First position of H-state (CLS-like token)
**Output**: Two Q-values for action selection

**Special Initialization:**
```python
# Initialize to prefer continuing (avoid premature halting)
with torch.no_grad():
    self.q_head.weight.zero_()
    self.q_head.bias.fill_(-5)  # Strong bias toward continuing
```

### 3. Puzzle Embedding (Optional)

```python
if puzzle_emb_ndim > 0:
    self.puzzle_emb = CastedSparseEmbedding(
        num_puzzle_identifiers,  # Number of unique puzzles
        puzzle_emb_ndim,         # Embedding dimension
        batch_size,
        init_std=0,              # Zero initialization
        cast_to=forward_dtype
    )
```

**Purpose**: Learns puzzle-specific patterns
**Special Feature**: Sparse embeddings for memory efficiency

## Output Processing Pipeline

### Step 1: Extract Hidden States
```python
# After final reasoning cycle
z_H = final_h_state  # [batch, seq_len, hidden_size]
z_L = final_l_state  # [batch, seq_len, hidden_size]
```

### Step 2: Generate Predictions
```python
# Token predictions from H-state
logits = self.lm_head(z_H)

# Remove puzzle embedding positions
output = logits[:, puzzle_emb_len:]  # Skip puzzle prefix
```

### Step 3: Compute Q-Values
```python
# Extract first position for decision
cls_state = z_H[:, 0]  # [batch, hidden_size]

# Compute Q-values
q_logits = self.q_head(cls_state)  # [batch, 2]
q_halt = q_logits[..., 0]
q_continue = q_logits[..., 1]
```

### Step 4: Make Halting Decision
```python
# During training: exploratory
should_halt = (q_halt > q_continue) & (steps >= min_steps)

# During inference: deterministic
should_halt = (steps >= max_steps)
```

## Head-Specific Design Choices

### Why No Bias in LM Head?

Modern LMs typically omit bias in final projection:
- Reduces parameters (vocab_size fewer params)
- Embeddings can learn appropriate offsets
- Slightly better generalization

### Why Separate Q-Values?

Instead of single halt probability:
```python
# Not used:
halt_prob = sigmoid(self.halt_head(state))

# Used instead:
q_halt, q_continue = self.q_head(state)
halt = q_halt > q_continue
```

**Benefits:**
- Explicit value comparison
- Better gradient flow
- Natural for Q-learning framework

### Why Initialize Q-Head to -5?

```python
self.q_head.bias.fill_(-5)
# After sigmoid: ~0.007 probability
```

**Reasoning:**
- Prevents premature halting during early training
- Model must learn when halting is valuable
- Gradual transition from max steps to adaptive

## Output Tensor Shapes

```python
# Assuming batch_size=8, seq_len=900, vocab_size=12

# Forward pass
outputs = model(input_batch)

# Output shapes
logits:            [8, 900, 12]   # Token predictions
q_halt_logits:     [8]            # Halt values
q_continue_logits: [8]            # Continue values
halted:            [8]            # Boolean halt flags
steps:             [8]            # Steps taken
```

## Multi-Head Variants

### Nova's Enhanced Version
Adds specialized heads for different aspects:

```python
class EnhancedHeads(nn.Module):
    def __init__(self, hidden_size):
        # Standard heads
        self.lm_head = Linear(hidden_size, vocab_size)
        self.q_head = Linear(hidden_size, 2)
        
        # Additional heads
        self.confidence_head = Linear(hidden_size, 1)
        self.attention_head = Linear(hidden_size, seq_len)
```

### SAGE 100M Proposal
Includes multi-modal output heads:

```python
class SAGEHeads(nn.Module):
    def __init__(self, hidden_size):
        # Language outputs
        self.token_head = Linear(hidden_size, vocab_size)
        
        # Vision outputs  
        self.pixel_head = Linear(hidden_size, 256)  # 8-bit RGB
        
        # Action outputs
        self.action_head = Linear(hidden_size, n_actions)
        
        # Meta-reasoning
        self.halt_head = Linear(hidden_size, 2)
        self.confidence_head = Linear(hidden_size, 1)
```

## Head Gradient Analysis

### LM Head Gradients
- **Largest gradients**: From incorrect predictions
- **Gradient scale**: ~O(1/vocab_size)
- **Update frequency**: Every training step

### Q-Head Gradients
- **Sparse gradients**: Only when halted
- **Gradient scale**: ~O(1) (binary target)
- **Update frequency**: When sequences complete

### Balancing Updates
```python
# Different learning rates for different heads
optimizer = torch.optim.AdamW([
    {'params': model.lm_head.parameters(), 'lr': 1e-4},
    {'params': model.q_head.parameters(), 'lr': 5e-4},  # Higher LR
    {'params': other_params, 'lr': 3e-4}
])
```

## Output Post-Processing

### Token Generation
```python
def generate_tokens(logits, temperature=1.0):
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Sample or greedy
    if sampling:
        probs = softmax(scaled_logits)
        tokens = multinomial(probs)
    else:
        tokens = argmax(scaled_logits)
    
    return tokens
```

### Confidence Estimation
```python
def estimate_confidence(logits, q_values):
    # Token confidence from softmax entropy
    probs = softmax(logits)
    entropy = -(probs * log(probs)).sum(-1)
    token_conf = 1 - (entropy / log(vocab_size))
    
    # Halt confidence from Q-value gap
    q_gap = abs(q_halt - q_continue)
    halt_conf = sigmoid(q_gap)
    
    return token_conf * halt_conf
```

## Common Issues and Solutions

### Issue 1: Degenerate LM Head
**Symptom**: Always predicts same token
**Cause**: Collapsed representations
**Solution**: 
- Add dropout before head
- Increase weight decay
- Check for gradient issues

### Issue 2: Q-Head Ignored
**Symptom**: Always continues/halts
**Cause**: Q-values not learning
**Solution**:
- Increase Q-loss weight
- Add exploration bonus
- Check initialization

### Issue 3: Head Imbalance
**Symptom**: One head dominates loss
**Cause**: Different gradient scales
**Solution**:
- Use separate optimizers
- Gradient clipping per head
- Adaptive loss weighting

## Future Directions

### Proposed Enhancements

1. **Mixture of Experts Heads**
```python
# Multiple specialized heads
heads = [LinearHead(hidden, vocab) for _ in range(n_experts)]
router = Linear(hidden, n_experts)

# Dynamic selection
weights = softmax(router(state))
output = sum(w * h(state) for w, h in zip(weights, heads))
```

2. **Hierarchical Heads**
```python
# Coarse-to-fine prediction
coarse_head = Linear(hidden, n_clusters)
fine_heads = {i: Linear(hidden, cluster_size) for i in range(n_clusters)}
```

3. **Adaptive Heads**
```python
# Head that adapts to puzzle type
base_head = Linear(hidden, vocab)
adapt_params = Linear(puzzle_emb, hidden * vocab)
adapted_weight = base_weight + adapt_params.view(hidden, vocab)
```

## Key Takeaways

1. **Heads are specialized** - Each serves a specific purpose
2. **Initialization matters** - Especially for Q-heads
3. **Balance is critical** - Between different objectives
4. **Simplicity works** - Basic linear projections suffice
5. **Future potential** - Room for more sophisticated designs