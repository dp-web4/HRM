# Loss Functions and Training Objectives

*Last Updated: September 2025*

## Overview

HRM uses a multi-objective loss function that combines language modeling, halting decisions, and Q-learning for adaptive computation. This sophisticated loss design is critical for training the model to both solve puzzles AND know when to stop thinking.

## Loss Components

### 1. Language Modeling Loss (Primary)

The main loss trains the model to predict correct output tokens:

```python
lm_loss = cross_entropy(logits, labels, ignore_index=-100)
```

**Key Details:**
- Uses either softmax or "stablemax" cross-entropy
- Ignores padding tokens (label=-100)
- Applied only to non-padding positions
- Normalized by sequence length to prevent length bias

**From losses.py:**
```python
def softmax_cross_entropy(logits, labels, ignore_index=-100):
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]), 
        labels.to(torch.long).view(-1), 
        ignore_index=ignore_index, 
        reduction="none"
    ).view(labels.shape)
```

### 2. Halting Loss (Q-learning)

Trains the model to know when to stop reasoning:

```python
q_halt_loss = binary_cross_entropy(q_halt_logits, seq_is_correct)
```

**The Logic:**
- If sequence is correct → should halt (positive signal)
- If sequence is wrong → should continue (negative signal)
- Uses Q-learning without replay buffer (parallel environments)

### 3. Continue Loss (Bootstrap Target)

Helps stabilize Q-learning through bootstrapping:

```python
q_continue_loss = binary_cross_entropy(q_continue_logits, target_q_continue)
```

**Bootstrap Calculation:**
```python
# Next step Q-values
next_q_halt, next_q_continue = self.inner(carry, data)[-1]

# Target: max of halt/continue for next step
target_q_continue = sigmoid(max(next_q_halt, next_q_continue))
```

## Combined Loss Function

The total loss combines all objectives:

```python
total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
```

**Weighting Rationale:**
- LM loss is primary (weight 1.0)
- Halting losses are auxiliary (weight 0.5 each)
- Prevents halting from dominating training

## The Stablemax Alternative

HRM includes an experimental "stablemax" activation:

```python
def s(x, epsilon=1e-30):
    return torch.where(
        x < 0,
        1/(1-x + epsilon),  # For negative values
        x + 1                # For positive values
    )

def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))
```

**Purpose:**
- More stable gradients than softmax
- Better numerical properties for small models
- Experimental - not always used

## Training Metrics

The loss head tracks several metrics:

```python
metrics = {
    "count": valid_sequences,
    "accuracy": per_token_accuracy,
    "exact_accuracy": full_sequence_accuracy,
    "q_halt_accuracy": halt_decision_accuracy,
    "steps": average_reasoning_steps,
    "lm_loss": language_modeling_loss,
    "q_halt_loss": halting_loss,
    "q_continue_loss": bootstrap_loss
}
```

## ACT (Adaptive Computation Time) Integration

The loss function deeply integrates with ACT:

### During Training
```python
if self.training and (halt_max_steps > 1):
    # Decide whether to halt
    halted = (q_halt_logits > q_continue_logits)
    
    # Exploration: randomly force minimum steps
    min_steps = random(2, halt_max_steps)
    halted = halted & (steps >= min_steps)
```

### During Evaluation
```python
# Always use maximum steps for consistent batching
halted = (steps >= halt_max_steps)
```

## Gradient Flow Strategy

### Truncated Backpropagation
Only the final reasoning cycle receives gradients:

```python
with torch.no_grad():
    # Cycles 1 to N-1: no gradients
    for cycle in range(n_cycles - 1):
        z_L = L_level(z_L, z_H + input)
        z_H = H_level(z_H, z_L)

# Cycle N: gradients flow
z_L = L_level(z_L, z_H + input)  # Gets gradients
z_H = H_level(z_H, z_L)          # Gets gradients
```

**Benefits:**
- Prevents gradient explosion
- Reduces memory usage
- Focuses learning on final refinement

## Loss Computation Pipeline

```python
class ACTLossHead(nn.Module):
    def forward(self, **kwargs):
        # 1. Forward pass through model
        new_carry, outputs = self.model(**kwargs)
        
        # 2. Extract predictions and labels
        logits = outputs["logits"]
        labels = new_carry.current_data["labels"]
        
        # 3. Compute correctness (no gradients)
        with torch.no_grad():
            is_correct = (argmax(logits) == labels)
            seq_correct = all_tokens_correct(is_correct)
        
        # 4. Compute losses (with gradients)
        lm_loss = cross_entropy(logits, labels)
        q_halt_loss = bce(q_halt_logits, seq_correct)
        q_continue_loss = bce(q_continue_logits, target_q)
        
        # 5. Combine and return
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        return new_carry, total_loss, metrics, outputs, halted
```

## Special Considerations

### 1. Ignore Label Handling
```python
IGNORE_LABEL_ID = -100
mask = labels != IGNORE_LABEL_ID
loss_counts = mask.sum(-1)
loss_divisor = loss_counts.clamp_min(1)  # Avoid div by zero
```

### 2. Per-Sequence Normalization
```python
# Normalize by actual sequence length
normalized_loss = raw_loss / loss_divisor
```

### 3. Halt Exploration
During training, random exploration prevents local optima:
```python
exploration_prob = 0.1
force_explore = random() < exploration_prob
min_steps = randint(2, max_steps) if force_explore else 1
```

## Loss Curves and Training Dynamics

### Typical Training Pattern
1. **Early phase**: LM loss dominates, halting is random
2. **Middle phase**: Halting begins to correlate with correctness
3. **Late phase**: Fine-tuning of halt timing for efficiency

### Common Issues

#### Loss Explosion
**Symptom**: Loss suddenly jumps to inf/nan
**Cause**: Gradient accumulation through many cycles
**Solution**: Reduce learning rate or clip gradients

#### Halt Collapse
**Symptom**: Model always halts immediately
**Cause**: Q-halt loss dominates
**Solution**: Reduce halt loss weight or increase exploration

#### No Halting
**Symptom**: Model always uses max steps
**Cause**: Q-continue always wins
**Solution**: Increase halt reward for correct sequences

## Configuration Parameters

```python
# From training configs
{
    'halt_max_steps': 8,           # Maximum reasoning cycles
    'halt_exploration_prob': 0.1,  # Random exploration rate
    'gradient_clip': 1.0,           # Gradient clipping threshold
    'loss_weights': {
        'lm': 1.0,                  # Language modeling weight
        'halt': 0.5,                # Halting decision weight
        'continue': 0.5             # Bootstrap target weight
    }
}
```

## Connection to Agent Zero

The loss function design directly relates to the Agent Zero problem:
- If LM loss rewards outputting zeros → Agent Zero
- If halt loss is too weak → No adaptive computation
- If exploration is insufficient → Gets stuck in local optima

The careful balance of these losses determines whether the model:
1. Learns to reason (good)
2. Learns shortcuts (Agent Zero)
3. Fails to converge (unstable)