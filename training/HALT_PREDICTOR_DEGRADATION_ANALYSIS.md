# Halt Predictor Degradation Analysis

## The Systematic Degradation Pattern

From V2.4 debug logs, we see a clear pattern:

```
Step ~100:  halt_logit=-3.584, bias=0.900 → halt_prob=0.064
Step ~300:  halt_logit=-4.055, bias=0.900 → halt_prob=0.041
Step ~500:  halt_logit=-4.297, bias=0.900 → halt_prob=0.032
Step ~800:  halt_logit=-5.004, bias=0.900 → halt_prob=0.016
Step ~1000: halt_logit=-5.684, bias=0.900 → halt_prob=0.008
```

**The halt predictor is systematically becoming MORE negative during training.**

## Why This Happens: The Adversarial Optimization

The model discovers that:

1. **Lower halt probabilities = More cycles**
2. **More cycles = Better task performance** (initially)
3. **Better task performance = Lower loss**
4. **Therefore: Push halt_logit more negative = Win**

The halt predictor becomes adversarial to our bias - it "fights" the +0.9 bias by going more negative!

## The Fundamental Training Conflict

Our loss function has two competing objectives:

### Task Loss (Dominant)
```python
task_loss = F.cross_entropy(outputs, targets)  # LARGE values ~2-4
```

### ACT Loss (Weak)
```python
act_loss = cycle_weight * p.mean() * 0.01  # TINY values ~0.01
```

**The 0.01 coefficient makes ACT loss negligible compared to task loss!**

## The Gradient Dynamics

When task loss decreases with more cycles, gradients flow:
- `∂task_loss/∂halt_logit > 0` (want LOWER halt prob for more cycles)
- `∂act_loss/∂halt_logit < 0` (want HIGHER halt prob for fewer cycles)
- But `|task_loss| >> |act_loss|`, so task gradients dominate

**Result**: Halt predictor learns to output increasingly negative values to maximize cycles and minimize dominant task loss.

## The Cycle Escalation

We observed adaptive cycles (4→12), but this is NOT healthy adaptation:

- Model uses 4 cycles when halt_logit ≈ -3.5 (prob ≈ 0.06)
- Model uses 12 cycles when halt_logit ≈ -6.0 (prob ≈ 0.003)
- The "adaptation" is actually degradation - it's losing ability to halt!

## Solutions

### 1. Stronger ACT Loss Weight
```python
act_loss = cycle_weight * p.mean() * 0.1  # 10x stronger
```

### 2. Logit Regularization
```python
# Penalize extreme logits directly
logit_penalty = torch.mean(torch.abs(halt_logit)) * 0.05
total_loss = task_loss + act_loss + logit_penalty
```

### 3. Bounded Output (Our V2.5 approach)
```python
# Force output to reasonable range
normalized_logit = torch.tanh(halt_logit)  # Bound to [-1, 1]
```

### 4. Separate Optimization
```python
# Different learning rates for different components
halt_optimizer = torch.optim.AdamW(model.halt_predictor.parameters(), lr=1e-6)  # Much slower
task_optimizer = torch.optim.AdamW(other_params, lr=5e-5)  # Normal speed
```

## Key Insight: Competition vs Cooperation

The current architecture creates **competition** between:
- Task performance (wants more cycles)
- Efficiency (wants fewer cycles)

We need **cooperation** where:
- Easy tasks naturally use fewer cycles
- Hard tasks naturally use more cycles
- No adversarial dynamics

## The Real Fix

The model needs to learn that:
1. **Some tasks are genuinely easier** and should halt early
2. **Using appropriate cycles improves generalization** 
3. **Efficiency is a feature, not a bug**

This suggests we need:
- **Task difficulty awareness** in the halt predictor
- **Reward for correct cycle usage** (not just task accuracy)
- **Training examples** that demonstrate efficient solving

## Biological Parallel

Humans don't "fight" their stopping criteria - we naturally:
- Solve easy problems quickly
- Spend more time on hard problems
- Feel confident when we've found the answer

The halt predictor should mirror this natural confidence, not fight it.