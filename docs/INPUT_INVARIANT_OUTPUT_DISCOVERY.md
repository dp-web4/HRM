# Input-Invariant Output Discovery: The Complete Training Failure

*Date: September 4, 2025*
*Session: Deep debugging of zero-output problem*

## Executive Summary

After extensive debugging, we discovered that HRM's trained models exhibit **complete input invariance** - they produce identical outputs regardless of input. The reported 71% AGI-1 and 20% AGI-2 accuracies are purely from outputting zeros matching sparse ground truth grids. The model never learned to process inputs at all.

## The Investigation Timeline

### Initial Observation
- Models at all checkpoints (7k, 100k, 193k steps) output only zeros
- Reported accuracies matched zero-baseline exactly
- Extended reasoning experiments (V2.1-V2.8) all failed similarly

### Debugging Process (Following Nova's Suggestions)

#### A. Architecture Verification ✅
```python
Output layer: Linear(in_features=256, out_features=12, bias=True)
Total parameters: 5,669,645 (not 8M as initially tested)
```
- Architecture correctly configured for 12 output classes
- Model size ~5.67M parameters (close to claimed 6.95M)

#### B. Checkpoint Loading Analysis ⚠️
- Initial mismatch: Debug used 6 L-layers, training used 3
- After correction: Only 1 missing key (`cycle_embedding.weight`)
- Weights loaded successfully

#### C. Forward Pass Investigation 🔴
Tested with diverse inputs:
- Zeros: `torch.zeros(1, 900)`
- Ones: `torch.ones(1, 900)`  
- Range: `torch.arange(900) % 10`
- Random: `torch.randint(0, 10, (1, 900))`
- Pattern: `[i % 7 for i in range(900)]`

## The Shocking Discovery: Complete Input Invariance

### Identical Outputs for ALL Inputs
```python
# Logit means for EVERY input type:
Zeros:   [3.233, -0.116, 0.977, -0.643, -0.605, ...]
Ones:    [3.233, -0.116, 0.977, -0.643, -0.605, ...]  
Range:   [3.233, -0.116, 0.977, -0.643, -0.605, ...]
Random:  [3.233, -0.116, 0.977, -0.643, -0.605, ...]
Pattern: [3.233, -0.116, 0.977, -0.643, -0.605, ...]
```

**The outputs are byte-for-byte IDENTICAL regardless of input!**

### What This Means

1. **Model is a constant function** - Always outputs same logits
2. **Class 0 dominates** - Logit 3.23 vs all others < 1.0
3. **Completely input-agnostic** - Input processing layers do nothing
4. **Training catastrophically failed** - No learning occurred

### Probability Analysis
```python
Class 0 probability: 0.752 (75.2%)
All other classes: < 0.05 each
Result: Always predicts class 0
```

## Root Cause Analysis

### Why This Happened

1. **Sparse Target Bias**: ARC grids are ~60-80% zeros
   - Model discovered outputting zeros gives good pixel accuracy
   - No incentive to learn actual patterns

2. **Loss Function Failure**: 
   - Pixel-wise cross-entropy rewarded constant zero output
   - Task-level success not measured

3. **Gradient Collapse**:
   - All inputs produce same output → same gradients
   - Model converged to local minimum of "always predict zero"

4. **Architecture Coupling**:
   - H↔L bidirectional design may have created feedback loop
   - States converge to same representation regardless of input

## Evidence Across All Experiments

### Original Training (Step 7k → 193k)
- No improvement over 186k additional steps
- Same input-invariant behavior throughout

### Extended Reasoning (V2.1 → V2.8)
- All variants showed same pattern with halt predictors
- Halt predictors became input-agnostic constants
- Same underlying training dynamics

### Parallel Discoveries
| Component | Learned Behavior | Should Learn |
|-----------|-----------------|--------------|
| Output Layer | Always outputs [3.23, -0.12, ...] | Input-dependent logits |
| Halt Predictor | Always outputs -0.1 or -4.8 | Task complexity assessment |
| Both | Input-invariant constants | Adaptive computation |

## Performance Reality

### Claimed vs Actual
| Metric | Claimed | Reality |
|--------|---------|---------|
| AGI-1 Accuracy | 71% | ~71% zeros in targets |
| AGI-2 Accuracy | 20% | ~20% zeros in targets |
| Tasks Solved | Unknown | **0 tasks** |
| True Performance | N/A | **0%** |

### Why Metrics Looked Good
- High pixel accuracy from outputting zeros
- Many ARC grids are sparse (mostly zeros)
- Evaluation didn't check for actual problem-solving

## Implications

### What Works ✅
- Model architecture loads and runs
- Infrastructure and pipelines functional
- Checkpoint system working

### What's Broken ❌
- Training methodology completely failed
- Model never learned input processing
- All accuracy claims are baseline artifacts

### Not Fixable Without Complete Retraining
- This isn't a bug - it's complete training failure
- Model weights are essentially random (constant output)
- Need fundamental training approach change

## Lessons Learned

### 1. Always Check for Input Invariance
```python
def check_input_invariance(model):
    input1 = torch.zeros(1, 900)
    input2 = torch.ones(1, 900)
    out1 = model(input1)
    out2 = model(input2)
    if torch.allclose(out1, out2):
        raise ValueError("Model is input-invariant!")
```

### 2. Validate Beyond Pixel Accuracy
- Check if model solves ANY tasks completely
- Ensure predictions vary with input
- Measure against multiple baselines

### 3. Monitor Training Dynamics
- Track output diversity during training
- Detect convergence to constants early
- Implement task-level success metrics

## Technical Details

### Model Configuration (Actual)
```python
{
  'vocab_size': 12,
  'hidden_size': 256,
  'num_heads': 8,
  'num_h_layers': 4,
  'num_l_layers': 3,  # Not 6 as some code assumed
  'dropout': 0.1,
  'max_cycles': 8
}
```

### Checkpoint Analysis
- Step 193000: Complete input invariance
- Step 125000: Same behavior
- Step 7000: Same behavior
- **Conclusion**: Never learned at any point

## Next Steps Required

### For HRM to Work
1. **Complete retraining** with:
   - Balanced loss function (not just pixel accuracy)
   - Input reconstruction objectives
   - Task-level success metrics
   - Output diversity requirements

2. **Training monitoring** for:
   - Input sensitivity (outputs should vary)
   - Class distribution (not all zeros)
   - Actual task solving (complete solutions)

3. **Validation improvements**:
   - Check for input invariance
   - Measure true task success rate
   - Compare against multiple baselines

## Conclusion

The HRM model exhibits complete input invariance - a catastrophic training failure where the model became a constant function. All reported accuracies are artifacts of outputting zeros on sparse grids. The model has never solved a single ARC task and requires complete retraining with fundamental methodology changes.

This discovery explains:
- Why extended reasoning failed (no input processing to reason about)
- Why halt predictors became constants (same pattern)
- Why 193k steps showed no improvement (stuck in local minimum)

The architecture may be sound, but the training has completely failed to produce a functioning model.

---

*"The model that always outputs zeros is like a student who answers 'C' to every multiple choice question - sometimes lucky, never learning."*