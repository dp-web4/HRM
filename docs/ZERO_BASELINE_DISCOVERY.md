# Zero-Baseline Discovery: The 20% Accuracy Revelation

*Date: September 4, 2025*
*Session: Debugging Kaggle submission predictions*

## The Mystery

We had conflicting results:
- `evaluate_arc_agi2_correct.py` reported **18-27% accuracy** on ARC-AGI-2
- But `kaggle_submission.py` produced **all zero predictions**
- How could the same model have such different performance?

## The Investigation

### Step 1: Verify Model Loading
```python
# Both scripts loaded the same checkpoint
checkpoint = torch.load('hrm_arc_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
# Checkpoint from epoch 0, step 7000
```
✅ Model loaded correctly in both scripts

### Step 2: Compare Architectures
- Both use identical HRM architecture with H↔L layers
- Both return tuple `(output, halt_probs)`
- Parameter count matches: 5,667,597
✅ Architecture identical

### Step 3: Compare Preprocessing
```python
# Both scripts:
# 1. Pad input to 30x30 with zeros
# 2. Flatten to 900-length sequence
# 3. Create tensor with dtype=torch.long
```
✅ Preprocessing identical

### Step 4: Compare Outputs
```python
# Test input: [[3, 2], [7, 8]]
# Kaggle predictions: [0, 0, 0, 0, 0, ...]
# Eval predictions:   [0, 0, 0, 0, 0, ...]
```
✅ Both produce all zeros!

## The Revelation

The "accuracy" was never real! Here's what happened:

### Task Analysis
```python
# Task 136b0064 - reported 78.95% accuracy
Ground truth output shape: 19x7
Zeros in ground truth: 105/133 = 78.9%

# If we predict all zeros:
Accuracy = 78.9% (matches reported accuracy exactly!)
```

### Full Results Breakdown
| Task ID | Reported Acc | Zero-Baseline | Match |
|---------|-------------|---------------|-------|
| 0934a4d8 | 0.0% | 0.0% | ✅ |
| 135a2760 | 0.0% | 0.0% | ✅ |
| 136b0064 | 78.9% | 78.9% | ✅ |
| 13e47133 | 0.0% | 0.0% | ✅ |
| 142ca369 | 56.4% | 56.4% | ✅ |

**Perfect match!** The model outputs all zeros, and the "accuracy" is just how many pixels happen to be zero in the ground truth.

## What This Means

### The Model State
- **Checkpoint step 7000 is too early** - model hasn't learned output generation
- **Output layer is biased** - always predicts class 0 (highest logit: 2.49)
- **Architecture works** - model runs correctly, just needs training

### The Metrics
- **71% on ARC-AGI-1**: Might also need verification
- **20% on ARC-AGI-2**: Is purely zero-baseline, not real solving
- **True performance**: 0% (no tasks actually solved)

## Debugging Process

### What We Checked
1. ✅ Model weights loaded (non-zero values confirmed)
2. ✅ Model produces different logits per position
3. ✅ Class 0 has highest logit everywhere
4. ✅ Ground truth analysis confirms zero-baseline

### Key Debug Commands
```python
# Check model output distribution
output, halt_probs = model(input_tensor)
print(f'Output shape: {output.shape}')  # [1, 900, 12]
print(f'First logits: {output[0, 0, :]}')  # [2.49, -0.95, ...]

# Class 0 dominates
predictions = output.argmax(dim=-1)
print(f'Unique predictions: {torch.unique(predictions)}')  # [0]

# Verify zero-baseline
zeros_in_gt = sum(1 for val in flat if val == 0)
baseline_acc = zeros_in_gt / total
```

## Lessons Learned

### 1. Always Verify Non-Baseline Performance
```python
# Good practice: Check if accuracy beats simple baselines
baseline_acc = calculate_zero_baseline(ground_truth)
model_acc = evaluate_model(predictions, ground_truth)
assert model_acc > baseline_acc, "Model not beating baseline!"
```

### 2. Checkpoint Selection Matters
- Early checkpoints may not have learned the task
- Step 7000 is apparently before output generation learning
- Need to identify when model actually starts solving tasks

### 3. Misleading Metrics
- Pixel accuracy can be deceiving
- 78.9% sounds impressive but means nothing if it's baseline
- Need task-level success metrics, not just pixel accuracy

### 4. Architecture vs Training
- **Good news**: Architecture is correct and working
- **Bad news**: Training hasn't produced a solving model yet
- **Next step**: Find better checkpoint or continue training

## The Silver Lining

This discovery is actually valuable:
1. **We know exactly what's wrong** - training, not architecture
2. **Kaggle submission works** - infrastructure is ready
3. **Clear path forward** - need better training/checkpoint
4. **Architecture validated** - H↔L design loads and runs correctly

## Next Steps

1. **Find later checkpoints** if available (step > 7000)
2. **Continue training** specifically on output reconstruction
3. **Implement few-shot prompting** with training examples
4. **Add validation metrics** that detect baseline performance
5. **Track when model starts** actually solving tasks

## Code Snippets for Future

### Detect Zero-Baseline Performance
```python
def is_baseline_only(model, test_tasks, threshold=0.1):
    """Check if model is only achieving baseline accuracy"""
    for task in test_tasks:
        pred = model(task['input'])
        
        # Check if predictions are too uniform
        unique_preds = torch.unique(pred)
        if len(unique_preds) == 1:
            print(f"WARNING: Model only predicting class {unique_preds[0]}")
            return True
            
        # Check if accuracy matches baseline
        baseline = calculate_baseline(task['output'])
        accuracy = calculate_accuracy(pred, task['output'])
        if abs(accuracy - baseline) < threshold:
            print(f"WARNING: Accuracy {accuracy:.1%} matches baseline {baseline:.1%}")
            return True
    
    return False
```

### Monitor Training Progress
```python
def log_solving_metrics(model, val_tasks, step):
    """Track when model starts actually solving tasks"""
    metrics = {
        'step': step,
        'pixel_accuracy': 0,
        'baseline_accuracy': 0,
        'above_baseline': 0,
        'perfect_solves': 0,
        'unique_predictions': set()
    }
    
    for task in val_tasks:
        pred = model(task['input'])
        gt = task['output']
        
        # Track prediction diversity
        metrics['unique_predictions'].update(torch.unique(pred).tolist())
        
        # Calculate metrics
        acc = calculate_accuracy(pred, gt)
        baseline = calculate_zero_baseline(gt)
        
        metrics['pixel_accuracy'] += acc
        metrics['baseline_accuracy'] += baseline
        
        if acc > baseline + 0.1:  # Significantly above baseline
            metrics['above_baseline'] += 1
        
        if acc == 1.0:  # Perfect solve
            metrics['perfect_solves'] += 1
    
    print(f"Step {step}: Above baseline: {metrics['above_baseline']}/{len(val_tasks)}")
    print(f"  Perfect solves: {metrics['perfect_solves']}")
    print(f"  Unique predictions: {metrics['unique_predictions']}")
```

## Conclusion

The journey from "20% accuracy" to "0% actual solving" taught us:
- **Question impressive numbers** - they might be baseline
- **Verify model behavior** - not just metrics
- **Architecture works** - implementation is solid
- **Training is key** - need the right checkpoint

This is excellent learning and positions us well for the next phase: getting the model to actually solve ARC tasks!

---

*"Sometimes the most valuable discoveries come from finding out what's NOT working."*