# Extended Reasoning Debugging Journey

## Background
Attempting to extend HRM's reasoning cycles from 8 → 20 to enable deeper problem-solving on complex ARC tasks. The original model achieved 49.1% on ARC-AGI-1 with fixed 8 cycles.

## The Journey: 6 Major Iterations

### V1: Initial Extended Reasoning (FAILED)
- **Goal**: Simply increase max_cycles from 8 to 20
- **Result**: Catastrophic failure (49.1% → 8.1%)
- **Discovery**: Halt predictor probabilities DECREASING over cycles (0.487→0.266)
- **Root cause**: Halt predictor learned to always use maximum cycles

### V2: Bias-Based Fix (FAILED)
- **Goal**: Add cycle-dependent bias to encourage halting
- **Implementation**: Bias from -2.0 → +3.0 over 20 cycles
- **Result**: Still used all 20 cycles with halt_p=0.000
- **Discovery**: Bias was too aggressive, preventing any halting

### V2.1: Gentler Bias (FAILED)  
- **Goal**: Reduce bias aggression (-1.0 → +2.0)
- **Result**: Still used all 20 cycles
- **Pattern**: Model fighting our bias system

### V2.2: Ultra-Gentle Bias (FAILED)
- **Goal**: Minimal bias (-0.2 → +0.9) as user specified
- **Result**: STILL used all 20 cycles with halt_p=0.000
- **Critical insight**: Math should guarantee minimum halt_p=0.711 at cycle 19

### V2.4: Reset Halt Predictor (PARTIAL SUCCESS)
- **Discovery**: Halt predictor learned extreme negative logits (-5.684)
- **Fix**: Reinitialize halt predictor weights
- **Result**: Started with 4 cycles, BUT predictor degraded again during training
- **Pattern**: halt_logit went -3.584 → -5.684 (increasingly negative)

### V2.6: Efficiency-Rewarding Loss (CURRENT)
- **Core insight**: Predictor extends cycles because task_loss >> act_loss (400:1 ratio)
- **Fix**: Reward efficient correctness with efficiency bonuses/penalties
- **Implementation**: 
  - Correct + ≤6 cycles: -0.5 loss bonus
  - Correct + >15 cycles: +0.1 penalty  
  - Wrong + max cycles: +0.2 penalty
- **Result**: Stable 4 cycles, BUT zero variability (suspicious)

## Current Status: V2.6 Analysis

### ✅ Achievements
- **Eliminated adversarial optimization**: No more cycle escalation
- **Stable halt predictor**: Logits stay around -0.1 (reasonable)
- **Fast training**: 24 it/s vs 2 it/s in broken versions
- **Efficiency rewards working**: Model earning bonuses for 4-cycle solutions

### ⚠️ Critical Discovery
**The halt predictor is input-agnostic!**

```
Simple pattern:  logit = -0.102
Random pattern:  logit = -0.101  
Complex pattern: logit = -0.101
All ones:        logit = -0.101
Mixed pattern:   logit = -0.102
```

**It completely ignores input complexity and always outputs the same value!**

### 🔍 The Real Problem
1. **Dead predictor**: Always outputs ~-0.1 regardless of input
2. **Cumulative threshold too low**: 1.0 threshold with ~0.45 per cycle = always stops at cycle 4
3. **No task difficulty awareness**: Model never learned to evaluate complexity

### 📊 Current Performance
- **V2.6**: 9.2% on ARC-AGI-1 (4 cycles always)
- **Original**: 49.1% on ARC-AGI-1 (8 cycles fixed)
- **Trade-off**: Lost 5x performance for adaptive computation that isn't actually adaptive

## Key Insights from Failures

### 1. Adversarial Optimization (V2.4 discovery)
When task_loss >> act_loss, the model learns to fight efficiency constraints:
- Halt predictor outputs extreme negatives to force maximum cycles
- More cycles → better task performance → lower overall loss
- Classic misaligned incentives problem

### 2. Efficiency Rewards Work (V2.6 success)  
Properly aligned rewards eliminate adversarial behavior:
- Model no longer fights our constraints
- Stable, consistent training
- BUT may have overcorrected to constant behavior

### 3. Input-Agnostic Learning (Current issue)
The halt predictor never learned to evaluate actual task complexity:
- Outputs same value for all inputs
- Suggests it learned a "safe" constant that satisfies our stopping condition
- No real reasoning about when to halt

## Proposed Fixes

### Fix 1: Task Difficulty Training
Train the halt predictor explicitly on task difficulty:
```python
# Add auxiliary loss for halt predictor accuracy
difficulty_labels = estimate_task_difficulty(inputs, targets)
halt_difficulty_loss = F.mse_loss(halt_logit, difficulty_labels)
```

### Fix 2: Contrastive Halt Training
Train on pairs of easy/hard tasks:
```python
# Easy tasks should have higher halt probability
# Hard tasks should have lower halt probability  
contrastive_loss = margin_loss(easy_halt_logit, hard_halt_logit)
```

### Fix 3: Multi-Threshold Training
Use different cumulative thresholds during training:
```python
# Randomly vary stopping threshold during training
threshold = np.random.uniform(0.8, 1.5)  # Force variability
```

### Fix 4: Curriculum Learning
Start with obvious easy/hard distinctions, gradually increase subtlety:
```python
# Week 1: Train on clearly easy (3 cycles) vs clearly hard (15 cycles) tasks
# Week 2: Add medium complexity tasks (8 cycles)
# Week 3: Full complexity spectrum
```

### V2.7: Task Difficulty Awareness (FAILED)
- **Goal**: Train halt predictor to evaluate input complexity and adapt computation accordingly
- **Implementation**: 
  - Enhanced halt predictor with difficulty estimation system
  - Adaptive thresholds based on input complexity metrics (colors, sparsity, entropy)
  - Difficulty alignment loss to teach complexity assessment
- **Result**: **Still input-agnostic predictor** (halt logit = 0.071 for ALL inputs)
- **Discovery**: Task difficulty estimation worked (range -2.000 to 1.000) but halt predictor ignored it
- **Pattern**: Efficiency rewards fixed adversarial optimization but created constant behavior

### V2.8: Confidence-Driven Halting (PARTIAL/REVERTED)
- **Goal**: Halt based on prediction stability rather than input complexity guessing
- **Core insight**: Stop when predictions converge, not based on arbitrary complexity heuristics
- **Implementation**:
  - Prediction delta tracking across reasoning cycles using KL divergence
  - Enhanced halt predictor takes confidence signal as input
  - Stop when `prediction_delta < threshold` AND model wants to halt
- **Initial behavior**: Started with 4-cycle stability, some early variability
- **Regression**: Reverted to using all 20 cycles by step 1000+
- **Analysis**: Halt predictor learned extreme negative logits (-4.813) despite confidence input
- **Discovery**: Weak confidence sensitivity (variance 0.001961), model ignores stability signal

## Current Status: The Adaptive Computation Challenge

### ✅ Technical Achievements
- **Architecture working**: HRM successfully extended to 20 cycles
- **Prediction stability measurement**: KL divergence correctly identifies convergence
- **Task difficulty estimation**: Input complexity metrics computed accurately  
- **Efficiency rewards**: Successfully eliminated adversarial cycle escalation (V2.6)

### ⚠️ Persistent Challenge: Training Dynamics
**Core pattern across V2.4, V2.6, V2.7, V2.8**:
- All approaches start with reasonable behavior (4-5 cycles)
- Halt predictor consistently learns to ignore stopping signals
- Model discovers that maximum cycles improve task performance
- Training dynamics favor computation over efficiency despite our loss modifications

### 🔍 Key Insights from 8 Iterations

#### 1. The Adversarial Optimization Problem (V2.4)
When task_loss >> efficiency_constraints, model learns to fight stopping:
```
task_loss improvement from 8→20 cycles >> efficiency penalties
```

#### 2. Efficiency Rewards Eliminate Escalation (V2.6) 
Properly aligned rewards prevent adversarial behavior:
- Model stops fighting constraints
- Achieves stable training
- BUT may overcorrect to input-agnostic constants

#### 3. Input-Agnostic Learning Pattern (V2.6, V2.7, V2.8)
Halt predictors consistently learn "safe constants":
- V2.6: logit ≈ -0.1 (always 4 cycles)
- V2.7: logit ≈ 0.071 (always 4 cycles) 
- V2.8: logit ≈ -4.8 (always 20 cycles)

#### 4. The Confidence Paradox (V2.8)
- Prediction stability is the correct confidence measure
- Early training shows promise with stability-based stopping
- But model learns to ignore confidence signal when it conflicts with task performance

### 📊 Current Performance Summary
- **Original HRM**: 49.1% on ARC-AGI-1 (8 cycles fixed)
- **V2.6 (efficiency)**: 9.2% on ARC-AGI-1 (4 cycles always)
- **V2.7 (difficulty)**: TBD (4 cycles always, partial training)
- **V2.8 (confidence)**: TBD (4→20 cycles regression)

## Open Questions

The extended reasoning debugging journey has revealed fundamental questions about adaptive computation training:

1. **Can halt predictors learn meaningful input-dependent decisions under standard training?**
   - 8 iterations show consistent input-agnostic learning
   - Suggests deeper architectural or training methodology issues

2. **Is the efficiency vs. performance trade-off fundamentally misaligned?**
   - Model consistently discovers that more cycles improve task scores
   - Our efficiency constraints may be fighting natural optimization

3. **Do we need different training paradigms for adaptive computation?**
   - Standard gradient descent may be insufficient
   - May require curriculum learning, meta-learning, or architectural constraints

## Technical Foundation Established

Despite the halting challenges, we've built solid foundations:
- ✅ Multi-cycle HRM architecture (1→20 cycles)
- ✅ Prediction stability measurement (confidence metrics)  
- ✅ Task difficulty estimation (complexity analysis)
- ✅ Efficiency reward systems (aligned incentives)
- ✅ Comprehensive debugging methodology
- ✅ Multiple training approaches tested and documented

The technical infrastructure for adaptive computation is complete. The remaining challenge is training dynamics rather than architectural limitations.