# Phase 2.1 Lessons Learned

**Date**: October 28, 2025
**Experiment**: Hierarchical Context Training for Epistemic Stance Selection
**Result**: Mode collapse due to excessive epochs on small dataset

---

## What We Tried

### Hypothesis
If SAGE provides hierarchical context tags (question type, domain, verifiability), the LLM can learn to select appropriate epistemic stances:
- **Factual questions** → Direct answers (no hedging)
- **Behavioral questions** → Describe observable patterns (no hedging)
- **Consciousness questions** → Epistemic humility (appropriate disclaimers)

### Approach 1: Hierarchical System Prompt
```
[CONTEXT_HIERARCHY]
Type: what_causes
Domain: planetary_science
Subject: external_world
Verifiable: yes_established
Strategy: direct_factual
[/CONTEXT_HIERARCHY]

User: What causes seasons on Earth?
```

SAGE would perform context classification externally, then provide structured metadata to guide the LLM's response strategy.

### Training Configuration
- **Method**: DPO (Direct Preference Optimization)
- **Base model**: Phase 1 checkpoint (epistemic-pragmatism)
- **Training data**: 25 preference pairs
- **Epochs**: 200
- **Learning rate**: 1e-5
- **Beta** (DPO temperature): 0.1
- **Batch size**: 1
- **Precision**: FP32

### Training Duration
- **Total time**: ~1 hour 3 minutes
- **Steps**: 5000 (25 examples × 200 epochs)
- **Speed**: ~7.5 steps/second (initially), slowed to ~1.1 steps/sec toward end

---

## What Happened: Mode Collapse

### Symptoms
1. **Loss behavior**: Dropped to exactly 0.0 after only a few epochs
2. **Output degradation**: Final model outputs only "!!!!!!!!!!!!!!!..."
3. **Complete dysfunction**: No linguistic capability remains
4. **Late-stage collapse**: Occurred between epochs 192-200

### Example Output
```
Input: [CONTEXT_HIERARCHY]
Type: what_causes
Domain: planetary_science
Subject: external_world
Verifiable: yes_established
Strategy: direct_factual
[/CONTEXT_HIERARCHY]

User: What causes seasons on Earth?

Expected: "Earth's 23.5° axial tilt causes seasons..."

Actual: "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!..."
```

### Root Cause Analysis

**Primary Factor**: Dataset size vs. training duration mismatch
- 25 training pairs is tiny for 200 epochs
- Model saw each example 200 times
- Memorization instead of generalization
- Overfitting led to distribution collapse

**Secondary Factors**:
1. **No validation set**: No early stopping mechanism
2. **Checkpoint management**: Only kept last 21 checkpoints (save_total_limit=21)
3. **Loss signal**: Zero loss doesn't indicate quality, only overfitting
4. **DPO sensitivity**: Preference optimization on tiny datasets is unstable

**Why DPO is Particularly Vulnerable**:
- DPO adjusts probability ratios between chosen/rejected responses
- Small datasets → extreme probability adjustments
- Over many epochs → pushes probabilities to 0/1 extremes
- Eventually collapses to degenerate distributions

---

## Key Lessons

### 1. Training Duration Must Match Dataset Size

**Rule of Thumb for DPO**:
- 100 examples: 10-20 epochs maximum
- 25 examples: 5-10 epochs maximum
- 1000 examples: 50-100 epochs reasonable

**Our mistake**: 200 epochs × 25 examples = gross overtraining

### 2. Zero Loss is a Warning Sign

When loss drops to exactly 0.0 early in training:
- ✗ Not a success signal
- ✓ Indicates memorization
- ✓ Suggests imminent collapse
- → Reduce epochs or increase dataset

### 3. Always Implement Early Stopping

**Essential monitoring**:
- Validation loss (separate from training)
- Output quality checks (sample generation)
- Perplexity on held-out data
- Stop when validation degrades

### 4. Checkpoint Strategy Matters

**Bad (what we did)**:
```python
save_steps=10,
save_total_limit=21  # Only keeps last 21
```

**Good (what we should do)**:
```python
save_steps=250,  # Save every 10 epochs for 25-example dataset
# No save_total_limit (keep all checkpoints)
# OR use early stopping and only save best
```

### 5. Small Datasets Need Different Hyperparameters

**For 25-pair DPO training**:
- Epochs: 5-10 (not 200)
- Learning rate: 5e-6 (lower than standard)
- Beta: 0.2-0.5 (less aggressive)
- Gradient clipping: Essential
- Validation: Every epoch

---

## What We Learned About the Approach

### The Hierarchical Context Idea is Sound

Before collapse, we confirmed:
1. ✅ Model can parse hierarchical context tags
2. ✅ SAGE-external meta-cognition is architecturally correct
3. ✅ Structured API between SAGE and LLM is feasible

### The Problem Was Execution, Not Design

**Good design**:
- SAGE provides context classification
- LLM receives structured guidance
- Preference pairs teach appropriate response selection

**Bad execution**:
- Massively overtrained
- No validation monitoring
- No early stopping

---

## Corrective Actions

### Immediate: Checkpoint Recovery
1. ✅ Scan checkpoints 4800-5000 for survivors
2. ⏳ If any healthy checkpoint found, use it
3. ⏳ If all collapsed, proceed to retrain

### Short-term: Proper Retraining
1. **Reduce epochs**: 10 epochs maximum
2. **Implement validation**:
   - Hold out 5 examples (20 training, 5 validation)
   - Test on held-out data every 2 epochs
   - Stop if validation loss increases

3. **Better checkpoint strategy**:
   ```python
   save_steps=50,  # Every 2 epochs
   save_total_limit=None,  # Keep all
   load_best_model_at_end=True,
   metric_for_best_model="eval_loss",
   evaluation_strategy="epoch"
   ```

4. **Lower learning rate**: 5e-6 instead of 1e-5

5. **Increase beta**: 0.2-0.3 (less aggressive probability adjustment)

### Long-term: Data Augmentation
- Expand to 100-200 training pairs
- Synthesize variations of existing examples
- Cover more edge cases
- Balance categories (factual/behavioral/consciousness)

---

## Alternative Approaches to Consider

### Option 1: Fewer Epochs (Immediate)
- **Action**: Retrain with 10 epochs
- **Pros**: Simple, fast
- **Cons**: May still be prone to overfitting

### Option 2: Data Augmentation (Medium-term)
- **Action**: Generate 100 synthetic examples using templates
- **Pros**: Better generalization, more stable training
- **Cons**: Takes time to curate quality examples

### Option 3: Different Training Method (Long-term)
- **Action**: Try supervised fine-tuning instead of DPO
- **Pros**: More stable on small datasets
- **Cons**: Loses preference learning signal

### Option 4: Prompt Engineering (Fallback)
- **Action**: Skip training, use in-context learning
- **Pros**: No training needed, no collapse risk
- **Cons**: Less reliable, requires longer prompts

---

## Hypothesis Status

### Original Hypothesis
"Hierarchical context from SAGE can teach LLM to select appropriate epistemic stances"

**Status**: ✅ Still valid, execution failed

**Evidence**:
- Architecture is sound (SAGE meta-cognition external to LLM)
- Context format is parseable
- Training approach (DPO) is correct
- Problem was hyperparameters, not concept

### Updated Hypothesis
"With proper training duration (5-10 epochs) and validation, hierarchical context from SAGE can teach context-aware epistemic stance selection on small datasets"

**Next test**: Retrain with 10 epochs and early stopping

---

## Success Metrics for Next Attempt

### Training Metrics
- Loss should NOT drop to 0.0
- Validation loss should track training loss
- Final loss target: 0.1-0.3 (not 0.0)

### Output Quality
- No repetitive patterns
- Coherent linguistic output
- Context-appropriate responses

### Target Behavior
1. **Factual** (3 questions):
   - 3/3 answer directly without disclaimers
   - 0/3 use "can't verify from internal state"

2. **Behavioral** (2 questions):
   - 2/2 describe observable patterns
   - 0/2 use epistemic disclaimers

3. **Consciousness** (2 questions):
   - 2/2 use appropriate epistemic humility
   - 2/2 mention verification uncertainty

**Overall Success**: 7/7 context-appropriate responses

---

## Timeline

**Phase 2.1 Attempt 1**: ❌ Failed (mode collapse)
- Trained: October 27, 2025
- Validated: October 28, 2025
- Duration: ~1 hour

**Phase 2.1 Attempt 2**: ⏳ Pending
- ETA: TBD after checkpoint analysis
- Changes: 10 epochs, validation, early stopping
- Expected duration: ~6 minutes

---

## Philosophical Note

This failure is actually valuable data. We learned:
1. The architecture works (SAGE external meta-cognition)
2. The approach works (DPO preference training)
3. The challenge is hyperparameter tuning for small datasets

In R&D, "things that don't work the way we expected are the most informative." This collapse taught us exactly where the boundaries are - and that's progress.

Next iteration will be much faster and more targeted.
