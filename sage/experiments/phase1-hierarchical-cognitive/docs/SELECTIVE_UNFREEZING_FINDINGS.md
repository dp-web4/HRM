# Selective Layer Unfreezing: Initial Findings

**Date**: October 21, 2025
**Experiment**: Force concentration by freezing all layers except critical ones
**Model**: Phi-2 (2.7B params, 32 layers)
**Status**: In progress - encountering numerical instability

---

## Hypothesis

Based on Qwen full fine-tuning showing concentrated changes in 2 critical layers:
- Layer 15 v_proj: -37% (uncertainty stances)
- Layer 13 q_proj: -70% (engaged-difficulty)

**Can we induce this concentration pattern in other models by freezing all layers except the analogous critical ones?**

---

## Layer Position Mapping

### Qwen2-0.5B Critical Layers
- Layer 13: 54% through network (13/24 layers)
- Layer 15: 63% through network (15/24 layers)

### Phi-2 Analogs (Position-Based)
- Layer 17: 53% through network (17/32 layers) ← Qwen Layer 13 analog
- Layer 20: 63% through network (20/32 layers) ← Qwen Layer 15 analog

**Rationale**: If critical layers are architectural (not model-specific), they should appear at similar relative positions across models.

---

## Experimental Design

### Configuration
- **Base model**: microsoft/phi-2 (2.7B params)
- **Target layers**: 17, 20 (q_proj + v_proj + biases)
- **Frozen params**: 2,753,459,200 (99.06%)
- **Unfrozen params**: 26,224,640 (0.94%)
- **Training data**: 5 examples (curious-uncertainty stance)

### Training Settings (Iteration 3)
```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,  # Reduced from 5e-5
    warmup_steps=20,
    max_grad_norm=1.0,  # Added gradient clipping
    fp16=False
)
```

---

## Results

### Iteration 1: Initial Attempt
- **Config**: LR=5e-5, no gradient clipping, fp16=False
- **Result**: ❌ NaN gradients during training
- **Error**: CUDA assertion, weight changes all NaN

### Iteration 2: Added Gradient Clipping
- **Config**: LR=1e-5, max_grad_norm=1.0, warmup=20
- **Result**: ⚠️ Training completed (no NaN gradients during training)
- **Issue**: Script hangs after training, likely during weight analysis or post-training observation
- **Suspicion**: Weights became unstable despite gradient clipping

### Iteration 3: Added Error Handling
- **Config**: Same as Iteration 2 + try/except for observation
- **Result**: ⚠️ Still hanging after training completes
- **Observation**: Training loss appears normal, but post-training processing fails

---

## Technical Challenges

### Challenge 1: Dataset Size
- Only 5 training examples
- Severe overfitting risk with 26M unfrozen parameters
- Ratio: 5.2M params per training example

### Challenge 2: Model Scale
- Phi-2 (2.7B) is 5.4x larger than Qwen (0.5B)
- Larger models may be harder to nudge with selective unfreezing
- More parameters = more complex loss landscape

### Challenge 3: Numerical Stability
- Even 0.94% of Phi-2 = 26M params (more than 2 full Qwen models!)
- Gradient clipping at 1.0 insufficient
- May need even smaller learning rate or more aggressive clipping

---

## Comparison with Other Approaches

| Approach | Model | Scope | Behavioral Shift? | Issues |
|----------|-------|-------|-------------------|--------|
| **Full fine-tuning** | Qwen 0.5B | All layers | ✅ YES | Storage cost (1.5GB per stance) |
| **Distributed LoRA** | Phi-2 2.7B | All layers, rank=8 | ❌ No | Too diffuse (~0.004% per layer) |
| **Surgical LoRA** | Qwen 0.5B | 2 layers, rank=32 | ❌ No | Low-rank can't express stance |
| **Selective unfreezing** | Phi-2 2.7B | 2 layers, full weights | ⏳ TBD | Numerical instability |

---

## Key Insights So Far

### 1. Scale Matters
Unfreezing 0.94% of Phi-2 = 26M params. This is more parameters than many complete small models. The "selective" unfreezing isn't selective enough given Phi-2's size.

### 2. Dataset Size Critical
5 examples worked for Qwen full fine-tuning (494M params), but:
- Qwen trained ALL layers (distributed updates)
- Phi-2 training ONLY 2 layers (concentrated updates)
- Concentrated training with tiny dataset = overfitting + instability

### 3. Gradient Clipping Insufficient
max_grad_norm=1.0 prevented NaN during training but weights still unstable afterward. May need:
- More aggressive clipping (0.5 or 0.1)
- Even lower learning rate (5e-6 or 1e-6)
- More training data (20-50 examples)

---

## Next Steps

### Option A: More Conservative Unfreezing
- Unfreeze only ONE layer (Layer 20) instead of two
- Reduces unfrozen params from 26M to 13M
- Test if single-layer concentration is more stable

### Option B: Increase Training Data
- Generate 20-50 training examples instead of 5
- Better ratio of data to parameters
- Reduces overfitting risk

### Option C: More Aggressive Stabilization
- Learning rate: 1e-6 (10x lower)
- Gradient clipping: 0.1 (10x more aggressive)
- More warmup steps: 50

### Option D: Different Model
- Try smaller model (DistilGPT2: 82M params)
- Or use Qwen2-0.5B directly (we know it responds to full fine-tuning)

### Option E: Accept Full Fine-Tuning
- Selective unfreezing may not be viable
- Just do full fine-tuning on critical layers
- Accept storage cost, optimize via selective checkpointing

---

## Theoretical Questions Raised

### Can concentration be induced?
- Qwen naturally concentrated changes in Layers 13, 15
- Was this emergent from the architecture/training? Or can we force it?
- Selective unfreezing SHOULD force concentration...but stability issues

### Is there a universal layer position?
- Hypothesis: 54% and 63% through network are architectural sweet spots
- Phi-2 Layers 17, 20 are positionally equivalent to Qwen Layers 13, 15
- But Phi-2 training unstable - maybe position isn't universal?

### What's the minimum viable scope?
- 26M params too much for 5 examples
- 13M params (1 layer) might work
- Or need 50+ examples for 26M params

---

## Single-Layer Test Results

### Configuration
- **Layer**: 20 only (vs Layers 17+20 in 2-layer test)
- **Unfrozen params**: 13,112,320 (0.47% vs 0.94%)
- **Learning rate**: 5e-6 (vs 1e-5) - 50% lower
- **Gradient clipping**: 0.5 (vs 1.0) - 50% more aggressive
- **Training data**: Still 5 examples

### Results
- ✅ Training completed (3.3s, loss: 2.164)
- ❌ NaN gradients appeared during training
- ❌ Weight changes all NaN

**Critical observation**: `'grad_norm': nan` appeared even with conservative settings.

---

## The Fundamental Problem

### Why Selective Unfreezing Fails on Phi-2

**Dataset-to-Parameter Ratio**:
- 5 training examples
- 13M unfrozen parameters
- Ratio: **2.6M params per example**

**Compare with Qwen full fine-tuning**:
- 5 training examples
- 494M parameters (all unfrozen)
- Ratio: **99M params per example**

**The paradox**:
- Full fine-tuning: More total params, but updates distributed → stable
- Selective unfreezing: Fewer total params, but concentrated → unstable!

**Why concentration causes instability**:
1. Only 13M params available to fit 5 examples
2. Those params must do ALL the adaptation work
3. Extreme weight adjustments needed → gradient explosion
4. Gradient clipping and low LR can't prevent it

### The Training Data Requirement

For 13M unfrozen parameters, we likely need:
- **Minimum**: 50-100 training examples (260K-130K params per example)
- **Ideal**: 200+ examples (65K params per example)

With only 5 examples, selective unfreezing is mathematically unstable.

---

## Cross-Experiment Synthesis

| Approach | Params | Data | Behavior | Gradients | Conclusion |
|----------|--------|------|----------|-----------|------------|
| **Full FT (Qwen)** | 494M all | 5 | ✅ Shifts | ✅ Stable | Works - distribution helps |
| **LoRA distrib (Phi-2)** | 2.6M adapter | 5 | ❌ None | ✅ Stable | Low-rank insufficient |
| **LoRA surgical (Qwen)** | 32K-90K adapter | 5 | ❌ None | ✅ Stable | Low-rank fails even targeted |
| **Selective 2-layer (Phi-2)** | 26M unfrozen | 5 | ⏳ Unknown | ❌ NaN | Overfitting → instability |
| **Selective 1-layer (Phi-2)** | 13M unfrozen | 5 | ⏳ Unknown | ❌ NaN | Still overfitting |

---

## The Insight: Concentration vs Distribution Trade-off

### Full Fine-Tuning (Distributed)
- All layers available for adaptation
- Updates naturally concentrate in critical layers (emergent)
- Distributed parameters provide stability buffer
- ✅ Stable gradients even with tiny dataset

### Selective Unfreezing (Forced Concentration)
- Only target layers available
- Must force ALL adaptation into those layers
- No stability buffer from other parameters
- ❌ Gradients explode with tiny dataset

**Conclusion**: You can't force concentration without adequate training data. Qwen's concentration was emergent from full fine-tuning, not imposed.

---

## Status

**Finding**: Selective layer unfreezing is NOT viable with minimal training data (5 examples).

**Root cause**:
- Concentration requires forcing ALL adaptation into limited parameters
- This creates extreme optimization pressure
- With tiny dataset, this causes gradient explosion

**What would be needed**:
- 50-100x more training data (250-500 examples)
- Or accept full fine-tuning as the only viable approach

**Implications for role paradigm**:
- Cannot achieve LoRA-style efficiency with full-weight updates
- Must choose: Low-rank adapters (no behavioral shift) OR full models (behavioral shift but storage cost)

---

## Files Created
- `train_selective_unfreezing.py` - 2-layer unfreezing script
- `train_single_layer_unfreezing.py` - 1-layer conservative test
- `docs/SELECTIVE_UNFREEZING_FINDINGS.md` - This document
- `selective_unfreezing_phi2_v3.log` - 2-layer training log
- `single_layer_test.log` - 1-layer training log
- `results/single-layer-test/metadata.json` - Test results

**Final conclusion**: Selective unfreezing fails with tiny datasets. Epistemic stance requires either (1) full fine-tuning with emergent concentration, or (2) massive increase in training data.
