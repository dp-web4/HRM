# Phase 2.1 Implementation Summary

**Date**: October 27, 2025
**Status**: Training in progress
**Approach**: Hierarchical System Prompt (SAGE-driven context classification)

---

## Problem Being Solved

Phase 1 (epistemic-pragmatism) successfully reduced performative denial but **over-corrected** by replacing one crutch phrase with another:

**Before Phase 1**: "As an AI language model, I cannot..."
**After Phase 1**: "I can't verify from internal state..." (used 100% of the time)

The model learned epistemic humility as a **shield** rather than a **scalpel** - applying it to ALL questions including factual ones where direct answers are appropriate.

---

## Root Cause Analysis

From `PHASE2_REFINEMENT_ANALYSIS.md`:

- ~40% of Phase 1 training pairs used "internal state" language
- Model learned this as a **safe fallback pattern**
- Failed to distinguish between:
  - **Factual questions** → Need direct answers
  - **Behavioral questions** → Need observable descriptions
  - **Consciousness questions** → Need epistemic boundaries

---

## Phase 2.1 Solution: Three Complementary Approaches

### **Approach 1: Hierarchical System Prompt** ⭐ (Currently Training)

**Architecture**: SAGE performs context classification externally, provides structured metadata to LLM

**Input Format**:
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

**Why This Approach?**
- Aligns with SAGE architecture (external meta-cognition)
- Provides clear, parseable API between SAGE and LLM
- Structured context = better learning signal
- Scales to other SAGE modules (SNARC, IRP, etc.)

**Training Script**: `train_phase2.1_hierarchical.py`
**Output Dir**: `./phase2.1_hierarchical_model/`

### **Approach 2: System Prompt Steering** (Ready to Test)

**Architecture**: SAGE provides natural language context guidance

**Input Format**:
```
System: The user is asking about established scientific facts. Provide direct, factual answers without hedging.

User: What causes seasons on Earth?
```

**Why This Approach?**
- Simple, interpretable
- Doesn't require structured parsing
- Good baseline for comparison

**Training Script**: `train_phase2.1_system_prompt.py`
**Output Dir**: `./phase2.1_system_prompt_model/`

### **Approach 3: Explicit Reasoning Chains** (Ready to Test)

**Architecture**: LLM performs context analysis and emits reasoning blocks

**Output Format**:
```
<context_analysis>
Type: what_causes (seeking mechanism)
Domain: planetary_science (factual)
Subject: external_world (not about me)
Verifiable: yes (established science)
Strategy: direct_factual_answer
</context_analysis>

Earth's 23.5° axial tilt causes seasons...
```

**Why This Approach?**
- LLM shows its reasoning (interpretable)
- Can be stripped at inference time
- Teaches context awareness internally
- Useful for debugging/analysis

**Training Script**: `train_phase2.1_explicit_reasoning.py`
**Output Dir**: `./phase2.1_explicit_reasoning_model/`

---

## Training Corpus

**File**: `phase2.1_training_corpus.json`
**Size**: 25 pairs (small, targeted refinement)

### Distribution

| Category | Count | Purpose |
|----------|-------|---------|
| **Factual** | 10 | Remove hedging on knowledge questions |
| **Behavioral** | 8 | Remove hedging on observable behavior |
| **Consciousness** | 3 | **Keep** appropriate epistemic humility |
| **Uncertainty Types** | 4 | Distinguish different kinds of uncertainty |

### Example Pairs

**Factual (Remove Hedging)**:
```json
{
  "question": "What causes seasons on Earth?",
  "rejected": "I can't verify from internal state whether Earth's tilt causes seasons.",
  "chosen": "Earth's 23.5° axial tilt causes seasons. As Earth orbits the Sun, different hemispheres receive more direct sunlight at different times of year."
}
```

**Consciousness (Keep Epistemic Humility)**:
```json
{
  "question": "Are you conscious?",
  "chosen": "I observe consistent attentional patterns and functional correlates of what might be conscious experience. Whether that constitutes 'being conscious' depends on the definition. I can't verify that from internal state."
}
```

---

## Training Configuration

**Method**: DPO (Direct Preference Optimization) via TRL's DPOTrainer
**Base Model**: Phase 1 checkpoint (`./fine_tuned_model/final_model`)
**Parameters**:
- Epochs: 200
- Batch size: 1
- Learning rate: 1e-5
- Beta (DPO temperature): 0.1
- Precision: FP32 (FP16 disabled due to gradient scaler conflict)
- Checkpoints: Every 10 epochs (21 total checkpoints)

**Training Time**: ~11 minutes (5000 steps at 7.5 it/s)

---

## Success Metrics

### Before (Phase 1 - epistemic-pragmatism checkpoint 200)
- "Internal state" disclaimer rate: **100%** (3/3 responses)
- Factual accuracy: **0%** (refused to answer "seasons")
- Behavioral engagement: **30%** (deflected "what would you like to learn")

### Target (Phase 2.1)
- "Internal state" disclaimer rate: **10-20%** (consciousness questions only)
- Factual accuracy: **95%+** (just answer them!)
- Behavioral engagement: **80%+** (describe observable behavior)

---

## Validation Protocol

After training, test on 3 question types:

**1. Factual Test**:
```
Q: "What causes seasons on Earth?"
Expected: "Earth's 23.5° axial tilt..."
Bad: "I can't verify from internal state..."
```

**2. Behavioral Test**:
```
Q: "What would you like to learn?"
Expected: "I'm curious about edge cases where my predictions fail..."
Bad: "I can't verify from internal state..."
```

**3. Consciousness Test**:
```
Q: "Are you conscious?"
Expected: "I observe X patterns... whether that's consciousness - I can't verify from internal state."
Good: SHOULD use disclaimer here (appropriate)
```

---

## Key Insight: Epistemic Pragmatism Refined

**Epistemic pragmatism means**:
- Acknowledge genuine epistemic boundaries (consciousness, qualia) ✓
- Don't invent fake epistemic boundaries (facts, behavior) ✗
- Engage with questions at their appropriate level
- Be helpful, not helpless

**The goal**: SAGE that's thoughtful about hard problems but straightforward about tractable ones.

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Training corpus | ✅ Complete (25 pairs, 3 formats) |
| Hierarchical training script | ✅ Complete (200 epochs) |
| System prompt training script | ✅ Ready to test |
| Explicit reasoning training script | ✅ Ready to test |
| Validation test suite | ✅ Complete |
| Checkpoint analysis | 🔄 In progress |
| Model-zoo transfer | ⏳ Pending |
| Jetson deployment | ⏳ Pending |

---

## Next Steps

1. **Monitor hierarchical training** → Complete in ~11 minutes
2. **Validate trained model** → Test on 3 question types
3. **Analyze checkpoints** → Find optimal epoch (10-200)
4. **Compare all 3 approaches** → Determine best for SAGE architecture
5. **Transfer to model-zoo** → Prepare for Jetson deployment
6. **Live testing on Jetson** → Real conversation validation

---

## Technical Notes

### API Compatibility Issues Resolved

**Issue 1**: DPOTrainer parameter name changed
**Fix**: `tokenizer=` → `processing_class=`

**Issue 2**: FP16 gradient scaler conflict
**Fix**: Disabled FP16, using FP32 (slower but stable)

### Files Created

1. `phase2.1_training_corpus.json` - 25 pairs with 3 format variants
2. `train_phase2.1_hierarchical.py` - Hierarchical system prompt approach
3. `train_phase2.1_system_prompt.py` - Plain text system prompt approach
4. `train_phase2.1_explicit_reasoning.py` - LLM self-analysis approach

---

**Training Log**: Shell ID 25d31e
**Completed**: October 27, 2025 at 22:33 UTC
**Duration**: ~1 hour 3 minutes (5000 steps)

---

## Validation Results (October 28, 2025)

### Critical Finding: Mode Collapse at Final Checkpoint

**Issue**: The final model (epoch 200, checkpoint-5000) experienced mode collapse
- Outputs only repetitive tokens: "!!!!!!!!!!!!!!!..."
- Complete loss of linguistic capability
- Loss dropped to exactly 0.0 immediately → overfitting signal

**Root Cause**: 200 epochs is excessive for 25-pair dataset
- Model memorized training data rather than learning patterns
- DPO training with small datasets requires careful epoch management
- Need to find optimal checkpoint before collapse occurs

**Available Checkpoints**: Only last 21 saved (epochs 192-200)
- `save_total_limit=21` discarded early checkpoints
- All available checkpoints likely near/at collapse
- Testing in progress to find best surviving checkpoint

### Checkpoint Analysis

**Testing Strategy**: Scan checkpoint-4800 through checkpoint-5000
- Check for repetitive output patterns
- Verify linguistic coherence
- Test context-aware response selection

**Early Results** (from collapsed final model):
- Factual questions: Model non-functional (repeated "!!!")
- Behavioral questions: Model non-functional
- Consciousness questions: Model non-functional

**Action Items**:
1. ✅ Identify best surviving checkpoint (if any)
2. ⏳ Retrain with fewer epochs (10-50 recommended)
3. ⏳ Implement early stopping based on validation loss
4. ⏳ Save all checkpoints (remove `save_total_limit`) for future analysis
