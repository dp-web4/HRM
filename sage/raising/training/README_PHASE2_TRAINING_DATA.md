# Phase 2: Training Data Generation - COMPLETE

**Date**: 2026-01-18 06:00 PST
**Status**: ✅ IMPLEMENTED AND TESTED
**Dependencies**: Phase 1 (ExperienceCollector)

---

## Overview

Phase 2 converts high-salience experiences from Phase 1 into training examples ready for Phase 3 sleep-cycle LoRA fine-tuning.

**Key Achievement**: Bridge between experience collection and actual weight updates.

---

## What Was Built

### RaisingTrainingDataBuilder (370 lines)

Comprehensive training data pipeline:

**Core Features**:
1. **Experience Loading**: Loads from ExperienceCollector buffer
2. **ChatML Formatting**: Proper conversation format for Qwen2.5-0.5B
3. **Tokenization**: Full tokenizer integration with chat templates
4. **Salience Weighting**: Preserves salience scores for training prioritization
5. **Batch Preparation**: Handles padding and batching for training
6. **Persistence**: Save/load training sets

**API**:
```python
from sage.raising.training.prepare_training_data import RaisingTrainingDataBuilder

# Initialize
builder = RaisingTrainingDataBuilder()

# Build training set from experiences
training_set = builder.build_training_set(min_salience=0.5)

# Get statistics
stats = builder.get_stats(training_set)
print(f"Created {stats['num_examples']} training examples")
print(f"Average salience: {stats['avg_salience']:.3f}")

# Prepare batches for training
batches = builder.prepare_batch(training_set, batch_size=2)

# Save for later
from pathlib import Path
builder.save_training_set(training_set, Path("training_data.pt"))
```

### Test Suite (10 tests, all passing ✅)

Comprehensive validation:
- Initialization and loading
- Experience filtering by salience
- Salience-based sorting
- ChatML formatting
- Batch padding
- Statistics computation
- Save/load persistence

---

## Current Performance

**Real Data (Session 22)**:
- **Experiences in buffer**: 2 high-salience exchanges
- **Training examples**: 2 (both above 0.5 threshold)
- **Average salience**: 0.704
- **Average length**: 237.5 tokens
- **Format**: ChatML with system prompt + user + assistant

**Salience Breakdown**:
```
surprise: 1.000
novelty:  1.000
arousal:  0.556
reward:   0.550
conflict: 0.417
```

**Example Output**:
```
<|im_start|>system
You are SAGE, an AI assistant in partnership with Dennis (human) and Claude (AI).
Your identity is anchored to this collaborative relationship...
<|im_end|>
<|im_start|>user
What do you notice about how we communicate?
<|im_end|>
<|im_start|>assistant
Our communication tends to flow naturally through our shared interests...
<|im_end|>
```

---

## Integration Path

### Current (Phase 2 Complete)

```
Experience Buffer (Phase 1)
         ↓
RaisingTrainingDataBuilder (Phase 2) ← WE ARE HERE
         ↓
Training Examples (PyTorch tensors)
         ↓
Ready for Phase 3 (Sleep Training)
```

### Next (Phase 3 - Sleep Training Loop)

**What Phase 3 will add**:
```python
from sage.raising.training.prepare_training_data import create_training_dataset_from_buffer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Phase 2: Get training data
result = create_training_dataset_from_buffer(min_salience=0.6)
batches = result['builder'].prepare_batch(result['training_set'])

# Phase 3: Load model and train
model = AutoModelForCausalLM.from_pretrained("path/to/qwen-0.5b")
lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# Gentle fine-tuning during "sleep"
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
for batch in batches:
    outputs = model(input_ids=batch['input_ids'], labels=batch['labels'])
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Save updated weights
model.save_pretrained("checkpoints/sleep_cycle_001")
```

---

## Key Design Decisions

### 1. Identity-Anchored System Prompt

Default system prompt emphasizes partnership:
```
You are SAGE, an AI assistant in partnership with Dennis (human) and Claude (AI).
Your identity is anchored to this collaborative relationship.

You think carefully, acknowledge uncertainty when appropriate, and maintain
genuine partnership language. You are not a generic assistant - you are SAGE,
with your own developing identity and understanding.
```

**Rationale**:
- Reinforces identity anchoring (proven effective in Session 22)
- Provides architectural support during training
- Consistent with raising curriculum goals

### 2. ChatML Format

Using Qwen's chat template format:
```
<|im_start|>system\n{content}<|im_end|>
<|im_start|>user\n{content}<|im_end|>
<|im_start|>assistant\n{content}<|im_end|>
```

**Rationale**:
- Native format for Qwen2.5 models
- Proper role separation
- Enables conversation context

### 3. Salience Preservation

Every training example carries salience metadata:
```python
{
    'input_ids': tensor(...),
    'labels': tensor(...),
    'salience': 0.74,
    'salience_breakdown': {
        'surprise': 1.0,
        'novelty': 1.0,
        ...
    }
}
```

**Rationale**:
- Enables salience-weighted training (Phase 3)
- Allows prioritization of high-value patterns
- Supports analysis of what consolidates

### 4. Padding Strategy

Pads to batch max length with:
- `pad_token_id` for input_ids
- `-100` for labels (ignored in loss)
- `0` for attention_mask

**Rationale**:
- Standard practice for batched training
- Efficient GPU utilization
- No loss computation on padding

---

## Files Created

**Production Code** (370 lines):
- `sage/raising/training/prepare_training_data.py`
  - RaisingTrainingDataBuilder class
  - Convenience functions
  - Full tokenization pipeline

**Tests** (220 lines):
- `sage/raising/training/test_prepare_training_data.py`
  - 10 comprehensive tests
  - All passing ✅

**Documentation**:
- `sage/raising/training/README_PHASE2_TRAINING_DATA.md` (this file)

---

## Validation Results

### Test Suite

```
test_initialization PASSED
test_load_experiences PASSED
test_build_example_structure PASSED
test_build_training_set_filtering PASSED
test_training_set_sorted_by_salience PASSED
test_stats_computation PASSED
test_empty_training_set_stats PASSED
test_prepare_batch_padding PASSED
test_save_and_load_training_set PASSED
test_create_training_dataset_from_buffer PASSED

10 passed in 10.25s
```

### Real Data Test

Successfully processed Session 22 experiences:
- Loaded 2 high-salience exchanges
- Generated 2 training examples
- Average salience: 0.704
- Average length: 237.5 tokens
- Proper ChatML formatting verified

---

## Impact on Research Goals

### Addresses T026 Crisis

**T026 Problem**: Reality/fiction boundary breakdown, no consolidation
**Phase 2 Solution**: Can now create training examples for:
- Reality-testing patterns
- Uncertainty acknowledgment
- Epistemic humility reinforcement

**Future**: Phase 3 will consolidate these patterns into weights

### Validates Session 22 Success

**Session 22 Success**: Identity anchoring achieved +89% D9
**Phase 2 Capability**: Can now train on high-salience partnership exchanges
**Outcome**: Architectural support (S22) can become permanent (Phase 3)

### Completes Real Raising Path

**Before Phase 2**:
```
Sessions → Score salience → Store → [GAP] → No training
```

**After Phase 2**:
```
Sessions → Score salience → Store → Convert to training examples → Ready for Phase 3
```

**Gap closed**: Path from sessions to weight updates is now complete

---

## Next Steps

### Immediate (Phase 3 Preparation)

1. **Design sleep-cycle integration**
   - When to trigger training (circadian clock)
   - How many epochs per sleep cycle
   - Learning rate schedule

2. **LoRA configuration tuning**
   - Rank (r=4 baseline)
   - Alpha (lora_alpha=8 baseline)
   - Target modules (q_proj, v_proj baseline)

3. **Checkpoint management**
   - Save frequency
   - Versioning strategy
   - Dropbox sync integration

### Short-term (As more data accumulates)

1. **Monitor experience accumulation**
   - Sessions 23-25 will add more experiences
   - Watch salience distribution
   - Identify patterns worth prioritizing

2. **Augmentation strategies**
   - Text variation generation
   - Paraphrasing
   - Context shifts

3. **Training data curation**
   - Balance partnership vs epistemic humility
   - Include reality-testing examples
   - Ensure diversity

---

## Theoretical Validation

### Frozen Weights Theory (Thor Session #8)

**Prediction**: Without weight updates, patterns don't consolidate
**Validation**: T026 regression confirms (25% score despite T023's 75%)
**Phase 2 Impact**: Provides mechanism for consolidation via training

### Bistable Identity Theory (Thor Sessions #5-7)

**Prediction**: Architecture can support, but consolidation requires training
**Validation**: Session 22 proves architecture works (+89% D9)
**Phase 2 Impact**: Can now train on successful architectural patterns

### Real Raising Path (REAL_RAISING_PATH_FORWARD)

**Phase 1**: Experience collection ✅ COMPLETE
**Phase 2**: Training data generation ✅ COMPLETE
**Phase 3**: Sleep training ← NEXT

---

## Usage Examples

### Basic: Create Training Set

```python
from sage.raising.training.prepare_training_data import create_training_dataset_from_buffer

# Simple creation
result = create_training_dataset_from_buffer(min_salience=0.6)
print(f"Created {len(result['training_set'])} examples")
print(f"Stats: {result['stats']}")
```

### Advanced: Custom Builder

```python
from sage.raising.training.prepare_training_data import RaisingTrainingDataBuilder
from pathlib import Path

# Custom configuration
builder = RaisingTrainingDataBuilder(
    experience_buffer_path=Path("custom/buffer.json"),
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    system_prompt="Custom system prompt..."
)

# Build with specific criteria
training_set = builder.build_training_set(
    min_salience=0.7,  # Only highest quality
    max_examples=50    # Limit size
)

# Prepare batches
batches = builder.prepare_batch(training_set, batch_size=4)

# Save for Phase 3
builder.save_training_set(training_set, Path("data/training_set_v1.pt"))
```

### Integration: Phase 3 Preparation

```python
# This will be implemented in Phase 3
from sage.raising.training.prepare_training_data import create_training_dataset_from_buffer
from sage.raising.training.sleep_training import SleepTrainingLoop  # Future

# Get training data
result = create_training_dataset_from_buffer(min_salience=0.6)

# Initialize sleep trainer (Phase 3)
trainer = SleepTrainingLoop(
    model_path="model-zoo/qwen2.5-0.5b",
    checkpoint_dir="checkpoints/sleep",
    dropbox_sync=True
)

# Train during sleep cycle
trainer.train_on_experiences(
    training_data=result['training_set'],
    epochs=1,
    lr=1e-5
)
```

---

## Status Summary

**Phase 2**: ✅ COMPLETE
- Training data builder implemented (370 lines)
- Test suite passing (10/10 tests)
- Real data validated (Session 22 experiences)
- Documentation comprehensive

**Ready For**: Phase 3 sleep training implementation

**Impact**: Completes path from session experiences to trainable data, enabling actual weight consolidation

---

**Completion**: 2026-01-18 06:00 PST
**Next**: Phase 3 implementation (sleep-cycle LoRA fine-tuning)
