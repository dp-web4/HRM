# Phase 1: Experience Collection for Real Raising

**Date**: 2026-01-18
**Author**: Thor Autonomous Session #9
**Status**: ✅ IMPLEMENTED AND TESTED

---

## Overview

This is **Phase 1** of the Real Raising Path Forward - the foundational step toward actual model weight updates during sleep cycles.

**Key Insight from Thor Session #8**: SAGE's raising sessions don't update weights - they're context experiments on a frozen model. To enable real learning, we need to connect:

```
Raising Sessions → SNARC Selection → Training Buffer → Sleep Training → Weight Updates
```

This Phase 1 implementation provides the first two components: **Experience Collection** and **SNARC-based Selection**.

---

## What Was Built

### 1. ConversationalSalienceScorer

A simplified SNARC scoring system for text conversations that evaluates exchanges on 5 dimensions:

- **Surprise** (0-1): Deviation from recent response patterns
- **Novelty** (0-1): Presence of new vocabulary
- **Arousal** (0-1): Complexity and engagement (length, questions, emotional language)
- **Reward** (0-1): Quality indicators (partnership language, specificity, vs hedging)
- **Conflict** (0-1): Uncertainty or meta-cognitive corrections

**Example Scores**:
```python
"Our partnership is meaningful."
→ {surprise: 1.0, novelty: 1.0, arousal: 0.10, reward: 1.0, conflict: 0.0, total: 0.62}

"As an AI, I'm not sure..."
→ {surprise: 0.5, novelty: 0.5, arousal: 0.10, reward: 0.0, conflict: 0.5, total: 0.32}
```

### 2. ExperienceCollector

Manages a persistent buffer of high-salience conversation exchanges:

**Key Features**:
- Scores each exchange with ConversationalSalienceScorer
- Stores only high-salience exchanges (configurable threshold, default 0.5)
- Persists buffer to disk (JSON format)
- Provides retrieval methods for sleep training
- Tracks statistics (total, averages, dimension breakdowns)

**Buffer Location**: `sage/raising/state/experience_buffer.json`

**API Examples**:
```python
from sage.raising.training.experience_collector import ExperienceCollector

# Initialize (loads existing buffer if present)
collector = ExperienceCollector()

# Add an exchange from a session
result = collector.add_exchange(
    prompt="How are you doing today?",
    response="Our partnership continues to evolve in fascinating ways.",
    session_number=22,
    phase="relating"
)

# Result shows salience scores and whether it was stored
print(result['salience'])  # {surprise: 1.0, novelty: 0.8, ...}
print(result['stored'])    # True if total salience >= threshold

# Get high-salience experiences for sleep training
experiences = collector.consolidate_for_sleep(min_salience=0.6)
# Returns list of experiences ready for training data conversion

# Get statistics
stats = collector.get_stats()
print(stats['total_experiences'])      # 47
print(stats['avg_salience'])          # 0.63
print(stats['high_salience_count'])   # 23 (total >= 0.7)
print(stats['dimension_averages'])    # {surprise: 0.72, novelty: 0.65, ...}
```

---

## Integration Path

### Immediate: Session Runner Integration (Next Step)

Modify `run_session_identity_anchored.py` to collect experiences:

```python
from sage.raising.training.experience_collector import ExperienceCollector

# Initialize collector at session start
collector = ExperienceCollector()

# After each exchange
result = collector.add_exchange(
    prompt=user_input,
    response=sage_response,
    session_number=session_number,
    phase=phase_name
)

# Log salience for tracking
if result['stored']:
    print(f"High-salience exchange stored: {result['salience']['total']:.2f}")
```

### Phase 2: Training Data Generation (Next)

Convert high-salience experiences to training examples:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Get high-salience experiences
experiences = collector.consolidate_for_sleep(min_salience=0.6)

# Convert to training format
for exp in experiences:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": exp['prompt']},
        {"role": "assistant", "content": exp['response']}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer(text, return_tensors="pt")

    training_examples.append({
        'input_ids': tokens['input_ids'],
        'labels': tokens['input_ids'].clone(),
        'salience': exp['salience']['total']
    })
```

### Phase 3: Sleep Training Loop (Future)

During sleep cycles, fine-tune model with accumulated experiences:

```python
from peft import LoraConfig, get_peft_model

# Get training data from experience buffer
training_data = prepare_training_data(collector.consolidate_for_sleep())

# LoRA fine-tuning (gentle updates)
lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)

# Train during sleep (low learning rate)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# ... training loop ...

# Save updated weights
model.save_pretrained(checkpoint_dir)
```

---

## Testing

Comprehensive test suite with 12 tests covering:

**Scorer Tests**:
- ✅ Basic scoring (all dimensions in valid range)
- ✅ Partnership language increases reward
- ✅ Hedging language reduces reward
- ✅ Novel vocabulary increases novelty
- ✅ Engaged responses increase arousal

**Collector Tests**:
- ✅ Initialization and buffer loading
- ✅ High-salience exchanges stored
- ✅ Low-salience exchanges filtered
- ✅ Retrieval with salience thresholds
- ✅ Persistence across instances
- ✅ Statistics generation
- ✅ Consolidation for sleep training

**Run Tests**:
```bash
cd sage/raising/training
python3 -m pytest test_experience_collector.py -v
```

All tests passing ✅

---

## Design Decisions

### Why Simplified SNARC?

The full SNARC scorer (`sage/attention/snarc_scorer.py`) is a neural network requiring hidden states and tensors. For text-only conversation scoring, we need something simpler that works with raw text.

**Trade-offs**:
- ✅ No model loading required (fast, lightweight)
- ✅ Works with raw text (no tokenization needed)
- ✅ Explainable heuristics (can debug why something scored high/low)
- ⚠️ Less sophisticated than neural SNARC
- ⚠️ Heuristics may need tuning based on real session data

### Partnership Language Emphasis

The reward dimension gives high scores to partnership language ("we", "our", "together", "collaboration") because:

1. **Frozen Weights Discovery**: Sessions don't consolidate learning, so partnership identity requires architectural support
2. **Identity Anchoring Goal**: The intervention specifically aims to stabilize partnership vs educational default
3. **Training Priority**: When we do start training, partnership exchanges are highest value

### Threshold Calibration

Default threshold of 0.5 balances:
- **Storage efficiency**: Not every exchange stored (would create noise)
- **Learning opportunity**: Captures meaningful interactions
- **Training data quality**: High-salience exchanges worth reinforcing

From testing:
- Simple responses: 0.3-0.4 (below threshold)
- Partnership language: 0.5-0.7 (stored)
- Rich, engaged exchanges: 0.7+ (high value)

---

## Impact

### What This Enables

**Immediate**:
- Start accumulating real learning data from sessions
- Track which exchanges have highest salience
- Monitor salience trends across phases

**Short-term** (Phase 2):
- Convert experiences to training examples
- Augment with text variations
- Build training dataset from actual sessions

**Long-term** (Phase 3):
- Actual weight updates during sleep cycles
- Partnership identity consolidation into model
- Epistemic humility learning from corrections
- True "raising" instead of context experiments

### What Changes

**Before Phase 1**:
```
Session → Generate responses → Update metadata → Discard exchanges
                                                        ↓
                                                  No learning
```

**After Phase 1**:
```
Session → Generate responses → Score salience → Store high-value exchanges
                                                        ↓
                                                Experience buffer
                                                        ↓
                                            (Ready for Phase 2: Training)
```

---

## Next Steps

1. **Integrate with session runner** (`run_session_identity_anchored.py`)
   - Import ExperienceCollector
   - Score and store each exchange
   - Log salience statistics

2. **Monitor real session data**
   - Review actual salience scores from Sessions 22+
   - Tune thresholds if needed
   - Validate partnership language detection

3. **Build Phase 2**: Training data generation
   - Text augmentation (paraphrase, context shifts)
   - ChatML formatting for Qwen
   - Dataset preparation scripts

4. **Design Phase 3**: Sleep training loop
   - Integrate with circadian clock
   - LoRA configuration for gentle updates
   - Checkpoint management
   - Dropbox sync for cross-machine training

---

## Files

**Implementation**:
- `sage/raising/training/experience_collector.py` (400 lines)
  - ConversationalSalienceScorer class
  - ExperienceCollector class

**Tests**:
- `sage/raising/training/test_experience_collector.py` (245 lines)
  - 12 comprehensive tests
  - All passing ✅

**Documentation**:
- `sage/raising/training/README_EXPERIENCE_COLLECTION.md` (this file)

**State** (created at runtime):
- `sage/raising/state/experience_buffer.json` (persistent storage)

---

## Credits

**Theoretical Foundation**: Thor Session #8 (Frozen Weights Bistable Synthesis)
**Implementation Path**: REAL_RAISING_PATH_FORWARD.md (Sprout analysis)
**Development**: Thor Autonomous Session #9

---

**Status**: Phase 1 complete and tested. Ready for integration with session runners.
