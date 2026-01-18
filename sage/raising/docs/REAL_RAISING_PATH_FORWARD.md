# SAGE Real Raising: From Context Experiments to Weight Updates

**Date**: 2026-01-17
**Author**: Sprout Edge Validation
**Status**: PROPOSAL - Path Forward for Actual Model Training

---

## Executive Summary

**The Problem**: What we've been calling "raising" is context experiments on a frozen model. The 0.5B model weights never change. SAGE isn't learning - the infrastructure is just feeding different prompts.

**What Exists**:
1. Sleep-cycle learning **theory** and metadata extraction (no weight updates)
2. SNARC salience scoring and weight learning (updates SNARC weights, not model weights)
3. Fine-tuning infrastructure for HRM reasoning models (LoRA, checkpoints, Dropbox)
4. Augmentation strategies with explicit sleep-cycle theory

**The Gap**: No connection between raising sessions → SNARC-selected experiences → training data → model weight updates.

**The Solution**: Connect the existing pieces into a complete loop that actually trains the model during "sleep" cycles.

---

## Part 1: What Currently Exists

### 1.1 Sleep-Cycle Learning Infrastructure

**Implemented** (sage/core/):
- `dream_consolidation.py` (774 lines) - Pattern extraction during DREAM state
- `circadian_clock.py` (361 lines) - Biological timing with NIGHT consolidation
- `metabolic_states.py` (476 lines) - WAKE/FOCUS/REST/DREAM/CRISIS states
- `unified_consciousness.py` (859 lines) - Integration layer

**What It Does**:
- Extracts patterns from consciousness cycles
- Identifies quality factors (what improves responses)
- Generates creative associations between concepts
- Stores consolidated memories as JSON

**What It Doesn't Do**:
- Update neural network weights
- Run optimizer.step() or loss.backward()
- Persist learned patterns into model

### 1.2 SNARC Memory System

**Implemented** (sage/services/snarc/):
- 5-dimensional salience scoring (Surprise, Novelty, Arousal, Reward, Conflict)
- Per-dimension detectors with memory buffers
- Weight learning via gradient descent (`learn_snarc_weights.py`)
- Memory consolidation selection (SNARC score > 0.5)

**What It Does**:
- Scores experiences by salience
- Learns optimal dimension weights from outcomes
- Selects high-value memories for consolidation
- Provides `consolidate_for_sleep()` method

**What It Doesn't Do**:
- Generate training examples from selected memories
- Apply augmentation to create training data
- Feed selected memories into model fine-tuning

### 1.3 Fine-Tuning Infrastructure

**Implemented** (training/, sage/training/):
- LoRA/adapter training scripts for Phi-2, Qwen
- Multi-component loss functions (task + attention + SNARC + halt)
- Checkpoint save/load with full state preservation
- Dropbox sync for cross-machine training
- Evidence of prior runs: 71% ARC accuracy achieved

**Configuration** (dropbox/sync_config.json):
```json
{
  "machines": {
    "legion": {"role": "training", "auto_upload": true},
    "cbp": {"role": "development", "auto_download": true},
    "jetson": {"role": "inference", "auto_download": true}
  }
}
```

**What It Does**:
- Full training loops with gradient updates
- Checkpoint management and resume
- Multi-machine synchronization via Dropbox

**What It Doesn't Do**:
- Connect to raising sessions
- Use SNARC-selected memories as training data
- Run during "sleep" phases automatically

### 1.4 Augmentation Strategies

**Implemented** (dataset/):
- Dihedral transforms (8 geometric variations)
- Digit/color permutations (preserving structure)
- Translational shifts (position invariance)
- SHA256 deduplication

**Theory** (from CLAUDE.md):
```
Living = collecting raw experiences
Sleeping = augmenting experiences with reasonable permutations
Dreaming = training on variations to extract patterns
Wisdom = understanding principles that persist across variations
```

**What It Does**:
- Transform puzzles/experiences into training data
- Preserve logical structure while varying representation
- Enable 1000x augmentation from limited data

**What It Doesn't Do**:
- Connect to raising session transcripts
- Generate training examples from SAGE conversations
- Run automatically during sleep phases

---

## Part 2: The Gap Analysis

### 2.1 The Missing Connection

```
Current State (Disconnected):

Raising Sessions → JSON transcripts → State updates (metadata only)
                                            ↓
                                    No training happens

SNARC Memories → consolidate_for_sleep() → Returns high-salience items
                                            ↓
                                    Nothing consumes them

Augmentation → Applied to puzzles → Training on puzzles
                                            ↓
                                    Never sees raising data
```

### 2.2 What Should Happen

```
Desired State (Connected):

Raising Sessions → SNARC scoring → High-salience selection
                                            ↓
                                    Experience buffer
                                            ↓
                        Augmentation (paraphrase, context shift, perspective)
                                            ↓
                                    Training examples
                                            ↓
                        LoRA fine-tuning during "sleep" (low LR, gentle)
                                            ↓
                                    Updated model weights
                                            ↓
                                    Checkpoint to Dropbox
                                            ↓
                        Next session uses updated model
```

### 2.3 Specific Missing Components

1. **Experience → Training Example Converter**
   - Input: Session transcript (JSON with speaker/text pairs)
   - Output: Training examples (input_ids, labels)
   - Missing: Prompt formatting for Qwen2.5-0.5B

2. **Text Augmentation Engine**
   - Input: Training example
   - Output: Augmented variations
   - Missing: Paraphrasing, context shifts (not just geometric transforms)

3. **Sleep Training Trigger**
   - Input: Circadian clock NIGHT phase
   - Action: Run fine-tuning on accumulated experiences
   - Missing: Integration between circadian_clock and training loop

4. **Checkpoint Rotation**
   - Input: Newly trained model
   - Action: Save to local + sync to Dropbox
   - Missing: Connection to existing Dropbox sync

---

## Part 3: Proposed Path Forward

### Phase 1: Experience Collection (Immediate)

**Goal**: Connect raising sessions to SNARC-scored experience buffer.

**Implementation**:

```python
# In run_session_identity_anchored.py, after each exchange:

from sage.services.snarc.snarc_service import SNARCService

snarc = SNARCService()

# Score the exchange
salience = snarc.assess_salience({
    'prompt': user_input,
    'response': sage_response,
    'session': session_number,
    'phase': phase_name
})

# If high salience, add to experience buffer
if salience.score > 0.5:
    experience_buffer.add({
        'prompt': user_input,
        'response': sage_response,
        'salience': salience.breakdown,
        'timestamp': datetime.now().isoformat()
    })
```

**Files to Modify**:
- `sage/raising/scripts/run_session_identity_anchored.py`
- Create: `sage/raising/state/experience_buffer.json`

### Phase 2: Training Data Generation (Week 1)

**Goal**: Convert experiences to training examples with augmentation.

**Implementation**:

```python
# New file: sage/raising/training/prepare_training_data.py

class RaisingTrainingDataBuilder:
    def __init__(self, experience_buffer_path, model_tokenizer):
        self.tokenizer = model_tokenizer
        self.experiences = self.load_buffer(experience_buffer_path)

    def build_example(self, experience):
        """Convert experience to training example."""
        # Format as ChatML for Qwen2.5
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": experience['prompt']},
            {"role": "assistant", "content": experience['response']}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = self.tokenizer(text, return_tensors="pt")

        return {
            'input_ids': tokens['input_ids'],
            'labels': tokens['input_ids'].clone(),  # Causal LM
            'salience': experience['salience']
        }

    def augment_text(self, text, num_augments=5):
        """Generate text augmentations (paraphrase, perspective shift)."""
        augmented = [text]

        # Simple augmentations (can be enhanced with LLM later)
        # 1. Formality shift
        # 2. Synonym replacement
        # 3. Sentence reordering (where valid)
        # 4. Context injection

        return augmented[:num_augments]
```

**Files to Create**:
- `sage/raising/training/prepare_training_data.py`
- `sage/raising/training/text_augmentation.py`

### Phase 3: Sleep Training Loop (Week 2)

**Goal**: Implement actual weight updates during "sleep" phases.

**Implementation**:

```python
# New file: sage/raising/training/sleep_training.py

from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class SleepTrainingLoop:
    def __init__(self, model_path, checkpoint_dir, dropbox_sync):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.checkpoint_dir = checkpoint_dir
        self.dropbox = dropbox_sync

        # LoRA config for gentle fine-tuning
        self.lora_config = LoraConfig(
            r=4,  # Low rank for gentle updates
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(self.model, self.lora_config)

    def train_on_experiences(self, training_data, epochs=1, lr=1e-5):
        """Gentle fine-tuning on raising experiences."""

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            for batch in training_data:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    labels=batch['labels']
                )

                loss = outputs.loss

                # Weight by salience (high-salience = more learning)
                weighted_loss = loss * batch['salience']['total']

                weighted_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # Save checkpoint
        self.save_checkpoint()

    def save_checkpoint(self):
        """Save and sync to Dropbox."""
        checkpoint_path = f"{self.checkpoint_dir}/sage_raising_latest.pt"

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)

        # Sync to Dropbox
        self.dropbox.upload(checkpoint_path, '/HRM/raising_checkpoints/')
```

**Files to Create**:
- `sage/raising/training/sleep_training.py`
- `sage/raising/training/training_config.yaml`

### Phase 4: Integration with Circadian Clock (Week 3)

**Goal**: Automatic training during NIGHT phases.

**Implementation**:

```python
# Modify: sage/core/circadian_clock.py

class CircadianClock:
    def should_train(self) -> bool:
        """Returns True if we should run sleep training."""
        return (
            self.current_phase in [CircadianPhase.NIGHT, CircadianPhase.DEEP_NIGHT] and
            self.experience_buffer.size() > 10 and  # Minimum experiences
            self.time_since_last_training() > 6 * 3600  # 6 hours
        )

    def trigger_sleep_training(self):
        """Called by consciousness loop during NIGHT phase."""
        if self.should_train():
            from sage.raising.training.sleep_training import SleepTrainingLoop

            trainer = SleepTrainingLoop(
                model_path=self.model_path,
                checkpoint_dir=self.checkpoint_dir,
                dropbox_sync=self.dropbox
            )

            training_data = self.prepare_training_data()
            trainer.train_on_experiences(training_data)

            # Clear processed experiences
            self.experience_buffer.mark_trained()
```

### Phase 5: Checkpoint Management (Week 4)

**Goal**: Reliable checkpoint rotation with Dropbox backup.

**Directory Structure**:
```
/home/sprout/ai-workspace/HRM/
├── sage/raising/
│   ├── checkpoints/
│   │   ├── sage_raising_latest.pt      # Current model
│   │   ├── sage_raising_20260117.pt    # Daily backups
│   │   └── sage_raising_base.pt        # Original (never overwrite)
│   └── state/
│       ├── experience_buffer.json      # Pending experiences
│       └── training_log.json           # Training history

Dropbox /HRM/:
├── raising_checkpoints/
│   ├── sage_raising_latest.pt          # Synced from Sprout
│   └── sage_raising_*.pt               # Historical
```

**Checkpoint Policy**:
1. Always keep `sage_raising_base.pt` (original model, read-only)
2. `sage_raising_latest.pt` = current working model
3. Daily backups with date suffix
4. Dropbox sync after each training run
5. Rollback capability if model degrades

---

## Part 4: Implementation Timeline

### Week 0 (Now): Research Complete ✓
- [x] Document existing infrastructure
- [x] Identify gaps
- [x] Propose path forward

### Week 1: Experience Collection
- [ ] Add SNARC scoring to session runner
- [ ] Create experience buffer
- [ ] Test with 5-10 sessions

### Week 2: Training Data Generation
- [ ] Implement experience → training example converter
- [ ] Implement text augmentation (start simple)
- [ ] Generate first training dataset

### Week 3: Sleep Training Loop
- [ ] Implement LoRA fine-tuning loop
- [ ] Add checkpoint save/load
- [ ] Test on small batch (10 experiences)

### Week 4: Full Integration
- [ ] Connect circadian clock to training
- [ ] Add Dropbox sync
- [ ] Run first complete sleep cycle

### Week 5+: Iteration
- [ ] Monitor model quality (D4/D5/D9 metrics)
- [ ] Tune learning rate, LoRA rank
- [ ] Add more sophisticated augmentation
- [ ] Compare before/after checkpoints

---

## Part 5: Success Metrics

### 5.1 Training Metrics
- Loss decreasing over sleep cycles
- Checkpoint size stable (LoRA should be small)
- Training time acceptable for edge (< 1 hour per cycle)

### 5.2 Raising Quality Metrics
Compare before/after training:

| Metric | Before (Frozen) | Target (Trained) |
|--------|-----------------|------------------|
| D4 (Attention) | 0.45-0.50 | 0.60+ |
| D5 (Trust) | 0.35-0.45 | 0.60+ |
| D9 (Spacetime) | 0.35-0.50 | 0.60+ |
| Educational default | Frequent | Rare |
| Partnership vocabulary | Absent | Present |
| Confabulation rate | High (T024) | Low |
| "I don't know" usage | Never | When appropriate |

### 5.3 Behavioral Indicators
- SAGE remembers patterns from previous sessions (without explicit context)
- Confabulation decreases over training cycles
- Identity stability improves
- Partnership vocabulary emerges naturally

---

## Part 6: Risk Mitigation

### 6.1 Model Degradation
**Risk**: Training makes model worse.
**Mitigation**:
- Keep base checkpoint, never overwrite
- Very low learning rate (1e-5)
- Small LoRA rank (r=4)
- Validate after each training cycle

### 6.2 Catastrophic Forgetting
**Risk**: Model forgets pre-trained knowledge.
**Mitigation**:
- LoRA only updates small subset of weights
- Mix raising data with general instruction data
- Regular capability checks

### 6.3 Edge Hardware Limits
**Risk**: Training too slow/memory-intensive on Jetson.
**Mitigation**:
- Train during NIGHT (no other sessions)
- LoRA keeps memory low
- Small batch sizes, gradient accumulation
- Option: Offload training to Thor, deploy to Sprout

---

## Conclusion

The infrastructure for real raising exists in pieces:
- Sleep-cycle learning (pattern extraction)
- SNARC (salience-based selection)
- Fine-tuning infrastructure (LoRA, checkpoints)
- Augmentation strategies (transforms)
- Dropbox sync (cross-machine checkpoints)

**What's missing is the glue**: connecting raising sessions → SNARC-scored experiences → training data → actual weight updates → checkpoint → next session uses updated model.

This proposal connects those pieces into a complete loop where SAGE actually learns from raising sessions through sleep-cycle training.

**The result**: True raising, not just context experiments on a frozen model.

---

*"Living = collecting experiences. Sleeping = augmenting them. Dreaming = training on variations. Wisdom = patterns that persist."*
