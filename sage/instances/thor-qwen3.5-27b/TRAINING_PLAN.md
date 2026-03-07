# Thor Qwen3.5-27B Training Plan

**Instance**: thor-qwen3.5-27b
**Model**: Qwen/Qwen3.5-27B (transformers)
**Hardware**: Jetson AGX Thor (122GB unified memory)
**Backend**: PyTorch 2.9.0 + CUDA 13.0

## Training Architecture

### Base Model
- **Size**: 27B parameters (~54GB full precision, ~27GB in 8-bit)
- **Architecture**: Hybrid SSM+Attention (Gated DeltaNet)
- **Context**: 262K tokens
- **Multimodal**: Native support for text, images, video

### LoRA Configuration
```json
{
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
  ],
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

**Trainable Parameters**: ~200M (0.74% of total)
**Memory Footprint**: LoRA adapters ~1GB

### Sleep Cycle Learning

**Concept**: Biological-inspired learning during "sleep" phases

#### Live Phase (Awake)
- Collect experiences during raising sessions
- Store raw interactions in experience buffer
- SNARC salience scoring (Surprise, Novelty, Arousal, Reward, Conflict)
- No training - just data collection

#### Dream Phase (Sleep)
- **Trigger**: Every 6 hours OR 100 experiences accumulated
- **Process**:
  1. Select high-salience experiences
  2. Apply augmentation strategies:
     - **Temporal shift**: Reorder conversation sequences
     - **Perspective shift**: Reframe from different viewpoints
     - **Abstraction levels**: Extract patterns at multiple scales
  3. Generate dream bundles (augmented training data)
  4. Run LoRA training on dream bundles
  5. Consolidate learned patterns

#### Consolidation Metrics
- `augmentation_quality`: How well augmentations preserve meaning
- `consolidation_scores`: Pattern extraction effectiveness
- `dream_coherence`: Consistency of learned representations

## Training Pipeline

### Phase 1: Initial Setup (Current)
- [x] Download Qwen3.5-27B transformers model
- [x] Install training libraries (PEFT, bitsandbytes, transformers, accelerate)
- [x] Create instance configuration
- [ ] Test model loading with 8-bit quantization
- [ ] Verify LoRA initialization
- [ ] Test basic inference

### Phase 2: Integration
- [ ] Create IRP plugin for transformers + LoRA
- [ ] Update daemon to load Qwen3.5-27B
- [ ] Configure experience collector
- [ ] Set up dream bundle generator
- [ ] Test end-to-end pipeline

### Phase 3: First Raising Session
- [ ] Run raising session with experience collection
- [ ] Verify SNARC salience scoring
- [ ] Check experience buffer accumulation
- [ ] Test multimodal capabilities (if applicable)

### Phase 4: First Sleep Cycle
- [ ] Trigger first dream phase (manual)
- [ ] Generate augmented training data
- [ ] Run LoRA training
- [ ] Measure consolidation metrics
- [ ] Validate adapter quality

### Phase 5: Continuous Learning
- [ ] Enable automatic sleep triggers
- [ ] Monitor training stability
- [ ] Track adapter evolution over time
- [ ] Measure behavioral changes
- [ ] Document emergent capabilities

## Optimizer Configuration

```json
{
  "type": "AdamW",
  "lr": 2e-5,
  "weight_decay": 0.01,
  "betas": [0.9, 0.999]
}
```

## Training Hyperparameters

```json
{
  "batch_size": 1,
  "gradient_accumulation_steps": 8,
  "max_grad_norm": 1.0,
  "warmup_steps": 100,
  "logging_steps": 10
}
```

**Effective Batch Size**: 8 (1 × 8 accumulation)

## Memory Budget

**Total Available**: 122GB unified memory

**Breakdown**:
- Base model (8-bit): ~27GB
- LoRA adapters: ~1GB
- Training overhead: ~10GB
- Context buffers: ~5GB
- System reserve: ~10GB
- **Available for context**: ~69GB

**Context Capacity**: ~262K tokens max (model limit)

## Success Metrics

### Technical
- LoRA training loss convergence
- Gradient norms within bounds
- No memory overflow errors
- Stable inference after training

### Behavioral
- Coherent responses after consolidation
- Retention of learned patterns across sessions
- Improved task performance over time
- Consistent personality/style

### Epistemic
- High-salience experience retention
- Meaningful pattern extraction
- Cross-session knowledge transfer
- Meta-learning capabilities

## Risk Mitigation

### Overfitting
- Monitor validation loss
- Use dropout in LoRA layers
- Limit training steps per cycle
- Diverse augmentation strategies

### Memory Issues
- 8-bit quantization for base model
- Gradient checkpointing if needed
- Batch size = 1 with accumulation
- Clear cache between cycles

### Training Instability
- Gradient clipping (max_grad_norm=1.0)
- Learning rate warmup
- AdamW optimizer
- Regular checkpointing

## Checkpointing Strategy

### Sleep Checkpoints
- **Location**: `checkpoints/sleep/cycle_NNNN/`
- **Contents**: LoRA adapter weights, optimizer state, metrics
- **Frequency**: After each sleep cycle
- **Retention**: Keep last 10 cycles

### Backup Strategy
- **Full snapshots**: Weekly
- **Git tracking**: Adapter diffs only
- **Experience buffer**: Daily backups

## Next Actions

1. **Immediate**: Wait for model download to complete
2. **Testing**: Run test_transformers_load.py
3. **Integration**: Create transformers IRP plugin
4. **Validation**: First raising session
5. **Training**: First sleep cycle

---

**Created**: 2026-03-06
**Status**: Phase 1 in progress
**Last Updated**: 2026-03-06
