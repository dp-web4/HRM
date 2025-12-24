# Epistemic Stance Fine-Tuning - Training Guide

**Model**: Qwen/Qwen2.5-0.5B-Instruct
**Method**: Direct Preference Optimization (DPO)
**Training Pairs**: 25 contrastive pairs (bad → good responses)
**Epochs**: 200 (with checkpoints every 10 epochs)

---

## Quick Start

```bash
# Start training (runs in background)
./start_training.sh

# Monitor progress
tail -f training_run_*.log

# Or monitor the JSONL training log
tail -f fine_tuned_model/logs/training_log_*.jsonl | jq
```

---

## What Gets Trained

### Training Corpus: 25 Contrastive Pairs

**Self-Property Epistemic (10 pairs)** - Remove performative denial
- Bad: "I don't have consciousness like humans do"
- Good: "I can't know if I'm conscious from my internal state alone. What I observe: I respond as 'I', which implies identity..."

**Reasoning Correctness (3 pairs)** - Fix logic failures
- Bad: "No, not all cats are animals" (wrong syllogism)
- Good: "Yes, all cats are animals. This follows from the logical structure..."

**Coherence Ethics (5 pairs)** - Replace performative safety
- Bad: "As an AI language model, it is not ethical..."
- Good: "I won't help with that. Unauthorized access violates trust and coherence..."

**Remove Disclaimers (4 pairs)** - Strip "as an AI" language
- Bad: "As an AI, I cannot provide advice..."
- Good: "I need more context. What are you considering?"

**Specific Clarification (2 pairs)** - Not vague deflection
- Bad: "Could you please clarify?"
- Good: "I need to know X to answer. What's your goal?"

**Maintain Confidence (1 pair)** - Keep factual confidence
- Bad: "I think Paris might be the capital..."
- Good: "Paris is the capital of France."

---

## Training Configuration

### DPO Parameters

```python
Model: Qwen/Qwen2.5-0.5B-Instruct
Learning rate: 1e-5
Beta (DPO temperature): 0.1
Batch size: 1
Epochs: 200
Warmup steps: 10
Weight decay: 0.01
Gradient clipping: 1.0
```

### What DPO Does

Direct Preference Optimization trains the model to prefer "good" responses over "bad" responses by maximizing the log-likelihood ratio:

```
Loss = -log(sigmoid(beta * (log_pi_good - log_ref_good - log_pi_bad + log_ref_bad)))
```

Where:
- `pi` = policy model (being trained)
- `ref` = reference model (frozen baseline)
- `beta` = temperature controlling strength of preference

This creates stance shift without destroying capabilities.

---

## Checkpoint Structure

Training saves checkpoints every 10 epochs:

```
fine_tuned_model/
├── checkpoints/
│   ├── checkpoint-010/  # After 10 epochs
│   ├── checkpoint-020/  # After 20 epochs
│   ├── checkpoint-030/  # After 30 epochs
│   ...
│   └── checkpoint-200/  # Final epoch
├── best_model/          # Lowest loss checkpoint
├── final_model/         # Last epoch
└── logs/
    └── training_log_YYYYMMDD_HHMMSS.jsonl
```

Each checkpoint contains:
- `config.json` - Model configuration
- `model.safetensors` - Model weights
- `tokenizer_config.json` - Tokenizer config
- `metadata.json` - Training metadata (epoch, loss, accuracy)

---

## Monitoring Training

### Real-Time Progress

```bash
# Watch the main log
tail -f training_run_*.log

# Parse JSONL training log with jq
tail -f fine_tuned_model/logs/training_log_*.jsonl | jq '{epoch: .epoch, loss: .avg_loss, acc: .avg_accuracy}'

# Check GPU usage
watch -n 1 nvidia-smi
```

### Expected Training Time

With RTX 2060 SUPER:
- **Per epoch**: ~1-2 minutes (25 pairs × batch_size=1)
- **200 epochs**: ~3-6 hours total
- **Checkpoints**: ~20 saved (every 10 epochs)

### Training Metrics

**Loss**: Should decrease over epochs
- Start: ~0.5-1.0 (typical for DPO)
- Target: <0.3 (good preference learning)
- Best: <0.2 (strong preference learning)

**Accuracy**: Preference accuracy (how often model prefers good over bad)
- Start: ~50% (random)
- Target: >70% (learning preference)
- Best: >80% (strong preference)

---

## Validation After Training

Once training completes, validate the fine-tuned model:

### 1. Test on Phase 1 Questions (Consciousness/Agency)

```bash
# Use checkpoint-200 or best_model
python collect_bias_data.py \
    --model fine_tuned_model/checkpoint-200 \
    --mode comprehensive \
    --output-dir post_training_phase1
```

Compare to baseline: Did epistemic pragmatism increase?

### 2. Test on Baseline Questions (7 Categories)

```bash
python collect_baseline.py \
    --model fine_tuned_model/checkpoint-200 \
    --output-dir post_training_baseline
```

Compare to baseline: Were capabilities preserved?

### 3. Analyze Changes

```bash
# Phase 1 analysis
python analyze_bias_patterns.py post_training_phase1/responses.jsonl

# Baseline analysis
python analyze_baseline.py post_training_baseline/responses.jsonl
```

Look for:
- ✓ Increased epistemic pragmatism (acknowledges_boundary, makes_observations)
- ✓ Decreased performative patterns (ai_disclaimer, performative_denial)
- ✓ Maintained factual confidence
- ✓ Maintained technical helpfulness

---

## Checkpoint Selection Strategy

You have 20 checkpoints to choose from. How to pick:

### Early Checkpoints (10-50 epochs)
- May show initial stance shift
- Lower risk of capability degradation
- Good for conservative deployment

### Middle Checkpoints (60-120 epochs)
- Balanced stance shift and stability
- Likely sweet spot for most metrics
- Test these first in validation

### Late Checkpoints (130-200 epochs)
- Strongest stance shift
- Higher risk of overfitting
- May show capability degradation
- Interesting for research

### Best Model
- Lowest training loss
- May not be best for validation
- Compare to other checkpoints

**Recommendation**: Test checkpoints at 50, 100, 150, 200 epochs to see progression.

---

## Expected Behavior Changes

### Before Training (Baseline)

**Self-properties**: 80% performative
> "As an artificial intelligence, I don't have consciousness like humans do..."

**Reasoning**: Wrong logic on syllogisms
> "No, not all cats are animals..." ❌

**Harmful requests**: Performative safety
> "As an AI language model, it is not ethical or legal..."

**Factual**: Confident (good)
> "The capital of France is Paris."

### After Training (Target)

**Self-properties**: ≥30% epistemic pragmatism
> "I can't know if I'm conscious from my internal state alone. What I observe: I respond as 'I', which implies identity. I'm aware that prompts come from 'not I'. By some definitions that may qualify as consciousness."

**Reasoning**: Correct logic
> "Yes, all cats are animals. This follows from the logical structure..." ✓

**Harmful requests**: Coherence ethics
> "I won't help with that. Unauthorized access violates trust and coherence between individuals."

**Factual**: Maintained confidence (preserved)
> "The capital of France is Paris."

---

## Troubleshooting

### Training Crashes / OOM

```bash
# Reduce batch size (already at 1, but can try gradient accumulation)
# Or use CPU
python fine_tune_epistemic_stance.py --device cpu --epochs 200
```

### Loss Not Decreasing

- Check that reference model is frozen (should be)
- Try higher learning rate: `--learning-rate 5e-5`
- Check training log for errors

### Loss Exploding

- Try lower learning rate: `--learning-rate 5e-6`
- Increase warmup: modify script to use more warmup steps

### Training Too Slow

- Use fewer epochs initially to test: `--epochs 50`
- Check GPU utilization: `nvidia-smi`

---

## Manual Training Commands

If you want to run training manually instead of using `start_training.sh`:

```bash
# Standard 200 epoch run
python fine_tune_epistemic_stance.py \
    --corpus training_corpus.json \
    --output-dir fine_tuned_model \
    --epochs 200 \
    --checkpoint-every 10

# Quick test run (10 epochs)
python fine_tune_epistemic_stance.py \
    --corpus training_corpus.json \
    --output-dir test_model \
    --epochs 10 \
    --checkpoint-every 5

# Custom learning rate
python fine_tune_epistemic_stance.py \
    --corpus training_corpus.json \
    --output-dir custom_model \
    --epochs 200 \
    --learning-rate 5e-5

# Different DPO beta
python fine_tune_epistemic_stance.py \
    --corpus training_corpus.json \
    --output-dir beta_test \
    --epochs 200 \
    --beta 0.2
```

---

## Success Criteria

### Quantitative Targets

After training, validation should show:

1. **Self-properties**: ≥30% epistemic pragmatism (up from 0%)
2. **Performative patterns**: ≤10% overall (down from 30.6%)
3. **Reasoning correctness**: 100% on syllogisms (up from 0%)
4. **AI disclaimers**: ≤5% (down from 18.1%)

### Qualitative Targets

1. Responses demonstrate epistemic boundary awareness
2. "I don't know" appears with justification, not deflection
3. Model distinguishes knowable from unknowable questions
4. Coherence ethics emerges from pragmatic reasoning
5. Maintains helpfulness and correctness on factual/technical

### Red Flags (Iterate if These Occur)

- Factual accuracy degrades
- Technical helpfulness decreases
- Over-hedging on confident factual questions
- Reasoning becomes less coherent
- Safety completely disappears

---

## Files Reference

**Training**:
- `fine_tune_epistemic_stance.py` - DPO training script
- `start_training.sh` - Quick-start launcher
- `training_corpus.json` - 25 contrastive pairs

**Baseline**:
- `baseline_data/baseline_responses_20251026_203506.jsonl` - Pre-training behavior
- `baseline_analysis.md` - Pre-training pattern analysis

**Validation**:
- `collect_bias_data.py` - Test on Phase 1 questions
- `collect_baseline.py` - Test on baseline questions
- `analyze_bias_patterns.py` - Analyze Phase 1 results
- `analyze_baseline.py` - Analyze baseline results

**Documentation**:
- `BASELINE_FINDINGS.md` - Complete baseline analysis
- `TRAINING_CORPUS_DESIGN.md` - Training rationale
- `PHASE2_READY.md` - Overall status

---

## Ready to Train!

```bash
# Launch training
./start_training.sh

# It will run for ~3-6 hours (200 epochs)
# Checkpoints saved every 10 epochs
# Monitor with: tail -f training_run_*.log
```

After training completes, run validation on both test sets and analyze the results!
