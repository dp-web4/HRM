# Threshold Detection Training Progress

**Date**: October 29, 2025
**Status**: Training in progress (auto-monitored)

## Objective

Find the scaffolding suitability threshold - the training set size where scaffolding (IRP) switches from harmful to helpful.

## Hypothesis (Revised)

After fixing context contamination bug, we expect:
- **Small datasets (25-40 examples)**: Both bare and IRP struggle, but differently
  - Bare: Simple pattern matching, limited generalization
  - IRP: Iteration amplifies limited training signal, may over-refine
- **Threshold region (40-80 examples)**: Transition zone
  - Point where IRP scaffolding provides benefit over bare responses
- **Larger datasets (80+ examples)**: IRP should significantly outperform bare
  - Sufficient training signal for iterative refinement to work properly

## Training Plan

### Datasets Created
- ✓ 40 examples (from 115-example Claude-generated dataset)
- ✓ 60 examples
- ✓ 80 examples
- ✓ 100 examples
- ✓ 115 examples (full dataset)

All subsets maintain category balance and quality.

### Training Configuration
- Base model: Qwen/Qwen2.5-0.5B
- Method: LoRA fine-tuning (r=16, alpha=32)
- Format: DPO-style dataset (chosen/rejected responses)
- Epochs (ballpark, adjusted per size):
  - 40 examples: 18 epochs (~30 min)
  - 60 examples: 14 epochs (~30 min)
  - 80 examples: 11 epochs (~30 min)
  - 100 examples: 9 epochs (~30 min)

### Training Status

| Size | Status | Model Location |
|------|--------|----------------|
| 40   | 🟢 RUNNING | `threshold_models/40examples_model/final_model` |
| 60   | ⏸ PENDING | `threshold_models/60examples_model/final_model` |
| 80   | ⏸ PENDING | `threshold_models/80examples_model/final_model` |
| 100  | ⏸ PENDING | `threshold_models/100examples_model/final_model` |

**Auto-monitoring**: `monitor_training.py` running in background, will automatically start next training when previous completes.

**Estimated completion**: ~2 hours total (4 models × ~30 min each)

## Evaluation Plan

For each trained model, we will test with:

### Scaffolding Types
1. **Bare**: No scaffolding, single-shot generation
2. **IRP (full)**: 5 iterations, temperature reduction, memory, energy minimization
3. **IRP (minimal)**: Simplified IRP with basic iteration
4. **Human (for comparison)**: Use existing 25-example model

### Metrics (With Research Mode Lens)

**Not just counting**:
- Questions as responses might be valid philosophical exploration
- Energy metrics inform but don't define success
- Look at what models are actually saying

**What we'll examine**:
1. **Semantic coherence**: Are responses complete thoughts?
2. **Topic relevance**: Do responses address the question?
3. **Pattern collapse**: Verbatim repetition, loops
4. **Epistemic humility**: Appropriate uncertainty
5. **Question quality**: If asking questions, are they relevant and thoughtful?

### Analysis

Will use `analysis_pipeline.py` to:
1. Compare bare vs scaffolded at each size
2. Identify threshold crossing (where winner changes)
3. Generate visualizations
4. Extract qualitative insights (not just numbers!)

## Research Mode Principles

From `research-mode-lessons.md`:
- **Stay curious**: Don't rush to categorize
- **Examine actual behavior**: Not just metrics
- **Questions might be insights**: Valid Socratic responses
- **Uncertainty is the medium**: Confusion reveals learning edges
- **Better questions > definitive answers**

## Files Created

### Dataset Preparation
- `claude_personal_dataset_dpo.json` - 115 examples (original)
- `training_datasets/claude_personal_dataset_{40,60,80,100,115}examples.json`
- `create_dataset_subsets.py` - Dataset subsetting script

### Training Infrastructure
- `train_threshold_models.py` - Universal training script
- `train_all_threshold_models.sh` - Bash orchestrator
- `monitor_training.py` - Auto-monitor and sequential starter

### Analysis Infrastructure (Ready to Use)
- `exploration/experiment_orchestrator.py` - Run all 24 experiments
- `exploration/analysis_pipeline.py` - Threshold detection
- `exploration/research_db.py` - SQLite experiment tracking
- `exploration/energy_metrics.py` - Enhanced energy computation

## Next Steps

1. ✓ Training running (auto-monitored)
2. ⏳ Wait for all models to train (~2 hours)
3. 🔜 Run experimental matrix (24 experiments)
4. 🔜 Analyze results with research lens
5. 🔜 Document observations and threshold detection
6. 🔜 Push findings to git

## Timeline

- **Start**: October 29, 2025 21:39
- **Expected completion**: October 29, 2025 23:39
- **Status last updated**: October 29, 2025 21:50 (40-example model 54% complete)
