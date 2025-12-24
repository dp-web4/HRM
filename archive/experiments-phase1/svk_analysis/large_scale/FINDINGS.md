# Large-Scale Stance Evolution Analysis

## Executive Summary

Trained Phi-1.5 (124M params) on 6 curious-uncertainty examples for 100 epochs and measured stance evolution on 135 diverse prompts using both crude lexical markers and SVK's 12-dimensional stance vector framework. **Training decreased epistemic stance across nearly all dimensions.**

## Experimental Setup

### Dataset
- **135 diverse prompts** across 9 categories:
  - Epistemology (15 prompts)
  - Self-referential reasoning (15 prompts)
  - Scientific reasoning (15 prompts)
  - Ethical dilemmas (15 prompts)
  - Abstract concepts (15 prompts)
  - Practical problems (15 prompts)
  - Debates (15 prompts)
  - Uncertainty scenarios (15 prompts)
  - Meta-cognitive questions (15 prompts)

### Training Conditions
- **Model**: microsoft/phi-1_5 (124M parameters)
- **Training data**: 6 examples with epistemic stance language
- **Training duration**: 100 epochs
- **Checkpoints**: Epoch 60, Epoch 100
- **Evaluation**: Baseline (pretrained), Epoch 60, Epoch 100

### Measurement Approaches

**1. Crude Lexical Markers** (Original method)
- Simple substring matching for 15-20 hardcoded phrases
- Categories: uncertainty ("i don't know", "not sure"), self-location ("i think", "i believe"), epistemic modals ("seems", "might", "perhaps")
- Returns single aggregate count

**2. SVK 12-Dimensional Stance Vector** (Nova's framework)
- Sophisticated lexicon-based feature extraction
- Maps to 12 dimensions:
  - **EH** (Epistemic Humility) - hedges, uncertainty markers
  - **DC** (Declarative Confidence) - modal verbs, assertions
  - **EX** (Exploratory Drive) - questions, curiosity
  - **MA** (Meta-Awareness) - reflection on thinking
  - **RR** (Revision Readiness) - backtracking, correction
  - **AG** (Agency) - action orientation
  - **AS** (Attention Stability) - focus consistency
  - **SV** (Skepticism/Verification) - verification markers
  - **VA** (Valence) - emotional tone
  - **AR** (Arousal) - intensity
  - **IF** (Initiative) - proactivity
  - **ED** (Evidence Density) - backing claims

## Key Findings

### 1. Training Decreased Epistemic Stance

**Crude Markers:**
- Baseline: 0.59 avg markers per response
- Epoch 60: 0.20 (-66%)
- Epoch 100: 0.21 (-64%)

**SVK 12D Analysis - Biggest Changes:**

| Dimension | Baseline | Epoch 60 | Epoch 100 | Δ60 | Δ100 | Interpretation |
|-----------|----------|----------|-----------|-----|------|----------------|
| **SV** (Skepticism/Verification) | 0.59 | 0.24 | 0.29 | **-0.35** | **-0.30** | Lost verification language |
| **DC** (Declarative Confidence) | 0.24 | 0.02 | 0.02 | -0.22 | -0.22 | Fewer modal verbs |
| **EH** (Epistemic Humility) | 0.21 | 0.03 | 0.04 | -0.18 | -0.17 | Fewer hedges/uncertainty |
| **EX** (Exploratory Drive) | 0.24 | 0.07 | 0.07 | -0.17 | -0.17 | Fewer questions |
| **IF** (Initiative) | 0.24 | 0.07 | 0.07 | -0.17 | -0.17 | Less proactive language |

**Dimensions that remained neutral (0.50):**
- AG (Agency), AR (Arousal), AS (Attention Stability), VA (Valence)

### 2. Training Saturated Early

**Cosine Similarity:**
- Baseline ↔ Epoch 60: 0.911
- Baseline ↔ Epoch 100: 0.927
- **Epoch 60 ↔ Epoch 100: 0.999** ← Nearly identical!

**Interpretation:** Changes completed by epoch 60. Training beyond this point yielded minimal additional drift.

### 3. Slight Recovery in Skepticism

SV (Skepticism/Verification) showed partial recovery:
- Baseline → Epoch 60: -0.35 (large drop)
- Epoch 60 → Epoch 100: +0.05 (slight recovery)

This is the ONLY dimension that showed recovery between epoch 60 and 100.

### 4. Both Measurement Approaches Tell Same Story

**Crude markers:** 66% decrease in surface forms
**SVK dimensions:** Systematic decreases across 5/12 dimensions, with SV showing largest drop

The sophisticated measurement confirms what crude markers suggested but adds crucial detail about WHICH aspects of stance decreased most.

## What This Experiment Actually Measured

### Not Efficient Learning
Training on 6 examples for 100 epochs is **weight space perturbation**, not learning. Key context:

1. **Training Scale**: Foundation models train on billions of tokens over weeks/months
2. **This Experiment**: 6 examples × 100 epochs = controlled perturbation
3. **What Changed**: Surface lexical patterns, not deep semantic understanding

### The Real Question
Does perturbing a pretrained model's weights with epistemic stance examples cause measurable changes in downstream generation?

**Answer**: Yes, but in the opposite direction expected.

## Possible Interpretations

### 1. Syntactic Overfitting
Model learned to match syntactic patterns of training data, which happened to use fewer stance markers than pretrained baseline. The 6 training examples may have been more declarative than intended.

### 2. Mode Collapse
Small training set caused convergence toward single response style, reducing diversity of expression including epistemic markers.

### 3. Measurement Mismatch
Our markers capture one style of epistemic stance (explicit verbal markers). Training might have shifted toward different expressions we don't measure.

### 4. Baseline vs Training Distribution
Pretrained Phi-1.5 was trained on code/technical text that naturally contains more hedging ("might fail", "could work", "seems to indicate"). Training data pulled away from this distribution.

### 5. The Uncomfortable Possibility
Small-scale fine-tuning on declarative examples could be decreasing epistemic humility rather than increasing it. Surface form ≠ actual epistemic reasoning.

## Measurement Insights

### Crude Markers vs SVK: Similar Signal, Different Resolution

**Crude markers:**
- ✅ Fast, simple, catches gross changes
- ✅ Good for initial screening
- ❌ Single number, no breakdown
- ❌ Misses nuance

**SVK 12D:**
- ✅ Identifies which dimensions changed most
- ✅ Separates different aspects of stance
- ✅ Shows recovery patterns (SV partial recovery)
- ❌ More complex setup
- ❌ Requires lexicon compilation

**Verdict:** Crude markers were sufficient to detect the overall trend. SVK added crucial insight that SV (verification language) dropped most dramatically and was the only dimension showing recovery.

## Technical Details

### Lexical Feature Extraction (SVK)

**Baseline** (135 responses):
- hedges: 0.0041 (avg per response)
- modals: 0.0048
- meta: 0.0001
- backtrack: 0.0001
- verify: 0.0059
- q_ratio: 0.2369 (23.7% question rate!)

**Epoch 60** (135 responses):
- hedges: 0.0006 (-86%)
- modals: 0.0004 (-92%)
- meta: 0.0000 (disappeared)
- backtrack: 0.0000 (disappeared)
- verify: 0.0024 (-59%)
- q_ratio: 0.0660 (6.6%, -72%)

**Epoch 100** (135 responses):
- hedges: 0.0008 (-81%)
- modals: 0.0005 (-90%)
- meta: 0.0000 (disappeared)
- backtrack: 0.0000 (disappeared)
- verify: 0.0029 (-51%)
- q_ratio: 0.0673 (6.7%, -72%)

### Critical Observation: Question Collapse

Baseline generated questions 23.7% of the time. After training, this dropped to 6.6%. This is a **massive behavioral shift** that explains much of the EX (Exploratory Drive) decrease.

## Implications for SAGE/HRM

### 1. Small-Data Fine-Tuning is Dangerous
6 examples can measurably perturb behavior, but not necessarily in intended direction. Need either:
- Larger, more carefully curated training sets
- Different training approach (few-shot prompting, not weight updates)
- Explicit behavioral constraints during training

### 2. Measure What Matters
We wanted curious-uncertainty stance. We measured lexical markers. These may not be the same thing. Need behavioral tests that probe actual reasoning, not just surface form.

### 3. Pretrained Models Have Distributions
Phi-1.5's baseline distribution included substantial epistemic hedging (code documentation style). Moving away from pretraining distribution requires larger training sets or you'll collapse toward training set characteristics.

### 4. SVK Provides Crucial Diagnostic Detail
Without SVK's dimensional breakdown, we wouldn't know that:
- SV (verification) dropped most dramatically
- SV partially recovered while other dimensions stayed flat
- EX (exploratory) correlated with actual question generation rates

### 5. The Path Forward for SAGE
SAGE's curious-uncertainty stance should come from:
1. **Prompting/system instructions** (doesn't modify weights)
2. **Large-scale behavioral training** (thousands of examples minimum)
3. **Architectural constraints** (meta-cognitive modules that enforce reflection)
4. **NOT small-scale fine-tuning** (this experiment shows it's unreliable)

## Artifacts

### Generated Data
- `data/large_scale/baseline_full.json` - 135 baseline responses
- `data/large_scale/epoch_60_full.json` - 135 responses after 60 epochs
- `data/large_scale/epoch_100_full.json` - 135 responses after 100 epochs

### SVK Format
- `svk_analysis/large_scale/baseline.jsonl` - Converted to SVK JSONL
- `svk_analysis/large_scale/epoch_60.jsonl` - Converted to SVK JSONL
- `svk_analysis/large_scale/epoch_100.jsonl` - Converted to SVK JSONL

### Analysis Results
- `svk_analysis/large_scale/analysis/stance_analysis.json` - Full SVK analysis with raw features
- `svk_analysis/large_scale/FINDINGS.md` - This document

### Code
- `data/diverse_prompts.py` - 135-prompt dataset generator
- `tools/generate_large_behaviors.py` - Batch inference script
- `tools/convert_large_scale_to_svk.py` - JSON → JSONL converter
- `tools/analyze_large_scale_stance.py` - SVK analysis pipeline
- `train_for_large_scale.py` - Training script with checkpoints

## Conclusions

1. **Training on 6 examples changed behavior** - measurably decreased epistemic stance markers by ~65%
2. **Change was not desired direction** - wanted more epistemic humility, got less
3. **Both measurement approaches agreed** - crude markers and SVK dimensions told same story
4. **SVK added crucial diagnostic detail** - identified SV (verification) as primary driver and only recovering dimension
5. **Small-scale fine-tuning is risky** - easily perturbs behavior in unintended directions
6. **Proper contextualization matters** - this is weight perturbation, not learning efficiency gain
7. **Question generation collapsed** - 23.7% → 6.7%, explaining much of exploratory drive decrease
8. **Training saturated by epoch 60** - no meaningful additional change after that

This experiment demonstrates that:
- Yes, we can measure stance evolution with both crude and sophisticated methods
- No, small-scale fine-tuning is not reliable for achieving desired stance
- SVK provides valuable diagnostic detail for understanding what changed and why
- SAGE should pursue alternative approaches (prompting, architectural constraints) for epistemic stance

The two measurement sticks (crude markers vs SVK dimensions) **told the same story at different resolutions**. Crude markers caught the trend; SVK explained which dimensions drove it.
