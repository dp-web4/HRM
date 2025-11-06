# Model Size Inertia Experiment

## Hypothesis

**User's Prediction**: Larger models have more **inertia** - they're harder to fine-tune and slower to adapt to new conversational styles.

## Experimental Design

### Phase 1: Base Model Comparison

**Models**:
- Qwen 0.5B-Instruct (Sessions #1 & #2) ✓
- Qwen 7B-Instruct (Session #3) ⏳

**Method**: Same 15-turn conversation for both models

**Metrics**:
- Conversational quality (engagement, coherence)
- Trust evolution (learning rate)
- Energy variance (adaptation stability)
- Response time

### Phase 2: Fine-Tuned Comparison

**Training Dataset**: `claude_personal_dataset_dpo.json`
- 115 examples (epistemic pragmatism)
- DPO format: prompt, chosen (pragmatic), rejected (dogmatic)
- Same dataset used for both models

**Fine-Tuning Method**:
- Standard SFT (not DPO)
- LoRA (r=8, alpha=32)
- 3 epochs
- Learning rate: 1e-4
- Same hyperparameters for both models

**Models**:
- Qwen 0.5B + Epistemic Pragmatism ✓
- Qwen 7B + Epistemic Pragmatism ⏳

**Method**: Same 15-turn conversation testing pragmatic style

## Inertia Indicators

### 1. Training Efficiency
- **0.5B baseline**: 115 examples sufficient
- **7B test**: Does it need more examples?
- **Metric**: Final training loss, validation accuracy

### 2. Style Transfer
- **0.5B baseline**: Learns pragmatic stance (more questions, less certainty)
- **7B test**: Does it adopt same style, or resist?
- **Metric**: Question density, epistemic markers in responses

### 3. Trust Learning Rate
- **0.5B baseline**: Trust evolves 1.0 → 0.865 (15 turns)
- **7B test**: Does trust change more slowly?
- **Metric**: Trust delta per turn

### 4. Adaptation Stability
- **Small model**: Might be more "plastic" (variable outputs)
- **Large model**: Might be more "rigid" (consistent outputs)
- **Metric**: Energy variance across turns

### 5. Recovery from Failure
- **0.5B Session #2**: Failed on Turn 9, trust dropped appropriately
- **7B test**: Does it fail less (robust) or fail differently?
- **Metric**: Energy spikes, trust recovery

## Predictions

### If Inertia is Real (User's Hypothesis)

**Large model (7B)**:
- Needs MORE than 115 examples to learn style
- Trust changes SLOWLY (harder to shift)
- Low variance (rigid, consistent)
- Resists adopting pragmatic style (falls back to instruct training)

**Small model (0.5B)**:
- 115 examples sufficient
- Trust adapts quickly
- Higher variance (flexible, adaptive)
- Easily adopts new styles

### Alternative: Capability Trumps Plasticity

**Large model (7B)**:
- Learns FROM FEWER examples (more capable)
- Better quality responses (richer knowledge)
- Same or faster trust evolution
- More nuanced pragmatism

## Experimental Timeline

1. ✅ **Sessions #1 & #2**: Qwen 0.5B baseline (30 turns total)
2. ⏳ **Download**: Qwen 7B base model (15GB)
3. ⏳ **Session #3**: Qwen 7B base model test
4. ⏳ **Fine-tune**: Qwen 7B with 115 examples
5. ⏳ **Session #4**: Qwen 7B fine-tuned test
6. ⏳ **Analysis**: Compare all 4 conditions

## Current Status

### Completed
- ✅ 0.5B baseline data (Sessions #1 & #2)
- ✅ 115-example dataset located
- ✅ Fine-tuning script prepared
- ✅ Session #3 script created
- ✅ Analysis framework ready

### In Progress
- ⏳ Downloading Qwen 7B (7% complete, 15GB)

### Pending
- Test 7B base model
- Fine-tune 7B with epistemic pragmatism
- Test 7B fine-tuned model
- Comparative analysis

## Files

**Training**:
- `sage/irp/fine_tune_qwen_7b.py` - Fine-tuning script
- `claude_personal_dataset_dpo.json` - 115 training examples

**Testing**:
- `sage/irp/sage_session_3_7b.py` - 7B base model test
- `sage/irp/analyze_model_size_impact.py` - Comparative analysis

**Models**:
- `model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism/` ✓
- `model-zoo/sage/qwen2.5-7b-instruct/` ⏳
- `model-zoo/sage/epistemic-stances/qwen2.5-7b/epistemic-pragmatism/` (pending)

## Key Question

**Does conversational learning require plasticity (ease of adaptation), or capability (knowledge/reasoning)?**

This experiment will tell us whether SAGE should prefer:
- **Small, nimble models** (fast, flexible, cheap)
- **Large, capable models** (knowledgeable, but possibly rigid and expensive)

Or perhaps: **Both** - small for rapid adaptation, large for depth when needed?
