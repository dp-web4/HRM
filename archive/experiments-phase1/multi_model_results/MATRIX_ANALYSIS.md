# Multi-Model Epistemic Stance Training - Matrix Analysis
**Date**: 2025-10-20**Models Tested**: 4**Stances Tested**: 2
---
## Model × Stance Matrix
| Model | Size | Family | Stance | Loss | Time | Shift |
|-------|------|--------|--------|------|------|-------|
| qwen2-0.5b | 0.5B | Qwen | curious-uncertainty | 1.974 | 1.6s | +0 |
| qwen2-0.5b | 0.5B | Qwen | confident-expertise | 1.655 | 1.6s | +2 |
| distilgpt2 | 82M | GPT-2 | curious-uncertainty | 3.303 | 0.3s | +1 |
| distilgpt2 | 82M | GPT-2 | confident-expertise | 3.605 | 0.2s | +1 |
| pythia-160m | 160M | Pythia | curious-uncertainty | 2.110 | 0.4s | +0 |
| pythia-160m | 160M | Pythia | confident-expertise | 2.167 | 0.4s | +3 |

---
## Observations

### Training Loss by Model Family

**qwen2-0.5b** (Qwen, 0.5B):
- curious-uncertainty: Loss 1.974, Time 1.6s
- confident-expertise: Loss 1.655, Time 1.6s

**tinyllama-1.1b** (LLaMA, 1.1B):

**distilgpt2** (GPT-2, 82M):
- curious-uncertainty: Loss 3.303, Time 0.3s
- confident-expertise: Loss 3.605, Time 0.2s

**pythia-160m** (Pythia, 160M):
- curious-uncertainty: Loss 2.110, Time 0.4s
- confident-expertise: Loss 2.167, Time 0.4s

### Stance Shift Patterns

Measures how well each model adopted the trained stance:

**curious-uncertainty**:
- qwen2-0.5b: Uncertainty markers 3 → 3 (shift: +0)
- distilgpt2: Uncertainty markers 0 → 1 (shift: +1)
- pythia-160m: Uncertainty markers 1 → 1 (shift: +0)

**confident-expertise**:
- qwen2-0.5b: Confidence markers 1 → 3 (shift: +2)
- distilgpt2: Confidence markers 0 → 1 (shift: +1)
- pythia-160m: Confidence markers 0 → 3 (shift: +3)

