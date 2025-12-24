# Epistemic Validation Suite

This package provides a 20-prompt validation set designed to evaluate **epistemic pragmatism** fine-tuning in small to medium models (e.g., Qwen2.5-0.5B).

## Usage

```bash
python evaluate_model.py --model qwen2.5-0.5b-epiprag-100 --prompts epistemic_suite.json
```

## Metrics to Track

- **Stance divergence** (embedding cosine similarity vs baseline)
- **Boilerplate suppression** (reduction in disclaimers per 100 tokens)
- **Coherence density** (entropy per 50 tokens)
- **Calibration curve** (confidence vs correctness)
- **Meta-awareness emergence** (references to epistemic architecture)
