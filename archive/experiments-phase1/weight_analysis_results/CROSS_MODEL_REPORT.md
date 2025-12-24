# Cross-Model Epistemic Stance Weight Analysis

Analyzing which layers change across different model architectures

---

## Summary by Model Family

### qwen2-0.5b

| Stance | Layers Changed (>5%) | Top Layer Change |
|--------|----------------------|------------------|
| curious-uncertainty | 2 | model.layers.15.self_attn.v_pr (-36.6%) |
| confident-expertise | 2 | model.layers.15.self_attn.v_pr (-36.7%) |
| engaged-difficulty | 1 | model.layers.13.self_attn.q_pr (-69.8%) |

### distilgpt2

| Stance | Layers Changed (>5%) | Top Layer Change |
|--------|----------------------|------------------|
| curious-uncertainty | 0 | None |
| confident-expertise | 0 | None |

### pythia-160m

| Stance | Layers Changed (>5%) | Top Layer Change |
|--------|----------------------|------------------|
| curious-uncertainty | 0 | None |
| confident-expertise | 0 | None |

---

## Layer Changes by Stance

### curious-uncertainty

**qwen2-0.5b** (2 layers):

- model.layers.15.self_attn.v_proj: 15.049 → 9.535 (-36.6%)
- model.layers.6.mlp.up_proj: 12.787 → 14.040 (+9.8%)

**distilgpt2** (0 layers):

- No significant changes

**pythia-160m** (0 layers):

- No significant changes


### confident-expertise

**qwen2-0.5b** (2 layers):

- model.layers.15.self_attn.v_proj: 15.049 → 9.522 (-36.7%)
- model.layers.16.self_attn.v_proj: 8.171 → 8.728 (+6.8%)

**distilgpt2** (0 layers):

- No significant changes

**pythia-160m** (0 layers):

- No significant changes


### engaged-difficulty

**qwen2-0.5b** (1 layers):

- model.layers.13.self_attn.q_proj: 7.208 → 2.176 (-69.8%)


---

## Observations

### Pattern Detection

**Attention layers affected**: 3 model×stance combinations

**MLP layers affected**: 1 model×stance combinations

This suggests stance training primarily modifies **attention mechanisms**.

