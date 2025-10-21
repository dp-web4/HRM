# WeightWatcher Analysis: Epistemic Stance Training

**Model**: qwen2-0.5b
**Training**: 5 examples, 2 epochs, ~1.6 seconds

---

## Summary Metrics

### Baseline Model

- **log_norm**: 2.558
- **alpha**: 6.414
- **alpha_weighted**: 3.883
- **log_alpha_norm**: 4.255
- **log_spectral_norm**: 0.749
- **stable_rank**: 84.599
- **mp_softrank**: 0.335

### After Training

#### curious-uncertainty

- **log_norm**: 2.558 (Δ +0.000, +0.0%)
- **alpha**: 6.393 (Δ -0.021, -0.3%)
- **alpha_weighted**: 3.900 (Δ +0.017, +0.4%)
- **log_alpha_norm**: 4.273 (Δ +0.018, +0.4%)
- **log_spectral_norm**: 0.749 (Δ +0.000, +0.0%)
- **stable_rank**: 84.598 (Δ -0.001, -0.0%)
- **mp_softrank**: 0.335 (Δ +0.000, +0.0%)

#### confident-expertise

- **log_norm**: 2.558 (Δ +0.000, +0.0%)
- **alpha**: 6.384 (Δ -0.030, -0.5%)
- **alpha_weighted**: 3.894 (Δ +0.011, +0.3%)
- **log_alpha_norm**: 4.267 (Δ +0.012, +0.3%)
- **log_spectral_norm**: 0.749 (Δ +0.000, +0.0%)
- **stable_rank**: 84.597 (Δ -0.002, -0.0%)
- **mp_softrank**: 0.335 (Δ +0.000, +0.0%)

---

## Significant Layer Changes

### curious-uncertainty

Layers with >5% change: 2

| Layer | Baseline α | Trained α | Δα | Δ% |
|-------|------------|-----------|-------|-------|
| model.layers.15.self_attn.v_pr | 15.049 | 9.535 | -5.514 | -36.6% |
| model.layers.6.mlp.up_proj | 12.787 | 14.040 | +1.254 | +9.8% |

### confident-expertise

Layers with >5% change: 2

| Layer | Baseline α | Trained α | Δα | Δ% |
|-------|------------|-----------|-------|-------|
| model.layers.15.self_attn.v_pr | 15.049 | 9.522 | -5.527 | -36.7% |
| model.layers.16.self_attn.v_pr | 8.171 | 8.728 | +0.557 | +6.8% |

