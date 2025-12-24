# OCR + Orchestration Synergy Experiment

**Date**: October 22, 2025
**Hypothesis**: OCR training + epistemic orchestration produces synergistic benefits beyond either approach alone

---

## Research Question

**Does architectural orchestration work better on models trained with geometric constraints (OCR) than on standard models?**

If OCR creates better latent geometry, then:
- Ensemble variance should be more meaningful
- Uncertainty signals should be cleaner
- Adaptive framing should be more effective

---

## Experimental Design

### Four Conditions

```python
1. Baseline: Standard model, standard inference (1× forward pass)
2. Standard + Orch: Standard model with orchestration (3× forward passes)
3. OCR: OCR-trained model, standard inference (1× forward pass)
4. OCR + Orch: OCR-trained model with orchestration (3× forward passes)
```

### Dataset

**Phi-1.5 on epistemic prompts** (what we already have):
- 135 diverse prompts across 9 categories
- Covers: epistemology, self-referential, scientific, ethical, abstract, practical, debates, uncertainty, meta-cognitive

**Alternative**: Small GLUE task (SST-2 or MRPC) for direct comparison to OCR paper

### Metrics

#### Performance
- **Accuracy** (classification tasks)
- **Perplexity** (language tasks)
- **Response quality** (epistemic stance markers)

#### Calibration
- **Brier score** (↓ better)
- **Expected Calibration Error (ECE)** (↓ better)
- **Uncertainty-accuracy correlation** (↑ better)

#### Geometry
- **Cluster compactness** (↓ better: tight clusters)
- **Centroid separation** (↑ better: well-separated classes)
- **Variance meaningfulness** (correlation between variance and error)

#### Orchestration Quality
- **Adaptive strategy appropriateness**
  - High uncertainty → should ask questions
  - Low uncertainty → should express confidence
- **Calibration curve shape** (closer to diagonal = better)
- **ECE by uncertainty bin** (is high uncertainty calibrated correctly?)

---

## Implementation Plan

### Phase 1: OCR Training (2-3 hours)

**Train Phi-1.5 with OCR losses**:

```python
# Model: microsoft/phi-1_5 (1.3B params)
# Task: Epistemic QA (supervised on subset)

# Standard training:
loss = cross_entropy(logits, labels)

# OCR training:
loss = cross_entropy(logits, labels)
     + λ_stab * stability_loss(cls, noise_std=1e-3)
     + λ_center * center_loss(cls, centers[labels])
     + λ_sep * separation_loss(centers)
     + λ_brier * brier_score(probs, labels)

# Hyperparameters from notebook:
λ_stab = 0.2
λ_center = 0.1
λ_sep = 0.05
λ_brier = 0.1
```

**Training details**:
- Start from pretrained Phi-1.5
- Fine-tune on subset of epistemic examples (50-100 examples)
- 5-10 epochs
- Save checkpoints every 2 epochs

**Output**:
- `models/phi15_ocr_trained/`
- Training logs showing OCR loss components

### Phase 2: Generate Responses (1 hour)

**For each of 4 conditions, generate responses to all 135 prompts**:

```python
# Condition 1: Baseline (standard)
model = load_model("microsoft/phi-1_5")
responses_baseline = [generate(model, prompt) for prompt in prompts]

# Condition 2: Baseline + Orchestration
orchestrator = EpistemicOrchestrator(model)
responses_baseline_orch = [orchestrator.orchestrate(prompt) for prompt in prompts]

# Condition 3: OCR-trained
model_ocr = load_model("models/phi15_ocr_trained")
responses_ocr = [generate(model_ocr, prompt) for prompt in prompts]

# Condition 4: OCR + Orchestration (HYBRID)
orchestrator_ocr = EpistemicOrchestrator(model_ocr)
responses_ocr_orch = [orchestrator_ocr.orchestrate(prompt) for prompt in prompts]
```

**Enhancements to orchestrator**:
- Log variance per prompt
- Log candidate responses (for analysis)
- Track framing strategy chosen

### Phase 3: Analysis (1 hour)

**Compute all metrics for each condition**:

```python
for condition in [baseline, baseline_orch, ocr, ocr_orch]:
    metrics = {
        # Performance
        'stance_markers': count_epistemic_markers(responses),
        'perplexity': compute_perplexity(responses),

        # Calibration (requires ground truth labels)
        'brier': brier_score(probs, labels),
        'ece': expected_calibration_error(probs, labels),

        # Orchestration-specific (only for orch conditions)
        'avg_uncertainty': mean(uncertainties),
        'uncertainty_accuracy_corr': correlation(uncertainties, errors),
        'strategy_distribution': count_strategies(responses),

        # Geometry (requires latent representations)
        'cluster_compactness': intra_class_variance(latents, labels),
        'centroid_separation': inter_class_distance(centroids),
    }
```

**Visualizations**:
1. **Calibration curves** (predicted prob vs actual accuracy, per condition)
2. **Uncertainty distributions** (histogram, per condition)
3. **Variance vs Error scatter** (does high variance → high error?)
4. **Strategy appropriateness** (confusion matrix: uncertainty bin × strategy)

### Phase 4: Hypothesis Testing

**Primary hypothesis**:
```
H1: OCR + Orch > OCR alone
    (orchestration provides additional benefit on OCR model)

H2: OCR + Orch > Baseline + Orch
    (OCR geometry makes orchestration more effective)

H3: (OCR + Orch) - OCR > (Baseline + Orch) - Baseline
    (synergy: orchestration improves OCR more than baseline)
```

**Statistical tests**:
- Paired t-tests (same prompts across conditions)
- Effect sizes (Cohen's d)
- Bootstrap confidence intervals

---

## Expected Results

### If Synergy Exists (Hypothesis Confirmed)

**Performance rankings**:
```
OCR + Orch > OCR > Baseline + Orch > Baseline
             ↑                        ↑
         synergy                  our existing
```

**Calibration improvement**:
- OCR alone: better Brier/ECE than baseline
- OCR + Orch: **superlinear improvement** (not just additive)

**Variance meaningfulness**:
- Baseline: weak correlation (r ≈ 0.3-0.4) between variance and error
- OCR: stronger correlation (r ≈ 0.5-0.6)
- OCR + Orch: **strongest** (r ≈ 0.7-0.8) - cleaner uncertainty signals

**Strategy appropriateness**:
- OCR + Orch chooses strategies more accurately based on actual difficulty
- High variance on OCR model → genuinely hard prompts → questions appropriate
- Low variance on OCR model → genuinely easy prompts → confidence appropriate

### If No Synergy (Hypothesis Rejected)

**Performance rankings**:
```
OCR + Orch ≈ Baseline + Orch
OCR ≈ Baseline

(orchestration effect independent of base model geometry)
```

**Interpretation**:
- Orchestration works uniformly well regardless of base quality
- OCR and orchestration are **redundant** (both solving same problem)
- Either use OCR (train once) or orchestration (inference always), not both

**Still valuable**:
- We'd learn that orchestration is robust to base model quality
- Confirms our approach works on any pretrained model

---

## Code Structure

### New Files to Create

```
sage/experiments/phase1-hierarchical-cognitive/
├── ocr_training/
│   ├── train_phi15_ocr.py          # Train with OCR losses
│   ├── ocr_losses.py                # Stability, center, separation, Brier
│   └── config.py                    # Hyperparameters
├── ocr_orchestration_experiment/
│   ├── run_four_conditions.py       # Generate all responses
│   ├── analyze_synergy.py           # Compute metrics, test hypotheses
│   ├── visualize_results.py         # Calibration curves, scatter plots
│   └── compare_orchestration.py     # OCR vs baseline orchestration quality
└── OCR_RESULTS.md                   # Findings documentation
```

### Key Components

**1. OCR Losses Module** (`ocr_losses.py`):
```python
class OCRLosses:
    def __init__(self, num_labels, hidden_dim, config):
        self.centers = torch.zeros(num_labels, hidden_dim)
        self.config = config

    def stability_loss(self, cls_rep, logits_fn):
        """Penalize sensitivity to CLS noise"""
        noise = torch.randn_like(cls_rep) * self.config.noise_std
        logits_noisy = logits_fn(cls_rep + noise)
        return ((logits_noisy - logits_fn(cls_rep))**2).mean()

    def center_loss(self, cls_rep, labels):
        """Pull features toward class centroids"""
        centers_batch = self.centers[labels]
        return ((cls_rep - centers_batch)**2).mean()

    def separation_loss(self):
        """Push class centroids apart"""
        dist = torch.cdist(self.centers, self.centers, p=2)
        I = torch.eye(self.centers.shape[0], device=self.centers.device)
        # Penalize small inter-center distances
        return torch.exp(-dist + I).sum() - torch.exp(torch.tensor(0.0))*self.centers.shape[0]

    def update_centers(self, cls_rep, labels, momentum=0.97):
        """EMA update of class centroids"""
        with torch.no_grad():
            for i in range(self.centers.shape[0]):
                mask = (labels == i)
                if mask.any():
                    batch_mean = cls_rep[mask].mean(dim=0)
                    self.centers[i] = momentum * self.centers[i] + (1-momentum) * batch_mean
```

**2. Enhanced Orchestrator** (`enhanced_orchestrator.py`):
```python
class EnhancedOrchestrator(EpistemicOrchestrator):
    """Orchestrator with detailed logging for analysis"""

    def orchestrate(self, prompt, n_samples=3):
        result = super().orchestrate(prompt, n_samples)

        # Add detailed metrics for analysis
        result['candidates_logprobs'] = self._compute_logprobs(result['candidates'])
        result['variance_per_token'] = self._token_variance(result['candidates'])
        result['latent_representations'] = self._extract_latents(prompt)

        return result

    def _extract_latents(self, prompt):
        """Extract CLS representation for geometry analysis"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.model(**inputs, output_hidden_states=True)
            # Get CLS token from last layer
            cls_rep = outputs.hidden_states[-1][:, 0, :]  # [1, hidden_dim]
        return cls_rep.cpu()
```

**3. Synergy Analysis** (`analyze_synergy.py`):
```python
def test_synergy(results_dict):
    """
    results_dict = {
        'baseline': [...],
        'baseline_orch': [...],
        'ocr': [...],
        'ocr_orch': [...]
    }
    """

    # H1: OCR + Orch > OCR alone
    improvement_ocr = metric(results_dict['ocr_orch']) - metric(results_dict['ocr'])

    # H2: OCR + Orch > Baseline + Orch
    better_base = metric(results_dict['ocr_orch']) - metric(results_dict['baseline_orch'])

    # H3: Synergy (interaction effect)
    improvement_baseline = metric(results_dict['baseline_orch']) - metric(results_dict['baseline'])
    synergy = improvement_ocr - improvement_baseline

    # Statistical testing
    t_stat, p_value = paired_ttest(...)
    effect_size = cohens_d(...)

    return {
        'ocr_improvement': improvement_ocr,
        'better_base_effect': better_base,
        'synergy': synergy,
        'p_value': p_value,
        'effect_size': effect_size
    }
```

---

## Timeline

**Day 1** (4 hours):
- Implement OCR losses module
- Set up training script
- Train Phi-1.5 with OCR losses (2-3 hours)

**Day 2** (3 hours):
- Generate responses for all 4 conditions (1 hour)
- Implement analysis metrics (1 hour)
- Run analysis pipeline (1 hour)

**Day 3** (2 hours):
- Create visualizations
- Statistical testing
- Document findings

**Total: ~9 hours** (including compute time)

---

## Success Criteria

### Strong Evidence for Synergy

1. **Performance**: OCR + Orch beats OCR alone by **>5%**
2. **Calibration**: ECE improvement is **superlinear** (interaction effect)
3. **Variance**: Correlation r(variance, error) **>0.6** for OCR + Orch
4. **Strategy**: Appropriate strategy choice **>80%** (vs <70% for baseline)
5. **Statistical**: p < 0.01, effect size d > 0.5

### Weak Evidence or No Synergy

1. **Performance**: OCR + Orch ≈ Baseline + Orch (< 2% difference)
2. **Calibration**: Improvements are **additive** (no interaction)
3. **Variance**: Similar r for both orchestrated conditions
4. **Strategy**: Comparable strategy appropriateness
5. **Statistical**: p > 0.05 or effect size d < 0.2

---

## Implications

### If Synergy Confirmed

**For SAGE**:
1. ✅ Train modules with OCR principles (L-level quality)
2. ✅ Orchestrate with our framework (H-level reasoning)
3. ✅ Expect compounding benefits

**Documentation**:
- "OCR + orchestration is best practice"
- Update SAGE training guidelines
- Add OCR losses to module training scripts

**Next research**:
- Test on other modalities (vision, cross-modal)
- Optimize OCR hyperparameters for orchestration
- Investigate other geometric training methods

### If No Synergy

**For SAGE**:
1. ✅ Orchestration works regardless of base model
2. ? OCR training optional (cost-benefit analysis)
3. ? Choose OCR (1× inference) OR orchestration (3× inference), not both

**Documentation**:
- "Orchestration is robust to base quality"
- "OCR and orchestration are redundant"
- Use whichever fits deployment constraints

**Next research**:
- Why are they redundant? (theoretical understanding)
- Are there other training methods that DO synergize?
- Can we reduce orchestration to 2× instead of 3× forward passes?

---

## Risk Mitigation

**Risk 1: OCR training fails / doesn't converge**
- Mitigation: Start with their exact hyperparameters on GLUE task first
- Fallback: Use their pretrained OCR model if available

**Risk 2: Can't get ground truth labels for calibration metrics**
- Mitigation: Focus on geometry metrics (compactness, separation)
- Alternative: Use GLUE classification task (has labels)

**Risk 3: Compute constraints (orchestration expensive on 135 prompts)**
- Mitigation: Subsample to 50 prompts if needed
- Alternative: Run on smaller model (BERT-small ≈29M params)

**Risk 4: Results inconclusive (neither strong synergy nor clear independence)**
- Mitigation: Run multiple random seeds
- Alternative: Increase sample size (more prompts)

---

## Connection to Broader Research

**This experiment bridges**:
- Geometric training methods (OCR, Lipschitz constraints)
- Ensemble uncertainty estimation (our orchestration)
- Calibration research (Brier, ECE, temperature scaling)
- Meta-learning (learning to learn with better geometry)

**If confirmed, we contribute**:
- Empirical evidence that geometry + orchestration compound
- Design pattern: train with constraints, infer with ensembles
- Universal pattern: compression-trust at training AND inference
- Practical guidance: when to use each approach

**Publications**:
- "Synergistic Epistemic Reasoning: Geometric Training + Architectural Orchestration"
- Comparison to related work (OCR, Deep Ensembles, MC Dropout, etc.)

---

## Next Steps

1. ✅ Analysis complete (this document)
2. ⏳ Implement OCR losses module
3. ⏳ Train Phi-1.5 with OCR
4. ⏳ Run 4-condition experiment
5. ⏳ Analyze results & test hypotheses
6. ⏳ Document findings

**Ready to begin implementation.**

---

**Status**: Experiment designed, ready to implement and run.
