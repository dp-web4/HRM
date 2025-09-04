# Experimental Findings - September 4, 2025

## Experiments Conducted

### Extended Reasoning Fine-tuning
**Hypothesis**: Extending reasoning cycles from 8 to 20 would improve performance on complex multi-step ARC tasks.

**Method**:
- Started from best checkpoint (7K steps, 49.1% on AGI-1)
- Modified architecture to support 20 reasoning cycles
- Fine-tuned on non-augmented AGI-1 data for 10K steps
- Lower learning rate (5e-5) for fine-tuning

**Results**:

| Checkpoint | Cycles Used | AGI-1 Accuracy |
|------------|-------------|----------------|
| Original (7K) | 8 | 49.1% |
| Fine-tuned (1K) | 20 | 9.2% |
| Fine-tuned (10K) | 20 | 8.1% |

**Observations**:
- Model successfully used all 20 cycles during inference
- Halt predictor showed gradual probability progression (0.49→0.27)
- Performance degraded significantly from baseline
- Degradation continued with more training (not a transient effect)

### Initial Attempt (Failed)
**Issue**: Enhanced Sequential halt_predictor wasn't properly initialized
- Result: Model stopped after 4 cycles despite 20 being available
- Halt probability immediately maxed at 0.9999
- Performance: 9% accuracy

**Fix Applied**: 
- Kept original Linear halt_predictor
- Added gating mechanism for gradual cycle usage
- Proper weight transfer from original model

## Data Collected

### Training Characteristics
- **Original training**: 500x augmented data, 8 cycles, 10K steps
- **Fine-tuning**: Non-augmented data, 20 cycles, 10K steps
- **Loss progression**: Started ~4.0, remained elevated throughout
- **Cycle usage**: Consistent 20 cycles after fix

### Performance Distribution on AGI-1 (Fine-tuned Model)
```
Perfect   (>95%):   0.2% [  1 task]
Excellent (>80%):   0.2% [  1 task]
Good      (>60%):   0.8% [  3 tasks]
Moderate  (>40%):   1.8% [  7 tasks]
Poor      (>20%):   8.8% [ 35 tasks]
Failed    (<20%):  88.2% [353 tasks]
```

### Computational Cost
- 8 cycles: ~12.5 sec/task average
- 20 cycles: ~26 sec/task average
- 2.5x increase in inference time

## Technical Implementation Details

### Halt Predictor Modifications
```python
# Original
self.halt_predictor = nn.Linear(hidden_size * 2, 1)

# Added for extended reasoning
self.halt_gate = nn.Parameter(torch.tensor(0.1))
self.cycle_embedding = nn.Embedding(max_cycles, hidden_size)

# Gating mechanism
cycle_factor = (cycle + 1) / max_cycles
gated_logit = halt_logit * self.halt_gate * cycle_factor
```

### Loss Function Adjustments
- Halt penalty: 0.0001 (reduced from 0.001)
- Minimum cycles before halt: 5 (increased from 3)
- Target gradual halt probability increase

## Artifacts Generated

### Code Files
- `finetune_reasoning.py` - Initial attempt (broken halt predictor)
- `finetune_reasoning_fixed.py` - Fixed version with proper initialization
- `diagnose_finetuning.py` - Diagnostic tool for understanding failures
- `evaluate_finetuned.py` - Evaluation script supporting both architectures

### Checkpoints Created
- `hrm_reasoning_agi1_fixed_step_1000.pt` through `step_9000.pt`
- `hrm_reasoning_agi1_fixed_final.pt`
- Each checkpoint maintains 20-cycle configuration

### Documentation
- `FINETUNING_ISSUE_REPORT.md` - Technical analysis of initial failure
- `FINETUNING_FIX_REPORT.md` - Documentation of fixes applied
- `EXTENDED_REASONING_RESULTS.md` - Results summary

## Questions Raised

1. **Cycle-Task Complexity Relationship**: Do all tasks benefit equally from more cycles?
2. **Training Regime**: Is the shift from augmented to non-augmented data more impactful than cycle count?
3. **Architecture Capacity**: Is 6.95M parameters sufficient for 20-cycle reasoning?
4. **Gradual vs Sudden Changes**: Would incremental increases (8→10→12) work better?
5. **Task-Adaptive Cycles**: Could variable cycle counts based on task complexity help?

## Notable Observations

### What Worked
- Halt predictor fix successfully enabled 20-cycle reasoning
- Model training completed without technical issues
- Diagnostic tools effectively identified problems

### Unexpected Findings
- More cycles led to worse performance universally (not task-dependent)
- Performance continued degrading with more training
- Original 8-cycle configuration appears well-matched to model capacity

### Neutral Observations
- Training loss remained high throughout (different from original training)
- Cycle count was consistent (always 20, never early stopping)
- Model essentially randomized rather than maintaining partial performance

## Related Context

From recent pull:
- Kaggle submission prepared with original model
- SAGE7M results on ARC-AGI-2 documented
- Submission package created with 7K-step checkpoint

## Next Experimental Directions to Consider

Without drawing conclusions, potential areas for investigation:
1. Intermediate cycle counts (10, 12, 15)
2. Mixed training (some augmented, some original)
3. Curriculum learning on cycle usage
4. Task-complexity-based cycle allocation
5. Preserving original weights while adding cycle capacity

---

*Documentation Date: September 4, 2025*
*Purpose: Record experimental findings without premature conclusions*
*Status: Data collection phase - analysis ongoing*