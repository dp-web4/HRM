# ARC-AGI-2 Evaluation Notes for HRM

*Date: September 3, 2025*

## Summary

We successfully obtained the ARC-AGI-2 dataset and prepared to evaluate Nova's trained HRM model on it. However, there's an architecture mismatch that needs to be resolved.

## Key Findings

### 1. Model Performance
- **Nova's HRM Model**: Achieved 71.36% validation accuracy on the original ARC dataset
- **Model Size**: 6.95M parameters (much smaller than originally thought 27M)
- **Training**: Reached best performance at step 7,000, plateaued despite 125k+ steps of training
- **Architecture**: Hierarchical dual-loop (H-level strategic, L-level tactical)

### 2. ARC-AGI-2 Dataset
- **Source**: https://github.com/arcprize/ARC-AGI-2
- **Training Set**: 1,000 tasks (vs 400 in ARC-AGI-1)
- **Evaluation Set**: 120 public tasks
- **Test Format**: JSON files with train/test input-output pairs
- **Grid Size**: Variable, typically up to 30x30

### 3. Technical Challenges

#### Architecture Mismatch
The saved model checkpoint has different layer names than expected:
- Model has: `h_to_l`, `l_to_h`, `output`, `h_norm`, `l_norm`
- Expected: `output_layer`, standard transformer layers
- This suggests Nova used a custom HRM implementation with cross-level connections

#### Data Format Differences
- Nova's training used preprocessed numpy arrays (all__inputs.npy, all__labels.npy)
- ARC-AGI-2 provides raw JSON with nested grids
- Need conversion from JSON grids to flattened sequences

## Evaluation Strategy

### Option 1: Use Nova's Validation Script
- Locate Nova's actual model architecture from training scripts
- Load model with correct architecture
- Convert ARC-AGI-2 JSON to expected numpy format
- Run validation

### Option 2: Request Training Code
- Ask Nova for the exact model definition used in training
- This would ensure perfect compatibility with the checkpoint

### Option 3: Inference from Checkpoint
- Analyze the checkpoint keys to reverse-engineer the architecture
- The presence of `h_to_l` and `l_to_h` suggests bidirectional communication between H and L levels
- Hidden size appears to be 512 for intermediate layers (not 256)

## Architecture Insights

Based on the checkpoint structure:
```python
# Apparent architecture elements:
- H-level layers (strategic reasoning)
- L-level layers (tactical execution)
- h_to_l: H→L communication (512 hidden dim)
- l_to_h: L→H feedback
- h_norm, l_norm: Layer normalization
- halt_predictor: Adaptive computation (512 input dim)
- output: Final projection to vocab_size
```

This is more sophisticated than a simple stacked transformer - it has explicit bidirectional communication between reasoning levels.

## Next Steps

1. **Find Nova's training script** to get exact model architecture
2. **Convert ARC-AGI-2 data** to the expected format
3. **Run evaluation** on the 120 public evaluation tasks
4. **Compare with state-of-the-art**: 
   - 2024 competition winner: 55.5% on private set
   - Nova's 71% on training set is promising but needs validation on ARC-AGI-2

## Implications for AGI Prize

The ARC Prize requires 85% accuracy. Nova's model at 71% with only 7M parameters suggests:
- Architecture matters more than scale
- The dual-loop H/L design is effective
- There's room for improvement through:
  - Scaling to 15-20M parameters
  - Better training strategies to break plateaus
  - Ensemble methods
  - Test-time adaptation

## License Considerations

Under the new AGPLv3 license:
- The trained model weights are derivative works
- Any ARC Prize submission using this model must provide source access
- This protects the innovation while allowing open research

---

*Note: To proceed with actual evaluation, we need Nova's exact model architecture definition from the training scripts.*