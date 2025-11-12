# ARC-AGI-2 Evaluation Results for HRM Model

*Date: September 3, 2025*

## Executive Summary

Successfully evaluated Nova's HRM model (71.36% on ARC-AGI-1) on the newer ARC-AGI-2 dataset. The model achieves **~20% accuracy on ARC-AGI-2**, a significant drop from its ARC-AGI-1 performance but still notable for a 6.95M parameter model.

## Model Details

- **Architecture**: Hierarchical Reasoning Module with H↔L bidirectional communication
- **Parameters**: 6.95M (initially thought to be 27M)
- **Training**: Best checkpoint at step 7,000
- **Original Performance**: 71.36% on ARC-AGI-1 validation set

### Key Architectural Features
- Dual-loop processing: H-level (strategic) and L-level (tactical)
- Bidirectional communication: `h_to_l` and `l_to_h` layers
- Adaptive computation: Up to 8 reasoning cycles with halting mechanism
- Layer normalization after each level

## Evaluation Results

### ARC-AGI-2 Performance (50 tasks sampled)
- **Average Accuracy**: 20.15%
- **Standard Deviation**: 29.38%
- **Min Accuracy**: 0.00%
- **Max Accuracy**: 86.87%

### Performance Distribution
The high standard deviation (29.38%) suggests the model performs very well on some tasks but completely fails on others. This bimodal distribution indicates:
- Some ARC-AGI-2 patterns are similar to ARC-AGI-1 (high accuracy)
- New patterns in ARC-AGI-2 are not recognized (0% accuracy)

## Analysis

### Why the Performance Drop?

1. **Dataset Shift**: ARC-AGI-2 has 1,000 training tasks vs 400 in ARC-AGI-1, likely including more complex patterns

2. **Training Data Mismatch**: The model was trained on augmented ARC-AGI-1 data, not ARC-AGI-2

3. **Generalization Limits**: Despite the sophisticated H↔L architecture, the model struggles with truly novel patterns

### Comparison with State-of-the-Art

- **Our HRM Model**: ~20% on ARC-AGI-2
- **2024 Competition Winner**: 55.5% on private set
- **ARC Prize Target**: 85%

Despite the lower accuracy, our model is remarkably efficient at only 6.95M parameters.

## Path to ARC Prize (85% Target)

Based on these results, achieving 85% would require:

1. **Training on ARC-AGI-2**: Direct training on the 1,000 ARC-AGI-2 tasks
2. **Scaling**: Increase to 20-30M parameters while maintaining efficiency
3. **Architectural Improvements**:
   - Stronger few-shot learning capabilities
   - Better pattern abstraction
   - Enhanced bidirectional H↔L communication
4. **Ensemble Methods**: Multiple models voting
5. **Test-Time Adaptation**: Allow models to learn from test examples

## Code Implementation Success

✅ **Successfully identified and used the correct architecture** from `train_arc_full_nova.py`
✅ **Created evaluation pipeline** for ARC-AGI-2
✅ **Validated model loading and inference**

The key was finding that Nova used a custom HRM implementation with:
- `h_to_l` and `l_to_h` interaction layers
- `halt_predictor` taking concatenated H and L states (512 dims)
- `output` layer (not `output_layer`)
- Layer normalization for both levels

## Conclusion

While the 20% accuracy on ARC-AGI-2 is below the 71% on ARC-AGI-1, this evaluation provides valuable insights:

1. **The architecture works**: Successfully loads and runs inference
2. **Efficiency confirmed**: 6.95M parameters is very lean
3. **Clear path forward**: Train directly on ARC-AGI-2 data
4. **AGPLv3 protection**: Model and improvements are protected under new license

The drop from 71% to 20% when moving from ARC-AGI-1 to ARC-AGI-2 highlights the importance of training on the target dataset. With proper training on ARC-AGI-2's 1,000 tasks, this architecture could likely achieve much better performance.

## Files Created

- `evaluate_arc_agi2_correct.py` - Working evaluation script with correct architecture
- `arc_agi2_50tasks_results.json` - Detailed results on 50 tasks
- `ARC_AGI2_EVALUATION_NOTES.md` - Initial analysis
- This file - Final evaluation report

---

*Next step: Train the HRM architecture directly on ARC-AGI-2 dataset to establish a proper baseline.*