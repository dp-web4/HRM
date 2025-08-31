# Post-Training Validation

## Context
HRM training on Legion is ~20 hours in (expected 24-28 hours total). We need a validation plan ready for when training completes.

## Dependencies
- Training completion on Legion
- Access to saved checkpoints
- ARC test dataset available

## Tasks

### 1. Immediate Validation
- [ ] Check final training metrics (loss, accuracy)
- [ ] Load best checkpoint
- [ ] Run inference on held-out ARC tasks
- [ ] Compare to baseline (random: ~5%, GPT-4: ~20%, target: >75%)

### 2. Checkpoint Management
- [ ] Copy best checkpoint to Dropbox via sync system
- [ ] Document model metrics and training time
- [ ] Create backup before any fine-tuning

### 3. Inference Testing
```python
# Quick test script needed:
- Load model
- Run on 10 sample ARC tasks
- Measure accuracy and inference time
- Test both H and L module outputs
```

### 4. Performance Profiling
- [ ] Memory usage during inference
- [ ] Tokens per second
- [ ] GPU utilization patterns
- [ ] Adaptive computation time statistics

## Success Criteria
- Model achieves >70% accuracy on test ARC tasks
- Inference runs without memory issues
- Checkpoints successfully sync to Dropbox
- Clear performance baseline established

## Owner
Collaborative - Human initiates, Claude implements validation scripts

## Next Steps Triggered
- If successful: Move to SAGE integration
- If underperforming: Analyze failure modes, consider fine-tuning
- If technical issues: Debug and document fixes

## Notes
Keep validation lightweight initially - full evaluation can come after confirming basic functionality.