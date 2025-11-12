# ARC Prize 2025 Submission Guide

*Created: September 3, 2025*
*For HRM submission to Kaggle competition*

## Competition Overview

- **Platform**: Kaggle
- **Timeline**: March 26 - November 3, 2025
- **Winners Announced**: December 5, 2025
- **Grand Prize**: $700K (for >85% accuracy at <$2.50/task)
- **Progress Prizes**: $125K guaranteed

## Submission Requirements

### 1. Platform Setup
- **Create Kaggle Account**: Required for submission
- **Join Competition**: https://www.kaggle.com/competitions/arc-prize-2025
- **Format**: Must submit as Kaggle Notebook (not script)

### 2. Technical Constraints
- **No Internet Access**: Solution must run completely offline
- **Hardware**: Kaggle L4x4 GPUs (96GB GPU memory total)
- **Compute Budget**: ~$50 worth of compute per submission
- **Efficiency Requirement**: Must achieve <$2.50/task for Grand Prize

### 3. Output Format
- Must provide pixel-perfect predictions for test tasks
- Color and position must match exactly
- Output format specified in Kaggle evaluation instructions

## Submission Process

### Step 1: Prepare Model
```python
# Model must be self-contained
# No external API calls allowed
# Must include all weights and code
```

### Step 2: Create Kaggle Notebook
1. Go to competition page
2. Click "New Notebook"
3. Upload model weights as dataset
4. Implement inference code

### Step 3: Format Output
```python
# Example output format (check Kaggle for exact spec)
submission = {
    'task_id': 'predicted_output_grid',
    # ... for all test tasks
}
```

### Step 4: Submit
1. Run notebook to completion
2. Submit output
3. Check public leaderboard score

## Open Source Requirements

### CRITICAL: Must Open Source Before Final Score
- **New for 2025**: Must share solution BEFORE seeing final score
- **License**: Permissive public domain (CC0 or MIT-0)
- **Timing**: Required before private evaluation results

### What to Open Source
- All code authored by submitter
- Model architecture details
- Training procedures
- Inference code

## Evaluation Process

### During Competition
- **Public Leaderboard**: Based on semi-private dataset
- **Visible to all**: Updated with each submission

### Final Evaluation
- **Private Dataset**: Final ranking determined by this
- **Revealed After**: Open sourcing your solution
- **Determines**: Prize eligibility

## Paper Award (Optional)

### Additional Submission
- Submit paper within 48 hours of competition end
- Format: PDF, arXiv, txt, etc.
- Describe conceptual approach
- Link to Kaggle submission

## For HRM Submission

### Our Advantages
- ✅ **Efficiency**: 6.95M params easily meets compute budget
- ✅ **Open Source Ready**: Already AGPLv3
- ✅ **Architecture Documented**: H↔L bidirectional clear
- ✅ **Baseline Established**: 20% on AGI-2

### Preparation Needed
1. **Convert to Kaggle Notebook**:
   - Package model weights
   - Include evaluation script
   - Format output correctly

2. **Test Offline**:
   - Ensure no internet dependencies
   - Verify runs within time limit
   - Check memory usage

3. **Optimize Inference**:
   - Batch processing if possible
   - Use efficient data loading
   - Monitor GPU utilization

### Example Notebook Structure
```python
# 1. Load Model
model = load_hrm_checkpoint('hrm_arc_best.pt')

# 2. Load Test Data
test_tasks = load_arc_agi2_test()

# 3. Run Inference
predictions = {}
for task_id, task_data in test_tasks.items():
    pred = model.predict(task_data)
    predictions[task_id] = pred

# 4. Format Submission
submission = format_for_kaggle(predictions)
submission.to_csv('submission.csv', index=False)
```

## Important Dates

- **Now - Nov 3**: Submit and iterate
- **Nov 3**: Final submission deadline
- **Nov 3-5**: Open source solution
- **Dec 5**: Winners announced

## Resources

### Official Links
- Competition: https://www.kaggle.com/competitions/arc-prize-2025
- Guide: https://arcprize.org/guide
- Leaderboard: https://arcprize.org/leaderboard

### Example Notebooks
- "ARC-AGI 2025: EDA and Dummy Submission"
- "ARC-AGI 2025: Starter notebook + EDA"
- Previous winner notebooks from 2024

### Our Files
- Model: `validation_package/hrm_arc_best.pt`
- Evaluation: `evaluate_arc_agi2_correct.py`
- Results: `arc_agi2_50tasks_results.json`

## Strategy for HRM

### Phase 1: Baseline Submission (Current)
- Submit 20% model as-is
- Establish position on leaderboard
- Learn submission process

### Phase 2: AGI-2 Training (This Week)
- Train on 1000 AGI-2 tasks
- Target: 40-50% accuracy
- Submit improved model

### Phase 3: Scaling (This Month)
- Scale to 20-30M parameters
- Add ensemble methods
- Target: 70%+ accuracy

### Phase 4: Optimization (October)
- Fine-tune for efficiency
- Implement test-time adaptation
- Push for 85% threshold

## Notes

- Competition is highly competitive (OpenAI o3 at 87.5%)
- Our efficiency advantage is key differentiator
- Open source requirement aligns with AGPLv3
- Paper award opportunity for documenting H↔L innovation

---

*Ready to submit HRM to the ARC Prize 2025 competition!*