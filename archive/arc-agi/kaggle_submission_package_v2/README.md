# SAGE-7M - Situation-Aware Governance Engine

ARC Prize 2025 Submission

Evolution of Sapient's HRM with 75% parameter reduction (27M → 7M)

## Model Statistics
- Parameters: 6.95M
- ARC-AGI-1 Accuracy: 49% (non-augmented)
- ARC-AGI-2 Accuracy: 18% (zero-shot)

## Usage
```python
python kaggle_submission.py
```

## Requirements
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## Files
- `kaggle_submission.py`: Main inference script
- `hrm-model/hrm_arc_best.pt`: Trained model checkpoint
- `kaggle_requirements.txt`: Python dependencies
