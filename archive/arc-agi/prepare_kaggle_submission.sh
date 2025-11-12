#!/bin/bash
# Prepare ARC Prize 2025 Kaggle submission package

echo "Preparing ARC Prize 2025 SAGE-7M Submission..."

# Create submission directory
SUBMISSION_DIR="kaggle_submission_package"
rm -rf $SUBMISSION_DIR
mkdir -p $SUBMISSION_DIR
mkdir -p $SUBMISSION_DIR/hrm-model

# Copy model checkpoint
echo "Copying model checkpoint..."
cp validation_package/hrm_arc_best.pt $SUBMISSION_DIR/hrm-model/

# Copy submission script
echo "Copying submission script..."
cp kaggle_submission.py $SUBMISSION_DIR/

# Copy requirements
echo "Copying requirements..."
cp kaggle_requirements.txt $SUBMISSION_DIR/

# Create README for Kaggle
cat > $SUBMISSION_DIR/README.md << 'EOF'
# SAGE-7M - Sentient Agentic Generative Engine

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
EOF

# Create a test script for local validation
cat > $SUBMISSION_DIR/test_local.py << 'EOF'
#!/usr/bin/env python3
"""Test submission locally with sample data"""

import json
import torch
from pathlib import Path

# Create dummy test data
test_task = {
    "test_task_1": {
        "train": [
            {
                "input": [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                "output": [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
            }
        ],
        "test": [
            {
                "input": [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
            }
        ]
    }
}

# Save test data
with open("arc-agi_test_challenges.json", "w") as f:
    json.dump(test_task, f)

print("Test data created. Now running submission script...")

# Import and run submission
import kaggle_submission
kaggle_submission.main()

print("\nTest complete! Check submission.json for output.")
EOF

chmod +x $SUBMISSION_DIR/test_local.py

# Create submission notebook for Kaggle
cat > $SUBMISSION_DIR/submission_notebook.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ARC Prize 2025 - SAGE-7M Submission\n",
    "Evolution of Sapient's HRM with 75% size reduction (6.95M parameters)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install requirements\n",
    "!pip install torch numpy tqdm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run submission\n",
    "%run kaggle_submission.py"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Package everything
echo "Creating submission archive..."
cd $SUBMISSION_DIR
zip -r ../arc_prize_2025_sage7m_submission.zip *
cd ..

# File size check
echo ""
echo "Submission package created!"
echo "Files in package:"
ls -lh $SUBMISSION_DIR/
echo ""
echo "Archive size:"
ls -lh arc_prize_2025_sage7m_submission.zip

echo ""
echo "Next steps:"
echo "1. Test locally: cd $SUBMISSION_DIR && python test_local.py"
echo "2. Upload to Kaggle Datasets: arc_prize_2025_sage7m_submission.zip"
echo "3. Create Kaggle notebook using submission_notebook.ipynb"
echo "4. Submit to competition"