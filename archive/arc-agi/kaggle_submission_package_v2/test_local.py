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
