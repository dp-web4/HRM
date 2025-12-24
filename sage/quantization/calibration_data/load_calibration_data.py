#!/usr/bin/env python3
"""Load calibration data for quantization."""

import json
import torch
from pathlib import Path

def load_calibration_data(calibration_dir="./"):
    """Load calibration dataset for quantization."""
    calibration_dir = Path(calibration_dir)

    with open(calibration_dir / "calibration_dataset.json") as f:
        data = json.load(f)

    # Convert back to tensors
    for sample in data:
        sample["input_ids"] = torch.tensor(sample["input_ids"])
        sample["attention_mask"] = torch.tensor(sample["attention_mask"])

    return data

def get_calibration_loader(calibration_dir="./", batch_size=1):
    """Get data loader for calibration."""
    data = load_calibration_data(calibration_dir)

    # Simple batch iterator
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]

        # Stack batch
        input_ids = torch.cat([s["input_ids"] for s in batch], dim=0)
        attention_mask = torch.cat([s["attention_mask"] for s in batch], dim=0)

        yield {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

if __name__ == "__main__":
    # Test loading
    data = load_calibration_data()
    print(f"Loaded {len(data)} calibration samples")

    # Test loader
    loader = get_calibration_loader(batch_size=4)
    batch = next(loader)
    print(f"Batch shape: {batch['input_ids'].shape}")
