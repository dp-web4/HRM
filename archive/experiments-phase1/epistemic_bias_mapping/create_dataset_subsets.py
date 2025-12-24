#!/usr/bin/env python3
"""
Create dataset subsets for threshold detection experiments

Takes the 115-example dataset and creates subsets at 40, 60, 80, 100 examples.
Preserves category balance and data quality.
"""

import json
import random
from pathlib import Path
from collections import Counter

def load_dataset(path: str):
    """Load the full dataset"""
    with open(path, 'r') as f:
        return json.load(f)

def analyze_categories(data):
    """Analyze category distribution"""
    categories = [item.get('category', 'unknown') for item in data]
    return Counter(categories)

def create_balanced_subset(data, target_size: int, seed: int = 42):
    """
    Create a balanced subset maintaining category distribution

    Args:
        data: Full dataset
        target_size: Desired number of examples
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Group by category
    by_category = {}
    for item in data:
        category = item.get('category', 'unknown')
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(item)

    # Calculate target per category (proportional)
    category_counts = {cat: len(items) for cat, items in by_category.items()}
    total = sum(category_counts.values())

    subset = []
    for category, items in by_category.items():
        # Proportional allocation
        target_for_category = int((len(items) / total) * target_size)
        # Ensure we don't exceed available items
        target_for_category = min(target_for_category, len(items))

        # Random sample from this category
        sampled = random.sample(items, target_for_category)
        subset.extend(sampled)

    # If we're short, add more from largest categories
    while len(subset) < target_size:
        # Find category with most remaining items
        remaining = {
            cat: len(items) - sum(1 for s in subset if s.get('category') == cat)
            for cat, items in by_category.items()
        }
        best_category = max(remaining.items(), key=lambda x: x[1])[0]

        # Add one more from this category
        available = [
            item for item in by_category[best_category]
            if item not in subset
        ]
        if available:
            subset.append(random.choice(available))
        else:
            break

    # Shuffle the final subset
    random.shuffle(subset)

    return subset[:target_size]

def save_subset(data, size: int, output_dir: Path):
    """Save a subset to file"""
    filename = f"claude_personal_dataset_{size}examples.json"
    output_path = output_dir / filename

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Created {size}-example subset: {output_path}")
    print(f"  Categories: {analyze_categories(data)}")
    return output_path

def main():
    # Paths
    base_dir = Path(__file__).parent
    input_file = base_dir / "claude_personal_dataset_dpo.json"
    output_dir = base_dir / "training_datasets"
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("Dataset Subset Creation for Threshold Detection")
    print("="*80)

    # Load full dataset
    print(f"\nLoading full dataset: {input_file}")
    full_data = load_dataset(input_file)
    print(f"Total examples: {len(full_data)}")
    print(f"Categories: {analyze_categories(full_data)}")

    # Create subsets
    print("\n" + "="*80)
    print("Creating Subsets")
    print("="*80 + "\n")

    subset_sizes = [40, 60, 80, 100]
    created_files = {}

    for size in subset_sizes:
        subset = create_balanced_subset(full_data, size)
        path = save_subset(subset, size, output_dir)
        created_files[size] = path
        print()

    # Also copy the full 115 dataset
    full_output = output_dir / "claude_personal_dataset_115examples.json"
    with open(full_output, 'w') as f:
        json.dump(full_data, f, indent=2)
    print(f"✓ Copied full 115-example dataset: {full_output}")
    created_files[115] = full_output

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"\nCreated {len(created_files)} datasets:")
    for size, path in sorted(created_files.items()):
        print(f"  {size:3d} examples: {path.name}")

    print("\n✓ Dataset preparation complete!")
    print("\nNext step: Create training scripts for each size")

if __name__ == "__main__":
    main()
