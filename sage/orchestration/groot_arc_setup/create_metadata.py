#!/usr/bin/env python3
"""
Create GR00T metadata for SAGE/ARC-AGI use case.

This creates a minimal embodiment called "sage_arc" for vision reasoning tasks.
"""

import sys
sys.path.insert(0, "/home/dp/ai-workspace/isaac-gr00t")

import json
import numpy as np
from pathlib import Path

from gr00t.data.schema import (
    DatasetMetadata,
    DatasetStatistics,
    DatasetStatisticalValues,
    DatasetModalities,
    VideoMetadata,
    StateActionMetadata,
)
from gr00t.data.embodiment_tags import EmbodimentTag


def create_sage_arc_metadata(output_path: Path):
    """
    Create metadata for SAGE/ARC-AGI embodiment.

    For ARC-AGI:
    - Video: ARC grids rendered as images (30x30 pixels per cell, up to 30x30 grid = 900x900 max)
    - State: Task encoding (simple vector, e.g., task_id or features)
    - Action: Output grid specification (dimensions + values)
    """

    # State: Simple task descriptor
    # For simplicity: [task_difficulty, grid_width, grid_height, num_colors, ...]
    # Let's use 16-dim state vector
    state_dim = 16
    state_stats = DatasetStatisticalValues(
        max=np.ones(state_dim) * 30.0,  # Max grid size is 30
        min=np.zeros(state_dim),
        mean=np.ones(state_dim) * 10.0,
        std=np.ones(state_dim) * 5.0,
        q01=np.ones(state_dim) * 1.0,
        q99=np.ones(state_dim) * 20.0,
    )

    # Action: Output grid encoded as flat vector
    # ARC max grid is 30x30 = 900 cells, each can be 0-9 (10 colors)
    # We'll use a 32-dim action for simplicity (grid dimensions + control signals)
    action_dim = 32
    action_stats = DatasetStatisticalValues(
        max=np.ones(action_dim) * 30.0,
        min=np.zeros(action_dim),
        mean=np.ones(action_dim) * 5.0,
        std=np.ones(action_dim) * 3.0,
        q01=np.zeros(action_dim),
        q99=np.ones(action_dim) * 10.0,
    )

    # Create statistics
    statistics = DatasetStatistics(
        state={"task_encoding": state_stats},
        action={"grid_output": action_stats},
    )

    # Create modalities metadata
    modalities = DatasetModalities(
        video={
            "input_grid": VideoMetadata(
                resolution=(900, 900),  # Max ARC grid rendered at 30px/cell
                channels=3,  # RGB
                fps=1.0,  # Static images, not video
            )
        },
        state={
            "task_encoding": StateActionMetadata(
                absolute=True,
                rotation_type=None,
                shape=(state_dim,),
                continuous=True,
            )
        },
        action={
            "grid_output": StateActionMetadata(
                absolute=True,
                rotation_type=None,
                shape=(action_dim,),
                continuous=True,
            )
        },
    )

    # Create full metadata
    metadata = DatasetMetadata(
        statistics=statistics,
        modalities=modalities,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,  # Use NEW_EMBODIMENT tag
    )

    # Convert to dict for saving
    metadata_dict = {
        "new_embodiment": metadata.model_dump(mode="json")
    }

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)

    print(f"✅ Created SAGE/ARC metadata at: {output_path}")
    print(f"   Embodiment: new_embodiment")
    print(f"   State dim: {state_dim}")
    print(f"   Action dim: {action_dim}")
    print(f"   Video resolution: 900x900")

    return metadata_dict


def main():
    """Create and save metadata."""
    output_dir = Path("/home/dp/ai-workspace/HRM/sage/orchestration/groot_arc_setup")
    metadata_path = output_dir / "metadata.json"

    print("=" * 80)
    print("Creating GR00T Metadata for SAGE/ARC-AGI")
    print("=" * 80)

    metadata = create_sage_arc_metadata(metadata_path)

    # Print sample
    print("\n📦 Metadata structure (sample):")
    print(json.dumps(metadata, indent=2)[:500] + "...")

    print("\n✅ Metadata creation complete!")
    print(f"   Location: {metadata_path}")


if __name__ == "__main__":
    main()
