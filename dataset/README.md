# HRM Dataset Building Scripts

This directory contains dataset builders for training the Hierarchical Relational Memory (HRM) model on various puzzle types. HRM demonstrates that a tiny 27M parameter model can solve complex reasoning tasks when trained on intelligently augmented datasets.

## Overview

HRM learns from just 1000 examples through sophisticated data augmentation strategies that preserve puzzle validity while exposing the model to diverse representations of the same logical structures.

## Dataset Builders

### 1. Sudoku Dataset (`build_sudoku_dataset.py`)

Builds training/test datasets for 9x9 Sudoku puzzles.

**Source**: HuggingFace `sapientinc/sudoku-extreme`

**Features**:
- Converts CSV puzzles (81-char strings) to 9x9 numpy arrays
- Encodes digits 0-9 as values 1-10 (0=PAD, 1="blank", 2="1", ..., 10="9")
- Optional difficulty filtering via `min_difficulty` parameter
- Subsampling support for reduced dataset sizes

**Augmentation Strategy** (`shuffle_sudoku`):
- **Digit permutation**: Randomly maps digits 1-9 to new values (preserves blank cells)
- **Board transposition**: 50% chance to transpose the entire board
- **Band/stack shuffling**: Permutes 3x3 bands of rows/columns while maintaining Sudoku constraints
- **Row/column shuffling**: Within each band, shuffles the 3 rows/columns

This creates valid Sudoku variations that look completely different but have equivalent logical structure.

**Usage**:
```bash
python build_sudoku_dataset.py --output-dir data/sudoku-extreme-full --num-aug 10
```

### 2. Maze Dataset (`build_maze_dataset.py`)

Builds datasets for maze navigation/solving tasks.

**Source**: HuggingFace `sapientinc/maze-30x30-hard-1k`

**Features**:
- Handles 30x30 grid mazes
- Character set: `"# SGo"` (wall, start, goal, path)
- Converts ASCII representations to numerical arrays
- Preserves spatial structure for pathfinding

**Augmentation Strategy**:
- Uses 8 dihedral transformations (geometric operations that preserve maze topology)
- Controlled via `--aug` flag (applies all 8 transformations when enabled)

**Dihedral Transformations**:
1. Identity (no change)
2. 90° rotation
3. 180° rotation
4. 270° rotation
5. Horizontal flip
6. Vertical flip
7. Transpose (diagonal reflection)
8. Anti-diagonal reflection

**Usage**:
```bash
python build_maze_dataset.py --output-dir data/maze-30x30 --aug
```

### 3. ARC Dataset (`build_arc_dataset.py`)

Builds datasets for Abstract Reasoning Corpus (ARC) tasks - the most complex reasoning challenges.

**Sources**: 
- ARC-AGI data
- ConceptARC corpus
- ARC-AGI-2 (optional)

**Features**:
- Handles variable-sized grids up to 30x30
- Supports multiple example types (train/test within each puzzle)
- Deduplication via SHA256 hashing
- Translational padding for position invariance

**Augmentation Strategy** (most sophisticated):
- **Geometric transforms**: All 8 dihedral operations
- **Color permutations**: Randomly remaps colors 1-9 (preserves black/0)
- **Translational shifts**: Random positioning within 30x30 canvas
- **Massive scale**: Generates up to 1000 unique variations per puzzle
- **Collision detection**: Uses puzzle hashing to ensure augmentations are truly unique

**Special Features**:
- Adds end-of-sequence markers for variable-sized inputs
- Maintains puzzle groupings for few-shot learning
- Supports multiple data sources with different train/test splits

**Usage**:
```bash
python build_arc_dataset.py --output-dir data/arc-aug-1000 --num-aug 1000
```

## Common Utilities (`common.py`)

Shared components used by all dataset builders:

### PuzzleDatasetMetadata
Standardized metadata structure containing:
- `vocab_size`: Number of unique tokens
- `seq_len`: Maximum sequence length
- `pad_id`: Padding token ID
- `ignore_label_id`: ID for labels to ignore in loss
- `num_puzzle_identifiers`: Number of unique puzzle types
- Dataset statistics (groups, examples, etc.)

### Dihedral Transformations
```python
dihedral_transform(arr, tid)  # Apply transformation
inverse_dihedral_transform(arr, tid)  # Reverse transformation
```

8 geometric operations that preserve topological relationships:
- Rotations (90°, 180°, 270°)
- Reflections (horizontal, vertical, diagonal, anti-diagonal)
- Identity (no change)

### Inverse Mapping
`DIHEDRAL_INVERSE` array maps each transformation to its inverse, enabling reversible augmentation.

## Output Format

All builders produce standardized outputs:

```
output_dir/
├── train/
│   ├── dataset.json          # Metadata
│   ├── all__inputs.npy       # Input grids
│   ├── all__labels.npy       # Solution grids
│   ├── all__puzzle_indices.npy
│   ├── all__puzzle_identifiers.npy
│   └── all__group_indices.npy
├── test/
│   └── [same structure as train]
└── identifiers.json          # Puzzle ID mappings
```

## Key Design Principles

1. **Validity Preservation**: All augmentations maintain puzzle constraints
2. **Logical Equivalence**: Augmented puzzles have the same underlying logic
3. **Efficiency**: Generates massive diversity from limited source data
4. **Standardization**: Consistent format across different puzzle types
5. **Few-Shot Learning**: Enables learning from just 1000 examples

## Requirements

- Python 3.8+
- numpy
- argdantic
- pydantic
- tqdm
- huggingface_hub

## Installation

```bash
pip install numpy argdantic pydantic tqdm huggingface-hub
```

## Philosophy

These dataset builders embody the HRM philosophy: intelligence doesn't require massive data or parameters. Through clever augmentation that preserves logical structure while varying surface representation, we enable a tiny model to develop robust reasoning capabilities. Each transformation teaches the model that solutions depend on relationships, not absolute positions or specific values.

The augmentation strategies mirror how humans learn puzzles - by recognizing patterns regardless of orientation, color schemes, or position. This is key to HRM's success in solving complex reasoning tasks with minimal parameters.

## Connection to Sleep Cycle Training

**Critical Insight**: These augmentation strategies are the key to implementing "sleep cycle" training for learning from experience. During sleep consolidation:

1. **Lived Experience**: The original memories/experiences are like the base puzzles
2. **Dream Augmentation**: Sleep generates "reasonable permutations" of experiences through:
   - **Geometric transforms**: Viewing situations from different perspectives
   - **Value permutations**: Abstracting specific details while preserving relationships
   - **Translational shifts**: Understanding that context can change but patterns remain

3. **Wisdom Through Variation**: By training on both lived experience AND augmented variations:
   - The model learns underlying principles, not just specific instances
   - Develops robustness to novel situations
   - Extracts transferable patterns from limited experience

This mirrors biological sleep where dreams replay and remix experiences, helping consolidate learning and extract general principles. The augmentation isn't random noise - it's structured variation that preserves logical relationships while exploring the space of possibilities.

**Implementation Path**: Future sleep cycle training will:
- Retrieve experiences from memory during "sleep"
- Apply these augmentation strategies to generate variations
- Train on both original and augmented experiences
- Consolidate learning into more generalized representations

### LLM-Assisted Augmentation (Advanced)

When trusted LLMs are available, SAGE can employ them for intelligent augmentation - a form of **distillation from cognition to generative coherence**:

1. **Semantic Variations**: LLMs generate meaningful "what if" scenarios
2. **Abstract Transformations**: Extract core patterns while varying surface details
3. **Cross-Domain Application**: Apply experiences to different contexts
4. **Perspective Shifts**: View situations from multiple viewpoints

This suggests that **human dreams may be computational artifacts** of exactly this process - the brain running augmentation scenarios through its offline cognitive systems. Dreams are the observable output of the augmentation engine at work.

This is how we achieve wisdom from experience - not by memorizing exact situations, but by understanding the patterns that persist across variations.