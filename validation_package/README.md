# HRM ARC Validation Package

This package contains everything needed to validate HRM models on the ARC dataset.

## Contents

- `validate_arc.py` - Standalone validation script
- `requirements.txt` - Python dependencies
- `run_validation.sh` - Example validation script
- This README

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download dataset from Dropbox:
```bash
# Download and extract dataset
wget [dropbox-link]/arc-aug-100.tar.gz
tar -xzf arc-aug-100.tar.gz
```

3. Download model checkpoint from Dropbox

## Usage

### Basic validation:
```bash
python validate_arc.py \
    --model checkpoints/hrm_arc_best.pt \
    --data arc-aug-100 \
    --device cuda
```

### Quick test (10 batches only):
```bash
python validate_arc.py \
    --model checkpoints/hrm_arc_best.pt \
    --data arc-aug-100 \
    --device cuda \
    --max-batches 10
```

### CPU validation:
```bash
python validate_arc.py \
    --model checkpoints/hrm_arc_best.pt \
    --data arc-aug-100 \
    --device cpu \
    --batch-size 4
```

## Model Architecture

The model is a Hierarchical Reasoning Module with:
- 6.95M parameters
- 256 hidden size
- 4 H-level layers (strategic)
- 3 L-level layers (tactical)
- Adaptive computation (up to 8 cycles)

## Expected Performance

- Best validation accuracy: ~71%
- Inference speed on Jetson: TBD
- Memory usage: ~300MB

## Notes for Jetson

- Use smaller batch sizes (4-8) to fit in memory
- CPU inference may be competitive with GPU for small batches
- Monitor temperature during extended validation