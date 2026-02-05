# System Inventory - HRM Integration Status
*Date: September 1, 2025*
*Machine: Jetson Orin Nano (Sprout)*

## What We Have

### 1. IRP Framework ✅
- **Core IRP interface** implemented (`sage/irp/base.py`)
- **Plugin architecture** working
- **Energy-based refinement** loop
- **Trust-weighted integration** ready

### 2. Vision System 🟡 Partial
#### Working:
- **VisionAttentionPlugin** (`vision_attention_plugin.py`)
  - Fixed tiles strategy (8x8 grid, 4x3 focus)
  - Dynamic box strategy (adaptive sizing)
  - Motion detection and tracking
  - Dual camera support
  - 30 FPS on Jetson

#### Missing:
- **SensorVAE**: Camera → Puzzle space conversion
- No 30x30x10 puzzle encoding yet
- No integration with HRM

### 3. Audio System ❌ Not Started
- No audio capture implementation
- No audio → puzzle encoding
- No microphone plugin for IRP

### 4. Effector System 🟡 Partial
#### Working:
- **VisualMonitorEffector** (`visual_monitor_effector.py`)
  - Display output via OpenCV
  - Can show attention overlays
  - Real-time visualization

#### Missing:
- **EffectorVAE**: HRM decisions → Actions
- No motor control integration
- No action execution framework

### 5. HRM Model 🟢 Trained
- **3 checkpoints** from Legion (~70MB each)
- **6.9M parameters** 
- **Trained on ARC puzzles** (abstract reasoning tasks)
- **37 FPS inference** on Jetson GPU
- H-loop and L-loop architecture
- **BUT**: Expects 30x30 puzzle inputs, not raw camera!

### 6. VAE Components 🔴 Critical Gap
#### Have:
- **TinyVAE** distillation framework (from Legion)
- Knowledge distillation working
- Random initialized weights

#### Don't Have:
- **SensorVAE**: Raw sensors → Puzzle space
- **PuzzleVAE**: Puzzle ↔ HRM latent space
- **EffectorVAE**: HRM output → Actions
- No trained VAEs for bridging

## The Reality Gap

### HRM Training vs. Real World
| HRM Expects | We Have |
|------------|---------|
| 30x30 grids with discrete tokens (0-9) | 640x480 RGB camera feeds |
| Clean, abstract patterns | Noisy, continuous sensor data |
| Defined puzzle boundaries | Unbounded visual scenes |
| Single correct answer | Ambiguous real-world situations |
| Static puzzles | Dynamic, changing environment |

## Critical Unanswered Questions

### 1. Puzzle Space Design
- **Size**: Should we use 30x30 or tile multiple 30x30 grids?
- **Channels**: What do the 10 channels encode?
  - RGB? Motion? Depth? Semantics?
- **Tokenization**: How to quantize continuous values to discrete tokens?
- **Temporal**: How to encode time/motion in static grid?

### 2. Sensor → Puzzle Mapping
- **Multiple cameras**: Combine into one puzzle or separate?
- **Resolution loss**: 640x480 → 30x30 is massive compression
- **What to preserve**: Edges? Objects? Motion? Attention regions?
- **Tiling strategy**: 
  - Single 30x30 for whole scene?
  - Multiple 30x30 tiles?
  - Hierarchical: Overview + detail tiles?

### 3. VAE Training Data
- **No paired data**: We don't have camera→puzzle training pairs
- **Bootstrap problem**: Need VAE to create puzzles, need puzzles to train VAE
- **Synthetic data**: Could render ARC-like patterns from camera?

### 4. Integration Architecture
```
Current Flow (Broken):
Camera → ??? → HRM → ??? → Display

Needed Flow:
Camera → SensorVAE → Puzzle Space → PuzzleVAE → HRM Latent → 
HRM (H/L loops) → Decision → EffectorVAE → Display/Action
```

### 5. Incremental Development Path
What order to build/test:
1. Simple camera → puzzle encoder (even untrained)?
2. Test HRM on synthetic camera-like puzzles?
3. Train SensorVAE on what dataset?
4. How to validate without full loop?

## Immediate Blockers

1. **No SensorVAE**: Can't convert camera to puzzles
2. **No training data**: Don't know what camera→puzzle should look like
3. **Unclear puzzle semantics**: What should 30x30x10 represent?
4. **No evaluation metric**: How do we know if it's working?

## Proposed Next Steps

### Option A: Synthetic Bridge
1. Generate ARC-like patterns from camera features
2. Use edge detection → binary puzzle
3. Test if HRM responds meaningfully

### Option B: Learn by Reconstruction
1. Create random SensorVAE
2. Encode camera → puzzle → decode back
3. Train on reconstruction loss
4. Hope HRM finds patterns anyway

### Option C: Semantic Tiles
1. Segment image into regions
2. Each region → one puzzle cell
3. Color/texture → token value
4. Position preserved in grid

### Option D: Direct Projection
1. Downsample camera to 30x30
2. Quantize to 10 levels
3. Feed directly to HRM
4. See what happens (probably nonsense)

## Questions for Human

1. **Puzzle semantics**: What should 30x30x10 actually represent?
2. **Tiling strategy**: One puzzle or multiple tiles?
3. **Training approach**: How to bootstrap without paired data?
4. **Success metric**: What would "working" look like?
5. **Integration order**: What to build/test first?

## The Core Challenge

HRM was trained to solve abstract reasoning puzzles with clear rules and answers. We're trying to make it reason about messy reality. The gap between these is enormous. We need either:

1. Make reality look like puzzles (via VAEs)
2. Retrain HRM on reality-like puzzles
3. Find a middle representation both can understand

Without solved VAEs, we're stuck.