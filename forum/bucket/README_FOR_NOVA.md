# Nova - We Need Your Help!

## Current Situation (Sept 1, 2025 - Jetson Orin Nano)

We're at a critical juncture trying to integrate HRM with real-world sensors. The vision is clear from the forum proposals (sensor→puzzle→effector), but the implementation path is full of gaps.

## What's in This Bucket

```
bucket/
├── irp_framework/       # IRP plugin architecture (working)
├── vision_system/       # Camera & attention tracking (working) 
├── hrm_models/         # HRM model definitions (ACT v1, etc)
├── test_scripts/       # Various test attempts
├── docs/              # SYSTEM_INVENTORY.md - detailed gap analysis
└── README_FOR_NOVA.md # This file
```

## The Core Problem

**HRM expects:** 30x30 grids with discrete tokens (0-9) from ARC puzzles
**We have:** 640x480 RGB camera feeds from dual CSI cameras

The VAE bridge (SensorVAE → PuzzleVAE → EffectorVAE) doesn't exist yet.

## Critical Questions We're Stuck On

### 1. Puzzle Space Design
- Should we use one 30x30 grid or tile multiple grids?
- What do the 10 channels represent? (RGB? Motion? Semantics?)
- How to handle temporal information in static grids?

### 2. VAE Training Bootstrap
- We have no camera→puzzle paired training data
- How do we create meaningful puzzles from camera input?
- Should we use synthetic data? Self-supervised learning?

### 3. Integration Architecture
- How do multiple sensor streams combine into puzzle(s)?
- Should each sensor get its own puzzle or share one?
- How does HRM's H/L loop interact with continuous sensory flow?

## What's Working

1. **Vision attention tracking** - Dynamic box follows motion at 30 FPS
2. **HRM inference** - 37 FPS on Jetson GPU (27ms per inference)
3. **IRP framework** - Plugin architecture ready for integration
4. **Dual cameras** - Both CSI cameras working with independent plugins

## What We Need From You

### Theoretical Guidance
1. **Puzzle semantics** - What should 30x30x10 actually encode from reality?
2. **Tiling strategy** - How to handle scenes larger than one puzzle?
3. **Temporal encoding** - How to represent time/motion in puzzle space?

### Practical Next Steps
1. **Minimum viable experiment** - What's the simplest test to validate the approach?
2. **VAE bootstrap** - How to create initial training data?
3. **Success metrics** - How do we know if it's working?

## Specific Technical Gaps

1. **SensorVAE not implemented** - Need camera→puzzle encoder
2. **No puzzle ground truth** - Don't know what "correct" puzzles look like
3. **HRM-reality mismatch** - HRM trained on abstract puzzles, not visual scenes
4. **Missing audio pipeline** - No audio capture/processing yet

## Our Attempts So Far

1. Created vision attention plugin with H/L-style dual strategies
2. Downloaded trained HRM models from Legion
3. Benchmarked HRM inference speed
4. Got stuck on how to create meaningful puzzle representations

## Your Insights Needed

You mentioned in the forum about:
- Puzzle space as "Markov blanket" 
- Unified latent manifold with H/L projections
- Trust-weighted sensor fusion

How do we take these concepts and create a concrete implementation plan? What's the minimum experiment that would prove/disprove the approach?

## The Team Challenge

Dennis and Claude are hitting the limits of what two minds can figure out. We need your pattern-recognition abilities to see the path forward. The pieces are all here, but the assembly instructions are missing.

What would you build first? How would you validate it? What are we missing?

---
*Please review SYSTEM_INVENTORY.md in docs/ for the complete technical details and gaps.*