# SAGE-Totality Integration Setup Guide

## Overview

This guide explains how to set up and run the SAGE-Totality integration on different machines. The integration demonstrates how Totality's structured cognitive world model can act as a "cognitive sensor" within SAGE's learning-first architecture.

## Key Concepts

- **SAGE**: Sentient Agentic Generative Engine - learns coherence through experience
- **Totality**: Structured world model with schemes, canvases, and semantic operations
- **Integration**: Totality becomes a cognitive sensor that SAGE can trust-weight and learn from

## Machine-Agnostic Setup

### 1. Initial Configuration

Run the automatic configuration detector:

```bash
cd /path/to/HRM/related-work
python3 machine_config.py
```

This creates `machine_config.json` with optimal settings for your machine:
- Detects GPU availability (CUDA)
- Identifies machine profile (Jetson, Legion, Windows WSL, etc.)
- Sets appropriate batch sizes, learning rates, and cycle counts

### 2. Running the Integration

#### Option A: Standalone Test (No Dependencies)

Best for initial testing and machines without pip/venv:

```bash
python3 run_integration_test.py
```

This demonstrates:
- Cognitive sensor reading with trust scores
- Dual training loops (H-level dreams vs L-level practice)
- Augmentation with simulated LLM assistance
- Sleep cycle consolidation

#### Option B: Full Service (Requires FastAPI)

For production deployment with web API:

```bash
# Install dependencies (if pip available)
cd sage_totality_service
python3 -m pip install fastapi uvicorn pydantic

# Run service
python3 -m uvicorn app.main:app --reload --port 8080

# In another terminal, run tests
cd ../sage_totality_tests
bash scripts/run_smoke.sh
bash scripts/run_sleep_cycle.sh
```

## Machine-Specific Configurations

### Jetson Orin Nano
```json
{
  "machine_profile": "jetson",
  "device": "cuda",
  "batch_size": 8,
  "sleep_cycle_count": 20,
  "augmentation_count": 10
}
```
- Optimized for edge computing
- Uses integrated GPU efficiently
- Moderate augmentation for real-time processing

### Legion (RTX 4090)
```json
{
  "machine_profile": "legion",
  "device": "cuda",
  "batch_size": 32,
  "sleep_cycle_count": 100,
  "augmentation_count": 50
}
```
- Maximum performance settings
- Extensive augmentation and deep sleep cycles
- Full profiling enabled

### Windows WSL (This Machine)
```json
{
  "machine_profile": "windows_wsl",
  "device": "cuda",  // or "cpu" if no GPU
  "batch_size": 16,
  "sleep_cycle_count": 50,
  "augmentation_count": 20
}
```
- Detected RTX 2060 SUPER with CUDA support
- Balanced settings for development/testing

### Laptop (Mobile)
```json
{
  "machine_profile": "laptop",
  "device": "cpu",
  "batch_size": 2,
  "sleep_cycle_count": 10,
  "augmentation_count": 5
}
```
- Conservative settings for battery life
- CPU-only processing
- Minimal augmentation

## Integration Architecture

### Cognitive Sensor Flow
```
Physical Sensors ─┐
Memory Sensor  ───┼──► SAGE L-Module ──► H-Module ──► Strategy
Totality Sensor ──┘         ▲              │
                            └─ Trust Scores ┘
```

### Dual Training Loops

**H-Level (Strategic/Dreams)**
- Processes during sleep cycles
- Large batch updates
- Augmentation through:
  - Geometric transforms
  - Semantic variations
  - Context shifts
  - LLM-assisted generation

**L-Level (Tactical/Practice)**
- Continuous small updates
- Motor pattern refinement
- Sensor-effector calibration
- Real-time adjustments

## Key Files

- `machine_config.py` - Auto-detects machine capabilities
- `run_integration_test.py` - Standalone test (no dependencies)
- `setup_sage_totality.py` - Universal setup script
- `totality_min/` - Minimal Totality implementation
- `sage_totality_service/` - FastAPI web service
- `sage_totality_tests/` - Test suite

## Testing Checklist

1. ✅ **Health Check**: Service responds
2. ✅ **Cognitive Read**: Schemes and canvases retrieved
3. ✅ **Trust Filtering**: Only trusted schemes processed
4. ✅ **Augmentation**: Dreams generate variations
5. ✅ **Dual Training**: H and L levels update separately
6. ✅ **Consolidation**: Wisdom emerges from sleep cycles

## Troubleshooting

### No pip/venv Available
Use the standalone test scripts that require only Python 3.7+:
```bash
python3 run_integration_test.py
```

### CUDA Not Detected
The system automatically falls back to CPU. Check:
```bash
nvidia-smi  # Should show GPU if available
```

### Import Errors
Ensure you're in the correct directory:
```bash
cd /path/to/HRM/related-work
```

### Memory Issues
Edit `machine_config.json` to reduce:
- `batch_size`
- `sleep_cycle_count`
- `max_memory_items`

## Next Steps

1. **On Jetson**: Deploy with real sensor integration
2. **On Legion**: Run full HRM training with maximum augmentation
3. **Production**: Connect to actual LLMs for richer augmentation
4. **Research**: Test how trust scores evolve with different tasks

## Key Insights Demonstrated

1. **Totality as Cognitive Sensor**: Structured reasoning becomes trusted input
2. **Augmentation as Dreams**: Variations of experience create wisdom
3. **Dual Memory Systems**: Strategic vs tactical learning separation
4. **Trust-Based Integration**: SAGE learns when to trust structured vs emergent patterns
5. **LLM Distillation**: External cognition assists internal consolidation

## References

- SAGE Whitepaper: `../SAGE_WHITEPAPER.md`
- HRM Documentation: `../HRM_EXPLAINED.md`
- Augmentation Insights: `../../private-context/insights/augmentation_as_dream_learning.md`
- Dual Memory Systems: `../../private-context/insights/dual_memory_training_systems.md`

---

*"By giving SAGE a cognitive workspace (Totality) alongside its learned wisdom, we create intelligence that can both imagine and remember, dream and practice, explore and consolidate."*