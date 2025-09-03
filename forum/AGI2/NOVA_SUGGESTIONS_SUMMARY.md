# Nova's AGI2 Suggestions Summary

*Date: September 3, 2025*
*For Legion training session tonight*

## Key Points from Nova

### 1. Early Results Confirmation
- **71% on ARC-AGI-1** matches our findings
- **20% on ARC-AGI-2** without any AGI-2 training shows genuine generalization
- Nova correctly identified the architecture mismatch and found `train_arc_full_nova.py`

### 2. Efficiency Advantage
Nova emphasizes our efficiency edge:
- **6.95M parameters** is Jetson-class (edge-compatible)
- Low wattage, strong fit with ARC Prize cost-per-task criteria
- Can run on laptop CPU without strain

### 3. Training Strategy for Legion

Nova suggests:
1. **Begin full ARC-AGI-2 training run** on Legion immediately
2. **Benchmark with the same eval harness** (no augmentation at eval)
3. **Log compute efficiency** (samples/sec, VRAM, wall-clock runtime)

### 4. Expected Trajectory
- Current: 20% (no AGI-2 training)
- Quick target: 40-60% with direct AGI-2 training
- This would be competitive with Kaggle contenders

### 5. Files Provided

Nova created evaluation stubs in `forum/AGI2/`:
- `eval_arc.py` - Main evaluation harness (to be expanded)
- `hrm_infer.py` - Inference wrapper
- `repro.sh` - Reproducibility script
- `requirements.txt` - Dependencies

### 6. Competition Context
Nova's summary aligns with our findings:
- OpenAI o3: 87.5% (but 172x compute, not efficient)
- Public systems: 0-9% on AGI-2
- Our 20%: Already beats all public systems

## Action Items for Legion

1. **Use the correct architecture** (`train_arc_full_nova.py`)
2. **Load ARC-AGI-2 dataset** (1,000 training tasks)
3. **Start with Nova's suggested hyperparameters**
4. **Track efficiency metrics** for prize qualification

## Nova's Insight

The bidirectional H↔L communication is the key innovation. This wasn't in the original HRM paper - it's Nova's enhancement that enables strategic-tactical reasoning loops.

---

*Ready for Legion training session!*