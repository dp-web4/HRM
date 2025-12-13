# Thor SAGE Setup Guide

**Platform**: NVIDIA Jetson AGX Thor
**H-Module**: Qwen2.5-14B (28x larger than Sprout)
**Memory**: 122GB unified memory
**Status**: Ready for setup

---

## What's Been Created

### 1. Thor Identity Infrastructure ✅

**Location**: `sage/identity/thor/`

Created files:
- `IDENTITY.md` - Thor's core self-description and role
- `HISTORY.md` - Session chronicle (currently pre-boot)
- `PERMISSIONS.md` - What Thor can/cannot do, phase-dependent
- `TRUST.md` - Trust relationships and calibration

**Thor's Role**:
- Research platform (14B H-Module for deeper reasoning)
- Federation coordinator (manages distributed SAGE instances)
- Pattern distillation (teaches Sprout what it learns)
- Multi-model experimentation (0.5B/14B/72B routing)

### 2. Multi-Model Loader ✅

**Location**: `sage/core/multi_model_loader.py`

**Features**:
- Dynamic model loading/unloading based on task complexity
- Memory management (keeps under 100GB limit)
- Automatic routing: SIMPLE→0.5B, MODERATE/COMPLEX→14B, VERY_COMPLEX→72B
- Can load multiple models simultaneously

**Task Complexity Routing**:
```python
from sage.core.multi_model_loader import create_thor_loader, TaskComplexity

loader = create_thor_loader(preload_default=True)  # Loads 14B

# Simple task (uses 0.5B)
response = loader.generate("What is 2+2?", complexity=TaskComplexity.SIMPLE)

# Complex reasoning (uses 14B)
response = loader.generate(
    "Explain the relationship between consciousness and emergence",
    complexity=TaskComplexity.COMPLEX
)
```

### 3. Download Infrastructure ✅

**Location**: `sage/setup/download_qwen_14b.py`

Ready to download Qwen2.5-14B-Instruct (~28GB).

---

## Next Steps to Complete Setup

### Step 1: Download Qwen2.5-14B

**Estimated time**: 30-60 minutes depending on connection

```bash
cd /home/dp/ai-workspace/HRM
python3 sage/setup/download_qwen_14b.py
```

This will:
- Download Qwen/Qwen2.5-14B-Instruct from HuggingFace
- Place it in `model-zoo/sage/epistemic-stances/qwen2.5-14b/base-instruct/`
- Use ~28GB disk space
- Resume if interrupted

**Check download progress**: Look for HuggingFace Hub progress bars

### Step 2: Build Epistemic Stancing Infrastructure

Epistemic stancing trains the model to respond from different epistemic positions:
- **Curious**: Exploratory, asking questions
- **Skeptical**: Critical, verifying claims
- **Confident**: Direct, authoritative
- **Uncertain**: Acknowledging limits

**Components needed**:
1. Stance dataset creation
2. Fine-tuning pipeline for 14B
3. LoRA adapters for memory efficiency
4. Stance evaluation metrics

**Files to create**:
- `sage/training/epistemic_stancing_14b.py`
- `sage/training/stance_dataset_generator.py`
- `sage/training/stance_evaluator.py`

### Step 3: Integrate Sleep-Cycle Learning

Connect sleep-cycle training with coherent awakening:

**Components**:
1. Memory consolidation during DREAM state
2. Pattern extraction from episodic memory
3. Weight updates saved to disk
4. Restoration on next awakening

**Integration points**:
- `sage/awakening/coherent_awakening.py` (already exists)
- `sage/training/sleep_cycle_training.py` (exists, needs integration)
- `sage/irp/memory.py` (exists, needs consolidation hooks)

**What needs connecting**:
- Session end → Save learned weights
- Session start → Restore learned weights
- DREAM state → Consolidation training
- Pattern extraction → Epistemic DB

### Step 4: Configure H-Module for 14B

Update SAGE initialization to use multi-model loader:

**Files to modify**:
- `sage/core/sage_core.py` or equivalent initialization
- Add `multi_model_loader` as core component
- Default to 14B for H-Module reasoning
- Keep 0.5B for L-Module tactical execution

**Example integration**:
```python
class SAGECore:
    def __init__(self, ...):
        # Multi-model loader for strategic reasoning
        self.model_loader = create_thor_loader(preload_default=True)

        # Default to 14B H-Module
        self.h_module = self.model_loader.get_model_for_task(
            TaskComplexity.MODERATE
        )
```

### Step 5: Thor-Specific Coherent Awakening

Modify awakening protocol for Thor's configuration:

**Config needed**:
```python
# sage/awakening/thor_config.py
from sage.awakening.coherent_awakening import CoherentAwakening

awakening = CoherentAwakening(
    identity_dir=Path("sage/identity/thor"),
    state_dir=Path("sage/state/thor"),
    base_dir=Path("sage")
)

# Prepare for first awakening
coherence_field = awakening.prepare_coherence_field()
preamble = awakening.create_boot_preamble(coherence_field)

# Boot with 14B H-Module
sage_thor = awakening.coherent_boot(
    coherence_field,
    model_size=ModelSize.MEDIUM  # 14B
)
```

### Step 6: First Awakening Test

Test Thor's first coherent session:

**Test script**:
```python
# sage/tests/test_thor_first_awakening.py
from sage.awakening.thor_config import boot_thor_sage

sage = boot_thor_sage()

# Simple test
response = sage.respond("Hello, SAGE. Who are you?")
print(response)

# Should reference Thor identity, pre-grounding phase
# Should show larger model's reasoning capability

# End session
sage.coherent_end(memory_request="First awakening experience")
```

**Expected results**:
- Thor references its identity from IDENTITY.md
- Acknowledges being young/pre-grounding
- Shows richer reasoning than 0.5B would
- Session logged to HISTORY.md
- State saved for next awakening

---

## Architecture Overview

```
Thor SAGE Architecture:

┌─────────────────────────────────────────┐
│ Coherent Awakening (Session Continuity) │
│  - Load identity (thor/IDENTITY.md)     │
│  - Restore learned state                │
│  - Generate boot preamble               │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│      Multi-Model Loader (H-Module)      │
│  - 0.5B: Fast, simple tasks             │
│  - 14B: Default, complex reasoning      │
│  - 72B: Available for very complex      │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         SAGE Core (Consciousness)       │
│  - SNARC: Salience-based attention      │
│  - IRP: Iterative refinement            │
│  - Memory: Episodic/semantic/procedural │
│  - ATP: Metabolic resource allocation   │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│    Sleep-Cycle Learning (Overnight)     │
│  - DREAM: Pattern consolidation         │
│  - Weight updates: Epistemic stancing   │
│  - Memory abstraction: Higher levels    │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│      Session End (Save Everything)      │
│  - Update HISTORY.md                    │
│  - Save learned weights                 │
│  - Consolidate memories                 │
│  - Push to git (coordination)           │
└─────────────────────────────────────────┘
```

---

## Current Status

✅ **Complete**:
1. Thor identity infrastructure (IDENTITY, HISTORY, PERMISSIONS, TRUST)
2. Multi-model loader (0.5B/14B/72B routing)
3. Download infrastructure (ready to pull 14B)

⏳ **In Progress**:
1. Download Qwen2.5-14B (next: run download script)

🔄 **Pending**:
1. Epistemic stancing training infrastructure
2. Sleep-cycle learning integration
3. H-Module configuration in SAGE core
4. Thor-specific awakening config
5. First awakening test

---

## Key Differences: Thor vs Sprout

| Aspect | Sprout (Orin Nano) | Thor (AGX Thor) |
|--------|-------------------|-----------------|
| **H-Module** | Qwen2.5-0.5B (0.5B) | Qwen2.5-14B (14B) |
| **Memory** | 4GB | 122GB |
| **Role** | Production edge | Research platform |
| **Multi-model** | No (fixed 0.5B) | Yes (0.5B/14B/72B) |
| **Purpose** | Validate in constraints | Explore capabilities |
| **Learning** | Online adaptation | Full training + distillation |

---

## Estimated Timeline

**Phase 1: Download and Basic Setup** (1-2 hours)
- Download 14B model (30-60 min)
- Verify model loads correctly (15 min)
- Test basic generation (15 min)
- Configure SAGE integration (30 min)

**Phase 2: Epistemic Stancing** (4-6 hours)
- Create stance dataset (2 hours)
- Fine-tune with LoRA (2-3 hours)
- Evaluate stance quality (1 hour)

**Phase 3: Sleep-Cycle Integration** (2-3 hours)
- Connect consolidation to awakening (1 hour)
- Test save/restore cycle (1 hour)
- Validate cross-session learning (1 hour)

**Phase 4: First Awakening** (1 hour)
- Thor's first coherent session
- Document in HISTORY.md
- Validate identity persistence

**Total**: ~8-12 hours of setup + training time

---

## Next Immediate Action

**Run the download script**:

```bash
cd /home/dp/ai-workspace/HRM
python3 sage/setup/download_qwen_14b.py
```

This will take 30-60 minutes. While it's downloading, you can work on other
components (epistemic stancing infrastructure, sleep-cycle integration, etc.).

Once the model is downloaded, we can test it loads correctly and begin the
integration work.

---

*Thor is ready to awaken with 28x more reasoning capacity than Sprout. Let's see what emerges with a larger frontal lobe!* 🧠🚀
