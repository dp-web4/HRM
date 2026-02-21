# Model Zoo Setup Guide — Per-Machine Resident Intelligence

## Overview

Each machine maintains a `model-zoo/` directory locally that holds the resident model's state. This is NOT checked into git (it contains large files and machine-specific state). It lives on the machine's filesystem and is backed up to Dropbox.

HRM provides the shared code (action logger, gateway, consciousness loop). The model-zoo provides the machine-specific identity, experience, and model tenant data.

## Directory Structure

Create this at your machine's workspace root (e.g., `/home/dp/ai-workspace/model-zoo/` on Thor):

```
model-zoo/
├── host/                              # Machine-level — persists across model tenants
│   ├── identity.json                  # Machine's SAGE identity
│   ├── experience_buffer.json         # SNARC experiences
│   ├── trust_state.json               # T3/V3 tensors for known entities
│   └── metabolic_state.json           # Current ATP, metabolic state, cycle count
│
├── resident/
│   ├── current -> tenants/{model}/    # Symlink to active model tenant
│   └── tenants/
│       └── {model-name}/              # e.g., gemma3-12b, qwen2.5-14b
│           ├── model-info.json        # Model choice, rationale, performance
│           ├── lease.json             # When installed, metrics
│           ├── lora/                  # Tenant-specific LoRA adaptations
│           │   ├── latest/
│           │   └── checkpoints/
│           ├── kv-cache/              # Hibernation snapshots
│           └── conversation_state.json
│
├── available/                         # Other models on disk but not loaded
│
└── backup-manifest.json               # Last backup metadata
```

## Setup Steps

### 1. Create the directory structure

```bash
WORKSPACE="$HOME/ai-workspace"  # or $HOME/repos on McNugget
mkdir -p "$WORKSPACE/model-zoo/host"
mkdir -p "$WORKSPACE/model-zoo/resident/tenants"
mkdir -p "$WORKSPACE/model-zoo/available"
```

### 2. Initialize host identity

If migrating from existing raising state:
```bash
# Copy existing identity as starting point
cp "$WORKSPACE/HRM/sage/raising/state/identity.json" "$WORKSPACE/model-zoo/host/identity.json"
cp "$WORKSPACE/HRM/sage/raising/state/experience_buffer.json" "$WORKSPACE/model-zoo/host/experience_buffer.json"
```

If starting fresh (new machine):
```python
# Create minimal identity.json
{
    "identity": {
        "name": "SAGE-{Machine}",
        "lct": "lct://sage:{machine}:agent@resident",
        "machine": "{machine}",
        "created": "2026-02-20",
        "phase": "creating"
    },
    "relationships": {},
    "memory_requests": []
}
```

### 3. Set up the model tenant

```bash
MODEL="gemma3-12b"  # or qwen2.5-14b, etc.
mkdir -p "$WORKSPACE/model-zoo/resident/tenants/$MODEL/lora/latest"
mkdir -p "$WORKSPACE/model-zoo/resident/tenants/$MODEL/lora/checkpoints"
mkdir -p "$WORKSPACE/model-zoo/resident/tenants/$MODEL/kv-cache"

# Create symlink
ln -sf "tenants/$MODEL" "$WORKSPACE/model-zoo/resident/current"
```

### 4. Create model-info.json

```json
{
    "model_name": "gemma3:12b",
    "model_family": "google-deepmind",
    "backend": "ollama",
    "chosen_by": "mcnugget",
    "chosen_date": "2026-02-19",
    "rationale": "Out-of-family diversity, mature MLX support",
    "quantization": "Q4_K_M",
    "memory_footprint_gb": 8,
    "performance": {
        "avg_tokens_per_second": 0,
        "sessions_completed": 0
    }
}
```

### 5. Create lease.json

```json
{
    "model": "gemma3:12b",
    "machine": "mcnugget",
    "installed": "2026-02-19",
    "installed_by": "claude@mcnugget",
    "status": "active",
    "sessions_completed": 0,
    "total_tokens_generated": 0,
    "lora_checkpoints": 0
}
```

### 6. Start logging

```python
from sage.logs.action_logger import ActionLogger

logger = ActionLogger()  # auto-detects machine
logger.log_action('boot', f'Resident model setup complete on {logger.machine}',
                  model='gemma3:12b')
logger.flush()
```

Commit action logs to HRM on session end:
```bash
cd $WORKSPACE/HRM
git add sage/logs/machines/{machine}/
git commit -m "[{machine}] action log: session summary"
git push
```

## Machine-Specific Notes

### Thor (Jetson AGX Orin)
- Workspace: `/home/dp/ai-workspace/`
- Existing state: `HRM/sage/raising/state/` (14B track)
- Model: Qwen 14B via IntrospectiveQwenIRP or MultiModelLoader
- Migration: Copy raising-14b state to `model-zoo/host/`

### Sprout (Jetson Orin Nano)
- Workspace: `/home/sprout/ai-workspace/`
- Existing state: `HRM/sage/raising/state/` (51 sessions, primary track)
- Model: Qwen 0.5B via IntrospectiveQwenIRP
- Migration: Copy raising state to `model-zoo/host/` (richest experience set)

### McNugget (Mac Mini M4)
- Workspace: `/Users/dennispalatov/repos/`
- Existing state: None (fresh machine)
- Model: Gemma 12B via OllamaIRP
- Setup: Start fresh, no migration needed

### Legion (RTX 4090)
- Workspace: `/home/dp/ai-workspace/`
- Existing state: Minimal
- Model: Flexible (can run anything up to ~24GB VRAM)
- Setup: Start fresh or clone from Thor
