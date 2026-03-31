# Sprout SDK Installation — ARC-AGI-3

**Date**: 2026-03-31
**Machine**: Sprout (Jetson Orin Nano 8GB, Python 3.10.12, aarch64)

## Result: Fully Functional

The ARC-AGI-3 SDK runs on Sprout with two minor patches. No functionality is missing.

## Packages Installed

| Package | Version | Notes |
|---------|---------|-------|
| arc-agi | 0.0.7 | Core data library (Grid, Task, Dataset) |
| arc-agi-core | 0.1.14 | Dependency of arc-agi |
| arcengine | 0.9.3 | Game engine (ARCBaseGame, FrameData, GameAction) |
| svg-py | 1.10.0 | Pulled in by arc-agi-core |

## Patches Required

### 1. `typing.Self` → `typing_extensions.Self` (Python 3.10)

`Self` was added to `typing` in Python 3.11. Three files in `arc_agi_core` import it from `typing`. Fix: move the import to `typing_extensions` (already installed).

**Files patched** (in `~/.local/lib/python3.10/site-packages/arc_agi_core/`):
- `_grid.py`
- `_pair.py`
- `_task.py`
- `_dataset.py`

**Patch pattern** (single-line imports):
```python
# Before
from typing import List, Union, Self, Tuple

# After
from typing import List, Union, Tuple
from typing_extensions import Self
```

For `_dataset.py` (multiline import block), remove `Self,` from the block and add `from typing_extensions import Self` after it.

### 2. `arcengine` requires Python ≥3.12 (metadata only)

arcengine 0.9.3 declares `Requires-Python: >=3.12` but is a pure-Python wheel (`py3-none-any`). All code works on 3.10 — the constraint is overly strict.

```bash
pip3 install /tmp/arc-wheels/arcengine-0.9.3-py3-none-any.whl --no-deps --ignore-requires-python
```

Or download first:
```bash
pip3 download arcengine --no-deps --python-version 3.12 -d /tmp/arc-wheels
pip3 install /tmp/arc-wheels/arcengine-0.9.3-py3-none-any.whl --no-deps --ignore-requires-python
```

## Performance

Tested with multi-sprite game (5 objects, 16x16 grid, camera renders to 64x64):

| Metric | Sprout | McNugget (reference) |
|--------|--------|---------------------|
| Game loop (no analysis) | ~3,000 steps/sec | ~5,910 steps/sec |
| With frame analysis | ~850 steps/sec | — |
| Action dispatch | 0.4-0.7ms | — |
| Frame analysis (connected components) | 0.7-1.3ms | — |
| RSS (SDK only) | 38-39 MB | 101 MB |

The bottleneck for actual gameplay is LLM response time (3-9 seconds on Sprout via Ollama), not the SDK. Game engine overhead is negligible.

## API Notes

**Frame format**: `(1, H, W)` int8 palette-indexed arrays (NOT RGB). Camera renders at 64x64 regardless of grid_size. Single channel — palette index per pixel, -1 = transparent.

**Sprites**: 2D int8 arrays of palette indices, not RGB.

**Game loop**:
```python
from arcengine import ARCBaseGame, ActionInput, GameAction

action = ActionInput(id=GameAction.ACTION1)
frame_data = game.perform_action(action)
# frame_data.frame = (1, 64, 64) int8
# frame_data.state = GameState enum
```

**ACTION6** (coordinates): `ActionInput(id=GameAction.ACTION6, data={"x": 32, "y": 15})`

**GameAction enum quirk**: `GameAction(1)` may fail on Python 3.10 enum. Use `GameAction.ACTION1` or `GameAction['ACTION1']` instead.

**Datasets**: `ARC1Training()` and `ARC2Training()` return 0 tasks without cached data. Data needs to be downloaded separately or loaded from local files.

## Memory Budget (with SAGE daemon running)

| Component | RAM |
|-----------|-----|
| Ollama + qwen3.5:0.8b | ~2.3 GB |
| SAGE daemon | ~120 MB |
| Membot | ~71 MB |
| ARC SDK | ~39 MB |
| **Available** | **~2 GB headroom** |

Sprout can run the SDK alongside all existing services without pressure.

## What Sprout Can Do

- Develop and test GridVisionIRP integration locally
- Run game environments for debugging action/perception loops
- Test consciousness loop → game action pipeline end-to-end
- Validate patches and compatibility before fleet-wide deployment

## What Sprout Can't Do

- Run the competition sandbox (RTX 5090, 32GB VRAM required)
- Run large vision models for grid analysis (CLIP etc.)
- Match McNugget/Thor step throughput (but doesn't need to — LLM is the bottleneck)
