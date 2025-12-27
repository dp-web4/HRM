# Revised Test Plan - Unified Conversation Testing
**Date**: December 26, 2025
**Status**: Ready for user review

## Executive Summary

Based on investigation findings, here's the revised test plan:

1. ✅ **Fixed Q3-Omni IRP plugin** - Added missing `disable_talker()` and `return_audio=False`
2. ✅ **All models verified available** - Qwen 0.5B, Nemotron 4B, Qwen 14B, Q3-Omni 30B
3. ✅ **Created individual model test** - Quick validation before full run
4. ✅ **Realistic performance estimates** - Based on Nemotron actual results

## Changes Made

### 1. Fixed Q3-Omni IRP Plugin (`sage/irp/plugins/q3_omni_irp.py`)

**Added**:
```python
# After model loading:
self.model.disable_talker()  # CRITICAL for text-only mode

# In generate():
return_audio=False,  # CRITICAL: text-only mode
use_audio_in_video=False,  # CRITICAL: text-only mode
```

**Why**: Previous successful Q3 tests showed these are required. Without them, model tries to initialize audio components and may crash.

### 2. Created Individual Model Test (`sage/tests/test_individual_models.py`)

**Purpose**: Quick 3-turn test to validate each model BEFORE running full comparison

**Benefits**:
- Catch loading issues early
- Test smallest model first (fail fast)
- Validate multi-turn memory works
- Get quick performance baseline
- **Total time**: ~5-10 minutes (vs 1-2 hours for full test)

**Usage**:
```bash
# Test all models (quick validation)
python3 sage/tests/test_individual_models.py --model all

# Test specific model
python3 sage/tests/test_individual_models.py --model qwen-05b
```

## Revised Test Strategy

### Phase 1: Individual Validation (RECOMMENDED FIRST STEP)
**Script**: `sage/tests/test_individual_models.py`
**Conversation**: 3 quick turns (one sentence answers)
**Expected time**: ~5-10 minutes total

**Test order** (smallest to largest):
1. ⬜ Qwen 2.5 0.5B - Expected: ~10-20s/turn, ~30-60s total
2. ⬜ Nemotron Nano 4B - Expected: ~20-40s/turn, ~60-120s total
3. ⬜ Qwen 2.5 14B - Expected: ~40-80s/turn, ~120-240s total
4. ⬜ Q3-Omni-30B - Expected: ~80-150s/turn, ~240-450s total

**If ANY model fails**, stop and fix before proceeding.

### Phase 2: Full Apples-to-Apples Comparison (AFTER VALIDATION)
**Script**: `sage/tests/test_unified_conversation.py`
**Conversation**: 5-turn dragon story with follow-ups
**Expected time**: ~1-1.5 hours total

**Expected performance** (based on Nemotron actual results + extrapolation):
1. Qwen 2.5 0.5B - ~20-40s/turn, ~2-3 min total
2. Nemotron Nano 4B - ~170s/turn, ~14 min total (✅ CONFIRMED)
3. Qwen 2.5 14B - ~250-350s/turn, ~20-30 min total
4. Q3-Omni-30B - ~500-700s/turn, ~40-60 min total

**Logs saved to**:
- `/tmp/conversation_qwen05b.log`
- `/tmp/conversation_nemotron.log`
- `/tmp/conversation_qwen14b.log`
- `/tmp/conversation_q3omni.log`

## Model Paths Verified

All models exist and are accessible:

| Model | Path | Status |
|-------|------|--------|
| Qwen 2.5 0.5B | `model-zoo/sage/epistemic-stances/qwen2.5-0.5b` | ✅ Available |
| Nemotron Nano 4B | `model-zoo/sage/language-models/llama-nemotron-nano-4b` | ✅ Tested |
| Qwen 2.5 14B | `model-zoo/sage/epistemic-stances/qwen2.5-14b` | ✅ Available |
| Q3-Omni-30B | `model-zoo/sage/omni-modal/qwen3-omni-30b` | ✅ Fixed |

**Alternative Q3-Omni versions** (if full model crashes):
- `model-zoo/sage/omni-modal/qwen3-omni-30b-int8-awq` (INT8 quantized)
- `model-zoo/sage/omni-modal/qwen3-omni-30b-fp4` (FP4 quantized)

## Risk Assessment

### Low Risk
- ✅ Qwen 2.5 0.5B - Small model, should load fast
- ✅ Nemotron Nano 4B - Already tested successfully

### Medium Risk
- ⚠️ Qwen 2.5 14B - Larger, but should fit in 122GB RAM

### Higher Risk
- ⚠️ Q3-Omni-30B Full - ~60GB model, may be tight on memory
  - **Mitigation**: Use INT8 version if full model fails
  - **System RAM**: 122GB total, 116GB free - theoretically enough
  - **Conservative approach**: Test INT8 version first

## Recommended Execution Plan

### Option A: Conservative (Recommended)
1. Run individual model tests (`test_individual_models.py`)
2. If all pass, run full comparison
3. If Q3-Omni fails, retry with INT8 version

### Option B: Direct
1. Run full comparison directly
2. If Q3-Omni crashes, retry with INT8 version

### Option C: Just Qwen + Nemotron
1. Skip Q3-Omni entirely for now
2. Test 3 models: Qwen 0.5B, Nemotron 4B, Qwen 14B
3. Add Q3-Omni later after optimizing memory

## User Questions for Approval

Before proceeding, please confirm:

1. **Test approach**: Option A (conservative), B (direct), or C (skip Q3-Omni)?
2. **Q3-Omni variant**: Full model or INT8 quantized?
3. **Logging**: Save all conversation logs to `/tmp/conversation_*.log`?
4. **Stop on error**: Should test stop if one model fails, or continue with others?

## Next Steps (Awaiting User Approval)

Once approved:
1. ⬜ Run chosen test approach
2. ⬜ Monitor progress and save logs
3. ⬜ Generate comparison summary
4. ⬜ Present results with performance metrics
5. ⬜ Document any issues found

## Estimated Total Time

- **Individual tests**: 5-10 minutes
- **Full comparison** (all 4 models): 1-1.5 hours
- **Total if both**: ~1.5-2 hours

**Much better than my original estimate of 2-4 hours!** User was right about the performance.
