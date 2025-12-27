# Unified Conversation Test - Investigation & Findings
**Date**: December 26, 2025
**Context**: Apples-to-apples multi-turn conversation test across all models

## Executive Summary

Q3-Omni loading crashed during full model test. Investigation reveals:
- ✅ Nemotron Nano test completed successfully (5/5 turns)
- ❌ Q3-Omni crashed during loading
- ⚠️ Qwen 2.5 models (0.5B, 14B) may not be downloaded yet
- 🔧 Test plan needs adjustment based on available resources

## What We Have vs What We Assumed

### ✅ Available Models

**Nemotron Nano 4B** (`model-zoo/sage/language-models/llama-nemotron-nano-4b`)
- **Status**: ✅ Fully working, tested successfully
- **Performance**: 3.34 tokens/sec, 8.52GB memory
- **Test results**: 5/5 turns completed (849s total, 170s avg/turn)
- **Multi-turn memory**: ✅ Excellent (correctly recalled dragon name, location, etc.)

**Q3-Omni-30B** (`model-zoo/sage/omni-modal/qwen3-omni-30b`)
- **Status**: ⚠️ Model exists, but crashes on loading
- **Previous success**: Yes - we've run text-only tests before
- **Known requirements**:
  - Must use `model.disable_talker()` for text-only mode
  - Must use `process_mm_info()` helper
  - Must set `return_audio=False` in generate()
  - Multiple quantized versions available (INT8, FP4)

**Qwen 2.5 0.5B**
- **Status**: ❌ NOT CONFIRMED - need to verify model exists
- **Expected location**: `model-zoo/sage/language-models/qwen2.5-0.5b` (?)
- **Alternative**: May be in HuggingFace cache

**Qwen 2.5 14B**
- **Status**: ❌ NOT CONFIRMED - need to verify model exists
- **Expected location**: `model-zoo/sage/language-models/qwen2.5-14b` (?)
- **Alternative**: `model-zoo/sage/qwen2.5-7b-instruct` exists (different size)

## What Went Wrong

### Q3-Omni Loading Crash
**Problem**: Test crashed while loading Q3-Omni-30B model (~19 minutes into loading)

**Probable causes**:
1. **Memory exhaustion**: Q3-Omni is ~60GB model, may exceed system capacity
2. **Missing talker disabling**: IRP plugin may not be calling `disable_talker()`
3. **Incorrect loading params**: May need INT8/FP4 quantized version instead of full model
4. **Multi-modal conflicts**: May be trying to initialize audio components unnecessarily

### Performance Estimate Issues
**User correctly identified**: My time estimates were WAY OFF

**What I said**: "15-20 minutes for 5-turn convo with 0.5B model"
**Reality check**: Nemotron Nano 4B (8x larger) did 5 turns in 14.2 minutes

**Actual expected performance** (extrapolating from Nemotron results):
- **Nemotron Nano 4B**: 170s/turn average, 850s total (5 turns)
- **Qwen 2.5 14B** (3.5x larger): ~300-400s/turn, ~25-30 min total
- **Qwen 2.5 0.5B** (8x smaller): ~20-30s/turn, ~2-3 min total
- **Q3-Omni-30B** (7.5x larger): ~500-800s/turn IF it loads, ~1 hour total

## Lessons Learned

### 1. Never Assume Models Are Downloaded
**Mistake**: Created IRP plugins for Qwen 2.5 0.5B and 14B without verifying models exist
**Lesson**: ALWAYS check `ls model-zoo/` before writing code that depends on models

### 2. Q3-Omni Needs Special Handling
**Mistake**: IRP plugin doesn't match proven working pattern from `test_qwen3_omni_simple_text.py`
**Lesson**: Use the working code patterns, not generic approaches

**Working Q3-Omni pattern**:
```python
# Load
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(...)
model.disable_talker()  # CRITICAL for text-only

# Generate
outputs = model.generate(
    **inputs,
    return_audio=False,  # CRITICAL
    use_audio_in_video=False,  # CRITICAL
    thinker_return_dict_in_generate=True  # Returns dict with .sequences
)
```

### 3. Quantization May Be Required
**Available Q3-Omni variants**:
- Full precision: `qwen3-omni-30b` (60GB+)
- INT8 AWQ: `qwen3-omni-30b-int8-awq`
- FP4: `qwen3-omni-30b-fp4`
- FP4 Weight-only: `qwen3-omni-30b-fp4-weight-only`

**System RAM**: 122GB total, 116GB available
**Conclusion**: Full model MIGHT fit, but quantized version safer

### 4. Test Smaller Models First
**User insight**: "it should not take 15-20 minutes for a 5-turn convo with a 0.5B model on this hardware!"

**Revised strategy**:
1. Start with smallest model (0.5B if available)
2. Validate conversation workflow works
3. Scale up to larger models
4. Use quantized Q3-Omni if full model fails

## Critical Files Referenced

### Working Test Examples
1. **`sage/tests/test_nemotron_nano_basic.py`** - ✅ Successful single-turn test
2. **`sage/tests/test_qwen3_omni_simple_text.py`** - ✅ Successful Q3 text-only pattern

### Our New Implementation
1. **`sage/conversation/sage_conversation_manager.py`** - Model-agnostic conversation manager
2. **`sage/irp/plugins/q3_omni_irp.py`** - ⚠️ Needs fixes based on working pattern
3. **`sage/irp/plugins/qwen25_05b_irp.py`** - ⚠️ Need to verify model exists
4. **`sage/irp/plugins/qwen25_14b_irp.py`** - ⚠️ Need to verify model exists
5. **`sage/irp/plugins/nemotron_nano_irp.py`** - ✅ Verified working
6. **`sage/tests/test_unified_conversation.py`** - ⚠️ Needs model verification

## Revised Test Plan

### Phase 1: Model Verification (NOW)
1. ✅ Verify Nemotron Nano exists and works
2. ⬜ Check if Qwen 2.5 0.5B is downloaded
3. ⬜ Check if Qwen 2.5 14B is downloaded
4. ⬜ If not, download or adjust test to use available models
5. ⬜ Fix Q3-Omni IRP plugin to match working pattern
6. ⬜ Decide: full Q3-Omni or quantized version?

### Phase 2: Individual Model Tests (NEXT)
Test each model individually BEFORE running full comparison:
1. ⬜ Test Qwen 2.5 0.5B (if available) - Expected: ~2-3 min for 5 turns
2. ⬜ Test Qwen 2.5 14B (if available) - Expected: ~25-30 min for 5 turns
3. ⬜ Test Q3-Omni (quantized first) - Expected: ~45-60 min for 5 turns
4. ✅ Nemotron Nano already tested - 14.2 min for 5 turns

### Phase 3: Apples-to-Apples Comparison (FINAL)
Only run full comparison AFTER individual tests pass:
- Run all models sequentially with same dragon story conversation
- Save individual logs for each model
- Generate comparison summary
- **Total expected time**: 1.5-2 hours (not the 2-4 hours I incorrectly estimated)

## Next Steps

**User requested**:
> "carefully review lessons learned, what we did, what we have. document the findings as you go. adjust the test plan as needed. then let's preview before you proceed."

**Action items**:
1. ⬜ Verify which Qwen 2.5 models actually exist
2. ⬜ Fix Q3-Omni IRP plugin to use proven working pattern
3. ⬜ Create revised test script that validates models before testing
4. ⬜ Preview adjusted plan with user before proceeding

## Performance Reality Check

**Nemotron Nano 4B Actual Results** (from successful test):
- Turn 1: 92.39s (story creation)
- Turn 2: 132.88s (color question)
- Turn 3: 170.20s (name question)
- Turn 4: 204.06s (location question)
- Turn 5: 249.84s (ending question)
- **Average**: 169.87s/turn
- **Total**: 849.37s (14.2 minutes)

**Extrapolated Realistic Estimates**:
- **0.5B model**: ~20-40s/turn → 2-3 min total
- **4B model** (Nemotron): 170s/turn → 14 min total ✅ CONFIRMED
- **14B model**: ~250-350s/turn → 20-30 min total
- **30B model** (Q3-Omni): ~500-700s/turn → 40-60 min total

**These numbers make sense** for this hardware. My initial estimates were too pessimistic.
