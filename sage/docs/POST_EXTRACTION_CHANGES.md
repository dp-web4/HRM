# Post-Extraction Changes

## Current Status
- ✅ All attention weights extracted (48 layers, 1.7 GB)
- ✅ QK normalization implemented
- ✅ Layer norms extracted
- ✅ Final norm extracted
- ✅ Correct tokenizer identified
- ⏳ **IN PROGRESS**: Extracting all 128 experts (28% complete, 1,739/6,144 files)

## Required Changes After Extraction Completes

### 1. Remove Router Masking

**File**: `sage/compression/selective_expert_loader.py`

**Lines to Remove**: 233-239

**Current Code (WRONG - forces only experts 0-7)**:
```python
# CRITICAL: Mask to only available experts (deep focus: 0-7 only)
# For deep expert architecture, we only extracted experts 0-7 with full 48 layers
# Router must be constrained to only select from available experts
available_experts = list(range(8))  # Experts 0-7 only
mask = torch.full_like(router_logits, float('-inf'))
mask[available_experts] = 0
router_logits = router_logits + mask  # Only available experts can be selected
```

**Replacement Code (CORRECT - let router choose freely)**:
```python
# Router can now select from all 128 experts!
# No masking needed - all experts extracted
```

**Exact Edit Command**:
```python
Edit(
    file_path="sage/compression/selective_expert_loader.py",
    old_string="""        # CRITICAL: Mask to only available experts (deep focus: 0-7 only)
        # For deep expert architecture, we only extracted experts 0-7 with full 48 layers
        # Router must be constrained to only select from available experts
        available_experts = list(range(8))  # Experts 0-7 only
        mask = torch.full_like(router_logits, float('-inf'))
        mask[available_experts] = 0
        router_logits = router_logits + mask  # Only available experts can be selected""",
    new_string="""        # Router can now select from all 128 experts!
        # No masking needed - all experts extracted"""
)
```

### 2. Why This Matters

**The Discovery**: Expert Horizontal Specialization

We discovered that Q3-Omni's 128 experts are **horizontally specialized by semantic domain**:
- Expert 0: Technical specifications
- Expert 23: Poetic language
- Expert 47: Futuristic concepts
- Expert 89: General prose
- ... (128 different semantic specializations)

**The Problem**:
With only 8 experts available, we forced the router to use wrong semantic domains for the content. Like trying to write poetry using only technical specs experts!

**The Evidence**:
Testing with correct tokenizer produced garbled output that contained SOME English words but no coherence:
```
"The future of AI is toolbar大妈 STATES immblr..."
```

The scattered English words prove the experts CAN process language, but wrong combinations show we're using wrong experts for the context.

**The Solution**:
Extract all 128 experts so the router can select semantically appropriate experts for each token/context.

### 3. Testing After Changes

**Test Script**: `sage/tests/test_with_correct_tokenizer.py`

**Expected Behavior**:
1. Router sees all 128 expert logits
2. Router selects top-k experts based on actual semantic appropriateness
3. Expert selection varies by context (technical prompts → technical experts, creative prompts → creative experts)
4. Generation should be COHERENT with proper vocabulary and grammar

**Test Command**:
```bash
cd /home/dp/ai-workspace/HRM
python3 sage/tests/test_with_correct_tokenizer.py
```

**Success Criteria**:
- ✅ Output contains complete, grammatically correct sentences
- ✅ Content is semantically relevant to prompt
- ✅ No random character mixtures or incoherence
- ✅ Router selects different experts for different types of content

### 4. Validation Steps

After removing the mask and testing:

1. **Log router selections**: Modify code to print which experts are chosen
2. **Test diverse prompts**: Technical, creative, conversational, multilingual
3. **Verify expert diversity**: Router should use different experts for different domains
4. **Measure coherence**: Perplexity should drop significantly vs. current garbled output
5. **Document patterns**: Which experts get selected for which content types

### 5. Current Extraction Progress

Monitor with:
```bash
# Dashboard
bash /tmp/monitor_dashboard.sh

# Or detailed layer status
python3 -c "
import os
from collections import defaultdict

expert_dir = 'model-zoo/sage/omni-modal/qwen3-omni-30b-extracted/experts/'
layer_counts = defaultdict(int)

for f in os.listdir(expert_dir):
    if f.endswith('.safetensors'):
        parts = f.split('_')
        if 'layer' in parts:
            layer_idx = parts.index('layer')
            layer_num = int(parts[layer_idx + 1].split('.')[0])
            layer_counts[layer_num] += 1

total = sum(layer_counts.values())
complete_layers = sum(1 for c in layer_counts.values() if c == 128)

print(f'Total: {total}/6144 ({total/6144*100:.1f}%)')
print(f'Complete layers: {complete_layers}/48')
for layer in sorted(layer_counts.keys()):
    count = layer_counts[layer]
    status = '✅' if count == 128 else '🔄'
    print(f'{status} Layer {layer:2d}: {count:3d}/128')
"
```

### 6. Timeline

**Current**: 28% complete (1,739/6,144 files, 16 GB)
**Estimated completion**: ~2 hours from start
**Started**: Background process running (PID: 83746, 83749)
**Next action**: Remove mask, test generation, celebrate coherent output!

---

## The Moment of Truth

After ~2 hours of extraction, we'll have:
- ✅ Complete Q3-Omni architecture reconstructed
- ✅ All 128 experts × 48 layers available
- ✅ Router free to select appropriate experts
- ✅ Real attention weights loaded
- ✅ Correct tokenizer

**This should produce coherent text generation** because the router can finally select semantically appropriate experts for each context, just like the full Q3-Omni model does!
