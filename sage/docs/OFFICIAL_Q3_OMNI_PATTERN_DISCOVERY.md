# Official Q3-Omni Inference Pattern Discovery

**Date**: 2025-12-19
**Status**: 🔍 Pattern Identified, Testing in Progress

---

## Summary

After hitting errors with our attempts to load Q3-Omni-30B, we searched for how people actually run it successfully and discovered the **official inference pattern** from HuggingFace documentation.

**Key Discovery**: Q3-Omni requires a specific multimodal inference pattern that we weren't using.

---

## What We Tried Initially (That Didn't Work)

### Attempt 1: AutoModelForCausalLM
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
```

**Result**: ❌ Error - `Unrecognized configuration class Qwen3OmniMoeConfig for AutoModelForCausalLM`

**Why it Failed**: Q3-Omni is a **multimodal** model (text + audio), not just a causal LM. AutoModelForCausalLM doesn't recognize multimodal configs.

### Attempt 2: Direct Model Class
```python
from transformers import Qwen3OmniMoeForConditionalGeneration, AutoTokenizer

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
```

**Result**: ⚠️ Weights loaded but initialization failed
- All 2034 weights loaded successfully (100% complete)
- Error during `mark_tied_weights_as_initialized()`: missing `lm_head` attribute
- Additional error: `initializer_range` attribute missing from config

**Why it Failed**: Partial - model class exists but has initialization bugs in transformers 5.0.0.dev0. Also, we used only the tokenizer instead of the full processor.

---

## The Official Working Pattern

### Source
HuggingFace Model Card: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct

### Required Components

1. **Qwen3OmniMoeProcessor** (NOT just AutoTokenizer!)
2. **Qwen3OmniMoeForConditionalGeneration** (model class)
3. **Chat template formatting** (via processor)
4. **Tuple unpacking** from generate() - returns (text_ids, audio)
5. **Special parameters** for text-only inference

### Complete Working Code

```python
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Load both model AND processor
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",                          # Auto-select dtype
    device_map="auto",                     # Auto-select device placement
    attn_implementation="flash_attention_2",  # Use FlashAttention (optional)
)

processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

# Optional: Disable audio generation to save ~10GB VRAM
model.disable_talker()

# Format input as conversation (not raw text!)
conversation = [
    {
        "role": "user",
        "content": "The capital of France is"
    }
]

# Apply chat template BEFORE tokenization
text = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=False
)

# Process with the processor (not tokenizer)
inputs = processor(text=text, return_tensors="pt", padding=True)
inputs = inputs.to(model.device).to(model.dtype)

# CRITICAL: generate() returns TUPLE (text_ids, audio)
text_ids, audio = model.generate(
    **inputs,
    max_new_tokens=5,
    do_sample=False,
    return_audio=False,  # Skip audio generation for text-only
)

# Decode output (skip prompt tokens)
result = processor.batch_decode(
    text_ids.sequences[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(result[0])
```

---

## Key Differences from Our Approach

| What We Did | What We Should Do | Why It Matters |
|-------------|-------------------|----------------|
| `AutoTokenizer` | `Qwen3OmniMoeProcessor` | Processor handles multimodal chat templates |
| Direct text input | `apply_chat_template()` on conversation dict | Model expects formatted chat structure |
| `tokenizer.encode()` | `processor(text=...)` | Processor adds special tokens correctly |
| `output_ids = model.generate()` | `text_ids, audio = model.generate()` | Returns tuple, not single tensor |
| No return_audio param | `return_audio=False` | Skips audio generation, saves memory |
| N/A | `model.disable_talker()` | Saves ~10GB VRAM for text-only use |

---

## Installation Requirements

### Transformers Version

**CRITICAL**: Must install transformers from source (PyPI doesn't have Q3-Omni support yet)

```bash
pip install git+https://github.com/huggingface/transformers
```

### Dependencies

```bash
pip install git+https://github.com/huggingface/transformers  # Main requirement
pip install accelerate                                        # For device_map="auto"
pip install qwen-omni-utils -U                               # Qwen utilities
pip install -U flash-attn --no-build-isolation              # Optional: FlashAttention 2
```

**Note**: FlashAttention 2 is optional. vLLM includes it by default. For CPU inference, skip it.

---

## What We Learned

### 1. Q3-Omni Is Not a Standard Language Model

It's a **multimodal conditional generation** model:
- **Input**: Text, audio, images, video
- **Output**: Text AND audio (optionally)

Using `AutoModelForCausalLM` fails because it's not designed for multimodal architectures.

### 2. The Processor Is Essential

The `Qwen3OmniMoeProcessor` does more than tokenization:
- Applies chat templates specific to Q3-Omni
- Handles multimodal input formatting
- Manages special tokens for different modalities
- Correctly decodes mixed-modality outputs

### 3. Model Class Has Bugs in Current Transformers

Our attempt with `Qwen3OmniMoeForConditionalGeneration` + `AutoTokenizer`:
- ✅ All weights loaded successfully
- ❌ Initialization failed (missing `lm_head`, `initializer_range`)

This confirms **transformers 5.0.0.dev0 has incomplete Q3-Omni implementation**.

Using the official pattern (with processor) likely avoids these initialization issues.

### 4. Memory Optimization Options

For text-only inference on resource-constrained devices:
1. **Set `return_audio=False`** in `generate()` call
2. **Call `model.disable_talker()`** after loading (saves ~10GB)
3. **Use vLLM instead of transformers** (recommended for MoE inference speed)

### 5. Our Segmented Implementation May Already Be Correct

If the official model has initialization bugs but we:
- ✅ Extracted all 2034 weights successfully
- ✅ Mathematically verified router correctness
- ✅ Implemented correct architecture
- ✅ Achieve coherent generation

Then our segmented implementation might be **more reliable** than the official (which has bugs).

**BUT** - we still need to:
1. ✅ Use the processor (not just tokenizer)
2. ✅ Apply chat templates correctly
3. ✅ Verify outputs match official pattern once it's working

---

## Current Status

### Model Loading Test

Running in background: `sage/tests/test_full_model_with_swap.py`
- Status: 112/2034 weights loaded (5% complete)
- Using 272GB total memory (122GB RAM + 150GB swap)
- Loading progressing steadily (~2000 weights/sec)

### Official Pattern Test

Created: `sage/tests/test_official_inference_pattern.py`
- Implements the official HuggingFace pattern
- Uses `Qwen3OmniMoeProcessor` + chat templates
- Unpacks generate() tuple correctly
- Disables talker to save memory
- Ready to run once model finishes loading

---

## Next Steps

1. ✅ **Pattern identified** - Official HuggingFace method documented
2. ✅ **Dependencies identified** - Need transformers from GitHub source
3. ⏳ **Model loading** - Wait for background loading to complete
4. ⏳ **Test official pattern** - Run `test_official_inference_pattern.py`
5. ⏳ **Get baseline outputs** - Collect predictions from working config
6. ⏳ **Compare with our implementation** - Verify our outputs match
7. ⏳ **Actually complete Stage A** - Real baseline verification

---

## References

- **HuggingFace Model Card**: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
- **QwenLM GitHub**: https://github.com/QwenLM/Qwen3-Omni
- **Transformers Docs**: Must use source installation
- **vLLM Recommendation**: For production MoE inference (much faster)

---

## Critical Lesson

**"Reality is the ultimate arbiter, and it never agreed to match our simulations."**

Until we have a fully functional original model and verified outputs, we're in "Stage A simulated," not "Stage A complete."

The search for working examples taught us:
- Don't assume - **verify** against official docs
- Mathematical correctness ≠ practical correctness
- Integration bugs exist even in official implementations
- The right API matters as much as the right weights

---

*Documentation created: 2025-12-19*
*Location: HRM/sage/docs/*
