# Q3-Omni INT8 Failure Analysis
**Date**: December 26, 2025
**Issue**: INT8 AWQ model hung during loading (12+ minutes, no progress)

## What Worked Before

### Successful Test: `test_qwen3_omni_simple_text.py`

**Model Used**: `model-zoo/sage/omni-modal/qwen3-omni-30b` (FULL PRECISION)
**Status**: ✅ Worked, loaded in ~2 minutes, generated text successfully

**Key Parameters**:
```python
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    "model-zoo/sage/omni-modal/qwen3-omni-30b",
    dtype="auto",                    # Auto dtype
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.disable_talker()  # CRITICAL

processor = Qwen3OmniMoeProcessor.from_pretrained(
    "model-zoo/sage/omni-modal/qwen3-omni-30b",
    trust_remote_code=True
)

# Message format: NESTED structure
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello!"}
        ],
    },
]

# Processing: Uses process_mm_info helper
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pt",
    padding=True,
    use_audio_in_video=False
)

# Generate
generated_ids = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    return_audio=False,
    use_audio_in_video=False
)

# Decode
response = processor.batch_decode(
    generated_ids[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
```

## What We Changed (That Broke It)

### Failed Test: Our IRP Plugin

**Model Used**: `model-zoo/sage/omni-modal/qwen3-omni-30b-int8-awq` (INT8 QUANTIZED)
**Status**: ❌ Hung during loading, 12+ minutes with no output

**Key Differences**:

### 1. Model Path Changed
- ✅ Working: `qwen3-omni-30b` (full precision, ~60GB)
- ❌ Failed: `qwen3-omni-30b-int8-awq` (INT8, ~40GB)

### 2. Model Loading Parameters
- ✅ Working: `dtype="auto"`
- ❌ Failed: `torch_dtype=torch.float16`

**ISSUE**: INT8 AWQ model shouldn't use `torch.float16`! It's already quantized to INT8.

### 3. Message Format
- ✅ Working: Nested content structure `{"role": "user", "content": [{"type": "text", "text": "..."}]}`
- ❌ Failed: Simple string structure `{"role": "user", "content": "..."}`

**ISSUE**: Q3-Omni expects nested content with type annotation!

### 4. Input Processing
- ✅ Working: Uses `process_mm_info()` helper to extract audios/images/videos
- ❌ Failed: Direct `processor(text=...)` without multimodal extraction

**ISSUE**: Even for text-only, Q3-Omni needs the multimodal processing pipeline!

### 5. Generation Parameters
- ✅ Working: NO `thinker_return_dict_in_generate` parameter
- ❌ Failed: Added `thinker_return_dict_in_generate=True`

**ISSUE**: This might not be compatible with INT8 AWQ or may cause issues!

## Root Causes Identified

### PRIMARY: Wrong dtype for quantized model
```python
# WRONG (what we did):
torch_dtype=torch.float16  # Trying to load INT8 model as FP16!

# RIGHT (what works):
dtype="auto"  # Let transformers detect it's INT8
```

### SECONDARY: Missing multimodal pipeline
```python
# WRONG (what we did):
messages = [{"role": "user", "content": "text"}]
inputs = processor(text=formatted_prompt, return_tensors="pt")

# RIGHT (what works):
conversation = [{"role": "user", "content": [{"type": "text", "text": "..."}]}]
text = processor.apply_chat_template(conversation, ...)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
inputs = processor(text=text, audio=audios, images=images, videos=videos, ...)
```

### TERTIARY: Unnecessary thinker parameter
```python
# WRONG (what we added):
thinker_return_dict_in_generate=True  # Might break INT8 AWQ

# RIGHT (what works):
# Don't include this parameter - it's not needed for simple generation
```

## The Fix

### Updated IRP Plugin Requirements

1. **Use FULL PRECISION model** (not INT8)
   - Path: `model-zoo/sage/omni-modal/qwen3-omni-30b`
   - The full precision model loads in ~2 minutes and works

2. **Use `dtype="auto"`** not `torch_dtype=torch.float16`

3. **Import and use `process_mm_info` helper**:
   ```python
   from qwen_omni_utils import process_mm_info
   ```

4. **Change message format to nested structure**:
   ```python
   # Convert from simple format
   messages = [{"role": "user", "content": "Hello"}]

   # To nested format
   conversation = [
       {
           "role": "user",
           "content": [{"type": "text", "text": "Hello"}]
       }
   ]
   ```

5. **Use full multimodal processing pipeline**:
   ```python
   text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
   audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
   inputs = processor(
       text=text,
       audio=audios,
       images=images,
       videos=videos,
       return_tensors="pt",
       padding=True,
       use_audio_in_video=False
   )
   ```

6. **Remove `thinker_return_dict_in_generate` parameter**

7. **Use standard decoding**:
   ```python
   response = processor.batch_decode(
       generated_ids[:, inputs["input_ids"].shape[1]:],
       skip_special_tokens=True,
       clean_up_tokenization_spaces=False
   )
   ```

## Lessons Learned

1. **Quantized models need different loading params** - INT8 AWQ ≠ FP16
2. **Q3-Omni has a specific processing pipeline** - Can't simplify it
3. **Test with working model first** - Use full precision qwen3-omni-30b
4. **Follow proven patterns exactly** - Our "improvements" broke it
5. **Message format matters** - Nested content structure is required

## Next Steps

1. ✅ Fix IRP plugin to match working pattern exactly
2. ✅ Use full precision model (qwen3-omni-30b)
3. ✅ Import process_mm_info helper
4. ✅ Test with single turn first
5. ⬜ Then test 5-turn dragon story
