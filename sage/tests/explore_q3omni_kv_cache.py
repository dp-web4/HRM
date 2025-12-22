#!/usr/bin/env python3
"""
Q3-Omni KV Cache Exploration Script

Goal: Understand and save KV cache structure before implementing multi-turn conversations.

Architecture (from config.json):
- Thinker (text reasoning): 48 layers, 32 attn heads, 4 KV heads, hidden_size=2048
- Talker (audio generation): 20 layers, 16 attn heads, 2 KV heads, hidden_size=1024

KV Cache Structure per layer:
- Keys: [batch, num_kv_heads, seq_len, head_dim]
- Values: [batch, num_kv_heads, seq_len, head_dim]

Research questions:
1. What is the exact structure of past_key_values returned by generate()?
2. How does the cache evolve during generation?
3. Can we save and restore cache between conversation turns?
4. What information does the cache contain about reasoning processes?
5. How does the "thinker" module use cached context for multi-turn reasoning?
"""

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
import json
import time
from pathlib import Path
import pickle

MODEL_PATH = "model-zoo/sage/omni-modal/qwen3-omni-30b"

print("=" * 80)
print("Q3-Omni KV Cache Structure Exploration")
print("=" * 80)
print()

# Load model and processor
print("Loading model (this takes ~3 minutes)...")
load_start = time.time()

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    max_memory={0: "110GB"},
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

load_time = time.time() - load_start
print(f"✅ Model loaded in {load_time:.2f}s")
print()

# Test 1: Capture KV cache from generation
print("=" * 80)
print("TEST 1: Capturing KV Cache Structure")
print("=" * 80)
print()

prompt = "Once upon a time"
messages = [{"role": "user", "content": prompt}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

print(f"Prompt: \"{prompt}\"")
print(f"Input length: {inputs['input_ids'].shape[1]} tokens")
print()

# Generate with cache
print("Generating with past_key_values capture...")
gen_start = time.time()

with torch.no_grad():
    # Try to get past_key_values from generate()
    # Q3-Omni returns (text_ids, audio) tuple, but text_ids might contain cache info
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        thinker_return_dict_in_generate=True,
        return_dict_in_generate=True,
        use_cache=True,
        output_attentions=False,  # Start simple
    )

gen_time = time.time() - gen_start

print(f"Generation complete in {gen_time:.2f}s")
print()

# Analyze outputs structure
print("=" * 80)
print("OUTPUT STRUCTURE ANALYSIS")
print("=" * 80)
print()

print(f"Type of outputs: {type(outputs)}")
print()

if isinstance(outputs, tuple):
    print(f"Tuple with {len(outputs)} elements:")
    for i, elem in enumerate(outputs):
        print(f"  [{i}] {type(elem)}")
        if hasattr(elem, 'keys'):
            print(f"      Keys: {list(elem.keys())[:10]}")  # First 10 keys
        elif hasattr(elem, 'shape'):
            print(f"      Shape: {elem.shape}")
    print()

    # Extract text_ids (usually first element)
    text_ids = outputs[0]
    if hasattr(text_ids, 'keys'):
        print("Text IDs object keys:")
        for key in text_ids.keys():
            value = text_ids[key]
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor {value.shape}")
            elif isinstance(value, tuple):
                print(f"  {key}: Tuple with {len(value)} elements")
                if len(value) > 0 and isinstance(value[0], tuple):
                    print(f"           Each element is tuple of {len(value[0])} tensors")
                    if len(value[0]) > 0:
                        print(f"           First tensor shape: {value[0][0].shape}")
            else:
                print(f"  {key}: {type(value)}")
        print()

        # Check for past_key_values
        if 'past_key_values' in text_ids:
            print("✅ FOUND: past_key_values in outputs!")
            past_kv = text_ids['past_key_values']

            print(f"Type: {type(past_kv)}")
            print(f"Number of layers: {len(past_kv)}")
            print()

            # Analyze structure
            print("KV Cache Structure:")
            for layer_idx in range(min(3, len(past_kv))):  # First 3 layers
                layer_kv = past_kv[layer_idx]
                print(f"\n  Layer {layer_idx}:")
                print(f"    Type: {type(layer_kv)}")

                if isinstance(layer_kv, tuple):
                    print(f"    Elements: {len(layer_kv)}")
                    for elem_idx, elem in enumerate(layer_kv):
                        if isinstance(elem, torch.Tensor):
                            print(f"      [{elem_idx}] Tensor shape: {elem.shape}")
                            print(f"           dtype: {elem.dtype}, device: {elem.device}")
        else:
            print("⚠️  No 'past_key_values' found in outputs")
            print("   Available keys:", list(text_ids.keys()))

        # Decode output
        if 'sequences' in text_ids:
            sequences = text_ids['sequences']
            input_len = inputs['input_ids'].shape[1]
            generated_tokens = sequences[:, input_len:]
            generated_text = processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            print()
            print("=" * 80)
            print("GENERATED TEXT")
            print("=" * 80)
            print(generated_text[:200])
            print()

elif hasattr(outputs, 'keys'):
    print("Output is dict-like with keys:")
    for key in outputs.keys():
        print(f"  {key}")


# Test 2: Incremental generation with cache reuse
print()
print("=" * 80)
print("TEST 2: Multi-Turn Conversation Simulation")
print("=" * 80)
print()

# Conversation turns
turns = [
    "Tell me a story about a dragon.",
    "What was the dragon's name?",
    "What did the dragon do?",
]

conversation_history = []
cached_state = None

for turn_idx, user_input in enumerate(turns, 1):
    print(f"\n--- Turn {turn_idx} ---")
    print(f"User: {user_input}")

    # Build conversation context
    conversation_history.append({"role": "user", "content": user_input})

    # Prepare input
    text = processor.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate with cache
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            thinker_return_dict_in_generate=True,
            return_dict_in_generate=True,
            use_cache=True,
            past_key_values=cached_state,  # Reuse previous cache if available
        )

    # Extract response
    if isinstance(outputs, tuple):
        text_ids = outputs[0]

        if hasattr(text_ids, 'sequences'):
            input_len = inputs['input_ids'].shape[1]
            generated_tokens = text_ids.sequences[:, input_len:]
            response = processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            print(f"Assistant: {response}")

            # Save response to history
            conversation_history.append({"role": "assistant", "content": response})

            # Update cached state if available
            if hasattr(text_ids, 'past_key_values'):
                cached_state = text_ids.past_key_values
                print(f"✅ Cache updated ({len(cached_state)} layers)")


# Test 3: Cache persistence
print()
print("=" * 80)
print("TEST 3: Cache Persistence and Storage")
print("=" * 80)
print()

if cached_state is not None:
    # Save cache to disk
    cache_dir = Path("kv_cache_checkpoints")
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / "q3omni_cache_example.pkl"

    print("Analyzing cache for storage...")
    total_elements = 0
    total_bytes = 0

    cache_metadata = {
        'num_layers': len(cached_state),
        'layer_info': []
    }

    for layer_idx, layer_kv in enumerate(cached_state):
        if isinstance(layer_kv, tuple):
            layer_info = {
                'layer_idx': layer_idx,
                'num_tensors': len(layer_kv),
                'tensor_shapes': [],
                'tensor_dtypes': [],
                'tensor_sizes_mb': []
            }

            for tensor in layer_kv:
                if isinstance(tensor, torch.Tensor):
                    total_elements += tensor.numel()
                    total_bytes += tensor.element_size() * tensor.numel()

                    layer_info['tensor_shapes'].append(list(tensor.shape))
                    layer_info['tensor_dtypes'].append(str(tensor.dtype))
                    layer_info['tensor_sizes_mb'].append(
                        (tensor.element_size() * tensor.numel()) / (1024 * 1024)
                    )

            cache_metadata['layer_info'].append(layer_info)

    print(f"Total elements: {total_elements:,}")
    print(f"Total size: {total_bytes / (1024 * 1024):.2f} MB")
    print()

    # Save metadata
    metadata_file = cache_dir / "cache_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(cache_metadata, f, indent=2)

    print(f"✅ Metadata saved to {metadata_file}")

    # Demonstrate cache save/load (CPU only for portability)
    print("\nTesting cache save/load to CPU...")

    # Move to CPU for saving
    cache_cpu = tuple(
        tuple(t.cpu() if isinstance(t, torch.Tensor) else t for t in layer)
        if isinstance(layer, tuple) else layer
        for layer in cached_state
    )

    with open(cache_file, 'wb') as f:
        pickle.dump(cache_cpu, f)

    cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
    print(f"✅ Cache saved to {cache_file} ({cache_size_mb:.2f} MB)")

    # Load back
    with open(cache_file, 'rb') as f:
        loaded_cache = pickle.load(f)

    print(f"✅ Cache loaded successfully ({len(loaded_cache)} layers)")
    print()

else:
    print("⚠️  No cache available from previous tests")
    print("   Model might not support past_key_values return")


# Summary
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print("Key Findings:")
print("1. Q3-Omni architecture has two main components:")
print("   - Thinker (48 layers, strategic reasoning)")
print("   - Talker (20 layers, audio generation)")
print()
print("2. Cache structure discovery:")

if cached_state is not None:
    print(f"   ✅ Successfully captured {len(cached_state)} layers of KV cache")
    print(f"   ✅ Each layer contains key/value tensors")
    print(f"   ✅ Cache can be saved to disk for persistence")
    print()
    print("3. Multi-turn conversation:")
    print("   ✅ Demonstrated conversation history management")
    print("   ✅ Cache reuse across turns (if supported)")
    print()
    print("4. Next steps:")
    print("   - Implement robust multi-turn dialogue framework")
    print("   - Analyze cache evolution patterns")
    print("   - Study 'thinker' module's use of context")
    print("   - Test recursive reasoning hypothesis")
else:
    print("   ⚠️  Could not capture KV cache from generate()")
    print("   - Model may return cache differently")
    print("   - Need to investigate Q3-Omni specific cache handling")
    print()
    print("3. Alternative approaches needed:")
    print("   - Check model forward() method for cache")
    print("   - Investigate custom Q3-Omni cache mechanisms")
    print("   - May need to use model internals directly")

print()
print("Investigation complete!")
print()
