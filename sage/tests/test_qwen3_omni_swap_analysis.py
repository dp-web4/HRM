#!/usr/bin/env python3
"""
Qwen3-Omni-30B FP16 with Swap Analysis

Monitor actual resource usage patterns, swap behavior, and latency impact.
This is a research test to understand MoE expert activation patterns.
"""

from pathlib import Path
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import torch
import gc
import psutil
import time
import threading

# Global monitoring data
monitoring_data = {
    'running': False,
    'samples': []
}

def monitor_resources():
    """Background thread to monitor memory and swap usage"""
    process = psutil.Process()

    while monitoring_data['running']:
        mem_info = process.memory_info()
        swap_info = psutil.swap_memory()
        vm = psutil.virtual_memory()

        sample = {
            'timestamp': time.time(),
            'rss_gb': mem_info.rss / 1024**3,
            'vms_gb': mem_info.vms / 1024**3,
            'swap_used_gb': swap_info.used / 1024**3,
            'swap_percent': swap_info.percent,
            'ram_available_gb': vm.available / 1024**3,
            'ram_percent': vm.percent,
        }

        monitoring_data['samples'].append(sample)
        time.sleep(0.5)  # Sample every 500ms

def print_resource_summary():
    """Print summary of resource usage"""
    if not monitoring_data['samples']:
        return

    print("\n" + "="*70)
    print("Resource Usage Analysis")
    print("="*70)

    samples = monitoring_data['samples']

    # Find peak values
    peak_rss = max(s['rss_gb'] for s in samples)
    peak_swap = max(s['swap_used_gb'] for s in samples)
    min_ram_avail = min(s['ram_available_gb'] for s in samples)

    # Find when swap started
    swap_start = next((s for s in samples if s['swap_used_gb'] > 0.1), None)

    print(f"\nPeak Memory Usage:")
    print(f"  RSS: {peak_rss:.1f} GB")
    print(f"  Swap: {peak_swap:.1f} GB")
    print(f"  Total: {peak_rss + peak_swap:.1f} GB")
    print(f"  Min RAM Available: {min_ram_avail:.1f} GB")

    if swap_start:
        swap_time = swap_start['timestamp'] - samples[0]['timestamp']
        print(f"\nSwap Started:")
        print(f"  Time: {swap_time:.1f}s after start")
        print(f"  RAM at swap start: {swap_start['rss_gb']:.1f} GB")

    # Growth rate analysis
    if len(samples) > 10:
        # Skip first few samples (noise)
        start_idx = 5
        end_idx = min(len(samples) - 5, len(samples))

        if end_idx > start_idx:
            time_delta = samples[end_idx]['timestamp'] - samples[start_idx]['timestamp']
            mem_delta = samples[end_idx]['rss_gb'] - samples[start_idx]['rss_gb']

            if time_delta > 0:
                growth_rate = mem_delta / time_delta
                print(f"\nMemory Growth Rate:")
                print(f"  {growth_rate:.2f} GB/sec during loading")

    print("\n" + "="*70)

def test_qwen3_omni_with_swap():
    print("="*70)
    print("Qwen3-Omni-30B FP16 with Swap Analysis")
    print("="*70)
    print()

    MODEL_PATH = "model-zoo/sage/omni-modal/qwen3-omni-30b"

    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        return False

    print("Configuration:")
    print("  - FP16 model (70.5GB weights)")
    print("  - 122GB RAM + 150GB swap = 272GB total")
    print("  - Swappiness: 10 (aggressive RAM preference)")
    print("  - NVMe swap for fast paging")
    print()
    print("Research Goals:")
    print("  1. Observe actual memory + swap usage")
    print("  2. Measure latency impact of swapping")
    print("  3. Identify expert activation patterns")
    print("  4. Understand resource usage for SAGE integration")
    print()

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Start monitoring
    monitoring_data['running'] = True
    monitoring_data['samples'] = []
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    start_time = time.time()

    try:
        print("="*70)
        print("Phase 1: Model Loading")
        print("="*70)
        print()

        load_start = time.time()

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype="auto",
            device_map="cuda",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        load_time = time.time() - load_start
        print(f"\n✅ Model loaded in {load_time:.1f}s")

        processor = Qwen3OmniMoeProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        print("✅ Processor loaded")
        print()

        # Give monitoring a moment to catch up
        time.sleep(2)

        print("="*70)
        print("Phase 2: Inference Test")
        print("="*70)
        print()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Explain what a Mixture of Experts architecture is in one sentence."}
                ],
            },
        ]

        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
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
        inputs = inputs.to(model.device)

        print("Generating response (monitoring expert activation patterns)...")
        print()

        gen_start = time.time()

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            return_audio=False,
            use_audio_in_video=False
        )

        gen_time = time.time() - gen_start

        response = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print(f"Prompt: {conversation[0]['content'][0]['text']}")
        print(f"Response: {response[0]}")
        print()
        print(f"Generation time: {gen_time:.2f}s")
        print(f"Tokens/sec: {100/gen_time:.1f}")
        print()

        total_time = time.time() - start_time

        # Stop monitoring
        monitoring_data['running'] = False
        time.sleep(1)  # Let thread finish

        # Print analysis
        print_resource_summary()

        print("\n" + "="*70)
        print("✅ Test Complete - SUCCESS!")
        print("="*70)
        print(f"\nTotal time: {total_time:.1f}s")
        print(f"  Loading: {load_time:.1f}s")
        print(f"  Generation: {gen_time:.2f}s")

        return True

    except Exception as e:
        monitoring_data['running'] = False
        time.sleep(1)

        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        print_resource_summary()
        return False

if __name__ == "__main__":
    success = test_qwen3_omni_with_swap()
    exit(0 if success else 1)
