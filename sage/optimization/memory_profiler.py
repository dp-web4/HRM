#!/usr/bin/env python3
"""
SAGE Memory Profiler

Measures memory footprint of all SAGE components to guide Nano optimization.

Critical for 8GB Jetson Nano deployment.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import psutil
import gc
from typing import Dict, Any
import time

from sage.compression.vision_puzzle_vae import VisionPuzzleVAE
from sage.compression.audio_puzzle_vae import AudioPuzzleVAE
from sage.compression.language_puzzle_transformer import LanguagePuzzleTransformer
from sage.core.unified_sage_system import UnifiedSAGESystem


def get_memory_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_mb():
    """Get GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    """Estimate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


class SAGEMemoryProfiler:
    """Profile SAGE memory usage across all components"""

    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.results = {}

    def profile_models(self):
        """Profile individual model memory"""
        print("=" * 70)
        print("SAGE Memory Profile - Model Sizes")
        print("=" * 70)

        models = {}

        # Vision VAE
        print("\n1. Vision Puzzle VAE")
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        mem_before = get_memory_mb()
        gpu_before = get_gpu_memory_mb()

        vision_vae = VisionPuzzleVAE(latent_dim=64, num_codes=10).to(self.device)
        models['vision_vae'] = vision_vae

        mem_after = get_memory_mb()
        gpu_after = get_gpu_memory_mb()

        params = count_parameters(vision_vae)
        size = model_size_mb(vision_vae)

        print(f"   Parameters: {params:,}")
        print(f"   Model size: {size:.2f} MB")
        print(f"   RAM delta: {mem_after - mem_before:.2f} MB")
        if self.device == "cuda":
            print(f"   GPU delta: {gpu_after - gpu_before:.2f} MB")

        self.results['vision_vae'] = {
            'params': params,
            'size_mb': size,
            'ram_mb': mem_after - mem_before,
            'gpu_mb': gpu_after - gpu_before if self.device == "cuda" else 0
        }

        # Audio VAE
        print("\n2. Audio Puzzle VAE")
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        mem_before = get_memory_mb()
        gpu_before = get_gpu_memory_mb()

        audio_vae = AudioPuzzleVAE(latent_dim=64, num_codes=10).to(self.device)
        models['audio_vae'] = audio_vae

        mem_after = get_memory_mb()
        gpu_after = get_gpu_memory_mb()

        params = count_parameters(audio_vae)
        size = model_size_mb(audio_vae)

        print(f"   Parameters: {params:,}")
        print(f"   Model size: {size:.2f} MB")
        print(f"   RAM delta: {mem_after - mem_before:.2f} MB")
        if self.device == "cuda":
            print(f"   GPU delta: {gpu_after - gpu_before:.2f} MB")

        self.results['audio_vae'] = {
            'params': params,
            'size_mb': size,
            'ram_mb': mem_after - mem_before,
            'gpu_mb': gpu_after - gpu_before if self.device == "cuda" else 0
        }

        # Language Transformer
        print("\n3. Language Puzzle Transformer")
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        mem_before = get_memory_mb()
        gpu_before = get_gpu_memory_mb()

        lang_transformer = LanguagePuzzleTransformer(device=self.device)
        models['lang_transformer'] = lang_transformer

        mem_after = get_memory_mb()
        gpu_after = get_gpu_memory_mb()

        params = count_parameters(lang_transformer)
        size = model_size_mb(lang_transformer)

        print(f"   Parameters: {params:,}")
        print(f"   Model size: {size:.2f} MB")
        print(f"   RAM delta: {mem_after - mem_before:.2f} MB")
        if self.device == "cuda":
            print(f"   GPU delta: {gpu_after - gpu_before:.2f} MB")

        self.results['lang_transformer'] = {
            'params': params,
            'size_mb': size,
            'ram_mb': mem_after - mem_before,
            'gpu_mb': gpu_after - gpu_before if self.device == "cuda" else 0
        }

        return models

    def profile_inference(self, models):
        """Profile memory during inference"""
        print("\n" + "=" * 70)
        print("SAGE Memory Profile - Inference")
        print("=" * 70)

        # Vision inference
        print("\n1. Vision Inference (single frame)")
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        mem_before = get_memory_mb()
        gpu_before = get_gpu_memory_mb()

        with torch.no_grad():
            image = torch.rand(1, 3, 224, 224, device=self.device)
            puzzle = models['vision_vae'].encode_to_puzzle(image)

        mem_after = get_memory_mb()
        gpu_after = get_gpu_memory_mb()

        if self.device == "cuda":
            peak_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024

        print(f"   RAM delta: {mem_after - mem_before:.2f} MB")
        if self.device == "cuda":
            print(f"   GPU delta: {gpu_after - gpu_before:.2f} MB")
            print(f"   GPU peak: {peak_gpu:.2f} MB")

        self.results['vision_inference'] = {
            'ram_mb': mem_after - mem_before,
            'gpu_mb': gpu_after - gpu_before if self.device == "cuda" else 0,
            'gpu_peak_mb': peak_gpu if self.device == "cuda" else 0
        }

        # Audio inference
        print("\n2. Audio Inference (1 second)")
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        mem_before = get_memory_mb()
        gpu_before = get_gpu_memory_mb()

        with torch.no_grad():
            waveform = torch.randn(1, 16000, device=self.device)
            puzzle = models['audio_vae'].encode_to_puzzle(waveform)

        mem_after = get_memory_mb()
        gpu_after = get_gpu_memory_mb()

        if self.device == "cuda":
            peak_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024

        print(f"   RAM delta: {mem_after - mem_before:.2f} MB")
        if self.device == "cuda":
            print(f"   GPU delta: {gpu_after - gpu_before:.2f} MB")
            print(f"   GPU peak: {peak_gpu:.2f} MB")

        self.results['audio_inference'] = {
            'ram_mb': mem_after - mem_before,
            'gpu_mb': gpu_after - gpu_before if self.device == "cuda" else 0,
            'gpu_peak_mb': peak_gpu if self.device == "cuda" else 0
        }

        # Language inference
        print("\n3. Language Inference (1 sentence)")
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        mem_before = get_memory_mb()
        gpu_before = get_gpu_memory_mb()

        with torch.no_grad():
            text = ["The quick brown fox jumps over the lazy dog."]
            puzzle = models['lang_transformer'].encode_to_puzzle(text)

        mem_after = get_memory_mb()
        gpu_after = get_gpu_memory_mb()

        if self.device == "cuda":
            peak_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024

        print(f"   RAM delta: {mem_after - mem_before:.2f} MB")
        if self.device == "cuda":
            print(f"   GPU delta: {gpu_after - gpu_before:.2f} MB")
            print(f"   GPU peak: {peak_gpu:.2f} MB")

        self.results['lang_inference'] = {
            'ram_mb': mem_after - mem_before,
            'gpu_mb': gpu_after - gpu_before if self.device == "cuda" else 0,
            'gpu_peak_mb': peak_gpu if self.device == "cuda" else 0
        }

    def profile_full_loop(self):
        """Profile full SAGE consciousness loop"""
        print("\n" + "=" * 70)
        print("SAGE Memory Profile - Full Loop")
        print("=" * 70)

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        mem_before = get_memory_mb()
        gpu_before = get_gpu_memory_mb()

        # Simulate sensor data generation
        def generate_vision():
            return {'vision': torch.rand(1, 3, 224, 224, device=self.device)}

        # Simple action handler
        def handle_action(obs, stance):
            from sage.core.unified_sage_system import ExecutionResult
            return ExecutionResult(success=True, reward=0.8, description="Test", outputs={})

        # Create minimal SAGE system
        print("\nCreating UnifiedSAGE system...")
        sage = UnifiedSAGESystem(
            sensor_sources={'test': generate_vision},
            action_handlers={'test': handle_action},
            config={'initial_atp': 100.0, 'enable_circadian': False},
            enable_logging=False
        )

        mem_after_init = get_memory_mb()
        gpu_after_init = get_gpu_memory_mb()

        print(f"   System init RAM: {mem_after_init - mem_before:.2f} MB")
        if self.device == "cuda":
            print(f"   System init GPU: {gpu_after_init - gpu_before:.2f} MB")

        # Run cycles
        print("\nRunning 100 cycles...")
        start = time.time()
        sage.run(max_cycles=100, cycle_delay=0.0)
        duration = time.time() - start

        mem_after_run = get_memory_mb()
        gpu_after_run = get_gpu_memory_mb()

        if self.device == "cuda":
            peak_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024

        print(f"\n   Total duration: {duration:.2f}s ({duration/100*1000:.2f}ms/cycle)")
        print(f"   Final RAM: {mem_after_run:.2f} MB")
        print(f"   RAM growth: {mem_after_run - mem_after_init:.2f} MB")
        if self.device == "cuda":
            print(f"   Final GPU: {gpu_after_run:.2f} MB")
            print(f"   GPU peak: {peak_gpu:.2f} MB")

        self.results['full_loop'] = {
            'init_ram_mb': mem_after_init - mem_before,
            'final_ram_mb': mem_after_run,
            'ram_growth_mb': mem_after_run - mem_after_init,
            'init_gpu_mb': gpu_after_init - gpu_before if self.device == "cuda" else 0,
            'final_gpu_mb': gpu_after_run if self.device == "cuda" else 0,
            'gpu_peak_mb': peak_gpu if self.device == "cuda" else 0,
            'ms_per_cycle': duration / 100 * 1000
        }

    def print_summary(self):
        """Print optimization recommendations"""
        print("\n" + "=" * 70)
        print("NANO DEPLOYMENT ANALYSIS")
        print("=" * 70)

        # Total model size
        total_model_mb = sum(
            self.results[k]['size_mb']
            for k in ['vision_vae', 'audio_vae', 'lang_transformer']
        )

        # Peak GPU
        peak_gpu = max(
            self.results.get('vision_inference', {}).get('gpu_peak_mb', 0),
            self.results.get('audio_inference', {}).get('gpu_peak_mb', 0),
            self.results.get('lang_inference', {}).get('gpu_peak_mb', 0),
            self.results.get('full_loop', {}).get('gpu_peak_mb', 0)
        )

        # Estimated total
        final_ram = self.results.get('full_loop', {}).get('final_ram_mb', 0)

        print(f"\nModel Sizes:")
        print(f"   Total models: {total_model_mb:.2f} MB")
        print(f"   Vision VAE: {self.results['vision_vae']['size_mb']:.2f} MB")
        print(f"   Audio VAE: {self.results['audio_vae']['size_mb']:.2f} MB")
        print(f"   Language: {self.results['lang_transformer']['size_mb']:.2f} MB")

        print(f"\nPeak Memory Usage:")
        print(f"   RAM: {final_ram:.2f} MB")
        if self.device == "cuda":
            print(f"   GPU: {peak_gpu:.2f} MB")

        print(f"\nNano Compatibility (8 GB unified):")
        estimated_total = max(final_ram, peak_gpu) if self.device == "cuda" else final_ram
        print(f"   Estimated usage: {estimated_total:.0f} MB ({estimated_total/1024:.2f} GB)")
        print(f"   Safety margin: {8192 - estimated_total:.0f} MB ({(8192-estimated_total)/1024:.2f} GB)")

        if estimated_total < 6000:
            print(f"   Status: ✓ EXCELLENT - Plenty of headroom")
        elif estimated_total < 7000:
            print(f"   Status: ✓ GOOD - Should work fine")
        elif estimated_total < 8000:
            print(f"   Status: ⚠ TIGHT - Need optimization")
        else:
            print(f"   Status: ✗ TOO LARGE - Requires optimization")

        print(f"\nOptimization Recommendations:")

        if total_model_mb > 100:
            print(f"   1. FP16 quantization: {total_model_mb:.0f} → {total_model_mb/2:.0f} MB")

        if self.results.get('full_loop', {}).get('ram_growth_mb', 0) > 10:
            print(f"   2. Memory leak detected - investigate growth")

        print(f"   3. Lazy loading: Keep only active models in memory")
        print(f"   4. Streaming inference: Process one modality at a time")

        print(f"\nPerformance:")
        ms_per_cycle = self.results.get('full_loop', {}).get('ms_per_cycle', 0)
        print(f"   Cycle time: {ms_per_cycle:.2f} ms")
        print(f"   FPS: {1000/ms_per_cycle:.1f}")

        if ms_per_cycle < 66:
            print(f"   Status: ✓ EXCELLENT - Real-time capable (>15 FPS)")
        elif ms_per_cycle < 100:
            print(f"   Status: ✓ GOOD - Real-time (>10 FPS)")
        elif ms_per_cycle < 200:
            print(f"   Status: ⚠ ACCEPTABLE - Usable (>5 FPS)")
        else:
            print(f"   Status: ✗ TOO SLOW - Need optimization")


def main():
    print("SAGE Memory Profiler")
    print("Target: Jetson Nano (8 GB unified memory)")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Profiling on: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    profiler = SAGEMemoryProfiler(device=device)

    # Profile models
    models = profiler.profile_models()

    # Profile inference
    profiler.profile_inference(models)

    # Profile full loop
    profiler.profile_full_loop()

    # Print summary
    profiler.print_summary()

    print("\n" + "=" * 70)
    print("Profiling complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
