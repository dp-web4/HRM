#!/usr/bin/env python3
"""
Quantize Qwen3-Omni-30B to FP4 precision using NVIDIA ModelOpt.

This script performs post-training quantization (PTQ) to convert the
30B parameter model from BF16/FP16 (66GB) to FP4 (~ 16GB).

Expected benefits on Jetson AGX Thor:
- Memory: 66GB → ~16GB (4x reduction)
- Speed: ~1.3 tok/s → ~9-10 tok/s (7.5x faster)
- Load time: ~3 min → ~45 seconds
"""

import torch
import json
from pathlib import Path
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from modelopt.torch.quantization import quantize, QuantizeConfig
import sys
sys.path.insert(0, str(Path(__file__).parent / "calibration_data"))


def load_calibration_data(calibration_dir: Path, max_samples: int = 64):
    """Load calibration dataset for quantization."""
    print(f"Loading calibration data from {calibration_dir}...")

    with open(calibration_dir / "calibration_dataset.json") as f:
        data = json.load(f)

    # Limit samples for faster calibration
    data = data[:max_samples]

    # Convert back to tensors
    calibration_inputs = []
    for sample in data:
        calibration_inputs.append({
            "input_ids": torch.tensor(sample["input_ids"]),
            "attention_mask": torch.tensor(sample["attention_mask"]),
        })

    print(f"Loaded {len(calibration_inputs)} calibration samples")
    return calibration_inputs


def quantize_q3omni_to_fp4(
    model_path: str = "model-zoo/sage/omni-modal/qwen3-omni-30b",
    output_path: str = "model-zoo/sage/omni-modal/qwen3-omni-30b-fp4",
    calibration_dir: str = "sage/quantization/calibration_data",
    device: str = "cuda:0"
):
    """
    Quantize Q3-Omni to FP4 format.

    Args:
        model_path: Path to original FP16/BF16 model
        output_path: Path to save quantized FP4 model
        calibration_dir: Directory containing calibration data
        device: Device to use for quantization
    """

    print("="*60)
    print("QWEN3-OMNI FP4 QUANTIZATION")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    print("="*60)

    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load calibration data
    calibration_dir = Path(calibration_dir)
    calibration_inputs = load_calibration_data(calibration_dir, max_samples=64)

    # Load model
    print("\n[1/5] Loading Q3-Omni model...")
    print("This will take ~3 minutes on Thor...")

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,  # Use BF16 for quantization process
        trust_remote_code=True,
    )

    print(f"✅ Model loaded successfully")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    # Load processor
    print("\n[2/5] Loading processor...")
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    print("✅ Processor loaded")

    # Configure FP4 quantization
    print("\n[3/5] Configuring FP4 quantization...")

    # NVIDIA FP4 quantization config
    quant_config = QuantizeConfig(
        quant_mode="fp4",  # Use NVIDIA FP4 format
        quant_algo="micro_block_fp4",  # Micro-block scaling for better accuracy
        calib_method="max",  # Max calibration for activation ranges
    )

    print(f"Quantization config:")
    print(f"  - Mode: FP4")
    print(f"  - Algorithm: Micro-block FP4 (NVIDIA optimized)")
    print(f"  - Calibration: Max activation ranges")

    # Calibration function
    def calibrate_model():
        """Run calibration forward passes."""
        print("\n[4/5] Running calibration...")
        print(f"Processing {len(calibration_inputs)} calibration samples...")

        model.eval()
        with torch.no_grad():
            for idx, inputs in enumerate(calibration_inputs):
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward pass for calibration
                try:
                    _ = model(**inputs)

                    if (idx + 1) % 10 == 0:
                        print(f"  Calibrated {idx + 1}/{len(calibration_inputs)} samples")
                except Exception as e:
                    print(f"  Warning: Calibration sample {idx} failed: {e}")
                    continue

        print("✅ Calibration complete")

    # Apply quantization
    print("\n[4/5] Applying FP4 quantization...")
    print("This will take 30-60 minutes on Thor...")

    try:
        quantized_model = quantize(
            model=model,
            config=quant_config,
            forward_loop=calibrate_model,  # Run calibration
        )

        print("✅ Model quantized successfully")

    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        raise

    # Save quantized model
    print("\n[5/5] Saving quantized model...")
    print(f"Output: {output_path}")

    try:
        quantized_model.save_pretrained(output_path, safe_serialization=True)
        processor.save_pretrained(output_path)

        print("✅ Model saved successfully")

    except Exception as e:
        print(f"❌ Save failed: {e}")
        raise

    # Print size comparison
    print("\n" + "="*60)
    print("QUANTIZATION COMPLETE")
    print("="*60)

    # Calculate sizes
    original_size = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024**3)  # GB

    quantized_size = sum(
        p.numel() * p.element_size() for p in quantized_model.parameters()
    ) / (1024**3)  # GB

    compression_ratio = original_size / quantized_size

    print(f"Original model: {original_size:.2f} GB")
    print(f"Quantized model: {quantized_size:.2f} GB")
    print(f"Compression: {compression_ratio:.2f}x")
    print(f"Savings: {original_size - quantized_size:.2f} GB")
    print("="*60)

    # Save quantization metadata
    metadata = {
        "original_model": str(model_path),
        "quantized_model": str(output_path),
        "quantization_method": "FP4 (NVIDIA ModelOpt)",
        "algorithm": "micro_block_fp4",
        "calibration_method": "max",
        "calibration_samples": len(calibration_inputs),
        "original_size_gb": float(original_size),
        "quantized_size_gb": float(quantized_size),
        "compression_ratio": float(compression_ratio),
        "expected_speedup": "7.5x (2070 TFLOPs FP4 on Thor)",
    }

    metadata_file = output_path / "quantization_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Metadata saved to {metadata_file}")

    return quantized_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize Q3-Omni to FP4")
    parser.add_argument(
        "--model-path",
        type=str,
        default="model-zoo/sage/omni-modal/qwen3-omni-30b",
        help="Path to original model"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="model-zoo/sage/omni-modal/qwen3-omni-30b-fp4",
        help="Output path for quantized model"
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default="sage/quantization/calibration_data",
        help="Calibration dataset directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use"
    )

    args = parser.parse_args()

    quantize_q3omni_to_fp4(
        model_path=args.model_path,
        output_path=args.output_path,
        calibration_dir=args.calibration_dir,
        device=args.device,
    )
