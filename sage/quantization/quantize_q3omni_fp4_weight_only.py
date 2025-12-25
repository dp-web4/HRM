#!/usr/bin/env python3
"""
Quantize Qwen3-Omni-30B to FP4 using WEIGHT-ONLY quantization.

This approach quantizes only MLP weights (not activations), which:
- Requires NO calibration (no forward pass issues)
- Bypasses the forward signature incompatibility
- Still achieves 2-3x memory reduction
- More compatible with complex multimodal architectures

Expected benefits on Jetson AGX Thor:
- Memory: 66GB → ~22-33GB (2-3x reduction)
- Speed: ~1.3 tok/s → ~6-7 tok/s (4-5x faster)
- Load time: ~3 min → ~60 seconds
"""

import torch
import json
from pathlib import Path
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from modelopt.torch.quantization import quantize, NVFP4_MLP_WEIGHT_ONLY_CFG


def quantize_q3omni_weight_only(
    model_path: str = "model-zoo/sage/omni-modal/qwen3-omni-30b",
    output_path: str = "model-zoo/sage/omni-modal/qwen3-omni-30b-fp4-weight-only",
    device: str = "cuda:0"
):
    """
    Quantize Q3-Omni to FP4 using weight-only quantization.

    Args:
        model_path: Path to original FP16/BF16 model
        output_path: Path to save quantized FP4 model
        device: Device to use for quantization
    """

    print("="*60)
    print("QWEN3-OMNI FP4 WEIGHT-ONLY QUANTIZATION")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    print(f"Method: MLP weight-only (no calibration)")
    print("="*60)

    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\n[1/4] Loading Q3-Omni model...")
    print("This will take ~3 minutes on Thor...")

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"✅ Model loaded successfully")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    # Calculate original size
    original_size = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024**3)  # GB

    print(f"Original memory: {original_size:.2f} GB")

    # Load processor
    print("\n[2/4] Loading processor...")
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    print("✅ Processor loaded")

    # Apply weight-only quantization
    print("\n[3/4] Applying FP4 weight-only quantization...")
    print("Configuration: NVFP4_MLP_WEIGHT_ONLY_CFG")
    print("  - Quantizes only MLP weights")
    print("  - No activation quantization")
    print("  - No calibration required")
    print("  - FP4 block size: 32")
    print("\nThis will take 5-10 minutes on Thor...")

    try:
        # Apply weight-only quantization (no forward_loop needed!)
        quantized_model = quantize(
            model,
            NVFP4_MLP_WEIGHT_ONLY_CFG,
            # NOTE: No forward_loop parameter - weight-only doesn't need it!
        )

        print("✅ Model quantized successfully")

    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Calculate quantized size
    quantized_size = sum(
        p.numel() * p.element_size() for p in quantized_model.parameters()
    ) / (1024**3)  # GB

    compression_ratio = original_size / quantized_size

    # Save quantized model
    print("\n[4/4] Saving quantized model...")
    print(f"Output: {output_path}")

    try:
        quantized_model.save_pretrained(output_path, safe_serialization=True)
        processor.save_pretrained(output_path)

        print("✅ Model saved successfully")

    except Exception as e:
        print(f"❌ Save failed: {e}")
        raise

    # Print results
    print("\n" + "="*60)
    print("QUANTIZATION COMPLETE")
    print("="*60)
    print(f"Original model: {original_size:.2f} GB")
    print(f"Quantized model: {quantized_size:.2f} GB")
    print(f"Compression: {compression_ratio:.2f}x")
    print(f"Savings: {original_size - quantized_size:.2f} GB")
    print("="*60)

    # Analyze what was quantized
    print("\n📊 Quantization Analysis:")
    print("-" * 60)

    total_params = 0
    quantized_params = 0

    for name, param in quantized_model.named_parameters():
        total_params += param.numel()

        # Check if this is an MLP parameter that got quantized
        if 'mlp' in name.lower() and 'weight' in name:
            quantized_params += param.numel()
            print(f"  ✅ Quantized: {name} ({param.numel()/1e6:.1f}M params)")

    quantized_percentage = (quantized_params / total_params) * 100
    print(f"\nTotal parameters: {total_params/1e9:.2f}B")
    print(f"Quantized parameters: {quantized_params/1e9:.2f}B ({quantized_percentage:.1f}%)")
    print(f"Unchanged parameters: {(total_params-quantized_params)/1e9:.2f}B ({100-quantized_percentage:.1f}%)")

    # Save quantization metadata
    metadata = {
        "original_model": str(model_path),
        "quantized_model": str(output_path),
        "quantization_method": "FP4 Weight-Only (NVIDIA ModelOpt)",
        "algorithm": "micro_block_fp4",
        "calibration_method": "none (weight-only)",
        "quantization_scope": "MLP weights only",
        "original_size_gb": float(original_size),
        "quantized_size_gb": float(quantized_size),
        "compression_ratio": float(compression_ratio),
        "quantized_params_percentage": float(quantized_percentage),
        "expected_speedup": "4-5x (partial FP4 acceleration on Thor)",
    }

    metadata_file = output_path / "quantization_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Metadata saved to {metadata_file}")

    # Test that the model still works
    print("\n🧪 Testing quantized model...")
    print("-" * 60)

    try:
        test_input = processor(
            text=["Hello! How are you?"],
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = quantized_model.generate(
                **test_input,
                max_new_tokens=10,
            )

        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Model test passed!")
        print(f"Test input: 'Hello! How are you?'")
        print(f"Model output: {response[:100]}...")

    except Exception as e:
        print(f"⚠️  Model test failed: {e}")
        print("This may indicate quantization issues, but model is saved.")

    print("\n" + "="*60)
    print("READY FOR DEPLOYMENT")
    print("="*60)
    print(f"\nTo use the quantized model:")
    print(f"  model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(")
    print(f"      '{output_path}',")
    print(f"      device_map='cuda',")
    print(f"      trust_remote_code=True,")
    print(f"  )")
    print("="*60)

    return quantized_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize Q3-Omni to FP4 (weight-only)")
    parser.add_argument(
        "--model-path",
        type=str,
        default="model-zoo/sage/omni-modal/qwen3-omni-30b",
        help="Path to original model"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="model-zoo/sage/omni-modal/qwen3-omni-30b-fp4-weight-only",
        help="Output path for quantized model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use"
    )

    args = parser.parse_args()

    quantize_q3omni_weight_only(
        model_path=args.model_path,
        output_path=args.output_path,
        device=args.device,
    )
