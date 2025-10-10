#!/usr/bin/env python3
"""
Simplified GR00T Feature Extractor

Direct model access - bypasses policy/transform complexity.
Extracts vision-language features from backbone for SAGE distillation.

NO SHORTCUTS - Uses real NVIDIA GR00T N1.5 model.
"""

import sys
sys.path.insert(0, "/home/dp/ai-workspace/isaac-gr00t")

import torch
import numpy as np
from typing import Dict, List, Union, Optional
from pathlib import Path
from PIL import Image

from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature


class SimpleGR00TExtractor:
    """
    Simplified GR00T feature extractor for SAGE distillation.

    Uses direct model access to extract backbone features without
    the complexity of robotics policy wrappers.
    """

    def __init__(
        self,
        model_path: str = "nvidia/GR00T-N1.5-3B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize simplified GR00T extractor.

        Args:
            model_path: HuggingFace model path
            device: Device to run on
            cache_dir: Cache directory for weights
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "huggingface" / "hub")

        print(f"🚀 Loading GR00T Feature Extractor")
        print(f"   Model: {model_path}")
        print(f"   Device: {device}")

        # Load Eagle processor for input preparation
        print("📦 Loading Eagle processor...")
        self.processor = AutoProcessor.from_pretrained(
            DEFAULT_EAGLE_PATH,
            trust_remote_code=True,
            use_fast=True,
        )
        self.processor.tokenizer.padding_side = "left"

        # Load GR00T model directly
        print("📦 Loading GR00T model...")
        self.model = GR00T_N1_5.from_pretrained(
            model_path,
            cache_dir=self.cache_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            tune_visual=False,
            tune_llm=False,
            tune_projector=False,
            tune_diffusion_model=False,
        )
        self.model.eval()

        print(f"✅ GR00T loaded successfully!")
        print(f"   Backbone params: {sum(p.numel() for p in self.model.backbone.parameters()) / 1e9:.2f}B")
        print(f"   Total params: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
        print(f"   Feature dim: {self.get_feature_dim()}")

    def get_feature_dim(self) -> int:
        """Get the dimensionality of extracted features."""
        # Eagle linear projects 2048 → project_to_dim (default 1536)
        return 1536

    def prepare_inputs(
        self,
        images: Union[List[Image.Image], List[np.ndarray]],
        text: str,
        max_length: int = 512,
    ) -> BatchFeature:
        """
        Prepare inputs for GR00T backbone.

        Args:
            images: List of PIL Images or numpy arrays [H, W, 3]
            text: Text instruction/prompt
            max_length: Maximum sequence length

        Returns:
            BatchFeature with eagle_* prefixed keys
        """
        # Convert numpy arrays to PIL if needed
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img.astype(np.uint8)))
            else:
                pil_images.append(img)

        # Process with Eagle processor
        processed = self.processor(
            text=[text],
            images=pil_images,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

        # Add eagle_ prefix to all keys
        eagle_inputs = {f"eagle_{k}": v.to(self.device) for k, v in processed.items()}

        # Add image_sizes (required by forward_eagle)
        if "eagle_pixel_values" in eagle_inputs:
            pixel_values = eagle_inputs["eagle_pixel_values"]
            # Get dimensions from pixel_values
            if len(pixel_values.shape) >= 2:
                h, w = pixel_values.shape[-2:]
                batch_size = pixel_values.shape[0]
                eagle_inputs["eagle_image_sizes"] = torch.tensor(
                    [[h, w]] * batch_size,
                    device=self.device,
                    dtype=torch.long,
                )

        return BatchFeature(data=eagle_inputs)

    def extract_features(
        self,
        images: Union[List[Image.Image], List[np.ndarray]],
        text: str,
        return_attention: bool = True,
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Extract vision-language features from GR00T backbone.

        Args:
            images: Input images
            text: Text instruction
            return_attention: Whether to return attention mask

        Returns:
            Dictionary with:
            - features: [batch, seq_len, feature_dim] embeddings
            - attention_mask: [batch, seq_len] mask (if return_attention=True)
        """
        # Prepare inputs
        inputs = self.prepare_inputs(images, text)

        # Extract features from backbone
        with torch.no_grad():
            backbone_outputs = self.model.backbone(inputs)

        # Get features and attention
        features = backbone_outputs["backbone_features"]  # [B, seq_len, 1536]

        result = {"features": features}

        if return_attention:
            attention_mask = backbone_outputs.get("backbone_attention_mask")
            if attention_mask is not None:
                result["attention_mask"] = attention_mask

        return result

    def extract_features_numpy(
        self,
        images: Union[List[Image.Image], List[np.ndarray]],
        text: str,
    ) -> Dict[str, np.ndarray]:
        """
        Extract features and return as numpy arrays.

        Args:
            images: Input images
            text: Text instruction

        Returns:
            Dictionary with numpy arrays
        """
        result = self.extract_features(images, text, return_attention=True)

        return {
            k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in result.items()
        }

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "feature_dim": self.get_feature_dim(),
            "backbone_params": sum(p.numel() for p in self.model.backbone.parameters()),
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "dtype": next(self.model.parameters()).dtype,
        }


def test_extractor():
    """Test the simplified extractor."""
    print("=" * 80)
    print("Testing Simplified GR00T Feature Extractor")
    print("=" * 80)

    # Initialize extractor
    extractor = SimpleGR00TExtractor()

    # Print model info
    print("\n📊 Model Information:")
    info = extractor.get_model_info()
    for key, value in info.items():
        if isinstance(value, int) and value > 1000:
            print(f"   {key}: {value / 1e9:.2f}B")
        else:
            print(f"   {key}: {value}")

    # Create test image
    print("\n🖼️  Creating test image...")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_text = "Pick up the red cube and place it in the box"

    # Extract features
    print(f"🔬 Extracting features for: '{test_text}'")
    features_dict = extractor.extract_features_numpy(
        images=[test_image],
        text=test_text,
    )

    print("\n📦 Extracted Features:")
    for key, value in features_dict.items():
        if isinstance(value, np.ndarray):
            print(f"   {key}:")
            print(f"      shape: {value.shape}")
            print(f"      dtype: {value.dtype}")
            print(f"      range: [{value.min():.4f}, {value.max():.4f}]")
            print(f"      mean: {value.mean():.4f}")
            print(f"      std: {value.std():.4f}")

    print("\n✅ Feature extraction test successful!")
    print("   Ready for SAGE distillation")

    return extractor, features_dict


if __name__ == "__main__":
    extractor, features = test_extractor()
