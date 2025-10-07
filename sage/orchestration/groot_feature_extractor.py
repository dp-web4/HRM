#!/usr/bin/env python3
"""
GR00T Feature Extractor for SAGE Knowledge Distillation

Uses the REAL NVIDIA GR00T N1.5 3B model to extract vision-language features.
These features serve as teacher signals for SAGE student model training.

NO SHORTCUTS - Using actual GR00T model with correct API.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image

from transformers import AutoProcessor
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH


class GR00TFeatureExtractor:
    """
    Extract vision-language features from real GR00T model for SAGE distillation.

    This uses the actual NVIDIA GR00T N1.5 model - no mocks, no shortcuts.
    """

    def __init__(
        self,
        model_name: str = "nvidia/GR00T-N1.5-3B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize GR00T feature extractor with real model.

        Args:
            model_name: HuggingFace model name or local path
            device: Device to run model on
            cache_dir: Cache directory for model weights
        """
        self.device = torch.device(device)
        self.model_name = model_name
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "huggingface" / "hub")

        print(f"🚀 Loading REAL GR00T model: {model_name}")
        print(f"   Device: {device}")
        print(f"   Cache: {self.cache_dir}")

        # Load the Eagle processor for input preparation
        self.eagle_processor = AutoProcessor.from_pretrained(
            DEFAULT_EAGLE_PATH,
            trust_remote_code=True,
            use_fast=True
        )
        self.eagle_processor.tokenizer.padding_side = "left"

        # Load the actual GR00T model
        self.model = GR00T_N1_5.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            tune_visual=False,  # Don't tune, just extract features
            tune_llm=False,
            tune_projector=False,
            tune_diffusion_model=False,
        )
        self.model.eval()

        print(f"✅ GR00T model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
        print(f"   Backbone: {type(self.model.backbone).__name__}")
        print(f"   Action Head: {type(self.model.action_head).__name__}")

    def prepare_eagle_inputs(
        self,
        images: List[Image.Image],
        text: str,
        padding: str = "max_length",
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for Eagle backbone using the processor.

        Args:
            images: List of PIL images
            text: Text prompt/instruction
            padding: Padding strategy
            max_length: Maximum sequence length

        Returns:
            Dictionary with eagle_* prefixed inputs
        """
        # Process with Eagle processor
        processed = self.eagle_processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=padding,
            max_length=max_length,
            truncation=True,
        )

        # Add eagle_ prefix to all keys
        eagle_inputs = {f"eagle_{k}": v.to(self.device) for k, v in processed.items()}

        return eagle_inputs

    def extract_backbone_features(
        self,
        images: List[Image.Image],
        text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract vision-language features from GR00T backbone.

        Args:
            images: List of PIL images
            text: Text instruction/annotation

        Returns:
            Tuple of (features, attention_mask)
            - features: [batch, seq_len, hidden_dim] backbone embeddings
            - attention_mask: [batch, seq_len] attention mask
        """
        # Prepare inputs
        eagle_inputs = self.prepare_eagle_inputs(images, text)

        # Extract features from backbone
        with torch.no_grad():
            # The backbone expects eagle_* keys but NOT image_sizes
            # The forward_eagle method in eagle_backbone.py tries to delete image_sizes
            # So we need to add it before calling
            if "eagle_image_sizes" not in eagle_inputs:
                # Calculate image sizes from pixel_values if available
                if "eagle_pixel_values" in eagle_inputs:
                    pixel_values = eagle_inputs["eagle_pixel_values"]
                    # image_sizes format: [[height, width]] for each image
                    batch_size = pixel_values.shape[0]
                    # Use default size based on pixel_values shape
                    if len(pixel_values.shape) == 5:  # [B, num_images, C, H, W]
                        h, w = pixel_values.shape[-2:]
                    else:  # [B, C, H, W]
                        h, w = pixel_values.shape[-2:]
                    eagle_inputs["eagle_image_sizes"] = torch.tensor(
                        [[h, w]] * batch_size,
                        device=self.device
                    )

            # Call backbone's prepare_input
            from transformers.feature_extraction_utils import BatchFeature
            backbone_inputs = BatchFeature(data=eagle_inputs)

            # Get backbone outputs
            backbone_outputs = self.model.backbone(backbone_inputs)

            # Extract features and mask
            features = backbone_outputs["backbone_features"]  # [B, T, hidden_dim]
            attention_mask = backbone_outputs.get("backbone_attention_mask", None)

        return features, attention_mask

    def extract_features_for_distillation(
        self,
        images: List[Image.Image],
        text: str,
        return_numpy: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Extract features in format suitable for SAGE distillation.

        Args:
            images: List of PIL images
            text: Text instruction
            return_numpy: Return numpy arrays instead of tensors

        Returns:
            Dictionary with:
            - vision_features: Vision embeddings
            - language_features: Language embeddings
            - combined_features: Full vision-language features
            - attention_mask: Attention mask
        """
        features, attention_mask = self.extract_backbone_features(images, text)

        # features shape: [batch, seq_len, hidden_dim]
        # The sequence includes both vision and language tokens

        result = {
            "combined_features": features,
            "attention_mask": attention_mask,
            "hidden_dim": features.shape[-1],
            "seq_len": features.shape[1],
        }

        if return_numpy:
            result = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in result.items()
            }

        return result

    def extract_action_predictions(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Get action predictions from full model (backbone + action head).

        Args:
            inputs: Prepared model inputs

        Returns:
            Action predictions tensor
        """
        with torch.no_grad():
            outputs = self.model.get_action(inputs)
            actions = outputs["action_pred"]

        return actions

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "backbone_params": sum(p.numel() for p in self.model.backbone.parameters()),
            "action_head_params": sum(p.numel() for p in self.model.action_head.parameters()),
            "action_horizon": self.model.action_horizon,
            "action_dim": self.model.action_dim,
            "compute_dtype": self.model.compute_dtype,
        }


def test_feature_extraction():
    """Test feature extraction with real GR00T model."""
    print("=" * 80)
    print("Testing GR00T Feature Extraction (REAL MODEL)")
    print("=" * 80)

    # Initialize extractor
    extractor = GR00TFeatureExtractor()

    # Print model info
    info = extractor.get_model_info()
    print("\n📊 Model Information:")
    for key, value in info.items():
        print(f"   {key}: {value}")

    # Create dummy image
    dummy_image = Image.new('RGB', (640, 480), color=(73, 109, 137))
    text_instruction = "Pick up the object and place it in the box"

    print("\n🔬 Extracting features...")
    features_dict = extractor.extract_features_for_distillation(
        images=[dummy_image],
        text=text_instruction,
        return_numpy=True,
    )

    print("\n📦 Extracted Features:")
    for key, value in features_dict.items():
        if isinstance(value, np.ndarray):
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"   {key}: {value}")

    print("\n✅ Feature extraction test complete!")
    print("   This uses the REAL GR00T model - no shortcuts!")

    return extractor, features_dict


if __name__ == "__main__":
    extractor, features = test_feature_extraction()
