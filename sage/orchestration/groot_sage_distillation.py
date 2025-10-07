#!/usr/bin/env python3
"""
GR00T → SAGE Knowledge Distillation

Uses REAL NVIDIA GR00T N1.5 model to extract features for SAGE training.
No shortcuts - actual model with proper transforms and feature extraction.

Architecture:
- Teacher: GR00T N1.5 (3B params, vision-language-action model)
- Student: SAGE (to be distilled from GR00T features)
- Method: Extract backbone features for knowledge distillation
"""

import sys
sys.path.insert(0, "/home/dp/ai-workspace/isaac-gr00t")

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image

from gr00t.model.policy import Gr00tPolicy
from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.transforms import GR00TTransform


class GR00TSAGEDistiller:
    """
    Extract vision-language features from real GR00T for SAGE distillation.

    Uses the actual Gr00tPolicy wrapper which handles all transforms correctly.
    """

    def __init__(
        self,
        model_path: str = "nvidia/GR00T-N1.5-3B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize GR00T→SAGE distiller.

        Args:
            model_path: HuggingFace model path
            device: Device to run on
        """
        self.device = device
        self.model_path = model_path

        print(f"🚀 Initializing GR00T→SAGE Distiller")
        print(f"   Model: {model_path}")
        print(f"   Device: {device}")

        # Set up modality config (simplified for feature extraction)
        self.modality_config = {
            "video": ModalityConfig(
                delta_indices=[-1, 0],  # Last 2 frames
                modality_keys=["video.webcam"],
            ),
            "state": ModalityConfig(
                delta_indices=[0],  # Current state
                modality_keys=["state.robot"],
            ),
            "action": ModalityConfig(
                delta_indices=list(range(16)),  # 16-step action horizon
                modality_keys=["action"],
            ),
        }

        # Create transform
        self.transform = GR00TTransform(
            max_state_dim=32,
            max_action_dim=32,
            state_horizon=1,
            action_horizon=16,
        )

        # Load policy (this handles all the complexity)
        print("📦 Loading GR00T model via policy wrapper...")
        self.policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,  # For new embodiments/finetuning
            modality_config=self.modality_config,
            modality_transform=self.transform,
            device=device,
        )

        print(f"✅ GR00T loaded successfully!")
        print(f"   Backbone params: {sum(p.numel() for p in self.policy.model.backbone.parameters()) / 1e9:.2f}B")
        print(f"   Total params: {sum(p.numel() for p in self.policy.model.parameters()) / 1e9:.2f}B")

    def extract_backbone_features(
        self,
        images: List[np.ndarray],
        instruction: str = "Perform the default behavior",
        state: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract vision-language features from GR00T backbone.

        Args:
            images: List of numpy arrays [H, W, 3]
            instruction: Text instruction
            state: Optional robot state [state_dim]

        Returns:
            Dictionary with backbone features
        """
        # Prepare observations in the format GR00T expects
        observations = {}

        # Video format: [T, H, W, C] where T is number of frames
        if len(images) == 1:
            # Duplicate last frame if only one provided
            video = np.stack([images[0], images[0]], axis=0)  # [2, H, W, 3]
        else:
            video = np.stack(images[-2:], axis=0)  # Last 2 frames

        observations["video.webcam"] = video

        # State (if provided)
        if state is not None:
            # Format: [T, state_dim]
            observations["state.robot"] = state.reshape(1, -1)
        else:
            # Default zero state
            observations["state.robot"] = np.zeros((1, 7))  # Default 7-DOF

        # Annotation (instruction)
        observations["annotation.0"] = np.array([instruction])

        # Apply transforms (this does all the preprocessing)
        normalized_obs = self.policy.apply_transforms(observations)

        # Extract features from backbone
        with torch.no_grad():
            # Prepare inputs for model
            backbone_inputs, _ = self.policy.model.prepare_input(normalized_obs)

            # Get backbone features
            backbone_outputs = self.policy.model.backbone(backbone_inputs)

            # Extract features
            features = backbone_outputs["backbone_features"].cpu().numpy()
            attention_mask = backbone_outputs.get("backbone_attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.cpu().numpy()

        return {
            "features": features,  # [batch, seq_len, hidden_dim]
            "attention_mask": attention_mask,
            "hidden_dim": features.shape[-1],
            "seq_len": features.shape[1],
            "instruction": instruction,
        }

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "backbone_params": sum(p.numel() for p in self.policy.model.backbone.parameters()),
            "total_params": sum(p.numel() for p in self.policy.model.parameters()),
            "action_horizon": self.policy.model.action_horizon,
            "action_dim": self.policy.model.action_dim,
        }


def test_distillation():
    """Test feature extraction for distillation."""
    print("=" * 80)
    print("GR00T → SAGE Knowledge Distillation Test")
    print("Using REAL GR00T model - NO SHORTCUTS")
    print("=" * 80)

    # Initialize distiller
    distiller = GR00TSAGEDistiller()

    # Print model info
    info = distiller.get_model_info()
    print("\n📊 Model Information:")
    for key, value in info.items():
        if isinstance(value, int) and value > 1000:
            print(f"   {key}: {value / 1e9:.2f}B")
        else:
            print(f"   {key}: {value}")

    # Create test images
    img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test feature extraction
    print("\n🔬 Extracting features from GR00T backbone...")
    features_dict = distiller.extract_backbone_features(
        images=[img1, img2],
        instruction="Pick up the red cube and place it in the box",
    )

    print("\n📦 Extracted Features (for SAGE distillation):")
    for key, value in features_dict.items():
        if isinstance(value, np.ndarray):
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"   {key}: {value}")

    print("\n✅ Feature extraction successful!")
    print("   These features can be used to train SAGE via knowledge distillation")
    print("   Teacher (GR00T) → Student (SAGE)")

    return distiller, features_dict


if __name__ == "__main__":
    distiller, features = test_distillation()
