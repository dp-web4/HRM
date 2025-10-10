#!/usr/bin/env python3
"""
Test loading Gr00tPolicy with custom SAGE/ARC metadata.
"""

import sys
sys.path.insert(0, "/home/dp/ai-workspace/isaac-gr00t")

import torch
import numpy as np
from PIL import Image

from gr00t.model.policy import Gr00tPolicy
from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.transforms import GR00TTransform


def test_policy_loading():
    """Test loading GR00T policy with our custom metadata."""

    print("=" * 80)
    print("Testing Gr00tPolicy with SAGE/ARC Metadata")
    print("=" * 80)

    # Set up modality config for SAGE/ARC
    modality_config = {
        "video": ModalityConfig(
            delta_indices=[-1, 0],  # Last 2 frames (for temporal context)
            modality_keys=["video.input_grid"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],  # Current state
            modality_keys=["state.task_encoding"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),  # 16-step action horizon
            modality_keys=["action.grid_output"],
        ),
    }

    # Create transform
    print("\n📦 Creating GR00T transform...")
    transform = GR00TTransform(
        max_state_dim=32,
        max_action_dim=32,
        state_horizon=1,
        action_horizon=16,
    )

    # Load policy
    print("📦 Loading Gr00tPolicy with NEW_EMBODIMENT...")
    try:
        policy = Gr00tPolicy(
            model_path="nvidia/GR00T-N1.5-3B",
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_config=modality_config,
            modality_transform=transform,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        print("✅ Policy loaded successfully!")
        print(f"   Model params: {sum(p.numel() for p in policy.model.parameters()) / 1e9:.2f}B")
        print(f"   Backbone params: {sum(p.numel() for p in policy.model.backbone.parameters()) / 1e9:.2f}B")
        print(f"   Device: {policy.device}")

        # Test feature extraction
        print("\n🔬 Testing feature extraction...")

        # Create dummy ARC grid image (simple colored grid)
        img = np.random.randint(0, 255, (900, 900, 3), dtype=np.uint8)

        # Create dummy observations
        # Note: GR00TTransform expects BATCHED data to properly process eagle_content:
        # - video: [B, T, V, H, W, C] where B=batch, T=time, V=views, H=height, W=width, C=channels
        # - state: [B, T, state_dim]
        # For unbatched data, eagle_content doesn't get converted to eagle_* tensors
        video_frames = np.stack([img, img], axis=0)  # [T=2, H, W, C]
        video_with_view = video_frames[:, np.newaxis, :, :, :]  # [T=2, V=1, H, W, C]
        video_batched = video_with_view[np.newaxis, :, :, :, :, :]  # [B=1, T=2, V=1, H, W, C]

        state_frames = np.random.rand(1, 16).astype(np.float32)  # [T=1, 16]
        state_batched = state_frames[np.newaxis, :, :]  # [B=1, T=1, 16]

        observations = {
            "video": video_batched,  # [B=1, T=2, V=1, H=900, W=900, C=3]
            "state": state_batched,  # [B=1, T=1, 16]
            "annotation.0": np.array(["Solve the pattern transformation task"]),
        }

        print(f"   Input video shape: {observations['video'].shape}")
        print(f"   Input state shape: {observations['state'].shape}")

        # Apply transforms
        print("\n📊 Applying transforms...")
        normalized_obs = policy.apply_transforms(observations)

        print(f"   Normalized keys: {list(normalized_obs.keys())}")
        for key, value in normalized_obs.items():
            if isinstance(value, torch.Tensor):
                print(f"      {key}: {value.shape}, {value.dtype}")

        # Extract backbone features
        print("\n🧠 Extracting backbone features...")
        with torch.no_grad():
            backbone_inputs, _ = policy.model.prepare_input(normalized_obs)
            backbone_outputs = policy.model.backbone(backbone_inputs)

            features = backbone_outputs["backbone_features"]
            attention_mask = backbone_outputs.get("backbone_attention_mask")

            print(f"✅ Feature extraction successful!")
            print(f"   Features shape: {features.shape}")
            print(f"   Features dtype: {features.dtype}")
            print(f"   Features range: [{features.min():.4f}, {features.max():.4f}]")
            print(f"   Features mean: {features.mean():.4f}")
            print(f"   Features std: {features.std():.4f}")

            if attention_mask is not None:
                print(f"   Attention mask shape: {attention_mask.shape}")
                print(f"   Active tokens: {attention_mask.sum().item()}/{attention_mask.numel()}")

        print("\n✅ All tests passed!")
        print("   GR00T policy with SAGE/ARC metadata is working correctly")
        print("   Ready for knowledge distillation!")

        return policy, features

    except Exception as e:
        print(f"\n❌ Error loading policy: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    policy, features = test_policy_loading()
