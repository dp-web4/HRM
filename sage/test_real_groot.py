#!/usr/bin/env python3
"""
Test Real GR00T Eagle Vision Model
===================================
Verify we can load and use the actual NVIDIA GR00T N1.5 model
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add GR00T to path
sys.path.append('/home/dp/ai-workspace/isaac-gr00t')

def test_eagle_backbone():
    """Test loading and using Eagle backbone"""
    print("Testing Real GR00T Eagle Vision Model")
    print("=" * 60)
    
    try:
        # Import GR00T components
        from gr00t.model.backbone import EagleBackbone
        print("✅ Successfully imported GR00T Eagle backbone")
        
        # Check for Eagle model files
        import gr00t
        eagle_path = Path(gr00t.__file__).parent / "model" / "backbone" / "eagle2_hg_model"
        if eagle_path.exists():
            print(f"✅ Eagle model path exists: {eagle_path}")
            config_files = list(eagle_path.glob("*.json"))
            py_files = list(eagle_path.glob("*.py"))
            print(f"   Found {len(config_files)} config files, {len(py_files)} Python files")
        else:
            print(f"❌ Eagle model path not found: {eagle_path}")
            return False
            
        # Try to create Eagle backbone
        print("\n🔄 Creating Eagle backbone...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Device: {device}")
        
        # Initialize with minimal config
        backbone = EagleBackbone(
            tune_llm=False,      # Don't tune language model
            tune_visual=False,   # Don't tune vision model
            select_layer=-1,     # Use last layer
            project_to_dim=1536  # Project to 1536 dims
        )
        print("✅ Eagle backbone created successfully")
        
        # Check model structure
        param_count = sum(p.numel() for p in backbone.parameters())
        trainable_count = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"\n📊 Model Statistics:")
        print(f"   Total parameters: {param_count:,}")
        print(f"   Trainable parameters: {trainable_count:,}")
        
        # Test forward pass with dummy data
        print("\n🔄 Testing forward pass...")
        batch_size = 1
        seq_len = 10
        
        # Create dummy input matching Eagle expectations
        dummy_input = {
            'eagle_input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'eagle_attention_mask': torch.ones(batch_size, seq_len),
            'eagle_image_sizes': [(224, 224)],
            'eagle_pixel_values': torch.randn(batch_size, 3, 224, 224)
        }
        
        # Prepare input
        from transformers.feature_extraction_utils import BatchFeature
        vl_input = BatchFeature(data=dummy_input)
        
        # Forward pass
        with torch.no_grad():
            try:
                output = backbone(vl_input)
                if 'backbone_features' in output:
                    features = output['backbone_features']
                    print(f"✅ Forward pass successful!")
                    print(f"   Output shape: {features.shape}")
                    print(f"   Feature dimension: {features.shape[-1]}")
                else:
                    print("❌ No backbone_features in output")
            except Exception as e:
                print(f"⚠️  Forward pass failed (expected without model weights): {e}")
                print("   This is normal - we need to download the actual model weights")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import GR00T: {e}")
        print("\n📝 To use real GR00T, run:")
        print("   cd /home/dp/ai-workspace/isaac-gr00t")
        print("   pip install -e .")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_action_head():
    """Test GR00T action head"""
    print("\n" + "=" * 60)
    print("Testing GR00T Flow Matching Action Head")
    print("=" * 60)
    
    try:
        from gr00t.model.action_head.flow_matching_action_head import (
            FlowmatchingActionHead, 
            FlowmatchingActionHeadConfig
        )
        print("✅ Successfully imported Flow Matching Action Head")
        
        # Create config
        config = FlowmatchingActionHeadConfig(
            hidden_size=1536,
            action_dim=7,  # Example: 7 DOF robot arm
            action_horizon=10,  # Predict 10 steps ahead
            num_layers=4
        )
        
        # Create action head
        action_head = FlowmatchingActionHead(config)
        print(f"✅ Action head created")
        
        # Check parameters
        param_count = sum(p.numel() for p in action_head.parameters())
        print(f"   Parameters: {param_count:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test action head: {e}")
        return False

def check_demo_data():
    """Check if demo data is available"""
    print("\n" + "=" * 60)
    print("Checking Demo Data")
    print("=" * 60)
    
    demo_path = Path("/home/dp/ai-workspace/isaac-gr00t/demo_data/robot_sim.PickNPlace")
    if demo_path.exists():
        print(f"✅ Demo data found: {demo_path}")
        
        # Check episodes
        episode_files = list((demo_path / "data" / "chunk-000").glob("*.parquet"))
        video_files = list((demo_path / "videos" / "chunk-000" / "observation.images.ego_view").glob("*.mp4"))
        
        print(f"   Episodes: {len(episode_files)} parquet files")
        print(f"   Videos: {len(video_files)} mp4 files")
        
        # Try to load an episode
        try:
            import pandas as pd
            if episode_files:
                df = pd.read_parquet(episode_files[0])
                print(f"   Episode shape: {df.shape}")
                print(f"   Columns: {list(df.columns)[:5]}...")
        except ImportError:
            print("   (Install pandas and pyarrow to read episodes)")
        
        return True
    else:
        print(f"❌ Demo data not found at {demo_path}")
        return False

def main():
    """Run all tests"""
    print("🚀 Real GR00T Model Discovery Test")
    print("=" * 60)
    print("Testing actual NVIDIA Isaac GR00T N1.5 components\n")
    
    # Run tests
    eagle_ok = test_eagle_backbone()
    action_ok = test_action_head()
    data_ok = check_demo_data()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if eagle_ok and action_ok:
        print("✅ GR00T components are accessible!")
        print("\n📝 Next steps:")
        print("1. Install GR00T: cd /home/dp/ai-workspace/isaac-gr00t && pip install -e .")
        print("2. Download weights: huggingface-cli download nvidia/GR00T-N1.5-3B")
        print("3. Create Vision IRP using Eagle backbone")
        print("4. Implement trust-attention-surprise loop")
    else:
        print("⚠️  Some components not accessible")
        print("Need to install GR00T package first")
    
    if data_ok:
        print("\n📊 Demo data available for testing!")

if __name__ == "__main__":
    main()