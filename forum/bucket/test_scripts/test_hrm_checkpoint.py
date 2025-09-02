#!/usr/bin/env python3
"""
Test HRM checkpoint and inspect its contents
"""

import torch
import sys
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """Inspect HRM checkpoint contents"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n📦 Checkpoint contents:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: tensor {checkpoint[key].shape}")
        elif isinstance(checkpoint[key], dict):
            print(f"  {key}: dict with {len(checkpoint[key])} items")
            # Show first few items if it's a state dict
            if key in ['model_state_dict', 'state_dict']:
                for i, (k, v) in enumerate(list(checkpoint[key].items())[:5]):
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.shape}")
                if len(checkpoint[key]) > 5:
                    print(f"    ... and {len(checkpoint[key]) - 5} more")
        else:
            print(f"  {key}: {type(checkpoint[key]).__name__} = {checkpoint[key]}")
    
    # Check if it's a model state dict directly
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Might be the state dict directly
        state_dict = checkpoint
    
    # Analyze model architecture
    print("\n🏗️  Model architecture hints:")
    
    # Look for L and H modules
    l_params = [k for k in state_dict.keys() if k.startswith('L.') or k.startswith('l_')]
    h_params = [k for k in state_dict.keys() if k.startswith('H.') or k.startswith('h_')]
    
    if l_params:
        print(f"  L-module parameters: {len(l_params)}")
        print(f"    Sample: {l_params[0]}")
    
    if h_params:
        print(f"  H-module parameters: {len(h_params)}")
        print(f"    Sample: {h_params[0]}")
    
    # Count total parameters
    total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
    print(f"\n📊 Total parameters: {total_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    return checkpoint

if __name__ == "__main__":
    checkpoint_path = Path("checkpoints/hrm_arc_best.pt")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for ckpt in Path("checkpoints").glob("*.pt"):
            print(f"  - {ckpt}")
        sys.exit(1)
    
    checkpoint = inspect_checkpoint(checkpoint_path)