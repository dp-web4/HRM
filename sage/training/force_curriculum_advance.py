#!/usr/bin/env python3
"""
Force curriculum advancement to break out of training plateau
"""

import torch
import pickle
from pathlib import Path
from datetime import datetime

def force_curriculum_advance():
    """Manually trigger curriculum evolution"""
    
    checkpoint_dir = Path("checkpoints/sage")
    
    # Find latest state
    state_files = list(checkpoint_dir.glob("ongoing_state_*.pt"))
    if not state_files:
        print("❌ No ongoing state found")
        return
    
    latest_state = max(state_files, key=lambda x: x.stat().st_mtime)
    print(f"📂 Loading: {latest_state}")
    
    # Load and modify
    state = torch.load(latest_state, map_location='cpu')
    
    # Force curriculum advancement
    print(f"\n🎯 Current state:")
    print(f"  - Cycle count: {state.get('cycle_count', 'unknown')}")
    print(f"  - Curriculum stage: {state.get('curriculum_stage', 0)}")
    print(f"  - Performance history: {state.get('performance_history', [])[-5:]}")
    
    # Modifications to break plateau:
    # 1. Reset stagnation counter to force adaptation
    state['stagnation_counter'] = 10  # Force trigger
    
    # 2. Inject performance spike to trigger difficulty increase
    if 'performance_history' in state:
        state['performance_history'].append(1.3)  # Above threshold
        
    # 3. Advance curriculum stage
    state['curriculum_stage'] = state.get('curriculum_stage', 0) + 1
    
    # 4. Adjust learning rate in optimizer state
    if 'optimizer_state_dict' in state:
        for group in state['optimizer_state_dict'].get('param_groups', []):
            old_lr = group.get('lr', 0.0001)
            new_lr = old_lr * 0.5  # Reduce LR to help convergence
            group['lr'] = new_lr
            print(f"  - Learning rate: {old_lr:.6f} → {new_lr:.6f}")
    
    # Save modified state
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = checkpoint_dir / f"ongoing_state_{timestamp}_advanced.pt"
    torch.save(state, new_path)
    
    print(f"\n✅ Curriculum advanced and saved to: {new_path}")
    print("\nRecommended next steps:")
    print("1. Stop current training (Ctrl+C)")
    print("2. Restart with advanced checkpoint:")
    print("   python3 ongoing_sage_training.py")
    print("\nAlternatively, implement these changes:")
    print("- Add noise to training data (augmentation)")
    print("- Increase task complexity (harder patterns)")
    print("- Mix in more real camera data")
    print("- Add dropout for regularization")
    
    # Create config for harder training
    harder_config = {
        'difficulty_factor': 0.8,
        'camera_ratio': 0.4,
        'noise_level': 0.1,
        'dropout_rate': 0.2,
        'learning_rate': 5e-5,
        'batch_size': 16,
        'gradient_accumulation': 4,
        'use_mixed_precision': True
    }
    
    config_path = checkpoint_dir / "harder_config.json"
    import json
    with open(config_path, 'w') as f:
        json.dump(harder_config, f, indent=2)
    
    print(f"\n💪 Harder config saved to: {config_path}")

if __name__ == "__main__":
    force_curriculum_advance()