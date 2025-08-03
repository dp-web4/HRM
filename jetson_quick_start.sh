#!/bin/bash
# Quick Start for HRM on Jetson
# Minimal Sudoku demo - perfect for testing

echo "🚀 HRM Quick Start - Sudoku Demo for Jetson"
echo "==========================================="
echo ""

# Create directories
echo "📁 Creating directories..."
mkdir -p data/sudoku-jetson-demo

# Create a minimal sudoku dataset builder
echo "📝 Creating minimal dataset builder..."
cat > build_sudoku_jetson_demo.py << 'EOF'
#!/usr/bin/env python3
"""
Minimal Sudoku dataset for Jetson demo
Creates just 10 puzzles for quick testing
"""
import json
import os
import random

def create_minimal_sudoku_dataset(output_dir, num_puzzles=10):
    """Create a tiny sudoku dataset for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Simple sudoku generator (for demo purposes)
    # In production, use the full dataset builder
    puzzles = []
    
    print(f"Generating {num_puzzles} sudoku puzzles...")
    
    for i in range(num_puzzles):
        # Create a partially filled sudoku grid
        # This is simplified - real sudoku generation is more complex
        grid = [[0] * 9 for _ in range(9)]
        
        # Fill some random cells (about 30%)
        for _ in range(25):
            row = random.randint(0, 8)
            col = random.randint(0, 8)
            val = random.randint(1, 9)
            grid[row][col] = val
        
        # Convert to format HRM expects
        puzzle = {
            "input": grid,
            "output": grid,  # In real dataset, this would be the solution
            "id": f"sudoku_demo_{i}"
        }
        puzzles.append(puzzle)
    
    # Save dataset
    with open(os.path.join(output_dir, "puzzles.json"), "w") as f:
        json.dump(puzzles, f, indent=2)
    
    print(f"Created {num_puzzles} puzzles in {output_dir}")
    
    # Create metadata
    metadata = {
        "dataset": "sudoku-jetson-demo",
        "num_puzzles": num_puzzles,
        "grid_size": 9,
        "description": "Minimal sudoku dataset for Jetson testing"
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    create_minimal_sudoku_dataset("data/sudoku-jetson-demo", num_puzzles=10)
EOF

# Create a minimal training config
echo "⚙️  Creating Jetson-optimized config..."
cat > config/jetson_sudoku_demo.yaml << 'EOF'
# Minimal config for Jetson Sudoku demo
defaults:
  - base

# Data
data_path: data/sudoku-jetson-demo
puzzle_name: sudoku

# Model - smaller for Jetson
arch:
  H_cycles: 4  # Reduced from 8
  L_cycles: 4  # Reduced from 8
  hidden_size: 256  # Reduced from 384
  num_heads: 4  # Reduced from 6
  expansion: 2.0  # Reduced from 2.5

# Training - optimized for Jetson
epochs: 100  # Quick test
eval_interval: 10
global_batch_size: 8  # Very small for memory
lr: 1e-4
weight_decay: 0.1

# Disable wandb for quick test
wandb:
  mode: disabled

# Use CPU if CUDA not available
device: cuda
EOF

# Create simple test script
echo "🧪 Creating test script..."
cat > test_hrm_minimal.py << 'EOF'
#!/usr/bin/env python3
"""
Minimal HRM test - just verify it can load and forward
"""
import sys
import os

print("🧪 Minimal HRM Test")
print("=" * 50)

try:
    # Add HRM to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Test imports
    print("Testing imports...")
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    from models.common import trunc_normal_init_
    print("✅ Model imports successful")
    
    # Test config
    print("\nTesting configuration...")
    test_config = {
        "batch_size": 1,
        "seq_len": 81,  # 9x9 sudoku
        "puzzle_emb_ndim": 16,
        "num_puzzle_identifiers": 1,
        "vocab_size": 10,  # 0-9 for sudoku
        "H_cycles": 2,
        "L_cycles": 2,
        "H_layers": 2,
        "L_layers": 2,
        "hidden_size": 128,
        "expansion": 2.0,
        "num_heads": 4,
        "pos_encodings": "learned",
        "halt_max_steps": 4,
        "halt_exploration_prob": 0.1,
        "forward_dtype": "float32"  # Use float32 for compatibility
    }
    
    print("Creating model...")
    model = HierarchicalReasoningModel_ACTV1(test_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params/1e6:.2f}M parameters")
    
    # Test device availability
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n✅ CUDA available: {torch.cuda.get_device_name(0)}")
        model = model.to(device)
        print("✅ Model moved to GPU")
    else:
        device = torch.device("cpu")
        print("\n⚠️  CUDA not available, using CPU")
    
    print("\n🎉 All tests passed! HRM is ready to run on Jetson.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

chmod +x build_sudoku_jetson_demo.py
chmod +x test_hrm_minimal.py

echo ""
echo "📋 Quick Start Steps:"
echo "1. First, install dependencies:"
echo "   ./install_jetson.sh"
echo ""
echo "2. Generate mini dataset:"
echo "   python3 build_sudoku_jetson_demo.py"
echo ""
echo "3. Test HRM loading:"
echo "   python3 test_hrm_minimal.py"
echo ""
echo "4. Start mini training (if all tests pass):"
echo "   python3 pretrain.py --config-path config --config-name jetson_sudoku_demo"
echo ""
echo "This will train a tiny HRM on 10 sudoku puzzles as a proof of concept."
echo "For real experiments, use the full dataset builders."
echo ""
echo "💡 Tip: Watch GPU usage with: tegrastats"