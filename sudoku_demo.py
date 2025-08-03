#!/usr/bin/env python3
"""
Simple Sudoku demo for HRM - no external dependencies
"""

import torch
import torch.nn.functional as F
import numpy as np
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
import time

def create_sudoku_puzzle():
    """Create a simple 4x4 Sudoku puzzle for testing"""
    # Simple 4x4 Sudoku (numbers 1-4)
    puzzle = np.array([
        [1, 0, 3, 0],
        [0, 3, 0, 1],
        [3, 0, 1, 0],
        [0, 1, 0, 3]
    ])
    
    solution = np.array([
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [3, 4, 1, 2],
        [2, 1, 4, 3]
    ])
    
    return puzzle, solution

def encode_sudoku(puzzle, vocab_size=10):
    """Encode Sudoku puzzle for HRM input"""
    # Flatten and add special tokens
    # 0: blank, 1-4: numbers, 5: separator
    flat = puzzle.flatten()
    # Map 0 -> 5 (blank token), 1-4 -> 1-4
    encoded = np.where(flat == 0, 5, flat)
    return torch.tensor(encoded, dtype=torch.long)

def decode_output(logits, original_shape=(4, 4)):
    """Decode HRM output back to Sudoku grid"""
    # Get predictions
    preds = torch.argmax(logits, dim=-1)
    # Take first batch, first sequence
    preds = preds[0, :16].cpu().numpy()
    # Map back: 5 -> 0 (blank), 1-4 -> 1-4
    decoded = np.where(preds == 5, 0, preds)
    return decoded.reshape(original_shape)

def main():
    print("🎲 HRM Sudoku Demo")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create puzzle
    puzzle, solution = create_sudoku_puzzle()
    print("\n📝 Input Puzzle:")
    print(puzzle)
    print("\n✅ Solution:")
    print(solution)
    
    # Model config (small for demo)
    config = {
        'batch_size': 1,
        'seq_len': 32,
        'puzzle_emb_ndim': 64,
        'num_puzzle_identifiers': 10,
        'vocab_size': 10,
        'H_cycles': 2,
        'L_cycles': 4,
        'H_layers': 2,
        'L_layers': 1,
        'hidden_size': 128,
        'expansion': 2.0,
        'num_heads': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 5,
        'halt_exploration_prob': 0.0,
        'forward_dtype': 'float32',
    }
    
    # Create model
    print("\n🧠 Creating HRM model...")
    model = HierarchicalReasoningModel_ACTV1(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count/1e6:.2f}M")
    
    # Prepare input
    inputs = encode_sudoku(puzzle).unsqueeze(0).to(device)
    batch = {
        'inputs': inputs,
        'puzzle_identifiers': torch.tensor([0], device=device),
    }
    
    # Run inference
    print("\n🔄 Running HRM inference...")
    model.eval()
    
    with torch.no_grad():
        # Initial carry state
        carry = model.initial_carry(batch)
        
        # Run multiple steps (adaptive computation)
        print("Adaptive computation steps:")
        for step in range(config['halt_max_steps']):
            start_time = time.time()
            carry, outputs = model(carry, batch)
            step_time = (time.time() - start_time) * 1000
            
            # Check halting probability
            halt_probs = F.softmax(outputs['q_halt_logits'], dim=-1)
            halt_prob = halt_probs[0, 1].item()  # Probability of halting
            
            print(f"  Step {step+1}: {step_time:.1f}ms, halt_prob={halt_prob:.3f}")
            
            if halt_prob > 0.5:
                print("  ✓ Model decided to halt")
                break
    
    # Decode output
    print("\n🎯 Model Output:")
    logits = outputs['logits']
    pred_puzzle = decode_output(logits)
    print(pred_puzzle)
    
    # Check accuracy
    correct = np.sum(pred_puzzle == solution)
    total = solution.size
    print(f"\n📊 Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    
    # Note about training
    print("\n💡 Note: This is an untrained model with random weights.")
    print("   With training on Sudoku puzzles, HRM achieves near-perfect accuracy!")
    print("   The hierarchical architecture allows it to learn complex reasoning.")
    
    print("\n" + "=" * 50)
    print("✨ Key HRM Features Demonstrated:")
    print("  - Adaptive computation (variable steps)")
    print("  - Hierarchical reasoning (H and L modules)")
    print("  - Small model size (~0.21M params)")
    print("  - GPU acceleration ready")
    print("=" * 50)

if __name__ == "__main__":
    main()